"""Dataset validation, splitting, and treatment-contrast utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from causal_pipeline.config import (
    DataConfig,
    JsonValue,
    OutcomeType,
    PipelineConfig,
    TreatmentMode,
)

logger = logging.getLogger(__name__)

IPW_WEIGHTS_STABILIZED_DEFAULT = False
PROPENSITY_CLIP_FLOOR = 1e-6


class CausalDataset(BaseModel):
    """Observational cohort: column contract plus validated tabular data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: DataConfig
    df: pd.DataFrame

    @model_validator(mode="after")
    def validate_df_contract(self) -> CausalDataset:
        spec = self.data
        required = [spec.outcome, spec.treatment] + list(spec.confounders)
        required.extend(spec.effect_modifiers)
        if spec.group_id is not None:
            required.append(spec.group_id)

        missing = [column for column in required if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        if self.df[required].isna().any().any():
            raise ValueError("Configured columns contain missing values.")

        treatment = self.df[spec.treatment]
        observed = set(treatment.unique())
        expected = set(spec.treatment_values)
        if not observed.issubset(expected):
            extra = observed - expected
            raise ValueError(f"Unexpected treatment values in data: {extra}")

        if spec.control_value not in expected:
            raise ValueError("control_value must be in treatment_values.")

        outcome = self.df[spec.outcome]
        if spec.outcome_type == OutcomeType.BINARY:
            unique = set(outcome.dropna().unique())
            if not unique.issubset({0, 1}):
                raise ValueError("Binary outcome must be coded as 0/1.")
        return self

    @property
    def outcome(self) -> str:
        return self.data.outcome

    @property
    def treatment(self) -> str:
        return self.data.treatment

    @property
    def confounders(self) -> list[str]:
        return list(self.data.confounders)

    @property
    def effect_modifiers(self) -> list[str]:
        return list(self.data.effect_modifiers)

    @property
    def group_id(self) -> str | None:
        return self.data.group_id

    @property
    def outcome_type(self) -> OutcomeType:
        return self.data.outcome_type

    @property
    def treatment_mode(self) -> TreatmentMode:
        return self.data.treatment_mode

    @property
    def control_value(self) -> JsonValue:
        return self.data.control_value

    @property
    def treatment_values(self) -> list[JsonValue]:
        return list(self.data.treatment_values)

    @property
    def X_confounders(self) -> pd.DataFrame:
        return self.df[list(self.data.confounders)]

    @property
    def X_effect_modifiers(self) -> pd.DataFrame:
        return self.df[list(self.data.effect_modifiers)]

    @property
    def X_adjustment(self) -> pd.DataFrame:
        """Confounders plus any effect modifier that is not already a confounder."""
        columns = list(dict.fromkeys([*self.confounders, *self.effect_modifiers]))
        return self.df[columns]

    @property
    def X_controls(self) -> pd.DataFrame:
        """Confounders that are not also effect modifiers, for an X/W split."""
        modifiers = set(self.effect_modifiers)
        columns = [column for column in self.confounders if column not in modifiers]
        return self.df[columns]

    @property
    def treatment_series(self) -> pd.Series:
        return self.df[self.data.treatment]

    @property
    def outcome_series(self) -> pd.Series:
        return self.df[self.data.outcome]

    def subset(self, positions: np.ndarray) -> CausalDataset:
        """Copy the rows at these positions without changing the source frame."""
        return CausalDataset(data=self.data, df=self.df.iloc[positions].reset_index(drop=True))


class DataPartitions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    estimation: CausalDataset
    test: CausalDataset | None = None


def contrast_columns(control_value: JsonValue, treatment_values: list[JsonValue]) -> list[str]:
    """Return contrast column names such as 1_vs_0 for non-control arms."""
    arms = [value for value in treatment_values if value != control_value]
    control_label = arm_label(control_value)
    return [f"{arm_label(arm)}_vs_{control_label}" for arm in arms]


def arm_label(value: JsonValue) -> str:
    return str(value).replace(".", "p")


class DataSplitter:
    """Keep the full sample, or hold out a test set when policy evaluation is on."""

    def split(self, dataset: CausalDataset, config: PipelineConfig) -> DataPartitions:
        if not config.policy:
            logger.info("Using the full sample for cross-fit estimation.")
            return DataPartitions(
                estimation=CausalDataset(data=dataset.data, df=dataset.df.copy()),
            )

        logger.info("Holding out a test set for policy evaluation.")
        estimation_frame, test_frame = self.hold_out_test(
            df=dataset.df,
            group_id=dataset.group_id,
            treatment_column=dataset.treatment,
            treatment_mode=dataset.treatment_mode,
            test_fraction=config.split.test_fraction,
            random_state=config.split.random_state,
        )
        return DataPartitions(
            estimation=CausalDataset(data=dataset.data, df=estimation_frame),
            test=CausalDataset(data=dataset.data, df=test_frame),
        )

    def hold_out_test(
        self,
        df: pd.DataFrame,
        group_id: str | None,
        treatment_column: str,
        treatment_mode: TreatmentMode,
        test_fraction: float,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if group_id is not None:
            return self.split_grouped(
                df=df,
                group_id=group_id,
                first_fraction=1.0 - test_fraction,
                random_state=random_state,
            )
        estimation, test = train_test_split(
            df,
            test_size=test_fraction,
            random_state=random_state,
            stratify=df[treatment_column] if treatment_mode == TreatmentMode.BINARY else None,
            shuffle=True,
        )
        return estimation, test

    def split_grouped(
        self,
        df: pd.DataFrame,
        group_id: str,
        first_fraction: float,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=1.0 - first_fraction,
            random_state=random_state,
        )
        groups = df[group_id]
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def ipw_weights_binary(
    treatment: np.ndarray,
    propensity: np.ndarray,
    stabilized: bool = IPW_WEIGHTS_STABILIZED_DEFAULT,
) -> np.ndarray:
    """Horvitz-Thompson weights for binary treatment."""
    treatment_values = np.asarray(treatment, dtype=float)
    propensity_values = np.clip(propensity, PROPENSITY_CLIP_FLOOR, 1.0 - PROPENSITY_CLIP_FLOOR)
    if stabilized:
        treated_probability = float(np.mean(treatment_values))
        control_probability = 1.0 - treated_probability
        return (
            treatment_values * treated_probability / propensity_values
            + (1.0 - treatment_values) * control_probability / (1.0 - propensity_values)
        )
    return treatment_values / propensity_values + (1.0 - treatment_values) / (1.0 - propensity_values)


def ipw_weights_multi(
    treatment: np.ndarray,
    propensity_matrix: np.ndarray,
    treatment_levels: list[JsonValue],
) -> np.ndarray:
    """Generalized IPW weights 1 / e_{T_i}(X_i)."""
    level_to_index = {level: index for index, level in enumerate(treatment_levels)}
    arm_indices = np.array([level_to_index[value] for value in treatment])
    row_indices = np.arange(len(treatment))
    selected = propensity_matrix[row_indices, arm_indices]
    selected = np.clip(selected, PROPENSITY_CLIP_FLOOR, None)
    return 1.0 / selected

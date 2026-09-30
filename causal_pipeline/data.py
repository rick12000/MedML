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


class DataPartitions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    train: CausalDataset
    validation: CausalDataset
    test: CausalDataset


def contrast_columns(control_value: JsonValue, treatment_values: list[JsonValue]) -> list[str]:
    """Return contrast column names such as 1_vs_0 for non-control arms."""
    arms = [value for value in treatment_values if value != control_value]
    control_label = arm_label(control_value)
    return [f"{arm_label(arm)}_vs_{control_label}" for arm in arms]


def arm_label(value: JsonValue) -> str:
    return str(value).replace(".", "p")


class DataSplitter:
    """Split data into train, validation, and test partitions."""

    def split(self, dataset: CausalDataset, config: PipelineConfig) -> DataPartitions:
        logger.info("Splitting dataset into train, validation, and test partitions.")
        df_input = dataset.df
        data_config = dataset.data

        train_fraction = config.split.train_fraction
        validation_fraction = config.split.validation_fraction
        test_fraction = config.split.test_fraction
        random_state = config.split.random_state
        group_id = data_config.group_id
        treatment_col = data_config.treatment

        if not config.policy:
            if group_id is None:
                df_train, df_validation = train_test_split(
                    df_input,
                    test_size=validation_fraction,
                    random_state=random_state,
                    stratify=df_input[treatment_col]
                    if data_config.treatment_mode == TreatmentMode.BINARY
                    else None,
                )
            else:
                df_train, df_validation = self.split_grouped(
                    df=df_input,
                    group_id=group_id,
                    first_fraction=train_fraction,
                    random_state=random_state,
                )
            df_test = df_input.iloc[0:0].copy()
            return DataPartitions(
                train=CausalDataset(data=data_config, df=df_train),
                validation=CausalDataset(data=data_config, df=df_validation),
                test=CausalDataset(data=data_config, df=df_test),
            )

        val_test_fraction = validation_fraction + test_fraction
        if group_id is None:
            df_train, df_val_test = train_test_split(
                df_input,
                test_size=val_test_fraction,
                random_state=random_state,
                stratify=df_input[treatment_col]
                if data_config.treatment_mode == TreatmentMode.BINARY
                else None,
            )
            relative_test = test_fraction / val_test_fraction
            df_validation, df_test = train_test_split(
                df_val_test,
                test_size=relative_test,
                random_state=random_state,
                stratify=df_val_test[treatment_col]
                if data_config.treatment_mode == TreatmentMode.BINARY
                else None,
            )
        else:
            df_train, df_val_test = self.split_grouped(
                df=df_input,
                group_id=group_id,
                first_fraction=train_fraction,
                random_state=random_state,
            )
            relative_test = test_fraction / val_test_fraction
            df_validation, df_test = self.split_grouped(
                df=df_val_test,
                group_id=group_id,
                first_fraction=1.0 - relative_test,
                random_state=random_state + 1,
            )

        return DataPartitions(
            train=CausalDataset(data=data_config, df=df_train),
            validation=CausalDataset(data=data_config, df=df_validation),
            test=CausalDataset(data=data_config, df=df_test),
        )

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
    stabilized: bool = False,
) -> np.ndarray:
    """Horvitz-Thompson weights for binary treatment."""
    treatment = treatment.astype(float)
    propensity = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    if stabilized:
        treated_probability = float(np.mean(treatment))
        control_probability = 1.0 - treated_probability
        return (
            treatment * treated_probability / propensity
            + (1.0 - treatment) * control_probability / (1.0 - propensity)
        )
    return treatment / propensity + (1.0 - treatment) / (1.0 - propensity)


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
    selected = np.clip(selected, 1e-6, None)
    return 1.0 / selected

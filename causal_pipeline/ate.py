"""ATE estimator adapters, confidence intervals, and sensitivity."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from causallib.estimation import Standardization
from doubleml import DoubleMLData
from doubleml.irm import DoubleMLAPO, DoubleMLAPOS, DoubleMLIRM
from doubleml.plm import DoubleMLPLR
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator

from causal_pipeline.config import (
    ATEEstimatorSpec,
    ATEKind,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    IPWATEEstimatorSpec,
    IPWWeighting,
    JsonValue,
    OutcomeType,
    SensitivityConfig,
    require_classifier,
)
from causal_pipeline.crossfit import (
    aipw_arm_scores,
    clip_propensity,
    cross_fit_nuisances,
    VARIANCE_PROPENSITY_TREATED_AS_KNOWN,
    factual_predictions,
    hajek_influence,
    horvitz_thompson_influence,
    learner_random_state,
    normal_interval_from_scores,
    target_potential_outcomes,
)
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.scaling import clone_scaled_estimator
from causal_pipeline.utils import write_html

logger = logging.getLogger(__name__)

SENSITIVITY_STRENGTH_CAP = 0.9999
SENSITIVITY_BISECT_ITERATIONS = 30
SENSITIVITY_CONFOUNDING_RHO = 1.0
SENSITIVITY_POINT_BOUND = "theta"
SENSITIVITY_INTERVAL_BOUND = "ci"


class BaseATEEstimator(ABC):
    """Common interface for average treatment effect estimators."""

    @abstractmethod
    def fit(self, data: CausalDataset) -> BaseATEEstimator:
        ...

    @abstractmethod
    def estimate(self) -> pd.DataFrame:
        ...


def initialize_ate_estimator(
    spec: ATEEstimatorSpec,
    data: CausalDataset,
) -> BaseATEEstimator:
    if spec.kind == ATEKind.IPW:
        return IPWAdapter(spec=spec, data=data)
    if spec.kind == ATEKind.AIPW:
        return AIPWAdapter(spec=spec, data=data)
    if spec.kind == ATEKind.TMLE:
        return TMLEAdapter(spec=spec, data=data)
    if spec.kind == ATEKind.DML_IRM:
        return DoubleMLIRMAdapter(spec=spec, data=data)
    if spec.kind == ATEKind.DML_PLR:
        return DoubleMLPLRAdapter(spec=spec, data=data)
    if spec.kind == ATEKind.DML_APOS:
        return DoubleMLAPOSAdapter(spec=spec, data=data)
    raise ValueError(f"Unsupported ATE kind: {spec.kind}")


def doubleml_dataframe(
    outcome: pd.Series,
    treatment: pd.Series,
    confounders: pd.DataFrame,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Align outcome, treatment, and confounders on a RangeIndex for DoubleML."""
    columns = {
        "y": np.asarray(outcome),
        "d": np.asarray(treatment),
    }
    if groups is not None:
        columns["cluster"] = np.asarray(groups)
    return pd.concat(
        [
            pd.DataFrame(columns),
            confounders.reset_index(drop=True),
        ],
        axis=1,
    )


def observation_groups(data: CausalDataset) -> np.ndarray | None:
    if data.group_id is None:
        return None
    return data.df[data.group_id].to_numpy()


def doubleml_data_from_frame(frame: pd.DataFrame) -> DoubleMLData:
    if "cluster" in frame.columns:
        return DoubleMLData(frame, y_col="y", d_cols="d", cluster_cols="cluster")
    return DoubleMLData(frame, y_col="y", d_cols="d")


def factual_outcome_predictions(
    outcome_model: Standardization,
    X: pd.DataFrame,
    treatment: pd.Series,
    treatment_arms: list[JsonValue],
) -> np.ndarray:
    """Predicted outcome under each unit's observed treatment arm."""
    matrix = np.asarray(outcome_model.estimate_individual_outcome(X, treatment))
    if matrix.ndim == 1:
        return matrix
    if matrix.ndim != 2:
        raise ValueError(f"Unexpected outcome prediction shape: {matrix.shape}")
    level_to_column = {level: index for index, level in enumerate(treatment_arms)}
    columns = np.array([level_to_column[value] for value in treatment.to_numpy()])
    rows = np.arange(len(treatment))
    return matrix[rows, columns]


def treated_propensity(fitted_propensity: np.ndarray) -> np.ndarray:
    """P(T = treated | X) from a 1-d score or an arm-by-column matrix."""
    propensity = np.asarray(fitted_propensity)
    if propensity.ndim == 1:
        return propensity
    if propensity.ndim != 2 or propensity.shape[1] < 2:
        raise ValueError("Expected a binary propensity vector or a 2+ column propensity matrix.")
    return propensity[:, 1]


class IPWAdapter(BaseATEEstimator):
    def __init__(self, spec: IPWATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.propensity: pd.DataFrame | None = None
        self.groups: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> IPWAdapter:
        require_classifier(self.spec.propensity_learner, "IPW propensity_learner")
        self.treatment = data.treatment_series.copy()
        self.outcome = data.outcome_series.copy()
        _, propensity = cross_fit_nuisances(
            covariates=data.X_adjustment.copy(),
            treatment=self.treatment,
            outcome=self.outcome,
            treatment_values=data.treatment_values,
            propensity_learner=self.spec.propensity_learner,
            outcome_learner=None,
            binary_outcome=data.outcome_type == OutcomeType.BINARY,
            n_folds=self.spec.n_folds,
            random_state=learner_random_state(
                self.spec.propensity_learner,
                self.spec.random_state,
            ),
            clip_bounds=self.spec.clip_bounds,
            groups=observation_groups(data),
        )
        self.propensity = clip_propensity(propensity, self.spec.clip_bounds)
        self.groups = observation_groups(data)
        return self

    def estimate(self) -> pd.DataFrame:
        if self.treatment is None or self.outcome is None or self.propensity is None:
            raise RuntimeError("Estimator is not fitted.")
        if self.spec.weighting == IPWWeighting.HORVITZ_THOMPSON:
            arm_scores = horvitz_thompson_scores_by_arm(self.outcome, self.treatment, self.propensity)
        else:
            arm_scores = hajek_scores_by_arm(self.outcome, self.treatment, self.propensity)
        table = contrasts_from_arm_scores(
            arm_scores=arm_scores,
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=self.spec.confidence_level,
            estimand=self.spec.estimand,
            groups=self.groups,
        )
        table = annotate_propensity_clip(table, self.spec.clip_bounds)
        table["variance_assumption"] = VARIANCE_PROPENSITY_TREATED_AS_KNOWN
        return table


class AIPWAdapter(BaseATEEstimator):
    def __init__(self, spec: DoublyRobustATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.propensity: pd.DataFrame | None = None
        self.potential_outcomes: pd.DataFrame | None = None
        self.outcome_predictions: np.ndarray | None = None
        self.groups: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> AIPWAdapter:
        binary_outcome = data.outcome_type == OutcomeType.BINARY
        require_classifier(self.spec.propensity_learner, "AIPW propensity_learner")
        if binary_outcome:
            require_classifier(self.spec.outcome_learner, "AIPW outcome_learner")
        self.treatment = data.treatment_series.copy()
        self.outcome = data.outcome_series.copy()
        potential_outcomes, propensity = cross_fit_nuisances(
            covariates=data.X_adjustment.copy(),
            treatment=self.treatment,
            outcome=self.outcome,
            treatment_values=data.treatment_values,
            propensity_learner=self.spec.propensity_learner,
            outcome_learner=self.spec.outcome_learner,
            binary_outcome=binary_outcome,
            n_folds=self.spec.n_folds,
            random_state=learner_random_state(
                self.spec.propensity_learner,
                self.spec.random_state,
            ),
            clip_bounds=self.spec.clip_bounds,
            groups=observation_groups(data),
        )
        if potential_outcomes is None:
            raise RuntimeError("AIPW cross-fit did not return potential outcomes.")
        self.potential_outcomes = potential_outcomes
        self.propensity = clip_propensity(propensity, self.spec.clip_bounds)
        self.outcome_predictions = factual_predictions(potential_outcomes, self.treatment)
        self.groups = observation_groups(data)
        return self

    def estimate(self) -> pd.DataFrame:
        if (
            self.treatment is None
            or self.outcome is None
            or self.propensity is None
            or self.potential_outcomes is None
        ):
            raise RuntimeError("Estimator is not fitted.")
        table = contrasts_from_arm_scores(
            arm_scores=aipw_scores_by_arm(
                self.outcome,
                self.treatment,
                self.potential_outcomes,
                self.propensity,
            ),
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=self.spec.confidence_level,
            estimand=self.spec.estimand,
            groups=self.groups,
        )
        return annotate_propensity_clip(table, self.spec.clip_bounds)


class TMLEAdapter(BaseATEEstimator):
    def __init__(self, spec: DoublyRobustATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.propensity: pd.DataFrame | None = None
        self.potential_outcomes: pd.DataFrame | None = None
        self.outcome_predictions: np.ndarray | None = None
        self.groups: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> TMLEAdapter:
        binary_outcome = data.outcome_type == OutcomeType.BINARY
        require_classifier(self.spec.propensity_learner, "TMLE propensity_learner")
        if binary_outcome:
            require_classifier(self.spec.outcome_learner, "TMLE outcome_learner")
        self.treatment = data.treatment_series.copy()
        self.outcome = data.outcome_series.copy()
        initial_outcomes, propensity = cross_fit_nuisances(
            covariates=data.X_adjustment.copy(),
            treatment=self.treatment,
            outcome=self.outcome,
            treatment_values=data.treatment_values,
            propensity_learner=self.spec.propensity_learner,
            outcome_learner=self.spec.outcome_learner,
            binary_outcome=binary_outcome,
            n_folds=self.spec.n_folds,
            random_state=learner_random_state(
                self.spec.propensity_learner,
                self.spec.random_state,
            ),
            clip_bounds=self.spec.clip_bounds,
            groups=observation_groups(data),
        )
        if initial_outcomes is None:
            raise RuntimeError("TMLE cross-fit did not return potential outcomes.")
        self.propensity = clip_propensity(propensity, self.spec.clip_bounds)
        self.potential_outcomes = target_potential_outcomes(
            outcome=self.outcome,
            treatment=self.treatment,
            potential_outcomes=initial_outcomes,
            propensity=self.propensity,
            treatment_values=data.treatment_values,
            control_value=data.control_value,
            reduced=self.spec.reduced,
        )
        self.outcome_predictions = factual_predictions(self.potential_outcomes, self.treatment)
        self.groups = observation_groups(data)
        return self

    def estimate(self) -> pd.DataFrame:
        if (
            self.treatment is None
            or self.outcome is None
            or self.propensity is None
            or self.potential_outcomes is None
        ):
            raise RuntimeError("Estimator is not fitted.")
        table = contrasts_from_arm_scores(
            arm_scores=aipw_scores_by_arm(
                self.outcome,
                self.treatment,
                self.potential_outcomes,
                self.propensity,
            ),
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=self.spec.confidence_level,
            estimand=self.spec.estimand,
            groups=self.groups,
        )
        return annotate_propensity_clip(table, self.spec.clip_bounds)


class DoubleMLIRMAdapter(BaseATEEstimator):
    def __init__(self, spec: DoubleMLATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None

    def fit(self, data: CausalDataset) -> DoubleMLIRMAdapter:
        if len(data.treatment_values) != 2:
            raise ValueError(
                "DoubleML IRM supports two treatment arms. Use DML_APOS for multi-arm treatment."
            )
        X = data.X_adjustment.copy()
        treatment = binary_treatment_indicator(data.treatment_series, data.control_value)
        outcome = data.outcome_series.copy()
        dml_data = doubleml_data_from_frame(
            doubleml_dataframe(outcome, treatment, X, observation_groups(data)),
        )
        ml_g = clone_scaled_estimator(self.spec.outcome_learner)
        ml_m = clone_scaled_estimator(self.spec.propensity_learner)
        require_classifier(ml_m, "DoubleML IRM propensity_learner")
        if data.outcome_type == OutcomeType.BINARY:
            require_classifier(ml_g, "DoubleML IRM outcome_learner")
        self.model = DoubleMLIRM(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            n_folds=self.spec.n_folds,
            n_rep=self.spec.n_rep,
            score="ATE",
        )
        self.model.fit()
        return self

    def estimate(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        summary = self.model.confint(level=self.spec.confidence_level)
        contrast = contrast_columns(self.data.control_value, self.data.treatment_values)[0]
        return pd.DataFrame(
            {
                "contrast": [contrast],
                "estimand": [self.spec.estimand],
                "estimate": [float(self.model.coef[0])],
                "ci_lower": [float(summary.iloc[0, 0])],
                "ci_upper": [float(summary.iloc[0, 1])],
            }
        )


class DoubleMLPLRAdapter(BaseATEEstimator):
    def __init__(self, spec: DoubleMLATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None

    def fit(self, data: CausalDataset) -> DoubleMLPLRAdapter:
        X = data.X_adjustment.copy()
        treatment = binary_treatment_indicator(data.treatment_series, data.control_value)
        outcome = data.outcome_series.copy()
        dml_data = doubleml_data_from_frame(
            doubleml_dataframe(outcome, treatment, X, observation_groups(data)),
        )
        ml_l = clone_scaled_estimator(self.spec.outcome_learner)
        ml_m = clone_scaled_estimator(self.spec.propensity_learner)
        require_classifier(ml_m, "DoubleML PLR propensity_learner")
        self.model = DoubleMLPLR(
            dml_data,
            ml_l=ml_l,
            ml_m=ml_m,
            n_folds=self.spec.n_folds,
            n_rep=self.spec.n_rep,
        )
        self.model.fit()
        return self

    def estimate(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        summary = self.model.confint(level=self.spec.confidence_level)
        contrast = contrast_columns(self.data.control_value, self.data.treatment_values)[0]
        return pd.DataFrame(
            {
                "contrast": [f"plr_{contrast}"],
                "estimand": ["partially_linear"],
                "estimate": [float(self.model.coef[0])],
                "ci_lower": [float(summary.iloc[0, 0])],
                "ci_upper": [float(summary.iloc[0, 1])],
            }
        )


class ClusterCapableDoubleMLAPOS(DoubleMLAPOS):
    """DoubleMLAPOS with the cluster sample splitting the other DoubleML models use.

    The installed DoubleMLAPOS calls the shared cluster splitter but never sets
    ``_n_folds_per_cluster``, and it copies only the row partition onto each
    potential-outcome model. Clustered data then cannot be fit. This subclass
    records the same fold counts as ``DoubleML`` and passes the cluster
    partition through.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        draw_sample_splitting = bool(kwargs.pop("draw_sample_splitting", True))
        super().__init__(*args, draw_sample_splitting=False, **kwargs)
        if self._is_cluster_data:
            self._n_folds_per_cluster = self._n_folds
            self._n_folds = self._n_folds ** self._dml_data.n_cluster_vars
        if draw_sample_splitting:
            self.draw_sample_splitting()
            self._initialize_dml_model()

    def _initialize_models(self) -> list[DoubleMLAPO]:
        if not self._is_cluster_data:
            return super()._initialize_models()
        modellist: list[DoubleMLAPO] = []
        model_arguments = {
            "obj_dml_data": self._dml_data,
            "ml_g": self._learner["ml_g"],
            "ml_m": self._learner["ml_m"],
            "score": self.score,
            "n_folds": self.n_folds,
            "n_rep": self.n_rep,
            "weights": self.weights,
            "ps_processor_config": self.ps_processor_config,
            "normalize_ipw": self.normalize_ipw,
            "draw_sample_splitting": False,
        }
        for level_index in range(self.n_treatment_levels):
            model = DoubleMLAPO(treatment_level=self._treatment_levels[level_index], **model_arguments)
            model.set_sample_splitting(
                all_smpls=self.smpls,
                all_smpls_cluster=self._smpls_cluster,
            )
            modellist.append(model)
        return modellist

    def fit(
        self,
        n_jobs_models: int | None = None,
        n_jobs_cv: int | None = None,
        store_predictions: bool = True,
        store_models: bool = False,
        external_predictions: dict | None = None,
    ) -> ClusterCapableDoubleMLAPOS:
        if not self._is_cluster_data:
            return super().fit(
                n_jobs_models=n_jobs_models,
                n_jobs_cv=n_jobs_cv,
                store_predictions=store_predictions,
                store_models=store_models,
                external_predictions=external_predictions,
            )
        # DoubleMLAPOS.fit concatenates one framework per treatment level.
        # That concatenation is not implemented for clustered scores, so the
        # level-specific models are kept and contrasts are formed later.
        if external_predictions is not None:
            self._check_external_predictions(external_predictions)
            external_by_level = self._rename_external_predictions(external_predictions)
        else:
            external_by_level = None
        fitted_models = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")(
            delayed(self._fit_model)(
                level_index,
                n_jobs_cv,
                store_predictions,
                store_models,
                external_by_level,
            )
            for level_index in range(self.n_treatment_levels)
        )
        for level_index, fitted in enumerate(fitted_models):
            self._modellist[level_index] = fitted
        return self


class DoubleMLAPOSAdapter(BaseATEEstimator):
    def __init__(self, spec: DoubleMLATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None

    def fit(self, data: CausalDataset) -> DoubleMLAPOSAdapter:
        X = data.X_adjustment.copy()
        treatment = data.treatment_series.copy()
        outcome = data.outcome_series.copy()
        dml_data = doubleml_data_from_frame(
            doubleml_dataframe(outcome, treatment, X, observation_groups(data)),
        )
        ml_g = clone_scaled_estimator(self.spec.outcome_learner)
        ml_m = clone_scaled_estimator(self.spec.propensity_learner)
        require_classifier(ml_m, "DoubleML APOS propensity_learner")
        if data.outcome_type == OutcomeType.BINARY:
            require_classifier(ml_g, "DoubleML APOS outcome_learner")
        self.model = ClusterCapableDoubleMLAPOS(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            treatment_levels=self.data.treatment_values,
            n_folds=self.spec.n_folds,
            n_rep=self.spec.n_rep,
        )
        self.model.fit()
        return self

    def estimate(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        return apos_contrast_table(
            model=self.model,
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=self.spec.confidence_level,
            estimand=self.spec.estimand,
        )


def apos_contrast_table(
    model: ClusterCapableDoubleMLAPOS,
    control_value: JsonValue,
    treatment_values: list[JsonValue],
    level: float,
    estimand: str,
) -> pd.DataFrame:
    """Difference of potential outcomes, one contrast at a time.

    ``DoubleMLAPOS.causal_contrast`` concatenates those differences, and the
    installed DoubleML build cannot concatenate clustered frameworks. Each
    contrast is already a single clustered framework, so the interval comes
    from that difference directly.
    """
    levels = list(model.treatment_levels)
    reference = model.modellist[treatment_level_index(levels, control_value)].framework
    names = contrast_columns(control_value, treatment_values)
    arms = [value for value in treatment_values if value != control_value]
    rows = []
    for name, arm in zip(names, arms, strict=True):
        contrast = model.modellist[treatment_level_index(levels, arm)].framework - reference
        intervals = contrast.confint(level=level)
        rows.append(
            {
                "contrast": name,
                "estimand": estimand,
                "estimate": float(np.ravel(contrast.thetas)[0]),
                "ci_lower": float(intervals.iloc[0, 0]),
                "ci_upper": float(intervals.iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def treatment_level_index(levels: list[object], arm: JsonValue) -> int:
    for index, level in enumerate(levels):
        if level == arm:
            return index
    raise KeyError(arm)


def binary_treatment_indicator(treatment: pd.Series, control_value: JsonValue) -> pd.Series:
    """Code the control arm as 0 and every other arm as 1 for binary DoubleML models."""
    indicator = (treatment.to_numpy() != control_value).astype(int)
    return pd.Series(indicator, index=treatment.index, name=treatment.name)


def estimator_supports_sensitivity(estimator: BaseATEEstimator) -> bool:
    """DoubleML publishes a score-based omitted-variable bound. Other ATE fits do not."""
    return isinstance(estimator, (DoubleMLIRMAdapter, DoubleMLPLRAdapter)) and estimator.model is not None


def contrasts_from_arm_scores(
    arm_scores: dict[JsonValue, np.ndarray],
    control_value: JsonValue,
    treatment_values: list[JsonValue],
    level: float,
    estimand: str,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    control_scores = arm_scores[control_value]
    names = contrast_columns(control_value, treatment_values)
    arms = [arm for arm in treatment_values if arm != control_value]
    rows = []
    for name, arm in zip(names, arms, strict=True):
        estimate, lower, upper = normal_interval_from_scores(
            arm_scores[arm] - control_scores,
            level,
            groups=groups,
        )
        rows.append(
            {
                "contrast": name,
                "estimand": estimand,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def hajek_scores_by_arm(
    outcome: pd.Series,
    treatment: pd.Series,
    propensity: pd.DataFrame,
) -> dict[JsonValue, np.ndarray]:
    outcome_values = outcome.to_numpy(dtype=float)
    treatment_values = treatment.to_numpy()
    return {
        arm: hajek_influence(
            outcome=outcome_values,
            treatment=treatment_values,
            propensity=propensity[arm].to_numpy(dtype=float),
            arm=arm,
        )
        for arm in propensity.columns
    }


def horvitz_thompson_scores_by_arm(
    outcome: pd.Series,
    treatment: pd.Series,
    propensity: pd.DataFrame,
) -> dict[JsonValue, np.ndarray]:
    outcome_values = outcome.to_numpy(dtype=float)
    treatment_values = treatment.to_numpy()
    return {
        arm: horvitz_thompson_influence(
            outcome=outcome_values,
            treatment=treatment_values,
            propensity=propensity[arm].to_numpy(dtype=float),
            arm=arm,
        )
        for arm in propensity.columns
    }


def annotate_propensity_clip(
    table: pd.DataFrame,
    clip_bounds: tuple[float, float],
) -> pd.DataFrame:
    annotated = table.copy()
    annotated["clip_lower"] = clip_bounds[0]
    annotated["clip_upper"] = clip_bounds[1]
    return annotated


def aipw_scores_by_arm(
    outcome: pd.Series,
    treatment: pd.Series,
    potential_outcomes: pd.DataFrame,
    propensity: pd.DataFrame,
) -> dict[JsonValue, np.ndarray]:
    outcome_values = outcome.to_numpy(dtype=float)
    treatment_values = treatment.to_numpy()
    return {
        arm: aipw_arm_scores(
            outcome=outcome_values,
            treatment=treatment_values,
            potential_outcome=potential_outcomes[arm].to_numpy(dtype=float),
            propensity=propensity[arm].to_numpy(dtype=float),
            arm=arm,
        )
        for arm in potential_outcomes.columns
    }


def contrasts_from_population_outcomes(
    population: pd.DataFrame | pd.Series,
    control_value: JsonValue,
    treatment_values: list[JsonValue],
    estimand: str,
) -> pd.DataFrame:
    series = population_as_series(population)
    names = contrast_columns(control_value, treatment_values)
    arms = [arm for arm in treatment_values if arm != control_value]
    rows = []
    for name, arm in zip(names, arms, strict=True):
        estimate = float(
            population_outcome_for_arm(series, arm)
            - population_outcome_for_arm(series, control_value),
        )
        rows.append(
            {
                "contrast": name,
                "estimand": estimand,
                "estimate": estimate,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }
        )
    return pd.DataFrame(rows)


def population_as_series(population: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(population, pd.DataFrame):
        if population.shape[1] != 1:
            raise ValueError("Population outcome table must be a Series or a single-column DataFrame.")
        return population.iloc[:, 0]
    return population


def population_outcome_for_arm(population: pd.Series, arm: JsonValue) -> float:
    if arm in population.index:
        return float(population[arm])
    matches = [index for index in population.index if str(index) == str(arm)]
    if len(matches) == 1:
        return float(population[matches[0]])
    raise KeyError(arm)


class SensitivityResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    summary: pd.DataFrame
    benchmarks: pd.DataFrame
    contour_path: str | None


class ATESensitivityAnalyzer:
    """Omitted-variable sensitivity for dedicated ATE estimators."""

    def __init__(self, sensitivity: SensitivityConfig) -> None:
        self.sensitivity = sensitivity

    def analyze(
        self,
        estimator: BaseATEEstimator,
        estimator_id: str,
        results_root: str,
    ) -> SensitivityResult | None:
        sensitivity_config = self.sensitivity
        if estimator_supports_sensitivity(estimator):
            return self.analyze_doubleml_native(
                estimator.model,
                estimator_id,
                results_root,
                sensitivity_config,
            )
        logger.info(
            "Skipping omitted-variable sensitivity for %s. "
            "The bound is reported only for cross-fit DoubleML scores.",
            estimator_id,
        )
        return None

    def analyze_doubleml_native(
        self,
        model: DoubleMLIRM | DoubleMLPLR,
        estimator_id: str,
        results_root: str,
        sensitivity_config: SensitivityConfig,
    ) -> SensitivityResult:
        model.sensitivity_analysis(
            level=sensitivity_config.confidence_level,
            null_hypothesis=sensitivity_config.null_effect,
        )
        sensitivity_params = model.sensitivity_params
        if sensitivity_params is None:
            raise RuntimeError("DoubleML sensitivity analysis did not populate sensitivity_params.")
        point_rv = float(np.ravel(sensitivity_params["rv"])[0])
        interval_rv = float(np.ravel(sensitivity_params["rva"])[0])
        benchmarks = doubleml_covariate_benchmarks(
            model=model,
            null_effect=sensitivity_config.null_effect,
            level=sensitivity_config.confidence_level,
        )
        best = benchmarks.iloc[benchmarks["benchmark_bias"].abs().argmax()]
        contour_dir = Path(results_root) / "ate" / estimator_id / "sensitivity"
        figure = model.sensitivity_plot(
            level=sensitivity_config.confidence_level,
            null_hypothesis=sensitivity_config.null_effect,
        )
        contour_path = str(write_html(contour_dir / "contour.html", figure))
        summary = pd.DataFrame(
            {
                "robustness_value_point": [point_rv],
                "robustness_value_interval": [interval_rv],
                "benchmark_variable": [best["covariate"]],
                "benchmark_treatment_strength": [best["benchmark_treatment_strength"]],
                "benchmark_outcome_strength": [best["benchmark_outcome_strength"]],
                "benchmark_bias": [best["benchmark_bias"]],
                "benchmark_multiple_to_null": [best["benchmark_multiple_to_null"]],
                "benchmark_multiple_to_null_interval": [best["benchmark_multiple_to_null_interval"]],
            }
        )
        return SensitivityResult(
            summary=summary,
            benchmarks=benchmarks,
            contour_path=contour_path,
        )


def doubleml_covariate_benchmarks(
    model: DoubleMLIRM | DoubleMLPLR,
    null_effect: float,
    level: float,
) -> pd.DataFrame:
    rows = []
    for covariate in model._dml_data.x_cols:
        benchmark = model.sensitivity_benchmark(benchmarking_set=[covariate]).iloc[0]
        treatment_strength = float(benchmark["cf_d"])
        outcome_strength = float(benchmark["cf_y"])
        rows.append(
            {
                "covariate": covariate,
                "benchmark_treatment_strength": treatment_strength,
                "benchmark_outcome_strength": outcome_strength,
                "benchmark_bias": float(benchmark["delta_theta"]),
                "benchmark_multiple_to_null": confounding_multiple_to_null(
                    model=model,
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    null_effect=null_effect,
                    level=level,
                    bound=SENSITIVITY_POINT_BOUND,
                ),
                "benchmark_multiple_to_null_interval": confounding_multiple_to_null(
                    model=model,
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    null_effect=null_effect,
                    level=level,
                    bound=SENSITIVITY_INTERVAL_BOUND,
                ),
            }
        )
    return pd.DataFrame(rows)


def confounding_multiple_to_null(
    model: DoubleMLIRM | DoubleMLPLR,
    treatment_strength: float,
    outcome_strength: float,
    null_effect: float,
    level: float,
    bound: str,
) -> float:
    """Smallest proportional multiple of a covariate benchmark whose bound covers the null."""
    if treatment_strength <= 0.0 and outcome_strength <= 0.0:
        return float("inf")
    largest_multiple = largest_proportional_multiple(
        treatment_strength=treatment_strength,
        outcome_strength=outcome_strength,
    )

    def covers_null(multiple: float) -> bool:
        lower, upper = doubleml_sensitivity_bounds(
            model=model,
            treatment_strength=multiple * treatment_strength,
            outcome_strength=multiple * outcome_strength,
            level=level,
            bound=bound,
        )
        return lower <= null_effect <= upper

    if covers_null(0.0):
        return 0.0
    if not covers_null(largest_multiple):
        return float("inf")
    lower_multiple = 0.0
    upper_multiple = largest_multiple
    for attempt in range(SENSITIVITY_BISECT_ITERATIONS):
        midpoint = (lower_multiple + upper_multiple) / 2.0
        if covers_null(midpoint):
            upper_multiple = midpoint
        else:
            lower_multiple = midpoint
    return float(upper_multiple)


def largest_proportional_multiple(treatment_strength: float, outcome_strength: float) -> float:
    peak = max(treatment_strength, outcome_strength)
    if peak <= 0.0:
        return 0.0
    return SENSITIVITY_STRENGTH_CAP / peak


def doubleml_sensitivity_bounds(
    model: DoubleMLIRM | DoubleMLPLR,
    treatment_strength: float,
    outcome_strength: float,
    level: float,
    bound: str,
) -> tuple[float, float]:
    result = model._framework._calc_sensitivity_analysis(
        cf_y=float(outcome_strength),
        cf_d=float(treatment_strength),
        rho=SENSITIVITY_CONFOUNDING_RHO,
        level=level,
    )
    lower = float(np.ravel(result[bound]["lower"])[0])
    upper = float(np.ravel(result[bound]["upper"])[0])
    return lower, upper

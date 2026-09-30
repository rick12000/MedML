"""ATE estimator adapters, confidence intervals, and sensitivity."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLData
from doubleml.irm import DoubleMLAPOS, DoubleMLIRM
from doubleml.plm import DoubleMLPLR
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator

from causal_pipeline.config import (
    DEFAULT_ATE_CONFIDENCE_LEVEL,
    DEFAULT_DML_N_FOLDS,
    ATEEstimatorSpec,
    ATEKind,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    IPWATEEstimatorSpec,
    JsonValue,
    OutcomeType,
    SensitivityConfig,
    clone_estimator,
    require_classifier,
)
from causal_pipeline.crossfit import (
    aipw_arm_scores,
    clip_propensity,
    cross_fit_nuisances,
    factual_predictions,
    hajek_influence,
    learner_random_state,
    normal_interval_from_scores,
    target_potential_outcomes,
)
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.utils import write_html

logger = logging.getLogger(__name__)

class BaseATEEstimator(ABC):
    """Common interface for average treatment effect estimators."""

    @abstractmethod
    def fit(self, data: CausalDataset) -> BaseATEEstimator:
        ...

    @abstractmethod
    def estimate(self) -> pd.DataFrame:
        ...


def create_ate_estimator(
    spec: ATEEstimatorSpec,
    data: CausalDataset,
    sensitivity_outcome_learner: BaseEstimator | None = None,
) -> BaseATEEstimator:
    del sensitivity_outcome_learner
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
) -> pd.DataFrame:
    """Align outcome, treatment, and confounders on a RangeIndex for DoubleML."""
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "y": np.asarray(outcome),
                    "d": np.asarray(treatment),
                }
            ),
            confounders.reset_index(drop=True),
        ],
        axis=1,
    )


def factual_outcome_predictions(
    outcome_model: Standardization,
    X: pd.DataFrame,
    treatment: pd.Series,
    treatment_values: list[JsonValue],
) -> np.ndarray:
    """Predicted outcome under each unit's observed treatment arm."""
    matrix = np.asarray(outcome_model.estimate_individual_outcome(X, treatment))
    if matrix.ndim == 1:
        return matrix
    if matrix.ndim != 2:
        raise ValueError(f"Unexpected outcome prediction shape: {matrix.shape}")
    level_to_column = {level: index for index, level in enumerate(treatment_values)}
    columns = np.array([level_to_column[value] for value in treatment.to_numpy()])
    rows = np.arange(len(treatment))
    return matrix[rows, columns]


def treated_propensity(fitted_propensity: object) -> np.ndarray:
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

    def fit(self, data: CausalDataset) -> IPWAdapter:
        require_classifier(self.spec.propensity_learner, "IPW propensity_learner")
        self.treatment = data.treatment_series.copy()
        self.outcome = data.outcome_series.copy()
        _, propensity = cross_fit_nuisances(
            X=data.X_adjustment.copy(),
            treatment=self.treatment,
            outcome=self.outcome,
            treatment_values=data.treatment_values,
            propensity_learner=self.spec.propensity_learner,
            outcome_learner=None,
            binary_outcome=data.outcome_type == OutcomeType.BINARY,
            n_folds=DEFAULT_DML_N_FOLDS,
            random_state=learner_random_state(self.spec.propensity_learner),
            clip_bounds=self.spec.clip_bounds,
        )
        self.propensity = clip_propensity(propensity, self.spec.clip_bounds)
        return self

    def estimate(self) -> pd.DataFrame:
        if self.treatment is None or self.outcome is None or self.propensity is None:
            raise RuntimeError("Estimator is not fitted.")
        return contrasts_from_arm_scores(
            arm_scores=_hajek_arm_scores(self.outcome, self.treatment, self.propensity),
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=DEFAULT_ATE_CONFIDENCE_LEVEL,
        )

class AIPWAdapter(BaseATEEstimator):
    def __init__(self, spec: DoublyRobustATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.propensity: pd.DataFrame | None = None
        self.potential_outcomes: pd.DataFrame | None = None
        self.outcome_predictions: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> AIPWAdapter:
        binary_outcome = data.outcome_type == OutcomeType.BINARY
        require_classifier(self.spec.propensity_learner, "AIPW propensity_learner")
        if binary_outcome:
            require_classifier(self.spec.outcome_learner, "AIPW outcome_learner")
        self.treatment = data.treatment_series.copy()
        self.outcome = data.outcome_series.copy()
        potential_outcomes, propensity = cross_fit_nuisances(
            X=data.X_adjustment.copy(),
            treatment=self.treatment,
            outcome=self.outcome,
            treatment_values=data.treatment_values,
            propensity_learner=self.spec.propensity_learner,
            outcome_learner=self.spec.outcome_learner,
            binary_outcome=binary_outcome,
            n_folds=DEFAULT_DML_N_FOLDS,
            random_state=learner_random_state(self.spec.propensity_learner),
            clip_bounds=self.spec.clip_bounds,
        )
        if potential_outcomes is None:
            raise RuntimeError("AIPW cross-fit did not return potential outcomes.")
        self.potential_outcomes = potential_outcomes
        self.propensity = clip_propensity(propensity, self.spec.clip_bounds)
        self.outcome_predictions = factual_predictions(potential_outcomes, self.treatment)
        return self

    def estimate(self) -> pd.DataFrame:
        if (
            self.treatment is None
            or self.outcome is None
            or self.propensity is None
            or self.potential_outcomes is None
        ):
            raise RuntimeError("Estimator is not fitted.")
        return contrasts_from_arm_scores(
            arm_scores=_aipw_arm_scores(
                self.outcome,
                self.treatment,
                self.potential_outcomes,
                self.propensity,
            ),
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=DEFAULT_ATE_CONFIDENCE_LEVEL,
        )

class TMLEAdapter(BaseATEEstimator):
    def __init__(self, spec: DoublyRobustATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.propensity: pd.DataFrame | None = None
        self.potential_outcomes: pd.DataFrame | None = None
        self.outcome_predictions: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> TMLEAdapter:
        binary_outcome = data.outcome_type == OutcomeType.BINARY
        require_classifier(self.spec.propensity_learner, "TMLE propensity_learner")
        if binary_outcome:
            require_classifier(self.spec.outcome_learner, "TMLE outcome_learner")
        self.treatment = data.treatment_series.copy()
        self.outcome = data.outcome_series.copy()
        initial_outcomes, propensity = cross_fit_nuisances(
            X=data.X_adjustment.copy(),
            treatment=self.treatment,
            outcome=self.outcome,
            treatment_values=data.treatment_values,
            propensity_learner=self.spec.propensity_learner,
            outcome_learner=self.spec.outcome_learner,
            binary_outcome=binary_outcome,
            n_folds=DEFAULT_DML_N_FOLDS,
            random_state=learner_random_state(self.spec.propensity_learner),
            clip_bounds=self.spec.clip_bounds,
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
        return self

    def estimate(self) -> pd.DataFrame:
        if (
            self.treatment is None
            or self.outcome is None
            or self.propensity is None
            or self.potential_outcomes is None
        ):
            raise RuntimeError("Estimator is not fitted.")
        return contrasts_from_arm_scores(
            arm_scores=_aipw_arm_scores(
                self.outcome,
                self.treatment,
                self.potential_outcomes,
                self.propensity,
            ),
            control_value=self.data.control_value,
            treatment_values=self.data.treatment_values,
            level=DEFAULT_ATE_CONFIDENCE_LEVEL,
        )

class DoubleMLIRMAdapter(BaseATEEstimator):
    def __init__(self, spec: DoubleMLATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None

    def fit(self, data: CausalDataset) -> DoubleMLIRMAdapter:
        X = data.X_adjustment.copy()
        treatment = binary_treatment_indicator(data.treatment_series, data.control_value)
        outcome = data.outcome_series.copy()
        dml_data = DoubleMLData(
            doubleml_dataframe(outcome, treatment, X),
            y_col="y",
            d_cols="d",
        )
        ml_g = clone_estimator(self.spec.outcome_learner)
        ml_m = clone_estimator(self.spec.propensity_learner)
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
                "estimand": ["ate"],
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
        dml_data = DoubleMLData(
            doubleml_dataframe(outcome, treatment, X),
            y_col="y",
            d_cols="d",
        )
        ml_l = clone_estimator(self.spec.outcome_learner)
        ml_m = clone_estimator(self.spec.propensity_learner)
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

class DoubleMLAPOSAdapter(BaseATEEstimator):
    def __init__(self, spec: DoubleMLATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None

    def fit(self, data: CausalDataset) -> DoubleMLAPOSAdapter:
        X = data.X_adjustment.copy()
        treatment = data.treatment_series.copy()
        outcome = data.outcome_series.copy()
        dml_data = DoubleMLData(
            doubleml_dataframe(outcome, treatment, X),
            y_col="y",
            d_cols="d",
        )
        ml_g = clone_estimator(self.spec.outcome_learner)
        ml_m = clone_estimator(self.spec.propensity_learner)
        require_classifier(ml_m, "DoubleML APOS propensity_learner")
        if data.outcome_type == OutcomeType.BINARY:
            require_classifier(ml_g, "DoubleML APOS outcome_learner")
        self.model = DoubleMLAPOS(
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
        framework = self.model.causal_contrast(reference_levels=self.data.control_value)
        intervals = framework.confint(level=self.spec.confidence_level)
        names = contrast_columns(self.data.control_value, self.data.treatment_values)
        rows = []
        for index, name in enumerate(names):
            rows.append(
                {
                    "contrast": name,
                    "estimand": "ate",
                    "estimate": float(framework.thetas[index]),
                    "ci_lower": float(intervals.iloc[index, 0]),
                    "ci_upper": float(intervals.iloc[index, 1]),
                }
            )
        return pd.DataFrame(rows)

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
    estimand: str = "ate",
) -> pd.DataFrame:
    control_scores = arm_scores[control_value]
    names = contrast_columns(control_value, treatment_values)
    arms = [arm for arm in treatment_values if arm != control_value]
    rows = []
    for name, arm in zip(names, arms, strict=True):
        estimate, lower, upper = normal_interval_from_scores(arm_scores[arm] - control_scores, level)
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


def _hajek_arm_scores(
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


def _aipw_arm_scores(
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
) -> pd.DataFrame:
    series = _population_series(population)
    names = contrast_columns(control_value, treatment_values)
    arms = [arm for arm in treatment_values if arm != control_value]
    rows = []
    for name, arm in zip(names, arms, strict=True):
        estimate = float(
            _population_outcome(series, arm) - _population_outcome(series, control_value),
        )
        rows.append(
            {
                "contrast": name,
                "estimand": "ate",
                "estimate": estimate,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _population_series(population: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(population, pd.DataFrame):
        if population.shape[1] != 1:
            raise ValueError("Population outcome table must be a Series or a single-column DataFrame.")
        return population.iloc[:, 0]
    return population


def _population_outcome(population: pd.Series, arm: JsonValue) -> float:
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
        training_data: CausalDataset,
        estimator_id: str,
        results_root: str,
    ) -> SensitivityResult | None:
        del training_data
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
        rv = float(np.ravel(sensitivity_params["rv"])[0])
        benchmarks = _doubleml_covariate_benchmarks(
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
                "robustness_value": [rv],
                "benchmark_variable": [best["covariate"]],
                "benchmark_treatment_strength": [best["benchmark_treatment_strength"]],
                "benchmark_outcome_strength": [best["benchmark_outcome_strength"]],
                "benchmark_bias": [best["benchmark_bias"]],
                "benchmark_multiple_to_null": [best["benchmark_multiple_to_null"]],
            }
        )
        return SensitivityResult(
            summary=summary,
            benchmarks=benchmarks,
            contour_path=contour_path,
        )


def _doubleml_covariate_benchmarks(
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
                "benchmark_multiple_to_null": _confounding_multiple_to_null(
                    model=model,
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    null_effect=null_effect,
                    level=level,
                ),
            }
        )
    return pd.DataFrame(rows)


def _confounding_multiple_to_null(
    model: DoubleMLIRM | DoubleMLPLR,
    treatment_strength: float,
    outcome_strength: float,
    null_effect: float,
    level: float,
) -> float:
    """Smallest multiple of a covariate benchmark whose DoubleML point bound covers the null."""
    if treatment_strength <= 0.0 and outcome_strength <= 0.0:
        return float("inf")

    def covers_null(multiple: float) -> bool:
        lower, upper = _doubleml_point_bounds(
            model=model,
            treatment_strength=min(multiple * treatment_strength, 0.9999),
            outcome_strength=min(multiple * outcome_strength, 0.9999),
            level=level,
        )
        return lower <= null_effect <= upper

    if covers_null(0.0):
        return 0.0
    upper_multiple = 10.0
    if not covers_null(upper_multiple):
        return float("inf")
    lower_multiple = 0.0
    for _ in range(30):
        midpoint = (lower_multiple + upper_multiple) / 2.0
        if covers_null(midpoint):
            upper_multiple = midpoint
        else:
            lower_multiple = midpoint
    return float(upper_multiple)


def _doubleml_point_bounds(
    model: DoubleMLIRM | DoubleMLPLR,
    treatment_strength: float,
    outcome_strength: float,
    level: float,
) -> tuple[float, float]:
    result = model._framework._calc_sensitivity_analysis(
        cf_y=outcome_strength,
        cf_d=treatment_strength,
        rho=1.0,
        level=level,
    )
    lower = float(np.ravel(result["theta"]["lower"])[0])
    upper = float(np.ravel(result["theta"]["upper"])[0])
    return lower, upper

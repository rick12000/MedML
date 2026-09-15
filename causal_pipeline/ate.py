"""ATE estimator adapters, confidence intervals, and sensitivity."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causallib.estimation import AIPW, IPW, Standardization, TMLE
from doubleml import DoubleMLData
from doubleml.irm import DoubleMLAPOS, DoubleMLIRM
from doubleml.plm import DoubleMLPLR
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LinearRegression

from causal_pipeline.config import (
    ATEEstimatorSpec,
    ATEKind,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    IPWATEEstimatorSpec,
    JsonValue,
    LearnerSpec,
    OutcomeType,
    SensitivityConfig,
    build_sklearn_learner,
)
from causal_pipeline.data import CausalDataset

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
    sensitivity_outcome_learner: LearnerSpec | None = None,
) -> BaseATEEstimator:
    if spec.kind == ATEKind.IPW:
        return IPWAdapter(
            spec=spec,
            data=data,
            sensitivity_outcome_learner=sensitivity_outcome_learner,
        )
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


class IPWAdapter(BaseATEEstimator):
    def __init__(
        self,
        spec: IPWATEEstimatorSpec,
        data: CausalDataset,
        sensitivity_outcome_learner: LearnerSpec | None = None,
    ) -> None:
        self.spec = spec
        self.data = data
        self.sensitivity_outcome_learner = sensitivity_outcome_learner
        self.model = None
        self.X: pd.DataFrame | None = None
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.fitted_propensity: np.ndarray | None = None
        self.outcome_predictions: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> IPWAdapter:
        X = data.X_confounders
        treatment = data.treatment_series
        outcome = data.outcome_series
        learner = build_sklearn_learner(self.spec.propensity_learner)
        clip_min, clip_max = self.spec.clip_bounds
        self.model = IPW(
            learner=learner,
            clip_min=clip_min,
            clip_max=clip_max,
            use_stabilized=self.spec.use_stabilized,
        )
        self.X = X.copy()
        self.treatment = treatment.copy()
        self.outcome = outcome.copy()
        self.model.fit(X, treatment)
        self.fitted_propensity = self.model.compute_propensity_matrix(X)
        if self.sensitivity_outcome_learner is not None:
            outcome_learner = build_sklearn_learner(self.sensitivity_outcome_learner)
            design = pd.concat(
                [treatment.reset_index(drop=True), X.reset_index(drop=True)],
                axis=1,
            )
            outcome_learner.fit(design, outcome)
            self.outcome_predictions = outcome_learner.predict(design).values
        return self

    def estimate(self) -> pd.DataFrame:
        if self.model is None or self.X is None or self.treatment is None or self.outcome is None:
            raise RuntimeError("Estimator is not fitted.")
        population = self.model.estimate_population_outcome(
            self.X,
            self.treatment,
            self.outcome,
        )
        return contrasts_from_population_outcomes(
            population,
            self.data.control_value,
            self.data.treatment_values,
        )

class AIPWAdapter(BaseATEEstimator):
    def __init__(self, spec: DoublyRobustATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.X: pd.DataFrame | None = None
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.fitted_propensity: np.ndarray | None = None
        self.outcome_predictions: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> AIPWAdapter:
        X = data.X_confounders
        treatment = data.treatment_series
        outcome = data.outcome_series
        outcome_learner = build_sklearn_learner(self.spec.outcome_learner)
        propensity_learner = build_sklearn_learner(self.spec.propensity_learner)
        predict_proba = self.data.outcome_type == OutcomeType.BINARY
        outcome_model = Standardization(
            outcome_learner,
            encode_treatment=True,
            predict_proba=predict_proba,
        )
        clip_min, clip_max = self.spec.clip_bounds
        weight_model = IPW(
            learner=propensity_learner,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.model = AIPW(outcome_model=outcome_model, weight_model=weight_model)
        self.X = X.copy()
        self.treatment = treatment.copy()
        self.outcome = outcome.copy()
        self.model.fit(X, treatment, outcome)
        self.fitted_propensity = weight_model.compute_propensity_matrix(X)
        design = pd.concat([treatment.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        self.outcome_predictions = outcome_model.predict(design, treatment).values
        return self

    def estimate(self) -> pd.DataFrame:
        if self.model is None or self.X is None or self.treatment is None or self.outcome is None:
            raise RuntimeError("Estimator is not fitted.")
        population = self.model.estimate_population_outcome(
            self.X,
            self.treatment,
            self.outcome,
        )
        return contrasts_from_population_outcomes(
            population,
            self.data.control_value,
            self.data.treatment_values,
        )

class TMLEAdapter(BaseATEEstimator):
    def __init__(self, spec: DoublyRobustATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.weight_model = None
        self.outcome_model = None
        self.X: pd.DataFrame | None = None
        self.treatment: pd.Series | None = None
        self.outcome: pd.Series | None = None
        self.fitted_propensity: np.ndarray | None = None
        self.outcome_predictions: np.ndarray | None = None

    def fit(self, data: CausalDataset) -> TMLEAdapter:
        X = data.X_confounders
        treatment = data.treatment_series
        outcome = data.outcome_series
        outcome_learner = build_sklearn_learner(self.spec.outcome_learner)
        propensity_learner = build_sklearn_learner(self.spec.propensity_learner)
        predict_proba = self.data.outcome_type == OutcomeType.BINARY
        self.outcome_model = Standardization(
            outcome_learner,
            encode_treatment=True,
            predict_proba=predict_proba,
        )
        clip_min, clip_max = self.spec.clip_bounds
        self.weight_model = IPW(
            learner=propensity_learner,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.model = TMLE(
            outcome_model=self.outcome_model,
            weight_model=self.weight_model,
            reduced=self.spec.reduced,
        )
        self.X = X.copy()
        self.treatment = treatment.copy()
        self.outcome = outcome.copy()
        self.model.fit(X, treatment, outcome)
        self.fitted_propensity = self.weight_model.compute_propensity_matrix(X)
        design = pd.concat([treatment.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        self.outcome_predictions = self.outcome_model.predict(design, treatment).values
        return self

    def estimate(self) -> pd.DataFrame:
        if self.model is None or self.X is None or self.treatment is None or self.outcome is None:
            raise RuntimeError("Estimator is not fitted.")
        population = self.model.estimate_population_outcome(
            self.X,
            self.treatment,
            self.outcome,
        )
        return contrasts_from_population_outcomes(
            population,
            self.data.control_value,
            self.data.treatment_values,
        )

class DoubleMLIRMAdapter(BaseATEEstimator):
    def __init__(self, spec: DoubleMLATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None

    def fit(self, data: CausalDataset) -> DoubleMLIRMAdapter:
        X = data.X_confounders
        treatment = data.treatment_series
        outcome = data.outcome_series
        df_dml = pd.DataFrame({"y": outcome, "d": treatment})
        df_dml = pd.concat([df_dml, X.reset_index(drop=True)], axis=1)
        dml_data = DoubleMLData(df_dml, y_col="y", d_cols="d")
        ml_g = build_sklearn_learner(self.spec.outcome_learner)
        ml_m = build_sklearn_learner(self.spec.propensity_learner)
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
        return pd.DataFrame(
            {
                "contrast": ["1_vs_0"],
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
        X = data.X_confounders
        treatment = data.treatment_series
        outcome = data.outcome_series
        df_dml = pd.DataFrame({"y": outcome, "d": treatment})
        df_dml = pd.concat([df_dml, X.reset_index(drop=True)], axis=1)
        dml_data = DoubleMLData(df_dml, y_col="y", d_cols="d")
        ml_l = build_sklearn_learner(self.spec.outcome_learner)
        ml_m = build_sklearn_learner(self.spec.propensity_learner)
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
        return pd.DataFrame(
            {
                "contrast": ["1_vs_0"],
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
        X = data.X_confounders
        treatment = data.treatment_series
        outcome = data.outcome_series
        df_dml = pd.DataFrame({"y": outcome, "d": treatment})
        df_dml = pd.concat([df_dml, X.reset_index(drop=True)], axis=1)
        dml_data = DoubleMLData(df_dml, y_col="y", d_cols="d")
        ml_g = build_sklearn_learner(self.spec.outcome_learner)
        ml_m = build_sklearn_learner(self.spec.propensity_learner)
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
        contrasts = self.model.causal_contrast(reference_levels=self.data.control_value)
        rows = []
        for contrast_name in contrasts.columns:
            rows.append(
                {
                    "contrast": contrast_name,
                    "estimate": float(contrasts[contrast_name].iloc[0]),
                    "ci_lower": float(contrasts[contrast_name].iloc[1]),
                    "ci_upper": float(contrasts[contrast_name].iloc[2]),
                }
            )
        return pd.DataFrame(rows)

def contrasts_from_population_outcomes(
    population: pd.DataFrame,
    control_value: JsonValue,
    treatment_values: list[JsonValue],
) -> pd.DataFrame:
    control_key = str(control_value)
    rows = []
    for arm in treatment_values:
        if arm == control_value:
            continue
        arm_key = str(arm)
        estimate = float(population[arm_key] - population[control_key])
        rows.append(
            {
                "contrast": f"{arm}_vs_{control_value}",
                "estimate": estimate,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }
        )
    return pd.DataFrame(rows)


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
    ) -> SensitivityResult:
        sensitivity_config = self.sensitivity
        if isinstance(estimator, DoubleMLIRMAdapter) and estimator.model is not None:
            return self.analyze_doubleml_native(
                estimator.model,
                estimator_id,
                results_root,
                sensitivity_config,
            )
        if isinstance(estimator, DoubleMLPLRAdapter) and estimator.model is not None:
            return self.analyze_doubleml_native(
                estimator.model,
                estimator_id,
                results_root,
                sensitivity_config,
            )
        return self.analyze_nonlinear(
            estimator=estimator,
            training_data=training_data,
            estimator_id=estimator_id,
            results_root=results_root,
            sensitivity_config=sensitivity_config,
        )

    def analyze_doubleml_native(
        self,
        model: DoubleMLIRM | DoubleMLPLR,
        estimator_id: str,
        results_root: str,
        sensitivity_config: SensitivityConfig,
    ) -> SensitivityResult:
        cf = float(sensitivity_config.confidence_level)
        analysis = model.sensitivity_analysis(cf=cf)
        benchmark = model.sensitivity_benchmark(cf=cf)
        contour_dir = f"{results_root}/ate/{estimator_id}/sensitivity"
        contour_path = f"{contour_dir}/contour.png"
        model.sensitivity_plot(filename=contour_path)
        summary = pd.DataFrame(
            {
                "robustness_value": [float(analysis["rv"].iloc[0])],
                "benchmark_variable": [str(benchmark.iloc[0, 0])],
                "benchmark_treatment_strength": [float(benchmark.iloc[0, 1])],
                "benchmark_outcome_strength": [float(benchmark.iloc[0, 2])],
                "benchmark_bias": [float(benchmark.iloc[0, 3])],
                "benchmark_multiple_to_null": [float(benchmark.iloc[0, 4])],
            }
        )
        return SensitivityResult(
            summary=summary,
            benchmarks=benchmark,
            contour_path=contour_path,
        )

    def analyze_nonlinear(
        self,
        estimator: BaseATEEstimator,
        training_data: CausalDataset,
        estimator_id: str,
        results_root: str,
        sensitivity_config: SensitivityConfig,
    ) -> SensitivityResult:
        estimate_table = estimator.estimate()
        tau_hat = float(estimate_table["estimate"].iloc[0])
        treatment = training_data.treatment_series.values
        outcome = training_data.outcome_series.values.astype(float)
        X = training_data.X_confounders

        if not hasattr(estimator, "fitted_propensity") or estimator.fitted_propensity is None:
            raise ValueError("Nonlinear sensitivity requires fitted propensity on the estimator adapter.")

        propensity = estimator.fitted_propensity
        if propensity.ndim > 1:
            propensity = propensity[:, 1]

        if not hasattr(estimator, "outcome_predictions") or estimator.outcome_predictions is None:
            raise ValueError("Nonlinear sensitivity requires outcome predictions on the estimator adapter.")

        predictions = estimator.outcome_predictions
        treated = (treatment != training_data.control_value).astype(float)
        omega = treated / propensity - (1.0 - treated) / (1.0 - propensity)
        residuals = outcome - predictions
        sensitivity_scale = float(np.sqrt(np.mean(residuals ** 2) * np.mean(omega ** 2)))

        benchmark_rows = []
        for column in X.columns:
            z = X[column].astype(float).values.reshape(-1, 1)
            cy2 = partial_r2_outcome_nonlinear(residuals, z)
            q_val = partial_q_propensity(omega, z, X.drop(columns=[column]))
            bias = sensitivity_scale * np.sqrt(cy2 * q_val / (1.0 - q_val))
            benchmark_rows.append(
                {
                    "covariate": column,
                    "benchmark_treatment_strength": q_val,
                    "benchmark_outcome_strength": cy2,
                    "benchmark_bias": bias,
                }
            )
        benchmarks = pd.DataFrame(benchmark_rows)
        best = benchmarks.iloc[benchmarks["benchmark_bias"].abs().argmax()]
        rv = robustness_value_equal_strength(tau_hat, sensitivity_scale)
        multiple = benchmark_multiple_to_null(
            tau_hat=tau_hat,
            null_effect=sensitivity_config.null_effect,
            q=float(best["benchmark_treatment_strength"]),
            cy2=float(best["benchmark_outcome_strength"]),
            scale=sensitivity_scale,
        )
        summary = pd.DataFrame(
            {
                "robustness_value": [rv],
                "benchmark_variable": [best["covariate"]],
                "benchmark_treatment_strength": [best["benchmark_treatment_strength"]],
                "benchmark_outcome_strength": [best["benchmark_outcome_strength"]],
                "benchmark_bias": [best["benchmark_bias"]],
                "benchmark_multiple_to_null": [multiple],
            }
        )
        contour_path = f"{results_root}/ate/{estimator_id}/sensitivity/contour.png"
        self.plot_sensitivity_contour(
            tau_hat=tau_hat,
            scale=sensitivity_scale,
            q=float(best["benchmark_treatment_strength"]),
            cy2=float(best["benchmark_outcome_strength"]),
            multiple=multiple,
            grid_size=sensitivity_config.benchmark_grid_size,
            multiplier_max=sensitivity_config.benchmark_multiplier_max,
            save_path=contour_path,
        )
        return SensitivityResult(summary=summary, benchmarks=benchmarks, contour_path=contour_path)

    def plot_sensitivity_contour(
        self,
        tau_hat: float,
        scale: float,
        q: float,
        cy2: float,
        multiple: float,
        grid_size: int,
        multiplier_max: float,
        save_path: str,
    ) -> None:
        multipliers = np.linspace(0.0, multiplier_max, grid_size)
        adjusted = []
        for mult_x in multipliers:
            row = []
            for mult_y in multipliers:
                q_adj = min(mult_x * q, 0.999)
                cy2_adj = min(mult_y * cy2, 0.999)
                bias = scale * np.sqrt(cy2_adj * q_adj / (1.0 - q_adj))
                row.append(tau_hat - bias)
            adjusted.append(row)
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.contour(multipliers, multipliers, adjusted, levels=15)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.plot([0, multiplier_max], [0, multiplier_max], linestyle="--", color="gray")
        axis.scatter([1.0], [1.0], color="red", label="observed benchmark")
        axis.scatter([multiple], [multiple], color="green", label="nulling point")
        axis.set_xlabel("Treatment-side multiple")
        axis.set_ylabel("Outcome-side multiple")
        axis.legend()
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def robustness_value_equal_strength(tau_hat: float, scale: float) -> float:
    if scale <= 0.0:
        return float("inf")
    ratio = abs(tau_hat) / scale
    return float(ratio / np.sqrt(1.0 + ratio ** 2))


def benchmark_multiple_to_null(
    tau_hat: float,
    null_effect: float,
    q: float,
    cy2: float,
    scale: float,
) -> float:
    target = abs(tau_hat - null_effect)
    if scale <= 0.0:
        return float("inf")
    multipliers = np.linspace(0.0, 10.0, 500)
    for mult in multipliers:
        q_adj = min(mult * q, 0.999)
        cy2_adj = min(mult * cy2, 0.999)
        bias = scale * np.sqrt(cy2_adj * q_adj / (1.0 - q_adj))
        if bias >= target:
            return float(mult)
    return float("inf")


def partial_r2_outcome_nonlinear(residuals: np.ndarray, z: np.ndarray) -> float:
    model = LinearRegression()
    model.fit(z, residuals)
    fitted = model.predict(z)
    rss_with = float(np.sum((residuals - fitted) ** 2))
    rss_without = float(np.sum(residuals ** 2))
    if rss_without == 0.0:
        return 0.0
    return 1.0 - rss_with / rss_without


def partial_q_propensity(
    omega: np.ndarray,
    z: np.ndarray,
    X_without_z: pd.DataFrame,
) -> float:
    design = pd.concat(
        [pd.DataFrame(z, columns=["z"]), X_without_z.reset_index(drop=True)],
        axis=1,
    )
    model = LinearRegression()
    model.fit(design, omega)
    fitted = model.predict(design)
    rss_with = float(np.sum((omega - fitted) ** 2))
    rss_without = float(np.sum(omega ** 2))
    if rss_without == 0.0:
        return 0.0
    return 1.0 - rss_with / rss_without

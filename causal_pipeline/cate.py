"""CATE estimator adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from causalml.inference.meta import (
    BaseDRClassifier,
    BaseDRRegressor,
    BaseRClassifier,
    BaseRRegressor,
    BaseSClassifier,
    BaseSRegressor,
    BaseTClassifier,
    BaseTRegressor,
    BaseXClassifier,
    BaseXRegressor,
)
from econml.dml import CausalForestDML

from causal_pipeline.config import CATEEstimatorSpec, CATEKind, OutcomeType, build_sklearn_learner
from causal_pipeline.data import CausalDataset, contrast_columns

logger = logging.getLogger(__name__)


class BaseCATEEstimator(ABC):
    """Common interface for conditional average treatment effect estimators."""

    @abstractmethod
    def fit(self, data: CausalDataset) -> BaseCATEEstimator:
        ...

    @abstractmethod
    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        ...

    @abstractmethod
    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        ...


def create_cate_estimator(spec: CATEEstimatorSpec, data: CausalDataset) -> BaseCATEEstimator:
    if spec.kind == CATEKind.S_LEARNER:
        return MetaLearnerAdapter(spec=spec, data=data, letter="s")
    if spec.kind == CATEKind.T_LEARNER:
        return MetaLearnerAdapter(spec=spec, data=data, letter="t")
    if spec.kind == CATEKind.X_LEARNER:
        return MetaLearnerAdapter(spec=spec, data=data, letter="x")
    if spec.kind == CATEKind.R_LEARNER:
        return MetaLearnerAdapter(spec=spec, data=data, letter="r")
    if spec.kind == CATEKind.DR_LEARNER:
        return MetaLearnerAdapter(spec=spec, data=data, letter="dr")
    if spec.kind == CATEKind.CAUSAL_FOREST:
        return CausalForestAdapter(spec=spec, data=data)
    if spec.kind == CATEKind.TARNET:
        return TARNetAdapter(spec=spec, data=data)
    if spec.kind == CATEKind.CFRNET:
        return CFRNetAdapter(spec=spec, data=data)
    if spec.kind == CATEKind.DRAGONNET:
        return DragonNetAdapter(spec=spec, data=data)
    raise ValueError(f"Unsupported CATE kind: {spec.kind}")


class MetaLearnerAdapter(BaseCATEEstimator):
    def __init__(self, spec: CATEEstimatorSpec, data: CausalDataset, letter: str) -> None:
        self.spec = spec
        self.data = data
        self.letter = letter
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> MetaLearnerAdapter:
        if self.spec.base_learner is None:
            raise ValueError(f"{self.spec.kind.value} requires base_learner.")

        X = data.X_effect_modifiers
        treatment = data.treatment_series
        outcome = data.outcome_series
        learner = build_sklearn_learner(self.spec.base_learner)
        binary = self.data.outcome_type == OutcomeType.BINARY
        mapping = {
            ("s", False): BaseSRegressor,
            ("s", True): BaseSClassifier,
            ("t", False): BaseTRegressor,
            ("t", True): BaseTClassifier,
            ("x", False): BaseXRegressor,
            ("x", True): BaseXClassifier,
            ("r", False): BaseRRegressor,
            ("r", True): BaseRClassifier,
            ("dr", False): BaseDRRegressor,
            ("dr", True): BaseDRClassifier,
        }
        model_cls = mapping[(self.letter, binary)]
        self.model = model_cls(learner=learner)
        self.model.fit(X, treatment=treatment, y=outcome)
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        effects = self.model.predict(X)
        if effects.ndim == 1:
            return pd.DataFrame({self.contrast_names[0]: effects})
        columns = self.contrast_names[: effects.shape[1]]
        return pd.DataFrame(effects, columns=columns)

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        if hasattr(self.model, "estimate_ate"):
            ate, lower, upper = self.model.estimate_ate(X)
            return pd.DataFrame(
                {
                    "contrast": [self.contrast_names[0]],
                    "estimate": [float(ate)],
                    "ci_lower": [float(lower)],
                    "ci_upper": [float(upper)],
                }
            )
        predictions = self.predict_effects(data)
        rows = []
        for column in predictions.columns:
            values = predictions[column].values
            rows.append(
                {
                    "contrast": column,
                    "estimate": float(np.mean(values)),
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
        return pd.DataFrame(rows)


class CausalForestAdapter(BaseCATEEstimator):
    def __init__(self, spec: CATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> CausalForestAdapter:
        if self.spec.outcome_learner is None or self.spec.propensity_learner is None:
            raise ValueError("causal_forest requires outcome_learner and propensity_learner.")

        X = data.X_effect_modifiers
        treatment = data.treatment_series
        outcome = data.outcome_series
        W = data.X_confounders
        params = dict(self.spec.params)
        self.model = CausalForestDML(
            model_y=build_sklearn_learner(self.spec.outcome_learner),
            model_t=build_sklearn_learner(self.spec.propensity_learner),
            discrete_treatment=True,
            discrete_outcome=self.data.outcome_type == OutcomeType.BINARY,
            **params,
        )
        self.model.fit(Y=outcome, T=treatment, X=X, W=W)
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        effects = self.model.effect(X)
        if effects.ndim == 1:
            return pd.DataFrame({self.contrast_names[0]: effects})
        columns = self.contrast_names[: effects.shape[1]]
        return pd.DataFrame(effects, columns=columns)

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        inference = self.model.ate_inference(X)
        return pd.DataFrame(
            {
                "contrast": [self.contrast_names[0]],
                "estimate": [float(inference.mean_point)],
                "ci_lower": [float(inference.conf_int_mean()[0][0])],
                "ci_upper": [float(inference.conf_int_mean()[1][0])],
            }
        )


class TARNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: CATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> TARNetAdapter:
        from catenets.models.jax import TARNet

        X = data.X_effect_modifiers
        treatment = data.treatment_series
        outcome = data.outcome_series
        self.model = TARNet(**dict(self.spec.params))
        self.model.fit(X.values, outcome.values, treatment.values)
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        effects = self.model.predict(X.values).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        predictions = self.predict_effects(data)
        return pd.DataFrame(
            {
                "contrast": [self.contrast_names[0]],
                "estimate": [float(predictions.iloc[:, 0].mean())],
                "ci_lower": [np.nan],
                "ci_upper": [np.nan],
            }
        )


class CFRNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: CATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> CFRNetAdapter:
        from catenets.models.jax import SNet1

        X = data.X_effect_modifiers
        treatment = data.treatment_series
        outcome = data.outcome_series
        params = dict(self.spec.params)
        params["penalty_disc"] = self.spec.penalty_disc
        self.model = SNet1(**params)
        self.model.fit(X.values, outcome.values, treatment.values)
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        effects = self.model.predict(X.values).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        predictions = self.predict_effects(data)
        return pd.DataFrame(
            {
                "contrast": [self.contrast_names[0]],
                "estimate": [float(predictions.iloc[:, 0].mean())],
                "ci_lower": [np.nan],
                "ci_upper": [np.nan],
            }
        )


class DragonNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: CATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> DragonNetAdapter:
        from causalml.inference.jax import DragonNet

        X = data.X_effect_modifiers
        treatment = data.treatment_series
        outcome = data.outcome_series
        self.model = DragonNet(**dict(self.spec.params))
        self.model.fit(X.values, outcome.values, treatment=treatment.values)
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        X = data.X_effect_modifiers
        effects = self.model.predict_tau(X.values).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        predictions = self.predict_effects(data)
        return pd.DataFrame(
            {
                "contrast": [self.contrast_names[0]],
                "estimate": [float(predictions.iloc[:, 0].mean())],
                "ci_lower": [np.nan],
                "ci_upper": [np.nan],
            }
        )

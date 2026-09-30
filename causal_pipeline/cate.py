"""CATE estimator adapters."""

from __future__ import annotations

import inspect
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
from sklearn.base import BaseEstimator

from causal_pipeline.config import (
    DEFAULT_DML_N_FOLDS,
    DEFAULT_PROPENSITY_CLIP,
    CATEEstimatorSpec,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    CFRNetCATEEstimatorSpec,
    DragonNetCATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    TARNetCATEEstimatorSpec,
    clone_estimator,
    require_classifier,
)
from causal_pipeline.crossfit import cross_fit_propensity_map, learner_random_state
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.diagnostics import probability_of_arm

_META_LEARNER_CLASSES = {
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


def mean_cate_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in predictions.columns:
        values = predictions[column].to_numpy(dtype=float)
        rows.append(
            {
                "contrast": column,
                "estimate": float(np.mean(values)),
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_meta_learner(
    letter: str,
    binary_outcome: bool,
    outcome_learner: BaseEstimator,
    effect_learner: BaseEstimator | None,
    propensity_learner: BaseEstimator | None,
    control_name: object,
) -> object:
    model_cls = _META_LEARNER_CLASSES[(letter, binary_outcome)]
    if letter in {"s", "t"}:
        return model_cls(learner=clone_estimator(outcome_learner), control_name=control_name)
    if effect_learner is None:
        raise ValueError(f"{letter}-learner requires effect_learner.")
    if letter == "x":
        return model_cls(
            outcome_learner=clone_estimator(outcome_learner),
            effect_learner=clone_estimator(effect_learner),
            control_name=control_name,
        )
    if letter == "r":
        if propensity_learner is None:
            raise ValueError("R-learner requires propensity_learner.")
        return model_cls(
            outcome_learner=clone_estimator(outcome_learner),
            effect_learner=clone_estimator(effect_learner),
            propensity_learner=clone_estimator(propensity_learner),
            control_name=control_name,
        )
    return model_cls(
        control_outcome_learner=clone_estimator(outcome_learner),
        treatment_outcome_learner=clone_estimator(outcome_learner),
        treatment_effect_learner=clone_estimator(effect_learner),
        control_name=control_name,
    )


class MetaLearnerAdapter(BaseCATEEstimator):
    def __init__(self, spec: MetaCATEEstimatorSpec, data: CausalDataset, letter: str) -> None:
        self.spec = spec
        self.data = data
        self.letter = letter
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> MetaLearnerAdapter:
        binary = self.data.outcome_type == OutcomeType.BINARY
        if binary:
            require_classifier(self.spec.outcome_learner, "CATE outcome_learner")
        self.model = build_meta_learner(
            letter=self.letter,
            binary_outcome=binary,
            outcome_learner=self.spec.outcome_learner,
            effect_learner=self.spec.effect_learner,
            propensity_learner=self.spec.propensity_learner,
            control_name=data.control_value,
        )
        features = data.X_adjustment.copy()
        propensity = None
        if self.letter in {"x", "r", "dr"}:
            if self.spec.propensity_learner is None:
                raise ValueError(f"{self.letter}-learner requires propensity_learner.")
            require_classifier(self.spec.propensity_learner, "CATE propensity_learner")
            propensity = cross_fit_propensity_map(
                X=features,
                treatment=data.treatment_series,
                learner=self.spec.propensity_learner,
                treatment_values=data.treatment_values,
                control_value=data.control_value,
                n_folds=DEFAULT_DML_N_FOLDS,
                random_state=learner_random_state(self.spec.propensity_learner),
                clip_bounds=DEFAULT_PROPENSITY_CLIP,
            )
        if propensity is None:
            self.model.fit(
                features,
                treatment=data.treatment_series,
                y=data.outcome_series,
            )
        else:
            self.model.fit(
                features,
                treatment=data.treatment_series,
                y=data.outcome_series,
                p=propensity,
            )
        if self.letter == "x":
            self._attach_propensity_model(features, data.treatment_series)
        return self

    def _attach_propensity_model(self, features: pd.DataFrame, treatment: pd.Series) -> None:
        """X-learner prediction reweights by e(X). Keep that model on the configured learner."""
        if self.spec.propensity_learner is None or self.model is None:
            return
        fitted = clone_estimator(self.spec.propensity_learner)
        fitted.fit(features, treatment)
        groups = list(getattr(self.model, "t_groups", []))
        self.model.propensity_model = {
            group: _ArmProbabilityModel(fitted, group) for group in groups
        }

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict(data.X_adjustment.copy())
        if effects.ndim == 1:
            return pd.DataFrame({self.contrast_names[0]: effects})
        columns = self.contrast_names[: effects.shape[1]]
        return pd.DataFrame(effects, columns=columns)

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        features = data.X_adjustment.copy()
        propensity = None
        if self.letter == "x" and getattr(self.model, "propensity_model", None):
            propensity = {
                group: predictor.predict(features)
                for group, predictor in self.model.propensity_model.items()
            }
        result = self.model.estimate_ate(
            features,
            data.treatment_series,
            data.outcome_series,
            p=propensity,
            pretrain=True,
            **_estimate_ate_interval_kwargs(self.model),
        )
        return _table_from_estimate_ate(result, self.contrast_names)


class CausalForestAdapter(BaseCATEEstimator):
    def __init__(self, spec: CausalForestCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> CausalForestAdapter:
        binary_outcome = self.data.outcome_type == OutcomeType.BINARY
        if binary_outcome:
            require_classifier(self.spec.outcome_learner, "Causal forest outcome_learner")
        require_classifier(self.spec.propensity_learner, "Causal forest propensity_learner")
        self.model = CausalForestDML(
            model_y=clone_estimator(self.spec.outcome_learner),
            model_t=clone_estimator(self.spec.propensity_learner),
            discrete_treatment=True,
            discrete_outcome=binary_outcome,
        )
        features = data.X_effect_modifiers.copy()
        controls = data.X_controls.copy()
        self.model.fit(
            Y=data.outcome_series,
            T=data.treatment_series,
            X=features if features.shape[1] else None,
            W=controls if controls.shape[1] else None,
        )
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        features = data.X_effect_modifiers.copy()
        effects = self.model.effect(features if features.shape[1] else None)
        if effects.ndim == 1:
            return pd.DataFrame({self.contrast_names[0]: effects})
        columns = self.contrast_names[: effects.shape[1]]
        return pd.DataFrame(effects, columns=columns)

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        features = data.X_effect_modifiers.copy()
        inference = self.model.ate_inference(features if features.shape[1] else None)
        lower, upper = inference.conf_int_mean()
        point = np.atleast_1d(np.squeeze(inference.mean_point))
        lower_bound = np.atleast_1d(np.squeeze(lower))
        upper_bound = np.atleast_1d(np.squeeze(upper))
        n_contrasts = min(len(self.contrast_names), len(point), len(lower_bound), len(upper_bound))
        return pd.DataFrame(
            {
                "contrast": self.contrast_names[:n_contrasts],
                "estimate": [float(value) for value in point[:n_contrasts]],
                "ci_lower": [float(value) for value in lower_bound[:n_contrasts]],
                "ci_upper": [float(value) for value in upper_bound[:n_contrasts]],
            }
        )


class TARNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: TARNetCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> TARNetAdapter:
        from catenets.models.jax import TARNet

        self.model = TARNet()
        features = data.X_adjustment.to_numpy()
        self.model.fit(features, data.outcome_series.to_numpy(), data.treatment_series.to_numpy())
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict(data.X_adjustment.to_numpy()).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        return mean_cate_table(self.predict_effects(data))


class CFRNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: CFRNetCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> CFRNetAdapter:
        from catenets.models.jax import CFRNet

        self.model = CFRNet(penalty_disc=self.spec.penalty_disc)
        features = data.X_adjustment.to_numpy()
        self.model.fit(features, data.outcome_series.to_numpy(), data.treatment_series.to_numpy())
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict(data.X_adjustment.to_numpy()).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        return mean_cate_table(self.predict_effects(data))


class DragonNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: DragonNetCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> DragonNetAdapter:
        from causalml.inference.jax import DragonNet

        self.model = DragonNet()
        self.model.fit(
            data.X_adjustment.to_numpy(),
            data.treatment_series.to_numpy(),
            data.outcome_series.to_numpy(),
        )
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict_tau(data.X_adjustment.to_numpy()).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        return mean_cate_table(self.predict_effects(data))


class _ArmProbabilityModel:
    """causalml calls propensity_model.predict and expects P(T = arm | X)."""

    def __init__(self, learner: BaseEstimator, arm: object) -> None:
        self.learner = learner
        self.arm = arm

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = pd.DataFrame(
            self.learner.predict_proba(X),
            columns=list(self.learner.classes_),
        )
        return probability_of_arm(probabilities, self.arm)


def _estimate_ate_interval_kwargs(model: object) -> dict[str, bool]:
    parameters = inspect.signature(model.estimate_ate).parameters
    if "return_ci" in parameters:
        return {"return_ci": True}
    return {}


def _table_from_estimate_ate(result: object, contrast_names: list[str]) -> pd.DataFrame:
    if isinstance(result, tuple):
        point, lower, upper = result[0], result[1], result[2]
    else:
        point = result
        lower = np.nan
        upper = np.nan
    point_values = np.atleast_1d(np.squeeze(point)).astype(float)
    lower_values = np.atleast_1d(np.squeeze(lower)).astype(float)
    upper_values = np.atleast_1d(np.squeeze(upper)).astype(float)
    n_contrasts = min(len(contrast_names), len(point_values))
    return pd.DataFrame(
        {
            "contrast": contrast_names[:n_contrasts],
            "estimate": point_values[:n_contrasts],
            "ci_lower": lower_values[:n_contrasts],
            "ci_upper": upper_values[:n_contrasts],
        }
    )

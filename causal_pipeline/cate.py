"""CATE estimator adapters."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    CATEEstimatorSpec,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    CFRNetCATEEstimatorSpec,
    DragonNetCATEEstimatorSpec,
    JsonValue,
    MetaCATEEstimatorSpec,
    OutcomeType,
    TARNetCATEEstimatorSpec,
    require_classifier,
)
from causal_pipeline.crossfit import cross_fit_propensity_map, dataset_groups, learner_random_state
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.diagnostics import probability_of_arm
from causal_pipeline.scaling import clone_scaled_estimator, fit_transform_covariates, transform_covariates

META_LEARNER_CLASSES = {
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


def reject_grouped_internal_crossfit(data: CausalDataset, estimator_name: str) -> None:
    """causalml and the neural libraries cross-fit by row and cannot keep groups intact."""
    if data.group_id is not None:
        raise NotImplementedError(
            f"{estimator_name} runs an internal row-wise cross-fit that cannot keep groups together."
        )


def binary_library_treatment(data: CausalDataset) -> np.ndarray:
    """Code control as 0 and the single treated arm as 1 for libraries that assume that coding."""
    non_control = [arm for arm in data.treatment_values if arm != data.control_value]
    if len(non_control) != 1:
        raise ValueError("TARNet, CFRNet, and DragonNet support one treated arm.")
    return (data.treatment_series.to_numpy() != data.control_value).astype(int)


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
    control_name: JsonValue,
) -> BaseEstimator:
    model_cls = META_LEARNER_CLASSES[(letter, binary_outcome)]
    if letter in {"s", "t"}:
        return model_cls(learner=clone_scaled_estimator(outcome_learner), control_name=control_name)
    if effect_learner is None:
        raise ValueError(f"{letter}-learner requires effect_learner.")
    if letter == "x":
        return model_cls(
            control_outcome_learner=clone_scaled_estimator(outcome_learner),
            treatment_outcome_learner=clone_scaled_estimator(outcome_learner),
            control_effect_learner=clone_scaled_estimator(effect_learner),
            treatment_effect_learner=clone_scaled_estimator(effect_learner),
            control_name=control_name,
        )
    if letter == "r":
        if propensity_learner is None:
            raise ValueError("R-learner requires propensity_learner.")
        return model_cls(
            outcome_learner=clone_scaled_estimator(outcome_learner),
            effect_learner=clone_scaled_estimator(effect_learner),
            propensity_learner=clone_scaled_estimator(propensity_learner),
            control_name=control_name,
        )
    return model_cls(
        control_outcome_learner=clone_scaled_estimator(outcome_learner),
        treatment_outcome_learner=clone_scaled_estimator(outcome_learner),
        treatment_effect_learner=clone_scaled_estimator(effect_learner),
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
        # The pipeline predicts each group from a model fit on the other groups.
        # Any cross-fit inside causalml only reshuffles that training subset.
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
            propensity_learner = self.spec.propensity_learner
            if propensity_learner is None:
                raise ValueError(f"{self.letter}-learner requires propensity_learner.")
            require_classifier(propensity_learner, "CATE propensity_learner")
            propensity = cross_fit_propensity_map(
                covariates=features,
                treatment=data.treatment_series,
                learner=propensity_learner,
                treatment_values=data.treatment_values,
                control_value=data.control_value,
                n_folds=self.spec.n_folds,
                random_state=learner_random_state(
                    propensity_learner,
                    self.spec.random_state,
                ),
                clip_bounds=self.spec.clip_bounds,
                groups=dataset_groups(data),
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
            self.attach_propensity_model(features, data.treatment_series)
        return self

    def attach_propensity_model(self, features: pd.DataFrame, treatment: pd.Series) -> None:
        """X-learner prediction reweights by e(X). Keep that model on the configured learner."""
        if self.spec.propensity_learner is None or self.model is None:
            return
        fitted = clone_scaled_estimator(self.spec.propensity_learner)
        fitted.fit(features, treatment)
        groups = list(getattr(self.model, "t_groups", []))
        self.model.propensity_model = {
            group: ArmProbabilityModel(fitted, group) for group in groups
        }

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict(data.X_adjustment.copy())
        learned_arms = list(getattr(self.model, "t_groups", []))
        columns = (
            contrast_columns(self.data.control_value, [self.data.control_value, *learned_arms])
            if learned_arms
            else self.contrast_names
        )
        if effects.ndim == 1:
            return pd.DataFrame({columns[0]: effects})
        return pd.DataFrame(effects, columns=columns[: effects.shape[1]])

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
            **estimate_ate_interval_kwargs(self.model),
        )
        return table_from_estimate_ate(
            parse_meta_learner_ate_result(result),
            self.contrast_names,
        )


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
            model_y=clone_scaled_estimator(self.spec.outcome_learner),
            model_t=clone_scaled_estimator(self.spec.propensity_learner),
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
            groups=dataset_groups(data),
        )
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        features = data.X_effect_modifiers.copy()
        feature_frame = features if features.shape[1] else None
        columns = {}
        for name, arm in zip(
            self.contrast_names,
            [value for value in self.data.treatment_values if value != self.data.control_value],
            strict=True,
        ):
            effect = self.model.effect(feature_frame, T0=self.data.control_value, T1=arm)
            columns[name] = np.ravel(effect)
        return pd.DataFrame(columns)

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted.")
        features = data.X_effect_modifiers.copy()
        feature_frame = features if features.shape[1] else None
        rows = []
        for name, arm in zip(
            self.contrast_names,
            [value for value in self.data.treatment_values if value != self.data.control_value],
            strict=True,
        ):
            inference = self.model.ate_inference(
                feature_frame,
                T0=self.data.control_value,
                T1=arm,
            )
            lower, upper = inference.conf_int_mean()
            rows.append(
                {
                    "contrast": name,
                    "estimate": float(np.squeeze(inference.mean_point)),
                    "ci_lower": float(np.squeeze(lower)),
                    "ci_upper": float(np.squeeze(upper)),
                }
            )
        return pd.DataFrame(rows)


class TARNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: TARNetCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.scaler = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> TARNetAdapter:
        reject_grouped_internal_crossfit(data, "TARNet")
        from catenets.models.jax import TARNet

        self.model = TARNet()
        self.scaler, features = fit_transform_covariates(data.X_adjustment)
        self.model.fit(features, data.outcome_series.to_numpy(), binary_library_treatment(data))
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict(transform_covariates(self.scaler, data.X_adjustment)).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        return mean_cate_table(self.predict_effects(data))


class CFRNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: CFRNetCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.scaler = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> CFRNetAdapter:
        reject_grouped_internal_crossfit(data, "CFRNet")
        from catenets.models.jax import CFRNet

        self.model = CFRNet(penalty_disc=self.spec.penalty_disc)
        self.scaler, features = fit_transform_covariates(data.X_adjustment)
        self.model.fit(features, data.outcome_series.to_numpy(), binary_library_treatment(data))
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict(transform_covariates(self.scaler, data.X_adjustment)).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        return mean_cate_table(self.predict_effects(data))


class DragonNetAdapter(BaseCATEEstimator):
    def __init__(self, spec: DragonNetCATEEstimatorSpec, data: CausalDataset) -> None:
        self.spec = spec
        self.data = data
        self.model = None
        self.scaler = None
        self.contrast_names = contrast_columns(
            data.control_value,
            data.treatment_values,
        )

    def fit(self, data: CausalDataset) -> DragonNetAdapter:
        reject_grouped_internal_crossfit(data, "DragonNet")
        from causalml.inference.jax import DragonNet

        self.model = DragonNet()
        self.scaler, features = fit_transform_covariates(data.X_adjustment)
        self.model.fit(
            features,
            binary_library_treatment(data),
            data.outcome_series.to_numpy(),
        )
        return self

    def predict_effects(self, data: CausalDataset) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Estimator is not fitted.")
        effects = self.model.predict_tau(transform_covariates(self.scaler, data.X_adjustment)).flatten()
        return pd.DataFrame({self.contrast_names[0]: effects})

    def estimate(self, data: CausalDataset) -> pd.DataFrame:
        return mean_cate_table(self.predict_effects(data))


class ArmProbabilityModel:
    """causalml calls propensity_model.predict and expects P(T = arm | X)."""

    def __init__(self, learner: BaseEstimator, arm: JsonValue) -> None:
        self.learner = learner
        self.arm = arm

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = pd.DataFrame(
            self.learner.predict_proba(features),
            columns=list(self.learner.classes_),
        )
        return probability_of_arm(probabilities, self.arm)


@dataclass(frozen=True)
class MetaLearnerAteResult:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def parse_meta_learner_ate_result(raw: tuple | float | np.ndarray) -> MetaLearnerAteResult:
    if isinstance(raw, tuple):
        point, lower, upper = raw[0], raw[1], raw[2]
    else:
        point = raw
        lower = np.nan
        upper = np.nan
    return MetaLearnerAteResult(
        point=np.atleast_1d(np.squeeze(point)).astype(float),
        lower=np.atleast_1d(np.squeeze(lower)).astype(float),
        upper=np.atleast_1d(np.squeeze(upper)).astype(float),
    )


def estimate_ate_interval_kwargs(model: BaseEstimator) -> dict[str, bool]:
    estimate_ate = getattr(model, "estimate_ate", None)
    if estimate_ate is None:
        return {}
    parameters = inspect.signature(estimate_ate).parameters
    if "return_ci" in parameters:
        return {"return_ci": True}
    return {}


def table_from_estimate_ate(
    parsed: MetaLearnerAteResult,
    contrast_names: list[str],
) -> pd.DataFrame:
    point_values = parsed.point
    lower_values = parsed.lower
    upper_values = parsed.upper
    n_contrasts = min(len(contrast_names), len(point_values))
    return pd.DataFrame(
        {
            "contrast": contrast_names[:n_contrasts],
            "estimate": point_values[:n_contrasts],
            "ci_lower": lower_values[:n_contrasts],
            "ci_upper": upper_values[:n_contrasts],
        }
    )

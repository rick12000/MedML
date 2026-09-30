"""Central Pydantic configuration and estimator cloning."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Union

import numpy as np
from econml.policy import PolicyTree
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeRegressor

ColumnName = Annotated[str, Field(min_length=1)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
ConfidenceLevel = Annotated[float, Field(gt=0.0, lt=1.0)]
JsonValue = str | int | float | bool | None


class OutcomeType(StrEnum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


class TreatmentMode(StrEnum):
    BINARY = "binary"
    MULTI = "multi"


class ATEKind(StrEnum):
    IPW = "ipw"
    AIPW = "aipw"
    TMLE = "tmle"
    DML_IRM = "dml_irm"
    DML_PLR = "dml_plr"
    DML_APOS = "dml_apos"


class CATEKind(StrEnum):
    S_LEARNER = "s_learner"
    T_LEARNER = "t_learner"
    X_LEARNER = "x_learner"
    R_LEARNER = "r_learner"
    DR_LEARNER = "dr_learner"
    CAUSAL_FOREST = "causal_forest"
    TARNET = "tarnet"
    CFRNET = "cfrnet"
    DRAGONNET = "dragonnet"


class PolicyKind(StrEnum):
    POLICY_TREE = "policy_tree"
    DR_POLICY_TREE = "dr_policy_tree"
    VIRTUAL_TWINS = "virtual_twins"
    MOB = "mob"


DEFAULT_PROPENSITY_CLIP: tuple[float, float] = (0.02, 0.98)
DEFAULT_DML_N_FOLDS: int = 5
DEFAULT_DML_N_REP: int = 1
DEFAULT_ATE_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_IPW_USE_STABILIZED: bool = False
DEFAULT_TMLE_REDUCED: bool = False
DEFAULT_CFRNET_PENALTY_DISC: float = 0.1
DEFAULT_POLICY_BOOTSTRAP_SAMPLES: int = 1000
DEFAULT_POLICY_RANDOM_STATE: int = 42
CATE_DEPENDENT_POLICY_KINDS = frozenset({PolicyKind.POLICY_TREE, PolicyKind.VIRTUAL_TWINS})


class ArbitraryTypesModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Return an unfitted copy so fitting cannot mutate a shared config object."""
    return clone(estimator)


def predict_outcome_mean(estimator: BaseEstimator, X) -> np.ndarray:
    """E[Y | X] as class probability for classifiers, otherwise model.predict."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim == 2:
            if proba.shape[1] != 2:
                raise ValueError("Only binary outcome classifiers are supported.")
            return proba[:, 1]
    return np.ravel(estimator.predict(X))


def require_classifier(estimator: BaseEstimator, role: str) -> None:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"{role} must be a classifier with predict_proba; got {type(estimator).__name__}."
        )


class DataConfig(BaseModel):
    outcome: ColumnName
    treatment: ColumnName
    confounders: list[ColumnName]
    effect_modifiers: list[ColumnName]
    group_id: ColumnName | None = None

    outcome_type: OutcomeType
    treatment_mode: TreatmentMode

    control_value: JsonValue
    treatment_values: list[JsonValue]

    @model_validator(mode="after")
    def validate_categorical_treatment(self) -> DataConfig:
        values = self.treatment_values
        if len(values) < 2:
            raise ValueError("treatment_values must contain at least two categorical arms.")
        if len(set(values)) != len(values):
            raise ValueError("treatment_values must be unique.")
        if self.control_value not in values:
            raise ValueError("control_value must be in treatment_values.")
        if self.treatment_mode == TreatmentMode.BINARY and len(values) != 2:
            raise ValueError("Binary treatment_mode requires exactly two treatment_values.")
        if self.treatment_mode == TreatmentMode.MULTI and len(values) < 3:
            raise ValueError("Multi treatment_mode requires at least three treatment_values.")
        for value in values:
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(
                    "Treatment must be categorical (binary or multi-arm). "
                    "Continuous treatment values are not supported."
                )
        return self


class IPWATEEstimatorSpec(ArbitraryTypesModel):
    kind: Literal[ATEKind.IPW]
    propensity_learner: BaseEstimator
    clip_bounds: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    use_stabilized: bool = DEFAULT_IPW_USE_STABILIZED


class DoublyRobustATEEstimatorSpec(ArbitraryTypesModel):
    kind: Literal[ATEKind.AIPW, ATEKind.TMLE]
    outcome_learner: BaseEstimator
    propensity_learner: BaseEstimator
    clip_bounds: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    reduced: bool = DEFAULT_TMLE_REDUCED


class DoubleMLATEEstimatorSpec(ArbitraryTypesModel):
    kind: Literal[ATEKind.DML_IRM, ATEKind.DML_PLR, ATEKind.DML_APOS]
    outcome_learner: BaseEstimator
    propensity_learner: BaseEstimator
    n_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    n_rep: PositiveInt = DEFAULT_DML_N_REP
    confidence_level: ConfidenceLevel = DEFAULT_ATE_CONFIDENCE_LEVEL


ATEEstimatorSpec = Annotated[
    Union[IPWATEEstimatorSpec, DoublyRobustATEEstimatorSpec, DoubleMLATEEstimatorSpec],
    Field(discriminator="kind"),
]


MetaCATEKind = Literal[
    CATEKind.S_LEARNER,
    CATEKind.T_LEARNER,
    CATEKind.X_LEARNER,
    CATEKind.R_LEARNER,
    CATEKind.DR_LEARNER,
]


class MetaCATEEstimatorSpec(ArbitraryTypesModel):
    kind: MetaCATEKind
    outcome_learner: BaseEstimator
    effect_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None

    @model_validator(mode="after")
    def validate_effect_learner(self) -> MetaCATEEstimatorSpec:
        two_stage = {CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER}
        if self.kind in two_stage and self.effect_learner is None:
            raise ValueError(
                "X-learner, R-learner, and DR-learner require effect_learner, "
                "a regressor for the continuous treatment-effect stage."
            )
        if self.kind in two_stage and self.propensity_learner is None:
            raise ValueError(
                "X-learner, R-learner, and DR-learner require propensity_learner."
            )
        return self


class CausalForestCATEEstimatorSpec(ArbitraryTypesModel):
    kind: Literal[CATEKind.CAUSAL_FOREST]
    outcome_learner: BaseEstimator
    propensity_learner: BaseEstimator


class TARNetCATEEstimatorSpec(BaseModel):
    kind: Literal[CATEKind.TARNET]


class CFRNetCATEEstimatorSpec(BaseModel):
    kind: Literal[CATEKind.CFRNET]
    penalty_disc: float = DEFAULT_CFRNET_PENALTY_DISC


class DragonNetCATEEstimatorSpec(BaseModel):
    kind: Literal[CATEKind.DRAGONNET]


CATEEstimatorSpec = Annotated[
    Union[
        MetaCATEEstimatorSpec,
        CausalForestCATEEstimatorSpec,
        TARNetCATEEstimatorSpec,
        CFRNetCATEEstimatorSpec,
        DragonNetCATEEstimatorSpec,
    ],
    Field(discriminator="kind"),
]


class SplitConfig(BaseModel):
    train_fraction: Fraction
    validation_fraction: Fraction
    test_fraction: Fraction = 0.0
    random_state: int = 42


class DiagnosticConfig(ArbitraryTypesModel):
    propensity_learner: BaseEstimator
    stabilized_weights: bool = False
    propensity_clip: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)


class SensitivityConfig(ArbitraryTypesModel):
    outcome_learner: BaseEstimator
    null_effect: float = 0.0
    benchmark_multiplier_max: float = 3.0
    benchmark_grid_size: PositiveInt = 100
    confidence_level: ConfidenceLevel = 0.95


class CATEEvaluationConfig(ArbitraryTypesModel):
    propensity_learner: BaseEstimator
    outcome_learner: BaseEstimator
    dr_crossfit_folds: PositiveInt = 5
    propensity_clip: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    calibration_bins: PositiveInt = 10
    rate_bootstrap_samples: PositiveInt = 1000
    random_state: int = 42


class PolicyTreeMethodSpec(ArbitraryTypesModel):
    kind: Literal[PolicyKind.POLICY_TREE]
    tree: PolicyTree = Field(default_factory=PolicyTree)


class DRPolicyTreeMethodSpec(ArbitraryTypesModel):
    kind: Literal[PolicyKind.DR_POLICY_TREE]
    outcome_learner: BaseEstimator
    propensity_learner: BaseEstimator


class VirtualTwinsMethodSpec(ArbitraryTypesModel):
    kind: Literal[PolicyKind.VIRTUAL_TWINS]
    tree: DecisionTreeRegressor = Field(default_factory=DecisionTreeRegressor)


class MOBMethodSpec(BaseModel):
    kind: Literal[PolicyKind.MOB]


PolicyMethodSpec = Annotated[
    Union[
        PolicyTreeMethodSpec,
        DRPolicyTreeMethodSpec,
        VirtualTwinsMethodSpec,
        MOBMethodSpec,
    ],
    Field(discriminator="kind"),
]


class PipelineConfig(BaseModel):
    data: DataConfig
    split: SplitConfig
    ate_estimators: list[ATEEstimatorSpec] | None = None
    cate_estimators: list[CATEEstimatorSpec] | None = None
    diagnostics: DiagnosticConfig | None = None
    sensitivity: SensitivityConfig | None = None
    cate_evaluation: CATEEvaluationConfig | None = None
    policy: list[PolicyMethodSpec] | None = None
    results_dir: str = "results"

    @model_validator(mode="after")
    def validate_pipeline_contract(self) -> PipelineConfig:
        ate_configured = bool(self.ate_estimators)
        cate_configured = bool(self.cate_estimators)
        policy_configured = bool(self.policy)

        if not ate_configured and not cate_configured:
            raise ValueError("PipelineConfig requires at least one ATE or CATE estimator.")

        if self.sensitivity is not None and not ate_configured:
            raise ValueError("Sensitivity analysis requires at least one ATE estimator.")

        if self.cate_evaluation is not None and not cate_configured and not policy_configured:
            raise ValueError(
                "CATE evaluation requires CATE estimators, unless policy methods need robust scores."
            )

        if policy_configured:
            if self.cate_evaluation is None:
                raise ValueError(
                    "Policy methods require cate_evaluation so held-out rules can be scored."
                )
            policy_kinds = {method.kind for method in self.policy or []}
            if policy_kinds & CATE_DEPENDENT_POLICY_KINDS and not cate_configured:
                raise ValueError(
                    "policy_tree and virtual_twins require at least one CATE estimator."
                )

        train = self.split.train_fraction
        validation = self.split.validation_fraction
        test = self.split.test_fraction
        total = train + validation + test

        if policy_configured:
            if test <= 0.0:
                raise ValueError(
                    "Policy methods require test_fraction > 0 so fitted rules can be evaluated on a held-out test split."
                )
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    "When policy methods are configured, train + validation + test must equal 1."
                )
        else:
            if abs(train + validation - 1.0) > 1e-9:
                raise ValueError(
                    "Without policy methods, train + validation must equal 1."
                )
            if test != 0.0:
                raise ValueError("Without policy methods, test_fraction must be 0.")
        return self

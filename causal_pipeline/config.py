"""Central Pydantic configuration and estimator cloning."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Union

import numpy as np
import pandas as pd
from econml.policy import PolicyTree
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


class OutcomeFavorability(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class IPWWeighting(StrEnum):
    HAJEK = "hajek"
    HORVITZ_THOMPSON = "horvitz_thompson"


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
    EXACT_POLICY_TREE = "exact_policy_tree"
    DR_POLICY_TREE = "dr_policy_tree"
    VIRTUAL_TWINS = "virtual_twins"
    MOB = "mob"


class PolicySearchMode(StrEnum):
    """Welfare maximizes total reward. CAPITAL maximizes a clinically constrained subgroup."""

    WELFARE = "welfare"
    CAPITAL = "capital"


DEFAULT_PROPENSITY_CLIP: tuple[float, float] = (0.02, 0.98)


def validate_propensity_overlap_handling(
    clip_bounds: tuple[float, float] | None,
    trim_bounds: tuple[float, float] | None,
) -> None:
    if clip_bounds is not None and trim_bounds is not None:
        raise ValueError(
            "At most one of clip_bounds and trim_bounds may be set; use clipping or trimming, not both."
        )
    for name, bounds in (("clip_bounds", clip_bounds), ("trim_bounds", trim_bounds)):
        if bounds is None:
            continue
        lower, upper = bounds
        if lower >= upper:
            raise ValueError(f"{name} must be ordered as (lower, upper) with lower < upper.")
DEFAULT_DML_N_FOLDS: int = 5
DEFAULT_DML_N_REP: int = 1
DEFAULT_ATE_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_TMLE_REDUCED: bool = False
DEFAULT_CFRNET_PENALTY_DISC: float = 0.1
DEFAULT_POLICY_BOOTSTRAP_SAMPLES: int = 1000
DEFAULT_POLICY_RANDOM_STATE: int = 42
DEFAULT_SPLIT_RANDOM_STATE: int = 42
DEFAULT_RESULTS_DIR: str = "results"
DEFAULT_SENSITIVITY_CONFIDENCE_LEVEL: ConfidenceLevel = 0.95
DEFAULT_CATE_EVAL_CROSSFIT_FOLDS: PositiveInt = 5
DEFAULT_CALIBRATION_BINS: PositiveInt = 10
DEFAULT_RATE_BOOTSTRAP_SAMPLES: PositiveInt = 1000
DEFAULT_ECETH_TOLERANCE: float = 0.01
DEFAULT_EXACT_POLICY_DEPTH: int = 2
DEFAULT_EXACT_POLICY_MIN_NODE_SIZE: PositiveInt = 1
DEFAULT_EXACT_POLICY_SPLIT_STEP: PositiveInt = 1
DEFAULT_MINIMUM_EFFECT: float = 0.0
DEFAULT_CLINICAL_THRESHOLD: float = 0.0
DEFAULT_NEGATIVE_EFFECT_PENALTY: float = 0.0
DEFAULT_ESTIMAND_ATE: str = "ate"
DEFAULT_DIAGNOSTIC_STABILIZED_WEIGHTS: bool = False
DEFAULT_TEST_FRACTION: Fraction = 0.0
DEFAULT_NULL_EFFECT: float = 0.0
DEFAULT_BENCHMARK_MULTIPLIER_MAX: float = 3.0
DEFAULT_BENCHMARK_GRID_SIZE: PositiveInt = 100
DEFAULT_LEARNER_RANDOM_STATE: int = 0
DEFAULT_NUISANCE_SCREEN_TEST_FRACTION: Fraction = 0.2
DEFAULT_NUISANCE_SCREEN_VALIDATION_FRACTION: Fraction = 0.25
DEFAULT_TUNE_ESTIMATORS: bool = False
CATE_DEPENDENT_POLICY_KINDS = frozenset(
    {PolicyKind.POLICY_TREE, PolicyKind.EXACT_POLICY_TREE, PolicyKind.VIRTUAL_TWINS}
)


class ArbitraryTypesModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EstimatorAliasMixin(BaseModel):
    """Optional human-chosen id for outputs, plots, and result directories."""

    alias: str | None = None

    @field_validator("alias", mode="before")
    @classmethod
    def normalize_alias(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("alias must be a string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError("alias must be non-empty when provided.")
        return stripped


def clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Return an unfitted copy so fitting cannot mutate a shared config object."""
    return clone(estimator)


def predict_outcome_mean(
    estimator: BaseEstimator,
    features: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """E[Y | X] as class probability for classifiers, otherwise model.predict."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(features))
        if proba.ndim == 2:
            if proba.shape[1] != 2:
                raise ValueError("Only binary outcome classifiers are supported.")
            return proba[:, 1]
    return np.ravel(estimator.predict(features))


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
    outcome_favorability: OutcomeFavorability = OutcomeFavorability.HIGHER_IS_BETTER
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


class IPWATEEstimatorSpec(EstimatorAliasMixin, ArbitraryTypesModel):
    kind: Literal[ATEKind.IPW]
    propensity_learner: BaseEstimator | None = None
    clip_bounds: tuple[float, float] | None = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    trim_bounds: tuple[float, float] | None = None
    weighting: IPWWeighting = IPWWeighting.HAJEK
    n_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    confidence_level: ConfidenceLevel = DEFAULT_ATE_CONFIDENCE_LEVEL
    random_state: int = DEFAULT_LEARNER_RANDOM_STATE
    estimand: str = DEFAULT_ESTIMAND_ATE

    @model_validator(mode="after")
    def validate_propensity_bounds(self) -> IPWATEEstimatorSpec:
        validate_propensity_overlap_handling(self.clip_bounds, self.trim_bounds)
        return self


class DoublyRobustATEEstimatorSpec(EstimatorAliasMixin, ArbitraryTypesModel):
    kind: Literal[ATEKind.AIPW, ATEKind.TMLE]
    outcome_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None
    clip_bounds: tuple[float, float] | None = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    trim_bounds: tuple[float, float] | None = None
    reduced: bool = DEFAULT_TMLE_REDUCED
    n_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    confidence_level: ConfidenceLevel = DEFAULT_ATE_CONFIDENCE_LEVEL
    random_state: int = DEFAULT_LEARNER_RANDOM_STATE
    estimand: str = DEFAULT_ESTIMAND_ATE

    @model_validator(mode="after")
    def validate_propensity_bounds(self) -> DoublyRobustATEEstimatorSpec:
        validate_propensity_overlap_handling(self.clip_bounds, self.trim_bounds)
        return self


class DoubleMLATEEstimatorSpec(EstimatorAliasMixin, ArbitraryTypesModel):
    kind: Literal[ATEKind.DML_IRM, ATEKind.DML_PLR, ATEKind.DML_APOS]
    outcome_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None
    n_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    n_rep: PositiveInt = DEFAULT_DML_N_REP
    confidence_level: ConfidenceLevel = DEFAULT_ATE_CONFIDENCE_LEVEL
    estimand: str = DEFAULT_ESTIMAND_ATE


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


class MetaCATEEstimatorSpec(EstimatorAliasMixin, ArbitraryTypesModel):
    kind: MetaCATEKind
    outcome_learner: BaseEstimator | None = None
    effect_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None
    n_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    clip_bounds: tuple[float, float] | None = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    trim_bounds: tuple[float, float] | None = None
    random_state: int = DEFAULT_LEARNER_RANDOM_STATE

    @model_validator(mode="after")
    def validate_propensity_bounds(self) -> MetaCATEEstimatorSpec:
        validate_propensity_overlap_handling(self.clip_bounds, self.trim_bounds)
        return self


class CausalForestCATEEstimatorSpec(EstimatorAliasMixin, ArbitraryTypesModel):
    kind: Literal[CATEKind.CAUSAL_FOREST]
    outcome_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None


class TARNetCATEEstimatorSpec(EstimatorAliasMixin):
    kind: Literal[CATEKind.TARNET]


class CFRNetCATEEstimatorSpec(EstimatorAliasMixin):
    kind: Literal[CATEKind.CFRNET]
    penalty_disc: float = DEFAULT_CFRNET_PENALTY_DISC


class DragonNetCATEEstimatorSpec(EstimatorAliasMixin):
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
    """Outer split. Policy off uses the full sample. Policy on holds out a test set."""

    test_fraction: Fraction = DEFAULT_TEST_FRACTION
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE


class DiagnosticConfig(ArbitraryTypesModel):
    propensity_learner: BaseEstimator | None = None
    stabilized_weights: bool = DEFAULT_DIAGNOSTIC_STABILIZED_WEIGHTS
    propensity_clip: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)


class SensitivityConfig(ArbitraryTypesModel):
    outcome_learner: BaseEstimator | None = None
    null_effect: float = DEFAULT_NULL_EFFECT
    benchmark_multiplier_max: float = DEFAULT_BENCHMARK_MULTIPLIER_MAX
    benchmark_grid_size: PositiveInt = DEFAULT_BENCHMARK_GRID_SIZE
    confidence_level: ConfidenceLevel = DEFAULT_SENSITIVITY_CONFIDENCE_LEVEL
    selection_column: ColumnName | None = None
    sampling_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None
    inclusion_fraction: float | None = None
    outcome_bounds: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> SensitivityConfig:
        if self.selection_column is not None and (
            self.sampling_learner is None or self.propensity_learner is None
        ):
            raise ValueError(
                "Observed-selection transport requires sampling_learner and propensity_learner."
            )
        if self.inclusion_fraction is not None and not 0.0 < self.inclusion_fraction < 1.0:
            raise ValueError("inclusion_fraction must lie strictly between 0 and 1.")
        if self.outcome_bounds is not None and self.outcome_bounds[0] >= self.outcome_bounds[1]:
            raise ValueError("outcome_bounds must be ordered as (lower, upper).")
        return self


class CATEEvaluationConfig(ArbitraryTypesModel):
    propensity_learner: BaseEstimator | None = None
    outcome_learner: BaseEstimator | None = None
    dr_crossfit_folds: PositiveInt = DEFAULT_CATE_EVAL_CROSSFIT_FOLDS
    propensity_clip: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    calibration_bins: PositiveInt = DEFAULT_CALIBRATION_BINS
    rate_bootstrap_samples: PositiveInt = DEFAULT_RATE_BOOTSTRAP_SAMPLES
    eceth_tolerance: float = Field(default=DEFAULT_ECETH_TOLERANCE, gt=0.0)
    eceth_bootstrap_samples: PositiveInt = DEFAULT_RATE_BOOTSTRAP_SAMPLES
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE


class PolicyTreeMethodSpec(ArbitraryTypesModel):
    kind: Literal[PolicyKind.POLICY_TREE]
    tree: PolicyTree = Field(default_factory=PolicyTree)


class DRPolicyTreeMethodSpec(ArbitraryTypesModel):
    kind: Literal[PolicyKind.DR_POLICY_TREE]
    outcome_learner: BaseEstimator | None = None
    propensity_learner: BaseEstimator | None = None


class ExactPolicyTreeMethodSpec(ArbitraryTypesModel):
    """Globally optimal depth-2 policy tree, in welfare or CAPITAL mode."""

    kind: Literal[PolicyKind.EXACT_POLICY_TREE]
    mode: PolicySearchMode = PolicySearchMode.WELFARE
    depth: Annotated[int, Field(ge=1, le=2)] = DEFAULT_EXACT_POLICY_DEPTH
    min_node_size: PositiveInt = DEFAULT_EXACT_POLICY_MIN_NODE_SIZE
    split_step: PositiveInt = DEFAULT_EXACT_POLICY_SPLIT_STEP
    minimum_effect: float = DEFAULT_MINIMUM_EFFECT
    clinical_threshold: float = DEFAULT_CLINICAL_THRESHOLD
    negative_effect_penalty: float = Field(default=DEFAULT_NEGATIVE_EFFECT_PENALTY, ge=0.0)


class VirtualTwinsMethodSpec(ArbitraryTypesModel):
    kind: Literal[PolicyKind.VIRTUAL_TWINS]
    tree: DecisionTreeRegressor = Field(default_factory=DecisionTreeRegressor)
    minimum_effect: float = Field(default=DEFAULT_MINIMUM_EFFECT, ge=0.0)


class MOBMethodSpec(BaseModel):
    kind: Literal[PolicyKind.MOB]


PolicyMethodSpec = Annotated[
    Union[
        PolicyTreeMethodSpec,
        ExactPolicyTreeMethodSpec,
        DRPolicyTreeMethodSpec,
        VirtualTwinsMethodSpec,
        MOBMethodSpec,
    ],
    Field(discriminator="kind"),
]


class NuisanceScreenConfig(BaseModel):
    """One train/validation/test pass for a logistic model and a random forest."""

    enabled: bool = True
    test_fraction: Fraction = DEFAULT_NUISANCE_SCREEN_TEST_FRACTION
    validation_fraction: Fraction = DEFAULT_NUISANCE_SCREEN_VALIDATION_FRACTION
    random_state: int = DEFAULT_LEARNER_RANDOM_STATE


class SubgroupSplit(BaseModel):
    """One pre-specified cutoff. The high side is every row at or above the cutoff."""

    column: ColumnName
    cutoff: float


class PipelineConfig(BaseModel):
    data: DataConfig
    split: SplitConfig
    ate_estimators: list[ATEEstimatorSpec] | None = None
    subgroup_splits: list[SubgroupSplit] | None = None
    cate_estimators: list[CATEEstimatorSpec] | None = None
    diagnostics: DiagnosticConfig | None = None
    sensitivity: SensitivityConfig | None = None
    cate_evaluation: CATEEvaluationConfig | None = None
    policy: list[PolicyMethodSpec] | None = None
    cate_crossfit_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    policy_bootstrap_samples: PositiveInt = DEFAULT_POLICY_BOOTSTRAP_SAMPLES
    policy_random_state: int = DEFAULT_POLICY_RANDOM_STATE
    results_dir: str = DEFAULT_RESULTS_DIR
    nuisance_screen: NuisanceScreenConfig = Field(default_factory=NuisanceScreenConfig)
    tune_estimators: bool = DEFAULT_TUNE_ESTIMATORS

    @model_validator(mode="after")
    def validate_pipeline_contract(self) -> PipelineConfig:
        ate_configured = bool(self.ate_estimators)
        cate_configured = bool(self.cate_estimators)
        policy_configured = bool(self.policy)

        if not ate_configured and not cate_configured:
            raise ValueError("PipelineConfig requires at least one ATE or CATE estimator.")

        if self.sensitivity is not None and not ate_configured:
            raise ValueError("Sensitivity analysis requires at least one ATE estimator.")

        if self.subgroup_splits and not ate_configured:
            raise ValueError("Subgroup splits require at least one ATE estimator.")
        if self.subgroup_splits:
            split_keys = [(split.column, split.cutoff) for split in self.subgroup_splits]
            if len(split_keys) != len(set(split_keys)):
                raise ValueError("Subgroup splits must be unique column and cutoff pairs.")

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
                    "policy_tree, exact_policy_tree, and virtual_twins require at least one CATE estimator."
                )

        test_fraction = self.split.test_fraction
        if policy_configured:
            if test_fraction <= 0.0 or test_fraction >= 1.0:
                raise ValueError(
                    "Policy methods require a test_fraction strictly between 0 and 1."
                )
        elif test_fraction != 0.0:
            raise ValueError("Without policy methods, test_fraction must be 0.")
        return self

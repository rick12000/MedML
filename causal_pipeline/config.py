"""Central Pydantic configuration and learner registry."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

ColumnName = Annotated[str, Field(min_length=1)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
ConfidenceLevel = Annotated[float, Field(gt=0.0, lt=1.0)]
JsonValue = str | int | float | bool | None
LearnerParams = dict[str, JsonValue]


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


DEFAULT_PROPENSITY_CLIP: tuple[float, float] = (0.02, 0.98)
DEFAULT_DML_N_FOLDS: int = 5
DEFAULT_DML_N_REP: int = 1
DEFAULT_ATE_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_IPW_USE_STABILIZED: bool = False
DEFAULT_TMLE_REDUCED: bool = False
DEFAULT_CFRNET_PENALTY_DISC: float = 0.1


class LearnerSpec(BaseModel):
    name: str
    params: LearnerParams = Field(default_factory=dict)


DEFAULT_LINEAR_REGRESSION_LEARNER = LearnerSpec(name="linear_regression", params={})
DEFAULT_RANDOM_FOREST_REGRESSOR_LEARNER = LearnerSpec(name="random_forest", params={})
DEFAULT_RANDOM_FOREST_CLASSIFIER_LEARNER = LearnerSpec(
    name="random_forest_classifier",
    params={},
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


class _ATEEstimatorCommon(BaseModel):
    params: LearnerParams = Field(default_factory=dict)


class IPWATEEstimatorSpec(_ATEEstimatorCommon):
    kind: Literal[ATEKind.IPW]
    propensity_learner: LearnerSpec
    clip_bounds: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    use_stabilized: bool = DEFAULT_IPW_USE_STABILIZED


class DoublyRobustATEEstimatorSpec(_ATEEstimatorCommon):
    kind: Literal[ATEKind.AIPW, ATEKind.TMLE]
    outcome_learner: LearnerSpec
    propensity_learner: LearnerSpec
    clip_bounds: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    reduced: bool = DEFAULT_TMLE_REDUCED


class DoubleMLATEEstimatorSpec(_ATEEstimatorCommon):
    kind: Literal[ATEKind.DML_IRM, ATEKind.DML_PLR, ATEKind.DML_APOS]
    outcome_learner: LearnerSpec
    propensity_learner: LearnerSpec
    clip_bounds: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    n_folds: PositiveInt = DEFAULT_DML_N_FOLDS
    n_rep: PositiveInt = DEFAULT_DML_N_REP
    confidence_level: ConfidenceLevel = DEFAULT_ATE_CONFIDENCE_LEVEL


ATEEstimatorSpec = Annotated[
    Union[IPWATEEstimatorSpec, DoublyRobustATEEstimatorSpec, DoubleMLATEEstimatorSpec],
    Field(discriminator="kind"),
]


class _CATEEstimatorCommon(BaseModel):
    params: LearnerParams = Field(default_factory=dict)


MetaCATEKind = Literal[
    CATEKind.S_LEARNER,
    CATEKind.T_LEARNER,
    CATEKind.X_LEARNER,
    CATEKind.R_LEARNER,
    CATEKind.DR_LEARNER,
]


class MetaCATEEstimatorSpec(_CATEEstimatorCommon):
    kind: MetaCATEKind
    base_learner: LearnerSpec


class CausalForestCATEEstimatorSpec(_CATEEstimatorCommon):
    kind: Literal[CATEKind.CAUSAL_FOREST]
    outcome_learner: LearnerSpec
    propensity_learner: LearnerSpec


NeuralCATEKind = Literal[CATEKind.TARNET, CATEKind.CFRNET, CATEKind.DRAGONNET]


class NeuralCATEEstimatorSpec(_CATEEstimatorCommon):
    kind: NeuralCATEKind
    penalty_disc: float = DEFAULT_CFRNET_PENALTY_DISC


CATEEstimatorSpec = Annotated[
    Union[MetaCATEEstimatorSpec, CausalForestCATEEstimatorSpec, NeuralCATEEstimatorSpec],
    Field(discriminator="kind"),
]


class SplitConfig(BaseModel):
    train_fraction: Fraction
    validation_fraction: Fraction
    test_fraction: Fraction = 0.0
    random_state: int = 42


class DiagnosticConfig(BaseModel):
    propensity_learner: LearnerSpec
    stabilized_weights: bool = False
    propensity_clip: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    plot_propensity_overlap: bool = True
    calculate_smd: bool = True


class SensitivityConfig(BaseModel):
    null_effect: float = 0.0
    benchmark_multiplier_max: float = 3.0
    benchmark_grid_size: PositiveInt = 100
    confidence_level: ConfidenceLevel = 0.95
    outcome_learner: LearnerSpec = Field(
        default_factory=lambda: DEFAULT_LINEAR_REGRESSION_LEARNER.model_copy(deep=True),
    )


class CATEEvaluationConfig(BaseModel):
    dr_crossfit_folds: PositiveInt = 5
    propensity_clip: tuple[float, float] = Field(default_factory=lambda: DEFAULT_PROPENSITY_CLIP)
    calibration_bins: PositiveInt = 10
    rate_bootstrap_samples: PositiveInt = 1000
    random_state: int = 42
    outcome_learner: LearnerSpec = Field(
        default_factory=lambda: DEFAULT_RANDOM_FOREST_REGRESSOR_LEARNER.model_copy(deep=True),
    )


class PolicyConfig(BaseModel):
    standard_policy_tree: bool = True
    dr_policy_tree: bool = True
    virtual_twins: bool = True
    mob: bool = True
    policy_tree_params: LearnerParams = Field(default_factory=dict)
    dr_policy_tree_params: LearnerParams = Field(default_factory=dict)
    virtual_twins_params: LearnerParams = Field(default_factory=dict)
    mob_params: LearnerParams = Field(default_factory=dict)
    dr_policy_regression_learner: LearnerSpec = Field(
        default_factory=lambda: DEFAULT_RANDOM_FOREST_REGRESSOR_LEARNER.model_copy(deep=True),
    )
    dr_policy_propensity_learner: LearnerSpec = Field(
        default_factory=lambda: DEFAULT_RANDOM_FOREST_CLASSIFIER_LEARNER.model_copy(deep=True),
    )


class PipelineConfig(BaseModel):
    data: DataConfig
    split: SplitConfig
    diagnostics: DiagnosticConfig
    ate_estimators: list[ATEEstimatorSpec]
    cate_estimators: list[CATEEstimatorSpec]
    sensitivity: SensitivityConfig | None = None
    cate_evaluation: CATEEvaluationConfig
    policy: PolicyConfig | None = None
    results_dir: str = "results"

    @model_validator(mode="after")
    def validate_split_fractions(self) -> PipelineConfig:
        train = self.split.train_fraction
        validation = self.split.validation_fraction
        test = self.split.test_fraction
        total = train + validation + test

        if self.policy is not None:
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    "When policy is configured, train + validation + test must equal 1."
                )
            if test <= 0.0:
                raise ValueError("When policy is configured, test_fraction must be > 0.")
        else:
            if abs(train + validation - 1.0) > 1e-9:
                raise ValueError(
                    "Without policy configuration, train + validation must equal 1."
                )
            if test != 0.0:
                raise ValueError("Without policy configuration, test_fraction must be 0.")
        return self


def build_sklearn_learner(spec: LearnerSpec) -> BaseEstimator:
    """Construct a sklearn-compatible estimator from a learner specification."""
    name = spec.name
    params = dict(spec.params)

    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "linear_regression":
        return LinearRegression(**params)
    if name == "random_forest_classifier":
        return RandomForestClassifier(**params)
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "random_forest_regressor":
        return RandomForestRegressor(**params)
    if name == "lightgbm_regressor":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**params)
    if name == "lightgbm_classifier":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**params)
    if name == "lightgbm":
        raise ValueError(
            "Use lightgbm_regressor or lightgbm_classifier instead of lightgbm.",
        )
    if name == "xgboost_regressor":
        import xgboost as xgb

        return xgb.XGBRegressor(**params)
    if name == "xgboost_classifier":
        import xgboost as xgb

        return xgb.XGBClassifier(**params)
    if name == "xgboost":
        raise ValueError(
            "Use xgboost_regressor or xgboost_classifier instead of xgboost.",
        )

    raise ValueError(f"Unknown learner name: {name}")

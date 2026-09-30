"""Project constants and assembled pipeline configuration (Python only)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DataConfig,
    DEFAULT_ATE_CONFIDENCE_LEVEL,
    DEFAULT_DML_N_FOLDS,
    DEFAULT_LEARNER_RANDOM_STATE,
    DEFAULT_POLICY_BOOTSTRAP_SAMPLES,
    DEFAULT_POLICY_RANDOM_STATE,
    DEFAULT_PROPENSITY_CLIP,
    DiagnosticConfig,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    DRPolicyTreeMethodSpec,
    IPWATEEstimatorSpec,
    MOBMethodSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    PolicyKind,
    PolicyTreeMethodSpec,
    SensitivityConfig,
    SplitConfig,
    TreatmentMode,
    VirtualTwinsMethodSpec,
)


RESULTS_DIR = "results"


DATA_OUTCOME_COLUMN = "outcome"
DATA_TREATMENT_COLUMN = "treatment"
DATA_CONFOUNDERS = ("age", "baseline_score", "biomarker", "comorbidity")
DATA_EFFECT_MODIFIERS = ("age", "baseline_score", "biomarker")
DATA_CONTROL_VALUE = 0
DATA_TREATMENT_VALUES = (0, 1)


SPLIT_RANDOM_STATE = 42
SPLIT_TEST_FRACTION_NO_POLICY = 0.0
SPLIT_TEST_FRACTION_WITH_POLICY = 0.20


LOGISTIC_MAX_ITER = 1000
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_RANDOM_STATE = 42


LOGISTIC_LEARNER = LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
LINEAR_REGRESSION_LEARNER = LinearRegression()
RANDOM_FOREST_LEARNER = RandomForestRegressor(
    n_estimators=RANDOM_FOREST_N_ESTIMATORS,
    random_state=RANDOM_FOREST_RANDOM_STATE,
)
RANDOM_FOREST_CLASSIFIER_LEARNER = RandomForestClassifier(
    n_estimators=RANDOM_FOREST_N_ESTIMATORS,
    random_state=RANDOM_FOREST_RANDOM_STATE,
)


DATA_CONFIG = DataConfig(
    outcome=DATA_OUTCOME_COLUMN,
    treatment=DATA_TREATMENT_COLUMN,
    confounders=list(DATA_CONFOUNDERS),
    effect_modifiers=list(DATA_EFFECT_MODIFIERS),
    outcome_type=OutcomeType.BINARY,
    treatment_mode=TreatmentMode.BINARY,
    control_value=DATA_CONTROL_VALUE,
    treatment_values=list(DATA_TREATMENT_VALUES),
)


INCLUDE_POLICY_DEFAULT = False

ATE_CROSSFIT_FOLDS = DEFAULT_DML_N_FOLDS
ATE_CONFIDENCE_LEVEL = DEFAULT_ATE_CONFIDENCE_LEVEL
ATE_CROSSFIT_RANDOM_STATE = DEFAULT_LEARNER_RANDOM_STATE
CATE_CROSSFIT_FOLDS = DEFAULT_DML_N_FOLDS
CATE_PROPENSITY_CLIP = DEFAULT_PROPENSITY_CLIP
POLICY_BOOTSTRAP_SAMPLES = DEFAULT_POLICY_BOOTSTRAP_SAMPLES
POLICY_RANDOM_STATE = DEFAULT_POLICY_RANDOM_STATE


def build_pipeline_config(include_policy: bool = INCLUDE_POLICY_DEFAULT) -> PipelineConfig:
    if include_policy:
        split = SplitConfig(
            test_fraction=SPLIT_TEST_FRACTION_WITH_POLICY,
            random_state=SPLIT_RANDOM_STATE,
        )
        policy = [
            PolicyTreeMethodSpec(kind=PolicyKind.POLICY_TREE),
            DRPolicyTreeMethodSpec(
                kind=PolicyKind.DR_POLICY_TREE,
                outcome_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
            ),
            VirtualTwinsMethodSpec(kind=PolicyKind.VIRTUAL_TWINS),
            MOBMethodSpec(kind=PolicyKind.MOB),
        ]
    else:
        split = SplitConfig(
            test_fraction=SPLIT_TEST_FRACTION_NO_POLICY,
            random_state=SPLIT_RANDOM_STATE,
        )
        policy = None

    return PipelineConfig(
        data=DATA_CONFIG,
        split=split,
        diagnostics=DiagnosticConfig(propensity_learner=LOGISTIC_LEARNER),
        ate_estimators=[
            IPWATEEstimatorSpec(
                kind=ATEKind.IPW,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=ATE_CROSSFIT_FOLDS,
                confidence_level=ATE_CONFIDENCE_LEVEL,
                random_state=ATE_CROSSFIT_RANDOM_STATE,
            ),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.AIPW,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=ATE_CROSSFIT_FOLDS,
                confidence_level=ATE_CONFIDENCE_LEVEL,
                random_state=ATE_CROSSFIT_RANDOM_STATE,
            ),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.TMLE,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=ATE_CROSSFIT_FOLDS,
                confidence_level=ATE_CONFIDENCE_LEVEL,
                random_state=ATE_CROSSFIT_RANDOM_STATE,
            ),
            DoubleMLATEEstimatorSpec(
                kind=ATEKind.DML_IRM,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                n_folds=ATE_CROSSFIT_FOLDS,
                confidence_level=ATE_CONFIDENCE_LEVEL,
            ),
        ],
        cate_estimators=[
            MetaCATEEstimatorSpec(
                kind=CATEKind.S_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.T_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.X_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                effect_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=CATE_CROSSFIT_FOLDS,
                clip_bounds=CATE_PROPENSITY_CLIP,
                random_state=ATE_CROSSFIT_RANDOM_STATE,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.R_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                effect_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=CATE_CROSSFIT_FOLDS,
                clip_bounds=CATE_PROPENSITY_CLIP,
                random_state=ATE_CROSSFIT_RANDOM_STATE,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.DR_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                effect_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=CATE_CROSSFIT_FOLDS,
                clip_bounds=CATE_PROPENSITY_CLIP,
                random_state=ATE_CROSSFIT_RANDOM_STATE,
            ),
            CausalForestCATEEstimatorSpec(
                kind=CATEKind.CAUSAL_FOREST,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
            ),
        ],
        sensitivity=SensitivityConfig(outcome_learner=LINEAR_REGRESSION_LEARNER),
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=LOGISTIC_LEARNER,
            outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
        ),
        policy=policy,
        cate_crossfit_folds=CATE_CROSSFIT_FOLDS,
        policy_bootstrap_samples=POLICY_BOOTSTRAP_SAMPLES,
        policy_random_state=POLICY_RANDOM_STATE,
        results_dir=RESULTS_DIR,
    )

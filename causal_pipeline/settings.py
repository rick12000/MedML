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
            IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=LOGISTIC_LEARNER),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.AIPW,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
            ),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.TMLE,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
            ),
            DoubleMLATEEstimatorSpec(
                kind=ATEKind.DML_IRM,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                propensity_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
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
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.R_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                effect_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.DR_LEARNER,
                outcome_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
                effect_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
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
        results_dir=RESULTS_DIR,
    )

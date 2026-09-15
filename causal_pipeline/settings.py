"""Project constants and assembled pipeline configuration (Python only)."""

from __future__ import annotations

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DataConfig,
    DiagnosticConfig,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    IPWATEEstimatorSpec,
    LearnerSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    PolicyConfig,
    SensitivityConfig,
    SplitConfig,
    TreatmentMode,
)


RESULTS_DIR = "results"


DATA_OUTCOME_COLUMN = "outcome"
DATA_TREATMENT_COLUMN = "treatment"
DATA_CONFOUNDERS = ("age", "baseline_score", "biomarker", "comorbidity")
DATA_EFFECT_MODIFIERS = ("age", "baseline_score", "biomarker")
DATA_CONTROL_VALUE = 0
DATA_TREATMENT_VALUES = (0, 1)


SPLIT_RANDOM_STATE = 42
SPLIT_TRAIN_FRACTION_NO_POLICY = 0.70
SPLIT_VALIDATION_FRACTION_NO_POLICY = 0.30
SPLIT_TEST_FRACTION_NO_POLICY = 0.0

SPLIT_TRAIN_FRACTION_WITH_POLICY = 0.60
SPLIT_VALIDATION_FRACTION_WITH_POLICY = 0.20
SPLIT_TEST_FRACTION_WITH_POLICY = 0.20


LOGISTIC_MAX_ITER = 1000
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_RANDOM_STATE = 42


LOGISTIC_LEARNER = LearnerSpec(
    name="logistic_regression",
    params={"max_iter": LOGISTIC_MAX_ITER},
)
RANDOM_FOREST_LEARNER = LearnerSpec(
    name="random_forest",
    params={
        "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
        "random_state": RANDOM_FOREST_RANDOM_STATE,
    },
)
RANDOM_FOREST_CLASSIFIER_LEARNER = LearnerSpec(
    name="random_forest_classifier",
    params={
        "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
        "random_state": RANDOM_FOREST_RANDOM_STATE,
    },
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


def build_pipeline_config(policy_enabled: bool = False) -> PipelineConfig:
    if policy_enabled:
        split = SplitConfig(
            train_fraction=SPLIT_TRAIN_FRACTION_WITH_POLICY,
            validation_fraction=SPLIT_VALIDATION_FRACTION_WITH_POLICY,
            test_fraction=SPLIT_TEST_FRACTION_WITH_POLICY,
            random_state=SPLIT_RANDOM_STATE,
        )
        policy = PolicyConfig()
    else:
        split = SplitConfig(
            train_fraction=SPLIT_TRAIN_FRACTION_NO_POLICY,
            validation_fraction=SPLIT_VALIDATION_FRACTION_NO_POLICY,
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
                outcome_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
            ),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.TMLE,
                outcome_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=LOGISTIC_LEARNER,
            ),
            DoubleMLATEEstimatorSpec(
                kind=ATEKind.DML_IRM,
                outcome_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
            ),
        ],
        cate_estimators=[
            MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER, base_learner=RANDOM_FOREST_LEARNER),
            MetaCATEEstimatorSpec(kind=CATEKind.T_LEARNER, base_learner=RANDOM_FOREST_LEARNER),
            MetaCATEEstimatorSpec(kind=CATEKind.X_LEARNER, base_learner=RANDOM_FOREST_LEARNER),
            MetaCATEEstimatorSpec(kind=CATEKind.R_LEARNER, base_learner=RANDOM_FOREST_LEARNER),
            MetaCATEEstimatorSpec(kind=CATEKind.DR_LEARNER, base_learner=RANDOM_FOREST_LEARNER),
            CausalForestCATEEstimatorSpec(
                kind=CATEKind.CAUSAL_FOREST,
                outcome_learner=RANDOM_FOREST_LEARNER,
                propensity_learner=RANDOM_FOREST_CLASSIFIER_LEARNER,
            ),
        ],
        sensitivity=SensitivityConfig(),
        cate_evaluation=CATEEvaluationConfig(),
        policy=policy,
        results_dir=RESULTS_DIR,
    )

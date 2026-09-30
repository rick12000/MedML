"""Example run configuration for the binary observational cohort.

This file is the analysis setup, not part of the causal_pipeline package.
Column names match the cohort written by scripts/generate_toy_cohort.py.
"""

from __future__ import annotations

from econml.policy import PolicyTree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

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
    ExactPolicyTreeMethodSpec,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    PolicyKind,
    PolicySearchMode,
    PolicyTreeMethodSpec,
    SensitivityConfig,
    SplitConfig,
    TreatmentMode,
    VirtualTwinsMethodSpec,
)


RESULTS_DIR = "results"

OUTCOME_COLUMN = "outcome"
TREATMENT_COLUMN = "treatment"
SELECTION_COLUMN = "included"
GROUP_COLUMN = "site"

# Twelve covariates. Confounders enter treatment and outcome. Effect modifiers
# enter the conditional treatment effect. Site is also the split cluster.
# lab_noise is pure noise so omitted-variable benchmarks have a weak covariate.
CONFOUNDERS = (
    "age",
    "baseline_score",
    "biomarker",
    "comorbidity",
    "sex",
    "bmi",
    "egfr",
    "smoker",
    "site",
    "lab_noise",
)
EFFECT_MODIFIERS = (
    "age",
    "baseline_score",
    "biomarker",
    "frailty",
    "symptom_years",
)
CONTROL_VALUE = 0
TREATMENT_VALUES = (0, 1)

# Binary outcome, so the unseen-population bound uses the unit interval.
OUTCOME_LOWER = 0.0
OUTCOME_UPPER = 1.0
# Share of the target population that meets the age inclusion rule.
POPULATION_INCLUSION_FRACTION = 0.50

SPLIT_RANDOM_STATE = 42
TEST_FRACTION = 0.20

CROSSFIT_FOLDS = 3
CONFIDENCE_LEVEL = 0.95
LEARNER_RANDOM_STATE = 0
PROPENSITY_CLIP = (0.02, 0.98)

LOGISTIC_MAX_ITER = 1000
FOREST_TREES = 40
FOREST_RANDOM_STATE = 42

POLICY_BOOTSTRAP_SAMPLES = 80
POLICY_RANDOM_STATE = 42
POLICY_TREE_DEPTH = 3
POLICY_MIN_LEAF = 40
EXACT_POLICY_DEPTH = 2
EXACT_POLICY_MIN_NODE_SIZE = 40
EXACT_POLICY_SPLIT_STEP = 40
WELFARE_MINIMUM_EFFECT = 0.03
CAPITAL_CLINICAL_THRESHOLD = 0.05
CAPITAL_NEGATIVE_EFFECT_PENALTY = 1.0
VIRTUAL_TWINS_MINIMUM_EFFECT = 0.03
VIRTUAL_TWINS_TREE_DEPTH = 3

EVALUATION_FOLDS = 3
CALIBRATION_BINS = 10
EVALUATION_BOOTSTRAP_SAMPLES = 80
ECETH_TOLERANCE = 0.01


LOGISTIC_LEARNER = LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
LINEAR_LEARNER = LinearRegression()
FOREST_REGRESSOR = RandomForestRegressor(
    n_estimators=FOREST_TREES,
    random_state=FOREST_RANDOM_STATE,
)
FOREST_CLASSIFIER = RandomForestClassifier(
    n_estimators=FOREST_TREES,
    random_state=FOREST_RANDOM_STATE,
)
GREEDY_POLICY_TREE = PolicyTree(
    max_depth=POLICY_TREE_DEPTH,
    min_samples_leaf=POLICY_MIN_LEAF,
    random_state=POLICY_RANDOM_STATE,
)
VIRTUAL_TWINS_TREE = DecisionTreeRegressor(
    max_depth=VIRTUAL_TWINS_TREE_DEPTH,
    min_samples_leaf=POLICY_MIN_LEAF,
    random_state=POLICY_RANDOM_STATE,
)


DATA_CONFIG = DataConfig(
    outcome=OUTCOME_COLUMN,
    treatment=TREATMENT_COLUMN,
    confounders=list(CONFOUNDERS),
    effect_modifiers=list(EFFECT_MODIFIERS),
    group_id=GROUP_COLUMN,
    outcome_type=OutcomeType.BINARY,
    treatment_mode=TreatmentMode.BINARY,
    control_value=CONTROL_VALUE,
    treatment_values=list(TREATMENT_VALUES),
)


def build_pipeline_config() -> PipelineConfig:
    """Binary cohort with diagnostics, ATE, CATE, selection, calibration, and policy."""
    return PipelineConfig(
        data=DATA_CONFIG,
        split=SplitConfig(
            test_fraction=TEST_FRACTION,
            random_state=SPLIT_RANDOM_STATE,
        ),
        diagnostics=DiagnosticConfig(
            propensity_learner=LOGISTIC_LEARNER,
            propensity_clip=PROPENSITY_CLIP,
        ),
        ate_estimators=[
            IPWATEEstimatorSpec(
                kind=ATEKind.IPW,
                propensity_learner=LOGISTIC_LEARNER,
                clip_bounds=PROPENSITY_CLIP,
                n_folds=CROSSFIT_FOLDS,
                confidence_level=CONFIDENCE_LEVEL,
                random_state=LEARNER_RANDOM_STATE,
            ),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.AIPW,
                outcome_learner=FOREST_CLASSIFIER,
                propensity_learner=LOGISTIC_LEARNER,
                clip_bounds=PROPENSITY_CLIP,
                n_folds=CROSSFIT_FOLDS,
                confidence_level=CONFIDENCE_LEVEL,
                random_state=LEARNER_RANDOM_STATE,
            ),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.TMLE,
                outcome_learner=FOREST_CLASSIFIER,
                propensity_learner=LOGISTIC_LEARNER,
                clip_bounds=PROPENSITY_CLIP,
                n_folds=CROSSFIT_FOLDS,
                confidence_level=CONFIDENCE_LEVEL,
                random_state=LEARNER_RANDOM_STATE,
            ),
            DoubleMLATEEstimatorSpec(
                kind=ATEKind.DML_IRM,
                outcome_learner=FOREST_CLASSIFIER,
                propensity_learner=FOREST_CLASSIFIER,
                n_folds=CROSSFIT_FOLDS,
                confidence_level=CONFIDENCE_LEVEL,
            ),
            DoubleMLATEEstimatorSpec(
                kind=ATEKind.DML_PLR,
                outcome_learner=FOREST_REGRESSOR,
                propensity_learner=FOREST_CLASSIFIER,
                n_folds=CROSSFIT_FOLDS,
                confidence_level=CONFIDENCE_LEVEL,
            ),
            DoubleMLATEEstimatorSpec(
                kind=ATEKind.DML_APOS,
                outcome_learner=FOREST_CLASSIFIER,
                propensity_learner=FOREST_CLASSIFIER,
                n_folds=CROSSFIT_FOLDS,
                confidence_level=CONFIDENCE_LEVEL,
            ),
        ],
        cate_estimators=[
            MetaCATEEstimatorSpec(
                kind=CATEKind.S_LEARNER,
                outcome_learner=FOREST_CLASSIFIER,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.T_LEARNER,
                outcome_learner=FOREST_CLASSIFIER,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.X_LEARNER,
                outcome_learner=FOREST_CLASSIFIER,
                effect_learner=FOREST_REGRESSOR,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=CROSSFIT_FOLDS,
                clip_bounds=PROPENSITY_CLIP,
                random_state=LEARNER_RANDOM_STATE,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.R_LEARNER,
                outcome_learner=FOREST_CLASSIFIER,
                effect_learner=FOREST_REGRESSOR,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=CROSSFIT_FOLDS,
                clip_bounds=PROPENSITY_CLIP,
                random_state=LEARNER_RANDOM_STATE,
            ),
            MetaCATEEstimatorSpec(
                kind=CATEKind.DR_LEARNER,
                outcome_learner=FOREST_CLASSIFIER,
                effect_learner=FOREST_REGRESSOR,
                propensity_learner=LOGISTIC_LEARNER,
                n_folds=CROSSFIT_FOLDS,
                clip_bounds=PROPENSITY_CLIP,
                random_state=LEARNER_RANDOM_STATE,
            ),
            CausalForestCATEEstimatorSpec(
                kind=CATEKind.CAUSAL_FOREST,
                outcome_learner=FOREST_CLASSIFIER,
                propensity_learner=FOREST_CLASSIFIER,
            ),
        ],
        sensitivity=SensitivityConfig(
            outcome_learner=LINEAR_LEARNER,
            sampling_learner=LOGISTIC_LEARNER,
            propensity_learner=LOGISTIC_LEARNER,
            selection_column=SELECTION_COLUMN,
            inclusion_fraction=POPULATION_INCLUSION_FRACTION,
            outcome_bounds=(OUTCOME_LOWER, OUTCOME_UPPER),
            confidence_level=CONFIDENCE_LEVEL,
        ),
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=LOGISTIC_LEARNER,
            outcome_learner=FOREST_CLASSIFIER,
            dr_crossfit_folds=EVALUATION_FOLDS,
            propensity_clip=PROPENSITY_CLIP,
            calibration_bins=CALIBRATION_BINS,
            rate_bootstrap_samples=EVALUATION_BOOTSTRAP_SAMPLES,
            eceth_tolerance=ECETH_TOLERANCE,
            eceth_bootstrap_samples=EVALUATION_BOOTSTRAP_SAMPLES,
            random_state=SPLIT_RANDOM_STATE,
        ),
        policy=[
            PolicyTreeMethodSpec(kind=PolicyKind.POLICY_TREE, tree=GREEDY_POLICY_TREE),
            ExactPolicyTreeMethodSpec(
                kind=PolicyKind.EXACT_POLICY_TREE,
                mode=PolicySearchMode.WELFARE,
                depth=EXACT_POLICY_DEPTH,
                min_node_size=EXACT_POLICY_MIN_NODE_SIZE,
                split_step=EXACT_POLICY_SPLIT_STEP,
                minimum_effect=WELFARE_MINIMUM_EFFECT,
            ),
            ExactPolicyTreeMethodSpec(
                kind=PolicyKind.EXACT_POLICY_TREE,
                mode=PolicySearchMode.CAPITAL,
                depth=EXACT_POLICY_DEPTH,
                min_node_size=EXACT_POLICY_MIN_NODE_SIZE,
                split_step=EXACT_POLICY_SPLIT_STEP,
                clinical_threshold=CAPITAL_CLINICAL_THRESHOLD,
                negative_effect_penalty=CAPITAL_NEGATIVE_EFFECT_PENALTY,
            ),
            DRPolicyTreeMethodSpec(
                kind=PolicyKind.DR_POLICY_TREE,
                outcome_learner=FOREST_REGRESSOR,
                propensity_learner=FOREST_CLASSIFIER,
            ),
            VirtualTwinsMethodSpec(
                kind=PolicyKind.VIRTUAL_TWINS,
                tree=VIRTUAL_TWINS_TREE,
                minimum_effect=VIRTUAL_TWINS_MINIMUM_EFFECT,
            ),
        ],
        cate_crossfit_folds=CROSSFIT_FOLDS,
        policy_bootstrap_samples=POLICY_BOOTSTRAP_SAMPLES,
        policy_random_state=POLICY_RANDOM_STATE,
        results_dir=RESULTS_DIR,
    )

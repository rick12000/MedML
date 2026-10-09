from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SPLIT_RANDOM_STATE,
    DiagnosticConfig,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    PolicyKind,
    PolicyTreeMethodSpec,
    SplitConfig,
    TreatmentMode,
)

TEST_LOGISTIC_MAX_ITER = 500
TEST_RANDOM_STATE = 0
TEST_FOREST_N_ESTIMATORS = 10
TEST_SYNTHETIC_N_OBSERVATIONS = 200
TEST_POLICY_CROSSFIT_FOLDS = 3
TEST_POLICY_BOOTSTRAP_SAMPLES = 20
RECOVERY_SAMPLE_SIZE = 4000
RECOVERY_RANDOM_STATE = 0
CONSTANT_TREATMENT_EFFECT = 1.5
CONSTANT_EFFECT_NOISE = 0.5
HETEROGENEOUS_EFFECT_INTERCEPT = 1.0
HETEROGENEOUS_EFFECT_SLOPE = 1.2
HETEROGENEOUS_EFFECT_NOISE = 0.5
RECOVERY_FOREST_TREES = 40
RECOVERY_FOREST_DEPTH = 8


def draw_confounded_treatment(
    n_observations: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.random.Generator]:
    """Covariates and a logistic propensity, the overlap design used by Nie and Wager."""
    generator = np.random.default_rng(random_state)
    covariate_one = generator.normal(size=n_observations)
    covariate_two = generator.normal(size=n_observations)
    propensity_logit = 0.5 * covariate_one - 0.4 * covariate_two
    propensity = 1.0 / (1.0 + np.exp(-propensity_logit))
    treatment = generator.binomial(1, propensity)
    return covariate_one, covariate_two, treatment, generator


def cohort_frame(
    covariate_one: np.ndarray,
    covariate_two: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": outcome,
            "t": treatment,
            "x1": covariate_one,
            "x2": covariate_two,
        }
    )


@pytest.fixture
def logistic_learner() -> BaseEstimator:
    return LogisticRegression(max_iter=TEST_LOGISTIC_MAX_ITER)


@pytest.fixture
def linear_learner() -> BaseEstimator:
    return LinearRegression()


@pytest.fixture
def recovery_regressor() -> BaseEstimator:
    return RandomForestRegressor(
        n_estimators=RECOVERY_FOREST_TREES,
        max_depth=RECOVERY_FOREST_DEPTH,
        random_state=TEST_RANDOM_STATE,
    )


@pytest.fixture
def recovery_classifier() -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=RECOVERY_FOREST_TREES,
        max_depth=RECOVERY_FOREST_DEPTH,
        random_state=TEST_RANDOM_STATE,
    )


@pytest.fixture
def forest_learner() -> BaseEstimator:
    return RandomForestRegressor(
        n_estimators=TEST_FOREST_N_ESTIMATORS,
        random_state=TEST_RANDOM_STATE,
    )


@pytest.fixture
def forest_classifier() -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=TEST_FOREST_N_ESTIMATORS,
        random_state=TEST_RANDOM_STATE,
    )


@pytest.fixture
def binary_data_config() -> DataConfig:
    return DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1", "x2"],
        effect_modifiers=["x1"],
        outcome_type=OutcomeType.BINARY,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )


@pytest.fixture
def continuous_data_config() -> DataConfig:
    return DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1"],
        effect_modifiers=["x1"],
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )


@pytest.fixture
def recovery_data_config() -> DataConfig:
    return DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1", "x2"],
        effect_modifiers=["x1"],
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )


@pytest.fixture
def constant_effect_cohort() -> tuple[pd.DataFrame, float]:
    """Partially linear outcome with a constant treatment coefficient."""
    covariate_one, covariate_two, treatment, generator = draw_confounded_treatment(
        n_observations=RECOVERY_SAMPLE_SIZE,
        random_state=RECOVERY_RANDOM_STATE,
    )
    outcome = (
        CONSTANT_TREATMENT_EFFECT * treatment
        + 0.8 * covariate_one
        + 0.4 * covariate_two
        + generator.normal(scale=CONSTANT_EFFECT_NOISE, size=RECOVERY_SAMPLE_SIZE)
    )
    return (
        cohort_frame(covariate_one, covariate_two, treatment, outcome),
        CONSTANT_TREATMENT_EFFECT,
    )


@pytest.fixture
def heterogeneous_effect_cohort() -> tuple[pd.DataFrame, np.ndarray]:
    """Linear conditional effect tau(x) = a + b x1, with the same confounding design."""
    covariate_one, covariate_two, treatment, generator = draw_confounded_treatment(
        n_observations=RECOVERY_SAMPLE_SIZE,
        random_state=RECOVERY_RANDOM_STATE + 1,
    )
    treatment_effect = HETEROGENEOUS_EFFECT_INTERCEPT + HETEROGENEOUS_EFFECT_SLOPE * covariate_one
    outcome = (
        treatment_effect * treatment
        + 0.6 * covariate_one
        + 0.3 * covariate_two
        + generator.normal(scale=HETEROGENEOUS_EFFECT_NOISE, size=RECOVERY_SAMPLE_SIZE)
    )
    return (
        cohort_frame(covariate_one, covariate_two, treatment, outcome),
        treatment_effect,
    )


@pytest.fixture
def pipeline_config_no_policy(
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    forest_classifier: BaseEstimator,
) -> PipelineConfig:
    return PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(
            test_fraction=0.0,
            random_state=DEFAULT_SPLIT_RANDOM_STATE,
        ),
        diagnostics=DiagnosticConfig(propensity_learner=logistic_learner),
        ate_estimators=[
            IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        ],
        cate_estimators=[
            MetaCATEEstimatorSpec(
                kind=CATEKind.S_LEARNER,
                outcome_learner=forest_classifier,
            ),
        ],
        sensitivity=None,
        cate_evaluation=None,
        policy=None,
        results_dir=DEFAULT_RESULTS_DIR,
    )


@pytest.fixture
def pipeline_config_with_policy(
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    forest_classifier: BaseEstimator,
) -> PipelineConfig:
    return PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(test_fraction=0.2, random_state=DEFAULT_SPLIT_RANDOM_STATE),
        diagnostics=DiagnosticConfig(propensity_learner=logistic_learner),
        ate_estimators=[
            IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        ],
        cate_estimators=[
            MetaCATEEstimatorSpec(
                kind=CATEKind.S_LEARNER,
                outcome_learner=forest_classifier,
            ),
        ],
        sensitivity=None,
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=logistic_learner,
            outcome_learner=forest_classifier,
            dr_crossfit_folds=TEST_POLICY_CROSSFIT_FOLDS,
            rate_bootstrap_samples=TEST_POLICY_BOOTSTRAP_SAMPLES,
            random_state=TEST_RANDOM_STATE,
        ),
        policy=[PolicyTreeMethodSpec(kind=PolicyKind.POLICY_TREE)],
        results_dir=DEFAULT_RESULTS_DIR,
    )


@pytest.fixture
def df_synthetic_binary() -> pd.DataFrame:
    rng = np.random.default_rng(TEST_RANDOM_STATE)
    treatment = rng.integers(0, 2, size=TEST_SYNTHETIC_N_OBSERVATIONS)
    covariate_one = rng.normal(size=TEST_SYNTHETIC_N_OBSERVATIONS)
    covariate_two = rng.normal(size=TEST_SYNTHETIC_N_OBSERVATIONS)
    logit = -0.3 + 0.8 * treatment + 0.4 * covariate_one
    outcome = (logit + rng.normal(size=TEST_SYNTHETIC_N_OBSERVATIONS) > 0).astype(int)
    return pd.DataFrame(
        {
            "y": outcome,
            "t": treatment,
            "x1": covariate_one,
            "x2": covariate_two,
        }
    )


@pytest.fixture
def df_synthetic_grouped(df_synthetic_binary: pd.DataFrame) -> pd.DataFrame:
    df_grouped = df_synthetic_binary.copy()
    df_grouped["patient_id"] = np.arange(len(df_grouped)) // 5
    return df_grouped



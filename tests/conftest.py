from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SPLIT_RANDOM_STATE,
    DiagnosticConfig,
    DoublyRobustATEEstimatorSpec,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    PolicyKind,
    PolicyTreeMethodSpec,
    SplitConfig,
    TreatmentMode,
)
from settings import DATA_CONFIG
from causal_pipeline.utils import ensure_directory

TEST_LOGISTIC_MAX_ITER = 500
TEST_RANDOM_STATE = 0
TEST_FOREST_N_ESTIMATORS = 10
TEST_SYNTHETIC_N_OBSERVATIONS = 200
TEST_INTEGRATION_FOREST_N_ESTIMATORS = 8
TEST_POLICY_CROSSFIT_FOLDS = 3
TEST_POLICY_BOOTSTRAP_SAMPLES = 20
TEST_INTEGRATION_BOOTSTRAP_SAMPLES = 8


@pytest.fixture
def logistic_learner() -> BaseEstimator:
    return LogisticRegression(max_iter=TEST_LOGISTIC_MAX_ITER)


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


@pytest.fixture
def integration_cache_dir() -> Path:
    return ensure_directory(Path("cache") / "pytest-integration")


@pytest.fixture
def integration_pipeline_config(integration_cache_dir: Path) -> PipelineConfig:
    logistic = LogisticRegression(max_iter=TEST_LOGISTIC_MAX_ITER)
    classifier = RandomForestClassifier(
        n_estimators=TEST_INTEGRATION_FOREST_N_ESTIMATORS,
        random_state=TEST_RANDOM_STATE,
    )
    regressor = RandomForestRegressor(
        n_estimators=TEST_INTEGRATION_FOREST_N_ESTIMATORS,
        random_state=TEST_RANDOM_STATE,
    )
    return PipelineConfig(
        data=DATA_CONFIG,
        split=SplitConfig(
            test_fraction=0.0,
            random_state=DEFAULT_SPLIT_RANDOM_STATE,
        ),
        diagnostics=DiagnosticConfig(propensity_learner=logistic),
        ate_estimators=[
            IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic),
            DoublyRobustATEEstimatorSpec(
                kind=ATEKind.AIPW,
                outcome_learner=classifier,
                propensity_learner=logistic,
            ),
        ],
        cate_estimators=[
            MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER, outcome_learner=classifier),
            MetaCATEEstimatorSpec(
                kind=CATEKind.X_LEARNER,
                outcome_learner=classifier,
                effect_learner=regressor,
                propensity_learner=logistic,
            ),
        ],
        sensitivity=None,
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=logistic,
            outcome_learner=classifier,
            dr_crossfit_folds=TEST_POLICY_CROSSFIT_FOLDS,
            rate_bootstrap_samples=TEST_INTEGRATION_BOOTSTRAP_SAMPLES,
            eceth_bootstrap_samples=TEST_INTEGRATION_BOOTSTRAP_SAMPLES,
            random_state=TEST_RANDOM_STATE,
        ),
        policy=None,
        results_dir=str(integration_cache_dir),
    )

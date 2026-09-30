from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
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
from causal_pipeline.settings import DATA_CONFIG
from causal_pipeline.utils import ensure_directory


@pytest.fixture
def logistic_learner() -> BaseEstimator:
    return LogisticRegression(max_iter=500)


@pytest.fixture
def forest_learner() -> BaseEstimator:
    return RandomForestRegressor(n_estimators=10, random_state=0)


@pytest.fixture
def forest_classifier() -> BaseEstimator:
    return RandomForestClassifier(n_estimators=10, random_state=0)


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
            train_fraction=0.7,
            validation_fraction=0.3,
            test_fraction=0.0,
            random_state=0,
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
        results_dir="results",
    )


@pytest.fixture
def pipeline_config_with_policy(
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    forest_classifier: BaseEstimator,
) -> PipelineConfig:
    return PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=0,
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
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=logistic_learner,
            outcome_learner=forest_classifier,
            dr_crossfit_folds=3,
            rate_bootstrap_samples=20,
            random_state=0,
        ),
        policy=[PolicyTreeMethodSpec(kind=PolicyKind.POLICY_TREE)],
        results_dir="results",
    )


@pytest.fixture
def df_synthetic_binary() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_observations = 200
    treatment = rng.integers(0, 2, size=n_observations)
    x1 = rng.normal(size=n_observations)
    x2 = rng.normal(size=n_observations)
    logit = -0.3 + 0.8 * treatment + 0.4 * x1
    outcome = (logit + rng.normal(size=n_observations) > 0).astype(int)
    return pd.DataFrame({"y": outcome, "t": treatment, "x1": x1, "x2": x2})


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
    logistic = LogisticRegression(max_iter=500)
    classifier = RandomForestClassifier(n_estimators=8, random_state=0)
    regressor = RandomForestRegressor(n_estimators=8, random_state=0)
    return PipelineConfig(
        data=DATA_CONFIG,
        split=SplitConfig(
            train_fraction=0.7,
            validation_fraction=0.3,
            test_fraction=0.0,
            random_state=0,
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
            dr_crossfit_folds=3,
            rate_bootstrap_samples=8,
            random_state=0,
        ),
        policy=None,
        results_dir=str(integration_cache_dir),
    )

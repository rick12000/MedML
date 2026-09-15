from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_pipeline.config import (
    ATEEstimatorSpec,
    ATEKind,
    CATEEstimatorSpec,
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
    DiagnosticConfig,
    LearnerSpec,
    OutcomeType,
    PipelineConfig,
    PolicyConfig,
    SensitivityConfig,
    SplitConfig,
    TreatmentMode,
)


@pytest.fixture
def logistic_learner() -> LearnerSpec:
    return LearnerSpec(name="logistic_regression", params={"max_iter": 500})


@pytest.fixture
def forest_learner() -> LearnerSpec:
    return LearnerSpec(
        name="random_forest",
        params={"n_estimators": 10, "random_state": 0},
    )


@pytest.fixture
def forest_classifier() -> LearnerSpec:
    return LearnerSpec(
        name="random_forest_classifier",
        params={"n_estimators": 10, "random_state": 0},
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
    logistic_learner: LearnerSpec,
    forest_learner: LearnerSpec,
) -> PipelineConfig:
    return PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(
            train_fraction=0.7,
            validation_fraction=0.3,
            test_fraction=0.0,
            random_state=0,
        ),
        diagnostics=DiagnosticConfig(
            propensity_learner=logistic_learner,
            plot_propensity_overlap=False,
        ),
        ate_estimators=[
            ATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        ],
        cate_estimators=[
            CATEEstimatorSpec(kind=CATEKind.S_LEARNER, base_learner=forest_learner),
        ],
        sensitivity=SensitivityConfig(enabled=False),
        cate_evaluation=CATEEvaluationConfig(
            dr_crossfit_folds=3,
            rate_bootstrap_samples=20,
            random_state=0,
        ),
        policy=PolicyConfig(enabled=False),
        results_dir="results",
    )


@pytest.fixture
def pipeline_config_with_policy(
    binary_data_config: DataConfig,
    logistic_learner: LearnerSpec,
    forest_learner: LearnerSpec,
) -> PipelineConfig:
    return PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=0,
        ),
        diagnostics=DiagnosticConfig(
            propensity_learner=logistic_learner,
            plot_propensity_overlap=False,
        ),
        ate_estimators=[
            ATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        ],
        cate_estimators=[
            CATEEstimatorSpec(kind=CATEKind.S_LEARNER, base_learner=forest_learner),
        ],
        sensitivity=SensitivityConfig(enabled=False),
        cate_evaluation=CATEEvaluationConfig(
            dr_crossfit_folds=3,
            rate_bootstrap_samples=20,
            random_state=0,
        ),
        policy=PolicyConfig(enabled=True),
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

"""Learner defaults and fold-local tuning helpers."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from causal_pipeline.learner_defaults import (
    DEFAULT_LOGISTIC_MAX_ITER,
    apply_defaults,
    default_outcome_regressor,
)
from causal_pipeline.learner_tuning import (
    TUNING_N_TRIALS,
    prepare_learner_for_outer_train,
    TuningTask,
)


def test_apply_defaults_raises_logistic_max_iter() -> None:
    model = apply_defaults(LogisticRegression(max_iter=100))
    assert model.get_params()["max_iter"] == DEFAULT_LOGISTIC_MAX_ITER


def test_random_forest_defaults_map_to_xgboost_regressor() -> None:
    model = apply_defaults(RandomForestRegressor(random_state=0))
    assert type(model).__name__ == "XGBRegressor"


def test_default_outcome_regressor_is_xgboost() -> None:
    model = default_outcome_regressor(random_state=0)
    assert type(model).__name__ == "XGBRegressor"


def test_prepare_learner_skips_study_when_tuning_disabled(
    df_synthetic_binary,
    binary_data_config,
) -> None:
    from causal_pipeline.data import CausalDataset

    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    template = LogisticRegression(max_iter=100)
    learner = prepare_learner_for_outer_train(
        template=template,
        features=dataset.X_adjustment,
        labels=dataset.treatment_series,
        task=TuningTask.CLASSIFICATION,
        tune=False,
        random_state=0,
    )
    assert learner.get_params()["max_iter"] == DEFAULT_LOGISTIC_MAX_ITER


def test_tuning_trial_budget_is_thirty_inclusive() -> None:
    assert TUNING_N_TRIALS == 30

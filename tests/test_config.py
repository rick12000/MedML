import pytest

from causal_pipeline.config import (
    DataConfig,
    LearnerSpec,
    OutcomeType,
    PipelineConfig,
    PolicyConfig,
    SplitConfig,
    TreatmentMode,
    build_sklearn_learner,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


def test_pipeline_config_accepts_valid_no_policy_split(pipeline_config_no_policy: PipelineConfig) -> None:
    assert pipeline_config_no_policy.split.test_fraction == 0.0
    assert pipeline_config_no_policy.policy.enabled is False


@pytest.mark.parametrize(
    "train_fraction,validation_fraction,test_fraction",
    [
        (0.6, 0.2, 0.2),
        (0.5, 0.4, 0.0),
    ],
)
def test_pipeline_config_rejects_invalid_split_when_policy_disabled(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(
            **{
                **pipeline_config_no_policy.model_dump(),
                "split": SplitConfig(
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                    test_fraction=test_fraction,
                    random_state=0,
                ),
            }
        )


def test_pipeline_config_requires_positive_test_when_policy_enabled(
    pipeline_config_with_policy: PipelineConfig,
) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(
            **{
                **pipeline_config_with_policy.model_dump(),
                "split": SplitConfig(
                    train_fraction=0.8,
                    validation_fraction=0.2,
                    test_fraction=0.0,
                    random_state=0,
                ),
            }
        )


@pytest.mark.parametrize(
    "learner_name,expected_type",
    [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestRegressor),
    ],
)
def test_build_sklearn_learner_returns_expected_estimator(
    learner_name: str,
    expected_type: type,
) -> None:
    if learner_name == "logistic_regression":
        spec = LearnerSpec(name=learner_name, params={"max_iter": 200})
    else:
        spec = LearnerSpec(name=learner_name, params={"n_estimators": 5, "random_state": 0})
    estimator = build_sklearn_learner(spec)
    assert isinstance(estimator, expected_type)


def test_build_sklearn_learner_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        build_sklearn_learner(LearnerSpec(name="not_a_learner", params={}))

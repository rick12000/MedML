import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal_pipeline.config import (
    ATEKind,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DataConfig,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    SplitConfig,
    TreatmentMode,
    clone_estimator,
    predict_outcome_mean,
    require_classifier,
)


def test_pipeline_config_accepts_valid_no_policy_split(pipeline_config_no_policy: PipelineConfig) -> None:
    assert pipeline_config_no_policy.split.test_fraction == 0.0
    assert not pipeline_config_no_policy.policy


def test_pipeline_config_rejects_missing_ate_and_cate(binary_data_config) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(
            data=binary_data_config,
            split=SplitConfig(
                train_fraction=0.7,
                validation_fraction=0.3,
                test_fraction=0.0,
                random_state=0,
            ),
        )


@pytest.mark.parametrize(
    "train_fraction,validation_fraction,test_fraction",
    [
        (0.6, 0.2, 0.2),
        (0.5, 0.4, 0.0),
    ],
)
def test_pipeline_config_rejects_invalid_split_without_policy(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(
            data=pipeline_config_no_policy.data,
            split=SplitConfig(
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                random_state=0,
            ),
            ate_estimators=pipeline_config_no_policy.ate_estimators,
            cate_estimators=pipeline_config_no_policy.cate_estimators,
            diagnostics=pipeline_config_no_policy.diagnostics,
        )


def test_pipeline_config_requires_positive_test_when_policy_configured(
    pipeline_config_with_policy: PipelineConfig,
) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(
            data=pipeline_config_with_policy.data,
            split=SplitConfig(
                train_fraction=0.8,
                validation_fraction=0.2,
                test_fraction=0.0,
                random_state=0,
            ),
            ate_estimators=pipeline_config_with_policy.ate_estimators,
            cate_estimators=pipeline_config_with_policy.cate_estimators,
            diagnostics=pipeline_config_with_policy.diagnostics,
            cate_evaluation=pipeline_config_with_policy.cate_evaluation,
            policy=pipeline_config_with_policy.policy,
        )


def test_empty_policy_list_is_treated_as_disabled(
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    config = PipelineConfig(
        data=pipeline_config_no_policy.data,
        split=pipeline_config_no_policy.split,
        ate_estimators=pipeline_config_no_policy.ate_estimators,
        cate_estimators=pipeline_config_no_policy.cate_estimators,
        policy=[],
    )
    assert not config.policy


def test_clone_estimator_fit_does_not_mutate_original() -> None:
    original = LogisticRegression(max_iter=200)
    cloned = clone_estimator(original)
    cloned.fit([[0.0], [1.0]], [0, 1])
    assert not hasattr(original, "coef_")
    assert hasattr(cloned, "coef_")


def test_ipw_ate_spec_requires_propensity_learner() -> None:
    with pytest.raises(ValidationError):
        IPWATEEstimatorSpec(kind=ATEKind.IPW)


def test_meta_cate_spec_requires_outcome_learner() -> None:
    with pytest.raises(ValidationError):
        MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER)


@pytest.mark.parametrize("kind", [CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER])
def test_two_stage_meta_learners_require_effect_learner(kind: CATEKind, forest_classifier) -> None:
    with pytest.raises(ValidationError):
        MetaCATEEstimatorSpec(kind=kind, outcome_learner=forest_classifier)


def test_causal_forest_cate_spec_requires_learners(forest_learner: RandomForestRegressor) -> None:
    with pytest.raises(ValidationError):
        CausalForestCATEEstimatorSpec(
            kind=CATEKind.CAUSAL_FOREST,
            outcome_learner=forest_learner,
        )


def test_data_config_rejects_non_integer_continuous_treatment() -> None:
    with pytest.raises(ValidationError):
        DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.BINARY,
            treatment_mode=TreatmentMode.BINARY,
            control_value=0.0,
            treatment_values=[0.0, 1.5],
        )


def test_data_config_rejects_binary_mode_with_three_arms() -> None:
    with pytest.raises(ValidationError):
        DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.BINARY,
            treatment_mode=TreatmentMode.BINARY,
            control_value=0,
            treatment_values=[0, 1, 2],
        )


def test_predict_outcome_mean_returns_positive_class_probability() -> None:
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    predicted = predict_outcome_mean(model, X)
    expected = model.predict_proba(X)[:, 1]
    assert predicted.shape == y.shape
    assert np.allclose(predicted, expected)


def test_require_classifier_rejects_regressor(forest_learner: RandomForestRegressor) -> None:
    with pytest.raises(TypeError):
        require_classifier(forest_learner, "outcome_learner")


def test_predict_outcome_mean_uses_regressor_predict(forest_learner: RandomForestRegressor) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.1, 0.4, 0.6, 0.9])
    forest_learner.fit(X, y)
    predicted = predict_outcome_mean(forest_learner, X)
    assert predicted.shape == y.shape
    assert np.allclose(predicted, forest_learner.predict(X))


@pytest.mark.parametrize("n_arms", [3, 10])
def test_multi_treatment_config_accepts_at_least_three_arms(n_arms: int) -> None:
    DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1"],
        effect_modifiers=["x1"],
        outcome_type=OutcomeType.BINARY,
        treatment_mode=TreatmentMode.MULTI,
        control_value=0,
        treatment_values=list(range(n_arms)),
    )


def test_multi_treatment_config_rejects_two_arms() -> None:
    with pytest.raises(ValidationError):
        DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.BINARY,
            treatment_mode=TreatmentMode.MULTI,
            control_value=0,
            treatment_values=[0, 1],
        )


def test_default_binary_config_uses_classifiers_for_outcome_and_propensity() -> None:
    from causal_pipeline.settings import build_pipeline_config

    config = build_pipeline_config(include_policy=False)
    assert config.data.outcome_type == OutcomeType.BINARY
    require_classifier(config.diagnostics.propensity_learner, "diagnostics")
    require_classifier(config.cate_evaluation.propensity_learner, "evaluation propensity")
    require_classifier(config.cate_evaluation.outcome_learner, "evaluation outcome")
    for spec in config.ate_estimators:
        require_classifier(spec.propensity_learner, spec.kind.value)
        if hasattr(spec, "outcome_learner"):
            require_classifier(spec.outcome_learner, spec.kind.value)
    for spec in config.cate_estimators:
        require_classifier(spec.outcome_learner, spec.kind.value)
        if spec.kind == CATEKind.CAUSAL_FOREST:
            require_classifier(spec.propensity_learner, spec.kind.value)

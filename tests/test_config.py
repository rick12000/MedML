from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal_pipeline.config import (
    ATEKind,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DataConfig,
    DEFAULT_SPLIT_RANDOM_STATE,
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
from causal_pipeline.learner_resolve import (
    fill_cate_spec_learners,
    materialize_ipw_propensity,
    materialize_meta_cate_learners,
)


def test_pipeline_config_accepts_valid_no_policy_split(
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    assert pipeline_config_no_policy.split.test_fraction == 0.0
    assert not pipeline_config_no_policy.policy


def test_pipeline_config_rejects_missing_ate_and_cate(binary_data_config: DataConfig) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(
            data=binary_data_config,
            split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
        )


@pytest.mark.parametrize("test_fraction", [0.2, 1.0])
def test_pipeline_config_rejects_a_test_split_without_policy(
    test_fraction: float,
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(
            data=pipeline_config_no_policy.data,
            split=SplitConfig(test_fraction=test_fraction, random_state=DEFAULT_SPLIT_RANDOM_STATE),
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
            split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
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


def test_ipw_ate_spec_accepts_missing_propensity_learner() -> None:
    spec = IPWATEEstimatorSpec(kind=ATEKind.IPW)
    learner = materialize_ipw_propensity(spec)
    assert type(learner).__name__ == "XGBClassifier"


def test_meta_cate_spec_accepts_missing_outcome_learner() -> None:
    spec = MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER)
    outcome, effect, propensity = materialize_meta_cate_learners(
        spec,
        outcome_type=OutcomeType.CONTINUOUS,
    )
    assert type(outcome).__name__ == "XGBRegressor"
    assert effect is None
    assert propensity is None


@pytest.mark.parametrize("kind", [CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER])
def test_two_stage_meta_learners_materialize_effect_and_propensity(
    kind: CATEKind,
) -> None:
    spec = MetaCATEEstimatorSpec(kind=kind)
    outcome, effect, propensity = materialize_meta_cate_learners(
        spec,
        outcome_type=OutcomeType.CONTINUOUS,
    )
    assert type(outcome).__name__ == "XGBRegressor"
    assert type(effect).__name__ == "XGBRegressor"
    assert type(propensity).__name__ == "XGBClassifier"


def test_causal_forest_cate_spec_accepts_missing_learners() -> None:
    spec = CausalForestCATEEstimatorSpec(kind=CATEKind.CAUSAL_FOREST)
    filled = fill_cate_spec_learners(spec, outcome_type=OutcomeType.CONTINUOUS)
    assert type(filled.outcome_learner).__name__ == "XGBRegressor"
    assert type(filled.propensity_learner).__name__ == "XGBClassifier"


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
    features = np.array([[0.0], [1.0], [0.0], [1.0]])
    labels = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(features, labels)
    predicted = predict_outcome_mean(estimator=model, features=features)
    expected = model.predict_proba(features)[:, 1]
    assert predicted.shape == labels.shape
    assert np.allclose(predicted, expected)


def test_require_classifier_rejects_regressor(forest_learner: RandomForestRegressor) -> None:
    with pytest.raises(TypeError):
        require_classifier(estimator=forest_learner, role="outcome_learner")


def test_predict_outcome_mean_uses_regressor_predict(forest_learner: RandomForestRegressor) -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0.1, 0.4, 0.6, 0.9])
    forest_learner.fit(features, labels)
    predicted = predict_outcome_mean(estimator=forest_learner, features=features)
    assert predicted.shape == labels.shape
    assert np.allclose(predicted, forest_learner.predict(features))


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

"""Fold-local tuning helpers for learners built from approved architectures."""

from __future__ import annotations

from optuna.samplers import GPSampler, TPESampler
from sklearn.linear_model import LogisticRegression

from causal_pipeline.config import CATEKind, MetaCATEEstimatorSpec
from causal_pipeline.data import CausalDataset
from causal_pipeline.learner_tuning import (
    TUNING_N_TRIALS,
    TuningTask,
    make_tuning_sampler,
    prepare_learner_for_outer_train,
    resolve_cate_learners,
)


def test_prepare_learner_skips_study_when_tuning_disabled(
    df_synthetic_binary,
    binary_data_config,
) -> None:
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
    assert learner.get_params()["max_iter"] == 100
    assert template.get_params()["max_iter"] == 100


def test_tuning_trial_budget_is_thirty_inclusive() -> None:
    assert TUNING_N_TRIALS == 30


def test_make_tuning_sampler_uses_tpe_without_torch(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal_pipeline.learner_tuning.importlib.util.find_spec",
        lambda name: None if name == "torch" else __import__("importlib.util").find_spec(name),
    )
    sampler = make_tuning_sampler(random_state=0)
    assert isinstance(sampler, TPESampler)


def _count_effect_tuning_calls(monkeypatch) -> list[int]:
    """Count effect-learner searches (regression, no outcome score) that actually tune."""
    calls = [0]
    real_prepare = prepare_learner_for_outer_train

    def wrapped(*args, **kwargs):
        if kwargs.get("tune") and kwargs.get("score_outcome") is None and (
            kwargs.get("task") == TuningTask.REGRESSION
        ):
            calls[0] += 1
        return real_prepare(*args, **{**kwargs, "tune": False})

    monkeypatch.setattr("causal_pipeline.learner_tuning.prepare_learner_for_outer_train", wrapped)
    return calls


def test_binary_meta_cate_skips_effect_learner_optuna(
    df_synthetic_binary,
    binary_data_config,
    monkeypatch,
) -> None:
    calls = _count_effect_tuning_calls(monkeypatch)
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    spec = MetaCATEEstimatorSpec(
        kind=CATEKind.X_LEARNER,
        alias="x",
        n_folds=3,
        random_state=0,
        effect_learner="xgboost",
    )
    resolve_cate_learners(train=dataset, spec=spec, tune=True)
    assert calls[0] == 0


def test_continuous_meta_cate_tunes_effect_learner(
    constant_effect_cohort,
    recovery_data_config,
    monkeypatch,
) -> None:
    calls = _count_effect_tuning_calls(monkeypatch)
    frame, _ = constant_effect_cohort
    dataset = CausalDataset(data=recovery_data_config, df=frame)
    spec = MetaCATEEstimatorSpec(
        kind=CATEKind.X_LEARNER,
        alias="x",
        n_folds=3,
        random_state=0,
        effect_learner="xgboost",
    )
    resolve_cate_learners(train=dataset, spec=spec, tune=True)
    assert calls[0] == 1


def test_make_tuning_sampler_uses_gp_when_torch_present(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal_pipeline.learner_tuning.importlib.util.find_spec",
        lambda name: object() if name == "torch" else __import__("importlib.util").find_spec(name),
    )
    sampler = make_tuning_sampler(random_state=0)
    assert isinstance(sampler, GPSampler)

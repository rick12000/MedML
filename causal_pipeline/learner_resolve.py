"""Materialize optional learner fields on estimator and analysis configs."""

from __future__ import annotations

from sklearn.base import BaseEstimator

from causal_pipeline.config import (
    CATEEstimatorSpec,
    CATEEvaluationConfig,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DEFAULT_LEARNER_RANDOM_STATE,
    OverlapConfig,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    DRPolicyTreeMethodSpec,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    SensitivityConfig,
)
from causal_pipeline.learner_defaults import (
    materialize_effect_learner,
    materialize_outcome_learner,
    materialize_propensity_learner,
)


def materialize_overlap_propensity(config: OverlapConfig) -> BaseEstimator:
    return materialize_propensity_learner(
        config.propensity_learner,
        random_state=config.random_state,
    )


def screen_outcome_learners(
    specs: list[object],
    outcome_type: OutcomeType,
    random_state: int,
) -> list[BaseEstimator]:
    """One copy of each configured outcome learner. Defaults apply when none are set."""
    learners: list[BaseEstimator] = []
    seen: set[str] = set()
    for spec in specs:
        if "outcome_learner" not in type(spec).model_fields:
            continue
        learner = materialize_outcome_learner(
            getattr(spec, "outcome_learner"),
            outcome_type=outcome_type,
            random_state=random_state,
        )
        identity = f"{type(learner).__name__}:{learner.get_params(deep=False)}"
        if identity in seen:
            continue
        seen.add(identity)
        learners.append(learner)
    if learners:
        return learners
    return [
        materialize_outcome_learner(None, outcome_type=outcome_type, random_state=random_state),
    ]


def materialize_sensitivity_outcome(
    config: SensitivityConfig,
    outcome_type: OutcomeType,
    random_state: int = DEFAULT_LEARNER_RANDOM_STATE,
) -> BaseEstimator:
    return materialize_outcome_learner(
        config.outcome_learner,
        outcome_type=outcome_type,
        random_state=random_state,
    )


def materialize_evaluation_learners(
    config: CATEEvaluationConfig,
    outcome_type: OutcomeType,
) -> tuple[BaseEstimator, BaseEstimator]:
    random_state = config.random_state
    propensity = materialize_propensity_learner(config.propensity_learner, random_state=random_state)
    outcome = materialize_outcome_learner(
        config.outcome_learner,
        outcome_type=outcome_type,
        random_state=random_state,
    )
    return propensity, outcome


def materialize_ipw_propensity(spec: IPWATEEstimatorSpec) -> BaseEstimator:
    return materialize_propensity_learner(
        spec.propensity_learner,
        random_state=spec.random_state,
    )


def materialize_doubly_robust_learners(
    spec: DoublyRobustATEEstimatorSpec,
    outcome_type: OutcomeType,
) -> tuple[BaseEstimator, BaseEstimator]:
    propensity = materialize_propensity_learner(spec.propensity_learner, random_state=spec.random_state)
    outcome = materialize_outcome_learner(
        spec.outcome_learner,
        outcome_type=outcome_type,
        random_state=spec.random_state,
    )
    return outcome, propensity


def materialize_doubleml_learners(
    spec: DoubleMLATEEstimatorSpec,
    outcome_type: OutcomeType,
) -> tuple[BaseEstimator, BaseEstimator]:
    propensity = materialize_propensity_learner(
        spec.propensity_learner,
        random_state=DEFAULT_LEARNER_RANDOM_STATE,
    )
    outcome = materialize_outcome_learner(
        spec.outcome_learner,
        outcome_type=outcome_type,
        random_state=DEFAULT_LEARNER_RANDOM_STATE,
    )
    return outcome, propensity


def materialize_meta_cate_learners(
    spec: MetaCATEEstimatorSpec,
    outcome_type: OutcomeType,
) -> tuple[BaseEstimator, BaseEstimator | None, BaseEstimator | None]:
    outcome = materialize_outcome_learner(
        spec.outcome_learner,
        outcome_type=outcome_type,
        random_state=spec.random_state,
    )
    two_stage = {CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER}
    if spec.kind not in two_stage:
        return outcome, None, None
    effect = materialize_effect_learner(spec.effect_learner, random_state=spec.random_state)
    propensity = materialize_propensity_learner(spec.propensity_learner, random_state=spec.random_state)
    return outcome, effect, propensity


def materialize_causal_forest_learners(
    spec: CausalForestCATEEstimatorSpec,
    outcome_type: OutcomeType,
) -> tuple[BaseEstimator, BaseEstimator]:
    outcome = materialize_outcome_learner(
        spec.outcome_learner,
        outcome_type=outcome_type,
        random_state=DEFAULT_LEARNER_RANDOM_STATE,
    )
    propensity = materialize_propensity_learner(
        spec.propensity_learner,
        random_state=DEFAULT_LEARNER_RANDOM_STATE,
    )
    return outcome, propensity


def fill_cate_spec_learners(
    spec: CATEEstimatorSpec,
    outcome_type: OutcomeType,
) -> CATEEstimatorSpec:
    if isinstance(spec, MetaCATEEstimatorSpec):
        outcome, effect, propensity = materialize_meta_cate_learners(spec, outcome_type=outcome_type)
        return spec.model_copy(
            update={
                "outcome_learner": outcome,
                "effect_learner": effect,
                "propensity_learner": propensity,
            },
        )
    if isinstance(spec, CausalForestCATEEstimatorSpec):
        outcome, propensity = materialize_causal_forest_learners(spec, outcome_type=outcome_type)
        return spec.model_copy(
            update={
                "outcome_learner": outcome,
                "propensity_learner": propensity,
            },
        )
    return spec


def materialize_dr_policy_learners(
    spec: DRPolicyTreeMethodSpec,
    outcome_type: OutcomeType,
    random_state: int = DEFAULT_LEARNER_RANDOM_STATE,
) -> tuple[BaseEstimator, BaseEstimator]:
    outcome = materialize_outcome_learner(
        spec.outcome_learner,
        outcome_type=outcome_type,
        random_state=random_state,
    )
    propensity = materialize_propensity_learner(spec.propensity_learner, random_state=random_state)
    return outcome, propensity

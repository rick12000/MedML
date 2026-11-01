"""Held-out policy values: each policy and each reference treatment on the same patients.

Every candidate assigns one treatment to each held-out patient. The patient's realized
gain is the oriented doubly robust score for that treatment, on the incremental scale
where control is zero. Differences between candidates are paired patient-level means,
so their intervals use the same bootstrap resamples for both sides.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_pipeline.config import JsonValue
from causal_pipeline.crossfit import dataset_groups
from causal_pipeline.data import CausalDataset
from causal_pipeline.estimator_identity import normalize_estimator_alias
from causal_pipeline.evaluation import resample_positions
from causal_pipeline.policy import (
    BOOTSTRAP_CI_LOWER_PERCENTILE,
    BOOTSTRAP_CI_UPPER_PERCENTILE,
    FittedPolicy,
    assigned_treatments,
    contrast_column_name,
    reward_direction,
)
from causal_pipeline.policy_figures import policy_slice_slug

logger = logging.getLogger(__name__)

NEVER_TREAT = "never_treat"
ALWAYS_TREAT = "always_treat"
OBSERVED = "observed"


@dataclass(frozen=True, eq=False)
class PolicyAssignment:
    """One candidate treatment rule and the gain each held-out patient realizes under it."""

    name: str
    method: str
    source_cate_model: str | None
    treatment: np.ndarray
    gain: np.ndarray
    share_treated: float


@dataclass(frozen=True, eq=False)
class PolicyValueReport:
    """Value table with one row per candidate, plus each candidate's patient-level assignment."""

    table: pd.DataFrame
    assignments: dict[str, PolicyAssignment]


def benefit_matrix(scores: pd.DataFrame, dataset: CausalDataset) -> pd.DataFrame:
    """Oriented DR gain for each treatment value, relative to control.

    Columns follow treatment_values. Control is zero by construction, so every gain
    is a difference from never treating. Lower-is-better outcomes are sign-flipped,
    so a positive gain always means benefit.
    """
    direction = reward_direction(dataset.data.outcome_favorability)
    control = dataset.control_value
    columns = list(scores.columns)
    gains = pd.DataFrame(0.0, index=scores.index, columns=dataset.treatment_values)
    for arm in dataset.treatment_values:
        if arm != control:
            contrast = scores[contrast_column_name(arm, control, columns)].to_numpy(dtype=float)
            gains[arm] = direction * contrast
    return gains


def make_assignment(
    name: str,
    method: str,
    source_cate_model: str | None,
    treatment: np.ndarray,
    gains: pd.DataFrame,
    control_value: JsonValue,
) -> PolicyAssignment:
    arm_index = gains.columns.get_indexer(treatment)
    if np.any(arm_index < 0):
        raise ValueError(f"{name} assigns a treatment outside treatment_values.")
    realized = gains.to_numpy(dtype=float)[np.arange(len(treatment)), arm_index]
    return PolicyAssignment(
        name=name,
        method=method,
        source_cate_model=source_cate_model,
        treatment=treatment,
        gain=realized,
        share_treated=float(np.mean(treatment != control_value)),
    )


def baseline_assignments(dataset: CausalDataset, gains: pd.DataFrame) -> list[PolicyAssignment]:
    """Never treat, treat everyone with each non-control arm, and the observed treatments."""
    control = dataset.control_value
    sample_size = len(dataset.df)
    assignments = [
        make_assignment(
            name=NEVER_TREAT,
            method=NEVER_TREAT,
            source_cate_model=None,
            treatment=np.full(sample_size, control, dtype=object),
            gains=gains,
            control_value=control,
        )
    ]
    for arm in dataset.treatment_values:
        if arm == control:
            continue
        assignments.append(
            make_assignment(
                name=normalize_estimator_alias(f"{ALWAYS_TREAT}_{arm}"),
                method=ALWAYS_TREAT,
                source_cate_model=None,
                treatment=np.full(sample_size, arm, dtype=object),
                gains=gains,
                control_value=control,
            )
        )
    assignments.append(
        make_assignment(
            name=OBSERVED,
            method=OBSERVED,
            source_cate_model=None,
            treatment=dataset.treatment_series.to_numpy(dtype=object),
            gains=gains,
            control_value=control,
        )
    )
    return assignments


def learned_assignment(
    policy: FittedPolicy,
    dataset: CausalDataset,
    gains: pd.DataFrame,
) -> PolicyAssignment:
    return make_assignment(
        name=policy_slice_slug(policy),
        method=policy.method,
        source_cate_model=policy.source_cate_model,
        treatment=assigned_treatments(policy=policy, dataset=dataset),
        gains=gains,
        control_value=dataset.control_value,
    )


def bootstrap_gain_means(
    gains: np.ndarray,
    groups: np.ndarray | pd.Series | None,
    samples: int,
    random_state: int,
) -> np.ndarray:
    """Mean gain of every column in each bootstrap replicate. Columns share the resamples."""
    generator = np.random.default_rng(random_state)
    means = np.empty((samples, gains.shape[1]), dtype=float)
    for replicate in range(samples):
        drawn = resample_positions(gains.shape[0], generator, groups, replace=True)
        means[replicate] = gains[drawn].mean(axis=0)
    return means


def policy_value_table(
    assignments: Sequence[PolicyAssignment],
    comparators: Sequence[PolicyAssignment],
    groups: np.ndarray | pd.Series | None,
    bootstrap_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """One row per assignment. Each comparator adds a paired difference in mean gain and its CI.

    Comparators must also appear in assignments, so their bootstrap columns exist.
    """
    gains = np.column_stack([assignment.gain for assignment in assignments])
    means = bootstrap_gain_means(gains, groups, bootstrap_samples, random_state)
    column_of = {assignment.name: index for index, assignment in enumerate(assignments)}
    rows: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        row: dict[str, object] = {
            "policy": assignment.name,
            "method": assignment.method,
            "source_cate_model": assignment.source_cate_model,
            "n_test": len(assignment.gain),
            "share_treated": assignment.share_treated,
        }
        for comparator in comparators:
            difference = means[:, index] - means[:, column_of[comparator.name]]
            lower, upper = np.percentile(
                difference,
                [BOOTSTRAP_CI_LOWER_PERCENTILE, BOOTSTRAP_CI_UPPER_PERCENTILE],
            )
            label = f"vs_{comparator.name}"
            row[label] = float(assignment.gain.mean() - comparator.gain.mean())
            row[f"{label}_ci_lower"] = float(lower)
            row[f"{label}_ci_upper"] = float(upper)
        rows.append(row)
    return pd.DataFrame(rows)


def patient_assignment_table(assignment: PolicyAssignment, observed: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observed_treatment": observed,
            "assigned_treatment": assignment.treatment,
            "gain": assignment.gain,
        }
    )


def evaluate_policy_values(
    policies: Sequence[FittedPolicy],
    test: CausalDataset,
    test_scores: pd.DataFrame,
    bootstrap_samples: int,
    random_state: int,
) -> PolicyValueReport:
    """Value every fitted policy and the reference treatments on the same held-out patients."""
    logger.info(
        "Valuing %s policy(ies) against never, always, and observed treatment on %s held-out patient(s)...",
        len(policies),
        len(test.df),
    )
    gains = benefit_matrix(scores=test_scores, dataset=test)
    baselines = baseline_assignments(dataset=test, gains=gains)
    learned = [learned_assignment(policy=policy, dataset=test, gains=gains) for policy in policies]
    assignments = [*baselines, *learned]
    table = policy_value_table(
        assignments=assignments,
        comparators=baselines,
        groups=dataset_groups(test),
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    logger.info("...Finished valuing policies")
    return PolicyValueReport(
        table=table,
        assignments={assignment.name: assignment for assignment in assignments},
    )

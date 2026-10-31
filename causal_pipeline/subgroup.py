"""Pre-specified binary splits of an already fitted average treatment effect."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from causal_pipeline.ate import TreatmentContrastScores
from causal_pipeline.config import SubgroupSplit
from causal_pipeline.data import CausalDataset, contrast_columns

logger = logging.getLogger(__name__)

ABOVE_CUTOFF_RULE = "greater_or_equal"
MINIMUM_SIDE_COUNT = 2
SUBGROUP_SPLIT_COLUMNS = [
    "estimator",
    "contrast",
    "column",
    "cutoff",
    "above_rule",
    "split_is_binary",
    "confidence_level",
    "n_above",
    "n_below",
    "effect_above",
    "effect_below",
    "effect_difference",
    "effect_se",
    "effect_ci_lower",
    "effect_ci_upper",
    "effect_pvalue",
    "outcome_above",
    "outcome_below",
    "outcome_difference",
    "outcome_se",
    "outcome_ci_lower",
    "outcome_ci_upper",
    "outcome_pvalue",
    "treatment_above",
    "treatment_below",
    "treatment_difference",
    "treatment_se",
    "treatment_ci_lower",
    "treatment_ci_upper",
    "treatment_pvalue",
]

OVERALL_LABEL = "All"


@dataclass(frozen=True)
class MeanDifference:
    above: float
    below: float
    difference: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    pvalue: float


def subgroup_split_table(
    dataset: CausalDataset,
    splits: list[SubgroupSplit],
    scores: TreatmentContrastScores,
    estimator_id: str,
    estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Wald tests for treatment-effect, split-difference, and group-difference gaps at declared cutoffs.

    The effect gap averages the estimator's treatment-contrast scores on each
    side. The group difference is the unadjusted mean outcome on those same rows.
    The split difference is the treated-minus-untreated mean outcome on each side,
    differenced between the two sides. Each contrast uses its own treated arm
    against the control arm, so the split difference is computed per contrast.
    When ``estimates`` is the fitted ATE table, one overall row per contrast is
    appended. Split columns on that row stay empty.
    """
    scored_rows = dataset.df.iloc[np.asarray(scores.positions, dtype=int)]
    outcome = scored_rows[dataset.outcome].to_numpy(dtype=float)
    treatment = scored_rows[dataset.treatment].to_numpy()
    contrast_values = scores.values.to_numpy(dtype=float)
    arm_by_contrast = contrast_arms(dataset, list(scores.values.columns))
    rows = []
    for split in splits:
        values = split_column_values(scored_rows, split.column)
        above = values >= split.cutoff
        below = ~above
        if int(above.sum()) < MINIMUM_SIDE_COUNT or int(below.sum()) < MINIMUM_SIDE_COUNT:
            raise ValueError(
                f"Column {split.column} at cutoff {split.cutoff} needs at least "
                f"{MINIMUM_SIDE_COUNT} rows on each side."
            )
        above_groups, below_groups = split_groups(scores.groups, above, below)
        outcome_gap = wald_mean_difference(
            above=outcome[above],
            below=outcome[below],
            level=scores.confidence_level,
            above_groups=above_groups,
            below_groups=below_groups,
        )
        for contrast_index, contrast in enumerate(scores.values.columns):
            effect_gap = wald_mean_difference(
                above=contrast_values[above, contrast_index],
                below=contrast_values[below, contrast_index],
                level=scores.confidence_level,
                above_groups=above_groups,
                below_groups=below_groups,
            )
            treatment_gap = treatment_split_gap(
                outcome=outcome,
                treatment=treatment,
                above=above,
                below=below,
                above_groups=above_groups,
                below_groups=below_groups,
                arm=arm_by_contrast[contrast],
                control=dataset.control_value,
                level=scores.confidence_level,
            )
            if treatment_gap is None:
                logger.warning(
                    "Split difference unavailable for %s at cutoff %s (%s): "
                    "fewer than two treated or untreated rows on a side.",
                    split.column,
                    split.cutoff,
                    contrast,
                )
            rows.append(
                {
                    "estimator": estimator_id,
                    "contrast": contrast,
                    "column": split.column,
                    "cutoff": split.cutoff,
                    "above_rule": ABOVE_CUTOFF_RULE,
                    "split_is_binary": is_binary_values(values),
                    "confidence_level": scores.confidence_level,
                    "n_above": int(above.sum()),
                    "n_below": int(below.sum()),
                    "effect_above": effect_gap.above,
                    "effect_below": effect_gap.below,
                    "effect_difference": effect_gap.difference,
                    "effect_se": effect_gap.standard_error,
                    "effect_ci_lower": effect_gap.ci_lower,
                    "effect_ci_upper": effect_gap.ci_upper,
                    "effect_pvalue": effect_gap.pvalue,
                    "outcome_above": outcome_gap.above,
                    "outcome_below": outcome_gap.below,
                    "outcome_difference": outcome_gap.difference,
                    "outcome_se": outcome_gap.standard_error,
                    "outcome_ci_lower": outcome_gap.ci_lower,
                    "outcome_ci_upper": outcome_gap.ci_upper,
                    "outcome_pvalue": outcome_gap.pvalue,
                    **treatment_fields(treatment_gap),
                }
            )
    if estimates is not None:
        for _, estimate_row in estimates.iterrows():
            estimate = float(estimate_row["estimate"])
            standard_error = float(estimate_row["standard_error"])
            if standard_error == 0.0:
                pvalue = 0.0 if estimate != 0.0 else 1.0
            else:
                pvalue = float(2.0 * stats.norm.sf(abs(estimate / standard_error)))
            row = dict.fromkeys(SUBGROUP_SPLIT_COLUMNS, np.nan)
            row.update(
                {
                    "estimator": estimator_id,
                    "contrast": estimate_row["contrast"],
                    "confidence_level": scores.confidence_level,
                    "effect_difference": estimate,
                    "effect_se": standard_error,
                    "effect_ci_lower": float(estimate_row["ci_lower"]),
                    "effect_ci_upper": float(estimate_row["ci_upper"]),
                    "effect_pvalue": pvalue,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=SUBGROUP_SPLIT_COLUMNS)


def available_subgroup_splits(
    columns: pd.Index,
    splits: list[SubgroupSplit],
) -> list[SubgroupSplit]:
    """Keep splits whose column is in the data. Warn for each missing column."""
    available = []
    for split in splits:
        if split.column in columns:
            available.append(split)
            continue
        logger.warning(
            "Skipping subgroup split on %s at cutoff %s because that column is not in the data.",
            split.column,
            split.cutoff,
        )
    if splits and not available:
        logger.warning("Skipping subgroup splits because none of the columns are in the data.")
    return available


def is_binary_values(values: np.ndarray) -> bool:
    """Strict check: every value is exactly 0 or 1."""
    return bool(np.isin(values, (0.0, 1.0)).all())


def split_column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Subgroup split column {column} is not in the dataset.")
    raw = frame[column]
    if not pd.api.types.is_numeric_dtype(raw) or raw.isna().any():
        raise ValueError(
            f"Subgroup split column {column} must be numeric and free of missing values."
        )
    return raw.to_numpy(dtype=float)


def split_groups(
    groups: np.ndarray | None,
    above: np.ndarray,
    below: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if groups is None:
        return None, None
    return groups[above], groups[below]


def contrast_arms(dataset: CausalDataset, contrasts: list[str]) -> dict[str, object]:
    """Map each contrast column such as ``1_vs_0`` to the treatment arm it compares to control."""
    arms = [value for value in dataset.treatment_values if value != dataset.control_value]
    arm_by_contrast = dict(zip(contrast_columns(dataset.control_value, dataset.treatment_values), arms))
    unknown = [contrast for contrast in contrasts if contrast not in arm_by_contrast]
    if unknown:
        raise ValueError(f"Contrast columns {unknown} do not match the configured treatment arms.")
    return arm_by_contrast


def arm_gap_with_influence(
    outcome: np.ndarray,
    treatment: np.ndarray,
    arm: object,
    control: object,
) -> tuple[float, np.ndarray] | None:
    """Mean outcome of ``arm`` minus ``control`` within one side, with per-row influence values.

    Rows in other arms are ignored. Returns None when either arm has fewer than
    two rows on the side, because the difference in means is then undefined.
    """
    treated = treatment == arm
    untreated = treatment == control
    n_treated = int(treated.sum())
    n_untreated = int(untreated.sum())
    if n_treated < MINIMUM_SIDE_COUNT or n_untreated < MINIMUM_SIDE_COUNT:
        return None
    treated_mean = float(np.mean(outcome[treated]))
    untreated_mean = float(np.mean(outcome[untreated]))
    influence = np.zeros(len(outcome))
    influence[treated] = (outcome[treated] - treated_mean) / n_treated
    influence[untreated] = -(outcome[untreated] - untreated_mean) / n_untreated
    return treated_mean - untreated_mean, influence


def treatment_split_gap(
    outcome: np.ndarray,
    treatment: np.ndarray,
    above: np.ndarray,
    below: np.ndarray,
    above_groups: np.ndarray | None,
    below_groups: np.ndarray | None,
    arm: object,
    control: object,
    level: float,
) -> MeanDifference | None:
    """Treated-minus-untreated outcome gap on each side, differenced between the sides."""
    above_gap = arm_gap_with_influence(outcome[above], treatment[above], arm, control)
    below_gap = arm_gap_with_influence(outcome[below], treatment[below], arm, control)
    if above_gap is None or below_gap is None:
        return None
    (above_difference, above_influence), (below_difference, below_influence) = above_gap, below_gap
    influence = np.concatenate([above_influence, -below_influence])
    if above_groups is None or below_groups is None:
        standard_error = float(np.sqrt(np.sum(influence**2)))
    else:
        labels = np.concatenate([above_groups, below_groups])
        standard_error = cluster_robust_standard_error(influence, labels)
    return wald_gap(
        difference=above_difference - below_difference,
        standard_error=standard_error,
        level=level,
        above=above_difference,
        below=below_difference,
    )


def treatment_fields(gap: MeanDifference | None) -> dict[str, float]:
    if gap is None:
        return {
            "treatment_above": np.nan,
            "treatment_below": np.nan,
            "treatment_difference": np.nan,
            "treatment_se": np.nan,
            "treatment_ci_lower": np.nan,
            "treatment_ci_upper": np.nan,
            "treatment_pvalue": np.nan,
        }
    return {
        "treatment_above": gap.above,
        "treatment_below": gap.below,
        "treatment_difference": gap.difference,
        "treatment_se": gap.standard_error,
        "treatment_ci_lower": gap.ci_lower,
        "treatment_ci_upper": gap.ci_upper,
        "treatment_pvalue": gap.pvalue,
    }


def wald_mean_difference(
    above: np.ndarray,
    below: np.ndarray,
    level: float,
    above_groups: np.ndarray | None,
    below_groups: np.ndarray | None,
) -> MeanDifference:
    if len(above) < MINIMUM_SIDE_COUNT or len(below) < MINIMUM_SIDE_COUNT:
        raise ValueError("Each side of a subgroup split needs at least two rows.")
    above_mean = float(np.mean(above))
    below_mean = float(np.mean(below))
    difference = above_mean - below_mean
    if above_groups is None and below_groups is None:
        standard_error = independent_difference_standard_error(above, below)
    elif above_groups is not None and below_groups is not None:
        standard_error = clustered_difference_standard_error(
            above=above,
            below=below,
            above_mean=above_mean,
            below_mean=below_mean,
            above_groups=above_groups,
            below_groups=below_groups,
        )
    else:
        raise ValueError("Cluster ids must be supplied for both sides of a split.")
    return wald_gap(
        difference=difference,
        standard_error=standard_error,
        level=level,
        above=above_mean,
        below=below_mean,
    )


def wald_gap(
    difference: float,
    standard_error: float,
    level: float,
    above: float,
    below: float,
) -> MeanDifference:
    quantile = float(stats.norm.ppf(0.5 + level / 2.0))
    if standard_error == 0.0:
        pvalue = 0.0 if difference != 0.0 else 1.0
    else:
        pvalue = float(2.0 * stats.norm.sf(abs(difference / standard_error)))
    return MeanDifference(
        above=above,
        below=below,
        difference=difference,
        standard_error=standard_error,
        ci_lower=difference - quantile * standard_error,
        ci_upper=difference + quantile * standard_error,
        pvalue=pvalue,
    )


def independent_difference_standard_error(above: np.ndarray, below: np.ndarray) -> float:
    above_variance = float(np.var(above, ddof=1))
    below_variance = float(np.var(below, ddof=1))
    return float(np.sqrt(above_variance / len(above) + below_variance / len(below)))


def clustered_difference_standard_error(
    above: np.ndarray,
    below: np.ndarray,
    above_mean: float,
    below_mean: float,
    above_groups: np.ndarray,
    below_groups: np.ndarray,
) -> float:
    influence = np.concatenate(
        [
            (above - above_mean) / len(above),
            -(below - below_mean) / len(below),
        ]
    )
    labels = np.concatenate([above_groups, below_groups])
    return cluster_robust_standard_error(influence, labels)


def cluster_robust_standard_error(influence: np.ndarray, labels: np.ndarray) -> float:
    """Standard error of a sum of per-row influence values, summed within each cluster first."""
    totals = pd.DataFrame({"influence": influence, "group": labels}).groupby("group")["influence"].sum()
    n_groups = len(totals)
    if n_groups < MINIMUM_SIDE_COUNT:
        raise ValueError("Cluster-robust subgroup intervals need at least two groups.")
    centered = totals.to_numpy(dtype=float) - float(totals.mean())
    scale = n_groups / (n_groups - 1) * float(np.sum(centered**2))
    return float(np.sqrt(scale))

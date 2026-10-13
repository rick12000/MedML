"""Pre-specified binary splits of an already fitted average treatment effect."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from causal_pipeline.ate import TreatmentContrastScores
from causal_pipeline.config import SubgroupSplit
from causal_pipeline.data import CausalDataset

logger = logging.getLogger(__name__)

ABOVE_CUTOFF_RULE = "greater_or_equal"
MINIMUM_SIDE_COUNT = 2
SUBGROUP_SPLIT_COLUMNS = [
    "estimator",
    "contrast",
    "column",
    "cutoff",
    "above_rule",
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
]


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
) -> pd.DataFrame:
    """Wald tests for treatment-effect and raw-outcome gaps at declared cutoffs.

    The effect gap averages the estimator's treatment-contrast scores on each
    side. The outcome gap is the unadjusted mean outcome on those same rows.
    """
    scored_rows = dataset.df.iloc[np.asarray(scores.positions, dtype=int)]
    outcome = scored_rows[dataset.outcome].to_numpy(dtype=float)
    contrast_values = scores.values.to_numpy(dtype=float)
    rows = []
    for split in splits:
        above = split_column_values(scored_rows, split.column) >= split.cutoff
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
            rows.append(
                {
                    "estimator": estimator_id,
                    "contrast": contrast,
                    "column": split.column,
                    "cutoff": split.cutoff,
                    "above_rule": ABOVE_CUTOFF_RULE,
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
                }
            )
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
    quantile = float(stats.norm.ppf(0.5 + level / 2.0))
    if standard_error == 0.0:
        pvalue = 0.0 if difference != 0.0 else 1.0
    else:
        pvalue = float(2.0 * stats.norm.sf(abs(difference / standard_error)))
    return MeanDifference(
        above=above_mean,
        below=below_mean,
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
    totals = (
        pd.DataFrame({"influence": influence, "group": labels}).groupby("group")["influence"].sum()
    )
    n_groups = len(totals)
    if n_groups < MINIMUM_SIDE_COUNT:
        raise ValueError("Cluster-robust subgroup intervals need at least two groups.")
    centered = totals.to_numpy(dtype=float) - float(totals.mean())
    scale = n_groups / (n_groups - 1) * float(np.sum(centered**2))
    return float(np.sqrt(scale))

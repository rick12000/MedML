"""Propensity overlap and covariate balance diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from causallib.utils.stat_utils import calc_weighted_standardized_mean_differences
from pydantic import BaseModel, ConfigDict

from causal_pipeline.config import JsonValue, OverlapConfig, TreatmentMode
from causal_pipeline.crossfit import cross_fit_nuisances, dataset_groups, learner_random_state
from causal_pipeline.learner_resolve import materialize_overlap_propensity
from causal_pipeline.data import CausalDataset, DataPartitions, ipw_weights_multi
from causal_pipeline.figures import (
    PANEL_HEIGHT,
    place_outside_legend,
    publication_style,
    series_style,
    style_axis,
)
from causal_pipeline.utils import save_figure, write_dataframe

AUSTIN_SMD_RESCALE = float(np.sqrt(2.0))
OVERLAP_HISTOGRAM_BINS = 30
OVERLAP_FIGURE_WIDTH = 4.6
OVERLAP_PANEL_HEIGHT = PANEL_HEIGHT
HISTOGRAM_ALPHA = 0.38
HISTOGRAM_EDGE_WIDTH = 0.8
PROPENSITY_AXIS_MIN = 0.0
PROPENSITY_AXIS_MAX = 1.0

logger = logging.getLogger(__name__)

PARTITION_COLUMN = "__overlap_partition"


def draw_propensity_histogram(
    axis: Axes,
    probabilities: np.ndarray,
    treatment: np.ndarray,
    treatment_values: list[JsonValue],
) -> None:
    for arm_index, arm in enumerate(treatment_values):
        style = series_style(arm_index)
        mask = treatment == arm
        axis.hist(
            probabilities[mask],
            bins=OVERLAP_HISTOGRAM_BINS,
            density=True,
            histtype="stepfilled",
            color=style.color,
            alpha=HISTOGRAM_ALPHA,
            edgecolor=style.color,
            linewidth=HISTOGRAM_EDGE_WIDTH,
            label=str(arm),
        )


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    propensity_overlap_path: str | None
    covariate_balance: pd.DataFrame
    covariate_balance_trimmed: pd.DataFrame | None = None
    overlap_counts: pd.DataFrame


class OverlapRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    partitions: DataPartitions
    diagnostics: DiagnosticResult


class OverlapPreprocessor:
    """Score the pooled sample once, write overlap diagnostics, then optionally trim."""

    def __init__(self, overlap: OverlapConfig, minimum_folds: int) -> None:
        self.overlap = overlap
        self.minimum_folds = minimum_folds

    def run(
        self,
        partitions: DataPartitions,
        results_root: str | None = None,
    ) -> OverlapRun:
        logger.info("Cross-fitting the overlap propensity on the pooled analysis sample.")
        pooled = stack_partitions(partitions)
        propensity_learner = materialize_overlap_propensity(self.overlap)
        _, propensity = cross_fit_nuisances(
            covariates=pooled.X_adjustment.copy(),
            treatment=pooled.treatment_series,
            outcome=pooled.outcome_series,
            treatment_values=pooled.treatment_values,
            propensity_learner=propensity_learner,
            outcome_learner=None,
            binary_outcome=False,
            n_folds=self.overlap.n_folds,
            random_state=learner_random_state(propensity_learner, self.overlap.random_state),
            clip_bounds=None,
            groups=dataset_groups(pooled),
            log_scope="overlap",
        )
        overlap_path = "diagnostics/propensity_overlap.png"
        save_path = None
        if results_root is not None:
            save_path = str(Path(results_root) / overlap_path)
        plot_overlap(
            treatment=pooled.treatment_series.to_numpy(),
            probabilities=propensity,
            treatment_mode=pooled.treatment_mode,
            treatment_values=pooled.treatment_values,
            control_value=pooled.control_value,
            save_path=save_path,
        )
        if save_path is not None:
            write_dataframe(
                Path(save_path).with_suffix(".csv"),
                propensity_overlap_frame(
                    treatment=pooled.treatment_series.to_numpy(),
                    probabilities=propensity,
                    treatment_mode=pooled.treatment_mode,
                    treatment_values=pooled.treatment_values,
                    control_value=pooled.control_value,
                ),
            )
        balance = covariate_balance(
            covariates=pooled.X_adjustment.copy(),
            dataset=pooled,
            weights=overlap_weights(
                dataset=pooled,
                propensity=propensity,
                stabilized=self.overlap.stabilized_weights,
            ),
        )
        trimmed_balance = None
        reduced = partitions
        if self.overlap.trim_bounds is not None:
            reduced = trim_partitions(
                partitions=partitions,
                propensity=propensity,
                trim_bounds=self.overlap.trim_bounds,
                minimum_folds=max(self.minimum_folds, self.overlap.n_folds),
            )
            trimmed_pool = stack_partitions(reduced)
            trimmed_balance = covariate_balance(
                covariates=trimmed_pool.X_adjustment.copy(),
                dataset=trimmed_pool,
                weights=overlap_weights(
                    dataset=trimmed_pool,
                    propensity=propensity.loc[trimmed_pool.df.index],
                    stabilized=self.overlap.stabilized_weights,
                ),
            )
        counts = overlap_counts(before=partitions, after=reduced)
        return OverlapRun(
            partitions=reduced,
            diagnostics=DiagnosticResult(
                propensity_overlap_path=overlap_path if save_path is not None else None,
                covariate_balance=balance,
                covariate_balance_trimmed=trimmed_balance,
                overlap_counts=counts,
            ),
        )


def stack_partitions(partitions: DataPartitions) -> CausalDataset:
    """Bind estimation and test rows without changing which partition each row belongs to."""
    frames = [labeled_partition(partitions.estimation, "estimation")]
    if partitions.test is not None:
        frames.append(labeled_partition(partitions.test, "test"))
    stacked = pd.concat(frames, axis=0)
    if stacked.index.has_duplicates:
        raise ValueError("Overlap scoring requires a unique row index.")
    return CausalDataset(data=partitions.estimation.data, df=stacked.drop(columns=PARTITION_COLUMN))


def labeled_partition(dataset: CausalDataset, name: str) -> pd.DataFrame:
    frame = dataset.df.copy()
    frame[PARTITION_COLUMN] = name
    return frame


def trim_partitions(
    partitions: DataPartitions,
    propensity: pd.DataFrame,
    trim_bounds: tuple[float, float],
    minimum_folds: int,
) -> DataPartitions:
    mask = covariate_overlap_mask(propensity, trim_bounds)
    estimation = filter_dataset(partitions.estimation, mask)
    test = None if partitions.test is None else filter_dataset(partitions.test, mask)
    require_overlap_sample(estimation, minimum_folds, "estimation")
    if test is not None:
        require_overlap_sample(test, 1, "test")
    logger.info(
        "Trimmed the overlap sample from %s to %s estimation rows.",
        len(partitions.estimation.df),
        len(estimation.df),
    )
    return DataPartitions(estimation=estimation, test=test)


def filter_dataset(dataset: CausalDataset, mask: pd.Series) -> CausalDataset:
    kept = dataset.df.loc[mask.reindex(dataset.df.index).fillna(False)].copy()
    return CausalDataset(data=dataset.data, df=kept)


def covariate_overlap_mask(propensity: pd.DataFrame, trim_bounds: tuple[float, float]) -> pd.Series:
    """Keep x when every arm probability lies inside the trim bounds."""
    lower, upper = trim_bounds
    values = propensity.to_numpy(dtype=float)
    inside = np.all((values >= lower) & (values <= upper), axis=1)
    finite = np.isfinite(values).all(axis=1)
    return pd.Series(inside & finite, index=propensity.index)


def require_overlap_sample(dataset: CausalDataset, minimum_folds: int, partition: str) -> None:
    counts = dataset.treatment_series.value_counts()
    for arm in dataset.treatment_values:
        available = int(counts.get(arm, 0))
        if available < minimum_folds:
            raise ValueError(
                f"Overlap trimming left {available} rows in arm {arm} on the {partition} partition. "
                f"Cross-fitting needs at least {minimum_folds}."
            )


def overlap_counts(before: DataPartitions, after: DataPartitions) -> pd.DataFrame:
    rows = [
        partition_count("estimation", before.estimation, after.estimation),
    ]
    if before.test is not None and after.test is not None:
        rows.append(partition_count("test", before.test, after.test))
    return pd.DataFrame(rows)


def partition_count(name: str, before: CausalDataset, after: CausalDataset) -> dict[str, int | str]:
    return {
        "partition": name,
        "rows_before": len(before.df),
        "rows_after": len(after.df),
    }


def plot_overlap(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    save_path: str | None = None,
) -> None:
    panel_count = 1 if treatment_mode == TreatmentMode.BINARY else len(treatment_values)
    with publication_style():
        figure, axes = plt.subplots(
            nrows=panel_count,
            figsize=(OVERLAP_FIGURE_WIDTH, OVERLAP_PANEL_HEIGHT * panel_count),
            squeeze=False,
        )
        if treatment_mode == TreatmentMode.BINARY:
            axis = axes[0, 0]
            non_control = next(arm for arm in treatment_values if arm != control_value)
            propensity = probability_of_arm(probabilities, non_control)
            draw_propensity_histogram(
                axis=axis,
                probabilities=propensity,
                treatment=treatment,
                treatment_values=treatment_values,
            )
            axis.set_title("Propensity overlap", loc="left", pad=4)
            axis.set_xlabel(f"P(T={non_control} | X)")
            axis.set_ylabel("Density")
            style_axis(axis)
            place_outside_legend(axis)
            axis.set_xlim(PROPENSITY_AXIS_MIN, PROPENSITY_AXIS_MAX)
        else:
            for panel_index, arm in enumerate(treatment_values):
                axis = axes[panel_index, 0]
                draw_propensity_histogram(
                    axis=axis,
                    probabilities=probability_of_arm(probabilities, arm),
                    treatment=treatment,
                    treatment_values=treatment_values,
                )
                axis.set_title(f"P(T={arm} | X)", loc="left", pad=4)
                style_axis(axis)
                place_outside_legend(axis)
                axis.set_xlim(PROPENSITY_AXIS_MIN, PROPENSITY_AXIS_MAX)
                if panel_index == len(treatment_values) - 1:
                    axis.set_xlabel("Probability")
                axis.set_ylabel("Density")
        figure.tight_layout()
        if save_path is not None:
            save_figure(save_path, figure)
        plt.close(figure)


def covariate_balance(
    covariates: pd.DataFrame,
    dataset: CausalDataset,
    weights: np.ndarray,
) -> pd.DataFrame:
    treatment = dataset.treatment_series
    treatment_values = treatment.to_numpy()
    if dataset.treatment_mode == TreatmentMode.BINARY:
        treated = treatment_values != dataset.control_value
        rows = []
        for column in covariates.columns:
            values = covariates[column].astype(float).to_numpy()
            rows.append(
                {
                    "covariate": column,
                    "unweighted_smd": smd_two_groups(values[treated], values[~treated]),
                    "weighted_smd": smd_two_groups(
                        values[treated],
                        values[~treated],
                        weight_a=weights[treated],
                        weight_b=weights[~treated],
                    ),
                }
            )
        return pd.DataFrame(rows)

    rows = []
    levels = dataset.treatment_values
    for column in covariates.columns:
        values = covariates[column].astype(float).to_numpy()
        for index_a, arm_a in enumerate(levels):
            for arm_b in levels[index_a + 1 :]:
                mask_a = treatment_values == arm_a
                mask_b = treatment_values == arm_b
                rows.append(
                    {
                        "covariate": column,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "unweighted_smd": smd_two_groups(values[mask_a], values[mask_b]),
                        "weighted_smd": smd_two_groups(
                            values[mask_a],
                            values[mask_b],
                            weight_a=weights[mask_a],
                            weight_b=weights[mask_b],
                        ),
                    }
                )
    pairwise = pd.DataFrame(rows)
    aggregate = pairwise.groupby("covariate", as_index=False).agg(
        max_abs_unweighted_smd=("unweighted_smd", lambda values: float(np.max(np.abs(values)))),
        max_abs_weighted_smd=("weighted_smd", lambda values: float(np.max(np.abs(values)))),
    )
    return pairwise.merge(aggregate, on="covariate", how="left")


def overlap_weights(
    dataset: CausalDataset,
    propensity: pd.DataFrame,
    stabilized: bool,
) -> np.ndarray:
    """Inverse-probability weights from an already cross-fit propensity."""
    levels = dataset.treatment_values
    matrix = propensity.loc[:, levels].to_numpy(dtype=float)
    weights = ipw_weights_multi(
        treatment=dataset.treatment_series.to_numpy(),
        propensity_matrix=matrix,
        treatment_levels=levels,
    )
    if not stabilized:
        return weights
    treatment = dataset.treatment_series.to_numpy()
    marginal = {arm: float(np.mean(treatment == arm)) for arm in levels}
    scale = np.array([marginal[value] for value in treatment], dtype=float)
    return weights * scale


def propensity_overlap_frame(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
) -> pd.DataFrame:
    frame = pd.DataFrame({"treatment": treatment})
    if treatment_mode == TreatmentMode.BINARY:
        non_control = next(arm for arm in treatment_values if arm != control_value)
        frame["propensity"] = probability_of_arm(probabilities, non_control)
        return frame
    for arm in treatment_values:
        frame[f"propensity_{arm}"] = probability_of_arm(probabilities, arm)
    return frame


def probability_of_arm(probabilities: pd.DataFrame, arm: JsonValue) -> np.ndarray:
    for column in probabilities.columns:
        if column == arm or str(column) == str(arm):
            return probabilities[column].to_numpy(dtype=float)
    raise KeyError(arm)


def smd_two_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    weight_a: np.ndarray | None = None,
    weight_b: np.ndarray | None = None,
) -> float:
    """Austin standardized difference, with the unweighted variance in the denominator."""
    weights_a = np.ones(len(group_a)) if weight_a is None else weight_a
    weights_b = np.ones(len(group_b)) if weight_b is None else weight_b
    difference = calc_weighted_standardized_mean_differences(
        group_a,
        group_b,
        weights_a,
        weights_b,
        weighted_var=False,
    )
    if not np.isfinite(difference):
        return 0.0
    return float(difference * AUSTIN_SMD_RESCALE)

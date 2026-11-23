"""Propensity overlap and covariate balance diagnostics."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from causallib.utils.stat_utils import calc_weighted_standardized_mean_differences
from pydantic import BaseModel, ConfigDict

from causal_pipeline.balance_figures import (
    COVARIATE_BALANCE_STEM,
    DEFAULT_MAX_COVARIATES,
    LOVE_MIN_PANEL_INCHES,
    LOVE_PANEL_PAD_INCHES,
    LOVE_ROW_INCHES,
    X_LABEL,
    balance_panels,
    draw_balance_panel,
    plot_covariate_balance,
)
from causal_pipeline.config import JsonValue, OverlapConfig, TreatmentMode
from causal_pipeline.crossfit import cross_fit_nuisances, dataset_groups, learner_random_state
from causal_pipeline.learners import build_learner
from causal_pipeline.data import (
    PROPENSITY_CLIP_FLOOR,
    CausalDataset,
    DataPartitions,
    ipw_weights_multi,
)
from causal_pipeline.figures import (
    DIAGNOSTIC_LABEL_SIZE,
    DIAGNOSTIC_LEGEND_SIZE,
    DIAGNOSTIC_TITLE_SIZE,
    INK,
    add_inch_axes,
    diagnostic_color,
    mark_panel,
    panel_letter,
    place_bottom_strip_legend,
    place_legend_under_axis,
    publication_style,
    style_diagnostic_axis,
    text_width,
    widest,
)
from causal_pipeline.utils import DEFAULT_FIGURE_FORMAT, figure_file, save_figure, write_dataframe

AUSTIN_SMD_RESCALE = float(np.sqrt(2.0))
OVERLAP_HISTOGRAM_BINS = 30
OVERLAP_FIGURE_WIDTH = 5.15
OVERLAP_PANEL_HEIGHT = 2.70
HISTOGRAM_ALPHA = 0.32
HISTOGRAM_EDGE_WIDTH = 1.05
PROPENSITY_AXIS_MIN = 0.0
PROPENSITY_AXIS_MAX = 1.0
PROPENSITY_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROPENSITY_TICK_LABELS = ("0", "0.25", "0.5", "0.75", "1")
OVERLAP_TITLE = "Propensity overlap"
DIAGNOSTICS_DIRNAME = "diagnostics"
PROPENSITY_OVERLAP_STEM = "propensity_overlap"
PROPENSITY_BALANCE_STEM = "propensity_balance"
DIAGNOSTIC_PAIR_FIGURE_NAME = figure_file(PROPENSITY_BALANCE_STEM)
PAIR_TOP_INCHES = 0.42
SAMPLE_SIZE_FONT_SIZE = 8.5
SAMPLE_SIZE_LINE_INCHES = 0.20
XLABEL_NOTE_GAP_INCHES = 0.50
NOTE_LEGEND_GAP_INCHES = 0.08
LEGEND_STRIP_INCHES = 0.34
OVERLAP_NOTE_WIDTH_INCHES = 4.3
PAIR_OVERLAP_WIDTH_INCHES = 3.55
PAIR_BALANCE_WIDTH_INCHES = 3.75
PAIR_COLUMN_GAP_INCHES = 0.40
PAIR_LEFT_MARGIN_INCHES = 0.72
PAIR_RIGHT_MARGIN_INCHES = 0.18
PAIR_PANEL_GAP_INCHES = 0.62
PAIR_LABEL_GAP_INCHES = 0.18

logger = logging.getLogger(__name__)

PARTITION_COLUMN = "__overlap_partition"


def draw_propensity_histogram(
    axis: Axes,
    probabilities: np.ndarray,
    treatment: np.ndarray,
    treatment_values: list[JsonValue],
) -> None:
    bins = np.linspace(PROPENSITY_AXIS_MIN, PROPENSITY_AXIS_MAX, OVERLAP_HISTOGRAM_BINS + 1)
    for arm_index, arm in enumerate(treatment_values):
        color = diagnostic_color(arm_index)
        mask = treatment == arm
        axis.hist(
            probabilities[mask],
            bins=bins,
            range=(PROPENSITY_AXIS_MIN, PROPENSITY_AXIS_MAX),
            density=True,
            histtype="stepfilled",
            color=color,
            alpha=HISTOGRAM_ALPHA,
            edgecolor=color,
            linewidth=HISTOGRAM_EDGE_WIDTH,
            label=str(arm),
            zorder=2,
        )
        axis.hist(
            probabilities[mask],
            bins=bins,
            range=(PROPENSITY_AXIS_MIN, PROPENSITY_AXIS_MAX),
            density=True,
            histtype="step",
            color=color,
            linewidth=HISTOGRAM_EDGE_WIDTH,
            label="_nolegend_",
            zorder=3,
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

    def __init__(
        self,
        overlap: OverlapConfig,
        minimum_folds: int,
        figure_format: str = DEFAULT_FIGURE_FORMAT,
    ) -> None:
        self.overlap = overlap
        self.minimum_folds = minimum_folds
        self.figure_format = figure_format

    def run(
        self,
        partitions: DataPartitions,
        results_root: str | None = None,
    ) -> OverlapRun:
        logger.info("Running overlap preprocessing on the pooled analysis sample...")
        pooled = stack_partitions(partitions)
        propensity_learner = build_learner(
            self.overlap.propensity_learner,
            "classification",
            self.overlap.random_state,
        )
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
        overlap_name = figure_file(PROPENSITY_OVERLAP_STEM, self.figure_format)
        overlap_path = f"{DIAGNOSTICS_DIRNAME}/{overlap_name}"
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
        if results_root is not None:
            plot_covariate_balance(
                balance=balance,
                save_path=str(
                    Path(results_root)
                    / DIAGNOSTICS_DIRNAME
                    / figure_file(COVARIATE_BALANCE_STEM, self.figure_format)
                ),
            )
            plot_diagnostic_pair(
                treatment=pooled.treatment_series.to_numpy(),
                probabilities=propensity,
                treatment_mode=pooled.treatment_mode,
                treatment_values=pooled.treatment_values,
                control_value=pooled.control_value,
                balance=balance,
                save_path=str(
                    Path(results_root)
                    / DIAGNOSTICS_DIRNAME
                    / figure_file(PROPENSITY_BALANCE_STEM, self.figure_format)
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
        logger.info("...Finished overlap preprocessing")
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
        "Trimmed the overlap sample from %s to %s estimation rows",
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


class ArmSampleSize(NamedTuple):
    label: str
    count: int
    effective_sample_size: float


def kish_effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS, (sum w)^2 / sum w^2. A constant scale cancels, so it is unchanged by stabilization."""
    finite = np.asarray(weights, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    weight_sum = float(np.sum(finite))
    square_sum = float(np.sum(np.square(finite)))
    if square_sum == 0.0:
        return 0.0
    return weight_sum**2 / square_sum


def arm_effective_sample_sizes(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
) -> list[ArmSampleSize]:
    """Kish ESS within each arm on the rows drawn in the overlap figure.

    Cross-fitting, when the overlap stage uses it, has already stored one
    out-of-fold propensity per row. Estimation and test rows are already
    pooled in that vector. Each row contributes its own weight once; fold
    membership is not averaged. The weight is the inverse probability of the
    received arm, floored the same way as the balance weights.
    """
    if len(treatment) != len(probabilities):
        raise ValueError("Treatment and propensity rows differ.")
    received = propensity_of_received_arm(
        treatment=treatment,
        probabilities=probabilities,
        treatment_mode=treatment_mode,
        treatment_values=treatment_values,
    )
    weights = 1.0 / np.clip(received, PROPENSITY_CLIP_FLOOR, None)
    sizes = []
    for arm in treatment_values:
        mask = treatment == arm
        sizes.append(
            ArmSampleSize(
                label=str(arm),
                count=int(np.sum(mask)),
                effective_sample_size=kish_effective_sample_size(weights[mask]),
            )
        )
    return sizes


def propensity_of_received_arm(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
) -> np.ndarray:
    """P(T = T_i | X_i). A missing binary column is the complement of the column that is present."""
    received = np.full(len(treatment), np.nan, dtype=float)
    known = [arm for arm in treatment_values if propensity_column(probabilities, arm) is not None]
    for arm in known:
        mask = treatment == arm
        received[mask] = probability_of_arm(probabilities, arm)[mask]
    if treatment_mode != TreatmentMode.BINARY or len(known) != 1:
        return received
    missing = [arm for arm in treatment_values if arm not in known]
    if len(missing) != 1:
        return received
    present_probability = probability_of_arm(probabilities, known[0])
    mask = treatment == missing[0]
    received[mask] = 1.0 - present_probability[mask]
    return received


def propensity_column(probabilities: pd.DataFrame, arm: JsonValue) -> str | None:
    for column in probabilities.columns:
        if column == arm or str(column) == str(arm):
            return str(column)
    return None


def format_effective_sample_size(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.1f}"


def arm_sample_label(sample: ArmSampleSize) -> str:
    ess = format_effective_sample_size(sample.effective_sample_size)
    return f"{sample.label}: n = {sample.count}, ESS = {ess}"


def format_arm_sample_sizes(samples: Sequence[ArmSampleSize]) -> str:
    return "; ".join(arm_sample_label(sample) for sample in samples)


def sample_size_note(samples: Sequence[ArmSampleSize], width_inches: float) -> str:
    """One line of arm labels when they fit under the panel, otherwise one arm per line."""
    lines: list[str] = []
    current = ""
    for sample in samples:
        part = arm_sample_label(sample)
        candidate = part if current == "" else f"{current}    {part}"
        if current and text_width(candidate, SAMPLE_SIZE_FONT_SIZE) > width_inches:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def sample_note_inches(note: str) -> float:
    line_count = note.count("\n") + 1 if note else 0
    return SAMPLE_SIZE_LINE_INCHES * line_count


def overlap_legend_inches(note: str) -> float:
    """Strip below an x-axis label: the sample-size line, then the legend."""
    return sample_note_inches(note) + NOTE_LEGEND_GAP_INCHES + LEGEND_STRIP_INCHES


def joint_bottom_inches(note: str) -> float:
    """Same strip, plus room for the x-axis label that hangs below the joint-panel spine."""
    return XLABEL_NOTE_GAP_INCHES + overlap_legend_inches(note)


def place_sample_size_note(figure: Figure, anchor: Axes, note: str, top_fraction: float) -> None:
    position = anchor.get_position()
    figure.text(
        (position.x0 + position.x1) / 2,
        top_fraction,
        note,
        ha="center",
        va="top",
        fontsize=SAMPLE_SIZE_FONT_SIZE,
        color=INK,
        linespacing=1.3,
        transform=figure.transFigure,
    )


def plot_overlap(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    save_path: str | None = None,
) -> None:
    logger.info(
        "Overlap Kish effective sample size: %s",
        format_arm_sample_sizes(
            arm_effective_sample_sizes(
                treatment=treatment,
                probabilities=probabilities,
                treatment_mode=treatment_mode,
                treatment_values=treatment_values,
            )
        ),
    )
    with publication_style():
        figure = compose_propensity_overlap(
            treatment=treatment,
            probabilities=probabilities,
            treatment_mode=treatment_mode,
            treatment_values=treatment_values,
            control_value=control_value,
        )
        if save_path is not None:
            save_figure(save_path, figure)
        plt.close(figure)


def plot_diagnostic_pair(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    balance: pd.DataFrame,
    save_path: str,
    max_covariates: int = DEFAULT_MAX_COVARIATES,
) -> None:
    with publication_style():
        figure = compose_diagnostic_pair(
            treatment=treatment,
            probabilities=probabilities,
            treatment_mode=treatment_mode,
            treatment_values=treatment_values,
            control_value=control_value,
            balance=balance,
            max_covariates=max_covariates,
        )
        save_figure(save_path, figure)
        plt.close(figure)


def compose_propensity_overlap(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
) -> Figure:
    panel_count = overlap_panel_count(treatment_mode=treatment_mode, treatment_values=treatment_values)
    note = sample_size_note(
        arm_effective_sample_sizes(
            treatment=treatment,
            probabilities=probabilities,
            treatment_mode=treatment_mode,
            treatment_values=treatment_values,
        ),
        width_inches=OVERLAP_NOTE_WIDTH_INCHES,
    )
    bottom_inches = overlap_legend_inches(note)
    figure_height = OVERLAP_PANEL_HEIGHT * panel_count + bottom_inches
    figure, grid = plt.subplots(
        nrows=panel_count,
        ncols=1,
        figsize=(OVERLAP_FIGURE_WIDTH, figure_height),
        squeeze=False,
    )
    axes = [grid[index, 0] for index in range(panel_count)]
    handle_axis = draw_overlap_panels(
        axes=axes,
        treatment=treatment,
        probabilities=probabilities,
        treatment_mode=treatment_mode,
        treatment_values=treatment_values,
        control_value=control_value,
    )
    if panel_count > 1:
        for index, axis in enumerate(axes):
            mark_panel(axis, panel_letter(index), x=-0.06, y=1.06, fontsize=DIAGNOSTIC_TITLE_SIZE)
    figure.tight_layout(rect=(0.0, bottom_inches / figure_height, 1.0, 1.0))
    place_sample_size_note(
        figure=figure,
        anchor=axes[-1],
        note=note,
        top_fraction=bottom_inches / figure_height,
    )
    place_bottom_strip_legend(
        figure=figure,
        anchor=axes[-1],
        handles_axis=handle_axis,
        ncol=len(treatment_values),
        fontsize=DIAGNOSTIC_LEGEND_SIZE,
    )
    return figure


def compose_diagnostic_pair(
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    balance: pd.DataFrame,
    max_covariates: int = DEFAULT_MAX_COVARIATES,
) -> Figure:
    """Propensity overlap on the left and covariate balance on the right."""
    panels = balance_panels(balance, max_covariates=max_covariates)
    overlap_count = overlap_panel_count(treatment_mode=treatment_mode, treatment_values=treatment_values)
    overlap_heights = [OVERLAP_PANEL_HEIGHT] * overlap_count
    balance_heights = [balance_panel_height(frame) for _, frame in panels]
    if overlap_count == 1 and len(panels) == 1:
        matched_height = max(overlap_heights[0], balance_heights[0])
        overlap_heights = [matched_height]
        balance_heights = [matched_height]
    left_stack = column_stack_height(overlap_heights)
    right_stack = column_stack_height(balance_heights)
    note = sample_size_note(
        arm_effective_sample_sizes(
            treatment=treatment,
            probabilities=probabilities,
            treatment_mode=treatment_mode,
            treatment_values=treatment_values,
        ),
        width_inches=PAIR_OVERLAP_WIDTH_INCHES,
    )
    bottom_inches = joint_bottom_inches(note)
    figure_height = PAIR_TOP_INCHES + max(left_stack, right_stack) + bottom_inches
    name_width = widest(
        (str(name) for _, frame in panels for name in frame["covariate"].tolist()),
        DIAGNOSTIC_LABEL_SIZE,
    )
    figure_width = (
        PAIR_LEFT_MARGIN_INCHES
        + PAIR_OVERLAP_WIDTH_INCHES
        + PAIR_COLUMN_GAP_INCHES
        + name_width
        + PAIR_LABEL_GAP_INCHES
        + PAIR_BALANCE_WIDTH_INCHES
        + PAIR_RIGHT_MARGIN_INCHES
    )
    figure = plt.figure(figsize=(figure_width, figure_height))
    overlap_axes = stack_panels(
        figure=figure,
        left=PAIR_LEFT_MARGIN_INCHES,
        width=PAIR_OVERLAP_WIDTH_INCHES,
        heights=overlap_heights,
    )
    balance_left = (
        PAIR_LEFT_MARGIN_INCHES + PAIR_OVERLAP_WIDTH_INCHES + PAIR_COLUMN_GAP_INCHES + name_width + PAIR_LABEL_GAP_INCHES
    )
    balance_axes = stack_panels(
        figure=figure,
        left=balance_left,
        width=PAIR_BALANCE_WIDTH_INCHES,
        heights=balance_heights,
    )
    handle_axis = draw_overlap_panels(
        axes=overlap_axes,
        treatment=treatment,
        probabilities=probabilities,
        treatment_mode=treatment_mode,
        treatment_values=treatment_values,
        control_value=control_value,
    )
    for index, ((contrast, frame), axis) in enumerate(zip(panels, balance_axes, strict=True)):
        draw_balance_panel(
            axis=axis,
            frame=frame,
            xlabel=X_LABEL if index == len(balance_axes) - 1 else "",
            contrast=contrast,
        )
    letter_panels([*overlap_axes, *balance_axes])
    place_sample_size_note(
        figure=figure,
        anchor=overlap_axes[-1],
        note=note,
        top_fraction=overlap_axes[-1].get_position().y0 - XLABEL_NOTE_GAP_INCHES / figure_height,
    )
    legend_clearance = XLABEL_NOTE_GAP_INCHES + sample_note_inches(note) + NOTE_LEGEND_GAP_INCHES
    place_legend_under_axis(
        figure=figure,
        anchor=overlap_axes[-1],
        handles_axis=handle_axis,
        ncol=len(treatment_values),
        clearance_inches=legend_clearance,
    )
    place_legend_under_axis(
        figure=figure,
        anchor=balance_axes[-1],
        handles_axis=balance_axes[0],
        ncol=2,
        clearance_inches=legend_clearance,
    )
    return figure


def overlap_panel_count(treatment_mode: TreatmentMode, treatment_values: Sequence[JsonValue]) -> int:
    if treatment_mode == TreatmentMode.BINARY:
        return 1
    return len(treatment_values)


def balance_panel_height(frame: pd.DataFrame) -> float:
    return max(LOVE_MIN_PANEL_INCHES, LOVE_ROW_INCHES * len(frame) + LOVE_PANEL_PAD_INCHES)


def column_stack_height(heights: Sequence[float]) -> float:
    if not heights:
        return 0.0
    return float(sum(heights) + PAIR_PANEL_GAP_INCHES * (len(heights) - 1))


def stack_panels(
    figure: Figure,
    left: float,
    width: float,
    heights: Sequence[float],
) -> list[Axes]:
    axes: list[Axes] = []
    top = PAIR_TOP_INCHES
    for height in heights:
        axes.append(add_inch_axes(figure, left, top, width, height))
        top += height + PAIR_PANEL_GAP_INCHES
    return axes


def letter_panels(axes: Sequence[Axes]) -> None:
    if len(axes) < 2:
        return
    for index, axis in enumerate(axes):
        mark_panel(axis, panel_letter(index), x=-0.06, y=1.06, fontsize=DIAGNOSTIC_TITLE_SIZE)


def draw_overlap_panels(
    axes: Sequence[Axes],
    treatment: np.ndarray,
    probabilities: pd.DataFrame,
    treatment_mode: TreatmentMode,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
) -> Axes:
    """Draw one density panel per propensity and return the axis that carries the legend."""
    if treatment_mode == TreatmentMode.BINARY:
        axis = axes[0]
        non_control = next(arm for arm in treatment_values if arm != control_value)
        draw_propensity_histogram(
            axis=axis,
            probabilities=probability_of_arm(probabilities, non_control),
            treatment=treatment,
            treatment_values=treatment_values,
        )
        finish_overlap_axis(axis, title=OVERLAP_TITLE, xlabel=f"P(T = {non_control} | X)")
        return axis
    for panel_index, arm in enumerate(treatment_values):
        axis = axes[panel_index]
        draw_propensity_histogram(
            axis=axis,
            probabilities=probability_of_arm(probabilities, arm),
            treatment=treatment,
            treatment_values=treatment_values,
        )
        xlabel = "Probability" if panel_index == len(treatment_values) - 1 else ""
        finish_overlap_axis(axis, title=f"P(T = {arm} | X)", xlabel=xlabel)
    return axes[0]


def finish_overlap_axis(axis: Axes, title: str, xlabel: str) -> None:
    style_diagnostic_axis(axis)
    axis.set_title(title, loc="center", pad=8, fontsize=DIAGNOSTIC_TITLE_SIZE, fontweight="bold", color=INK)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_xlim(PROPENSITY_AXIS_MIN, PROPENSITY_AXIS_MAX)
    axis.set_xticks(list(PROPENSITY_TICKS))
    axis.set_xticklabels(list(PROPENSITY_TICK_LABELS))
    if xlabel == "":
        axis.tick_params(axis="x", labelbottom=False)
    axis.xaxis.label.set_size(DIAGNOSTIC_LABEL_SIZE)
    axis.yaxis.label.set_size(DIAGNOSTIC_LABEL_SIZE)
    lower, upper = axis.get_ylim()
    ceiling = upper if upper > lower else 1.0
    axis.set_ylim(0.0, ceiling * 1.16 if ceiling > 0.0 else 1.0)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


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

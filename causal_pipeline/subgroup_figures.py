"""Forest-style table figures for pre-specified subgroup split results."""

from __future__ import annotations

from dataclasses import dataclass
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from causal_pipeline.figures import (
    AXIS_LABEL_SIZE,
    ERROR_LINE_WIDTH,
    INK,
    MARKER_SIZE,
    PUBLICATION_RC,
    REFERENCE_COLOR,
    REFERENCE_LINESTYLE,
    REFERENCE_LINEWIDTH,
    TICK_LABEL_SIZE,
    display_contrast,
    display_estimator,
    publication_style,
    style_axis,
)
from causal_pipeline.subgroup import ABOVE_CUTOFF_RULE
from causal_pipeline.utils import save_figure

SUBGROUP_SPLITS_FIGURE_NAME = "subgroup_splits.png"
ROW_HEIGHT_INCHES = 0.36
HEADER_HEIGHT_INCHES = 0.55
FIGURE_WIDTH_INCHES = 7.8
FOREST_CAP_TOLERANCE = 2.5
MIN_FOREST_HALF_WIDTH = 0.05


@dataclass(frozen=True)
class SubgroupVisualRow:
    estimator: str | None
    label: str
    n_above: int | None = None
    n_below: int | None = None
    effect_difference: float | None = None
    effect_ci_lower: float | None = None
    effect_ci_upper: float | None = None
    effect_pvalue: float | None = None
    outcome_difference: float | None = None
    outcome_ci_lower: float | None = None
    outcome_ci_upper: float | None = None
    outcome_pvalue: float | None = None


def plot_subgroup_splits_table(
    table: pd.DataFrame,
    split_order: list[tuple[str, float]],
    save_path: str,
) -> None:
    rows = build_visual_rows(table=table, split_order=split_order)
    if not rows:
        return
    confidence_level = float(table["confidence_level"].iloc[0])
    ci_percent = int(round(confidence_level * 100))
    with publication_style():
        figure = compose_subgroup_table(
            rows=rows,
            ci_percent=ci_percent,
        )
        save_figure(save_path, figure)
        plt.close(figure)


def build_visual_rows(
    table: pd.DataFrame,
    split_order: list[tuple[str, float]],
) -> list[SubgroupVisualRow]:
    if table.empty:
        return []
    estimators = list(dict.fromkeys(table["estimator"].astype(str)))
    rows: list[SubgroupVisualRow] = []
    for estimator in estimators:
        estimator_rows = table.loc[table["estimator"].astype(str) == estimator]
        first_row = True
        for column, cutoff in split_order:
            matches = estimator_rows.loc[
                (estimator_rows["column"] == column) & (estimator_rows["cutoff"] == cutoff)
            ]
            for _, record in matches.iterrows():
                split_label = split_row_label(
                    column=str(record["column"]),
                    cutoff=float(record["cutoff"]),
                    above_rule=str(record["above_rule"]),
                    contrast=str(record["contrast"]),
                    multi_contrast=estimator_rows["contrast"].nunique() > 1,
                )
                if first_row:
                    label = f"{display_estimator(estimator)}\n{split_label.strip()}"
                    first_row = False
                else:
                    label = split_label
                rows.append(
                    SubgroupVisualRow(
                        estimator=estimator,
                        label=label,
                        n_above=round_patient_count(record["n_above"]),
                        n_below=round_patient_count(record["n_below"]),
                        effect_difference=float(record["effect_difference"]),
                        effect_ci_lower=float(record["effect_ci_lower"]),
                        effect_ci_upper=float(record["effect_ci_upper"]),
                        effect_pvalue=float(record["effect_pvalue"]),
                        outcome_difference=float(record["outcome_difference"]),
                        outcome_ci_lower=float(record["outcome_ci_lower"]),
                        outcome_ci_upper=float(record["outcome_ci_upper"]),
                        outcome_pvalue=float(record["outcome_pvalue"]),
                    )
                )
    return rows


def split_row_label(
    column: str,
    cutoff: float,
    above_rule: str,
    contrast: str,
    multi_contrast: bool,
) -> str:
    cutoff_text = format(cutoff, ".6g")
    if above_rule == ABOVE_CUTOFF_RULE:
        base = f"{column} ≥ {cutoff_text}"
    else:
        base = f"{column} {above_rule} {cutoff_text}"
    if multi_contrast:
        return f"  {base} ({display_contrast(contrast)})"
    return f"  {base}"


def compose_subgroup_table(
    rows: list[SubgroupVisualRow],
    ci_percent: int,
) -> Figure:
    total_rows = len(rows)
    figure_height = HEADER_HEIGHT_INCHES + ROW_HEIGHT_INCHES * total_rows
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    grid = GridSpec(
        nrows=total_rows + 1,
        ncols=8,
        figure=figure,
        height_ratios=[HEADER_HEIGHT_INCHES / ROW_HEIGHT_INCHES] + [1.0] * total_rows,
        width_ratios=[2.5, 0.8, 1.15, 1.45, 0.55, 1.15, 1.45, 0.55],
        hspace=0.05,
        wspace=0.25,
        left=0.05,
        right=0.98,
        top=0.96,
        bottom=0.07,
    )
    write_column_headers(grid=grid, ci_percent=ci_percent)
    effect_axis = add_forest_axis(grid=grid, column=3, row_count=len(rows))
    outcome_axis = add_forest_axis(grid=grid, column=6, row_count=len(rows))
    effect_limits = forest_limits(
        estimates=[row.effect_difference for row in rows],
        lowers=[row.effect_ci_lower for row in rows],
        uppers=[row.effect_ci_upper for row in rows],
    )
    outcome_limits = forest_limits(
        estimates=[row.outcome_difference for row in rows],
        lowers=[row.outcome_ci_lower for row in rows],
        uppers=[row.outcome_ci_upper for row in rows],
    )
    effect_axis.set_xlim(effect_limits)
    outcome_axis.set_xlim(outcome_limits)
    row_positions = np.arange(len(rows), dtype=float)[::-1] + 1.0
    for grid_row, visual in enumerate(rows, start=1):
        data_y = float(row_positions[grid_row - 1])
        write_text_cell(grid, grid_row, 0, visual.label, ha="left", fontweight="regular")
        write_text_cell(
            grid,
            grid_row,
            1,
            f"{visual.n_above} / {visual.n_below}",
            ha="center",
        )
        write_text_cell(
            grid,
            grid_row,
            2,
            format_estimate_ci(
                visual.effect_difference,
                visual.effect_ci_lower,
                visual.effect_ci_upper,
            ),
            ha="center",
        )
        write_text_cell(grid, grid_row, 4, format_p_value(visual.effect_pvalue), ha="center")
        write_text_cell(
            grid,
            grid_row,
            5,
            format_estimate_ci(
                visual.outcome_difference,
                visual.outcome_ci_lower,
                visual.outcome_ci_upper,
            ),
            ha="center",
        )
        write_text_cell(grid, grid_row, 7, format_p_value(visual.outcome_pvalue), ha="center")
        draw_capped_interval(
            axis=effect_axis,
            y=data_y,
            estimate=visual.effect_difference,
            lower=visual.effect_ci_lower,
            upper=visual.effect_ci_upper,
            xlim=effect_axis.get_xlim(),
        )
        draw_capped_interval(
            axis=outcome_axis,
            y=data_y,
            estimate=visual.outcome_difference,
            lower=visual.outcome_ci_lower,
            upper=visual.outcome_ci_upper,
            xlim=outcome_axis.get_xlim(),
        )
    configure_forest_axis(
        axis=effect_axis,
        positions=row_positions,
        xlabel="ATE difference (above − below)",
    )
    configure_forest_axis(
        axis=outcome_axis,
        positions=row_positions,
        xlabel="Outcome difference (above − below)",
    )
    return figure


def write_column_headers(grid: GridSpec, ci_percent: int) -> None:
    headers = [
        "Subgroup",
        "Patients\n(≥ / <)",
        f"ATE diff\n({ci_percent}% CI)",
        "",
        "P value",
        f"Outcome diff\n({ci_percent}% CI)",
        "",
        "P value",
    ]
    for column, header in enumerate(headers):
        axis = grid.figure.add_subplot(grid[0, column])
        axis.axis("off")
        axis.text(
            0.5 if column in {1, 2, 4, 5, 7} else 0.0,
            0.5,
            header,
            ha="center" if column in {1, 2, 4, 5, 7} else "left",
            va="center",
            fontsize=AXIS_LABEL_SIZE,
            color=INK,
            fontweight="bold" if header else "regular",
        )


def write_text_cell(
    grid: GridSpec,
    grid_row: int,
    column: int,
    text: str,
    ha: str,
    fontweight: str = "regular",
) -> None:
    axis = grid.figure.add_subplot(grid[grid_row, column])
    axis.axis("off")
    axis.text(
        0.5 if ha == "center" else 0.0,
        0.5,
        text,
        ha=ha,
        va="center",
        fontsize=TICK_LABEL_SIZE,
        color=INK,
        fontweight=fontweight,
    )


def add_forest_axis(grid: GridSpec, column: int, row_count: int) -> Axes:
    if row_count < 1:
        raise ValueError("Subgroup figure needs at least one data row.")
    axis = grid.figure.add_subplot(grid[1 : row_count + 1, column])
    return axis


def configure_forest_axis(axis: Axes, positions: np.ndarray, xlabel: str) -> None:
    axis.axvline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=REFERENCE_LINEWIDTH)
    axis.set_yticks(positions)
    axis.set_yticklabels([])
    axis.set_ylim(float(positions.min()) - 0.6, float(positions.max()) + 0.6)
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, color=INK)
    style_axis(axis)
    axis.tick_params(axis="y", left=False)


def draw_capped_interval(
    axis: Axes,
    y: float,
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    xlim: tuple[float, float],
) -> None:
    if estimate is None or not np.isfinite(estimate):
        return
    xmin, xmax = xlim
    lower_value = lower if lower is not None and np.isfinite(lower) else estimate
    upper_value = upper if upper is not None and np.isfinite(upper) else estimate
    draw_lower = max(lower_value, xmin)
    draw_upper = min(upper_value, xmax)
    if draw_upper < draw_lower:
        draw_lower = draw_upper = float(np.clip(estimate, xmin, xmax))
    axis.plot(
        [draw_lower, draw_upper],
        [y, y],
        color=INK,
        linewidth=ERROR_LINE_WIDTH,
        solid_capstyle="butt",
        zorder=2,
    )
    axis.plot(
        estimate,
        y,
        linestyle="none",
        marker="o",
        color=INK,
        markersize=MARKER_SIZE,
        zorder=3,
    )
    cap_length = 0.08 * (xmax - xmin)
    if lower is not None and np.isfinite(lower) and lower < xmin:
        axis.annotate(
            "",
            xy=(xmin, y),
            xytext=(xmin + cap_length, y),
            arrowprops={"arrowstyle": "<|-", "color": INK, "lw": ERROR_LINE_WIDTH},
            zorder=4,
        )
    if upper is not None and np.isfinite(upper) and upper > xmax:
        axis.annotate(
            "",
            xy=(xmax, y),
            xytext=(xmax - cap_length, y),
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": ERROR_LINE_WIDTH},
            zorder=4,
        )
def forest_limits(
    estimates: list[float | None],
    lowers: list[float | None],
    uppers: list[float | None],
) -> tuple[float, float]:
    values = []
    for estimate, lower, upper in zip(estimates, lowers, uppers, strict=True):
        if estimate is not None and np.isfinite(estimate):
            values.append(estimate)
        if lower is not None and np.isfinite(lower):
            values.append(lower)
        if upper is not None and np.isfinite(upper):
            values.append(upper)
    if not values:
        return -1.0, 1.0
    array = np.asarray(values, dtype=float)
    endpoint_cap = float(np.max(np.abs(array)))
    endpoint_cap = max(endpoint_cap, MIN_FOREST_HALF_WIDTH)
    half_width = endpoint_cap * 1.08
    widest = float(np.max(array) - np.min(array))
    if widest > FOREST_CAP_TOLERANCE * endpoint_cap:
        half_width = FOREST_CAP_TOLERANCE * endpoint_cap
    half_width = max(half_width, MIN_FOREST_HALF_WIDTH)
    return -half_width, half_width


def round_patient_count(value: object) -> int:
    return int(round(float(value)))


def round_two(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.2f}"


def format_estimate_ci(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
) -> str:
    return f"{round_two(estimate)} ({round_two(lower)}-{round_two(upper)})"


def format_p_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 0.01:
        return "< 0.01"
    return f"{value:.2f}"

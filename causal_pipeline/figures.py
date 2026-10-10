"""Publication figures for causal estimates, diagnostics, and evaluation curves."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from causal_pipeline.utils import save_figure

logger = logging.getLogger(__name__)

# Okabe-Ito colors stay distinct for common color-vision deficiencies and in print.
SERIES_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
LINE_STYLES = ("solid", "dashed", "dotted", "dashdot")
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")
INK = "#1B1B1B"
REFERENCE_COLOR = "#6B6B6B"
REFERENCE_LINESTYLE = (0, (2.2, 1.8))
SINGLE_COLUMN_WIDTH = 3.54
PANEL_HEIGHT = 2.65
EFFECT_FIGURE_WIDTH = 6.10
FOREST_ROW_INCHES = 0.34
FOREST_PANEL_PAD = 0.85
CAPTION_LINE_INCHES = 0.16
CAPTION_GAP_INCHES = 0.20
AXIS_PAD_FRACTION = 0.08
LINE_WIDTH = 1.05
REFERENCE_LINEWIDTH = 0.7
SPINE_WIDTH = 0.6
TICK_LENGTH = 3.0
MARKER_SIZE = 4.8
ERROR_LINE_WIDTH = 0.8
ERROR_CAP_SIZE = 2.2
BAR_HEIGHT = 0.62
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.5
CAPTION_FONT_SIZE = 7.5
PANEL_LABEL_SIZE = 9.0
TITLE_SIZE = 9.5
TABLE_ROW_INCHES = 0.30
TABLE_HEADER_INCHES = 0.32
LEGEND_BAND_INCHES = 0.42
LEGEND_FONT_SIZE = 7.5
P_VALUE_FLOOR = 0.001
STATISTIC_SCIENTIFIC_FLOOR = 0.001
STATISTIC_SCIENTIFIC_CEILING = 1000.0
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
INTERVAL_CAPTION = "Horizontal lines are confidence intervals. The vertical line marks no effect."
POINT_CAPTION = "Bars are means of conditional treatment effects. The vertical line marks no effect."
IDEAL_CALIBRATION_LABEL = "Ideal"
TOC_FIGURE_WIDTH = 5.80
CALIBRATION_FIGURE_WIDTH = 5.20
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": AXIS_LABEL_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": AXIS_LABEL_SIZE,
    "axes.linewidth": SPINE_WIDTH,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.unicode_minus": False,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": SPINE_WIDTH,
    "ytick.major.width": SPINE_WIDTH,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": INK,
}

ESTIMATOR_LABELS = {
    "ipw": "IPW",
    "aipw": "AIPW",
    "tmle": "TMLE",
    "dml_irm": "DML-IRM",
    "dml_plr": "DML-PLR",
    "dml_apos": "DML-APOS",
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "r_learner": "R-learner",
    "dr_learner": "DR-learner",
    "causal_forest": "Causal forest",
    "tarnet": "TARNet",
    "cfrnet": "CFRNet",
    "dragonnet": "Dragonnet",
}


@dataclass(frozen=True)
class SeriesStyle:
    color: str
    marker: str
    linestyle: str


@dataclass(frozen=True)
class RankingSeries:
    estimator_id: str
    toc_fraction: np.ndarray
    toc_value: np.ndarray
    autoc: float
    autoc_p_value: float
    qini: float
    qini_p_value: float


@dataclass(frozen=True)
class CalibrationSeries:
    estimator_id: str
    predicted: np.ndarray
    observed: np.ndarray
    eceth: float
    eceth_standard_error: float
    eceth_p_value: float


@contextmanager
def publication_style() -> Iterator[None]:
    with plt.rc_context(PUBLICATION_RC):
        yield


def display_estimator(estimator_id: str) -> str:
    if estimator_id in ESTIMATOR_LABELS:
        return ESTIMATOR_LABELS[estimator_id]
    return estimator_id.replace("_", " ")


def display_contrast(contrast: str) -> str:
    return contrast.replace("_", " ")


def series_style(index: int) -> SeriesStyle:
    return SeriesStyle(
        color=SERIES_COLORS[index % len(SERIES_COLORS)],
        marker=MARKERS[index % len(MARKERS)],
        linestyle=LINE_STYLES[index % len(LINE_STYLES)],
    )


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < P_VALUE_FLOOR:
        return "< 0.001"
    return f"{p_value:.3f}"


def format_statistic(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    magnitude = abs(value)
    if magnitude != 0.0 and (
        magnitude < STATISTIC_SCIENTIFIC_FLOOR or magnitude >= STATISTIC_SCIENTIFIC_CEILING
    ):
        return f"{value:.2e}"
    return f"{value:.3f}"


def style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(SPINE_WIDTH)
    axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(
        axis="both",
        which="major",
        width=SPINE_WIDTH,
        length=TICK_LENGTH,
        direction="out",
        colors=INK,
        labelsize=TICK_LABEL_SIZE,
        pad=2,
    )
    axis.xaxis.label.set_size(AXIS_LABEL_SIZE)
    axis.yaxis.label.set_size(AXIS_LABEL_SIZE)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_size(AXIS_LABEL_SIZE)
    axis.title.set_color(INK)
    axis.title.set_fontweight("regular")
    axis.set_axisbelow(True)


def mark_panel(axis: Axes, letter: str) -> None:
    axis.text(
        -0.02,
        1.03,
        letter,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def place_outside_legend(axis: Axes) -> None:
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=LEGEND_FONT_SIZE,
    )


def add_horizontal_reference(axis: Axes, location: float) -> None:
    axis.axhline(
        location,
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=1,
    )


def add_vertical_reference(axis: Axes, location: float, zorder: float) -> None:
    axis.axvline(
        location,
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=zorder,
    )


def plot_interval_estimates(
    estimates: pd.DataFrame,
    save_path: str,
    estimator_column: str,
    contrast_column: str,
    estimate_column: str,
    lower_column: str,
    upper_column: str,
    xlabel: str,
) -> None:
    with publication_style():
        figure = compose_interval_estimates(
            estimates=estimates,
            estimator_column=estimator_column,
            contrast_column=contrast_column,
            estimate_column=estimate_column,
            lower_column=lower_column,
            upper_column=upper_column,
            xlabel=xlabel,
        )
        save_figure(save_path, figure)
        plt.close(figure)


def plot_point_estimates(
    estimates: pd.DataFrame,
    save_path: str,
    estimator_column: str,
    contrast_column: str,
    estimate_column: str,
    xlabel: str,
) -> None:
    with publication_style():
        figure = compose_point_estimates(
            estimates=estimates,
            estimator_column=estimator_column,
            contrast_column=contrast_column,
            estimate_column=estimate_column,
            xlabel=xlabel,
        )
        save_figure(save_path, figure)
        plt.close(figure)


def plot_ranking_curves(series: Sequence[RankingSeries], save_path: str) -> None:
    with publication_style():
        figure = compose_ranking_curves(series)
        save_figure(save_path, figure)
        plt.close(figure)


def plot_calibration_curves(series: Sequence[CalibrationSeries], save_path: str) -> None:
    with publication_style():
        figure = compose_calibration_curves(series)
        save_figure(save_path, figure)
        plt.close(figure)


def compose_interval_estimates(
    estimates: pd.DataFrame,
    estimator_column: str,
    contrast_column: str,
    estimate_column: str,
    lower_column: str,
    upper_column: str,
    xlabel: str,
) -> Figure:
    groups, estimator_order = effect_groups(
        estimates=estimates,
        estimator_column=estimator_column,
        contrast_column=contrast_column,
        estimate_column=estimate_column,
        extra_columns=(lower_column, upper_column),
    )
    figure, axes, bottom = effect_figure(groups=groups, caption_lines=1)
    for panel_index, (contrast, group) in enumerate(groups):
        axis = axes[panel_index]
        draw_interval_panel(
            axis=axis,
            group=group,
            estimator_column=estimator_column,
            estimate_column=estimate_column,
            lower_column=lower_column,
            upper_column=upper_column,
            estimator_order=estimator_order,
            xlabel=xlabel if panel_index == len(groups) - 1 else "",
            contrast=contrast,
        )
        if len(groups) > 1:
            mark_panel(axis, panel_letter(panel_index))
    finish_figure(figure, bottom)
    attach_figure_caption(figure, axes[-1], INTERVAL_CAPTION, bottom)
    return figure


def compose_point_estimates(
    estimates: pd.DataFrame,
    estimator_column: str,
    contrast_column: str,
    estimate_column: str,
    xlabel: str,
) -> Figure:
    groups, estimator_order = effect_groups(
        estimates=estimates,
        estimator_column=estimator_column,
        contrast_column=contrast_column,
        estimate_column=estimate_column,
        extra_columns=(),
    )
    figure, axes, bottom = effect_figure(groups=groups, caption_lines=1)
    for panel_index, (contrast, group) in enumerate(groups):
        axis = axes[panel_index]
        draw_point_panel(
            axis=axis,
            group=group,
            estimator_column=estimator_column,
            estimate_column=estimate_column,
            estimator_order=estimator_order,
            xlabel=xlabel if panel_index == len(groups) - 1 else "",
            contrast=contrast,
        )
        if len(groups) > 1:
            mark_panel(axis, panel_letter(panel_index))
    finish_figure(figure, bottom)
    attach_figure_caption(figure, axes[-1], POINT_CAPTION, bottom)
    return figure


def compose_ranking_curves(series: Sequence[RankingSeries]) -> Figure:
    if len(series) == 0:
        raise ValueError("A ranking figure needs at least one estimator.")
    column_labels, rows, name_colors = ranking_table(series)
    figure, axis, table_axis = annotated_curve_axes(
        width=max(TOC_FIGURE_WIDTH, 1.45 * len(series)),
        panel_height=PANEL_HEIGHT,
        table_rows=len(rows),
        top_legend=len(series) > 1,
    )
    for item in series:
        style = series_style(estimator_index(item.estimator_id, series))
        draw_curve(
            axis=axis,
            x_values=item.toc_fraction,
            y_values=item.toc_value,
            style=style,
            label=display_estimator(item.estimator_id) if len(series) > 1 else "",
            marker="none",
        )
    add_horizontal_reference(axis, 0.0)
    style_axis(axis)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Top fraction ranked by CATE")
    axis.set_ylabel("TOC")
    y_values = np.concatenate([item.toc_value for item in series])
    lower, upper = padded_limits(y_values, include_zero=True)
    axis.set_ylim(lower, upper)
    name_curve(axis=axis, series_count=len(series), title=display_estimator(series[0].estimator_id))
    draw_metric_table(
        axis=table_axis,
        column_labels=column_labels,
        rows=rows,
        name_colors=name_colors,
    )
    return figure


def compose_calibration_curves(series: Sequence[CalibrationSeries]) -> Figure:
    if len(series) == 0:
        raise ValueError("A calibration figure needs at least one estimator.")
    column_labels, rows, name_colors = calibration_table(series)
    figure, axis, table_axis = annotated_curve_axes(
        width=max(CALIBRATION_FIGURE_WIDTH, 1.55 * len(series)),
        panel_height=PANEL_HEIGHT + 0.15,
        table_rows=len(rows),
        top_legend=len(series) > 1,
    )
    coordinates: list[np.ndarray] = []
    for item in series:
        if item.predicted.size == 0 or item.observed.size == 0:
            raise ValueError("Calibration series has no bins.")
        style = series_style(estimator_index(item.estimator_id, series))
        order = np.argsort(item.predicted, kind="mergesort")
        draw_curve(
            axis=axis,
            x_values=item.predicted[order],
            y_values=item.observed[order],
            style=style,
            label=display_estimator(item.estimator_id) if len(series) > 1 else "",
            marker=style.marker,
        )
        coordinates.append(item.predicted)
        coordinates.append(item.observed)
    lower, upper = padded_limits(np.concatenate(coordinates), include_zero=False)
    axis.plot(
        [lower, upper],
        [lower, upper],
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        label=IDEAL_CALIBRATION_LABEL if len(series) > 1 else "",
        zorder=1,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_box_aspect(1)
    axis.set_xlabel("Mean predicted CATE")
    axis.set_ylabel("Mean robust proxy")
    style_axis(axis)
    name_curve(
        axis=axis,
        series_count=len(series),
        title=display_estimator(series[0].estimator_id),
    )
    draw_metric_table(
        axis=table_axis,
        column_labels=column_labels,
        rows=rows,
        name_colors=name_colors,
    )
    return figure


def effect_groups(
    estimates: pd.DataFrame,
    estimator_column: str,
    contrast_column: str,
    estimate_column: str,
    extra_columns: tuple[str, ...],
) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    required = (estimator_column, contrast_column, estimate_column, *extra_columns)
    missing = [column for column in required if column not in estimates.columns]
    if missing:
        raise ValueError(f"Effect table is missing columns: {missing}")
    if estimates.empty:
        raise ValueError("Effect table has no rows.")
    frame = estimates.loc[:, list(required)].copy()
    estimate_values = pd.to_numeric(frame[estimate_column], errors="coerce")
    if estimate_values.isna().any():
        raise ValueError("Effect estimates contain missing values.")
    frame[estimate_column] = estimate_values
    for column in extra_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    estimator_order = list(dict.fromkeys(frame[estimator_column].astype(str)))
    groups: list[tuple[str, pd.DataFrame]] = []
    for contrast, contrast_rows in frame.groupby(contrast_column, sort=False):
        ordered = order_estimators(
            group=contrast_rows,
            estimator_column=estimator_column,
            estimator_order=estimator_order,
        )
        groups.append((str(contrast), ordered))
    return groups, estimator_order


def order_estimators(
    group: pd.DataFrame,
    estimator_column: str,
    estimator_order: list[str],
) -> pd.DataFrame:
    rank = {estimator: index for index, estimator in enumerate(estimator_order)}
    ordered = group.copy()
    ordered["estimator_rank"] = ordered[estimator_column].astype(str).map(rank)
    return (
        ordered.sort_values("estimator_rank", kind="mergesort")
        .drop(columns="estimator_rank")
        .reset_index(drop=True)
    )


def effect_figure(
    groups: list[tuple[str, pd.DataFrame]],
    caption_lines: int,
) -> tuple[Figure, list[Axes], float]:
    heights = [max(PANEL_HEIGHT, FOREST_ROW_INCHES * len(group) + FOREST_PANEL_PAD) for _, group in groups]
    caption_inches = CAPTION_LINE_INCHES * caption_lines + CAPTION_GAP_INCHES
    figure_height = float(sum(heights) + caption_inches)
    bottom = caption_inches / figure_height
    figure, axes = plt.subplots(
        nrows=len(groups),
        ncols=1,
        figsize=(EFFECT_FIGURE_WIDTH, figure_height),
        squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )
    return figure, [axes[index, 0] for index in range(len(groups))], bottom


def draw_interval_panel(
    axis: Axes,
    group: pd.DataFrame,
    estimator_column: str,
    estimate_column: str,
    lower_column: str,
    upper_column: str,
    estimator_order: list[str],
    xlabel: str,
    contrast: str,
) -> None:
    positions = row_positions(len(group))
    extent = [group[estimate_column].to_numpy(dtype=float)]
    indexed = group.reset_index(drop=True)
    for row_number in range(len(indexed)):
        row = indexed.iloc[row_number]
        estimate = float(row[estimate_column])
        lower = float(row[lower_column])
        upper = float(row[upper_column])
        style = series_style(estimator_order.index(str(row[estimator_column])))
        position = float(positions[row_number])
        if np.isfinite(lower) and np.isfinite(upper):
            extent.append(np.array([lower, upper], dtype=float))
            axis.errorbar(
                x=estimate,
                y=position,
                xerr=[[max(estimate - lower, 0.0)], [max(upper - estimate, 0.0)]],
                fmt="o",
                color=style.color,
                ecolor=style.color,
                markersize=MARKER_SIZE,
                elinewidth=ERROR_LINE_WIDTH,
                capsize=ERROR_CAP_SIZE,
                capthick=ERROR_LINE_WIDTH,
                linestyle="none",
                zorder=3,
            )
        else:
            axis.plot(
                estimate,
                position,
                linestyle="none",
                marker="o",
                color=style.color,
                markersize=MARKER_SIZE,
                zorder=3,
            )
    configure_effect_axis(
        axis=axis,
        positions=positions,
        labels=[display_estimator(str(estimator)) for estimator in group[estimator_column]],
        extent=np.concatenate(extent),
        xlabel=xlabel,
        contrast=contrast,
    )


def draw_point_panel(
    axis: Axes,
    group: pd.DataFrame,
    estimator_column: str,
    estimate_column: str,
    estimator_order: list[str],
    xlabel: str,
    contrast: str,
) -> None:
    positions = row_positions(len(group))
    colors = [
        series_style(estimator_order.index(str(estimator))).color
        for estimator in group[estimator_column]
    ]
    values = group[estimate_column].to_numpy(dtype=float)
    axis.barh(
        y=positions,
        width=values,
        height=BAR_HEIGHT,
        color=colors,
        zorder=2,
    )
    configure_effect_axis(
        axis=axis,
        positions=positions,
        labels=[display_estimator(str(estimator)) for estimator in group[estimator_column]],
        extent=values,
        xlabel=xlabel,
        contrast=contrast,
        reference_zorder=4,
    )


def configure_effect_axis(
    axis: Axes,
    positions: np.ndarray,
    labels: list[str],
    extent: np.ndarray,
    xlabel: str,
    contrast: str,
    reference_zorder: float = 1,
) -> None:
    add_vertical_reference(axis, 0.0, reference_zorder)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_ylim(-0.55, len(labels) - 0.45)
    lower, upper = padded_limits(extent, include_zero=True)
    axis.set_xlim(lower, upper)
    axis.set_xlabel(xlabel)
    axis.set_title(display_contrast(contrast), loc="center", pad=6)
    style_axis(axis)


def row_positions(row_count: int) -> np.ndarray:
    return np.arange(row_count, dtype=float)[::-1]


def estimator_index(estimator_id: str, series: Sequence[RankingSeries] | Sequence[CalibrationSeries]) -> int:
    for index, item in enumerate(series):
        if item.estimator_id == estimator_id:
            return index
    raise ValueError(f"Estimator {estimator_id} is not in the figure series.")


def draw_curve(
    axis: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    style: SeriesStyle,
    label: str,
    marker: str,
) -> None:
    axis.plot(
        x_values,
        y_values,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=LINE_WIDTH,
        marker=marker,
        markersize=MARKER_SIZE,
        markerfacecolor=style.color,
        markeredgecolor=style.color,
        markeredgewidth=0.4,
        label=label,
        zorder=2,
    )


def annotated_curve_axes(
    width: float,
    panel_height: float,
    table_rows: int,
    top_legend: bool,
) -> tuple[Figure, Axes, Axes]:
    table_height = TABLE_HEADER_INCHES + TABLE_ROW_INCHES * table_rows
    top_band = LEGEND_BAND_INCHES if top_legend else 0.12
    figure_height = panel_height + table_height + top_band + 0.28
    figure = plt.figure(figsize=(width, figure_height), layout="constrained")
    grid = figure.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[panel_height, table_height],
        hspace=0.08,
    )
    plot_axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])
    table_axis.set_axis_off()
    return figure, plot_axis, table_axis


def name_curve(axis: Axes, series_count: int, title: str) -> None:
    if series_count == 1:
        axis.set_title(title, loc="center", pad=8, fontsize=TITLE_SIZE, color=INK, fontweight="regular")
        return
    add_legend(axis, loc="lower center", anchor=(0.5, 1.02))


def ranking_table(
    series: Sequence[RankingSeries],
) -> tuple[list[str], list[list[str]], list[str]]:
    if len(series) == 1:
        item = series[0]
        return (
            ["", "Estimate", "p"],
            [
                ["AUTOC", format_statistic(item.autoc), format_p_value(item.autoc_p_value)],
                ["Qini", format_statistic(item.qini), format_p_value(item.qini_p_value)],
            ],
            [INK, INK],
        )
    rows: list[list[str]] = []
    colors: list[str] = []
    for index, item in enumerate(series):
        rows.append(
            [
                display_estimator(item.estimator_id),
                format_statistic(item.autoc),
                format_p_value(item.autoc_p_value),
                format_statistic(item.qini),
                format_p_value(item.qini_p_value),
            ]
        )
        colors.append(text_color(series_style(index).color))
    return ["", "AUTOC", "p", "Qini", "p"], rows, colors


def calibration_table(
    series: Sequence[CalibrationSeries],
) -> tuple[list[str], list[list[str]], list[str]]:
    if len(series) == 1:
        item = series[0]
        return (
            ["", "Estimate", "SE", "p"],
            [[
                "ECETH",
                format_statistic(item.eceth),
                format_statistic(item.eceth_standard_error),
                format_p_value(item.eceth_p_value),
            ]],
            [INK],
        )
    rows: list[list[str]] = []
    colors: list[str] = []
    for index, item in enumerate(series):
        rows.append(
            [
                display_estimator(item.estimator_id),
                format_statistic(item.eceth),
                format_statistic(item.eceth_standard_error),
                format_p_value(item.eceth_p_value),
            ]
        )
        colors.append(text_color(series_style(index).color))
    return ["", "ECETH", "SE", "p"], rows, colors


def text_color(color: str) -> str:
    if color.upper() == "#F0E442":
        return INK
    return color


def draw_metric_table(
    axis: Axes,
    column_labels: Sequence[str],
    rows: Sequence[Sequence[str]],
    name_colors: Sequence[str],
) -> None:
    column_count = len(column_labels)
    name_width = 0.36 if column_count <= 4 else 0.28
    value_width = (1.0 - name_width) / (column_count - 1)
    table = axis.table(
        cellText=[list(row) for row in rows],
        colLabels=list(column_labels),
        loc="upper center",
        cellLoc="center",
        colWidths=[name_width, *([value_width] * (column_count - 1))],
        bbox=[0.04, 0.02, 0.92, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(CAPTION_FONT_SIZE)
    last_row = len(rows)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_linewidth(0.6)
        cell.set_edgecolor(INK)
        if row_index == 0:
            cell.visible_edges = "TB"
        elif row_index == last_row:
            cell.visible_edges = "B"
        else:
            cell.visible_edges = ""
        text = cell.get_text()
        text.set_fontsize(CAPTION_FONT_SIZE)
        text.set_color(INK)
        if column_index == 0:
            text.set_ha("left")
        else:
            text.set_ha("center")
        if row_index == 0:
            text.set_fontweight("regular")
        elif column_index == 0:
            text.set_color(name_colors[row_index - 1])


def padded_limits(values: np.ndarray, include_zero: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("The figure has no finite values to plot.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    if span == 0.0:
        span = 1.0
    pad = AXIS_PAD_FRACTION * span
    return lower - pad, upper + pad


def panel_letter(index: int) -> str:
    if index < len(PANEL_LETTERS):
        return PANEL_LETTERS[index]
    return str(index + 1)


def add_legend(axis: Axes, loc: str, anchor: tuple[float, float]) -> None:
    handles, labels = axis.get_legend_handles_labels()
    kept_handles = []
    kept_labels = []
    for handle, label in zip(handles, labels, strict=True):
        if label:
            kept_handles.append(handle)
            kept_labels.append(label)
    if not kept_labels:
        return
    axis.legend(
        kept_handles,
        kept_labels,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        borderaxespad=0.0,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=len(kept_labels),
        columnspacing=1.4,
    )


def finish_figure(figure: Figure, bottom: float) -> None:
    figure.tight_layout(rect=(0.0, bottom, 1.0, 1.0))


def attach_figure_caption(figure: Figure, axis: Axes, caption: str, bottom: float) -> None:
    position = axis.get_position()
    figure.text(
        position.x0,
        max(bottom - 0.015, 0.005),
        caption,
        ha="left",
        va="top",
        fontsize=CAPTION_FONT_SIZE,
        color=INK,
        transform=figure.transFigure,
    )


def attach_panel_captions(
    figure: Figure,
    axes: Sequence[Axes],
    captions: Sequence[str],
    bottom: float,
) -> None:
    caption_y = max(bottom - 0.008, 0.004)
    for axis, caption in zip(axes, captions, strict=True):
        position = axis.get_position()
        figure.text(
            position.x0,
            caption_y,
            caption,
            ha="left",
            va="top",
            fontsize=CAPTION_FONT_SIZE,
            color=INK,
            linespacing=1.35,
            transform=figure.transFigure,
        )

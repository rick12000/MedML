"""Publication figures for causal estimates, diagnostics, and evaluation curves."""

from __future__ import annotations

import logging
import zlib
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
LEGEND_FONT_SIZE = 7.5
P_VALUE_FLOOR = 0.001
STATISTIC_SCIENTIFIC_FLOOR = 0.001
STATISTIC_SCIENTIFIC_CEILING = 1000.0
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
INTERVAL_CAPTION = "Horizontal lines are confidence intervals. The vertical line marks no effect."
POINT_CAPTION = "Bars are means of conditional treatment effects. The vertical line marks no effect."
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
    "axes.grid": False,
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
    axis.grid(False)
    axis.set_axisbelow(True)


def mark_panel(axis: Axes, letter: str, x: float = -0.02) -> None:
    axis.text(
        x,
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


def drop_nonfinite_effect_estimates(
    estimates: pd.DataFrame,
    estimate_column: str,
    *,
    figure_name: str,
) -> pd.DataFrame:
    """Return rows with finite point estimates; log and drop the rest for summary figures."""
    if estimates.empty or estimate_column not in estimates.columns:
        return estimates
    values = pd.to_numeric(estimates[estimate_column], errors="coerce")
    keep = np.isfinite(values.to_numpy(dtype=float, copy=False))
    if keep.all():
        return estimates
    for _, row in estimates.loc[~keep].iterrows():
        logger.warning(
            "Omitting non-finite %s for estimator=%s contrast=%s from %s.",
            estimate_column,
            row.get("estimator", "?"),
            row.get("contrast", "?"),
            figure_name,
        )
    return estimates.loc[keep].reset_index(drop=True)


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


EVALUATION_NAME_WIDTH_INCHES = 1.25
EVALUATION_PANEL_WIDTH_INCHES = 2.15
EVALUATION_PANEL_GAP_INCHES = 0.45
EVALUATION_WIDTH_INCHES = EVALUATION_NAME_WIDTH_INCHES + 2 * EVALUATION_PANEL_WIDTH_INCHES + EVALUATION_PANEL_GAP_INCHES
EVALUATION_PANEL_TOP_INCHES = 0.08
EVALUATION_PANEL_TITLE_INCHES = 0.26
EVALUATION_PLOT_HEIGHT_INCHES = 1.60
EVALUATION_TICK_BAND_INCHES = 0.50
EVALUATION_TABLE_GAP_INCHES = 0.12
EVALUATION_AXIS_PAD_INCHES = 0.50
EVALUATION_AXIS_RIGHT_PAD_INCHES = 0.06
EVALUATION_LETTER_INSET_INCHES = 0.42
EVALUATION_LEFT_PANEL_X_INCHES = EVALUATION_NAME_WIDTH_INCHES
EVALUATION_RIGHT_PANEL_X_INCHES = (
    EVALUATION_LEFT_PANEL_X_INCHES + EVALUATION_PANEL_WIDTH_INCHES + EVALUATION_PANEL_GAP_INCHES
)
EVALUATION_GROUP_TITLE_Y_INCHES = 0.11
EVALUATION_GROUP_RULE_Y_INCHES = 0.22
EVALUATION_SUBHEADER_Y_INCHES = 0.36
EVALUATION_HEADER_RULE_Y_INCHES = 0.48
EVALUATION_TABLE_ROW_INCHES = 0.22
EVALUATION_BOTTOM_PAD_INCHES = 0.08
MUTED_TABLE_INK = "#4A4A4A"
# Okabe-Ito without black and yellow, so every line stays visible on white.
EVALUATION_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
CALIBRATION_GROUP = "Calibration"
RANKING_GROUP = "Ranking"
CALIBRATION_TITLE = "Calibration"
RANKING_TITLE = "Targeting Operator Curve (TOC)"
CALIBRATION_X_LABEL = "Mean Predicted CATE"
CALIBRATION_Y_LABEL = "Mean Robust Proxy"
RANKING_X_LABEL = "Top Fraction Ranked by Benefit"
RANKING_Y_LABEL = "Excess Benefit"
CALIBRATION_COLUMNS = ("ECETH", "p")
RANKING_COLUMNS = ("AUTOC", "p", "Qini", "p")
ESTIMATOR_ORDER = tuple(ESTIMATOR_LABELS)


@dataclass(frozen=True)
class EvaluationRow:
    label: str
    calibration: tuple[str, ...]
    ranking: tuple[str, ...]


def estimator_style(estimator_id: str) -> SeriesStyle:
    """A fixed style per estimator, so one estimator keeps its colour in every figure.

    Known estimators take their position in ``ESTIMATOR_LABELS``. Unknown ones hash to a
    slot after that, so the mapping never depends on the order of the series passed in.
    Lines stay solid so they never read as the dashed reference lines. Colour identifies the
    estimator; the marker changes only once the palette wraps.
    """
    if estimator_id in ESTIMATOR_ORDER:
        index = ESTIMATOR_ORDER.index(estimator_id)
    else:
        index = len(ESTIMATOR_ORDER) + zlib.crc32(estimator_id.encode("utf-8")) % 64
    cycle, slot = divmod(index, len(EVALUATION_COLORS))
    return SeriesStyle(
        color=EVALUATION_COLORS[slot],
        marker=MARKERS[cycle % len(MARKERS)],
        linestyle="solid",
    )


def plot_evaluation_figure(
    calibration: Sequence[CalibrationSeries],
    ranking: Sequence[RankingSeries],
    save_path: str,
) -> None:
    with publication_style():
        figure = compose_evaluation_figure(calibration=calibration, ranking=ranking)
        save_figure(save_path, figure)
        plt.close(figure)


def compose_evaluation_figure(
    calibration: Sequence[CalibrationSeries],
    ranking: Sequence[RankingSeries],
) -> Figure:
    """Calibration and targeting curves side by side, with one table that serves both panels."""
    if not calibration or not ranking:
        raise ValueError("An evaluation figure needs calibration and ranking series.")
    rows = evaluation_rows(calibration=calibration, ranking=ranking)
    plot_top = EVALUATION_PANEL_TOP_INCHES + EVALUATION_PANEL_TITLE_INCHES
    panel_bottom = plot_top + EVALUATION_PLOT_HEIGHT_INCHES + EVALUATION_TICK_BAND_INCHES
    table_top = panel_bottom + EVALUATION_TABLE_GAP_INCHES
    table_height = evaluation_table_height(len(rows))
    height = table_top + table_height + EVALUATION_BOTTOM_PAD_INCHES
    figure = plt.figure(figsize=(EVALUATION_WIDTH_INCHES, height))

    plot_width = EVALUATION_PANEL_WIDTH_INCHES - EVALUATION_AXIS_PAD_INCHES - EVALUATION_AXIS_RIGHT_PAD_INCHES
    calibration_axis = add_inch_axes(
        figure,
        EVALUATION_LEFT_PANEL_X_INCHES + EVALUATION_AXIS_PAD_INCHES,
        plot_top,
        plot_width,
        EVALUATION_PLOT_HEIGHT_INCHES,
    )
    ranking_axis = add_inch_axes(
        figure,
        EVALUATION_RIGHT_PANEL_X_INCHES + EVALUATION_AXIS_PAD_INCHES,
        plot_top,
        plot_width,
        EVALUATION_PLOT_HEIGHT_INCHES,
    )
    draw_calibration_axis(calibration_axis, calibration)
    draw_ranking_axis(ranking_axis, ranking)
    letter_x = -EVALUATION_LETTER_INSET_INCHES / plot_width
    mark_panel(calibration_axis, PANEL_LETTERS[0], x=letter_x)
    mark_panel(ranking_axis, PANEL_LETTERS[1], x=letter_x)

    table_axis = add_inch_axes(figure, 0.0, table_top, EVALUATION_WIDTH_INCHES, table_height)
    draw_evaluation_table(table_axis, rows)
    return figure


def inch_position(figure: Figure, x: float, y_from_top: float) -> tuple[float, float]:
    width, height = figure.get_size_inches()
    return x / width, 1.0 - y_from_top / height


def add_inch_axes(figure: Figure, left: float, top: float, width: float, height: float) -> Axes:
    figure_width, figure_height = figure.get_size_inches()
    return figure.add_axes(
        (left / figure_width, 1.0 - (top + height) / figure_height, width / figure_width, height / figure_height)
    )


def evaluation_table_height(row_count: int) -> float:
    return EVALUATION_HEADER_RULE_Y_INCHES + EVALUATION_TABLE_ROW_INCHES * row_count + 0.02


def evaluation_rows(
    calibration: Sequence[CalibrationSeries],
    ranking: Sequence[RankingSeries],
) -> list[EvaluationRow]:
    by_calibration = {item.estimator_id: item for item in calibration}
    by_ranking = {item.estimator_id: item for item in ranking}
    estimator_ids = list(dict.fromkeys([*by_ranking, *by_calibration]))
    rows = []
    for estimator_id in estimator_ids:
        item_calibration = by_calibration.get(estimator_id)
        item_ranking = by_ranking.get(estimator_id)
        rows.append(
            EvaluationRow(
                label=display_estimator(estimator_id),
                calibration=(
                    (
                        format_statistic(item_calibration.eceth),
                        format_p_value(item_calibration.eceth_p_value),
                    )
                    if item_calibration is not None
                    else ("NA",) * len(CALIBRATION_COLUMNS)
                ),
                ranking=(
                    (
                        format_statistic(item_ranking.autoc),
                        format_p_value(item_ranking.autoc_p_value),
                        format_statistic(item_ranking.qini),
                        format_p_value(item_ranking.qini_p_value),
                    )
                    if item_ranking is not None
                    else ("NA",) * len(RANKING_COLUMNS)
                ),
            )
        )
    return rows


def draw_calibration_axis(axis: Axes, series: Sequence[CalibrationSeries]) -> None:
    coordinates: list[np.ndarray] = []
    for item in series:
        if item.predicted.size == 0 or item.observed.size == 0:
            raise ValueError("Calibration series has no bins.")
        style = estimator_style(item.estimator_id)
        order = np.argsort(item.predicted, kind="mergesort")
        draw_curve(
            axis=axis,
            x_values=item.predicted[order],
            y_values=item.observed[order],
            style=style,
            label="",
            marker=style.marker,
        )
        coordinates.extend([item.predicted, item.observed])
    lower, upper = padded_limits(np.concatenate(coordinates), include_zero=False)
    axis.plot(
        [lower, upper],
        [lower, upper],
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=1,
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel(CALIBRATION_X_LABEL)
    axis.set_ylabel(CALIBRATION_Y_LABEL)
    axis.set_title(CALIBRATION_TITLE, loc="left", pad=6, fontsize=AXIS_LABEL_SIZE, color=INK)
    style_axis(axis)


def draw_ranking_axis(axis: Axes, series: Sequence[RankingSeries]) -> None:
    values: list[np.ndarray] = []
    for item in series:
        draw_curve(
            axis=axis,
            x_values=item.toc_fraction,
            y_values=item.toc_value,
            style=estimator_style(item.estimator_id),
            label="",
            marker="none",
        )
        values.append(item.toc_value)
    add_horizontal_reference(axis, 0.0)
    lower, upper = padded_limits(np.concatenate(values), include_zero=True)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(lower, upper)
    ticks = np.linspace(0.0, 1.0, 6)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{round(tick * 100)}%" for tick in ticks])
    axis.set_xlabel(RANKING_X_LABEL)
    axis.set_ylabel(RANKING_Y_LABEL)
    axis.set_title(RANKING_TITLE, loc="left", pad=6, fontsize=AXIS_LABEL_SIZE, color=INK)
    style_axis(axis)


def column_centers(left: float, width: float, count: int) -> list[float]:
    return [left + width * (index + 0.5) / count for index in range(count)]


def draw_evaluation_table(axis: Axes, rows: Sequence[EvaluationRow]) -> None:
    """One table under both panels. Each column group sits under its own panel."""
    axis.set_xlim(0.0, EVALUATION_WIDTH_INCHES)
    axis.set_ylim(evaluation_table_height(len(rows)), 0.0)
    axis.axis("off")
    calibration_centers = column_centers(
        EVALUATION_LEFT_PANEL_X_INCHES, EVALUATION_PANEL_WIDTH_INCHES, len(CALIBRATION_COLUMNS)
    )
    ranking_centers = column_centers(
        EVALUATION_RIGHT_PANEL_X_INCHES, EVALUATION_PANEL_WIDTH_INCHES, len(RANKING_COLUMNS)
    )
    for left, text in (
        (EVALUATION_LEFT_PANEL_X_INCHES, CALIBRATION_GROUP),
        (EVALUATION_RIGHT_PANEL_X_INCHES, RANKING_GROUP),
    ):
        center = left + EVALUATION_PANEL_WIDTH_INCHES / 2
        axis.text(center, EVALUATION_GROUP_TITLE_Y_INCHES, text, ha="center", va="center",
                  fontsize=CAPTION_FONT_SIZE, fontweight="bold", color=MUTED_TABLE_INK)
        axis.plot([left, left + EVALUATION_PANEL_WIDTH_INCHES], [EVALUATION_GROUP_RULE_Y_INCHES] * 2,
                  color=INK, linewidth=0.5, solid_capstyle="butt")
    axis.text(0.0, EVALUATION_SUBHEADER_Y_INCHES, "Estimator", ha="left", va="center",
              fontsize=CAPTION_FONT_SIZE, fontweight="bold", color=MUTED_TABLE_INK)
    for center, header in zip(calibration_centers, CALIBRATION_COLUMNS, strict=True):
        axis.text(center, EVALUATION_SUBHEADER_Y_INCHES, header, ha="center", va="center",
                  fontsize=CAPTION_FONT_SIZE, fontweight="bold", color=MUTED_TABLE_INK)
    for center, header in zip(ranking_centers, RANKING_COLUMNS, strict=True):
        axis.text(center, EVALUATION_SUBHEADER_Y_INCHES, header, ha="center", va="center",
                  fontsize=CAPTION_FONT_SIZE, fontweight="bold", color=MUTED_TABLE_INK)
    rule_width = EVALUATION_WIDTH_INCHES
    axis.plot([0.0, rule_width], [EVALUATION_HEADER_RULE_Y_INCHES] * 2, color=INK, linewidth=0.7,
              solid_capstyle="butt")
    for index, row in enumerate(rows):
        middle = EVALUATION_HEADER_RULE_Y_INCHES + EVALUATION_TABLE_ROW_INCHES * (index + 0.5)
        axis.text(0.0, middle, row.label, ha="left", va="center", fontsize=CAPTION_FONT_SIZE, color=INK)
        for center, value in zip(calibration_centers, row.calibration, strict=True):
            axis.text(center, middle, value, ha="center", va="center", fontsize=CAPTION_FONT_SIZE, color=INK)
        for center, value in zip(ranking_centers, row.ranking, strict=True):
            axis.text(center, middle, value, ha="center", va="center", fontsize=CAPTION_FONT_SIZE, color=INK)
    bottom = evaluation_table_height(len(rows)) - 0.01
    axis.plot([0.0, rule_width], [bottom, bottom], color=INK, linewidth=0.7, solid_capstyle="butt")


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

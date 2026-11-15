"""Love plot of covariate balance before and after inverse-probability weighting.

Each covariate is one row. Hollow circles are the unadjusted standardized mean difference
and filled circles are the weighted one, joined by a thin connector. Rows are ranked by
the unadjusted magnitude, largest first, and capped at ``max_covariates``. A pale band
and dotted lines mark the conventional ±0.1 balance threshold, and the dashed line marks
perfect balance.

Multi-arm analyses get one panel per arm contrast. The legend sits centred beneath the
chart in a single row of two columns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from causal_pipeline.figures import (
    DIAGNOSTIC_LABEL_SIZE,
    DIAGNOSTIC_LEGEND_SIZE,
    DIAGNOSTIC_TITLE_SIZE,
    EFFECT_FIGURE_WIDTH,
    INK,
    REFERENCE_COLOR,
    REFERENCE_LINEWIDTH,
    configure_effect_axis,
    diagnostic_color,
    display_contrast,
    mark_panel,
    panel_letter,
    place_bottom_strip_legend,
    publication_style,
    row_positions,
    style_diagnostic_axis,
)
from causal_pipeline.utils import save_figure

COVARIATE_BALANCE_FIGURE_NAME = "covariate_balance.png"
DEFAULT_MAX_COVARIATES = 10
REQUIRED_COLUMNS = ("covariate", "unweighted_smd", "weighted_smd")
UNADJUSTED_LABEL = "Unadjusted"
WEIGHTED_LABEL = "Weighted"
UNADJUSTED_COLOR = diagnostic_color(1)
WEIGHTED_COLOR = diagnostic_color(0)
BALANCE_BAND_COLOR = "#F1F3F5"
CONNECTOR_COLOR = "#C5CED6"
CONNECTOR_WIDTH = 1.0
UNADJUSTED_EDGE_WIDTH = 1.2
WEIGHTED_EDGE_WIDTH = 0.6
LOVE_MARKER_AREA = 38.0
THRESHOLD = 0.1
THRESHOLD_LINESTYLE = (0, (1.1, 1.6))
LOVE_ROW_INCHES = 0.30
LOVE_PANEL_PAD_INCHES = 0.95
LOVE_MIN_PANEL_INCHES = 2.15
LEGEND_HEIGHT_INCHES = 0.42
LABEL_FONT_SIZE = DIAGNOSTIC_LABEL_SIZE
LEGEND_FONT_SIZE = DIAGNOSTIC_LEGEND_SIZE
X_LABEL = "Standardized Mean Difference"
BALANCE_TITLE = "Covariate balance"


def plot_covariate_balance(
    balance: pd.DataFrame,
    save_path: str,
    max_covariates: int = DEFAULT_MAX_COVARIATES,
) -> None:
    with publication_style():
        figure = compose_covariate_balance(balance, max_covariates=max_covariates)
        save_figure(save_path, figure)
        plt.close(figure)


def compose_covariate_balance(
    balance: pd.DataFrame,
    max_covariates: int = DEFAULT_MAX_COVARIATES,
) -> Figure:
    if max_covariates < 1:
        raise ValueError("max_covariates must be at least 1.")
    panels = balance_panels(balance, max_covariates=max_covariates)
    heights = [
        max(LOVE_MIN_PANEL_INCHES, LOVE_ROW_INCHES * len(frame) + LOVE_PANEL_PAD_INCHES)
        for _, frame in panels
    ]
    figure_height = float(sum(heights) + LEGEND_HEIGHT_INCHES)
    figure, grid = plt.subplots(
        nrows=len(panels),
        ncols=1,
        figsize=(EFFECT_FIGURE_WIDTH, figure_height),
        squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )
    axes = [grid[index, 0] for index in range(len(panels))]
    for index, ((contrast, frame), axis) in enumerate(zip(panels, axes, strict=True)):
        is_last = index == len(panels) - 1
        draw_balance_panel(axis=axis, frame=frame, xlabel=X_LABEL if is_last else "", contrast=contrast)
        if len(panels) > 1:
            mark_panel(axis, panel_letter(index), x=-0.06, y=1.06, fontsize=DIAGNOSTIC_TITLE_SIZE)
    figure.tight_layout(rect=(0.0, LEGEND_HEIGHT_INCHES / figure_height, 1.0, 1.0))
    place_legend_below(figure, axes[-1], axes[0])
    return figure


def balance_panels(balance: pd.DataFrame, max_covariates: int) -> list[tuple[str, pd.DataFrame]]:
    """One panel for a binary contrast, one per arm pair for a multi-arm analysis."""
    missing = [column for column in REQUIRED_COLUMNS if column not in balance.columns]
    if missing:
        raise ValueError(f"Covariate balance table is missing columns: {missing}")
    if balance.empty:
        raise ValueError("Covariate balance table has no rows.")
    if "arm_a" not in balance.columns:
        return [("", top_covariates(balance, max_covariates))]
    return [
        (f"{arm_a}_vs_{arm_b}", top_covariates(frame, max_covariates))
        for (arm_a, arm_b), frame in balance.groupby(["arm_a", "arm_b"], sort=False)
    ]


def top_covariates(frame: pd.DataFrame, max_covariates: int) -> pd.DataFrame:
    """Covariates ranked by unadjusted |SMD|, largest first, keeping the first ``max_covariates``."""
    magnitude = np.abs(frame["unweighted_smd"].to_numpy(dtype=float))
    order = np.argsort(-magnitude, kind="mergesort")
    return frame.iloc[order[:max_covariates]]


def draw_balance_panel(axis: Axes, frame: pd.DataFrame, xlabel: str, contrast: str) -> None:
    positions = row_positions(len(frame))
    unadjusted = frame["unweighted_smd"].to_numpy(dtype=float)
    weighted = frame["weighted_smd"].to_numpy(dtype=float)
    axis.axvspan(-THRESHOLD, THRESHOLD, color=BALANCE_BAND_COLOR, linewidth=0, zorder=0)
    axis.hlines(
        positions,
        unadjusted,
        weighted,
        color=CONNECTOR_COLOR,
        linewidth=CONNECTOR_WIDTH,
        zorder=2,
    )
    for threshold in (-THRESHOLD, THRESHOLD):
        axis.axvline(
            threshold,
            color=REFERENCE_COLOR,
            linestyle=THRESHOLD_LINESTYLE,
            linewidth=REFERENCE_LINEWIDTH,
            zorder=1,
        )
    axis.scatter(
        unadjusted,
        positions,
        s=LOVE_MARKER_AREA,
        facecolors="white",
        edgecolors=UNADJUSTED_COLOR,
        linewidths=UNADJUSTED_EDGE_WIDTH,
        zorder=3,
        label=UNADJUSTED_LABEL,
    )
    axis.scatter(
        weighted,
        positions,
        s=LOVE_MARKER_AREA,
        color=WEIGHTED_COLOR,
        edgecolors="white",
        linewidths=WEIGHTED_EDGE_WIDTH,
        zorder=4,
        label=WEIGHTED_LABEL,
    )
    configure_effect_axis(
        axis=axis,
        positions=positions,
        labels=frame["covariate"].astype(str).tolist(),
        extent=np.concatenate([unadjusted, weighted, [-THRESHOLD, THRESHOLD]]),
        xlabel=xlabel,
        contrast=contrast,
        reference_zorder=1,
    )
    style_diagnostic_axis(axis)
    title = BALANCE_TITLE if contrast == "" else display_contrast(contrast)
    axis.set_title(
        title,
        loc="center",
        pad=8,
        fontsize=DIAGNOSTIC_TITLE_SIZE,
        fontweight="bold",
        color=INK,
    )
    axis.xaxis.label.set_size(LABEL_FONT_SIZE)


def place_legend_below(figure: Figure, chart_axis: Axes, handle_axis: Axes) -> None:
    """Two-column legend centred under the chart, in the strip reserved beneath the x-axis label."""
    place_bottom_strip_legend(
        figure=figure,
        anchor=chart_axis,
        handles_axis=handle_axis,
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
    )

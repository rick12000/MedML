"""Love plot of covariate balance before and after inverse-probability weighting.

Each covariate is one row. Hollow circles are the unadjusted standardized mean difference
and filled circles are the weighted one, joined by a thin connector. Rows are ranked by
the unadjusted magnitude, largest first, and capped at ``max_covariates``. Dotted lines
mark the conventional ±0.1 balance threshold, and the dashed line marks perfect balance.

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
    EFFECT_FIGURE_WIDTH,
    MARKER_SIZE,
    REFERENCE_COLOR,
    REFERENCE_LINEWIDTH,
    SERIES_COLORS,
    configure_effect_axis,
    mark_panel,
    panel_letter,
    publication_style,
    row_positions,
)
from causal_pipeline.utils import save_figure

COVARIATE_BALANCE_FIGURE_NAME = "covariate_balance.png"
DEFAULT_MAX_COVARIATES = 10
REQUIRED_COLUMNS = ("covariate", "unweighted_smd", "weighted_smd")
UNADJUSTED_LABEL = "Unadjusted"
WEIGHTED_LABEL = "Weighted"
# Okabe-Ito vermilion for the unadjusted state and blue for the weighted state, matching the series palette.
UNADJUSTED_COLOR = SERIES_COLORS[1]
WEIGHTED_COLOR = SERIES_COLORS[0]
CONNECTOR_COLOR = "#BDBDBD"
CONNECTOR_WIDTH = 0.8
UNADJUSTED_EDGE_WIDTH = 0.9
WEIGHTED_EDGE_WIDTH = 0.4
THRESHOLD = 0.1
THRESHOLD_LINESTYLE = (0, (1.0, 1.5))
LOVE_ROW_INCHES = 0.20
LOVE_PANEL_PAD_INCHES = 0.75
LOVE_MIN_PANEL_INCHES = 1.5
LEGEND_HEIGHT_INCHES = 0.32
LABEL_FONT_SIZE = 9.5
TICK_FONT_SIZE = 8.5
LEGEND_FONT_SIZE = 9.0
X_LABEL = "Standardized Mean Difference"


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
            mark_panel(axis, panel_letter(index))
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
    axis.hlines(positions, unadjusted, weighted, color=CONNECTOR_COLOR, linewidth=CONNECTOR_WIDTH, zorder=2)
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
        s=MARKER_SIZE**2,
        facecolors="white",
        edgecolors=UNADJUSTED_COLOR,
        linewidths=UNADJUSTED_EDGE_WIDTH,
        zorder=3,
        label=UNADJUSTED_LABEL,
    )
    axis.scatter(
        weighted,
        positions,
        s=MARKER_SIZE**2,
        color=WEIGHTED_COLOR,
        edgecolors=WEIGHTED_COLOR,
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
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.xaxis.label.set_size(LABEL_FONT_SIZE)
    axis.set_title(axis.get_title(), fontsize=LABEL_FONT_SIZE)


def place_legend_below(figure: Figure, chart_axis: Axes, handle_axis: Axes) -> None:
    """Two-column legend centred under the chart, in the strip reserved beneath the x-axis label."""
    handles, labels = handle_axis.get_legend_handles_labels()
    position = chart_axis.get_position()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=((position.x0 + position.x1) / 2, 0.0),
        bbox_transform=figure.transFigure,
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=2.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

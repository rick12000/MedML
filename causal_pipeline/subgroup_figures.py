"""Forest-style table for pre-specified subgroup split results.

The figure is drawn on one canvas measured in inches, so text, intervals, and
p-values share one row grid. Every data row carries exactly one interval per
forest column, centred on that row. Only a bottom X axis is drawn.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from causal_pipeline.figures import INK, REFERENCE_COLOR, display_contrast, display_estimator, publication_style
from causal_pipeline.subgroup import ABOVE_CUTOFF_RULE
from causal_pipeline.utils import save_figure

SUBGROUP_SPLITS_FIGURE_NAME = "subgroup_splits.png"
MUTED_INK = "#4A4A4A"
RULE_COLOR = "#D0D0D0"
ROW_INCHES = 0.30
HEADER_INCHES = 0.34
TOP_PAD_INCHES = 0.12
TITLE_INCHES = 0.04
AXIS_INCHES = 0.55
MARGIN_INCHES = 0.14
TEXT_FONT_SIZE = 7.6
HEADER_FONT_SIZE = 7.8
COLUMN_HEADER_SIZE = 7.4
FOREST_WIDTH_INCHES = 1.75
FOREST_GAP_INCHES = 0.14
SUBGROUP_WIDTH_INCHES = 2.55
PATIENTS_WIDTH_INCHES = 0.95
ESTIMATE_WIDTH_INCHES = 1.65
P_WIDTH_INCHES = 0.62
MARKER_SIZE = 4.2
TICK_FONT = 6.8
INTERVAL_WIDTH = 1.0
TICK_TARGET_COUNT = 5
LIMIT_HALF_WIDTH_FACTOR = 1.25
CAP_CLIP_FACTOR = 2.5
ARROW_INCHES = 0.07
P_FLOOR = 0.01


def plot_subgroup_splits_table(
    table: pd.DataFrame,
    split_order: list[tuple[str, float]],
    save_path: str,
) -> None:
    if table.empty:
        return
    ci_percent = int(round(float(table["confidence_level"].iloc[0]) * 100))
    items = build_items(table=table, split_order=split_order)
    if not items:
        return
    with publication_style():
        figure = compose_subgroup_table(items=items, ci_percent=ci_percent)
        save_figure(save_path, figure)
        plt.close(figure)


def build_items(table: pd.DataFrame, split_order: list[tuple[str, float]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    multi_contrast = table["contrast"].nunique() > 1
    for estimator in dict.fromkeys(table["estimator"].astype(str)):
        estimator_rows = table.loc[table["estimator"].astype(str) == estimator]
        items.append({"kind": "header", "label": display_estimator(estimator)})
        for column, cutoff in split_order:
            matches = estimator_rows.loc[
                (estimator_rows["column"] == column) & (estimator_rows["cutoff"] == cutoff)
            ]
            for _, record in matches.iterrows():
                items.append(
                    {
                        "kind": "data",
                        "label": split_label(
                            column=column,
                            cutoff=cutoff,
                            above_rule=str(record["above_rule"]),
                            contrast=str(record["contrast"]),
                            multi_contrast=multi_contrast,
                        ),
                        "n_above": int(round(float(record["n_above"]))),
                        "n_below": int(round(float(record["n_below"]))),
                        "effect": interval_values(record, "effect"),
                        "outcome": interval_values(record, "outcome"),
                        "effect_p": float(record["effect_pvalue"]),
                        "outcome_p": float(record["outcome_pvalue"]),
                    }
                )
    return items


def interval_values(record: pd.Series, prefix: str) -> tuple[float, float, float]:
    return (
        float(record[f"{prefix}_difference"]),
        float(record[f"{prefix}_ci_lower"]),
        float(record[f"{prefix}_ci_upper"]),
    )


def split_label(column: str, cutoff: float, above_rule: str, contrast: str, multi_contrast: bool) -> str:
    cutoff_text = format(cutoff, ".6g")
    comparison = "≥" if above_rule == ABOVE_CUTOFF_RULE else above_rule
    base = f"{column} {comparison} {cutoff_text}"
    if multi_contrast:
        base = f"{base} · {display_contrast(contrast)}"
    return base


def compose_subgroup_table(items: list[dict[str, object]], ci_percent: int) -> Figure:
    columns = column_layout()
    body_height = sum(HEADER_INCHES if item["kind"] == "header" else ROW_INCHES for item in items)
    height = MARGIN_INCHES + TITLE_INCHES + HEADER_INCHES + TOP_PAD_INCHES + body_height + AXIS_INCHES
    figure = plt.figure(figsize=(columns["total"], height))
    canvas = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    canvas.set_xlim(0.0, columns["total"])
    canvas.set_ylim(height, 0.0)
    canvas.axis("off")

    header_top = MARGIN_INCHES + TITLE_INCHES
    write_column_headers(canvas, columns, header_top, ci_percent)

    data_items = [item for item in items if item["kind"] == "data"]
    effect_scale = forest_scale([item["effect"] for item in data_items])
    outcome_scale = forest_scale([item["outcome"] for item in data_items])

    top = header_top + HEADER_INCHES + TOP_PAD_INCHES
    row_top = top
    for item in items:
        if item["kind"] == "header":
            canvas.text(
                columns["subgroup"][0],
                row_top + HEADER_INCHES / 2,
                str(item["label"]),
                fontsize=HEADER_FONT_SIZE,
                fontweight="bold",
                color=INK,
                va="center",
                ha="left",
            )
            if row_top > top:
                canvas.plot(
                    [0.0, columns["total"]],
                    [row_top, row_top],
                    color=RULE_COLOR,
                    linewidth=0.6,
                )
            row_top += HEADER_INCHES
            continue
        middle = row_top + ROW_INCHES / 2
        write_data_row(canvas, columns, item, middle)
        draw_interval(canvas, columns["effect_forest"], effect_scale, middle, item["effect"])
        draw_interval(canvas, columns["outcome_forest"], outcome_scale, middle, item["outcome"])
        row_top += ROW_INCHES

    axis_y = row_top + 0.12
    for key, scale, label in (
        ("effect_forest", effect_scale, "ATE difference (above − below)"),
        ("outcome_forest", outcome_scale, "Outcome difference (above − below)"),
    ):
        span = columns[key]
        zero_x = to_canvas_x(span, scale, 0.0)
        canvas.plot(
            [zero_x, zero_x],
            [top, axis_y],
            color=REFERENCE_COLOR,
            linestyle=(0, (2, 2)),
            linewidth=0.5,
            zorder=1,
        )
        draw_bottom_axis(canvas, span, scale, axis_y, label)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def column_layout() -> dict[str, object]:
    widths = [
        ("subgroup", SUBGROUP_WIDTH_INCHES),
        ("patients", PATIENTS_WIDTH_INCHES),
        ("effect_text", ESTIMATE_WIDTH_INCHES),
        ("effect_forest", FOREST_WIDTH_INCHES),
        ("effect_p", P_WIDTH_INCHES),
        ("outcome_text", ESTIMATE_WIDTH_INCHES),
        ("outcome_forest", FOREST_WIDTH_INCHES),
        ("outcome_p", P_WIDTH_INCHES),
    ]
    layout: dict[str, object] = {}
    cursor = MARGIN_INCHES
    for name, width in widths:
        layout[name] = (cursor, cursor + width)
        cursor += width
    layout["total"] = cursor + MARGIN_INCHES
    return layout


def write_column_headers(canvas, columns: dict[str, object], top: float, ci_percent: int) -> None:
    headers = {
        "subgroup": ("Subgroup", "left"),
        "patients": ("Patients (≥ / <)", "center"),
        "effect_text": (f"ATE diff ({ci_percent}% CI)", "center"),
        "effect_p": ("P", "center"),
        "outcome_text": (f"Outcome diff ({ci_percent}% CI)", "center"),
        "outcome_p": ("P", "center"),
    }
    for key, (text, align) in headers.items():
        span = columns[key]
        x = span[0] if align == "left" else (span[0] + span[1]) / 2
        canvas.text(
            x,
            top + HEADER_INCHES / 2,
            text,
            fontsize=COLUMN_HEADER_SIZE,
            color=MUTED_INK,
            ha=align,
            va="center",
            fontweight="bold",
        )
    canvas.plot(
        [columns["subgroup"][0], columns["outcome_p"][1]],
        [top + HEADER_INCHES, top + HEADER_INCHES],
        color=INK,
        linewidth=0.7,
    )


def write_data_row(canvas, columns: dict[str, object], item: dict[str, object], middle: float) -> None:
    def text(key: str, value: str, align: str = "center") -> None:
        span = columns[key]
        x = span[0] if align == "left" else (span[0] + span[1]) / 2
        canvas.text(
            x,
            middle,
            value,
            fontsize=TEXT_FONT_SIZE,
            color=INK,
            ha=align,
            va="center",
        )

    text("subgroup", str(item["label"]), align="left")
    text("patients", f"{item['n_above']} / {item['n_below']}")
    text("effect_text", format_estimate(item["effect"]))
    text("effect_p", format_p(float(item["effect_p"])))
    text("outcome_text", format_estimate(item["outcome"]))
    text("outcome_p", format_p(float(item["outcome_p"])))


def forest_scale(intervals: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Symmetric window around zero. Endpoints far beyond it are drawn as arrows, not stretched."""
    estimates = np.array([abs(item[0]) for item in intervals if np.isfinite(item[0])], dtype=float)
    half_widths = np.array(
        [abs(item[2] - item[1]) / 2 for item in intervals if np.isfinite(item[1]) and np.isfinite(item[2])],
        dtype=float,
    )
    anchor = float(estimates.max()) if estimates.size else 1.0
    spread = float(np.median(half_widths)) if half_widths.size else 0.0
    limit = max(anchor * LIMIT_HALF_WIDTH_FACTOR, CAP_CLIP_FACTOR * spread, 1e-6)
    return -limit, limit


def to_canvas_x(columns_span: tuple[float, float], scale: tuple[float, float], value: float) -> float:
    low, high = scale
    left, right = columns_span
    fraction = (value - low) / (high - low)
    return left + float(np.clip(fraction, 0.0, 1.0)) * (right - left)


def draw_interval(
    canvas,
    columns_span: tuple[float, float],
    scale: tuple[float, float],
    middle: float,
    interval: tuple[float, float, float],
) -> None:
    estimate, lower, upper = interval
    if not np.isfinite(estimate):
        return
    low, high = scale
    left = low if not np.isfinite(lower) else max(lower, low)
    right = high if not np.isfinite(upper) else min(upper, high)
    x_left = to_canvas_x(columns_span, scale, left)
    x_right = to_canvas_x(columns_span, scale, right)
    canvas.plot([x_left, x_right], [middle, middle], color=INK, linewidth=INTERVAL_WIDTH, solid_capstyle="butt", zorder=3)
    if np.isfinite(lower) and lower < low:
        canvas.annotate(
            "",
            xy=(x_left, middle),
            xytext=(x_left + ARROW_INCHES * 2, middle),
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.8, "mutation_scale": 6},
            zorder=4,
        )
    if np.isfinite(upper) and upper > high:
        canvas.annotate(
            "",
            xy=(x_right, middle),
            xytext=(x_right - ARROW_INCHES * 2, middle),
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.8, "mutation_scale": 6},
            zorder=4,
        )
    canvas.plot(
        to_canvas_x(columns_span, scale, estimate),
        middle,
        marker="o",
        markersize=MARKER_SIZE,
        color=INK,
        zorder=5,
    )


def draw_bottom_axis(
    canvas,
    columns_span: tuple[float, float],
    scale: tuple[float, float],
    y: float,
    label: str,
) -> None:
    left, right = columns_span
    canvas.plot([left, right], [y, y], color=INK, linewidth=0.6, zorder=2)
    for tick in nice_ticks(scale):
        x = to_canvas_x(columns_span, scale, tick)
        canvas.plot([x, x], [y, y + 0.05], color=INK, linewidth=0.6)
        canvas.text(x, y + 0.08, format_tick(tick), fontsize=TICK_FONT, color=INK, ha="center", va="top")
    canvas.text(
        (left + right) / 2,
        y + 0.30,
        label,
        fontsize=TICK_FONT,
        color=MUTED_INK,
        ha="center",
        va="top",
    )


def nice_ticks(scale: tuple[float, float]) -> list[float]:
    low, high = scale
    span = high - low
    raw_step = span / max(TICK_TARGET_COUNT - 1, 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if step >= raw_step:
            break
    start = np.ceil(low / step) * step
    ticks = np.arange(start, high + step * 0.5, step)
    return [float(round(tick, 10)) for tick in ticks if low <= tick <= high]


def format_tick(value: float) -> str:
    return f"{value:g}"


def format_estimate(interval: tuple[float, float, float]) -> str:
    estimate, lower, upper = interval
    return f"{estimate:.2f} ({lower:.2f}, {upper:.2f})"


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < P_FLOOR:
        return "<0.01"
    return f"{value:.2f}"

"""Table figure for held-out policy values.

Rows are the learned policies. Columns are the reference rules each policy is compared with:
every arm treated for everyone, treating no one, and the observed treatments. A cell is the
mean held-out benefit of the row policy minus the column rule, with its paired bootstrap
interval. Shaded cells are those whose interval excludes zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from causal_pipeline.figures import (
    INK,
    display_estimator,
    draw_line,
    publication_style,
    text_width,
    widest,
    write_text,
)
from causal_pipeline.policy import BOOTSTRAP_CI_LOWER_PERCENTILE, BOOTSTRAP_CI_UPPER_PERCENTILE
from causal_pipeline.policy_value import ALWAYS_TREAT, NEVER_TREAT, OBSERVED
from causal_pipeline.utils import save_figure

POLICY_VALUE_FIGURE_NAME = "value_comparison.png"
COMPARISON_PREFIX = "vs_"
CI_SUFFIXES = ("_ci_lower", "_ci_upper")
BASELINE_METHODS = frozenset({NEVER_TREAT, ALWAYS_TREAT, OBSERVED})
METHOD_LABELS = {
    "policy_tree": "Policy tree",
    "exact_policy_welfare": "Exact welfare tree",
    "exact_policy_capital": "Exact CAPITAL tree",
    "dr_policy_tree": "DR policy tree",
    "virtual_twins": "Virtual twins",
    "mob": "MOB",
}
ALWAYS_GROUP_TITLE = "Versus treat everyone with"
NEVER_GROUP_TITLE = "Versus treat no one"
OBSERVED_GROUP_TITLE = "Versus observed"
NEVER_LABEL = "Treat no one"
OBSERVED_LABEL = "Observed"
POLICY_HEADER = "Policy"
TABLE_TITLE_TEMPLATE = "Held-out policy value (n = {sample_size})"
FOOTNOTE_TEMPLATE = (
    "Each cell is the mean held-out benefit of the row policy minus the column rule. "
    "Positive values favour the row policy.",
    "Brackets: {confidence:g}% paired bootstrap interval over held-out patients. "
    "Shaded: the interval excludes zero (blue favours the row policy, rose favours the column rule).",
    "Treat no one is the zero-benefit reference. Observed is the treatment each patient received.",
)

POSITIVE_FILL = "#E3EEF8"
NEGATIVE_FILL = "#F6E1DD"
MUTED_INK = "#4A4A4A"
RULE_COLOR = "#D0D0D0"

MARGIN_INCHES = 0.10
TITLE_INCHES = 0.30
PARENT_INCHES = 0.24
SUB_INCHES = 0.24
HEADER_GAP_INCHES = 0.04
BODY_TOP_PAD_INCHES = 0.04
ROW_INCHES = 0.46
CELL_PAD_INCHES = 0.24
COLUMN_GROUP_GAP_INCHES = 0.16
LABEL_GAP_INCHES = 0.18
FOOTNOTE_LINE_INCHES = 0.17
FOOTNOTE_GAP_INCHES = 0.12
LINE_OFFSET_INCHES = 0.10
CELL_INSET_INCHES = 0.03

TITLE_FONT_SIZE = 9.0
PARENT_FONT_SIZE = 7.8
HEADER_FONT_SIZE = 7.4
TEXT_FONT_SIZE = 7.8
DETAIL_FONT_SIZE = 6.6
FOOTNOTE_FONT_SIZE = 6.6


@dataclass(frozen=True)
class ValueCell:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ValueRow:
    label: str
    detail: str
    cells: tuple[ValueCell, ...]


@dataclass(frozen=True)
class ValueColumn:
    group: str
    label: str


@dataclass(frozen=True)
class ValueTable:
    title: str
    footnotes: tuple[str, ...]
    columns: tuple[ValueColumn, ...]
    rows: tuple[ValueRow, ...]
    decimals: int


@dataclass(frozen=True)
class ValueLayout:
    width: float
    label_left: float
    columns: tuple[tuple[float, float], ...]
    groups: tuple[tuple[str, float, float], ...]


def plot_policy_value_table(table: pd.DataFrame, save_path: str) -> None:
    """Write the policy value table as a PNG. Nothing is written without learned policies."""
    model = build_value_table(table)
    if model is None:
        return
    with publication_style():
        figure = compose_value_table(model)
        save_figure(save_path, figure)
        plt.close(figure)


def build_value_table(table: pd.DataFrame) -> ValueTable | None:
    learned = table.loc[~table["method"].isin(BASELINE_METHODS)]
    if learned.empty:
        return None
    comparisons = sorted(
        (
            column
            for column in table.columns
            if column.startswith(COMPARISON_PREFIX) and not column.endswith(CI_SUFFIXES)
        ),
        key=comparison_rank,
    )
    columns = tuple(comparison_column(column[len(COMPARISON_PREFIX) :]) for column in comparisons)
    rows = tuple(build_row(record, comparisons) for _, record in learned.iterrows())
    estimates = np.array([cell.estimate for row in rows for cell in row.cells], dtype=float)
    decimals = 3 if float(np.max(np.abs(estimates))) < 1.0 else 2
    confidence = BOOTSTRAP_CI_UPPER_PERCENTILE - BOOTSTRAP_CI_LOWER_PERCENTILE
    return ValueTable(
        title=TABLE_TITLE_TEMPLATE.format(sample_size=int(table["n_test"].iloc[0])),
        footnotes=tuple(line.format(confidence=confidence) for line in FOOTNOTE_TEMPLATE),
        columns=columns,
        rows=rows,
        decimals=decimals,
    )


def comparison_rank(column: str) -> int:
    """Always-treat columns first, then treat no one, then observed. Arms keep table order."""
    name = column[len(COMPARISON_PREFIX) :]
    if name.startswith(f"{ALWAYS_TREAT}_"):
        return 0
    if name == NEVER_TREAT:
        return 1
    return 2


def comparison_column(name: str) -> ValueColumn:
    """Map a comparator name from the value table to its column heading."""
    if name.startswith(f"{ALWAYS_TREAT}_"):
        arm = name[len(ALWAYS_TREAT) + 1 :]
        return ValueColumn(group=ALWAYS_GROUP_TITLE, label=display_text(arm))
    if name == NEVER_TREAT:
        return ValueColumn(group=NEVER_GROUP_TITLE, label=NEVER_LABEL)
    if name == OBSERVED:
        return ValueColumn(group=OBSERVED_GROUP_TITLE, label=OBSERVED_LABEL)
    raise ValueError(f"Unknown comparison {name} in the policy value table.")


def build_row(record: pd.Series, comparisons: Sequence[str]) -> ValueRow:
    cells = tuple(
        ValueCell(
            estimate=float(record[column]),
            lower=float(record[f"{column}_ci_lower"]),
            upper=float(record[f"{column}_ci_upper"]),
        )
        for column in comparisons
    )
    method = str(record["method"])
    return ValueRow(
        label=METHOD_LABELS.get(method, display_text(method)),
        detail=estimator_label(record["source_cate_model"]),
        cells=cells,
    )


def estimator_label(source: object) -> str:
    """Estimator ids are ``<kind>__<overlap>__<digest>`` unless aliased, so the kind leads."""
    if not isinstance(source, str) or not source:
        return ""
    return display_estimator(source.split("__")[0])


def display_text(text: str) -> str:
    return text.replace("_", " ")


def format_estimate(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def format_bounds(cell: ValueCell, decimals: int) -> str:
    return f"[{cell.lower:.{decimals}f}, {cell.upper:.{decimals}f}]"


def cell_fill(cell: ValueCell) -> str | None:
    if cell.lower > 0.0:
        return POSITIVE_FILL
    if cell.upper < 0.0:
        return NEGATIVE_FILL
    return None


def build_layout(model: ValueTable) -> ValueLayout:
    label_width = max(
        text_width(POLICY_HEADER, HEADER_FONT_SIZE, "bold"),
        widest((row.label for row in model.rows), TEXT_FONT_SIZE, "bold"),
        widest((row.detail for row in model.rows), DETAIL_FONT_SIZE),
    )
    label_left = MARGIN_INCHES
    label_right = label_left + label_width + LABEL_GAP_INCHES
    groups = grouped_column_indices(model.columns)
    widths: dict[int, float] = {}
    for group_title, indices in groups:
        column_widths = [column_width(model, index) for index in indices]
        group_needed = text_width(group_title, PARENT_FONT_SIZE, "bold") + CELL_PAD_INCHES
        column_widths[-1] += max(0.0, group_needed - sum(column_widths))
        widths.update(zip(indices, column_widths, strict=True))
    table_width = label_right + sum(widths.values()) + COLUMN_GROUP_GAP_INCHES * (len(groups) - 1)
    footnote_width = widest(model.footnotes, FOOTNOTE_FONT_SIZE) + 2 * MARGIN_INCHES
    stretch = max(0.0, footnote_width - (table_width + MARGIN_INCHES)) / len(model.columns)

    cursor = label_right
    column_spans: list[tuple[float, float]] = []
    group_spans: list[tuple[str, float, float]] = []
    for group_title, indices in groups:
        group_left = cursor
        for index in indices:
            column_spans.append((cursor, cursor + widths[index] + stretch))
            cursor += widths[index] + stretch
        group_spans.append((group_title, group_left, cursor))
        cursor += COLUMN_GROUP_GAP_INCHES
    width = cursor - COLUMN_GROUP_GAP_INCHES + MARGIN_INCHES
    return ValueLayout(
        width=max(width, footnote_width),
        label_left=label_left,
        columns=tuple(column_spans),
        groups=tuple(group_spans),
    )


def grouped_column_indices(columns: Sequence[ValueColumn]) -> list[tuple[str, list[int]]]:
    """Consecutive columns that share a group title, in table order."""
    groups: list[tuple[str, list[int]]] = []
    for index, column in enumerate(columns):
        if groups and groups[-1][0] == column.group:
            groups[-1][1].append(index)
        else:
            groups.append((column.group, [index]))
    return groups


def column_width(model: ValueTable, index: int) -> float:
    cells = [row.cells[index] for row in model.rows]
    return CELL_PAD_INCHES + max(
        [text_width(model.columns[index].label, HEADER_FONT_SIZE, "bold")]
        + [text_width(format_estimate(cell.estimate, model.decimals), TEXT_FONT_SIZE) for cell in cells]
        + [text_width(format_bounds(cell, model.decimals), DETAIL_FONT_SIZE) for cell in cells]
    )


def compose_value_table(model: ValueTable) -> Figure:
    layout = build_layout(model)
    header_top = MARGIN_INCHES + TITLE_INCHES + HEADER_GAP_INCHES
    body_top = header_top + PARENT_INCHES + SUB_INCHES + BODY_TOP_PAD_INCHES
    body_bottom = body_top + ROW_INCHES * len(model.rows)
    footnote_top = body_bottom + FOOTNOTE_GAP_INCHES
    height = footnote_top + FOOTNOTE_LINE_INCHES * len(model.footnotes) + MARGIN_INCHES

    figure = plt.figure(figsize=(layout.width, height))
    canvas = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    canvas.set_xlim(0.0, layout.width)
    canvas.set_ylim(height, 0.0)
    canvas.axis("off")

    draw_title(canvas, model, layout)
    draw_header(canvas, model, layout, header_top, body_top)
    draw_body(canvas, model, layout, body_top)
    draw_line(canvas, layout.label_left, layout.width - MARGIN_INCHES, body_bottom, INK, 0.7)
    for line_index, footnote in enumerate(model.footnotes):
        write_text(
            canvas,
            layout.label_left,
            footnote_top + FOOTNOTE_LINE_INCHES * (line_index + 0.5),
            footnote,
            ha="left",
            size=FOOTNOTE_FONT_SIZE,
            color=MUTED_INK,
        )
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def draw_title(canvas, model: ValueTable, layout: ValueLayout) -> None:
    write_text(
        canvas,
        layout.label_left,
        MARGIN_INCHES + TITLE_INCHES / 2,
        model.title,
        ha="left",
        size=TITLE_FONT_SIZE,
        color=INK,
        weight="bold",
    )


def draw_header(
    canvas,
    model: ValueTable,
    layout: ValueLayout,
    header_top: float,
    body_top: float,
) -> None:
    parent_middle = header_top + PARENT_INCHES / 2
    sub_middle = header_top + PARENT_INCHES + SUB_INCHES / 2
    draw_line(canvas, layout.label_left, layout.width - MARGIN_INCHES, header_top, INK, 0.9)
    write_text(
        canvas,
        layout.label_left,
        sub_middle,
        POLICY_HEADER,
        ha="left",
        size=HEADER_FONT_SIZE,
        color=MUTED_INK,
        weight="bold",
    )
    for title, left, right in layout.groups:
        write_text(
            canvas,
            (left + right) / 2,
            parent_middle,
            title,
            size=PARENT_FONT_SIZE,
            color=INK,
            weight="bold",
        )
        draw_line(canvas, left, right, header_top + PARENT_INCHES, INK, 0.5)
    for column, (left, right) in zip(model.columns, layout.columns, strict=True):
        write_text(
            canvas,
            (left + right) / 2,
            sub_middle,
            column.label,
            size=HEADER_FONT_SIZE,
            color=MUTED_INK,
            weight="bold",
        )
    draw_line(canvas, layout.label_left, layout.width - MARGIN_INCHES, body_top - BODY_TOP_PAD_INCHES, INK, 0.7)


def draw_body(canvas, model: ValueTable, layout: ValueLayout, body_top: float) -> None:
    for row_index, row in enumerate(model.rows):
        top = body_top + ROW_INCHES * row_index
        middle = top + ROW_INCHES / 2
        for (left, right), cell in zip(layout.columns, row.cells, strict=True):
            fill = cell_fill(cell)
            if fill is not None:
                canvas.add_patch(
                    Rectangle(
                        (left + CELL_INSET_INCHES, top + CELL_INSET_INCHES),
                        right - left - 2 * CELL_INSET_INCHES,
                        ROW_INCHES - 2 * CELL_INSET_INCHES,
                        facecolor=fill,
                        edgecolor="none",
                        zorder=0,
                    )
                )
            center = (left + right) / 2
            write_text(
                canvas,
                center,
                middle - LINE_OFFSET_INCHES,
                format_estimate(cell.estimate, model.decimals),
                size=TEXT_FONT_SIZE,
                color=INK,
            )
            write_text(
                canvas,
                center,
                middle + LINE_OFFSET_INCHES,
                format_bounds(cell, model.decimals),
                size=DETAIL_FONT_SIZE,
                color=MUTED_INK,
            )
        write_text(
            canvas,
            layout.label_left,
            middle - LINE_OFFSET_INCHES,
            row.label,
            ha="left",
            size=TEXT_FONT_SIZE,
            color=INK,
            weight="bold",
        )
        write_text(
            canvas,
            layout.label_left,
            middle + LINE_OFFSET_INCHES,
            row.detail,
            ha="left",
            size=DETAIL_FONT_SIZE,
            color=MUTED_INK,
        )
        if row_index < len(model.rows) - 1:
            draw_line(canvas, layout.label_left, layout.width - MARGIN_INCHES, top + ROW_INCHES, RULE_COLOR, 0.5)

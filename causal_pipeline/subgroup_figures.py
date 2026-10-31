"""Forest-style table for pre-specified subgroup split results.

Two images are written per subgroup table.

The estimate figure has one row per subgroup split. When several treatment contrasts
were estimated, rows are grouped by contrast. The overall row (``All``) leads each group
and is highlighted. Each estimator gets an ATE block.

The difference figure has one row per subgroup split, grouped by contrast. It has two
unadjusted blocks. The split difference is the treated-minus-untreated outcome gap on
each side of the cutoff, differenced between the two sides. The group difference is the
mean outcome above the cutoff minus the mean outcome below it.

Columns are a shared descriptive part followed by blocks. The descriptive part is the
variable (split column), the splits (cutoff sides, or ``1 / 0`` for binary variables),
and the patient counts. Each block
holds a delta column, a confidence-interval column, a p column, and a forest column.

The figure is drawn on one canvas measured in inches. Column widths come from the text
each column holds, so spacing stays tight whatever the labels are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from causal_pipeline.figures import (
    INK,
    REFERENCE_COLOR,
    display_contrast,
    display_estimator,
    publication_style,
)
from causal_pipeline.subgroup import ABOVE_CUTOFF_RULE, OVERALL_LABEL
from causal_pipeline.utils import save_figure

SUBGROUP_SPLITS_FIGURE_NAME = "subgroup_splits.png"
SUBGROUP_DIFFERENCES_FIGURE_NAME = "subgroup_differences.png"

DELTA = "Δ"
ATE_KIND = "ate"
SPLIT_KIND = "split"
GROUP_KIND = "group"
SPLIT_BLOCK_KEY = "split_difference"
GROUP_BLOCK_KEY = "group_difference"
SPLIT_DIFFERENCE_TITLE = "Split Difference"
GROUP_DIFFERENCE_TITLE = "Group Difference"
AXIS_LABELS = {
    ATE_KIND: f"ATE {DELTA}",
    SPLIT_KIND: f"Split {DELTA}",
    GROUP_KIND: f"Group {DELTA}",
}
GROUP_HEADER = "Variable"
SPLITS_HEADER = "Splits"
PATIENTS_HEADER = "Patients"
P_HEADER = "p"
BINARY_SPLIT_TEXT = "1 / 0"

# Deep teal for the overall row: a calm accent that stays legible in print.
OVERALL_COLOR = "#1B5E7A"
OVERALL_BAND = "#E9F2F6"
MUTED_INK = "#4A4A4A"
RULE_COLOR = "#D0D0D0"

MARGIN_INCHES = 0.08
HEADER_PARENT_INCHES = 0.24
HEADER_SUB_INCHES = 0.22
TOP_PAD_INCHES = 0.06
GROUP_TITLE_INCHES = 0.26
ROW_INCHES = 0.27
AXIS_INCHES = 0.50
AXIS_TICKS_INCHES = 0.30

COLUMN_GAP_INCHES = 0.14
INNER_GAP_INCHES = 0.14
BLOCK_GAP_INCHES = 0.22
TEXT_GAP_INCHES = 0.10
FOREST_WIDTH_INCHES = 1.30

TEXT_FONT_SIZE = 7.6
PARENT_FONT_SIZE = 7.8
HEADER_FONT_SIZE = 7.4
TICK_FONT = 6.8
MARKER_SIZE = 4.2
INTERVAL_WIDTH = 1.0
ARROW_INCHES = 0.07
TICK_LENGTH_INCHES = 0.05
TICK_LABEL_OFFSET_INCHES = 0.08
AXIS_LABEL_OFFSET_INCHES = 0.30
TICK_TARGET_COUNT = 5
LIMIT_HALF_WIDTH_FACTOR = 1.25
CAP_CLIP_FACTOR = 2.5
P_FLOOR = 0.01


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    pvalue: float


@dataclass(frozen=True)
class Block:
    """One column block. ``kind`` picks the forest axis label and the shared forest scale."""

    key: str
    title: str
    kind: str


@dataclass(frozen=True)
class SubgroupRow:
    group: str
    splits: str
    patients: str
    is_overall: bool
    intervals: Mapping[str, Interval]


@dataclass(frozen=True)
class ContrastGroup:
    title: str
    rows: tuple[SubgroupRow, ...]


@dataclass(frozen=True)
class TableModel:
    blocks: tuple[Block, ...]
    groups: tuple[ContrastGroup, ...]
    ci_percent: int


@dataclass(frozen=True)
class EstimateCell:
    estimate: str
    pvalue: str
    bounds: str


@dataclass(frozen=True)
class Span:
    left: float
    right: float

    @property
    def center(self) -> float:
        return (self.left + self.right) / 2

    @property
    def width(self) -> float:
        return self.right - self.left


@dataclass(frozen=True)
class BlockColumns:
    block: Block
    delta: Span
    bounds: Span
    pvalue: Span
    forest: Span

    @property
    def span(self) -> Span:
        return Span(self.delta.left, self.forest.right)


@dataclass(frozen=True)
class TableLayout:
    left: float
    right: float
    group: Span
    splits: Span
    patients: Span
    blocks: tuple[BlockColumns, ...]

    @property
    def width(self) -> float:
        return self.right + MARGIN_INCHES


@dataclass(frozen=True)
class VerticalLayout:
    header_top: float
    body_top: float
    axis_y: float
    height: float


def plot_subgroup_splits_table(
    table: pd.DataFrame,
    split_order: Sequence[tuple[str, float]],
    save_path: str,
) -> None:
    """Write the estimate figure: one ATE block per estimator."""
    if table.empty:
        return
    model = build_table_model(table=table, split_order=split_order)
    if not any(group.rows for group in model.groups):
        return
    _save_model(model, save_path)


def plot_subgroup_differences_table(
    table: pd.DataFrame,
    split_order: Sequence[tuple[str, float]],
    save_path: str,
) -> None:
    """Write the difference figure: split difference first, then group difference."""
    if table.empty:
        return
    model = build_difference_model(table=table, split_order=split_order)
    if not any(group.rows for group in model.groups):
        return
    _save_model(model, save_path)


def write_subgroup_figures(
    table: pd.DataFrame,
    split_order: Sequence[tuple[str, float]],
    output_dir: str | Path,
) -> None:
    """Write both subgroup images into ``output_dir``."""
    output = Path(output_dir)
    plot_subgroup_splits_table(table, split_order, str(output / SUBGROUP_SPLITS_FIGURE_NAME))
    plot_subgroup_differences_table(table, split_order, str(output / SUBGROUP_DIFFERENCES_FIGURE_NAME))


def _save_model(model: TableModel, save_path: str) -> None:
    with publication_style():
        figure = compose_subgroup_table(model)
        save_figure(save_path, figure)
        plt.close(figure)


# Data model


def estimate_blocks(table: pd.DataFrame) -> tuple[Block, ...]:
    """One ATE block per estimator in ``table``.

    Safe to call on a table that stacks several scenarios, because it never groups rows.
    """
    estimators = unique_strings(table["estimator"])
    return tuple(
        Block(key=estimator, title=display_estimator(estimator), kind=ATE_KIND) for estimator in estimators
    )


def difference_blocks() -> tuple[Block, ...]:
    """Split difference first, then group difference. Both are estimator-free."""
    return (
        Block(key=SPLIT_BLOCK_KEY, title=SPLIT_DIFFERENCE_TITLE, kind=SPLIT_KIND),
        Block(key=GROUP_BLOCK_KEY, title=GROUP_DIFFERENCE_TITLE, kind=GROUP_KIND),
    )


def build_table_model(table: pd.DataFrame, split_order: Sequence[tuple[str, float]]) -> TableModel:
    return build_model(
        table=table,
        blocks=estimate_blocks(table),
        split_order=split_order,
        make_group=contrast_group,
    )


def build_difference_model(table: pd.DataFrame, split_order: Sequence[tuple[str, float]]) -> TableModel:
    return build_model(
        table=table,
        blocks=difference_blocks(),
        split_order=split_order,
        make_group=difference_group,
    )


def build_model(
    table: pd.DataFrame,
    blocks: tuple[Block, ...],
    split_order: Sequence[tuple[str, float]],
    make_group: Callable[[pd.DataFrame, Sequence[tuple[str, float]], str], ContrastGroup],
) -> TableModel:
    contrasts = unique_strings(table["contrast"])
    multi_contrast = len(contrasts) > 1
    groups = tuple(
        make_group(
            table.loc[table["contrast"].astype(str) == contrast],
            split_order,
            display_contrast(contrast) if multi_contrast else "",
        )
        for contrast in contrasts
    )
    ci_percent = round(float(table["confidence_level"].iloc[0]) * 100)
    return TableModel(blocks=blocks, groups=groups, ci_percent=ci_percent)


def contrast_group(frame: pd.DataFrame, split_order: Sequence[tuple[str, float]], title: str) -> ContrastGroup:
    split_rows = [
        row
        for column, cutoff in split_order
        if (row := split_row(frame, column, cutoff)) is not None
    ]
    overall = overall_row(frame, overall_patients(frame, split_order))
    rows = [overall, *split_rows] if overall is not None else split_rows
    return ContrastGroup(title=title, rows=tuple(rows))


def difference_group(frame: pd.DataFrame, split_order: Sequence[tuple[str, float]], title: str) -> ContrastGroup:
    """Split rows only. The difference blocks have no overall row."""
    rows = [
        row
        for column, cutoff in split_order
        if (row := difference_row(frame, column, cutoff)) is not None
    ]
    return ContrastGroup(title=title, rows=tuple(rows))


def matching_split_records(frame: pd.DataFrame, column: str, cutoff: float) -> pd.DataFrame:
    return frame.loc[(frame["column"] == column) & (frame["cutoff"] == cutoff)]


def overall_patients(frame: pd.DataFrame, split_order: Sequence[tuple[str, float]]) -> int | None:
    """Every split partitions the same scored sample, so any split gives the total."""
    for column, cutoff in split_order:
        matches = matching_split_records(frame, column, cutoff)
        if not matches.empty:
            first = matches.iloc[0]
            return int(first["n_above"]) + int(first["n_below"])
    return None


def overall_row(frame: pd.DataFrame, patients: int | None) -> SubgroupRow | None:
    overall = frame.loc[frame["cutoff"].isna()]
    if overall.empty:
        return None
    records = one_record_per_estimator(overall)
    return SubgroupRow(
        group=OVERALL_LABEL,
        splits="",
        patients="" if patients is None else str(patients),
        is_overall=True,
        intervals=present({key: interval_from(record, "effect") for key, record in records.items()}),
    )


def split_row(frame: pd.DataFrame, column: str, cutoff: float) -> SubgroupRow | None:
    matches = matching_split_records(frame, column, cutoff)
    if matches.empty:
        return None
    records = one_record_per_estimator(matches)
    first = next(iter(records.values()))
    return SubgroupRow(
        group=column,
        splits=split_text(
            cutoff=cutoff,
            above_rule=str(first["above_rule"]),
            is_binary=as_bool(first["split_is_binary"]),
        ),
        patients=f"{int(first['n_above'])} / {int(first['n_below'])}",
        is_overall=False,
        intervals=present({key: interval_from(record, "effect") for key, record in records.items()}),
    )


def difference_row(frame: pd.DataFrame, column: str, cutoff: float) -> SubgroupRow | None:
    """Split and group differences do not depend on the estimator, so one shared row per split."""
    matches = matching_split_records(frame, column, cutoff)
    if matches.empty:
        return None
    records = list(one_record_per_estimator(matches).values())
    check_shared_differences(records)
    first = records[0]
    return SubgroupRow(
        group=column,
        splits=split_text(
            cutoff=cutoff,
            above_rule=str(first["above_rule"]),
            is_binary=as_bool(first["split_is_binary"]),
        ),
        patients=f"{int(first['n_above'])} / {int(first['n_below'])}",
        is_overall=False,
        intervals=present(
            {
                SPLIT_BLOCK_KEY: interval_from(first, "treatment"),
                GROUP_BLOCK_KEY: interval_from(first, "outcome"),
            }
        ),
    )


def one_record_per_estimator(frame: pd.DataFrame) -> dict[str, pd.Series]:
    records: dict[str, pd.Series] = {}
    for _, record in frame.iterrows():
        estimator = str(record["estimator"])
        if estimator in records:
            raise ValueError(f"Estimator {estimator} has more than one row for one subgroup split.")
        records[estimator] = record
    return records


DIFFERENCE_CHECK_COLUMNS = (
    "n_above",
    "n_below",
    "treatment_difference",
    "treatment_ci_lower",
    "treatment_ci_upper",
    "treatment_pvalue",
    "outcome_difference",
    "outcome_ci_lower",
    "outcome_ci_upper",
    "outcome_pvalue",
)


def check_shared_differences(records: Sequence[pd.Series]) -> None:
    """Differences and counts do not depend on the estimator, so they must agree across it."""
    reference = np.array([float(records[0][column]) for column in DIFFERENCE_CHECK_COLUMNS])
    for record in records[1:]:
        values = np.array([float(record[column]) for column in DIFFERENCE_CHECK_COLUMNS])
        if not np.allclose(reference, values, equal_nan=True):
            raise ValueError("Split and group differences and counts must match across estimators for each split.")


def interval_from(record: pd.Series, prefix: str) -> Interval | None:
    estimate = float(record[f"{prefix}_difference"])
    if not np.isfinite(estimate):
        return None
    return Interval(
        estimate=estimate,
        lower=float(record[f"{prefix}_ci_lower"]),
        upper=float(record[f"{prefix}_ci_upper"]),
        pvalue=float(record[f"{prefix}_pvalue"]),
    )


def present(intervals: Mapping[str, Interval | None]) -> dict[str, Interval]:
    return {key: interval for key, interval in intervals.items() if interval is not None}


def as_bool(value: object) -> bool:
    """Flags come back as strings when the table has been round-tripped through CSV."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def split_text(cutoff: float, above_rule: str, is_binary: bool) -> str:
    """Binary splits are always ``1 / 0``. Continuous splits show both sides of the cutoff."""
    if is_binary:
        return BINARY_SPLIT_TEXT
    if above_rule != ABOVE_CUTOFF_RULE:
        raise ValueError(f"Unsupported above rule {above_rule}.")
    cut = format(cutoff, ".6g")
    return f"≥ {cut} / < {cut}"


def unique_strings(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(values.astype(str)))


# Formatting


def format_number(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2f}"


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < P_FLOOR:
        return f"<{P_FLOOR:g}"
    return f"{value:.2f}"


def estimate_cell(interval: Interval) -> EstimateCell:
    return EstimateCell(
        estimate=format_number(interval.estimate),
        pvalue=format_p(interval.pvalue),
        bounds=f"[{format_number(interval.lower)}, {format_number(interval.upper)}]",
    )


def format_tick(value: float) -> str:
    return f"{value:g}"


# Layout


def text_width(text: str, fontsize: float, fontweight: str = "normal") -> float:
    """Rendered width in inches under the active rc settings."""
    scratch = Figure()
    FigureCanvasAgg(scratch)
    artist = scratch.text(0.0, 0.0, text, fontsize=fontsize, fontweight=fontweight)
    extent = artist.get_window_extent(renderer=scratch.canvas.get_renderer())
    return float(extent.width / scratch.dpi)


def widest(texts: Iterable[str], fontsize: float, fontweight: str = "normal") -> float:
    return max((text_width(text, fontsize, fontweight) for text in texts), default=0.0)


def cell_for_row(row: SubgroupRow, key: str) -> EstimateCell | None:
    interval = row.intervals.get(key)
    return estimate_cell(interval) if interval is not None else None


def build_layout(model: TableModel) -> TableLayout:
    rows = [row for group in model.groups for row in group.rows]
    group_width = max(
        text_width(GROUP_HEADER, HEADER_FONT_SIZE, "bold"),
        widest((row.group for row in rows if not row.is_overall), TEXT_FONT_SIZE),
        widest((row.group for row in rows if row.is_overall), TEXT_FONT_SIZE, "bold"),
    )
    splits_width = max(
        text_width(SPLITS_HEADER, HEADER_FONT_SIZE, "bold"),
        widest((row.splits for row in rows), TEXT_FONT_SIZE),
    )
    patients_width = max(
        text_width(PATIENTS_HEADER, HEADER_FONT_SIZE, "bold"),
        widest((row.patients for row in rows), TEXT_FONT_SIZE),
    )
    cursor = MARGIN_INCHES
    group = Span(cursor, cursor + group_width)
    cursor = group.right + COLUMN_GAP_INCHES
    splits = Span(cursor, cursor + splits_width)
    cursor = splits.right + COLUMN_GAP_INCHES
    patients = Span(cursor, cursor + patients_width)
    cursor = patients.right + BLOCK_GAP_INCHES

    ci_header = f"{model.ci_percent}% CI"
    block_columns = []
    for block in model.blocks:
        cells = [cell for row in rows if (cell := cell_for_row(row, block.key)) is not None]
        delta_width = max(
            [text_width(AXIS_LABELS[block.kind], HEADER_FONT_SIZE, "bold")]
            + [text_width(cell.estimate, TEXT_FONT_SIZE) for cell in cells]
        )
        bounds_width = max(
            [text_width(ci_header, HEADER_FONT_SIZE, "bold")]
            + [text_width(cell.bounds, TEXT_FONT_SIZE) for cell in cells]
        )
        pvalue_width = max(
            [text_width(P_HEADER, HEADER_FONT_SIZE, "bold")]
            + [text_width(cell.pvalue, TEXT_FONT_SIZE, "bold") for cell in cells]
        )
        delta = Span(cursor, cursor + delta_width)
        cursor = delta.right + INNER_GAP_INCHES
        bounds = Span(cursor, cursor + bounds_width)
        cursor = bounds.right + INNER_GAP_INCHES
        pvalue = Span(cursor, cursor + pvalue_width)
        cursor = pvalue.right + TEXT_GAP_INCHES
        # The estimator title is centred over the whole block, so the forest widens to fit it.
        title_width = text_width(block.title, PARENT_FONT_SIZE, "bold")
        filled = pvalue.right - delta.left + TEXT_GAP_INCHES
        forest_width = max(FOREST_WIDTH_INCHES, title_width - filled)
        forest = Span(cursor, cursor + forest_width)
        cursor = forest.right + BLOCK_GAP_INCHES
        block_columns.append(
            BlockColumns(block=block, delta=delta, bounds=bounds, pvalue=pvalue, forest=forest)
        )
    return TableLayout(
        left=MARGIN_INCHES,
        right=cursor - BLOCK_GAP_INCHES,
        group=group,
        splits=splits,
        patients=patients,
        blocks=tuple(block_columns),
    )


def build_vertical_layout(model: TableModel) -> VerticalLayout:
    header_top = MARGIN_INCHES
    body_top = header_top + HEADER_PARENT_INCHES + HEADER_SUB_INCHES + TOP_PAD_INCHES
    body_height = sum(group_height(group) for group in model.groups)
    axis_y = body_top + body_height
    height = axis_y + axis_block_inches(with_label=True) + MARGIN_INCHES
    return VerticalLayout(header_top=header_top, body_top=body_top, axis_y=axis_y, height=height)


def axis_block_inches(*, with_label: bool) -> float:
    """Vertical space taken by one forest axis: ticks only, or ticks plus the axis title."""
    return AXIS_INCHES if with_label else AXIS_TICKS_INCHES


def group_height(group: ContrastGroup) -> float:
    title_height = GROUP_TITLE_INCHES if group.title else 0.0
    return title_height + ROW_INCHES * len(group.rows)


def forest_scales(model: TableModel) -> dict[str, tuple[float, float]]:
    """One symmetric scale per block kind, so every estimator's forest shares an axis."""
    by_kind: dict[str, list[Interval]] = {block.kind: [] for block in model.blocks}
    for block in model.blocks:
        for group in model.groups:
            for row in group.rows:
                interval = row.intervals.get(block.key)
                if interval is not None:
                    by_kind[block.kind].append(interval)
    return {kind: forest_scale(intervals) for kind, intervals in by_kind.items()}


def forest_scale(intervals: Sequence[Interval]) -> tuple[float, float]:
    """Symmetric window around zero. Endpoints far beyond it are drawn as arrows, not stretched."""
    anchor = max((abs(interval.estimate) for interval in intervals), default=1.0)
    half_widths = [
        abs(interval.upper - interval.lower) / 2
        for interval in intervals
        if np.isfinite(interval.lower) and np.isfinite(interval.upper)
    ]
    spread = float(np.median(half_widths)) if half_widths else 0.0
    limit = max(anchor * LIMIT_HALF_WIDTH_FACTOR, CAP_CLIP_FACTOR * spread, 1e-6)
    return -limit, limit


def to_canvas_x(span: Span, scale: tuple[float, float], value: float) -> float:
    low, high = scale
    fraction = (value - low) / (high - low)
    return span.left + float(np.clip(fraction, 0.0, 1.0)) * span.width


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


# Drawing


def compose_subgroup_table(model: TableModel) -> Figure:
    layout = build_layout(model)
    vertical = build_vertical_layout(model)
    figure = plt.figure(figsize=(layout.width, vertical.height))
    canvas = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    canvas.set_xlim(0.0, layout.width)
    canvas.set_ylim(vertical.height, 0.0)
    canvas.axis("off")
    scales = forest_scales(model)
    draw_header(canvas, model, layout, vertical)
    draw_body(canvas, model, layout, vertical, scales)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def draw_header(canvas, model: TableModel, layout: TableLayout, vertical: VerticalLayout) -> None:
    parent_middle = vertical.header_top + HEADER_PARENT_INCHES / 2
    sub_middle = vertical.header_top + HEADER_PARENT_INCHES + HEADER_SUB_INCHES / 2
    ci_header = f"{model.ci_percent}% CI"
    draw_rule(canvas, layout, vertical.header_top, INK, 0.9)
    write_text(canvas, layout.group.left, sub_middle, GROUP_HEADER, ha="left",
               size=HEADER_FONT_SIZE, color=MUTED_INK, weight="bold")
    write_text(canvas, layout.splits.center, sub_middle, SPLITS_HEADER,
               size=HEADER_FONT_SIZE, color=MUTED_INK, weight="bold")
    write_text(canvas, layout.patients.center, sub_middle, PATIENTS_HEADER,
               size=HEADER_FONT_SIZE, color=MUTED_INK, weight="bold")
    for columns in layout.blocks:
        write_text(canvas, columns.span.center, parent_middle, columns.block.title,
                   size=PARENT_FONT_SIZE, color=INK, weight="bold")
        draw_line(canvas, columns.span.left, columns.span.right,
                  vertical.header_top + HEADER_PARENT_INCHES, INK, 0.5)
        for span, text in ((columns.delta, AXIS_LABELS[columns.block.kind]), (columns.bounds, ci_header),
                           (columns.pvalue, P_HEADER)):
            write_text(canvas, span.center, sub_middle, text,
                       size=HEADER_FONT_SIZE, color=MUTED_INK, weight="bold")
    draw_rule(canvas, layout, vertical.body_top - TOP_PAD_INCHES, INK, 0.7)


def draw_body(
    canvas,
    model: TableModel,
    layout: TableLayout,
    vertical: VerticalLayout,
    scales: Mapping[str, tuple[float, float]],
) -> None:
    top = vertical.body_top
    for index, group in enumerate(model.groups):
        if group.title:
            if index > 0:
                draw_rule(canvas, layout, top, RULE_COLOR, 0.6)
            write_text(canvas, layout.group.left, top + GROUP_TITLE_INCHES / 2, group.title,
                       ha="left", size=HEADER_FONT_SIZE, color=INK, weight="bold")
            top += GROUP_TITLE_INCHES
        for row in group.rows:
            draw_row(canvas, model.blocks, layout, row, top, scales)
            top += ROW_INCHES
    draw_forest_axes(
        canvas,
        layout,
        scales,
        top=vertical.body_top,
        axis_y=vertical.axis_y,
        show_label=True,
    )


def draw_forest_axes(
    canvas,
    layout: TableLayout,
    scales: Mapping[str, tuple[float, float]],
    *,
    top: float,
    axis_y: float,
    show_label: bool,
) -> None:
    """One x-axis under every forest column. ``top`` is where the zero reference line starts."""
    draw_rule(canvas, layout, axis_y, INK, 0.7)
    for columns in layout.blocks:
        draw_forest_axis(
            canvas,
            columns,
            scales[columns.block.kind],
            top=top,
            axis_y=axis_y,
            show_label=show_label,
        )


def draw_row(
    canvas,
    blocks: Sequence[Block],
    layout: TableLayout,
    row: SubgroupRow,
    top: float,
    scales: Mapping[str, tuple[float, float]],
) -> None:
    color = OVERALL_COLOR if row.is_overall else INK
    weight = "bold" if row.is_overall else "normal"
    middle = top + ROW_INCHES / 2
    if row.is_overall:
        canvas.add_patch(
            Rectangle(
                (layout.left, top),
                layout.right - layout.left,
                ROW_INCHES,
                facecolor=OVERALL_BAND,
                edgecolor="none",
                zorder=0,
            )
        )
    write_text(canvas, layout.group.left, middle, row.group, ha="left", color=color, weight=weight)
    write_text(canvas, layout.splits.center, middle, row.splits, color=color, weight=weight)
    write_text(canvas, layout.patients.center, middle, row.patients, color=color, weight=weight)
    for columns in layout.blocks:
        interval = row.intervals.get(columns.block.key)
        if interval is None:
            continue
        cell = estimate_cell(interval)
        write_text(canvas, columns.delta.center, middle, cell.estimate, color=color, weight=weight)
        write_text(canvas, columns.bounds.center, middle, cell.bounds, color=color, weight=weight)
        write_text(canvas, columns.pvalue.center, middle, cell.pvalue, color=color, weight="bold")
        draw_interval(canvas, columns.forest, scales[columns.block.kind], middle, interval, color)


def draw_interval(
    canvas,
    span: Span,
    scale: tuple[float, float],
    y: float,
    interval: Interval,
    color: str,
) -> None:
    low, high = scale
    left = low if not np.isfinite(interval.lower) else max(interval.lower, low)
    right = high if not np.isfinite(interval.upper) else min(interval.upper, high)
    x_left = to_canvas_x(span, scale, left)
    x_right = to_canvas_x(span, scale, right)
    canvas.plot([x_left, x_right], [y, y], color=color, linewidth=INTERVAL_WIDTH,
                solid_capstyle="butt", zorder=3)
    arrow = {"arrowstyle": "-|>", "color": color, "lw": 0.8, "mutation_scale": 6}
    if np.isfinite(interval.lower) and interval.lower < low:
        canvas.annotate("", xy=(x_left, y), xytext=(x_left + ARROW_INCHES * 2, y),
                        arrowprops=arrow, zorder=4)
    if np.isfinite(interval.upper) and interval.upper > high:
        canvas.annotate("", xy=(x_right, y), xytext=(x_right - ARROW_INCHES * 2, y),
                        arrowprops=arrow, zorder=4)
    canvas.plot(to_canvas_x(span, scale, interval.estimate), y, marker="o", markersize=MARKER_SIZE,
                color=color, zorder=5)


def draw_forest_axis(
    canvas,
    columns: BlockColumns,
    scale: tuple[float, float],
    *,
    top: float,
    axis_y: float,
    show_label: bool,
) -> None:
    span = columns.forest
    zero_x = to_canvas_x(span, scale, 0.0)
    canvas.plot([zero_x, zero_x], [top, axis_y], color=REFERENCE_COLOR,
                linestyle=(0, (2, 2)), linewidth=0.5, zorder=1)
    for tick in nice_ticks(scale):
        x = to_canvas_x(span, scale, tick)
        canvas.plot([x, x], [axis_y, axis_y + TICK_LENGTH_INCHES], color=INK, linewidth=0.6)
        canvas.text(x, axis_y + TICK_LABEL_OFFSET_INCHES, format_tick(tick), fontsize=TICK_FONT,
                    color=INK, ha="center", va="top")
    if show_label:
        canvas.text(span.center, axis_y + AXIS_LABEL_OFFSET_INCHES, AXIS_LABELS[columns.block.kind],
                    fontsize=TICK_FONT, color=MUTED_INK, ha="center", va="top")


def draw_rule(canvas, layout: TableLayout, y: float, color: str, linewidth: float) -> None:
    draw_line(canvas, layout.left, layout.right, y, color, linewidth)


def draw_line(canvas, x_start: float, x_end: float, y: float, color: str, linewidth: float) -> None:
    canvas.plot([x_start, x_end], [y, y], color=color, linewidth=linewidth, solid_capstyle="butt")


def write_text(
    canvas,
    x: float,
    y: float,
    text: str,
    *,
    ha: str = "center",
    size: float = TEXT_FONT_SIZE,
    color: str = INK,
    weight: str = "normal",
) -> None:
    canvas.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", fontweight=weight)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.policy_value_figures import (
    NEGATIVE_FILL,
    POLICY_VALUE_FIGURE_NAME,
    POSITIVE_FILL,
    build_value_table,
    cell_fill,
    plot_policy_value_table,
)


def value_frame() -> pd.DataFrame:
    """A table shaped like policy/value_comparison.csv for a three-arm study."""
    comparisons = {
        "vs_never_treat": (0.4, 0.3, 0.5),
        "vs_always_treat_1": (-0.1, -0.2, -0.01),
        "vs_always_treat_2": (0.05, -0.02, 0.12),
        "vs_observed": (0.2, 0.1, 0.3),
    }
    row = {
        "policy": "policy_tree_s_learner",
        "method": "policy_tree",
        "source_cate_model": "s_learner__unclipped__5c2a49272b",
        "n_test": 200,
        "share_treated": 0.6,
    }
    for name, (estimate, lower, upper) in comparisons.items():
        row[name] = estimate
        row[f"{name}_ci_lower"] = lower
        row[f"{name}_ci_upper"] = upper
    never = {"policy": "never_treat", "method": "never_treat", "source_cate_model": None, "n_test": 200, "share_treated": 0.0}
    for name in comparisons:
        never[name] = 0.0
        never[f"{name}_ci_lower"] = 0.0
        never[f"{name}_ci_upper"] = 0.0
    return pd.DataFrame([never, row])


def test_value_table_orders_comparison_groups_and_labels_rows() -> None:
    model = build_value_table(value_frame())
    assert model is not None
    assert [column.group for column in model.columns] == [
        "Versus treat everyone with",
        "Versus treat everyone with",
        "Versus treat no one",
        "Versus observed",
    ]
    assert [column.label for column in model.columns] == ["1", "2", "Treat no one", "Observed"]
    assert [row.label for row in model.rows] == ["Policy tree"]
    assert model.rows[0].detail == "S-learner"
    assert model.title == "Held-out policy value (n = 200)"


def test_value_table_shades_only_cells_whose_interval_excludes_zero() -> None:
    model = build_value_table(value_frame())
    assert model is not None
    fills = [cell_fill(cell) for cell in model.rows[0].cells]
    # Order: always 1, always 2, treat no one, observed.
    assert fills == [NEGATIVE_FILL, None, POSITIVE_FILL, POSITIVE_FILL]


def test_value_table_writes_png_only_when_a_learned_policy_exists(test_output_dir: Path) -> None:
    destination = test_output_dir / POLICY_VALUE_FIGURE_NAME
    plot_policy_value_table(table=value_frame(), save_path=str(destination))
    assert destination.is_file()
    assert destination.stat().st_size > 0

    baselines_only = value_frame().iloc[:1]
    skipped = test_output_dir / "baselines_only.png"
    plot_policy_value_table(table=baselines_only, save_path=str(skipped))
    assert not skipped.exists()


def test_value_table_precision_follows_the_size_of_the_values() -> None:
    frame = value_frame()
    scaled = frame.copy()
    for column in scaled.columns:
        if column.startswith("vs_"):
            scaled[column] = scaled[column] * 10.0
    small = build_value_table(frame)
    large = build_value_table(scaled)
    assert small is not None and large is not None
    assert small.decimals == 3
    assert large.decimals == 2
    # Treat-no-one is the third column after sorting; its raw estimate 0.4 scales to 4.0.
    assert np.isclose(large.rows[0].cells[2].estimate, 4.0)

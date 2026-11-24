from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causal_pipeline.plotting.balance_figures import (
    COVARIATE_BALANCE_FIGURE_NAME,
    THRESHOLD,
    compose_covariate_balance,
    plot_covariate_balance,
)
from tests.cache_paths import artifact_path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _binary_balance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "covariate": ["age", "bmi", "sex"],
            "unweighted_smd": [0.02, -0.31, 0.12],
            "weighted_smd": [0.01, -0.04, -0.13],
        }
    )


def _multi_arm_balance() -> pd.DataFrame:
    rows = []
    for arm_a, arm_b in [(1, 0), (2, 0)]:
        for covariate, unweighted, weighted in [("age", 0.2, 0.05), ("bmi", -0.15, 0.02)]:
            rows.append(
                {
                    "covariate": covariate,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "unweighted_smd": unweighted,
                    "weighted_smd": weighted,
                }
            )
    return pd.DataFrame(rows)


def test_love_plot_orders_covariates_by_unadjusted_magnitude() -> None:
    figure = compose_covariate_balance(_binary_balance())
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["bmi", "sex", "age"]
    finally:
        plt.close(figure)


def test_love_plot_legend_names_both_adjustment_states_in_two_columns() -> None:
    figure = compose_covariate_balance(_binary_balance())
    try:
        assert len(figure.legends) == 1
        legend = figure.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == ["Unadjusted", "Weighted"]
        assert legend._ncols == 2
    finally:
        plt.close(figure)


def test_love_plot_caps_covariates_at_the_top_ten_by_unadjusted_smd_by_default() -> None:
    names = [f"cov{index}" for index in range(12)]
    magnitudes = np.arange(12, dtype=float) / 100.0
    balance = pd.DataFrame(
        {"covariate": names, "unweighted_smd": magnitudes, "weighted_smd": magnitudes / 10.0}
    )
    figure = compose_covariate_balance(balance)
    try:
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert len(labels) == 10
        assert labels == [f"cov{index}" for index in range(11, 1, -1)]
    finally:
        plt.close(figure)


def test_love_plot_max_covariates_is_configurable() -> None:
    figure = compose_covariate_balance(_binary_balance(), max_covariates=2)
    try:
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ["bmi", "sex"]
    finally:
        plt.close(figure)


def test_love_plot_x_label_is_title_case() -> None:
    figure = compose_covariate_balance(_binary_balance())
    try:
        assert figure.axes[0].get_xlabel() == "Standardized Mean Difference"
        assert not any(text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_love_plot_marks_both_balance_thresholds_and_no_difference_line() -> None:
    figure = compose_covariate_balance(_binary_balance())
    try:
        axis = figure.axes[0]
        for threshold in (-THRESHOLD, THRESHOLD):
            matches = [
                line
                for line in axis.lines
                if np.allclose(np.asarray(line.get_xdata(), dtype=float), threshold)
            ]
            assert len(matches) == 1
        zero_lines = [
            line
            for line in axis.lines
            if np.allclose(np.asarray(line.get_xdata(), dtype=float), 0.0)
        ]
        assert len(zero_lines) == 1
    finally:
        plt.close(figure)


def test_multi_arm_balance_gets_one_titled_panel_per_contrast() -> None:
    figure = compose_covariate_balance(_multi_arm_balance())
    try:
        assert len(figure.axes) == 2
        assert [axis.get_title() for axis in figure.axes] == ["1 vs 0", "2 vs 0"]
        assert [[text.get_text() for text in axis.texts] for axis in figure.axes] == [["a"], ["b"]]
    finally:
        plt.close(figure)


def test_covariate_balance_figure_is_written_as_png(test_output_dir: Path) -> None:
    destination = artifact_path(test_output_dir, "diagnostics", COVARIATE_BALANCE_FIGURE_NAME)
    plot_covariate_balance(balance=_binary_balance(), save_path=str(destination))
    assert destination.is_file()
    assert destination.read_bytes().startswith(PNG_SIGNATURE)


def test_multi_arm_covariate_balance_figure_is_written_as_png(test_output_dir: Path) -> None:
    destination = artifact_path(test_output_dir, "diagnostics", "covariate_balance_multi_arm.png")
    plot_covariate_balance(balance=_multi_arm_balance(), save_path=str(destination))
    assert destination.read_bytes().startswith(PNG_SIGNATURE)


def test_covariate_balance_figure_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        compose_covariate_balance(pd.DataFrame({"covariate": ["age"], "unweighted_smd": [0.1]}))

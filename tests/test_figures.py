from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.table import Table

from causal_pipeline.figures import (
    CalibrationSeries,
    RankingSeries,
    compose_calibration_curves,
    compose_interval_estimates,
    compose_point_estimates,
    compose_ranking_curves,
    format_p_value,
    plot_interval_estimates,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_p_values_use_journal_thresholds() -> None:
    assert format_p_value(0.0004) == "< 0.001"
    assert format_p_value(0.0312) == "0.031"
    assert format_p_value(0.5) == "0.500"


def test_interval_figure_shows_confidence_intervals_and_no_effect_line(tmp_path: Path) -> None:
    estimates = pd.DataFrame(
        {
            "estimator": ["ipw", "aipw", "ipw", "aipw"],
            "contrast": ["1_vs_0", "1_vs_0", "2_vs_0", "2_vs_0"],
            "estimate": [0.2, -0.1, 0.4, 0.05],
            "ci_lower": [0.05, -0.3, 0.1, -0.2],
            "ci_upper": [0.35, 0.1, 0.7, 0.3],
        }
    )
    figure = compose_interval_estimates(
        estimates=estimates,
        estimator_column="estimator",
        contrast_column="contrast",
        estimate_column="estimate",
        lower_column="ci_lower",
        upper_column="ci_upper",
        xlabel="Average treatment effect",
    )
    assert len(figure.axes) == 2
    for axis in figure.axes:
        zero_lines = [
            line
            for line in axis.lines
            if np.allclose(np.asarray(line.get_xdata(), dtype=float), 0.0)
        ]
        assert len(zero_lines) == 1
        assert len(axis.containers) == 2
        assert all(isinstance(container, ErrorbarContainer) for container in axis.containers)
        assert {tick.get_text() for tick in axis.get_yticklabels()} == {"IPW", "AIPW"}
    assert any("no effect" in text.get_text() for text in figure.texts)
    plt.close(figure)

    destination = tmp_path / "ate_intervals.png"
    plot_interval_estimates(
        estimates=estimates,
        save_path=str(destination),
        estimator_column="estimator",
        contrast_column="contrast",
        estimate_column="estimate",
        lower_column="ci_lower",
        upper_column="ci_upper",
        xlabel="Average treatment effect",
    )
    assert destination.is_file()
    assert destination.read_bytes().startswith(PNG_SIGNATURE)


def test_point_figure_shows_means_without_intervals() -> None:
    estimates = pd.DataFrame(
        {
            "estimator": ["s_learner", "t_learner"],
            "contrast": ["1_vs_0", "1_vs_0"],
            "estimate": [0.15, -0.05],
        }
    )
    figure = compose_point_estimates(
        estimates=estimates,
        estimator_column="estimator",
        contrast_column="contrast",
        estimate_column="estimate",
        xlabel="Mean conditional treatment effect",
    )
    axis = figure.axes[0]
    zero_lines = [
        line
        for line in axis.lines
        if np.allclose(np.asarray(line.get_xdata(), dtype=float), 0.0)
    ]
    assert len(figure.axes) == 1
    assert len(zero_lines) == 1
    assert len(axis.containers) == 1
    assert isinstance(axis.containers[0], BarContainer)
    assert not any(isinstance(container, ErrorbarContainer) for container in axis.containers)
    assert {tick.get_text() for tick in axis.get_yticklabels()} == {"S-learner", "T-learner"}
    plt.close(figure)


def test_interval_figure_rejects_missing_estimates() -> None:
    estimates = pd.DataFrame(
        {
            "estimator": ["ipw"],
            "contrast": ["1_vs_0"],
            "estimate": [np.nan],
            "ci_lower": [0.0],
            "ci_upper": [0.1],
        }
    )
    with pytest.raises(ValueError):
        compose_interval_estimates(
            estimates=estimates,
            estimator_column="estimator",
            contrast_column="contrast",
            estimate_column="estimate",
            lower_column="ci_lower",
            upper_column="ci_upper",
            xlabel="Average treatment effect",
        )


def test_ranking_and_calibration_figures_report_metrics_under_each_panel() -> None:
    fraction = np.linspace(0.0, 1.0, 6)
    ranking = [
        RankingSeries(
            estimator_id="s_learner",
            toc_fraction=fraction,
            toc_value=0.2 * fraction * (1.0 - fraction),
            autoc=0.042,
            autoc_p_value=0.031,
            qini=0.018,
            qini_p_value=0.0002,
        ),
        RankingSeries(
            estimator_id="t_learner",
            toc_fraction=fraction,
            toc_value=0.05 * fraction * (1.0 - fraction),
            autoc=0.011,
            autoc_p_value=0.42,
            qini=-0.004,
            qini_p_value=0.67,
        ),
    ]
    ranking_figure = compose_ranking_curves(ranking)
    assert len(ranking_figure.axes) == 2
    zero_lines = [
        line
        for line in ranking_figure.axes[0].lines
        if np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.0)
    ]
    assert len(zero_lines) == 1
    legend_labels = ranking_figure.axes[0].get_legend().get_texts()
    assert [text.get_text() for text in legend_labels] == ["S-learner", "T-learner"]
    ranking_cells = table_cells(ranking_figure)
    assert ranking_cells[0] == ["", "AUTOC", "p", "Qini", "p"]
    assert ranking_cells[1] == ["S-learner", "0.042", "0.031", "0.018", "< 0.001"]
    assert ranking_cells[2] == ["T-learner", "0.011", "0.420", "-0.004", "0.670"]
    single_ranking = compose_ranking_curves([ranking[0]])
    assert single_ranking.axes[0].get_title() == "S-learner"
    assert table_cells(single_ranking)[1] == ["AUTOC", "0.042", "0.031"]
    assert table_cells(single_ranking)[2] == ["Qini", "0.018", "< 0.001"]
    plt.close(ranking_figure)
    plt.close(single_ranking)

    calibration = [
        CalibrationSeries(
            estimator_id="s_learner",
            predicted=np.array([-0.2, 0.0, 0.3]),
            observed=np.array([-0.1, 0.05, 0.25]),
            eceth=0.004,
            eceth_standard_error=0.002,
            eceth_p_value=0.04,
        ),
        CalibrationSeries(
            estimator_id="causal_forest",
            predicted=np.array([-0.1, 0.1, 0.4]),
            observed=np.array([-0.05, 0.08, 0.2]),
            eceth=0.021,
            eceth_standard_error=0.01,
            eceth_p_value=0.2,
        ),
    ]
    calibration_figure = compose_calibration_curves(calibration)
    assert len(calibration_figure.axes) == 2
    ideal = [
        line
        for line in calibration_figure.axes[0].lines
        if line.get_label() == "Ideal"
    ]
    assert len(ideal) == 1
    calibration_cells = table_cells(calibration_figure)
    assert calibration_cells[0] == ["", "ECETH", "SE", "p"]
    assert calibration_cells[1] == ["S-learner", "0.004", "0.002", "0.040"]
    assert calibration_cells[2] == ["Causal forest", "0.021", "0.010", "0.200"]
    single_calibration = compose_calibration_curves([calibration[0]])
    assert single_calibration.axes[0].get_title() == "S-learner"
    assert table_cells(single_calibration)[1] == ["ECETH", "0.004", "0.002", "0.040"]
    plt.close(calibration_figure)
    plt.close(single_calibration)


def table_cells(figure: plt.Figure) -> list[list[str]]:
    tables = [child for child in figure.axes[1].get_children() if isinstance(child, Table)]
    cells = tables[0].get_celld()
    row_count = 0
    column_count = 0
    for row, column in cells:
        row_count = max(row_count, row + 1)
        column_count = max(column_count, column + 1)
    return [
        [cells[(row, column)].get_text().get_text() for column in range(column_count)]
        for row in range(row_count)
    ]

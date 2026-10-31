from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.text import Text

from causal_pipeline.figures import (
    CalibrationSeries,
    RankingSeries,
    compose_evaluation_figure,
    compose_interval_estimates,
    compose_point_estimates,
    drop_nonfinite_effect_estimates,
    estimator_style,
    format_p_value,
    plot_evaluation_figure,
    plot_interval_estimates,
    plot_point_estimates,
)
from tests.cache_paths import artifact_path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_p_values_use_journal_thresholds() -> None:
    assert format_p_value(0.0004) == "< 0.01"
    assert format_p_value(0.0312) == "0.03"
    assert format_p_value(0.5) == "0.50"


def test_interval_figure_shows_confidence_intervals_and_no_effect_line(test_output_dir: Path) -> None:
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

    destination = artifact_path(test_output_dir, "summary", "ate_intervals.png")
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


def test_point_figure_shows_means_without_intervals(test_output_dir: Path) -> None:
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
    plot_point_estimates(
        estimates=estimates,
        save_path=str(artifact_path(test_output_dir, "summary", "cate_mean_effects.png")),
        estimator_column="estimator",
        contrast_column="contrast",
        estimate_column="estimate",
        xlabel="Mean conditional treatment effect",
    )
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


def test_drop_nonfinite_effect_estimates_keeps_plottable_rows() -> None:
    estimates = pd.DataFrame(
        {
            "estimator": ["s_learner", "dr_learner_trim"],
            "contrast": ["1_vs_0", "1_vs_0"],
            "estimate": [0.12, np.nan],
        }
    )
    kept = drop_nonfinite_effect_estimates(
        estimates,
        "estimate",
        figure_name="cate_mean_effects.png",
    )
    assert kept.shape[0] == 1
    assert kept["estimator"].iloc[0] == "s_learner"
    compose_point_estimates(
        estimates=kept,
        estimator_column="estimator",
        contrast_column="contrast",
        estimate_column="estimate",
        xlabel="Mean conditional treatment effect",
    )


def test_evaluation_figure_puts_calibration_and_ranking_side_by_side(test_output_dir: Path) -> None:
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
        CalibrationSeries(
            estimator_id="t_learner",
            predicted=np.array([-0.15, 0.05, 0.25]),
            observed=np.array([-0.08, 0.04, 0.22]),
            eceth=0.009,
            eceth_standard_error=0.004,
            eceth_p_value=0.3,
        ),
    ]
    ranking.append(
        RankingSeries(
            estimator_id="causal_forest",
            toc_fraction=fraction,
            toc_value=0.1 * fraction * (1.0 - fraction),
            autoc=0.030,
            autoc_p_value=0.08,
            qini=0.012,
            qini_p_value=0.2,
        )
    )
    figure = compose_evaluation_figure(calibration=calibration, ranking=ranking)
    assert len(figure.axes) == 4
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert figure.legends
    zero_lines = [
        line
        for line in figure.axes[1].lines
        if np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.0)
    ]
    assert len(zero_lines) == 1
    texts = [text.get_text() for text in figure.findobj(Text)]
    for expected in [
        "S-learner",
        "T-learner",
        "Causal forest",
        "Estimator",
        "ECETH",
        "AUTOC",
        "(a)",
        "(b)",
        "Calibration",
        "Targeting Operator Curve (TOC)",
        "Mean Predicted CATE",
        "Mean Robust Proxy",
        "Top Fraction Ranked by Benefit",
        "Excess Benefit",
    ]:
        assert expected in texts
    assert "NA" not in texts
    assert "SE" not in texts
    assert not any("estimation sample" in text or "held-out" in text for text in texts)
    assert not any(text.startswith(("Left:", "Right:")) for text in texts)
    plt.close(figure)
    plot_evaluation_figure(
        calibration=calibration,
        ranking=ranking,
        save_path=str(artifact_path(test_output_dir, "cate", "comparison", "1_vs_0", "evaluation.png")),
    )


def test_evaluation_figure_marks_missing_estimator_as_na() -> None:
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
        )
    ]
    calibration = [
        CalibrationSeries(
            estimator_id="t_learner",
            predicted=np.array([-0.2, 0.0, 0.3]),
            observed=np.array([-0.1, 0.05, 0.25]),
            eceth=0.004,
            eceth_standard_error=0.002,
            eceth_p_value=0.04,
        )
    ]
    figure = compose_evaluation_figure(calibration=calibration, ranking=ranking)
    texts = [text.get_text() for text in figure.findobj(Text)]
    assert "NA" in texts
    plt.close(figure)


def test_estimator_style_is_fixed_per_estimator() -> None:
    assert estimator_style("causal_forest") == estimator_style("causal_forest")
    assert estimator_style("s_learner") != estimator_style("t_learner")
    assert estimator_style("custom_estimator") == estimator_style("custom_estimator")

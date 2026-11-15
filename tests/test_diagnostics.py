from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causal_pipeline.config import PipelineConfig, TreatmentMode
from causal_pipeline.data import CausalDataset, DataPartitions
from causal_pipeline.diagnostics import (
    DIAGNOSTIC_PAIR_FIGURE_NAME,
    OverlapPreprocessor,
    arm_effective_sample_sizes,
    compose_diagnostic_pair,
    covariate_overlap_mask,
    kish_effective_sample_size,
    plot_diagnostic_pair,
    smd_two_groups,
)
from tests.cache_paths import artifact_path


def test_kish_effective_sample_size_matches_the_count_when_weights_are_equal() -> None:
    weights = np.ones(4)
    assert kish_effective_sample_size(weights) == 4.0
    assert kish_effective_sample_size(weights * 5.0) == 4.0


def test_kish_effective_sample_size_shrinks_when_one_weight_dominates() -> None:
    assert kish_effective_sample_size(np.array([1.0, 1.0, 1.0, 3.0])) == 3.0


def test_arm_effective_sample_size_uses_the_probability_of_the_received_arm() -> None:
    sizes = arm_effective_sample_sizes(
        treatment=np.array([0, 0, 1, 1]),
        probabilities=pd.DataFrame({1: [0.2, 0.4, 0.8, 0.8]}),
        treatment_mode=TreatmentMode.BINARY,
        treatment_values=[0, 1],
    )
    by_arm = {sample.label: sample for sample in sizes}
    assert by_arm["0"].count == 2
    assert by_arm["1"].count == 2
    control_weights = 1.0 / np.array([0.8, 0.6])
    assert by_arm["0"].effective_sample_size == pytest.approx(kish_effective_sample_size(control_weights))
    assert by_arm["1"].effective_sample_size == pytest.approx(2.0)


def test_smd_two_groups_zero_for_identical_samples() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert smd_two_groups(group_a=values, group_b=values) == 0.0


def test_covariate_overlap_mask_requires_every_arm_inside_bounds() -> None:
    propensity = pd.DataFrame({0: [0.01, 0.4], 1: [0.99, 0.6]})
    mask = covariate_overlap_mask(propensity, (0.1, 0.9))
    assert mask.tolist() == [False, True]


def test_overlap_preprocessor_produces_balance_table(
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    preprocessor = OverlapPreprocessor(overlap=pipeline_config_no_policy.overlap, minimum_folds=2)
    dataset = CausalDataset(data=pipeline_config_no_policy.data, df=df_synthetic_binary)
    result = preprocessor.run(partitions=DataPartitions(estimation=dataset)).diagnostics
    assert not result.covariate_balance.empty
    assert set(result.covariate_balance.columns) >= {
        "covariate",
        "unweighted_smd",
        "weighted_smd",
    }
    assert len(result.covariate_balance) == len(pipeline_config_no_policy.data.confounders)


def test_overlap_preprocessor_writes_plot_under_results_root(
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
    test_output_dir: Path,
) -> None:
    preprocessor = OverlapPreprocessor(overlap=pipeline_config_no_policy.overlap, minimum_folds=2)
    dataset = CausalDataset(data=pipeline_config_no_policy.data, df=df_synthetic_binary)
    result = preprocessor.run(
        partitions=DataPartitions(estimation=dataset),
        results_root=str(test_output_dir),
    )
    overlap_path = test_output_dir / "diagnostics" / "propensity_overlap.png"
    assert overlap_path.is_file()
    assert (test_output_dir / "diagnostics" / "propensity_overlap.csv").is_file()
    assert result.diagnostics.propensity_overlap_path is not None
    assert (test_output_dir / "diagnostics" / "covariate_balance.png").is_file()
    assert (test_output_dir / "diagnostics" / DIAGNOSTIC_PAIR_FIGURE_NAME).is_file()


def test_diagnostic_pair_puts_propensity_left_of_balance() -> None:
    probabilities = pd.DataFrame({1: [0.15, 0.72, 0.28, 0.81, 0.44, 0.63]})
    balance = pd.DataFrame(
        {
            "covariate": ["age", "bmi", "sex"],
            "unweighted_smd": [0.02, -0.31, 0.12],
            "weighted_smd": [0.01, -0.04, -0.13],
        }
    )
    figure = compose_diagnostic_pair(
        treatment=np.array([0, 1, 0, 1, 0, 1]),
        probabilities=probabilities,
        treatment_mode=TreatmentMode.BINARY,
        treatment_values=[0, 1],
        control_value=0,
        balance=balance,
    )
    try:
        overlap_axes = [axis for axis in figure.axes if axis.get_ylabel() == "Density"]
        balance_axes = [axis for axis in figure.axes if axis.get_xlabel() == "Standardized Mean Difference"]
        assert len(overlap_axes) == 1
        assert len(balance_axes) == 1
        assert overlap_axes[0].get_position().x1 < balance_axes[0].get_position().x0
        assert overlap_axes[0].get_xlabel() == "P(T = 1 | X)"
        assert [tick.get_text() for tick in balance_axes[0].get_yticklabels()] == ["bmi", "sex", "age"]
        legend_text = sorted(text.get_text() for legend in figure.legends for text in legend.get_texts())
        assert legend_text == ["0", "1", "Unadjusted", "Weighted"]
        assert [text.get_text() for axis in figure.axes for text in axis.texts] == ["a", "b"]
        note = " ".join(text.get_text() for text in figure.texts)
        assert "0: n = 3, ESS = " in note
        assert "1: n = 3, ESS = " in note
    finally:
        plt.close(figure)


def test_diagnostic_pair_is_written_beside_the_separate_figures(test_output_dir: Path) -> None:
    probabilities = pd.DataFrame({1: [0.2, 0.8, 0.4, 0.6]})
    balance = pd.DataFrame(
        {
            "covariate": ["age", "bmi"],
            "unweighted_smd": [0.2, -0.4],
            "weighted_smd": [0.05, -0.03],
        }
    )
    destination = artifact_path(test_output_dir, "diagnostics", DIAGNOSTIC_PAIR_FIGURE_NAME)
    plot_diagnostic_pair(
        treatment=np.array([0, 1, 0, 1]),
        probabilities=probabilities,
        treatment_mode=TreatmentMode.BINARY,
        treatment_values=[0, 1],
        control_value=0,
        balance=balance,
        save_path=str(destination),
    )
    assert destination.is_file()
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

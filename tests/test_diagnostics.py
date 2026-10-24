from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset, DataPartitions
from causal_pipeline.diagnostics import OverlapPreprocessor, covariate_overlap_mask, smd_two_groups


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

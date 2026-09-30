from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset
from causal_pipeline.diagnostics import DiagnosticsRunner, smd_two_groups


def test_smd_two_groups_zero_for_identical_samples() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert smd_two_groups(group_a=values, group_b=values) == 0.0


def test_diagnostics_runner_produces_balance_table(
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    runner = DiagnosticsRunner(diagnostics=pipeline_config_no_policy.diagnostics)
    dataset = CausalDataset(data=pipeline_config_no_policy.data, df=df_synthetic_binary)
    result = runner.run(train=dataset)
    assert not result.covariate_balance.empty
    assert set(result.covariate_balance.columns) >= {
        "covariate",
        "unweighted_smd",
        "weighted_smd",
    }
    assert len(result.covariate_balance) == len(pipeline_config_no_policy.data.confounders)


def test_diagnostics_writes_overlap_plot_under_results_root(
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
    tmp_path: Path,
) -> None:
    runner = DiagnosticsRunner(diagnostics=pipeline_config_no_policy.diagnostics)
    dataset = CausalDataset(data=pipeline_config_no_policy.data, df=df_synthetic_binary)
    result = runner.run(train=dataset, results_root=str(tmp_path))
    overlap_path = tmp_path / "diagnostics" / "propensity_overlap.png"
    assert overlap_path.is_file()
    assert result.propensity_overlap_path is not None

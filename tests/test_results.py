from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset, DataPartitions
from causal_pipeline.results import ResultStore


def test_result_store_persists_partitions(
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
    test_output_dir: Path,
) -> None:
    store = ResultStore(config=pipeline_config_no_policy)
    data_config = pipeline_config_no_policy.data
    estimation = CausalDataset(data=data_config, df=df_synthetic_binary.iloc[:100])
    store.save_partitions(partitions=DataPartitions(estimation=estimation))
    assert (test_output_dir / "data" / "estimation.parquet").is_file()
    assert not (test_output_dir / "data" / "test.parquet").exists()
    held_out = CausalDataset(data=data_config, df=df_synthetic_binary.iloc[100:])
    store.save_partitions(
        partitions=DataPartitions(estimation=estimation, test=held_out),
    )
    assert (test_output_dir / "data" / "test.parquet").is_file()


def test_result_store_writes_csv_summary(
    pipeline_config_no_policy: PipelineConfig,
    test_output_dir: Path,
) -> None:
    store = ResultStore(config=pipeline_config_no_policy)
    ate_summary = pd.DataFrame({"estimator": ["ipw"], "estimate": [0.1]})
    cate_summary = pd.DataFrame({"estimator": ["s_learner"], "eceth": [0.01]})
    store.write_summaries(
        ate_summary=ate_summary,
        cate_summary=cate_summary,
        policy_summary=None,
    )
    assert (test_output_dir / "summary" / "ate_estimators.csv").is_file()
    assert (test_output_dir / "summary" / "cate_estimators.csv").is_file()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset, DataPartitions
from causal_pipeline.results import ResultStore


def test_result_store_persists_partitions(
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
    tmp_path: Path,
) -> None:
    config = pipeline_config_no_policy.model_copy(update={"results_dir": str(tmp_path)})
    store = ResultStore(config=config)
    data_config = config.data
    estimation = CausalDataset(data=data_config, df=df_synthetic_binary.iloc[:100])
    store.save_partitions(partitions=DataPartitions(estimation=estimation))
    assert (tmp_path / "data" / "estimation.parquet").is_file()
    assert not (tmp_path / "data" / "test.parquet").exists()
    held_out = CausalDataset(data=data_config, df=df_synthetic_binary.iloc[100:])
    store.save_partitions(
        partitions=DataPartitions(estimation=estimation, test=held_out),
    )
    assert (tmp_path / "data" / "test.parquet").is_file()


def test_result_store_writes_csv_summary(
    tmp_path: Path,
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    config = pipeline_config_no_policy.model_copy(update={"results_dir": str(tmp_path)})
    store = ResultStore(config=config)
    ate_summary = pd.DataFrame({"estimator": ["ipw"], "estimate": [0.1]})
    cate_summary = pd.DataFrame({"estimator": ["s_learner"], "eceth": [0.01]})
    store.write_summaries(
        ate_summary=ate_summary,
        cate_summary=cate_summary,
        policy_summary=None,
    )
    assert (tmp_path / "summary" / "ate_estimators.csv").is_file()
    assert (tmp_path / "summary" / "cate_estimators.csv").is_file()

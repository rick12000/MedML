import pandas as pd

from causal_pipeline.data import CausalDataset, DataPartitions
from causal_pipeline.diagnostics import DiagnosticResult
from causal_pipeline.results import ResultStore


def test_result_store_persists_partitions(
    pipeline_config_no_policy,
    df_synthetic_binary: pd.DataFrame,
    tmp_path,
) -> None:
    config = pipeline_config_no_policy.model_copy(update={"results_dir": str(tmp_path)})
    store = ResultStore(config)
    df_train = df_synthetic_binary.iloc[:100]
    df_validation = df_synthetic_binary.iloc[100:]
    data_config = config.data
    partitions = DataPartitions(
        train=CausalDataset(data=data_config, df=df_train),
        validation=CausalDataset(data=data_config, df=df_validation),
        test=CausalDataset(data=data_config, df=df_train.iloc[0:0]),
    )
    store.save_partitions(partitions)
    assert (tmp_path / "data" / "train.parquet").is_file()
    assert (tmp_path / "data" / "validation.parquet").is_file()


def test_result_store_writes_csv_summary(tmp_path, pipeline_config_no_policy) -> None:
    config = pipeline_config_no_policy.model_copy(update={"results_dir": str(tmp_path)})
    store = ResultStore(config)
    ate_summary = pd.DataFrame({"estimator": ["ipw"], "estimate": [0.1]})
    cate_summary = pd.DataFrame({"estimator": ["s_learner"], "eceth": [0.01]})
    store.write_summaries(ate_summary, cate_summary, policy_summary=None)
    assert (tmp_path / "summary" / "ate_estimators.csv").is_file()
    assert (tmp_path / "summary" / "cate_estimators.csv").is_file()

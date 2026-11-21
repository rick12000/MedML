from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.utils import read_dataframe


def test_same_config_and_data_give_identical_outputs(
    reproducibility_config: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    first_root = Path(reproducibility_config.results_dir)
    second_root = first_root.with_name("second")
    CausalPipeline(config=reproducibility_config).run(df_synthetic_binary)
    CausalPipeline(
        config=reproducibility_config.model_copy(update={"results_dir": str(second_root)})
    ).run(df_synthetic_binary)

    tables = [path.relative_to(first_root) for path in first_root.rglob("*") if path.suffix in {".csv", ".parquet"}]
    assert tables
    for table in tables:
        pd.testing.assert_frame_equal(read_dataframe(first_root / table), read_dataframe(second_root / table))

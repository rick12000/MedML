"""Project entrypoint: load cohort data and run the causal pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.settings import build_pipeline_config
from causal_pipeline.utils import read_dataframe

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = build_pipeline_config(include_policy=False)
    data_path = Path("data") / "analysis.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected prepared dataset at {data_path}. "
            "Place your observational cohort parquet file there before running.",
        )
    df_input = read_dataframe(data_path)
    pipeline = CausalPipeline(config)
    pipeline.run(df_input)


if __name__ == "__main__":
    main()

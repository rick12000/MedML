"""Result persistence and output directory conventions."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import DataPartitions
from causal_pipeline.diagnostics import DiagnosticResult
from causal_pipeline.utils import (
    ensure_directory,
    write_dataframe as persist_dataframe,
    write_text,
)

logger = logging.getLogger(__name__)


class ResultStore:
    """Create directories and persist pipeline outputs."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.root = ensure_directory(config.results_dir)

    def save_partitions(self, partitions: DataPartitions) -> None:
        data_dir = ensure_directory(self.root / "data")
        persist_dataframe(data_dir / "estimation.parquet", partitions.estimation.df)
        if partitions.test is not None:
            persist_dataframe(data_dir / "test.parquet", partitions.test.df)
        logger.info("Saved data partitions under %s", data_dir)

    def save_diagnostics(self, diagnostics: DiagnosticResult) -> None:
        diag_dir = ensure_directory(self.root / "diagnostics")
        if not diagnostics.covariate_balance.empty:
            persist_dataframe(
                diag_dir / "covariate_balance.csv",
                diagnostics.covariate_balance,
            )
        if diagnostics.propensity_overlap_path is not None:
            write_text(
                diag_dir / "propensity_overlap_path.txt",
                diagnostics.propensity_overlap_path,
            )
        logger.info("Saved diagnostics under %s", diag_dir)

    def ate_dir(self, estimator_id: str) -> Path:
        return ensure_directory(self.root / "ate" / estimator_id)

    def cate_dir(self, estimator_id: str) -> Path:
        return ensure_directory(self.root / "cate" / estimator_id)

    def write_dataframe(self, relative_path: str, df: pd.DataFrame) -> Path:
        return persist_dataframe(self.root / relative_path, df)

    def write_summaries(
        self,
        ate_summary: pd.DataFrame,
        cate_summary: pd.DataFrame,
        policy_summary: pd.DataFrame | None,
    ) -> None:
        summary_dir = ensure_directory(self.root / "summary")
        persist_dataframe(summary_dir / "ate_estimators.csv", ate_summary)
        persist_dataframe(summary_dir / "cate_estimators.csv", cate_summary)
        if policy_summary is not None:
            persist_dataframe(summary_dir / "policy_methods.csv", policy_summary)
        logger.info("Wrote aggregate summaries to %s", summary_dir)

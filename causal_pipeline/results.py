"""Result persistence and output directory conventions."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import DataPartitions
from causal_pipeline.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)


class ResultStore:
    """Create directories and persist pipeline outputs."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.root = Path(config.results_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_partitions(self, partitions: DataPartitions) -> None:
        data_dir = self.root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        partitions.train.df.to_parquet(data_dir / "train.parquet")
        partitions.validation.df.to_parquet(data_dir / "validation.parquet")
        partitions.test.df.to_parquet(data_dir / "test.parquet")
        logger.info("Saved data partitions under %s", data_dir)

    def save_diagnostics(self, diagnostics: DiagnosticResult) -> None:
        diag_dir = self.root / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        if not diagnostics.covariate_balance.empty:
            diagnostics.covariate_balance.to_csv(
                diag_dir / "covariate_balance.csv",
                index=False,
            )
        if diagnostics.propensity_overlap_path is not None:
            (diag_dir / "propensity_overlap_path.txt").write_text(
                diagnostics.propensity_overlap_path,
                encoding="utf-8",
            )
        logger.info("Saved diagnostics under %s", diag_dir)

    def ate_dir(self, estimator_id: str) -> Path:
        path = self.root / "ate" / estimator_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cate_dir(self, estimator_id: str) -> Path:
        path = self.root / "cate" / estimator_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_dataframe(self, relative_path: str, df: pd.DataFrame) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix == ".parquet":
            df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported output extension: {path.suffix}")
        return path

    def write_summaries(
        self,
        ate_summary: pd.DataFrame,
        cate_summary: pd.DataFrame,
        policy_summary: pd.DataFrame | None,
    ) -> None:
        summary_dir = self.root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        ate_summary.to_csv(summary_dir / "ate_estimators.csv", index=False)
        cate_summary.to_csv(summary_dir / "cate_estimators.csv", index=False)
        if policy_summary is not None:
            policy_summary.to_csv(summary_dir / "policy_methods.csv", index=False)
        logger.info("Wrote aggregate summaries to %s", summary_dir)

"""Pooled outputs from one completed analysis per imputed covariate draw."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.cate import mean_cate_table
from causal_pipeline.config import DEFAULT_ATE_CONFIDENCE_LEVEL, PipelineConfig
from causal_pipeline.diagnostics import plot_overlap
from causal_pipeline.estimator_identity import assign_estimator_ids, attach_estimator_metadata
from causal_pipeline.evaluation import (
    ContrastEvaluation,
    save_joint_cate_figures,
    write_ranking_curve_data,
)
from causal_pipeline.figures import CalibrationSeries, RankingSeries, plot_calibration_curves, plot_ranking_curves
from causal_pipeline.pooling import (
    average_aligned_frames,
    pool_contrast_evaluation,
    pool_curve,
    pool_keyed_estimates,
    pool_point_columns,
    pool_subgroup_table,
    reference_sample_size,
)
from causal_pipeline.results import ResultStore
from causal_pipeline.subgroup_figures import SUBGROUP_SPLITS_FIGURE_NAME, plot_subgroup_splits_table
from causal_pipeline.utils import read_dataframe, write_dataframe

logger = logging.getLogger(__name__)

RATE_NOT_POOLED = (
    "AUTOC and Qini are averaged across imputations. "
    "Their half-sample spreads are not sampling variances, so pooled standard errors "
    "and p-values are left missing."
)


@dataclass(frozen=True)
class CompletedAnalysis:
    """Estimates from one completed covariate draw, before pooling."""

    ate_by_estimator: dict[str, pd.DataFrame]
    df_subgroups: pd.DataFrame | None
    cate_by_estimator: dict[str, pd.DataFrame]
    mean_cate_by_estimator: dict[str, pd.DataFrame]
    cate_evaluation: dict[str, list[ContrastEvaluation]]
    df_balance: pd.DataFrame
    df_transport: pd.DataFrame | None
    estimation_rows: int
    cluster_count: int | None
    results_root: Path


def estimation_sample_size(completed_runs: list[CompletedAnalysis]) -> int:
    row_counts = {run.estimation_rows for run in completed_runs}
    cluster_counts = {run.cluster_count for run in completed_runs}
    if len(row_counts) != 1 or len(cluster_counts) != 1:
        raise ValueError("Imputations produced different estimation sample sizes.")
    return reference_sample_size(
        row_count=completed_runs[0].estimation_rows,
        cluster_count=completed_runs[0].cluster_count,
    )


def write_pooled_ate(
    config: PipelineConfig,
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
    sample_size: int,
) -> dict[str, pd.DataFrame]:
    """Rubin-pool each ATE contrast into ate/<id>/estimate.csv and ate/summary.csv."""
    ate_by_estimator: dict[str, pd.DataFrame] = {}
    df_summaries = []
    for estimator_id, spec in assign_estimator_ids(config.ate_estimators or []):
        df_pooled_ate = pool_keyed_estimates(
            frames=[run.ate_by_estimator[estimator_id] for run in completed_runs],
            keys=["contrast"],
            level=spec.confidence_level,
            sample_size=sample_size,
        )
        ate_by_estimator[estimator_id] = df_pooled_ate
        write_dataframe(results.ate_dir(estimator_id) / "estimate.csv", df_pooled_ate)
        df_summaries.append(df_pooled_ate)
    if df_summaries:
        results.write_dataframe("ate/summary.csv", pd.concat(df_summaries, ignore_index=True))
    return ate_by_estimator


def write_pooled_subgroups(
    config: PipelineConfig,
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
    sample_size: int,
) -> None:
    """Rubin-pool subgroup differences into ate/subgroup_splits.csv."""
    df_subgroups = [run.df_subgroups for run in completed_runs]
    if all(df_one is not None for df_one in df_subgroups):
        df_present = [df_one for df_one in df_subgroups if df_one is not None]
        df_pooled = pool_subgroup_table(
            frames=df_present,
            level_column="confidence_level",
            sample_size=sample_size,
        )
        results.write_dataframe("ate/subgroup_splits.csv", df_pooled)
        splits = config.subgroup_splits or []
        plot_subgroup_splits_table(
            table=df_pooled,
            split_order=[(split.column, split.cutoff) for split in splits],
            save_path=str(results.root / "ate" / SUBGROUP_SPLITS_FIGURE_NAME),
        )


def write_pooled_cate_predictions(
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
) -> dict[str, pd.DataFrame]:
    """Average cross-fit CATE predictions and write the between-imputation spread."""
    df_means: dict[str, pd.DataFrame] = {}
    if completed_runs[0].cate_by_estimator:
        for estimator_id in completed_runs[0].cate_by_estimator:
            df_mean, df_between = average_aligned_frames(
                [run.cate_by_estimator[estimator_id] for run in completed_runs]
            )
            df_means[estimator_id] = df_mean
            write_dataframe(results.cate_dir(estimator_id) / "crossfit_predictions.parquet", df_mean)
            write_dataframe(
                results.cate_dir(estimator_id) / "crossfit_predictions_between_sd.parquet",
                df_between,
            )
    return df_means


def write_pooled_cate_means(
    config: PipelineConfig,
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
    cate_by_estimator: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Mean of the averaged CATE predictions, with the spread of the per-draw means."""
    mean_cate_by_estimator: dict[str, pd.DataFrame] = {}
    specs = dict(assign_estimator_ids(config.cate_estimators or []))
    for estimator_id, df_predictions in cate_by_estimator.items():
        df_mean_cate = mean_cate_table(df_predictions)
        per_draw = [
            mean_cate_table(run.cate_by_estimator[estimator_id]).set_index("contrast")["estimate"]
            for run in completed_runs
        ]
        df_mean_cate["between_sd"] = df_mean_cate["contrast"].map(
            pd.concat(per_draw, axis=1).std(axis=1, ddof=1)
        )
        write_dataframe(
            results.cate_dir(estimator_id) / "ate_estimate.csv",
            attach_estimator_metadata(
                df_mean_cate,
                estimator_id=estimator_id,
                spec=specs[estimator_id],
            ),
        )
        mean_cate_by_estimator[estimator_id] = df_mean_cate
    return mean_cate_by_estimator


def pooled_cate_evaluation(
    completed_runs: list[CompletedAnalysis],
    sample_size: int,
) -> dict[str, list[ContrastEvaluation]]:
    """Rubin-pool ECETH. Average AUTOC and Qini and leave their sampling results missing."""
    evaluation_by_estimator: dict[str, list[ContrastEvaluation]] = {}
    if completed_runs[0].cate_evaluation:
        logger.warning(RATE_NOT_POOLED)
        for estimator_id in completed_runs[0].cate_evaluation:
            grouped = zip(
                *(run.cate_evaluation[estimator_id] for run in completed_runs),
                strict=True,
            )
            evaluation_by_estimator[estimator_id] = [
                pool_contrast_evaluation(
                    evaluations=list(series),
                    sample_size=sample_size,
                    level=DEFAULT_ATE_CONFIDENCE_LEVEL,
                    tolerance=float(np.mean([item.eceth_tolerance for item in series])),
                )
                for series in grouped
            ]
    return evaluation_by_estimator


def write_pooled_curves(
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
    cate_evaluation: dict[str, list[ContrastEvaluation]],
) -> None:
    """Average TOC and calibration curves when every draw used the same grid."""
    ranking_by_contrast: dict[str, list[RankingSeries]] = {}
    calibration_by_contrast: dict[str, list[CalibrationSeries]] = {}
    for estimator_id, contrasts in cate_evaluation.items():
        for item in contrasts:
            df_toc = read_curve_tables(completed_runs, item.toc_path)
            df_calibration = read_curve_tables(completed_runs, item.calibration_path)
            toc_aligned = all(len(df_one) == len(df_toc[0]) for df_one in df_toc)
            calibration_aligned = all(len(df_one) == len(df_calibration[0]) for df_one in df_calibration)
            if not toc_aligned:
                logger.warning(
                    "Skipping pooled TOC for %s %s because the grids differ.",
                    estimator_id,
                    item.contrast,
                )
            elif not calibration_aligned:
                logger.warning(
                    "Skipping pooled calibration for %s %s because the bins differ.",
                    estimator_id,
                    item.contrast,
                )
            else:
                df_toc_mean = pool_curve(frames=df_toc, x_column="toc_fraction", y_columns=["toc_value"])
                ranking = RankingSeries(
                    estimator_id=estimator_id,
                    toc_fraction=df_toc_mean["toc_fraction"].to_numpy(dtype=float),
                    toc_value=df_toc_mean["toc_value"].to_numpy(dtype=float),
                    autoc=item.rate_autoc,
                    autoc_p_value=item.rate_autoc_pvalue,
                    qini=item.rate_qini,
                    qini_p_value=item.rate_qini_pvalue,
                )
                toc_file = results.root / item.toc_path
                plot_ranking_curves(series=[ranking], save_path=str(toc_file))
                write_ranking_curve_data(toc_file, [ranking])
                ranking_by_contrast.setdefault(item.contrast, []).append(ranking)
                df_bins = []
                for df_one in df_calibration:
                    df_indexed = df_one.copy()
                    df_indexed["bin_index"] = np.arange(len(df_indexed))
                    df_bins.append(df_indexed)
                df_calibration_mean = pool_curve(
                    frames=df_bins,
                    x_column="bin_index",
                    y_columns=["mean_predicted_cate", "mean_robust_proxy"],
                )
                series = CalibrationSeries(
                    estimator_id=estimator_id,
                    predicted=df_calibration_mean["mean_predicted_cate"].to_numpy(dtype=float),
                    observed=df_calibration_mean["mean_robust_proxy"].to_numpy(dtype=float),
                    eceth=item.eceth,
                    eceth_standard_error=item.eceth_standard_error,
                    eceth_p_value=item.eceth_pvalue,
                )
                calibration_file = results.root / item.calibration_path
                plot_calibration_curves(series=[series], save_path=str(calibration_file))
                write_dataframe(calibration_file.with_suffix(".csv"), df_calibration_mean)
                calibration_by_contrast.setdefault(item.contrast, []).append(series)
    if ranking_by_contrast and calibration_by_contrast:
        save_joint_cate_figures(
            ranking_by_contrast=ranking_by_contrast,
            calibration_by_contrast=calibration_by_contrast,
            results_root=results.root,
            held_out=False,
        )


def read_curve_tables(completed_runs: list[CompletedAnalysis], relative_path: str) -> list[pd.DataFrame]:
    return [
        read_dataframe((run.results_root / relative_path).with_suffix(".csv"))
        for run in completed_runs
    ]


def write_pooled_balance(
    config: PipelineConfig,
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
) -> None:
    """Average covariate balance and the propensity-overlap scatter."""
    balance_tables = [run.df_balance for run in completed_runs if not run.df_balance.empty]
    if len(balance_tables) == len(completed_runs) and balance_tables:
        keys = [column for column in ("covariate", "arm_a", "arm_b") if column in balance_tables[0].columns]
        values = [column for column in balance_tables[0].columns if column not in keys]
        results.write_dataframe(
            "diagnostics/covariate_balance.csv",
            pool_point_columns(frames=balance_tables, keys=keys, values=values),
        )
    overlap_paths = [run.results_root / "diagnostics" / "propensity_overlap.csv" for run in completed_runs]
    if all(path.exists() for path in overlap_paths):
        overlap_tables = [read_dataframe(path) for path in overlap_paths]
        same_length = all(len(df_one) == len(overlap_tables[0]) for df_one in overlap_tables)
        if same_length:
            value_columns = [column for column in overlap_tables[0].columns if column != "treatment"]
            df_averaged = overlap_tables[0][["treatment"]].copy()
            for column in value_columns:
                stack = np.vstack([df_one[column].to_numpy(dtype=float) for df_one in overlap_tables])
                df_averaged[column] = np.mean(stack, axis=0)
            destination = results.root / "diagnostics" / "propensity_overlap.csv"
            write_dataframe(destination, df_averaged)
            data = config.data
            if "propensity" in df_averaged.columns:
                non_control = next(arm for arm in data.treatment_values if arm != data.control_value)
                df_probabilities = pd.DataFrame({non_control: df_averaged["propensity"]})
            else:
                df_probabilities = df_averaged.drop(columns=["treatment"])
            plot_overlap(
                treatment=df_averaged["treatment"].to_numpy(),
                probabilities=df_probabilities,
                treatment_mode=data.treatment_mode,
                treatment_values=data.treatment_values,
                control_value=data.control_value,
                save_path=str(destination.with_suffix(".png")),
            )


def write_pooled_transport(
    results: ResultStore,
    completed_runs: list[CompletedAnalysis],
) -> None:
    """Average transport point estimates. They have no sampling variance to pool."""
    transport_tables = [run.df_transport for run in completed_runs]
    if transport_tables and all(df_one is not None for df_one in transport_tables):
        present_tables = [df_one for df_one in transport_tables if df_one is not None]
        value_columns = [
            column
            for column in ("doubly_robust", "outcome_standardization", "inverse_odds")
            if column in present_tables[0].columns
        ]
        results.write_dataframe(
            "selection/transport.csv",
            pool_point_columns(frames=present_tables, keys=["contrast"], values=value_columns),
        )

"""High-level causal analysis orchestration."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import warnings
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from sklearn.base import BaseEstimator

from causal_pipeline.ate import (
    ATESensitivityAnalyzer,
    BaseATEEstimator,
    benchmark_sensitivity_models,
    initialize_ate_estimator,
    estimator_supports_sensitivity,
)
from causal_pipeline.cate import create_cate_estimator, ensure_cate_spec_supported, mean_cate_table
from causal_pipeline.config import (
    ATEEstimatorSpec,
    ATEKind,
    ArbitraryTypesModel,
    CATEEstimatorSpec,
    CATEKind,
    DEFAULT_LEARNER_RANDOM_STATE,
    DEFAULT_PROPENSITY_CLIP,
    NuisanceScreenConfig,
    OutcomeType,
    PipelineConfig,
    PolicyKind,
)
from causal_pipeline.learner_defaults import materialize_propensity_learner
from causal_pipeline.learner_defaults import (
    materialize_nuisance_screen_outcome_learners,
    materialize_nuisance_screen_propensity,
)
from causal_pipeline.learner_resolve import materialize_sensitivity_outcome
from causal_pipeline.learner_tuning import resolve_cate_spec_for_outer_train
from causal_pipeline.crossfit import cross_fit_predictions, cross_fit_splits, dataset_groups
from causal_pipeline.data import CausalDataset, DataSplitter, minimum_groups_for_crossfit
from causal_pipeline.diagnostics import OverlapPreprocessor
from causal_pipeline.imputed import (
    CompletedAnalysis,
    estimation_sample_size,
    pooled_cate_evaluation,
    write_pooled_ate,
    write_pooled_balance,
    write_pooled_cate_means,
    write_pooled_cate_predictions,
    write_pooled_curves,
    write_pooled_subgroups,
    write_pooled_transport,
)
from causal_pipeline.pooling import pool_nuisance_screens
from causal_pipeline.imputation import (
    EXCLUDED_IMPUTATION_SEED_OFFSET,
    CovariateImputer,
    complete_covariates,
    covariate_columns,
    imputation_warnings,
    miceforest_impute,
    missingness_report,
    observed_predictors,
    require_selection_observed,
    validate_missingness,
)
from causal_pipeline.estimator_identity import (
    assign_estimator_ids,
    attach_estimator_metadata,
    estimator_config_columns,
)
from causal_pipeline.evaluation import CATEEvaluator, ContrastEvaluation
from causal_pipeline.figures import (
    drop_nonfinite_effect_estimates,
    plot_interval_estimates,
    plot_point_estimates,
)
from causal_pipeline.nuisance_metrics import log_nuisance_metric, outcome_metric_from_predictions
from causal_pipeline.nuisance_screen import run_nuisance_screen
from causal_pipeline.parallel import (
    background_priority,
    imputation_draws_are_parallel,
    pool_workers,
    run_map,
    worker_library_jobs,
)
from causal_pipeline.policy import (
    POLICY_DATA_SPLIT_TEST,
    FittedPolicy,
    PolicyService,
)
from causal_pipeline.policy_figures import (
    POLICY_TREE_FIGURE_NAME,
    policy_slice_slug,
    save_policy_slice_figure,
)
from causal_pipeline.results import ResultStore
from causal_pipeline.selection import (
    plot_tipping_point,
    split_included_population,
    transport_included_effects,
    unseen_population_bound,
)
from causal_pipeline.subgroup import available_subgroup_splits, subgroup_split_table
from causal_pipeline.subgroup_figures import SUBGROUP_SPLITS_FIGURE_NAME, plot_subgroup_splits_table
from causal_pipeline.logging_config import configure_package_runtime
from causal_pipeline.utils import read_dataframe, write_dataframe

logger = logging.getLogger(__name__)

ATE_POOL_COLUMNS = (
    "within_variance",
    "between_variance",
    "total_variance",
    "degrees_of_freedom",
    "relative_increase",
    "fraction_missing_information",
    "imputations",
)


def pooled_estimate_columns(estimate_row: pd.Series) -> dict[str, object]:
    return {
        column: estimate_row[column]
        for column in ATE_POOL_COLUMNS
        if column in estimate_row.index
    }


ATE_INTERVAL_XLABEL = "Average treatment effect"
CATE_MEAN_XLABEL = "Mean conditional treatment effect"
DML_ATE_KINDS = frozenset(
    {
        ATEKind.DML_IRM,
        ATEKind.DML_PLR,
        ATEKind.DML_APOS,
    }
)
IN_PROCESS_CATE_KINDS = frozenset(
    {
        CATEKind.TARNET,
        CATEKind.CFRNET,
        CATEKind.DRAGONNET,
    }
)

R_EXTRA_INSTALL = 'pip install "medml[r]"'


def ensure_pipeline_runtime_support(config: PipelineConfig) -> None:
    """Fail before data work when the config needs optional packages that are missing."""
    for spec in config.cate_estimators or []:
        ensure_cate_spec_supported(spec=spec)
    if config.policy is not None and any(method.kind == PolicyKind.MOB for method in config.policy):
        if importlib.util.find_spec("rpy2") is None:
            raise ImportError(f"MOB requires optional dependency rpy2. Install with: {R_EXTRA_INSTALL}")


def predict_cate_fold(
    train: CausalDataset,
    held_out: CausalDataset,
    spec: CATEEstimatorSpec,
    outcome_records: list[tuple[np.ndarray, np.ndarray, object]] | None = None,
    tune_estimators: bool = False,
    forest_n_jobs: int | None = None,
) -> pd.DataFrame:
    fold_spec = resolve_cate_spec_for_outer_train(
        train=train,
        spec=spec,
        tune=tune_estimators,
    )
    estimator = create_cate_estimator(
        spec=fold_spec,
        data=train,
        forest_n_jobs=forest_n_jobs,
    )
    estimator.fit(data=train)
    if outcome_records is not None:
        predicted = estimator.factual_outcome(held_out)
        if predicted is not None:
            outcome_records.append(
                (
                    held_out.outcome_series.to_numpy(dtype=float),
                    np.asarray(predicted, dtype=float),
                    estimator.logged_outcome_model(),
                )
            )
    return estimator.predict_effects(data=held_out)


def cate_fold_predictor(
    spec: CATEEstimatorSpec,
    outcome_records: list[tuple[np.ndarray, np.ndarray, object]] | None = None,
    tune_estimators: bool = False,
    forest_n_jobs: int | None = None,
) -> Callable[[CausalDataset, CausalDataset], pd.DataFrame]:
    def predict_fold(train: CausalDataset, held_out: CausalDataset) -> pd.DataFrame:
        return predict_cate_fold(
            train=train,
            held_out=held_out,
            spec=spec,
            outcome_records=outcome_records,
            tune_estimators=tune_estimators,
            forest_n_jobs=forest_n_jobs,
        )

    return predict_fold


class IndependentFitKind(StrEnum):
    SCREEN = "screen"
    ATE = "ate"
    CATE = "cate"
    ROBUST_SCORES = "robust_scores"


class ImputedDraw(ArbitraryTypesModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    config: PipelineConfig
    df_included: pd.DataFrame
    df_excluded: pd.DataFrame | None
    results_root: str
    parallel: bool


class IndependentFit(ArbitraryTypesModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    kind: IndependentFitKind
    results_root: Path
    dataset: CausalDataset
    estimator_id: str
    ate_spec: ATEEstimatorSpec | None
    cate_spec: CATEEstimatorSpec | None
    tune_ate: bool
    tune_cate: bool
    cate_folds: int
    split_random_state: int
    forest_n_jobs: int
    screen: NuisanceScreenConfig | None
    propensity_learner: BaseEstimator | None
    outcome_learners: tuple[BaseEstimator, ...]
    evaluator: CATEEvaluator | None
    held_out: CausalDataset | None = None
    held_out_positions: np.ndarray | None = None


class IndependentFitResult(ArbitraryTypesModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    kind: IndependentFitKind
    estimator_id: str
    ate_model: BaseATEEstimator | None
    ate_estimate: pd.DataFrame | None
    cate_predictions: pd.DataFrame | None
    robust_scores: pd.DataFrame | None
    held_out_positions: np.ndarray | None = None
    outcome_observed: np.ndarray | None = None
    outcome_predicted: np.ndarray | None = None
    outcome_learner_name: str | None = None


def fit_imputed_draw(draw: ImputedDraw) -> CompletedAnalysis:
    """Fit one completed imputation on its own results directory."""
    logger.info("Fitting imputation draw under %s...", draw.results_root)
    store = ResultStore(config=draw.config, root=Path(draw.results_root))
    pipeline = CausalPipeline(config=draw.config, results=store)
    completed = pipeline.run_completed(
        df_included=draw.df_included,
        df_excluded=draw.df_excluded,
        results=store,
        fit_policy=False,
        fit_sensitivity=False,
        write_unseen_bounds=False,
        parallel=draw.parallel,
    )
    logger.info("...Finished imputation draw under %s", draw.results_root)
    return completed


def fit_ate_estimator(
    estimator_id: str,
    spec: ATEEstimatorSpec,
    train: CausalDataset,
    tune: bool,
) -> BaseATEEstimator:
    logger.info("Fitting ATE estimator %s...", estimator_id)
    logger.debug(
        "ATE sample: %s rows, %s columns.",
        len(train.df),
        train.X_adjustment.shape[1],
    )
    logger.debug(
        "Treatment balance (%s): %s.",
        train.treatment,
        train.treatment_series.value_counts().to_dict(),
    )
    estimator = initialize_ate_estimator(spec=spec, data=train)
    estimator.fit(train, tune_estimators=tune)
    logger.info("...Finished fitting ATE estimator %s", estimator_id)
    return estimator


def cross_fit_one_cate(
    estimation: CausalDataset,
    estimator_id: str,
    spec: CATEEstimatorSpec,
    n_folds: int,
    random_state: int,
    tune: bool,
    results_root: Path,
    forest_n_jobs: int,
) -> pd.DataFrame:
    logger.info("Cross-fitting CATE estimator %s...", estimator_id)
    outcome_records: list[tuple[np.ndarray, np.ndarray, object]] = []
    binary_outcome = estimation.outcome_type == OutcomeType.BINARY
    df_effects = cross_fit_predictions(
        dataset=estimation,
        n_folds=n_folds,
        random_state=random_state,
        predict_fold=cate_fold_predictor(
            spec,
            outcome_records,
            tune_estimators=tune,
            forest_n_jobs=forest_n_jobs,
        ),
    )
    log_pooled_factual_outcome(
        records=outcome_records,
        role=f"CATE/{estimator_id} outcome",
        binary_outcome=binary_outcome,
        metric_prefix="OOF",
    )
    write_dataframe(
        results_root / "cate" / estimator_id / "crossfit_predictions.parquet",
        df_effects,
    )
    logger.info("...Finished cross-fitting CATE estimator %s", estimator_id)
    return df_effects


def cate_map_width(spec: CATEEstimatorSpec, parallel: bool, n_folds: int) -> int:
    """Fold count dispatched as separate tasks. Neural fits stay in one process."""
    if spec.kind in IN_PROCESS_CATE_KINDS:
        return 1
    if pool_workers(parallel=parallel, n_tasks=n_folds) == 1:
        return 1
    return n_folds


def fit_cate_fold_job(job: IndependentFit) -> IndependentFitResult:
    if job.cate_spec is None or job.held_out is None or job.held_out_positions is None:
        raise ValueError("A CATE fold is missing its spec or held-out rows.")
    outcome_records: list[tuple[np.ndarray, np.ndarray, object]] = []
    predictions = predict_cate_fold(
        train=job.dataset,
        held_out=job.held_out,
        spec=job.cate_spec,
        outcome_records=outcome_records,
        tune_estimators=job.tune_cate,
        forest_n_jobs=job.forest_n_jobs,
    )
    observed = None
    predicted = None
    learner_label = None
    if outcome_records:
        observed, predicted, logged_model = outcome_records[0]
        learner_label = type(logged_model).__name__
    return IndependentFitResult(
        kind=job.kind,
        estimator_id=job.estimator_id,
        ate_model=None,
        ate_estimate=None,
        cate_predictions=predictions,
        robust_scores=None,
        held_out_positions=job.held_out_positions,
        outcome_observed=observed,
        outcome_predicted=predicted,
        outcome_learner_name=learner_label,
    )


def execute_independent_fit(job: IndependentFit) -> IndependentFitResult:
    if job.kind == IndependentFitKind.SCREEN:
        if job.screen is None or job.propensity_learner is None:
            raise ValueError("A nuisance screen fit is missing its learners.")
        run_nuisance_screen(
            dataset=job.dataset,
            screen=job.screen,
            results_root=job.results_root,
            propensity_learner=job.propensity_learner,
            outcome_learners=list(job.outcome_learners),
        )
        return IndependentFitResult(
            kind=job.kind,
            estimator_id=job.estimator_id,
            ate_model=None,
            ate_estimate=None,
            cate_predictions=None,
            robust_scores=None,
        )
    if job.kind == IndependentFitKind.ATE:
        if job.ate_spec is None:
            raise ValueError("An ATE fit is missing its spec.")
        model = fit_ate_estimator(
            estimator_id=job.estimator_id,
            spec=job.ate_spec,
            train=job.dataset,
            tune=job.tune_ate,
        )
        df_estimate = attach_estimator_metadata(
            model.estimate(),
            estimator_id=job.estimator_id,
            spec=job.ate_spec,
        )
        write_dataframe(job.results_root / "ate" / job.estimator_id / "estimate.csv", df_estimate)
        return IndependentFitResult(
            kind=job.kind,
            estimator_id=job.estimator_id,
            ate_model=model,
            ate_estimate=df_estimate,
            cate_predictions=None,
            robust_scores=None,
        )
    if job.kind == IndependentFitKind.CATE:
        if job.cate_spec is None:
            raise ValueError("A CATE fit is missing its spec.")
        if job.held_out is not None:
            return fit_cate_fold_job(job)
        predictions = cross_fit_one_cate(
            estimation=job.dataset,
            estimator_id=job.estimator_id,
            spec=job.cate_spec,
            n_folds=job.cate_folds,
            random_state=job.split_random_state,
            tune=job.tune_cate,
            results_root=job.results_root,
            forest_n_jobs=job.forest_n_jobs,
        )
        return IndependentFitResult(
            kind=job.kind,
            estimator_id=job.estimator_id,
            ate_model=None,
            ate_estimate=None,
            cate_predictions=predictions,
            robust_scores=None,
        )
    if job.kind != IndependentFitKind.ROBUST_SCORES:
        raise ValueError(f"Unknown independent fit: {job.kind}.")
    if job.evaluator is None:
        raise ValueError("Robust-score cross-fitting is missing its evaluator.")
    return IndependentFitResult(
        kind=job.kind,
        estimator_id=job.estimator_id,
        ate_model=None,
        ate_estimate=None,
        cate_predictions=None,
        robust_scores=job.evaluator.build_robust_scores(dataset=job.dataset),
    )


def cate_fit_job(
    estimation: CausalDataset,
    results_root: Path,
    estimator_id: str,
    spec: CATEEstimatorSpec,
    tune_estimators: bool,
    cate_folds: int,
    split_random_state: int,
    library_jobs: int,
    held_out: CausalDataset | None,
    held_out_positions: np.ndarray | None,
) -> IndependentFit:
    return IndependentFit(
        kind=IndependentFitKind.CATE,
        results_root=results_root,
        dataset=estimation,
        estimator_id=estimator_id,
        ate_spec=None,
        cate_spec=spec,
        tune_ate=False,
        tune_cate=tune_estimators,
        cate_folds=cate_folds,
        split_random_state=split_random_state,
        forest_n_jobs=library_jobs,
        screen=None,
        propensity_learner=None,
        outcome_learners=(),
        evaluator=None,
        held_out=held_out,
        held_out_positions=held_out_positions,
    )


def independent_fit_jobs(
    estimation: CausalDataset,
    results_root: Path,
    screen: NuisanceScreenConfig | None,
    ate_runs: list[tuple[str, ATEEstimatorSpec]],
    cate_runs: list[tuple[str, CATEEstimatorSpec]],
    tune_estimators: bool,
    cate_folds: int,
    split_random_state: int,
    evaluator: CATEEvaluator | None,
    parallel: bool,
) -> list[IndependentFit]:
    fold_widths = [cate_map_width(spec=spec, parallel=parallel, n_folds=cate_folds) for _, spec in cate_runs]
    n_tasks = len(ate_runs) + sum(fold_widths)
    if screen is not None:
        n_tasks += 1
    if evaluator is not None:
        n_tasks += 1
    library_jobs = worker_library_jobs(parallel=parallel, n_tasks=n_tasks)
    fold_splits = None
    if any(width > 1 for width in fold_widths):
        fold_splits = cross_fit_splits(
            features=np.arange(len(estimation.df)),
            treatment=estimation.treatment_series,
            n_folds=cate_folds,
            random_state=split_random_state,
            groups=dataset_groups(estimation),
        )
    jobs: list[IndependentFit] = []
    if screen is not None:
        jobs.append(
            IndependentFit(
                kind=IndependentFitKind.SCREEN,
                results_root=results_root,
                dataset=estimation,
                estimator_id="",
                ate_spec=None,
                cate_spec=None,
                tune_ate=False,
                tune_cate=False,
                cate_folds=cate_folds,
                split_random_state=split_random_state,
                forest_n_jobs=library_jobs,
                screen=screen,
                propensity_learner=materialize_nuisance_screen_propensity(
                    screen.propensity_learner,
                    random_state=screen.random_state,
                ),
                outcome_learners=tuple(
                    materialize_nuisance_screen_outcome_learners(
                        screen.outcome_learner,
                        outcome_type=estimation.outcome_type,
                        random_state=screen.random_state,
                    )
                ),
                evaluator=None,
            )
        )
    for estimator_id, spec in ate_runs:
        jobs.append(
            IndependentFit(
                kind=IndependentFitKind.ATE,
                results_root=results_root,
                dataset=estimation,
                estimator_id=estimator_id,
                ate_spec=spec,
                cate_spec=None,
                tune_ate=tune_estimators and spec.kind not in DML_ATE_KINDS,
                tune_cate=False,
                cate_folds=cate_folds,
                split_random_state=split_random_state,
                forest_n_jobs=library_jobs,
                screen=None,
                propensity_learner=None,
                outcome_learners=(),
                evaluator=None,
            )
        )
    for (estimator_id, spec), width in zip(cate_runs, fold_widths, strict=True):
        if width == 1 or fold_splits is None:
            jobs.append(
                cate_fit_job(
                    estimation=estimation,
                    results_root=results_root,
                    estimator_id=estimator_id,
                    spec=spec,
                    tune_estimators=tune_estimators,
                    cate_folds=cate_folds,
                    split_random_state=split_random_state,
                    library_jobs=library_jobs,
                    held_out=None,
                    held_out_positions=None,
                )
            )
            continue
        for train_positions, held_out_positions in fold_splits:
            jobs.append(
                cate_fit_job(
                    estimation=estimation.subset(train_positions),
                    results_root=results_root,
                    estimator_id=estimator_id,
                    spec=spec,
                    tune_estimators=tune_estimators,
                    cate_folds=cate_folds,
                    split_random_state=split_random_state,
                    library_jobs=library_jobs,
                    held_out=estimation.subset(held_out_positions),
                    held_out_positions=held_out_positions,
                )
            )
    if evaluator is not None:
        jobs.append(
            IndependentFit(
                kind=IndependentFitKind.ROBUST_SCORES,
                results_root=results_root,
                dataset=estimation,
                estimator_id="",
                ate_spec=None,
                cate_spec=None,
                tune_ate=False,
                tune_cate=False,
                cate_folds=cate_folds,
                split_random_state=split_random_state,
                forest_n_jobs=library_jobs,
                screen=None,
                propensity_learner=None,
                outcome_learners=(),
                evaluator=evaluator,
            )
        )
    return jobs


def collect_independent_fits(
    pieces: list[IndependentFitResult],
    row_count: int,
    results_root: Path,
    binary_outcome: bool,
) -> tuple[
    dict[str, BaseATEEstimator],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame | None,
]:
    ate_models: dict[str, BaseATEEstimator] = {}
    ate_by_estimator: dict[str, pd.DataFrame] = {}
    cate_by_estimator: dict[str, pd.DataFrame] = {}
    fold_pieces: dict[str, list[IndependentFitResult]] = {}
    robust_scores = None
    for piece in pieces:
        if piece.kind == IndependentFitKind.ATE:
            if piece.ate_model is None or piece.ate_estimate is None:
                raise RuntimeError(f"ATE fit {piece.estimator_id} returned no estimate.")
            ate_models[piece.estimator_id] = piece.ate_model
            ate_by_estimator[piece.estimator_id] = piece.ate_estimate
        elif piece.kind == IndependentFitKind.CATE and piece.held_out_positions is not None:
            fold_pieces.setdefault(piece.estimator_id, []).append(piece)
        elif piece.kind == IndependentFitKind.CATE:
            if piece.cate_predictions is None:
                raise RuntimeError(f"CATE fit {piece.estimator_id} returned no predictions.")
            cate_by_estimator[piece.estimator_id] = piece.cate_predictions
        elif piece.kind == IndependentFitKind.ROBUST_SCORES:
            robust_scores = piece.robust_scores
    for estimator_id, folds in fold_pieces.items():
        if estimator_id in cate_by_estimator:
            raise RuntimeError(f"CATE fit {estimator_id} returned both a full fit and fold fits.")
        cate_by_estimator[estimator_id] = publish_cate_folds(
            estimator_id=estimator_id,
            folds=folds,
            row_count=row_count,
            results_root=results_root,
            binary_outcome=binary_outcome,
        )
    return ate_models, ate_by_estimator, cate_by_estimator, robust_scores


def publish_cate_folds(
    estimator_id: str,
    folds: list[IndependentFitResult],
    row_count: int,
    results_root: Path,
    binary_outcome: bool,
) -> pd.DataFrame:
    predictions = combine_cate_folds(folds=folds, row_count=row_count)
    outcome_records = [
        (fold.outcome_observed, fold.outcome_predicted, fold.outcome_learner_name)
        for fold in folds
        if fold.outcome_observed is not None
        and fold.outcome_predicted is not None
        and fold.outcome_learner_name is not None
    ]
    log_pooled_factual_outcome(
        records=outcome_records,
        role=f"CATE/{estimator_id} outcome",
        binary_outcome=binary_outcome,
        metric_prefix="OOF",
    )
    write_dataframe(
        results_root / "cate" / estimator_id / "crossfit_predictions.parquet",
        predictions,
    )
    return predictions


def combine_cate_folds(folds: list[IndependentFitResult], row_count: int) -> pd.DataFrame:
    if not folds or folds[0].cate_predictions is None:
        raise RuntimeError("CATE fold fits returned no predictions.")
    combined = pd.DataFrame(
        index=np.arange(row_count),
        columns=list(folds[0].cate_predictions.columns),
        dtype=float,
    )
    filled = np.zeros(row_count, dtype=bool)
    for fold in folds:
        if fold.cate_predictions is None or fold.held_out_positions is None:
            raise RuntimeError("A CATE fold is missing predictions or positions.")
        positions = np.asarray(fold.held_out_positions)
        block = fold.cate_predictions.to_numpy()
        if len(block) != len(positions):
            raise ValueError("Fold prediction length does not match the held-out rows.")
        combined_values = combined.to_numpy(copy=True)
        combined_values[positions] = block
        combined = pd.DataFrame(combined_values, index=combined.index, columns=combined.columns)
        filled[positions] = True
    if not filled.all():
        raise RuntimeError("Cross-fitting left some rows without a prediction.")
    return combined.reset_index(drop=True)


def log_pooled_factual_outcome(
    records: list[tuple[np.ndarray, np.ndarray, object]],
    role: str,
    binary_outcome: bool,
    metric_prefix: str,
) -> None:
    if records:
        outcome = np.concatenate([observed for observed, _, _ in records])
        predicted = np.concatenate([values for _, values, _ in records])
        metric = outcome_metric_from_predictions(outcome, predicted, binary_outcome)
        if metric is not None:
            log_nuisance_metric(role, records[0][2], f"{metric_prefix} {metric.name}", metric.value)


class CausalPipeline:
    """End-to-end observational causal analysis pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        splitter: DataSplitter | None = None,
        overlap_preprocessor: OverlapPreprocessor | None = None,
        sensitivity_analyzer: ATESensitivityAnalyzer | None = None,
        evaluator: CATEEvaluator | None = None,
        policy_service: PolicyService | None = None,
        results: ResultStore | None = None,
        imputer: CovariateImputer | None = None,
    ) -> None:
        self.config = config
        self.splitter = splitter or DataSplitter()
        self.results = results or ResultStore(config=config)
        self.imputer = miceforest_impute if imputer is None else imputer
        self.overlap = overlap_preprocessor or OverlapPreprocessor(
            overlap=config.overlap,
            minimum_folds=minimum_groups_for_crossfit(config),
        )
        if sensitivity_analyzer is not None:
            self.sensitivity_analyzer = sensitivity_analyzer
        elif config.sensitivity is not None:
            self.sensitivity_analyzer = ATESensitivityAnalyzer(sensitivity=config.sensitivity)
        else:
            self.sensitivity_analyzer = None
        if evaluator is not None:
            self.evaluator = evaluator
        elif config.cate_evaluation is not None:
            self.evaluator = CATEEvaluator(evaluation=config.cate_evaluation)
        else:
            self.evaluator = None
        if policy_service is not None:
            self.policy_service = policy_service
        elif config.policy is not None:
            self.policy_service = PolicyService(
                methods=config.policy,
                bootstrap_samples=config.policy_bootstrap_samples,
                random_state=config.policy_random_state,
            )
        else:
            self.policy_service = None
        ensure_pipeline_runtime_support(config=config)

    def run(self, df_input: pd.DataFrame) -> None:
        """Fit the included sample once, or once per imputed covariate draw.

        Imputation off writes every estimate under the configured results directory.
        Imputation on writes each draw under results/imputation/<m> and the pooled
        estimates at the results root. Policy trees, omitted-variable sensitivity,
        and unit-level conditional effect intervals are omitted from that pooled output.
        The run executes at lower CPU priority so the rest of the machine stays responsive.
        """
        with background_priority():
            self._run(df_input)

    def _run(self, df_input: pd.DataFrame) -> None:
        configure_package_runtime()
        logger.info("Starting causal pipeline run...")
        logger.debug("Input cohort: %s rows, %s columns.", len(df_input), df_input.shape[1])
        sensitivity = self.config.sensitivity
        if sensitivity is not None and sensitivity.selection_column is not None:
            require_selection_observed(frame=df_input, column=sensitivity.selection_column)
        df_included, df_excluded = self.analysis_sample(df_input)
        validate_missingness(
            analysis_frame=df_included,
            excluded_frame=df_excluded,
            config=self.config,
        )
        for message in imputation_warnings(self.config):
            warnings.warn(message, UserWarning, stacklevel=2)
        if self.config.imputation.enabled:
            self.run_imputed(df_included=df_included, df_excluded=df_excluded)
        else:
            self.run_completed(
                df_included=df_included,
                df_excluded=df_excluded,
                results=self.results,
                fit_policy=bool(self.config.policy),
                fit_sensitivity=self.config.sensitivity is not None,
                write_unseen_bounds=True,
                parallel=self.config.parallel,
            )
        logger.info("...Finished causal pipeline run")

    def run_imputed(self, df_included: pd.DataFrame, df_excluded: pd.DataFrame | None) -> None:
        """Impute covariates, fit each completed draw, then pool at the results root."""
        imputation = self.config.imputation
        covariates = covariate_columns(self.config.data)
        logger.info(
            "Imputing %s covariate draws (%s iterations each)...",
            imputation.datasets,
            imputation.iterations,
        )
        self.results.write_dataframe(
            "imputation/missingness.csv",
            missingness_report(frame=df_included, covariates=covariates),
        )
        outcome_treatment = [self.config.data.treatment, self.config.data.outcome]
        df_draws = complete_covariates(
            frame=df_included,
            covariates=covariates,
            predictors=outcome_treatment,
            datasets=imputation.datasets,
            iterations=imputation.iterations,
            random_state=imputation.random_state,
            imputer=self.imputer,
        )
        df_excluded_draws = None
        if df_excluded is not None:
            df_excluded_draws = complete_covariates(
                frame=df_excluded,
                covariates=covariates,
                predictors=observed_predictors(frame=df_excluded, candidates=outcome_treatment),
                datasets=imputation.datasets,
                iterations=imputation.iterations,
                random_state=imputation.random_state + EXCLUDED_IMPUTATION_SEED_OFFSET,
                imputer=self.imputer,
            )
        draw_parallel = imputation_draws_are_parallel(
            parallel=self.config.parallel,
            n_draws=len(df_draws),
        )
        inner_parallel = self.config.parallel and not draw_parallel
        draws = [
            ImputedDraw(
                config=self.config,
                df_included=df_draw,
                df_excluded=None if df_excluded_draws is None else df_excluded_draws[draw_index],
                results_root=str(self.results.root / "imputation" / str(draw_index)),
                parallel=inner_parallel,
            )
            for draw_index, df_draw in enumerate(df_draws)
        ]
        completed_runs = run_map(fit_imputed_draw, draws, parallel=draw_parallel)
        logger.info("...Finished fitting %s imputation draws", len(draws))
        logger.info("Pooling imputation draws at the results root...")
        self.publish_imputed(completed_runs)
        logger.info("...Finished pooling imputation draws")

    def publish_imputed(self, completed_runs: list[CompletedAnalysis]) -> None:
        """Write pooled estimates at the results root. Per-draw files stay under imputation/<m>."""
        sample_size = estimation_sample_size(completed_runs)
        ate_by_estimator = write_pooled_ate(
            config=self.config,
            results=self.results,
            completed_runs=completed_runs,
            sample_size=sample_size,
        )
        cate_by_estimator = write_pooled_cate_predictions(
            results=self.results,
            completed_runs=completed_runs,
        )
        mean_cate_by_estimator = write_pooled_cate_means(
            config=self.config,
            results=self.results,
            completed_runs=completed_runs,
            cate_by_estimator=cate_by_estimator,
        )
        cate_evaluation = pooled_cate_evaluation(
            completed_runs=completed_runs,
            sample_size=sample_size,
        )
        self.persist_cate_evaluation(cate_evaluation=cate_evaluation, results=self.results)
        write_pooled_curves(
            results=self.results,
            completed_runs=completed_runs,
            cate_evaluation=cate_evaluation,
        )
        write_pooled_balance(config=self.config, results=self.results, completed_runs=completed_runs)
        write_pooled_transport(results=self.results, completed_runs=completed_runs)
        pool_nuisance_screens(
            imputation_roots=[run.results_root for run in completed_runs],
            destination=self.results.root,
        )
        write_pooled_subgroups(
            config=self.config,
            results=self.results,
            completed_runs=completed_runs,
            sample_size=sample_size,
        )
        if self.config.sensitivity is not None and ate_by_estimator:
            df_estimation = read_dataframe(completed_runs[0].results_root / "data" / "estimation.parquet")
            self.run_selection_analyses(
                estimation=CausalDataset(data=self.config.data, df=df_estimation),
                df_excluded=None,
                ate_by_estimator=ate_by_estimator,
                write_unseen_bounds=True,
                results=self.results,
            )
        df_ate_summary = self.build_ate_summary(
            ate_by_estimator,
            {},
            dict(assign_estimator_ids(self.config.ate_estimators or [])),
        )
        df_cate_summary = self.build_cate_summary(
            mean_cate_by_estimator,
            cate_evaluation,
            dict(assign_estimator_ids(self.config.cate_estimators or [])),
        )
        self.results.write_summaries(
            ate_summary=df_ate_summary,
            cate_summary=df_cate_summary,
            policy_summary=None,
        )
        self.write_summary_figures(
            ate_summary=df_ate_summary,
            cate_summary=df_cate_summary,
            results=self.results,
        )
        logger.info(
            "Pooled %s imputations into %s",
            len(completed_runs),
            self.results.root,
        )

    def run_completed(
        self,
        df_included: pd.DataFrame,
        df_excluded: pd.DataFrame | None,
        results: ResultStore,
        fit_policy: bool,
        fit_sensitivity: bool,
        write_unseen_bounds: bool,
        parallel: bool,
    ) -> CompletedAnalysis:
        """Fit ATE, CATE, diagnostics, and the optional policy holdout on one complete sample.

        `results` is the directory for this draw. The canonical store on the pipeline
        stays pointed at the results root, including while an imputation draw is fit.
        """
        configure_package_runtime()
        logger.info("Fitting analysis on results under %s...", results.root)
        dataset = CausalDataset(data=self.config.data, df=df_included)
        logger.debug(
            "Analysis sample: %s rows, %s columns.",
            len(dataset.df),
            dataset.X_adjustment.shape[1],
        )
        logger.debug(
            "Treatment balance (%s): %s.",
            dataset.treatment,
            dataset.treatment_series.value_counts().to_dict(),
        )
        partitions = self.splitter.split(dataset, self.config, hold_out=fit_policy)
        overlap_run = self.overlap.run(
            partitions=partitions,
            results_root=str(results.root),
        )
        partitions = overlap_run.partitions
        results.save_partitions(partitions)
        results.save_diagnostics(overlap_run.diagnostics)
        estimation = partitions.estimation
        ate_runs = assign_estimator_ids(self.config.ate_estimators or [])
        cate_runs = assign_estimator_ids(self.config.cate_estimators or [])
        ate_spec_by_id = dict(ate_runs)
        cate_spec_by_id = dict(cate_runs)
        score_cate = self.evaluator is not None and (bool(cate_runs) or fit_policy)
        jobs = independent_fit_jobs(
            estimation=estimation,
            results_root=results.root,
            screen=self.config.nuisance_screen if self.config.nuisance_screen.enabled else None,
            ate_runs=ate_runs,
            cate_runs=cate_runs,
            tune_estimators=self.config.tune_estimators,
            cate_folds=self.config.cate_crossfit_folds,
            split_random_state=self.config.split.random_state,
            evaluator=self.evaluator if score_cate else None,
            parallel=parallel,
        )
        logger.debug("Dispatching %s independent fit jobs", len(jobs))
        pieces = run_map(execute_independent_fit, jobs, parallel=parallel)
        ate_models, ate_by_estimator, cate_by_estimator, estimation_scores = collect_independent_fits(
            pieces=pieces,
            row_count=len(estimation.df),
            results_root=results.root,
            binary_outcome=estimation.outcome_type == OutcomeType.BINARY,
        )
        if ate_by_estimator:
            results.write_dataframe(
                "ate/summary.csv",
                pd.concat(list(ate_by_estimator.values()), ignore_index=True),
            )
        df_subgroups = self.write_subgroup_splits(
            estimation=estimation,
            ate_models=ate_models,
            results=results,
        )
        sensitivity_summaries = (
            self.run_ate_sensitivity(ate_models=ate_models, results=results, parallel=parallel)
            if fit_sensitivity
            else {}
        )
        df_transport = self.run_selection_analyses(
            estimation=estimation,
            df_excluded=df_excluded,
            ate_by_estimator=ate_by_estimator,
            write_unseen_bounds=write_unseen_bounds,
            results=results,
        )

        mean_cate_by_estimator = {
            estimator_id: mean_cate_table(df_predictions)
            for estimator_id, df_predictions in cate_by_estimator.items()
        }
        for estimator_id, df_mean_cate in mean_cate_by_estimator.items():
            write_dataframe(
                results.cate_dir(estimator_id) / "ate_estimate.csv",
                attach_estimator_metadata(
                    df_mean_cate,
                    estimator_id=estimator_id,
                    spec=cate_spec_by_id[estimator_id],
                ),
            )

        cate_evaluation: dict[str, list[ContrastEvaluation]] = {}
        if self.evaluator is not None and cate_by_estimator and estimation_scores is not None:
            cate_evaluation = self.evaluator.evaluate_cate_models(
                predictions=cate_by_estimator,
                robust_scores=estimation_scores,
                results_root=results.root,
                outcome_favorability=estimation.data.outcome_favorability,
                groups=dataset_groups(estimation),
            )
            self.persist_cate_evaluation(cate_evaluation=cate_evaluation, results=results)

        df_policy = None
        if fit_policy:
            if self.evaluator is None or partitions.test is None:
                raise ValueError(
                    "Policy methods require cate_evaluation and a held-out test set."
                )
            if self.policy_service is None:
                raise RuntimeError("Policy methods are configured but no policy service is available.")
            policies = self.policy_service.fit_policy_methods(
                estimation=estimation,
                predictions=cate_by_estimator,
                robust_scores=estimation_scores,
            )
            df_test_scores = self.evaluator.build_test_validation_scores(
                estimation=estimation,
                test=partitions.test,
                signal=self.config.test_validation_signal,
            )
            df_policy = self.policy_service.evaluate_policies(
                policies=policies,
                splits=[(partitions.test, df_test_scores, POLICY_DATA_SPLIT_TEST)],
            )
            results.write_dataframe("policy/summary.csv", df_policy)
            self.persist_policy_artifacts(
                policies=policies,
                df_policy=df_policy,
                estimation=estimation,
                df_estimation_scores=estimation_scores,
                test=partitions.test,
                df_test_scores=df_test_scores,
                results=results,
            )

        df_ate_summary = self.build_ate_summary(
            ate_by_estimator,
            sensitivity_summaries,
            ate_spec_by_id,
        )
        df_cate_summary = self.build_cate_summary(
            mean_cate_by_estimator,
            cate_evaluation,
            cate_spec_by_id,
        )
        results.write_summaries(
            ate_summary=df_ate_summary,
            cate_summary=df_cate_summary,
            policy_summary=df_policy,
        )
        self.write_summary_figures(
            ate_summary=df_ate_summary,
            cate_summary=df_cate_summary,
            results=results,
        )
        logger.info("...Finished analysis on results under %s", results.root)
        group_name = estimation.group_id
        return CompletedAnalysis(
            ate_by_estimator=ate_by_estimator,
            df_subgroups=df_subgroups,
            cate_by_estimator=cate_by_estimator,
            mean_cate_by_estimator=mean_cate_by_estimator,
            cate_evaluation=cate_evaluation,
            df_balance=overlap_run.diagnostics.covariate_balance,
            df_transport=df_transport,
            estimation_rows=len(estimation.df),
            cluster_count=None if group_name is None else int(estimation.df[group_name].nunique()),
            results_root=results.root,
        )

    def analysis_sample(self, df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        sensitivity = self.config.sensitivity
        df_included = df_input.copy()
        df_excluded = None
        if sensitivity is not None and sensitivity.selection_column is not None:
            data = self.config.data
            blocked = [data.outcome, data.treatment, *data.confounders, *data.effect_modifiers]
            df_included, df_excluded = split_included_population(
                df=df_input,
                selection_column=sensitivity.selection_column,
                blocked_columns=blocked,
            )
            logger.info(
                "Restricted the analysis to %s included patients and held out %s excluded patients",
                len(df_included),
                len(df_excluded),
            )
        return df_included, df_excluded

    def run_selection_analyses(
        self,
        estimation: CausalDataset,
        df_excluded: pd.DataFrame | None,
        ate_by_estimator: dict[str, pd.DataFrame],
        write_unseen_bounds: bool,
        results: ResultStore,
    ) -> pd.DataFrame | None:
        sensitivity = self.config.sensitivity
        df_transport = None
        if sensitivity is not None:
            if (
                df_excluded is not None
                and len(df_excluded) > 0
                and sensitivity.sampling_learner is not None
                and sensitivity.propensity_learner is not None
            ):
                covariates = list(estimation.X_adjustment.columns)
                missing = [column for column in covariates if column not in df_excluded.columns]
                if missing:
                    raise ValueError(f"Excluded patients are missing covariates: {missing}")
                df_transport = transport_included_effects(
                    included_covariates=estimation.X_adjustment.copy(),
                    included_treatment=estimation.treatment_series.copy(),
                    included_outcome=estimation.outcome_series.copy(),
                    excluded_covariates=df_excluded.loc[:, covariates].copy(),
                    outcome_learner=materialize_sensitivity_outcome(
                        sensitivity,
                        outcome_type=estimation.outcome_type,
                    ),
                    sampling_learner=sensitivity.sampling_learner,
                    propensity_learner=materialize_propensity_learner(
                        sensitivity.propensity_learner,
                        random_state=DEFAULT_LEARNER_RANDOM_STATE,
                    ),
                    treatment_values=estimation.treatment_values,
                    control_value=estimation.control_value,
                    propensity_clip=DEFAULT_PROPENSITY_CLIP,
                )
                results.write_dataframe("selection/transport.csv", df_transport)
            if write_unseen_bounds and sensitivity.inclusion_fraction is not None:
                outcome_lower = None
                outcome_upper = None
                if sensitivity.outcome_bounds is not None:
                    outcome_lower, outcome_upper = sensitivity.outcome_bounds
                for estimator_id, df_estimates in ate_by_estimator.items():
                    rows = []
                    for _, estimate_row in df_estimates.iterrows():
                        bound = unseen_population_bound(
                            included_effect=float(estimate_row["estimate"]),
                            inclusion_fraction=sensitivity.inclusion_fraction,
                            outcome_lower=outcome_lower,
                            outcome_upper=outcome_upper,
                        )
                        rows.append(bound.model_dump())
                        if outcome_lower is not None and outcome_upper is not None:
                            plot_tipping_point(
                                bound=bound,
                                outcome_lower=outcome_lower,
                                outcome_upper=outcome_upper,
                                save_path=str(
                                    results.ate_dir(estimator_id)
                                    / "selection"
                                    / f"tipping_{estimate_row['contrast']}.png"
                                ),
                            )
                    write_dataframe(
                        results.ate_dir(estimator_id) / "selection" / "unseen.csv",
                        pd.DataFrame(rows),
                    )
        return df_transport

    def fit_ate_estimators(
        self,
        train: CausalDataset,
        ate_runs: list[tuple[str, ATEEstimatorSpec]],
    ) -> dict[str, BaseATEEstimator]:
        return {
            estimator_id: fit_ate_estimator(
                estimator_id=estimator_id,
                spec=spec,
                train=train,
                tune=self.config.tune_estimators and spec.kind not in DML_ATE_KINDS,
            )
            for estimator_id, spec in ate_runs
        }

    def estimate_ate_models(
        self,
        ate_models: dict[str, BaseATEEstimator],
        spec_by_id: dict[str, ATEEstimatorSpec],
        results: ResultStore,
    ) -> dict[str, pd.DataFrame]:
        ate_by_estimator: dict[str, pd.DataFrame] = {}
        if ate_models:
            for estimator_id, model in ate_models.items():
                spec = spec_by_id[estimator_id]
                df_estimate = attach_estimator_metadata(
                    model.estimate(),
                    estimator_id=estimator_id,
                    spec=spec,
                )
                ate_by_estimator[estimator_id] = df_estimate
                write_dataframe(results.ate_dir(estimator_id) / "estimate.csv", df_estimate)
            df_summary = pd.concat(list(ate_by_estimator.values()), ignore_index=True)
            results.write_dataframe("ate/summary.csv", df_summary)
        return ate_by_estimator

    def write_subgroup_splits(
        self,
        estimation: CausalDataset,
        ate_models: dict[str, BaseATEEstimator],
        results: ResultStore,
    ) -> pd.DataFrame | None:
        configured = self.config.subgroup_splits
        splits = available_subgroup_splits(estimation.df.columns, configured or [])
        df_subgroups = None
        if configured and splits:
            logger.info(
                "Estimating pre-specified subgroup splits (%s cutoffs)...",
                len(splits),
            )
            df_tables = [
                subgroup_split_table(
                    dataset=estimation,
                    splits=splits,
                    scores=model.treatment_contrast_scores(),
                    estimator_id=estimator_id,
                    estimates=model.estimate(),
                )
                for estimator_id, model in ate_models.items()
            ]
            df_subgroups = pd.concat(df_tables, ignore_index=True)
            results.write_dataframe("ate/subgroup_splits.csv", df_subgroups)
            plot_subgroup_splits_table(
                table=df_subgroups,
                split_order=[(split.column, split.cutoff) for split in splits],
                save_path=str(results.root / "ate" / SUBGROUP_SPLITS_FIGURE_NAME),
            )
            logger.info("...Finished estimating pre-specified subgroup splits")
        return df_subgroups

    def run_ate_sensitivity(
        self,
        ate_models: dict[str, BaseATEEstimator],
        results: ResultStore,
        parallel: bool,
    ) -> dict[str, pd.DataFrame]:
        summaries: dict[str, pd.DataFrame] = {}
        if self.config.sensitivity is not None and ate_models:
            if self.sensitivity_analyzer is None:
                raise RuntimeError("Sensitivity is configured but no sensitivity analyzer is available.")
            analyzer = self.sensitivity_analyzer
            supported: list[tuple[str, BaseATEEstimator]] = []
            for estimator_id, model in ate_models.items():
                if estimator_supports_sensitivity(model):
                    supported.append((estimator_id, model))
                else:
                    logger.info(
                        "Skipping sensitivity for %s (DoubleML score bounds are the supported analysis)",
                        estimator_id,
                    )
                    self.clear_sensitivity_output(estimator_id, results)
            fitted_models = [
                (estimator_id, model.model)
                for estimator_id, model in supported
                if model.model is not None
            ]
            benchmark_frames = benchmark_sensitivity_models(
                models=fitted_models,
                null_effect=self.config.sensitivity.null_effect,
                level=self.config.sensitivity.confidence_level,
                parallel=parallel,
            )
            for estimator_id, model in supported:
                result = analyzer.analyze(
                    estimator=model,
                    estimator_id=estimator_id,
                    results_root=str(results.root),
                    benchmarks=benchmark_frames[estimator_id],
                )
                if result is None:
                    self.clear_sensitivity_output(estimator_id, results)
                else:
                    sensitivity_dir = results.ate_dir(estimator_id) / "sensitivity"
                    write_dataframe(sensitivity_dir / "summary.csv", result.summary)
                    write_dataframe(sensitivity_dir / "benchmarks.csv", result.benchmarks)
                    summaries[estimator_id] = result.summary
        return summaries

    def clear_sensitivity_output(self, estimator_id: str, results: ResultStore) -> None:
        """Drop a previous sensitivity folder so an unsupported estimator cannot keep an old bound."""
        sensitivity_dir = results.ate_dir(estimator_id) / "sensitivity"
        if sensitivity_dir.exists():
            shutil.rmtree(sensitivity_dir)

    def persist_cate_evaluation(
        self,
        cate_evaluation: dict[str, list[ContrastEvaluation]],
        results: ResultStore,
    ) -> None:
        for estimator_id, contrasts in cate_evaluation.items():
            for contrast_result in contrasts:
                contrast_dir = results.cate_dir(estimator_id) / contrast_result.contrast
                df_cate_evaluation = pd.DataFrame(
                    {
                        "eceth": [contrast_result.eceth],
                        "eceth_standard_error": [contrast_result.eceth_standard_error],
                        "eceth_pvalue": [contrast_result.eceth_pvalue],
                        "eceth_tolerance": [contrast_result.eceth_tolerance],
                        "rate_autoc": [contrast_result.rate_autoc],
                        "rate_autoc_standard_error": [contrast_result.rate_autoc_standard_error],
                        "rate_autoc_pvalue": [contrast_result.rate_autoc_pvalue],
                        "rate_qini": [contrast_result.rate_qini],
                        "rate_qini_standard_error": [contrast_result.rate_qini_standard_error],
                        "rate_qini_pvalue": [contrast_result.rate_qini_pvalue],
                    }
                )
                write_dataframe(contrast_dir / "evaluation.csv", df_cate_evaluation)

    def persist_policy_artifacts(
        self,
        policies: list[FittedPolicy],
        df_policy: pd.DataFrame,
        estimation: CausalDataset,
        df_estimation_scores: pd.DataFrame,
        test: CausalDataset,
        df_test_scores: pd.DataFrame,
        results: ResultStore,
    ) -> None:
        for policy in policies:
            slice_slug = policy_slice_slug(policy)
            slice_dir = results.policy_slice_dir(slice_slug)
            df_slice = df_policy.loc[
                (df_policy["method"] == policy.method)
                & (df_policy["source_cate_model"] == policy.source_cate_model)
            ]
            if not df_slice.empty:
                write_dataframe(slice_dir / "rule_performance.csv", df_slice)
            save_policy_slice_figure(
                policy=policy,
                estimation=estimation,
                train_scores=df_estimation_scores,
                test=test,
                test_scores=df_test_scores,
                save_path=str(slice_dir / POLICY_TREE_FIGURE_NAME),
            )
        logger.info("Saved policy slice tables and figures under %s", results.root / "policy")

    def write_summary_figures(
        self,
        ate_summary: pd.DataFrame,
        cate_summary: pd.DataFrame,
        results: ResultStore,
    ) -> None:
        summary_dir = results.root / "summary"
        wrote_figure = False
        ate_for_figure = drop_nonfinite_effect_estimates(
            ate_summary,
            "estimate",
            figure_name="ate_intervals.png",
        )
        if not ate_for_figure.empty:
            wrote_figure = True
            plot_interval_estimates(
                estimates=ate_for_figure,
                save_path=str(summary_dir / "ate_intervals.png"),
                estimator_column="estimator",
                contrast_column="contrast",
                estimate_column="estimate",
                lower_column="ci_lower",
                upper_column="ci_upper",
                xlabel=ATE_INTERVAL_XLABEL,
            )
        cate_for_figure = drop_nonfinite_effect_estimates(
            cate_summary,
            "mean_crossfit_cate",
            figure_name="cate_mean_effects.png",
        )
        if not cate_for_figure.empty:
            wrote_figure = True
            plot_point_estimates(
                estimates=cate_for_figure,
                save_path=str(summary_dir / "cate_mean_effects.png"),
                estimator_column="estimator",
                contrast_column="contrast",
                estimate_column="mean_crossfit_cate",
                xlabel=CATE_MEAN_XLABEL,
            )
        if wrote_figure:
            logger.info("Saved joint effect figures under %s", summary_dir)

    def build_ate_summary(
        self,
        ate_results: dict[str, pd.DataFrame],
        sensitivity_summaries: dict[str, pd.DataFrame],
        spec_by_id: dict[str, ATEEstimatorSpec],
    ) -> pd.DataFrame:
        rows = []
        for estimator_id, estimates in ate_results.items():
            config = estimator_config_columns(spec_by_id[estimator_id])
            sensitivity = (
                sensitivity_summaries[estimator_id]
                if estimator_id in sensitivity_summaries
                else None
            )
            for _, estimate_row in estimates.iterrows():
                row = {
                    "estimator": estimator_id,
                    **config,
                    "contrast": estimate_row["contrast"],
                    "estimand": estimate_row["estimand"],
                    "estimate": estimate_row["estimate"],
                    "ci_lower": estimate_row["ci_lower"],
                    "ci_upper": estimate_row["ci_upper"],
                    **pooled_estimate_columns(estimate_row),
                    "robustness_value_point": None,
                    "robustness_value_interval": None,
                    "benchmark_variable": None,
                    "benchmark_multiple_to_null": None,
                    "benchmark_multiple_to_null_interval": None,
                }
                if sensitivity is not None and not sensitivity.empty:
                    row["robustness_value_point"] = sensitivity["robustness_value_point"].iloc[0]
                    row["robustness_value_interval"] = sensitivity["robustness_value_interval"].iloc[0]
                    row["benchmark_variable"] = sensitivity["benchmark_variable"].iloc[0]
                    row["benchmark_multiple_to_null"] = sensitivity["benchmark_multiple_to_null"].iloc[0]
                    row["benchmark_multiple_to_null_interval"] = sensitivity[
                        "benchmark_multiple_to_null_interval"
                    ].iloc[0]
                rows.append(row)
        return pd.DataFrame(rows)

    def build_cate_summary(
        self,
        cate_ate_results: dict[str, pd.DataFrame],
        cate_evaluation: dict[str, list[ContrastEvaluation]],
        spec_by_id: dict[str, CATEEstimatorSpec],
    ) -> pd.DataFrame:
        rows = []
        for estimator_id, ate_table in cate_ate_results.items():
            config = estimator_config_columns(spec_by_id[estimator_id])
            contrast_evaluations = (
                cate_evaluation[estimator_id]
                if estimator_id in cate_evaluation
                else []
            )
            evaluation_by_contrast = {
                item.contrast: item for item in contrast_evaluations
            }
            for _, ate_row in ate_table.iterrows():
                contrast = ate_row["contrast"]
                evaluation = (
                    evaluation_by_contrast[contrast]
                    if contrast in evaluation_by_contrast
                    else None
                )
                rows.append(
                    {
                        "estimator": estimator_id,
                        **config,
                        "contrast": contrast,
                        "mean_crossfit_cate": ate_row["estimate"],
                        "mean_crossfit_cate_between_sd": (
                            ate_row["between_sd"] if "between_sd" in ate_row.index else None
                        ),
                        "eceth": evaluation.eceth if evaluation else None,
                        "eceth_standard_error": evaluation.eceth_standard_error if evaluation else None,
                        "eceth_pvalue": evaluation.eceth_pvalue if evaluation else None,
                        "eceth_tolerance": evaluation.eceth_tolerance if evaluation else None,
                        "rate_autoc": evaluation.rate_autoc if evaluation else None,
                        "rate_autoc_standard_error": (
                            evaluation.rate_autoc_standard_error if evaluation else None
                        ),
                        "rate_autoc_pvalue": evaluation.rate_autoc_pvalue
                        if evaluation
                        else None,
                        "rate_qini": evaluation.rate_qini if evaluation else None,
                        "rate_qini_standard_error": (
                            evaluation.rate_qini_standard_error if evaluation else None
                        ),
                        "rate_qini_pvalue": evaluation.rate_qini_pvalue
                        if evaluation
                        else None,
                    }
                )
        return pd.DataFrame(rows)

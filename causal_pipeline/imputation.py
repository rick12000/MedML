"""Covariate multiple imputation and the missingness contract."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from typing import Protocol

import pandas as pd

from causal_pipeline.config import DataConfig, PipelineConfig
from causal_pipeline.parallel import available_cores

logger = logging.getLogger(__name__)

MISSING_WHILE_DISABLED = (
    "Configured columns contain missing values. Multiple imputation is disabled."
)
TREATMENT_OUTCOME_MISSING = (
    "Multiple imputation is not applied to treatment or outcomes. "
    "Ensure there are no missing values in these columns."
)
GROUP_ID_MISSING = "group_id is not imputed. Ensure there are no missing values in this column."
SELECTION_COLUMN_MISSING = (
    "selection_column is not imputed. Ensure there are no missing values in this column."
)
MICEFOREST_INSTALL = 'pip install "medml[impute]"'
EXCLUDED_IMPUTATION_SEED_OFFSET = 1_009
LIGHTGBM_INNER_PREDICT_PATCHED = False
POLICY_SKIP_WARNING = (
    "Policy methods are skipped while multiple imputation is enabled. "
    "Policy trees are not pooled, and the estimation sample is the full cohort."
)
SENSITIVITY_SKIP_WARNING = (
    "Omitted-variable sensitivity is skipped while multiple imputation is enabled."
)
UNIT_CATE_INTERVAL_SKIP_WARNING = (
    "Unit-level conditional treatment effect intervals are skipped "
    "while multiple imputation is enabled."
)


class CovariateImputer(Protocol):
    """Complete covariate columns and return one frame per imputation."""

    def __call__(
        self,
        frame: pd.DataFrame,
        covariates: list[str],
        predictors: list[str],
        datasets: int,
        iterations: int,
        random_state: int,
    ) -> list[pd.DataFrame]:
        """Return `datasets` frames with the same index and columns as `frame`."""


def covariate_columns(data: DataConfig) -> list[str]:
    return list(dict.fromkeys([*data.confounders, *data.effect_modifiers]))


def analysis_columns(data: DataConfig) -> list[str]:
    columns = [data.outcome, data.treatment, *covariate_columns(data)]
    if data.group_id is not None:
        columns.append(data.group_id)
    return list(dict.fromkeys(columns))


def require_present_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def validate_missingness(
    analysis_frame: pd.DataFrame,
    excluded_frame: pd.DataFrame | None,
    config: PipelineConfig,
) -> None:
    """Reject missing treatment, outcome, and, when imputation is off, covariates."""
    data = config.data
    required = analysis_columns(data)
    require_present_columns(analysis_frame, required)
    if config.imputation.enabled:
        validate_imputed_missingness(
            analysis_frame=analysis_frame,
            excluded_frame=excluded_frame,
            config=config,
        )
        return
    if analysis_frame[required].isna().any().any():
        raise ValueError(MISSING_WHILE_DISABLED)


def validate_imputed_missingness(
    analysis_frame: pd.DataFrame,
    excluded_frame: pd.DataFrame | None,
    config: PipelineConfig,
) -> None:
    data = config.data
    outcome_treatment = [data.outcome, data.treatment]
    if analysis_frame[outcome_treatment].isna().any().any():
        raise ValueError(f"{TREATMENT_OUTCOME_MISSING} Columns: {outcome_treatment}.")
    if data.group_id is not None and analysis_frame[data.group_id].isna().any():
        raise ValueError(GROUP_ID_MISSING)
    for column in covariate_columns(data):
        if analysis_frame[column].isna().all():
            raise ValueError(
                f"Covariate {column} is missing in every row, so it cannot be imputed."
            )
    validate_subgroup_columns(analysis_frame, config)
    if excluded_frame is None:
        return
    require_present_columns(excluded_frame, covariate_columns(data))
    for column in covariate_columns(data):
        if excluded_frame[column].isna().all():
            raise ValueError(
                f"Covariate {column} is missing for every excluded row, so it cannot be imputed."
            )


def validate_subgroup_columns(frame: pd.DataFrame, config: PipelineConfig) -> None:
    covariates = set(covariate_columns(config.data))
    for split in config.subgroup_splits or []:
        if split.column in covariates or split.column not in frame.columns:
            continue
        if frame[split.column].isna().any():
            raise ValueError(
                f"Subgroup column {split.column} has missing values and is not a covariate. "
                "Multiple imputation fills only confounders and effect modifiers."
            )


def require_selection_observed(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"Selection column {column} is missing.")
    if frame[column].isna().any():
        raise ValueError(SELECTION_COLUMN_MISSING)


def missingness_report(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    rows = []
    row_count = len(frame)
    for column in covariates:
        missing = int(frame[column].isna().sum())
        rows.append(
            {
                "column": column,
                "n_missing": missing,
                "fraction_missing": missing / row_count if row_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def observed_predictors(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [
        column
        for column in candidates
        if column in frame.columns and not frame[column].isna().any()
    ]


def complete_covariates(
    frame: pd.DataFrame,
    covariates: list[str],
    predictors: list[str],
    datasets: int,
    iterations: int,
    random_state: int,
    imputer: CovariateImputer,
) -> list[pd.DataFrame]:
    """Impute covariates, leaving every other column as it was supplied."""
    completed = imputer(
        frame=frame.copy(),
        covariates=list(covariates),
        predictors=list(predictors),
        datasets=datasets,
        iterations=iterations,
        random_state=random_state,
    )
    if len(completed) != datasets:
        raise ValueError(
            f"The imputer returned {len(completed)} frames for {datasets} requested datasets."
        )
    return completed


def patch_lightgbm_inner_predict() -> None:
    """miceforest 6.0.5 calls a LightGBM 4.7 method positionally.

    LightGBM 4.7 made ``Booster.__inner_predict`` keyword-only. The released
    miceforest still passes the dataset index as a positional argument.
    """
    global LIGHTGBM_INNER_PREDICT_PATCHED
    if LIGHTGBM_INNER_PREDICT_PATCHED:
        return
    import lightgbm

    booster = lightgbm.Booster
    current = booster._Booster__inner_predict
    parameter = inspect.signature(current).parameters.get("data_idx")
    if parameter is None or parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
        LIGHTGBM_INNER_PREDICT_PATCHED = True
        return

    def inner_predict(self: object, data_idx: int = 0) -> object:
        return current(self, data_idx=data_idx)

    booster._Booster__inner_predict = inner_predict
    LIGHTGBM_INNER_PREDICT_PATCHED = True


def miceforest_impute(
    frame: pd.DataFrame,
    covariates: list[str],
    predictors: list[str],
    datasets: int,
    iterations: int,
    random_state: int,
) -> list[pd.DataFrame]:
    """Chained lightgbm imputation. Complete treatment and outcome stay predictors."""
    targets = [column for column in covariates if frame[column].isna().any()]
    if not targets:
        logger.info("Covariates are complete; each imputation repeats the observed sample")
        return [frame.copy() for _dataset in range(datasets)]
    if importlib.util.find_spec("miceforest") is None:
        raise ImportError(
            f"Multiple imputation requires miceforest. Install with: {MICEFOREST_INSTALL}"
        )
    import miceforest as miceforest_module

    model_columns = list(dict.fromkeys([*covariates, *predictors]))
    model_frame = frame.loc[:, model_columns].copy()
    logger.info(
        "Imputing %s datasets with miceforest (%s iterations)...",
        datasets,
        iterations,
    )
    logger.debug(
        "Imputation model frame: %s rows, %s columns.",
        len(model_frame),
        len(model_columns),
    )
    for column in predictors:
        if column not in model_frame.columns:
            continue
        if not pd.api.types.is_float_dtype(model_frame[column]):
            model_frame[column] = model_frame[column].astype("category")
    patch_lightgbm_inner_predict()
    kernel = miceforest_module.ImputationKernel(
        data=model_frame,
        num_datasets=datasets,
        random_state=random_state,
    )
    # Imputation runs in the parent. LightGBM would otherwise take every logical core.
    kernel.mice(iterations=iterations, verbose=False, num_threads=available_cores())
    completed_frames = []
    for dataset_index in range(datasets):
        completed_model = kernel.complete_data(dataset=dataset_index)
        if not completed_model.index.equals(frame.index):
            raise ValueError("The imputer returned rows that do not match the source index.")
        filled = frame.copy()
        filled.loc[:, covariates] = completed_model.loc[:, covariates]
        completed_frames.append(filled)
    logger.info("...Finished imputing %s datasets with miceforest", datasets)
    return completed_frames


def imputation_warnings(config: PipelineConfig) -> list[str]:
    if not config.imputation.enabled:
        return []
    messages = []
    if config.policy:
        messages.append(POLICY_SKIP_WARNING)
    if config.sensitivity is not None:
        messages.append(SENSITIVITY_SKIP_WARNING)
    if config.cate_estimators:
        messages.append(UNIT_CATE_INTERVAL_SKIP_WARNING)
    return messages

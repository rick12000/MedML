from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest
from xgboost import XGBRegressor

from causal_pipeline.learner_defaults import DEFAULT_XGBOOST_N_JOBS, apply_defaults
from causal_pipeline.parallel import (
    SINGLE_CORE_N_JOBS,
    WINDOWS_BELOW_NORMAL_PRIORITY,
    _windows_kernel32,
    background_priority,
    WORKER_MARKER_ENV,
    available_cores,
    pool_workers,
    worker_library_jobs,
)

from causal_pipeline.config import (
    ATEKind,
    CATEKind,
    DataConfig,
    DEFAULT_IMPUTATION_DATASETS,
    DEFAULT_SPLIT_RANDOM_STATE,
    DoubleMLATEEstimatorSpec,
    ImputationConfig,
    MetaCATEEstimatorSpec,
    PipelineConfig,
    SensitivityConfig,
    SplitConfig,
)
from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.utils import read_dataframe

PARALLEL_SPEEDUP_MULTIPLE = 1.5
CATE_KINDS = (CATEKind.S_LEARNER, CATEKind.T_LEARNER, CATEKind.X_LEARNER)


def elapsed_pipeline(frame: pd.DataFrame, config: PipelineConfig) -> float:
    started = perf_counter()
    CausalPipeline(config=config).run(frame)
    return perf_counter() - started


@pytest.mark.slow
def test_parallel_cate_folds_finish_sooner_than_serial_folds(
    require_parallel_cores: None,
    test_output_dir: Path,
    parallel_timing_sample: tuple[pd.DataFrame, DataConfig],
) -> None:
    frame, data_config = parallel_timing_sample
    elapsed: dict[bool, float] = {}
    for parallel in (True, False):
        config = PipelineConfig(
            data=data_config,
            split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
            cate_estimators=[
                MetaCATEEstimatorSpec(kind=kind, alias=kind.value, random_state=index)
                for index, kind in enumerate(CATE_KINDS)
            ],
            parallel=parallel,
            results_dir=str(test_output_dir / ("parallel" if parallel else "serial")),
        )
        elapsed[parallel] = elapsed_pipeline(frame, config)
    serial_summary = read_dataframe(test_output_dir / "serial" / "summary" / "cate_estimators.csv")
    parallel_summary = read_dataframe(test_output_dir / "parallel" / "summary" / "cate_estimators.csv")
    assert list(serial_summary["estimator"]) == [kind.value for kind in CATE_KINDS]
    assert list(parallel_summary["estimator"]) == list(serial_summary["estimator"])
    pd.testing.assert_series_equal(
        parallel_summary["mean_crossfit_cate"].reset_index(drop=True),
        serial_summary["mean_crossfit_cate"].reset_index(drop=True),
    )
    assert elapsed[False] > elapsed[True] * PARALLEL_SPEEDUP_MULTIPLE


@pytest.mark.slow
def test_parallel_imputation_draws_finish_sooner_than_serial_draws(
    require_parallel_cores: None,
    test_output_dir: Path,
    parallel_timing_sample: tuple[pd.DataFrame, DataConfig],
) -> None:
    frame, data_config = parallel_timing_sample
    elapsed: dict[bool, float] = {}
    for parallel in (True, False):
        config = PipelineConfig(
            data=data_config,
            split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
            cate_estimators=[
                MetaCATEEstimatorSpec(kind=CATEKind.T_LEARNER, alias="t_learner", random_state=0)
            ],
            imputation=ImputationConfig(enabled=True),
            parallel=parallel,
            results_dir=str(test_output_dir / ("parallel" if parallel else "serial")),
        )
        elapsed[parallel] = elapsed_pipeline(frame, config)
    for name in ("serial", "parallel"):
        draw_root = test_output_dir / name / "imputation"
        draw_dirs = [path for path in draw_root.iterdir() if path.is_dir()]
        summary = read_dataframe(test_output_dir / name / "summary" / "cate_estimators.csv")
        assert len(draw_dirs) == DEFAULT_IMPUTATION_DATASETS
        assert list(summary["estimator"]) == ["t_learner"]
    assert elapsed[False] > elapsed[True] * PARALLEL_SPEEDUP_MULTIPLE


@pytest.mark.slow
def test_parallel_sensitivity_refits_finish_sooner_than_serial_refits(
    require_parallel_cores: None,
    test_output_dir: Path,
    parallel_timing_sample: tuple[pd.DataFrame, DataConfig],
) -> None:
    frame, data_config = parallel_timing_sample
    elapsed: dict[bool, float] = {}
    for parallel in (True, False):
        config = PipelineConfig(
            data=data_config,
            split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
            ate_estimators=[DoubleMLATEEstimatorSpec(kind=ATEKind.DML_IRM, alias="dml_irm")],
            sensitivity=SensitivityConfig(null_effect=0.0),
            parallel=parallel,
            results_dir=str(test_output_dir / ("parallel" if parallel else "serial")),
        )
        elapsed[parallel] = elapsed_pipeline(frame, config)
    for name in ("serial", "parallel"):
        benchmarks = read_dataframe(
            test_output_dir / name / "ate" / "dml_irm" / "sensitivity" / "benchmarks.csv"
        )
        assert benchmarks["covariate"].tolist() == list(data_config.confounders)
    assert elapsed[False] > elapsed[True] * PARALLEL_SPEEDUP_MULTIPLE


def test_available_cores_leaves_one_logical_core_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert available_cores() == 15
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert available_cores() == 1


def test_pool_workers_never_exceed_the_core_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert pool_workers(parallel=True, n_tasks=100) == 15
    assert pool_workers(parallel=True, n_tasks=4) == 4
    assert pool_workers(parallel=False, n_tasks=100) == 1


def test_library_threads_are_pinned_inside_pool_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.delenv(WORKER_MARKER_ENV, raising=False)
    # Parent with a single task may use the whole budget.
    assert worker_library_jobs(parallel=False, n_tasks=1) == available_cores()
    # Parent with a pool gives every task one thread.
    assert worker_library_jobs(parallel=True, n_tasks=8) == SINGLE_CORE_N_JOBS
    # Inside a worker (for example a draw running its fits serially) nothing fans out.
    monkeypatch.setenv(WORKER_MARKER_ENV, "1")
    assert worker_library_jobs(parallel=False, n_tasks=1) == SINGLE_CORE_N_JOBS
    assert pool_workers(parallel=True, n_tasks=8) == 1


def test_xgboost_learners_default_to_one_thread_unless_set() -> None:
    assert apply_defaults(XGBRegressor()).get_params()["n_jobs"] == DEFAULT_XGBOOST_N_JOBS
    assert apply_defaults(XGBRegressor(n_jobs=4)).get_params()["n_jobs"] == 4


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priority classes")
def test_background_priority_lowers_then_restores_the_process_class() -> None:
    kernel32 = _windows_kernel32()
    handle = kernel32.GetCurrentProcess()
    before = kernel32.GetPriorityClass(handle)
    with background_priority():
        assert kernel32.GetPriorityClass(handle) == WINDOWS_BELOW_NORMAL_PRIORITY
    assert kernel32.GetPriorityClass(handle) == before

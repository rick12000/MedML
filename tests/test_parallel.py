from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest

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

"""Pipeline-style CATE evaluation figures on known recovery cohorts (cache/pytest/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causal_pipeline.config import DataConfig
from tests.cache_paths import artifact_path
from tests.evaluation_recovery import write_linear_meta_learner_evaluation_figures


@pytest.mark.parametrize(
    "scenario",
    ["heterogeneous_linear", "constant_effect"],
)
def test_recovery_cohorts_write_cate_evaluation_figures(
    scenario: str,
    recovery_data_config: DataConfig,
    heterogeneous_effect_cohort: tuple[pd.DataFrame, object],
    constant_effect_cohort: tuple[pd.DataFrame, float],
    test_output_dir: Path,
) -> None:
    if scenario == "heterogeneous_linear":
        frame, _truth = heterogeneous_effect_cohort
    else:
        frame, _truth = constant_effect_cohort
    results_root = artifact_path(test_output_dir, scenario)
    write_linear_meta_learner_evaluation_figures(
        frame=frame,
        data_config=recovery_data_config,
        results_root=results_root,
    )
    comparison = results_root / "cate" / "comparison" / "1_vs_0"
    assert (comparison / "toc.png").is_file()
    assert (comparison / "calibration.png").is_file()
    for learner in ("s_learner", "t_learner"):
        learner_dir = results_root / "cate" / learner / "1_vs_0"
        assert (learner_dir / "toc.png").is_file()
        assert (learner_dir / "calibration.png").is_file()

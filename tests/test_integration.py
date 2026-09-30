from __future__ import annotations

from pathlib import Path

import pytest

from causal_pipeline.config import PipelineConfig
from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.utils import read_dataframe
from generate_toy_cohort import generate_observational_cohort

INTEGRATION_RANDOM_STATE = 0


@pytest.mark.integration
@pytest.mark.parametrize("n_observations", [200, 1000])
def test_toy_cohort_pipeline_writes_summaries_to_cache(
    n_observations: int,
    integration_pipeline_config: PipelineConfig,
    integration_cache_dir: Path,
) -> None:
    df_input = generate_observational_cohort(
        n_observations,
        random_state=INTEGRATION_RANDOM_STATE,
    )
    CausalPipeline(config=integration_pipeline_config).run(df_input=df_input)

    ate_summary = read_dataframe(integration_cache_dir / "summary" / "ate_estimators.csv")
    cate_summary = read_dataframe(integration_cache_dir / "summary" / "cate_estimators.csv")
    assert ate_summary.shape[0] == 2
    assert cate_summary.shape[0] == 2
    assert {"ipw", "aipw"}.issubset(set(ate_summary["estimator"]))
    assert {"s_learner", "x_learner"}.issubset(set(cate_summary["estimator"]))
    assert ate_summary["estimate"].notna().all()
    assert cate_summary["mean_crossfit_cate"].notna().all()
    assert (integration_cache_dir / "diagnostics" / "propensity_overlap.png").is_file()
    assert (integration_cache_dir / "data" / "estimation.parquet").is_file()

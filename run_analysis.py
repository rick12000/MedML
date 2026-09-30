"""Example entry point for the binary observational analysis.

Generate the cohort first, from the repository root::

    python scripts/generate_toy_cohort.py

Then::

    python run_analysis.py

``settings.py`` is the run configuration. It turns on diagnostics, the ATE
estimators, the CATE estimators, omitted-variable and selection sensitivity,
calibration, and the policy methods. The cohort keeps patients under 65, with
covariates observed, so the selection step can transport the included effect.
"""

from __future__ import annotations

import logging
from pathlib import Path

from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.utils import read_dataframe
from settings import build_pipeline_config

logging.basicConfig(level=logging.INFO)

DATA_PATH = Path("data") / "analysis.parquet"
GENERATOR_COMMAND = "python scripts/generate_toy_cohort.py"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected the cohort at {DATA_PATH}. Generate it with: {GENERATOR_COMMAND}"
        )
    config = build_pipeline_config()
    cohort = read_dataframe(DATA_PATH)
    CausalPipeline(config).run(cohort)


if __name__ == "__main__":
    main()

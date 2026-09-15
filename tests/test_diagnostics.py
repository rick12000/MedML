import numpy as np
import pandas as pd

from causal_pipeline.data import CausalDataset
from causal_pipeline.diagnostics import DiagnosticsRunner, smd_two_groups


def test_smd_two_groups_zero_for_identical_samples() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert smd_two_groups(values, values) == 0.0


def test_diagnostics_runner_produces_balance_table(
    pipeline_config_no_policy,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    config = pipeline_config_no_policy.model_copy(
        update={
            "diagnostics": pipeline_config_no_policy.diagnostics.model_copy(
                update={"plot_propensity_overlap": False},
            ),
        },
    )
    runner = DiagnosticsRunner(config)
    dataset = CausalDataset(data=config.data, df=df_synthetic_binary)
    result = runner.run(dataset)
    assert not result.covariate_balance.empty
    assert set(result.covariate_balance.columns) >= {
        "covariate",
        "unweighted_smd",
        "weighted_smd",
    }
    assert len(result.covariate_balance) == len(config.data.confounders)

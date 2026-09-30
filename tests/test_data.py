from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_pipeline.config import DataConfig, PipelineConfig
from causal_pipeline.data import (
    CausalDataset,
    DataSplitter,
    contrast_columns,
    ipw_weights_binary,
    ipw_weights_multi,
)


def test_causal_dataset_exposes_confounder_columns(
    binary_data_config: DataConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    assert list(dataset.X_confounders.columns) == ["x1", "x2"]
    assert list(dataset.X_adjustment.columns) == ["x1", "x2"]
    assert list(dataset.X_controls.columns) == ["x2"]
    assert len(dataset.treatment_series) == len(df_synthetic_binary)


def test_causal_dataset_rejects_missing_columns(binary_data_config: DataConfig) -> None:
    df_missing_cols = pd.DataFrame({"y": [0, 1], "t": [0, 1]})
    with pytest.raises(ValueError):
        CausalDataset(data=binary_data_config, df=df_missing_cols)


def test_causal_dataset_rejects_non_binary_outcome_codes(
    binary_data_config: DataConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    df_invalid_outcome = df_synthetic_binary.copy()
    df_invalid_outcome["y"] = 2
    with pytest.raises(ValueError):
        CausalDataset(data=binary_data_config, df=df_invalid_outcome)


def test_contrast_columns_names_non_control_arms() -> None:
    columns = contrast_columns(control_value=0, treatment_values=[0, 1, 2])
    assert columns == ["1_vs_0", "2_vs_0"]


@pytest.mark.parametrize("n_observations", [50, 200])
def test_split_without_policy_partitions_cover_all_rows(
    n_observations: int,
    pipeline_config_no_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    df_input = df_synthetic_binary.iloc[:n_observations].copy()
    dataset = CausalDataset(data=pipeline_config_no_policy.data, df=df_input)
    partitions = DataSplitter().split(dataset=dataset, config=pipeline_config_no_policy)
    assert len(partitions.estimation.df) == n_observations
    assert partitions.test is None


def test_split_with_policy_allocates_test_rows(
    pipeline_config_with_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(
        data=pipeline_config_with_policy.data,
        df=df_synthetic_binary,
    )
    partitions = DataSplitter().split(dataset=dataset, config=pipeline_config_with_policy)
    total = len(df_synthetic_binary)
    assert partitions.test is not None
    assert len(partitions.estimation.df) + len(partitions.test.df) == total
    assert len(partitions.test.df) > 0


def test_ipw_weights_binary_stabilized_uses_marginal_probability() -> None:
    treatment = np.array([0.0, 0.0, 0.0, 1.0])
    propensity = np.array([0.2, 0.2, 0.2, 0.8])
    weights = ipw_weights_binary(
        treatment=treatment,
        propensity=propensity,
        stabilized=True,
    )
    assert np.isclose(weights[0], 0.75 / 0.8)
    assert np.isclose(weights[3], 0.25 / 0.8)


def test_ipw_weights_binary_shape_and_positive() -> None:
    treatment = np.array([0, 1, 1, 0])
    propensity = np.array([0.4, 0.6, 0.7, 0.3])
    weights = ipw_weights_binary(
        treatment=treatment,
        propensity=propensity,
        stabilized=False,
    )
    assert weights.shape == treatment.shape
    assert np.all(weights > 0)


def test_ipw_weights_multi_selects_arm_specific_propensity() -> None:
    treatment = np.array([0, 1])
    propensity_matrix = np.array([[0.6, 0.4], [0.3, 0.7]])
    weights = ipw_weights_multi(
        treatment=treatment,
        propensity_matrix=propensity_matrix,
        treatment_levels=[0, 1],
    )
    assert weights.shape == (2,)
    assert np.isclose(weights[0], 1.0 / 0.6)
    assert np.isclose(weights[1], 1.0 / 0.7)

import numpy as np
import pandas as pd
import pytest

from causal_pipeline.config import OutcomeType, TreatmentMode
from causal_pipeline.data import (
    CausalDataset,
    DataSplitter,
    contrast_columns,
    ipw_weights_binary,
    ipw_weights_multi,
)


def test_causal_dataset_exposes_confounder_columns(
    binary_data_config,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    assert list(dataset.X_confounders.columns) == ["x1", "x2"]
    assert len(dataset.treatment_series) == len(df_synthetic_binary)


def test_causal_dataset_rejects_missing_columns(binary_data_config) -> None:
    df_missing_cols = pd.DataFrame({"y": [0, 1], "t": [0, 1]})
    with pytest.raises(ValueError):
        CausalDataset(data=binary_data_config, df=df_missing_cols)


def test_causal_dataset_rejects_non_binary_outcome_codes(
    binary_data_config,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    df_invalid_outcome = df_synthetic_binary.copy()
    df_invalid_outcome["y"] = 2
    with pytest.raises(ValueError):
        CausalDataset(data=binary_data_config, df=df_invalid_outcome)


def test_contrast_columns_names_non_control_arms() -> None:
    columns = contrast_columns(0, [0, 1, 2])
    assert columns == ["1_vs_0", "2_vs_0"]


@pytest.mark.parametrize("n_observations", [50, 200])
def test_split_without_policy_partitions_cover_all_rows(
    n_observations: int,
    pipeline_config_no_policy,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    df_input = df_synthetic_binary.iloc[:n_observations].copy()
    dataset = CausalDataset(data=pipeline_config_no_policy.data, df=df_input)
    partitions = DataSplitter().split(dataset, pipeline_config_no_policy)
    assert len(partitions.train.df) + len(partitions.validation.df) == n_observations
    assert len(partitions.test.df) == 0
    assert list(partitions.test.df.columns) == list(df_input.columns)


def test_split_with_policy_allocates_test_rows(
    pipeline_config_with_policy,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(
        data=pipeline_config_with_policy.data,
        df=df_synthetic_binary,
    )
    partitions = DataSplitter().split(dataset, pipeline_config_with_policy)
    total = len(df_synthetic_binary)
    assert (
        len(partitions.train.df)
        + len(partitions.validation.df)
        + len(partitions.test.df)
        == total
    )
    assert len(partitions.test.df) > 0


def test_ipw_weights_binary_shape_and_positive() -> None:
    treatment = np.array([0, 1, 1, 0])
    propensity = np.array([0.4, 0.6, 0.7, 0.3])
    weights = ipw_weights_binary(treatment, propensity)
    assert weights.shape == treatment.shape
    assert np.all(weights > 0)


def test_ipw_weights_multi_selects_arm_specific_propensity() -> None:
    treatment = np.array([0, 1])
    propensity_matrix = np.array([[0.6, 0.4], [0.3, 0.7]])
    weights = ipw_weights_multi(treatment, propensity_matrix, [0, 1])
    assert weights.shape == (2,)
    assert np.isclose(weights[0], 1.0 / 0.6)
    assert np.isclose(weights[1], 1.0 / 0.7)

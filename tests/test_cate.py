import pandas as pd
import numpy as np
import pytest

from causal_pipeline.cate import create_cate_estimator, mean_cate_table
from causal_pipeline.config import CATEKind, MetaCATEEstimatorSpec
from causal_pipeline.data import CausalDataset


def test_binary_s_learner_rejects_regressor_outcome_model(
    binary_data_config,
    forest_learner,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    spec = MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER, outcome_learner=forest_learner)
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = create_cate_estimator(spec, dataset)
    with pytest.raises(TypeError):
        estimator.fit(dataset)


@pytest.mark.parametrize(
    "kind",
    [
        CATEKind.S_LEARNER,
        CATEKind.T_LEARNER,
        CATEKind.X_LEARNER,
        CATEKind.R_LEARNER,
        CATEKind.DR_LEARNER,
    ],
)
def test_binary_meta_learners_return_one_contrast_per_row(
    kind: CATEKind,
    binary_data_config,
    forest_classifier,
    forest_learner,
    logistic_learner,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    two_stage = {CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER}
    spec = MetaCATEEstimatorSpec(
        kind=kind,
        outcome_learner=forest_classifier,
        effect_learner=forest_learner if kind in two_stage else None,
        propensity_learner=logistic_learner if kind in two_stage else None,
    )
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = create_cate_estimator(spec, dataset)
    estimator.fit(dataset)
    predictions = estimator.predict_effects(dataset)
    assert predictions.shape[0] == len(df_synthetic_binary)
    assert predictions.shape[1] == 1
    ate = estimator.estimate(dataset)
    assert ate.shape[0] == 1
    assert ate["estimate"].notna().all()


def test_mean_cate_table_averages_each_contrast() -> None:
    predictions = pd.DataFrame({"1_vs_0": [0.0, 0.5, 1.0], "2_vs_0": [0.2, 0.2, 0.2]})
    table = mean_cate_table(predictions)
    assert table.shape == (2, 4)
    assert list(table["contrast"]) == ["1_vs_0", "2_vs_0"]
    assert np.isclose(table["estimate"].to_numpy(), [0.5, 0.2]).all()

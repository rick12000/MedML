from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from causal_pipeline.cate import create_cate_estimator, mean_cate_table
from causal_pipeline.config import CATEKind, DataConfig, MetaCATEEstimatorSpec
from causal_pipeline.crossfit import cross_fit_predictions
from causal_pipeline.data import CausalDataset

TWO_STAGE_META_LEARNERS = frozenset(
    {CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER}
)
CROSSFIT_FOLD_COUNT = 5


def test_binary_s_learner_rejects_regressor_outcome_model(
    binary_data_config: DataConfig,
    forest_learner: BaseEstimator,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    spec = MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER, outcome_learner=forest_learner)
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = create_cate_estimator(spec=spec, data=dataset)
    with pytest.raises(TypeError):
        estimator.fit(data=dataset)


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
    binary_data_config: DataConfig,
    forest_classifier: BaseEstimator,
    forest_learner: BaseEstimator,
    logistic_learner: BaseEstimator,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    spec = MetaCATEEstimatorSpec(
        kind=kind,
        outcome_learner=forest_classifier,
        effect_learner=forest_learner if kind in TWO_STAGE_META_LEARNERS else None,
        propensity_learner=logistic_learner if kind in TWO_STAGE_META_LEARNERS else None,
    )
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = create_cate_estimator(spec=spec, data=dataset)
    estimator.fit(data=dataset)
    predictions = estimator.predict_effects(data=dataset)
    assert predictions.shape[0] == len(df_synthetic_binary)
    assert predictions.shape[1] == 1
    ate = estimator.estimate(data=dataset)
    assert ate.shape[0] == 1
    assert ate["estimate"].notna().all()


def test_mean_cate_table_averages_each_contrast() -> None:
    predictions = pd.DataFrame({"1_vs_0": [0.0, 0.5, 1.0], "2_vs_0": [0.2, 0.2, 0.2]})
    table = mean_cate_table(predictions)
    assert table.shape == (2, 4)
    assert list(table["contrast"]) == ["1_vs_0", "2_vs_0"]
    assert np.isclose(table["estimate"].to_numpy(), [0.5, 0.2]).all()


def test_cross_fit_predictions_score_each_row_from_the_other_folds(
    binary_data_config: DataConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)

    def predict_fold(train: CausalDataset, held_out: CausalDataset) -> pd.DataFrame:
        complement_mean = float(train.outcome_series.mean())
        return pd.DataFrame({"effect": np.full(len(held_out.df), complement_mean)})

    predictions = cross_fit_predictions(
        dataset=dataset,
        n_folds=CROSSFIT_FOLD_COUNT,
        random_state=0,
        predict_fold=predict_fold,
    )
    full_mean = float(dataset.outcome_series.mean())
    assert predictions.shape == (len(df_synthetic_binary), 1)
    assert not np.isclose(predictions["effect"].to_numpy(), full_mean).all()

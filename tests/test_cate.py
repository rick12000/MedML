from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from causal_pipeline.cate import create_cate_estimator, mean_cate_table
from causal_pipeline.config import (
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DataConfig,
    MetaCATEEstimatorSpec,
)
from causal_pipeline.crossfit import cross_fit_predictions
from causal_pipeline.data import CausalDataset

TWO_STAGE_META_LEARNERS = frozenset(
    {CATEKind.X_LEARNER, CATEKind.R_LEARNER, CATEKind.DR_LEARNER}
)
CROSSFIT_FOLD_COUNT = 5
RECOVERY_FOLDS = 5
CATE_MEAN_TOLERANCE = 0.05
CATE_CORRELATION_FLOOR = 0.95
# TARNet, CFRNet, and DragonNet need the optional JAX stack and a much larger fit.
FAST_CATE_KINDS = [
    CATEKind.S_LEARNER,
    CATEKind.T_LEARNER,
    CATEKind.X_LEARNER,
    CATEKind.R_LEARNER,
    CATEKind.DR_LEARNER,
    CATEKind.CAUSAL_FOREST,
]


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


@pytest.mark.parametrize("architecture", ["linear", "forest"])
@pytest.mark.parametrize("kind", FAST_CATE_KINDS)
def test_cate_recovers_a_known_linear_effect(
    kind: CATEKind,
    architecture: str,
    recovery_data_config: DataConfig,
    heterogeneous_effect_cohort: tuple[pd.DataFrame, np.ndarray],
    linear_learner: BaseEstimator,
    logistic_learner: BaseEstimator,
    recovery_regressor: BaseEstimator,
    recovery_classifier: BaseEstimator,
) -> None:
    frame, truth = heterogeneous_effect_cohort
    dataset = CausalDataset(data=recovery_data_config, df=frame)
    outcome_learner = linear_learner if architecture == "linear" else recovery_regressor
    propensity_learner = logistic_learner if architecture == "linear" else recovery_classifier
    effect_learner = linear_learner if architecture == "linear" else recovery_regressor
    if kind == CATEKind.CAUSAL_FOREST:
        spec = CausalForestCATEEstimatorSpec(
            kind=kind,
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
        )
    else:
        spec = MetaCATEEstimatorSpec(
            kind=kind,
            outcome_learner=outcome_learner,
            effect_learner=effect_learner if kind in TWO_STAGE_META_LEARNERS else None,
            propensity_learner=propensity_learner if kind in TWO_STAGE_META_LEARNERS else None,
            n_folds=RECOVERY_FOLDS,
        )
    estimator = create_cate_estimator(spec=spec, data=dataset)
    np.random.seed(0)
    estimator.fit(data=dataset)
    predictions = estimator.predict_effects(data=dataset)
    estimated = predictions.iloc[:, 0].to_numpy(dtype=float)
    truth_mean = float(np.mean(truth))
    treated = frame.loc[frame["t"] == 1, "y"]
    control = frame.loc[frame["t"] == 0, "y"]
    unadjusted = float(treated.mean() - control.mean())
    assert predictions.shape == (len(frame), 1)
    assert abs(float(np.mean(estimated)) - truth_mean) < CATE_MEAN_TOLERANCE
    assert abs(float(np.mean(estimated)) - truth_mean) < abs(unadjusted - truth_mean)
    # A linear S-learner has no treatment-by-covariate term, so tau(x) stays flat.
    if kind == CATEKind.S_LEARNER and architecture == "linear":
        assert float(np.std(estimated)) < 0.05
    else:
        assert float(np.corrcoef(estimated, truth)[0, 1]) > CATE_CORRELATION_FLOOR
    ate = estimator.estimate(data=dataset)
    assert ate.shape[0] == 1
    assert abs(float(ate["estimate"].iloc[0]) - truth_mean) < CATE_MEAN_TOLERANCE


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

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.metrics import get_toc, rate_score

from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset
from causal_pipeline.evaluation import (
    CATEEvaluator,
    RATE_WEIGHTING_AUTOC,
    compute_eceth,
    rate_input_frame,
    eceth_hypothesis_test,
)

EVALUATION_N_OBSERVATIONS = 40
EVALUATION_BIN_COUNT = 4


def test_compute_eceth_is_zero_when_predictions_match_scores() -> None:
    tau_hat = np.array([0.0, 0.0, 1.0, 1.0])
    value = compute_eceth(tau_hat=tau_hat, gamma=tau_hat.copy(), n_bins=2)
    assert value == 0.0


def test_compute_eceth_leave_one_out_ignores_a_single_bin_outlier() -> None:
    tau_hat = np.array([0.0, 0.0, 1.0, 1.0])
    gamma = np.array([0.0, 10.0, 1.0, 1.0])
    value = compute_eceth(tau_hat=tau_hat, gamma=gamma, n_bins=2)
    assert value == 0.0


def test_compute_eceth_is_the_mean_squared_gap_when_the_bin_is_shifted() -> None:
    tau_hat = np.array([0.0, 0.0, 1.0, 1.0])
    gamma = np.ones(4)
    value = compute_eceth(tau_hat=tau_hat, gamma=gamma, n_bins=2)
    assert value == 0.5


def test_eceth_tolerance_test_rejects_a_perfectly_calibrated_model() -> None:
    tau_hat = np.array([0.0, 0.0, 1.0, 1.0])
    estimate, standard_error, p_value = eceth_hypothesis_test(
        tau_hat=tau_hat,
        gamma=tau_hat.copy(),
        n_bins=2,
        tolerance=0.01,
        bootstrap_samples=10,
        random_state=0,
    )
    assert estimate == 0.0
    assert standard_error == 0.0
    assert p_value == 0.0


def test_eceth_tolerance_test_does_not_reject_a_large_calibration_gap() -> None:
    tau_hat = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    gamma = np.ones(8)
    estimate, standard_error, p_value = eceth_hypothesis_test(
        tau_hat=tau_hat,
        gamma=gamma,
        n_bins=2,
        tolerance=0.01,
        bootstrap_samples=30,
        random_state=0,
    )
    assert estimate == 0.5
    assert standard_error > 0.0
    assert 0.0 <= p_value <= 1.0
    assert p_value > 0.5


def test_compute_eceth_returns_finite_scalar_for_well_specified_inputs() -> None:
    tau_hat = np.linspace(-1.0, 1.0, EVALUATION_N_OBSERVATIONS)
    gamma = tau_hat + np.random.default_rng(0).normal(scale=0.05, size=EVALUATION_N_OBSERVATIONS)
    value = compute_eceth(tau_hat=tau_hat, gamma=gamma, n_bins=EVALUATION_BIN_COUNT)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_rate_input_frame_yields_nonempty_toc_and_rate() -> None:
    tau_hat = np.linspace(-1.0, 1.0, EVALUATION_N_OBSERVATIONS)
    gamma = tau_hat + np.random.default_rng(0).normal(scale=0.05, size=EVALUATION_N_OBSERVATIONS)
    frame = rate_input_frame(tau_hat=tau_hat, gamma=gamma)
    assert list(frame.columns) == ["cate", "tau"]
    assert frame.shape[0] == EVALUATION_N_OBSERVATIONS
    toc = get_toc(frame, treatment_effect_col="tau")
    assert toc.shape[0] > 0
    assert toc.shape[1] == 1
    scores = rate_score(
        frame,
        treatment_effect_col="tau",
        weighting=RATE_WEIGHTING_AUTOC,
        return_ci=False,
    )
    assert scores.shape[0] == 1
    assert np.isfinite(float(scores.iloc[0]))


def test_robust_scores_have_one_row_per_observation(
    pipeline_config_with_policy: PipelineConfig,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    evaluator = CATEEvaluator(evaluation=pipeline_config_with_policy.cate_evaluation)
    dataset = CausalDataset(
        data=pipeline_config_with_policy.data,
        df=df_synthetic_binary,
    )
    scores = evaluator.build_robust_scores(dataset=dataset)
    assert scores.shape[0] == len(df_synthetic_binary)
    assert list(scores.columns) == ["1_vs_0"]
    assert np.isfinite(scores.to_numpy()).all()

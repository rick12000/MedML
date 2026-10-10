from __future__ import annotations

import numpy as np
from causalml.metrics import get_toc, rate_score

from causal_pipeline.evaluation import (
    RATE_WEIGHTING_AUTOC,
    RATE_WEIGHTING_QINI,
    calibration_bin_rows,
    compute_eceth,
    ranking_curve,
    rate_input_frame,
)

CALIBRATION_BINS = 5
CALIBRATION_GAP_TOLERANCE = 0.08
ECETH_TOLERANCE = 0.01
PROXY_MEAN_TOLERANCE = 0.05
TOP_FRACTION = 0.2
TOC_TOLERANCE = 0.10
RATE_TOLERANCE = 0.05
RATE_WEIGHTINGS = (RATE_WEIGHTING_AUTOC, RATE_WEIGHTING_QINI)


def toc_at_top_fraction(score: np.ndarray, effect: np.ndarray, fraction: float) -> float:
    curve = get_toc(rate_input_frame(tau_hat=score, gamma=effect), treatment_effect_col="tau")
    fractions = curve.index.to_numpy(dtype=float)
    values = curve.iloc[:, 0].to_numpy(dtype=float)
    return float(values[int(np.argmin(np.abs(fractions - fraction)))])


def test_correct_cate_is_calibrated_against_the_true_effect(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    predicted, proxy, truth = recovered_linear_cate
    rows = calibration_bin_rows(tau_hat=predicted, gamma=proxy, n_bins=CALIBRATION_BINS)
    gaps = [
        abs(row["mean_predicted_cate"] - row["mean_robust_proxy"])
        for row in rows
    ]
    assert abs(float(np.mean(proxy)) - float(np.mean(truth))) < PROXY_MEAN_TOLERANCE
    assert max(gaps) < CALIBRATION_GAP_TOLERANCE
    assert abs(compute_eceth(tau_hat=predicted, gamma=proxy, n_bins=CALIBRATION_BINS)) < ECETH_TOLERANCE


def test_ranking_curves_span_the_prioritized_fraction() -> None:
    score = np.linspace(-1.0, 1.0, 40)
    effect = np.linspace(-0.2, 0.5, 40)
    fraction, value = ranking_curve(tau_hat=score, gamma=effect)
    assert fraction.shape == value.shape
    assert fraction.shape[0] > 1
    assert np.isfinite(fraction).all()
    assert np.isfinite(value).all()
    assert float(np.min(fraction)) >= 0.0
    assert float(np.max(fraction)) <= 1.0


def test_prioritization_matches_the_oracle_ranking(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    predicted, proxy, truth = recovered_linear_cate
    # Fitted ranking, then the same scores with the ranking reversed.
    rankings = (
        (predicted, truth),
        (-predicted, -truth),
    )
    for score, oracle_score in rankings:
        estimated_toc = toc_at_top_fraction(score, proxy, TOP_FRACTION)
        oracle_toc = toc_at_top_fraction(oracle_score, truth, TOP_FRACTION)
        assert abs(estimated_toc - oracle_toc) < TOC_TOLERANCE
        for weighting in RATE_WEIGHTINGS:
            estimated_rate = float(
                rate_score(
                    rate_input_frame(tau_hat=score, gamma=proxy),
                    treatment_effect_col="tau",
                    weighting=weighting,
                    return_ci=False,
                ).iloc[0]
            )
            oracle_rate = float(
                rate_score(
                    rate_input_frame(tau_hat=oracle_score, gamma=truth),
                    treatment_effect_col="tau",
                    weighting=weighting,
                    return_ci=False,
                ).iloc[0]
            )
            assert abs(estimated_rate - oracle_rate) < RATE_TOLERANCE

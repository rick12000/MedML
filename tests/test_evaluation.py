from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import CATEEvaluationConfig, OutcomeFavorability
from causal_pipeline.evaluation import (
    CATEEvaluator,
    RATE_WEIGHTING_AUTOC,
    RATE_WEIGHTING_QINI,
    absolute_eceth_tolerance,
    calibration_bin_rows,
    compute_eceth,
    eceth_hypothesis_test,
    integrate_toc,
    resample_positions,
    ranking_curve,
    rate_hypothesis_test,
    targeting_operator_curve,
)
from causal_pipeline.figures import CalibrationSeries, RankingSeries, plot_calibration_curves, plot_ranking_curves
from tests.cache_paths import artifact_path

CALIBRATION_BINS = 5
CALIBRATION_GAP_TOLERANCE = 0.08
ECETH_TOLERANCE = 0.01
PROXY_MEAN_TOLERANCE = 0.05
TOP_FRACTION = 0.2
TOC_TOLERANCE = 0.10
RATE_TOLERANCE = 0.05
RATE_WEIGHTINGS = (RATE_WEIGHTING_AUTOC, RATE_WEIGHTING_QINI)
PERFECT_EFFECT = np.array([1.0, 2.0, 3.0, 4.0])
PERFECT_AUTOC = 0.75
PERFECT_QINI = 0.3125


def test_correct_cate_is_calibrated_against_the_true_effect(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_output_dir: Path,
) -> None:
    predicted, proxy, truth = recovered_linear_cate
    rows = calibration_bin_rows(tau_hat=predicted, gamma=proxy, n_bins=CALIBRATION_BINS)
    gaps = [abs(row["mean_predicted_cate"] - row["mean_robust_proxy"]) for row in rows]
    assert abs(float(np.mean(proxy)) - float(np.mean(truth))) < PROXY_MEAN_TOLERANCE
    assert max(gaps) < CALIBRATION_GAP_TOLERANCE
    assert abs(compute_eceth(tau_hat=predicted, gamma=proxy, n_bins=CALIBRATION_BINS)) < ECETH_TOLERANCE
    eceth, eceth_se, eceth_p = eceth_hypothesis_test(
        tau_hat=predicted,
        gamma=proxy,
        n_bins=CALIBRATION_BINS,
        tolerance=ECETH_TOLERANCE,
        bootstrap_samples=30,
        random_state=0,
    )
    calibration = CalibrationSeries(
        estimator_id="t_learner_heterogeneous_recovery",
        predicted=np.array([row["mean_predicted_cate"] for row in rows], dtype=float),
        observed=np.array([row["mean_robust_proxy"] for row in rows], dtype=float),
        eceth=eceth,
        eceth_standard_error=eceth_se,
        eceth_p_value=eceth_p,
    )
    plot_calibration_curves(
        series=[calibration],
        save_path=str(
            artifact_path(
                test_output_dir,
                "heterogeneous_recovery",
                "calibration_t_learner_bins.png",
            )
        ),
    )


@pytest.mark.parametrize("weighting", RATE_WEIGHTINGS)
def test_rate_integral_matches_the_area_under_the_toc(weighting: str) -> None:
    fraction, toc = targeting_operator_curve(PERFECT_EFFECT, PERFECT_EFFECT)
    reversed_fraction, reversed_toc = targeting_operator_curve(-PERFECT_EFFECT, PERFECT_EFFECT)
    expected = PERFECT_AUTOC if weighting == RATE_WEIGHTING_AUTOC else PERFECT_QINI
    assert fraction[0] == pytest.approx(0.0)
    assert fraction[-1] == pytest.approx(1.0)
    assert toc[0] == pytest.approx(0.0)
    assert toc[-1] == pytest.approx(0.0)
    assert integrate_toc(fraction, toc, weighting) == pytest.approx(expected)
    assert integrate_toc(reversed_fraction, reversed_toc, weighting) == pytest.approx(-expected)


def test_constant_effect_has_zero_toc_and_zero_rate() -> None:
    score = np.linspace(0.0, 1.0, 6)
    effect = np.full(6, 3.0)
    fraction, toc = targeting_operator_curve(score, effect)
    assert toc == pytest.approx(0.0)
    assert integrate_toc(fraction, toc, RATE_WEIGHTING_AUTOC) == pytest.approx(0.0)
    assert integrate_toc(fraction, toc, RATE_WEIGHTING_QINI) == pytest.approx(0.0)


def test_tied_priorities_do_not_depend_on_row_order() -> None:
    score = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    effect = np.array([0.0, 10.0, -4.0, 3.0, 7.0])
    order = np.array([2, 0, 4, 1, 3])
    fraction, toc = targeting_operator_curve(score, effect)
    other_fraction, other_toc = targeting_operator_curve(score[order], effect[order])
    assert fraction.shape == (len(score) + 1,)
    assert other_toc.shape == toc.shape
    assert np.allclose(fraction, other_fraction)
    assert np.allclose(toc, other_toc)


@pytest.mark.parametrize("weighting", RATE_WEIGHTINGS)
def test_lower_is_better_flips_the_effect_with_the_score(weighting: str) -> None:
    fraction, toc = ranking_curve(tau_hat=PERFECT_EFFECT, gamma=PERFECT_EFFECT, direction=-1.0)
    oriented = integrate_toc(fraction, toc, weighting)
    score_only_fraction, score_only_toc = targeting_operator_curve(-PERFECT_EFFECT, PERFECT_EFFECT)
    both_fraction, both_toc = targeting_operator_curve(-PERFECT_EFFECT, -PERFECT_EFFECT)
    assert oriented == pytest.approx(integrate_toc(both_fraction, both_toc, weighting))
    assert oriented > 0.0
    assert integrate_toc(score_only_fraction, score_only_toc, weighting) < 0.0


def test_rate_pvalue_supports_only_a_positive_ranking() -> None:
    score = np.linspace(-2.0, 2.0, 80)
    autoc, autoc_p, qini, qini_p = rate_hypothesis_test(
        tau_hat=score,
        gamma=score,
        bootstrap_samples=20,
        random_state=0,
    )
    bad_autoc, bad_autoc_p, bad_qini, bad_qini_p = rate_hypothesis_test(
        tau_hat=-score,
        gamma=score,
        bootstrap_samples=20,
        random_state=0,
    )
    assert autoc > 0.0
    assert qini > 0.0
    assert autoc_p < 0.05
    assert qini_p < 0.05
    assert bad_autoc < 0.0
    assert bad_qini < 0.0
    assert bad_autoc_p > 0.5
    assert bad_qini_p > 0.5


def test_half_sample_keeps_groups_intact() -> None:
    groups = np.repeat(np.arange(4), 2)
    generator = np.random.default_rng(0)
    draws: list[np.ndarray] = []
    while len(draws) < 5:
        draws.append(resample_positions(len(groups), generator, groups, replace=False))
    for positions in draws:
        selected = groups[positions]
        assert len(positions) == 4
        assert len(np.unique(selected)) == 2
        for group in np.unique(selected):
            assert int(np.sum(selected == group)) == int(np.sum(groups == group))


def test_tied_predictions_stay_in_one_calibration_bin() -> None:
    predicted = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    proxy = np.arange(len(predicted), dtype=float)
    rows = calibration_bin_rows(tau_hat=predicted, gamma=proxy, n_bins=4)
    sizes = [int(row["bin_size"]) for row in rows]
    tied_bin = rows[int(np.argmax(sizes))]
    assert tied_bin["bin_size"] == 4
    assert tied_bin["mean_predicted_cate"] == pytest.approx(1.0)


def test_cluster_bootstrap_resamples_whole_groups() -> None:
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    positions = resample_positions(len(groups), np.random.default_rng(1), groups, replace=True)
    selected = groups[positions]
    assert len(positions) == len(groups)
    for group in np.unique(groups):
        assert int(np.sum(selected == group)) % 2 == 0


def test_eceth_does_not_depend_on_the_order_of_tied_rows() -> None:
    predicted = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    proxy = np.array([0.0, 5.0, -1.0, 2.0, 2.5, 4.0])
    order = np.array([2, 0, 1, 5, 3, 4])
    original = compute_eceth(tau_hat=predicted, gamma=proxy, n_bins=2)
    reordered = compute_eceth(tau_hat=predicted[order], gamma=proxy[order], n_bins=2)
    assert original == pytest.approx(reordered)


def test_eceth_pvalue_does_not_call_a_biased_cate_calibrated() -> None:
    predicted = np.linspace(-1.0, 1.0, 50)
    proxy = np.zeros(50)
    estimate, _standard_error, p_value = eceth_hypothesis_test(
        tau_hat=predicted,
        gamma=proxy,
        n_bins=5,
        tolerance=1e-4,
        bootstrap_samples=20,
        random_state=0,
    )
    assert estimate > 1e-4
    assert p_value > 0.5


def test_eceth_tolerance_follows_the_prediction_scale() -> None:
    predicted = np.array([0.0, 2.0])
    gamma = np.array([10.0, 12.0])
    assert absolute_eceth_tolerance(
        tau_hat=predicted,
        gamma=gamma,
        tolerance_fraction=0.01,
    ) == pytest.approx(0.02)


def test_eceth_tolerance_uses_score_variance_for_a_constant_prediction() -> None:
    predicted = np.array([1.0, 1.0])
    gamma = np.array([0.0, 2.0])
    assert absolute_eceth_tolerance(
        tau_hat=predicted,
        gamma=gamma,
        tolerance_fraction=0.01,
    ) == pytest.approx(0.02)


def test_ranking_curves_span_the_prioritized_fraction() -> None:
    score = np.linspace(-1.0, 1.0, 40)
    effect = np.linspace(-0.2, 0.5, 40)
    fraction, value = ranking_curve(tau_hat=score, gamma=effect)
    assert fraction.shape == value.shape
    assert fraction.shape == (len(score) + 1,)
    assert np.isfinite(fraction).all()
    assert np.isfinite(value).all()
    assert float(fraction[0]) == pytest.approx(0.0)
    assert float(fraction[-1]) == pytest.approx(1.0)
    assert float(value[0]) == pytest.approx(0.0)
    assert float(value[-1]) == pytest.approx(0.0)


def test_evaluate_cate_models_writes_pipeline_style_artifacts(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_output_dir: Path,
) -> None:
    predicted, proxy, _truth = recovered_linear_cate
    predictions = {"t_learner": pd.DataFrame({"1_vs_0": predicted})}
    scores = pd.DataFrame({"1_vs_0": proxy})
    evaluation = CATEEvaluationConfig(
        propensity_learner=LogisticRegression(max_iter=500),
        outcome_learner=LinearRegression(),
        dr_crossfit_folds=3,
        calibration_bins=5,
        rate_bootstrap_samples=8,
        eceth_bootstrap_samples=8,
        random_state=0,
    )
    results = CATEEvaluator(evaluation=evaluation).evaluate_cate_models(
        predictions=predictions,
        robust_scores=scores,
        results_root=test_output_dir,
        outcome_favorability=OutcomeFavorability.HIGHER_IS_BETTER,
    )
    assert results["t_learner"]
    assert results["t_learner"][0].eceth_tolerance == pytest.approx(
        absolute_eceth_tolerance(
            tau_hat=predicted,
            gamma=proxy,
            tolerance_fraction=evaluation.eceth_tolerance,
        )
    )
    assert (test_output_dir / "cate" / "t_learner" / "1_vs_0" / "toc.png").is_file()
    assert (test_output_dir / "cate" / "t_learner" / "1_vs_0" / "calibration.png").is_file()
    assert (test_output_dir / "cate" / "comparison" / "1_vs_0" / "toc.png").is_file()


def test_prioritization_matches_the_oracle_ranking(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_output_dir: Path,
) -> None:
    predicted, proxy, truth = recovered_linear_cate
    rankings = (
        ("fitted_t_learner", predicted, truth, "fitted"),
        ("sign_flip_control", -predicted, -truth, "sign_flip"),
    )
    for label, score, oracle_score, folder in rankings:
        estimated_fraction, estimated_toc = ranking_curve(tau_hat=score, gamma=proxy)
        oracle_fraction, oracle_value = ranking_curve(tau_hat=oracle_score, gamma=truth)
        top = int(np.argmin(np.abs(estimated_fraction - TOP_FRACTION)))
        oracle_top = int(np.argmin(np.abs(oracle_fraction - TOP_FRACTION)))
        assert abs(float(estimated_toc[top]) - float(oracle_value[oracle_top])) < TOC_TOLERANCE
        autoc = integrate_toc(estimated_fraction, estimated_toc, RATE_WEIGHTING_AUTOC)
        qini = integrate_toc(estimated_fraction, estimated_toc, RATE_WEIGHTING_QINI)
        for weighting in RATE_WEIGHTINGS:
            estimated_rate = integrate_toc(estimated_fraction, estimated_toc, weighting)
            oracle_rate = integrate_toc(oracle_fraction, oracle_value, weighting)
            assert abs(estimated_rate - oracle_rate) < RATE_TOLERANCE
        plot_ranking_curves(
            series=[
                RankingSeries(
                    estimator_id=label,
                    toc_fraction=estimated_fraction,
                    toc_value=estimated_toc,
                    autoc=autoc,
                    autoc_p_value=float("nan"),
                    qini=qini,
                    qini_p_value=float("nan"),
                )
            ],
            save_path=str(
                artifact_path(
                    test_output_dir,
                    "heterogeneous_recovery",
                    folder,
                    "toc.png",
                )
            ),
        )
        plot_ranking_curves(
            series=[
                RankingSeries(
                    estimator_id=f"{label}_oracle",
                    toc_fraction=oracle_fraction,
                    toc_value=oracle_value,
                    autoc=autoc,
                    autoc_p_value=float("nan"),
                    qini=qini,
                    qini_p_value=float("nan"),
                )
            ],
            save_path=str(
                artifact_path(
                    test_output_dir,
                    "heterogeneous_recovery",
                    folder,
                    "toc_oracle.png",
                )
            ),
        )

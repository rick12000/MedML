from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from causalml.metrics import get_toc, rate_score
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import CATEEvaluationConfig, OutcomeFavorability
from causal_pipeline.evaluation import (
    CATEEvaluator,
    RATE_WEIGHTING_AUTOC,
    RATE_WEIGHTING_QINI,
    benefit_oriented_scores,
    calibration_bin_rows,
    compute_eceth,
    eceth_hypothesis_test,
    ranking_curve,
    rate_input_frame,
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


def toc_at_top_fraction(score: np.ndarray, effect: np.ndarray, fraction: float) -> float:
    curve = get_toc(rate_input_frame(tau_hat=score, gamma=effect), treatment_effect_col="tau")
    fractions = curve.index.to_numpy(dtype=float)
    values = curve.iloc[:, 0].to_numpy(dtype=float)
    return float(values[int(np.argmin(np.abs(fractions - fraction)))])


def test_correct_cate_is_calibrated_against_the_true_effect(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_output_dir: Path,
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


def test_benefit_oriented_scores_flip_both_ranking_inputs() -> None:
    tau_hat = np.array([0.2, -0.1, 0.4])
    gamma = np.array([0.3, 0.0, -0.2])
    oriented_tau, oriented_gamma = benefit_oriented_scores(tau_hat, gamma, direction=-1.0)
    assert oriented_tau.shape == tau_hat.shape
    assert oriented_gamma.shape == gamma.shape
    assert np.allclose(oriented_tau, -tau_hat)
    assert np.allclose(oriented_gamma, -gamma)


def test_lower_is_better_rate_matches_manual_sign_flip(
    recovered_linear_cate: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    predicted, proxy, _truth = recovered_linear_cate
    for weighting in RATE_WEIGHTINGS:
        directed_rate = float(
            rate_score(
                rate_input_frame(*benefit_oriented_scores(predicted, proxy, -1.0)),
                treatment_effect_col="tau",
                weighting=weighting,
                return_ci=False,
            ).iloc[0]
        )
        manual_flip_rate = float(
            rate_score(
                rate_input_frame(-predicted, -proxy),
                treatment_effect_col="tau",
                weighting=weighting,
                return_ci=False,
            ).iloc[0]
        )
        assert abs(directed_rate - manual_flip_rate) < RATE_TOLERANCE


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
        estimated_toc = toc_at_top_fraction(score, proxy, TOP_FRACTION)
        oracle_toc = toc_at_top_fraction(oracle_score, truth, TOP_FRACTION)
        assert abs(estimated_toc - oracle_toc) < TOC_TOLERANCE
        toc_fraction, toc_value = ranking_curve(tau_hat=score, gamma=proxy)
        autoc = float(
            rate_score(
                rate_input_frame(tau_hat=score, gamma=proxy),
                treatment_effect_col="tau",
                weighting=RATE_WEIGHTING_AUTOC,
                return_ci=False,
            ).iloc[0]
        )
        qini = float(
            rate_score(
                rate_input_frame(tau_hat=score, gamma=proxy),
                treatment_effect_col="tau",
                weighting=RATE_WEIGHTING_QINI,
                return_ci=False,
            ).iloc[0]
        )
        plot_ranking_curves(
            series=[
                RankingSeries(
                    estimator_id=label,
                    toc_fraction=toc_fraction,
                    toc_value=toc_value,
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
        oracle_fraction, oracle_value = ranking_curve(tau_hat=oracle_score, gamma=truth)
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

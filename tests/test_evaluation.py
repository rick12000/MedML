from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import (
    CATEEvaluationConfig,
    DataConfig,
    OutcomeFavorability,
    OutcomeType,
    TestValidationSignal,
    TreatmentMode,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.evaluation import (
    RATE_WEIGHTING_AUTOC,
    RATE_WEIGHTING_QINI,
    CATEEvaluator,
    absolute_eceth_tolerance,
    calibration_bin_rows,
    compute_eceth,
    eceth_hypothesis_test,
    integrate_toc,
    ranking_curve,
    rate_hypothesis_test,
    resample_positions,
    targeting_operator_curve,
)
from causal_pipeline.figures import (
    CalibrationSeries,
    RankingSeries,
    plot_evaluation_figure,
)
from causal_pipeline.learner_defaults import DEFAULT_LOGISTIC_MAX_ITER
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
    fraction, value = ranking_curve(tau_hat=predicted, gamma=proxy)
    ranking = ranking_from_curve("t_learner_heterogeneous_recovery", fraction, value)
    plot_evaluation_figure(
        calibration=[calibration],
        ranking=[ranking],
        save_path=str(
            artifact_path(test_output_dir, "heterogeneous_recovery", "evaluation_t_learner_bins.png")
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
    autoc, autoc_standard_error, autoc_p, qini, qini_standard_error, qini_p = rate_hypothesis_test(
        tau_hat=score,
        gamma=score,
        bootstrap_samples=20,
        random_state=0,
    )
    bad_autoc, bad_autoc_standard_error, bad_autoc_p, bad_qini, bad_qini_standard_error, bad_qini_p = (
        rate_hypothesis_test(
            tau_hat=-score,
            gamma=score,
            bootstrap_samples=20,
            random_state=0,
        )
    )
    assert autoc > 0.0
    assert autoc_standard_error > 0.0
    assert qini_standard_error > 0.0
    assert bad_autoc_standard_error > 0.0
    assert bad_qini_standard_error > 0.0
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
        propensity_learner=LogisticRegression(max_iter=DEFAULT_LOGISTIC_MAX_ITER),
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
    assert (test_output_dir / "cate" / "t_learner" / "1_vs_0" / "toc.csv").is_file()
    assert (test_output_dir / "cate" / "t_learner" / "1_vs_0" / "calibration.csv").is_file()
    assert (test_output_dir / "cate" / "comparison" / "1_vs_0" / "evaluation.png").is_file()


def calibration_from_rows(estimator_id: str, rows: list[dict[str, float]]) -> CalibrationSeries:
    return CalibrationSeries(
        estimator_id=estimator_id,
        predicted=np.array([row["mean_predicted_cate"] for row in rows], dtype=float),
        observed=np.array([row["mean_robust_proxy"] for row in rows], dtype=float),
        eceth=float("nan"),
        eceth_standard_error=float("nan"),
        eceth_p_value=float("nan"),
    )


def ranking_from_curve(estimator_id: str, fraction: np.ndarray, value: np.ndarray) -> RankingSeries:
    return RankingSeries(
        estimator_id=estimator_id,
        toc_fraction=fraction,
        toc_value=value,
        autoc=integrate_toc(fraction, value, RATE_WEIGHTING_AUTOC),
        autoc_p_value=float("nan"),
        qini=integrate_toc(fraction, value, RATE_WEIGHTING_QINI),
        qini_p_value=float("nan"),
    )


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
        for weighting in RATE_WEIGHTINGS:
            estimated_rate = integrate_toc(estimated_fraction, estimated_toc, weighting)
            oracle_rate = integrate_toc(oracle_fraction, oracle_value, weighting)
            assert abs(estimated_rate - oracle_rate) < RATE_TOLERANCE
        fitted_rows = calibration_bin_rows(tau_hat=score, gamma=proxy, n_bins=CALIBRATION_BINS)
        oracle_rows = calibration_bin_rows(tau_hat=oracle_score, gamma=truth, n_bins=CALIBRATION_BINS)
        for name, file_name, fraction_values, toc_values, rows in (
            (label, "evaluation.png", estimated_fraction, estimated_toc, fitted_rows),
            (f"{label}_oracle", "evaluation_oracle.png", oracle_fraction, oracle_value, oracle_rows),
        ):
            plot_evaluation_figure(
                calibration=[calibration_from_rows(name, rows)],
                ranking=[ranking_from_curve(name, fraction_values, toc_values)],
                save_path=str(artifact_path(test_output_dir, "heterogeneous_recovery", folder, file_name)),
            )


def test_estimation_full_fit_scores_test_outcomes_from_estimation_nuisances() -> None:
    generator = np.random.default_rng(0)
    estimation_treatment = np.tile([0, 1], 20)
    test_treatment = np.array([0, 0, 1, 1])
    estimation_covariate = generator.normal(size=len(estimation_treatment))
    test_covariate = generator.normal(size=len(test_treatment))
    estimation_outcome = estimation_covariate + estimation_treatment + generator.normal(scale=0.1, size=40)
    test_outcome = test_covariate + test_treatment
    data = DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x"],
        effect_modifiers=["x"],
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    estimation = CausalDataset(
        data=data,
        df=pd.DataFrame({"y": estimation_outcome, "t": estimation_treatment, "x": estimation_covariate}),
    )
    test = CausalDataset(
        data=data,
        df=pd.DataFrame({"y": test_outcome, "t": test_treatment, "x": test_covariate}),
    )
    evaluator = CATEEvaluator(
        evaluation=CATEEvaluationConfig(
            propensity_learner=LogisticRegression(max_iter=DEFAULT_LOGISTIC_MAX_ITER),
            outcome_learner=LinearRegression(),
            dr_crossfit_folds=5,
            random_state=0,
        )
    )
    scores = evaluator.build_test_validation_scores(
        estimation=estimation,
        test=test,
        signal=TestValidationSignal.ESTIMATION_FULL_FIT,
    )
    changed_outcome = test_outcome.copy()
    changed_outcome[2] += 1.0
    changed_test = CausalDataset(
        data=data,
        df=pd.DataFrame({"y": changed_outcome, "t": test_treatment, "x": test_covariate}),
    )
    changed_scores = evaluator.build_test_validation_scores(
        estimation=estimation,
        test=changed_test,
        signal=TestValidationSignal.ESTIMATION_FULL_FIT,
    )
    with pytest.raises(ValueError):
        evaluator.build_test_validation_scores(
            estimation=estimation,
            test=test,
            signal=TestValidationSignal.TEST_FIT,
        )

    assert scores.shape == (len(test_treatment), 1)
    assert list(scores.columns) == ["1_vs_0"]
    difference = changed_scores.iloc[:, 0].to_numpy() - scores.iloc[:, 0].to_numpy()
    assert np.flatnonzero(np.abs(difference) > 1e-8).tolist() == [2]
    assert difference[2] > 1.0

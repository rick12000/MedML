"""CATE evaluation: robust scores, calibration, ECETH, TOC, and RATE."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from causalml.metrics import get_toc, rate_score
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import t as student_t

from causal_pipeline.config import CATEEvaluationConfig, predict_outcome_mean
from causal_pipeline.scaling import clone_scaled_estimator
from causal_pipeline.crossfit import cross_fit_splits, dataset_groups
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.diagnostics import probability_of_arm
from causal_pipeline.figures import (
    CalibrationSeries,
    RankingSeries,
    plot_calibration_curves,
    plot_ranking_curves,
)
from causal_pipeline.utils import write_dataframe

logger = logging.getLogger(__name__)

RATE_WEIGHTING_AUTOC = "autoc"
RATE_WEIGHTING_QINI = "qini"


def calibration_bin_rows(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
) -> list[dict[str, float]]:
    order = np.argsort(tau_hat)
    bins = np.array_split(order, n_bins)
    rows: list[dict[str, float]] = []
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        rows.append(
            {
                "mean_predicted_cate": float(np.mean(tau_hat[bin_indices])),
                "mean_robust_proxy": float(np.mean(gamma[bin_indices])),
                "bin_size": len(bin_indices),
            }
        )
    return rows


def ranking_curve(tau_hat: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TOC curve. AUTOC and Qini RATE are two integrals of this same curve."""
    table = get_toc(rate_input_frame(tau_hat, gamma), treatment_effect_col="tau")
    return (
        table.index.to_numpy(dtype=float).copy(),
        table.iloc[:, 0].to_numpy(dtype=float).copy(),
    )


def save_joint_cate_figures(
    ranking_by_contrast: dict[str, list[RankingSeries]],
    calibration_by_contrast: dict[str, list[CalibrationSeries]],
    results_root: Path,
    held_out: bool,
) -> None:
    marker = "test_" if held_out else ""
    for contrast, ranking in ranking_by_contrast.items():
        comparison = results_root / "cate" / "comparison" / contrast
        plot_ranking_curves(
            series=ranking,
            save_path=str(comparison / f"{marker}toc.png"),
        )
        plot_calibration_curves(
            series=calibration_by_contrast[contrast],
            save_path=str(comparison / f"{marker}calibration.png"),
        )
        logger.info("Saved joint CATE figures for %s.", contrast)


def rate_input_frame(tau_hat: np.ndarray, gamma: np.ndarray) -> pd.DataFrame:
    """Frame for causalml RATE/TOC: ranking score plus DR proxy as the effect column."""
    return pd.DataFrame({"cate": np.asarray(tau_hat), "tau": np.asarray(gamma)})


class ContrastEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    contrast: str = Field(min_length=1)
    eceth: float
    eceth_standard_error: float
    eceth_pvalue: float
    eceth_tolerance: float
    rate_autoc: float
    rate_autoc_pvalue: float
    rate_qini: float
    rate_qini_pvalue: float
    calibration_path: str = Field(min_length=1)
    toc_path: str = Field(min_length=1)


class CATEEvaluator:
    """CATE metrics from cross-fitted predictions and doubly robust proxy scores."""

    def __init__(self, evaluation: CATEEvaluationConfig) -> None:
        self.evaluation = evaluation

    def build_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        return self.build_multi_arm_robust_scores(dataset)

    def build_multi_arm_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        covariates = dataset.X_adjustment.copy()
        treatment = dataset.treatment_series
        outcome_values = dataset.outcome_series.to_numpy(dtype=float)
        levels = dataset.treatment_values
        clip_min, clip_max = self.evaluation.propensity_clip
        gamma_by_arm = {level: np.zeros(len(outcome_values)) for level in levels}

        for train_idx, test_idx in cross_fit_splits(
            features=covariates,
            treatment=treatment,
            n_folds=self.evaluation.dr_crossfit_folds,
            random_state=self.evaluation.random_state,
            groups=dataset_groups(dataset),
        ):
            train_covariates = covariates.iloc[train_idx].copy()
            test_covariates = covariates.iloc[test_idx].copy()
            train_outcome = outcome_values[train_idx]
            train_treatment = treatment.iloc[train_idx]
            propensity_learner = clone_scaled_estimator(self.evaluation.propensity_learner)
            propensity_learner.fit(train_covariates, train_treatment)
            propensity = pd.DataFrame(
                propensity_learner.predict_proba(test_covariates),
                columns=list(propensity_learner.classes_),
            )
            for arm in levels:
                outcome_learner = clone_scaled_estimator(self.evaluation.outcome_learner)
                arm_mask = train_treatment.to_numpy() == arm
                outcome_learner.fit(train_covariates.iloc[arm_mask].copy(), train_outcome[arm_mask])
                outcome_mean = predict_outcome_mean(outcome_learner, test_covariates)
                arm_propensity = np.clip(probability_of_arm(propensity, arm), clip_min, clip_max)
                observed = treatment.iloc[test_idx].to_numpy() == arm
                gamma_by_arm[arm][test_idx] = (
                    outcome_mean
                    + observed * (outcome_values[test_idx] - outcome_mean) / arm_propensity
                )

        control = dataset.control_value
        contrasts = {}
        for column, arm in zip(
            contrast_columns(control, levels),
            [value for value in levels if value != control],
            strict=True,
        ):
            contrasts[column] = gamma_by_arm[arm] - gamma_by_arm[control]
        return pd.DataFrame(contrasts)

    def evaluate_cate_models(
        self,
        predictions: dict[str, pd.DataFrame],
        robust_scores: pd.DataFrame,
        results_root: Path,
        held_out: bool = False,
    ) -> dict[str, list[ContrastEvaluation]]:
        results: dict[str, list[ContrastEvaluation]] = {}
        ranking_by_contrast: dict[str, list[RankingSeries]] = {}
        calibration_by_contrast: dict[str, list[CalibrationSeries]] = {}
        for estimator_id, df_cate_predictions in predictions.items():
            contrast_results = []
            for contrast in df_cate_predictions.columns:
                if contrast not in robust_scores.columns:
                    continue
                tau_hat = df_cate_predictions[contrast].values
                gamma = robust_scores[contrast].values
                eceth, eceth_se, eceth_pvalue = eceth_hypothesis_test(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.evaluation.calibration_bins,
                    tolerance=self.evaluation.eceth_tolerance,
                    bootstrap_samples=self.evaluation.eceth_bootstrap_samples,
                    random_state=self.evaluation.random_state,
                )
                marker = "test_" if held_out else ""
                rows = calibration_bin_rows(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.evaluation.calibration_bins,
                )
                calibration_path = f"cate/{estimator_id}/{contrast}/{marker}calibration.png"
                calibration_file = results_root / calibration_path
                toc_fraction, toc_value = ranking_curve(tau_hat=tau_hat, gamma=gamma)
                rate_autoc, rate_autoc_p = self.rate_with_bootstrap(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    weighting=RATE_WEIGHTING_AUTOC,
                )
                rate_qini, rate_qini_p = self.rate_with_bootstrap(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    weighting=RATE_WEIGHTING_QINI,
                )
                ranking = RankingSeries(
                    estimator_id=estimator_id,
                    toc_fraction=toc_fraction,
                    toc_value=toc_value,
                    autoc=rate_autoc,
                    autoc_p_value=rate_autoc_p,
                    qini=rate_qini,
                    qini_p_value=rate_qini_p,
                )
                calibration = CalibrationSeries(
                    estimator_id=estimator_id,
                    predicted=np.array(
                        [row["mean_predicted_cate"] for row in rows],
                        dtype=float,
                    ),
                    observed=np.array(
                        [row["mean_robust_proxy"] for row in rows],
                        dtype=float,
                    ),
                    eceth=eceth,
                    eceth_standard_error=eceth_se,
                    eceth_p_value=eceth_pvalue,
                )
                toc_path = f"cate/{estimator_id}/{contrast}/{marker}toc.png"
                plot_ranking_curves(series=[ranking], save_path=str(results_root / toc_path))
                plot_calibration_curves(
                    series=[calibration],
                    save_path=str(calibration_file),
                )
                write_dataframe(calibration_file.with_suffix(".csv"), pd.DataFrame(rows))
                ranking_by_contrast.setdefault(contrast, []).append(ranking)
                calibration_by_contrast.setdefault(contrast, []).append(calibration)
                contrast_results.append(
                    ContrastEvaluation(
                        contrast=contrast,
                        eceth=eceth,
                        eceth_standard_error=eceth_se,
                        eceth_pvalue=eceth_pvalue,
                        eceth_tolerance=self.evaluation.eceth_tolerance,
                        rate_autoc=rate_autoc,
                        rate_autoc_pvalue=rate_autoc_p,
                        rate_qini=rate_qini,
                        rate_qini_pvalue=rate_qini_p,
                        calibration_path=calibration_path,
                        toc_path=toc_path,
                    )
                )
            results[estimator_id] = contrast_results
        save_joint_cate_figures(
            ranking_by_contrast=ranking_by_contrast,
            calibration_by_contrast=calibration_by_contrast,
            results_root=results_root,
            held_out=held_out,
        )
        return results

    def rate_with_bootstrap(
        self,
        tau_hat: np.ndarray,
        gamma: np.ndarray,
        weighting: str,
    ) -> tuple[float, float]:
        scored = rate_score(
            rate_input_frame(tau_hat, gamma),
            treatment_effect_col="tau",
            weighting=weighting,
            return_ci=True,
            n_bootstrap=self.evaluation.rate_bootstrap_samples,
            random_state=self.evaluation.random_state,
        )
        return float(scored["rate"].iloc[0]), float(scored["p_value"].iloc[0])


def compute_eceth(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
) -> float:
    """Leave-one-out ℓ2 calibration error of Xu and Yadlowsky (2022).

    Within each equal-count bin of predicted CATE, the bin mean of the doubly
    robust scores is recomputed without patient i. The estimator is the mean
    product of (score − prediction) and (leave-one-out bin mean − prediction).
    It can be slightly negative in finite samples; that is the debiased
    estimator, not a display truncation at zero.
    """
    predicted = np.asarray(tau_hat, dtype=float)
    proxy = np.asarray(gamma, dtype=float)
    if predicted.shape != proxy.shape or predicted.ndim != 1:
        raise ValueError("CATE predictions and proxy scores must be one-dimensional and aligned.")
    if len(predicted) < 2:
        raise ValueError("ECETH needs at least two observations.")
    order = np.argsort(predicted, kind="mergesort")
    products: list[np.ndarray] = []
    for bin_index in np.array_split(order, n_bins):
        if len(bin_index) < 2:
            continue
        proxy_bin = proxy[bin_index]
        predicted_bin = predicted[bin_index]
        leave_one_out = (float(np.sum(proxy_bin)) - proxy_bin) / (len(bin_index) - 1)
        products.append((proxy_bin - predicted_bin) * (leave_one_out - predicted_bin))
    if not products:
        raise ValueError("Each ECETH bin needs at least two observations for the leave-one-out mean.")
    return float(np.mean(np.concatenate(products)))


def eceth_hypothesis_test(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
    tolerance: float,
    bootstrap_samples: int,
    random_state: int,
) -> tuple[float, float, float]:
    """Test H0: ECETH ≥ tolerance against H1: ECETH < tolerance.

    Xu and Yadlowsky use a one-sided t-test with nonparametric bootstrap
    standard errors of the leave-one-out estimator. The nuisance scores are
    held fixed: the estimator is asymptotically linear, so the bootstrap
    resamples the pairs (prediction, score).
    """
    if bootstrap_samples < 2:
        raise ValueError("The ECETH test needs at least two bootstrap replicates.")
    predicted = np.asarray(tau_hat, dtype=float)
    proxy = np.asarray(gamma, dtype=float)
    estimate = compute_eceth(tau_hat=predicted, gamma=proxy, n_bins=n_bins)
    generator = np.random.default_rng(random_state)
    replicates = np.empty(bootstrap_samples)
    sample_size = len(predicted)
    for replicate in range(bootstrap_samples):
        drawn = generator.choice(sample_size, size=sample_size, replace=True)
        replicates[replicate] = compute_eceth(
            tau_hat=predicted[drawn],
            gamma=proxy[drawn],
            n_bins=n_bins,
        )
    standard_error = float(np.std(replicates, ddof=1))
    if standard_error == 0.0:
        p_value = 0.0 if estimate < tolerance else 1.0
        return estimate, standard_error, p_value
    statistic = (estimate - tolerance) / standard_error
    p_value = float(student_t.cdf(statistic, df=bootstrap_samples - 1))
    return estimate, standard_error, p_value

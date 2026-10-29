"""CATE evaluation: robust scores, calibration, ECETH, TOC, and RATE."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm
from sklearn.base import BaseEstimator

from causal_pipeline.config import (
    CATEEvaluationConfig,
    JsonValue,
    OutcomeFavorability,
    OutcomeType,
    TestValidationSignal,
    predict_outcome_mean,
)
from causal_pipeline.policy import reward_direction
from causal_pipeline.nuisance_metrics import log_crossfit_nuisances
from causal_pipeline.learner_resolve import materialize_evaluation_learners
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


def contrast_scores_from_arms(
    gamma_by_arm: dict[JsonValue, np.ndarray],
    dataset: CausalDataset,
) -> pd.DataFrame:
    control = dataset.control_value
    treated = [value for value in dataset.treatment_values if value != control]
    columns = contrast_columns(control, dataset.treatment_values)
    return pd.DataFrame(
        {
            name: gamma_by_arm[arm] - gamma_by_arm[control]
            for name, arm in zip(columns, treated, strict=True)
        }
    )

RATE_WEIGHTING_AUTOC = "autoc"
RATE_WEIGHTING_QINI = "qini"
MINIMUM_SAMPLE = 2


def require_aligned_scores(
    left: np.ndarray,
    right: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional and aligned.")
    if len(first) < MINIMUM_SAMPLE:
        raise ValueError(f"{label} needs at least two observations.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError(f"{label} must be finite.")
    return first, second


def prediction_bin_indices(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Equal-count bins. A tie stays in one bin so membership does not follow row order."""
    if n_bins < 1:
        raise ValueError("Calibration needs at least one bin.")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    sample_size = len(values)
    sizes = [len(chunk) for chunk in np.array_split(np.arange(sample_size), n_bins)]
    bins: list[np.ndarray] = []
    start = 0
    for edge in np.cumsum(sizes):
        end = int(edge)
        if end <= start:
            continue
        while end < sample_size and ordered[end] == ordered[end - 1]:
            end += 1
        bins.append(order[start:end])
        start = end
    return bins


def calibration_bin_rows(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
) -> list[dict[str, float]]:
    predicted, proxy = require_aligned_scores(
        tau_hat,
        gamma,
        "CATE predictions and proxy scores",
    )
    rows: list[dict[str, float]] = []
    for bin_indices in prediction_bin_indices(predicted, n_bins):
        rows.append(
            {
                "mean_predicted_cate": float(np.mean(predicted[bin_indices])),
                "mean_robust_proxy": float(np.mean(proxy[bin_indices])),
                "bin_size": len(bin_indices),
            }
        )
    return rows


def benefit_oriented_scores(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    direction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sign-adjust predictions and DR proxies for treat-when-beneficial ranking."""
    oriented_tau = direction * np.asarray(tau_hat, dtype=float)
    oriented_gamma = direction * np.asarray(gamma, dtype=float)
    return oriented_tau, oriented_gamma


def prioritized_effects(score: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Effects in descending priority. Tied scores contribute their mean effect."""
    order = np.argsort(-score, kind="mergesort")
    ordered_score = score[order]
    ordered_effect = effect[order]
    tie_ends = np.flatnonzero(np.diff(ordered_score)) + 1
    starts = np.concatenate(([0], tie_ends))
    ends = np.concatenate((tie_ends, [len(order)]))
    prioritized = np.empty(len(order), dtype=float)
    for start, end in zip(starts, ends, strict=True):
        prioritized[start:end] = float(np.mean(ordered_effect[start:end]))
    return prioritized


def targeting_operator_curve(
    score: np.ndarray,
    effect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TOC(q) is the top-q mean effect minus the sample mean. TOC(0) and TOC(1) are 0."""
    priority, values = require_aligned_scores(score, effect, "Priority scores and effects")
    prioritized = prioritized_effects(priority, values)
    counts = np.arange(1, len(priority) + 1, dtype=float)
    toc_at_counts = np.cumsum(prioritized) / counts - float(np.mean(values))
    fraction = np.concatenate(([0.0], counts / len(priority)))
    toc = np.concatenate(([0.0], toc_at_counts))
    toc[-1] = 0.0
    return fraction, toc


def integrate_toc(fraction: np.ndarray, toc: np.ndarray, weighting: str) -> float:
    """AUTOC integrates TOC. Qini integrates q · TOC. Both are areas, not normalized weights."""
    if weighting == RATE_WEIGHTING_AUTOC:
        integrand = np.asarray(toc, dtype=float)
    elif weighting == RATE_WEIGHTING_QINI:
        integrand = np.asarray(fraction, dtype=float) * np.asarray(toc, dtype=float)
    else:
        raise ValueError(
            f"Unknown RATE weighting {weighting!r}. Use {RATE_WEIGHTING_AUTOC!r} or {RATE_WEIGHTING_QINI!r}."
        )
    return float(np.trapezoid(integrand, fraction))


def ranking_curve(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    direction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Benefit-oriented TOC. Pass the result to integrate_toc for AUTOC or Qini."""
    oriented_tau, oriented_gamma = benefit_oriented_scores(tau_hat, gamma, direction)
    return targeting_operator_curve(oriented_tau, oriented_gamma)


def write_ranking_curve_data(save_path: str | Path, series: Sequence[RankingSeries]) -> Path:
    rows: list[dict[str, float | str]] = []
    for item in series:
        for fraction, value in zip(item.toc_fraction, item.toc_value, strict=True):
            rows.append(
                {
                    "estimator": item.estimator_id,
                    "toc_fraction": float(fraction),
                    "toc_value": float(value),
                }
            )
    destination = Path(save_path).with_suffix(".csv")
    write_dataframe(destination, pd.DataFrame(rows))
    return destination


def write_calibration_curve_data(save_path: str | Path, series: Sequence[CalibrationSeries]) -> Path:
    rows: list[dict[str, float | str]] = []
    for item in series:
        for predicted, observed in zip(item.predicted, item.observed, strict=True):
            rows.append(
                {
                    "estimator": item.estimator_id,
                    "mean_predicted_cate": float(predicted),
                    "mean_robust_proxy": float(observed),
                }
            )
    destination = Path(save_path).with_suffix(".csv")
    write_dataframe(destination, pd.DataFrame(rows))
    return destination


def save_joint_cate_figures(
    ranking_by_contrast: dict[str, list[RankingSeries]],
    calibration_by_contrast: dict[str, list[CalibrationSeries]],
    results_root: Path,
    held_out: bool,
) -> None:
    marker = "test_" if held_out else ""
    for contrast, ranking in ranking_by_contrast.items():
        comparison = results_root / "cate" / "comparison" / contrast
        toc_path = comparison / f"{marker}toc.png"
        calibration_path = comparison / f"{marker}calibration.png"
        plot_ranking_curves(series=ranking, save_path=str(toc_path))
        write_ranking_curve_data(toc_path, ranking)
        plot_calibration_curves(
            series=calibration_by_contrast[contrast],
            save_path=str(calibration_path),
        )
        write_calibration_curve_data(calibration_path, calibration_by_contrast[contrast])
        logger.info("Saved joint CATE figures for %s.", contrast)


class ContrastEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    contrast: str = Field(min_length=1)
    eceth: float
    eceth_standard_error: float
    eceth_pvalue: float
    eceth_tolerance: float
    rate_autoc: float
    rate_autoc_standard_error: float
    rate_autoc_pvalue: float
    rate_qini: float
    rate_qini_standard_error: float
    rate_qini_pvalue: float
    calibration_path: str = Field(min_length=1)
    toc_path: str = Field(min_length=1)


class CATEEvaluator:
    """CATE metrics from cross-fitted predictions and doubly robust proxy scores."""

    def __init__(self, evaluation: CATEEvaluationConfig) -> None:
        self.evaluation = evaluation

    def build_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        return self.build_multi_arm_robust_scores(dataset)

    def build_test_validation_scores(
        self,
        estimation: CausalDataset,
        test: CausalDataset,
        signal: TestValidationSignal,
    ) -> pd.DataFrame:
        """Scores for held-out policy value and CATE calibration.

        Cross-fit training scores are left untouched. Estimation-full-fit learns the
        nuisances on every estimation row, then inserts the test treatment and outcome.
        Test-fit cross-fits those nuisances inside the test set.
        """
        if signal == TestValidationSignal.TEST_FIT:
            return self.build_robust_scores(dataset=test)
        propensity_template, outcome_template = materialize_evaluation_learners(
            self.evaluation,
            outcome_type=estimation.outcome_type,
        )
        contrasts, propensity, outcomes = self.doubly_robust_contrasts(
            train=estimation,
            score_rows=test,
            propensity_template=propensity_template,
            outcome_template=outcome_template,
        )
        log_crossfit_nuisances(
            scope="CATE/test validation DR",
            treatment=test.treatment_series,
            outcome=test.outcome_series,
            propensity=propensity,
            propensity_learner=propensity_template,
            potential_outcomes=outcomes,
            outcome_learner=outcome_template,
            binary_outcome=test.outcome_type == OutcomeType.BINARY,
        )
        return contrasts

    def doubly_robust_contrasts(
        self,
        train: CausalDataset,
        score_rows: CausalDataset,
        propensity_template: BaseEstimator,
        outcome_template: BaseEstimator,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit arm nuisances on train and return contrast scores on score_rows."""
        clip_min, clip_max = self.evaluation.propensity_clip
        levels = train.treatment_values
        train_covariates = train.X_adjustment.copy()
        score_covariates = score_rows.X_adjustment.copy()
        train_outcome = train.outcome_series.to_numpy(dtype=float)
        train_treatment = train.treatment_series
        score_outcome = score_rows.outcome_series.to_numpy(dtype=float)
        score_treatment = score_rows.treatment_series.to_numpy()
        propensity_learner = clone_scaled_estimator(propensity_template)
        propensity_learner.fit(train_covariates, train_treatment)
        propensity = pd.DataFrame(
            propensity_learner.predict_proba(score_covariates),
            columns=list(propensity_learner.classes_),
        )
        outcomes = pd.DataFrame(index=score_covariates.index, columns=levels, dtype=float)
        gamma_by_arm: dict[JsonValue, np.ndarray] = {}
        for arm in levels:
            outcome_learner = clone_scaled_estimator(outcome_template)
            arm_mask = train_treatment.to_numpy() == arm
            outcome_learner.fit(train_covariates.iloc[arm_mask].copy(), train_outcome[arm_mask])
            outcome_mean = predict_outcome_mean(outcome_learner, score_covariates)
            outcomes[arm] = outcome_mean
            arm_propensity = np.clip(probability_of_arm(propensity, arm), clip_min, clip_max)
            observed = score_treatment == arm
            gamma_by_arm[arm] = (
                outcome_mean + observed * (score_outcome - outcome_mean) / arm_propensity
            )
        aligned_propensity = propensity.reindex(columns=levels)
        aligned_propensity.index = score_covariates.index
        return (
            contrast_scores_from_arms(gamma_by_arm, train),
            aligned_propensity,
            outcomes,
        )

    def build_multi_arm_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        covariates = dataset.X_adjustment.copy()
        treatment = dataset.treatment_series
        levels = dataset.treatment_values
        contrast_names = contrast_columns(dataset.control_value, levels)
        contrast_values = {name: np.zeros(len(dataset.df)) for name in contrast_names}
        oof_propensity = pd.DataFrame(np.nan, index=covariates.index, columns=levels, dtype=float)
        oof_outcomes = pd.DataFrame(np.nan, index=covariates.index, columns=levels, dtype=float)
        propensity_template, outcome_template = materialize_evaluation_learners(
            self.evaluation,
            outcome_type=dataset.outcome_type,
        )

        for train_idx, test_idx in cross_fit_splits(
            features=covariates,
            treatment=treatment,
            n_folds=self.evaluation.dr_crossfit_folds,
            random_state=self.evaluation.random_state,
            groups=dataset_groups(dataset),
        ):
            contrasts, propensity, outcomes = self.doubly_robust_contrasts(
                train=dataset.subset(train_idx),
                score_rows=dataset.subset(test_idx),
                propensity_template=propensity_template,
                outcome_template=outcome_template,
            )
            oof_propensity.iloc[test_idx] = propensity.to_numpy()
            oof_outcomes.iloc[test_idx] = outcomes.to_numpy()
            for name in contrast_names:
                contrast_values[name][test_idx] = contrasts[name].to_numpy()

        log_crossfit_nuisances(
            scope="CATE/evaluation DR",
            treatment=treatment,
            outcome=dataset.outcome_series,
            propensity=oof_propensity,
            propensity_learner=propensity_template,
            potential_outcomes=oof_outcomes,
            outcome_learner=outcome_template,
            binary_outcome=dataset.outcome_type == OutcomeType.BINARY,
        )
        return pd.DataFrame(contrast_values)

    def evaluate_cate_models(
        self,
        predictions: dict[str, pd.DataFrame],
        robust_scores: pd.DataFrame,
        results_root: Path,
        held_out: bool = False,
        outcome_favorability: OutcomeFavorability = OutcomeFavorability.HIGHER_IS_BETTER,
        groups: np.ndarray | pd.Series | None = None,
    ) -> dict[str, list[ContrastEvaluation]]:
        direction = reward_direction(outcome_favorability)
        results: dict[str, list[ContrastEvaluation]] = {}
        ranking_by_contrast: dict[str, list[RankingSeries]] = {}
        calibration_by_contrast: dict[str, list[CalibrationSeries]] = {}
        for estimator_id, df_cate_predictions in predictions.items():
            contrast_results = []
            for contrast in df_cate_predictions.columns:
                if contrast not in robust_scores.columns:
                    continue
                tau_hat = df_cate_predictions[contrast].to_numpy(dtype=float)
                gamma = robust_scores[contrast].to_numpy(dtype=float)
                absolute_tolerance = absolute_eceth_tolerance(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    tolerance_fraction=self.evaluation.eceth_tolerance,
                )
                eceth, eceth_se, eceth_pvalue = eceth_hypothesis_test(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.evaluation.calibration_bins,
                    tolerance=absolute_tolerance,
                    bootstrap_samples=self.evaluation.eceth_bootstrap_samples,
                    random_state=self.evaluation.random_state,
                    groups=groups,
                )
                marker = "test_" if held_out else ""
                rows = calibration_bin_rows(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.evaluation.calibration_bins,
                )
                calibration_path = f"cate/{estimator_id}/{contrast}/{marker}calibration.png"
                calibration_file = results_root / calibration_path
                toc_fraction, toc_value = ranking_curve(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    direction=direction,
                )
                rate_autoc, rate_autoc_se, rate_autoc_p, rate_qini, rate_qini_se, rate_qini_p = (
                    rate_hypothesis_test(
                        tau_hat=tau_hat,
                        gamma=gamma,
                        bootstrap_samples=self.evaluation.rate_bootstrap_samples,
                        random_state=self.evaluation.random_state,
                        groups=groups,
                        direction=direction,
                    )
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
                toc_file = results_root / toc_path
                plot_ranking_curves(series=[ranking], save_path=str(toc_file))
                write_ranking_curve_data(toc_file, [ranking])
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
                        eceth_tolerance=absolute_tolerance,
                        rate_autoc=rate_autoc,
                        rate_autoc_standard_error=rate_autoc_se,
                        rate_autoc_pvalue=rate_autoc_p,
                        rate_qini=rate_qini,
                        rate_qini_standard_error=rate_qini_se,
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


def absolute_eceth_tolerance(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    tolerance_fraction: float,
) -> float:
    """Fraction of Var(τ̂), or of Var(Γ) when every prediction is the same."""
    if tolerance_fraction <= 0.0:
        raise ValueError("The ECETH tolerance fraction must be positive.")
    predicted, scores = require_aligned_scores(
        tau_hat,
        gamma,
        "CATE predictions and proxy scores",
    )
    prediction_variance = float(np.var(predicted, ddof=1))
    scale = prediction_variance if prediction_variance > 0.0 else float(np.var(scores, ddof=1))
    return tolerance_fraction * scale


def resample_positions(
    sample_size: int,
    generator: np.random.Generator,
    groups: np.ndarray | pd.Series | None,
    replace: bool,
) -> np.ndarray:
    """Draw rows. Clusters stay whole. replace=False draws half the rows or clusters."""
    group_ids = None if groups is None else np.asarray(groups)
    if group_ids is not None and len(group_ids) != sample_size:
        raise ValueError("Group ids must have one entry per score.")
    if group_ids is None:
        size = sample_size if replace else sample_size // 2
        if not replace and size < MINIMUM_SAMPLE:
            raise ValueError("A half-sample needs at least four rows.")
        return generator.choice(sample_size, size=size, replace=replace)
    inverse, codes = pd.factorize(group_ids, sort=False)
    size = len(codes) if replace else len(codes) // 2
    if not replace and size < MINIMUM_SAMPLE:
        raise ValueError("A half-sample needs at least four groups.")
    chosen = generator.choice(len(codes), size=size, replace=replace)
    if not replace:
        return np.flatnonzero(np.isin(inverse, chosen))
    return np.concatenate([np.flatnonzero(inverse == group) for group in chosen])


def normal_tail(estimate: float, threshold: float, standard_error: float, lower: bool) -> float:
    """One-sided normal p-value. lower tests estimate < threshold; otherwise estimate > threshold."""
    if standard_error == 0.0:
        beyond = estimate < threshold if lower else estimate > threshold
        return 0.0 if beyond else 1.0
    z_score = (estimate - threshold) / standard_error
    return float(norm.cdf(z_score) if lower else norm.sf(z_score))


def rate_hypothesis_test(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    bootstrap_samples: int,
    random_state: int,
    groups: np.ndarray | pd.Series | None = None,
    direction: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    """AUTOC, the SD of its half-samples, and a normal tail that treats that SD as a sampling SE.

    The same three values follow for Qini. The half-sample SD is not a sampling
    variance, so multiple imputation does not pool these tails.
    """
    if bootstrap_samples < MINIMUM_SAMPLE:
        raise ValueError("The RATE test needs at least two bootstrap replicates.")
    oriented_tau, oriented_gamma = benefit_oriented_scores(tau_hat, gamma, direction)
    fraction, toc = targeting_operator_curve(oriented_tau, oriented_gamma)
    autoc = integrate_toc(fraction, toc, RATE_WEIGHTING_AUTOC)
    qini = integrate_toc(fraction, toc, RATE_WEIGHTING_QINI)
    generator = np.random.default_rng(random_state)
    autoc_draws = np.empty(bootstrap_samples)
    qini_draws = np.empty(bootstrap_samples)
    for replicate in range(bootstrap_samples):
        drawn = resample_positions(len(oriented_tau), generator, groups, replace=False)
        draw_fraction, draw_toc = targeting_operator_curve(oriented_tau[drawn], oriented_gamma[drawn])
        autoc_draws[replicate] = integrate_toc(draw_fraction, draw_toc, RATE_WEIGHTING_AUTOC)
        qini_draws[replicate] = integrate_toc(draw_fraction, draw_toc, RATE_WEIGHTING_QINI)
    autoc_error = float(np.std(autoc_draws, ddof=1))
    qini_error = float(np.std(qini_draws, ddof=1))
    return (
        autoc,
        autoc_error,
        normal_tail(autoc, 0.0, autoc_error, lower=False),
        qini,
        qini_error,
        normal_tail(qini, 0.0, qini_error, lower=False),
    )


def compute_eceth(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
) -> float:
    """Leave-one-out ℓ2 calibration error. A small negative value is the debiased estimator."""
    predicted, proxy = require_aligned_scores(
        tau_hat,
        gamma,
        "CATE predictions and proxy scores",
    )
    products: list[np.ndarray] = []
    for bin_index in prediction_bin_indices(predicted, n_bins):
        if len(bin_index) < MINIMUM_SAMPLE:
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
    groups: np.ndarray | pd.Series | None = None,
) -> tuple[float, float, float]:
    """Test H0: ECETH ≥ tolerance against H1: ECETH < tolerance. Tolerance is absolute."""
    if bootstrap_samples < MINIMUM_SAMPLE:
        raise ValueError("The ECETH test needs at least two bootstrap replicates.")
    predicted, proxy = require_aligned_scores(
        tau_hat,
        gamma,
        "CATE predictions and proxy scores",
    )
    estimate = compute_eceth(tau_hat=predicted, gamma=proxy, n_bins=n_bins)
    generator = np.random.default_rng(random_state)
    replicates = np.empty(bootstrap_samples)
    for replicate in range(bootstrap_samples):
        drawn = resample_positions(len(predicted), generator, groups, replace=True)
        replicates[replicate] = compute_eceth(
            tau_hat=predicted[drawn],
            gamma=proxy[drawn],
            n_bins=n_bins,
        )
    standard_error = float(np.std(replicates, ddof=1))
    return estimate, standard_error, normal_tail(estimate, tolerance, standard_error, lower=True)

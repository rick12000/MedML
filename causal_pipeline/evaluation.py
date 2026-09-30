"""CATE evaluation: robust scores, calibration, ECETH, TOC, and RATE."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causalml.metrics import get_toc, rate_score
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import StratifiedKFold

from causal_pipeline.config import CATEEvaluationConfig, clone_estimator, predict_outcome_mean
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.diagnostics import probability_of_arm
from causal_pipeline.utils import save_figure, write_dataframe

logger = logging.getLogger(__name__)


def rate_input_frame(tau_hat: np.ndarray, gamma: np.ndarray) -> pd.DataFrame:
    """Frame for causalml RATE/TOC: ranking score plus DR proxy as the effect column."""
    return pd.DataFrame({"cate": np.asarray(tau_hat), "tau": np.asarray(gamma)})


class ContrastEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    contrast: str = Field(min_length=1)
    eceth: float
    rate_autoc: float
    rate_autoc_pvalue: float
    rate_qini: float
    rate_qini_pvalue: float
    calibration_path: str = Field(min_length=1)
    toc_path: str = Field(min_length=1)


class CATEEvaluator:
    """Validation-set CATE evaluation using cross-fitted robust proxy scores."""

    def __init__(self, evaluation: CATEEvaluationConfig) -> None:
        self.evaluation = evaluation

    def build_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        return self.build_multi_arm_robust_scores(dataset)

    def build_multi_arm_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        X = dataset.X_adjustment.copy()
        treatment = dataset.treatment_series
        outcome = dataset.outcome_series.to_numpy(dtype=float)
        levels = dataset.treatment_values
        n_folds = min(self.evaluation.dr_crossfit_folds, int(treatment.value_counts().min()))
        if n_folds < 2:
            raise ValueError("Doubly robust scores need at least two units in every treatment arm.")
        clip_min, clip_max = self.evaluation.propensity_clip
        folds = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.evaluation.random_state,
        )
        gamma_by_arm = {level: np.zeros(len(outcome)) for level in levels}

        for train_idx, test_idx in folds.split(X, treatment):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = outcome[train_idx]
            t_train = treatment.iloc[train_idx]
            propensity_learner = clone_estimator(self.evaluation.propensity_learner)
            propensity_learner.fit(X_train, t_train)
            propensity = pd.DataFrame(
                propensity_learner.predict_proba(X_test),
                columns=list(propensity_learner.classes_),
            )
            for arm in levels:
                outcome_learner = clone_estimator(self.evaluation.outcome_learner)
                mask = t_train.to_numpy() == arm
                outcome_learner.fit(X_train.iloc[mask].copy(), y_train[mask])
                mu = predict_outcome_mean(outcome_learner, X_test)
                arm_propensity = np.clip(probability_of_arm(propensity, arm), clip_min, clip_max)
                observed = treatment.iloc[test_idx].to_numpy() == arm
                gamma_by_arm[arm][test_idx] = mu + observed * (outcome[test_idx] - mu) / arm_propensity

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
    ) -> dict[str, list[ContrastEvaluation]]:
        results: dict[str, list[ContrastEvaluation]] = {}
        for estimator_id, df_cate_predictions in predictions.items():
            contrast_results = []
            for contrast in df_cate_predictions.columns:
                if contrast not in robust_scores.columns:
                    continue
                tau_hat = df_cate_predictions[contrast].values
                gamma = robust_scores[contrast].values
                eceth = compute_eceth(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.evaluation.calibration_bins,
                )
                calibration_path = f"cate/{estimator_id}/{contrast}/calibration.png"
                calibration_file = results_root / calibration_path
                df_calibration = self.plot_calibration(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.evaluation.calibration_bins,
                    save_path=str(calibration_file),
                )
                write_dataframe(calibration_file.with_suffix(".csv"), df_calibration)
                toc_path = f"cate/{estimator_id}/{contrast}/toc.png"
                self.plot_toc(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    save_path=str(results_root / toc_path),
                )
                rate_autoc, rate_autoc_p = self.rate_with_bootstrap(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    weighting="autoc",
                )
                rate_qini, rate_qini_p = self.rate_with_bootstrap(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    weighting="qini",
                )
                contrast_results.append(
                    ContrastEvaluation(
                        contrast=contrast,
                        eceth=eceth,
                        rate_autoc=rate_autoc,
                        rate_autoc_pvalue=rate_autoc_p,
                        rate_qini=rate_qini,
                        rate_qini_pvalue=rate_qini_p,
                        calibration_path=calibration_path,
                        toc_path=toc_path,
                    )
                )
            results[estimator_id] = contrast_results
        return results

    def plot_calibration(
        self,
        tau_hat: np.ndarray,
        gamma: np.ndarray,
        n_bins: int,
        save_path: str,
    ) -> pd.DataFrame:
        order = np.argsort(tau_hat)
        bins = np.array_split(order, n_bins)
        rows = []
        predicted_means = []
        proxy_means = []
        for bin_indices in bins:
            if len(bin_indices) == 0:
                continue
            predicted_mean = float(np.mean(tau_hat[bin_indices]))
            proxy_mean = float(np.mean(gamma[bin_indices]))
            predicted_means.append(predicted_mean)
            proxy_means.append(proxy_mean)
            rows.append(
                {
                    "mean_predicted_cate": predicted_mean,
                    "mean_robust_proxy": proxy_mean,
                }
            )
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.scatter(predicted_means, proxy_means)
        limits = [
            min(predicted_means + proxy_means),
            max(predicted_means + proxy_means),
        ]
        axis.plot(limits, limits, linestyle="--", color="gray")
        axis.set_xlabel("Mean predicted CATE")
        axis.set_ylabel("Mean robust proxy")
        fig.tight_layout()
        save_figure(save_path, fig)
        plt.close(fig)
        return pd.DataFrame(rows)

    def plot_toc(self, tau_hat: np.ndarray, gamma: np.ndarray, save_path: str) -> None:
        toc = get_toc(rate_input_frame(tau_hat, gamma), treatment_effect_col="tau")
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(toc.index.to_numpy(), toc.iloc[:, 0].to_numpy())
        axis.set_xlabel("Top fraction ranked by CATE")
        axis.set_ylabel("TOC")
        fig.tight_layout()
        save_figure(save_path, fig)
        plt.close(fig)

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
    """Binned absolute calibration error of a CATE against doubly robust scores."""
    order = np.argsort(tau_hat)
    bins = np.array_split(order, n_bins)
    gaps = []
    weights = []
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        gaps.append(abs(float(np.mean(tau_hat[bin_indices])) - float(np.mean(gamma[bin_indices]))))
        weights.append(len(bin_indices))
    return float(np.average(gaps, weights=weights))

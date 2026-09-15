"""CATE evaluation: robust scores, calibration, ECETH, TOC, and RATE."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causalml.metrics import get_toc, rate_score
from scipy import stats
from sklearn.model_selection import KFold
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import LabelEncoder

from causal_pipeline.config import PipelineConfig, build_sklearn_learner
from causal_pipeline.data import CausalDataset

logger = logging.getLogger(__name__)


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

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def build_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        return self.build_multi_arm_robust_scores(dataset)

    def build_multi_arm_robust_scores(self, dataset: CausalDataset) -> pd.DataFrame:
        X = dataset.X_confounders
        treatment = dataset.treatment_series.values
        outcome = dataset.outcome_series.values.astype(float)
        levels = dataset.treatment_values
        encoder = LabelEncoder()
        encoder.fit(levels)
        treatment_encoded = encoder.transform(treatment)
        n_folds = self.config.cate_evaluation.dr_crossfit_folds
        clip_bounds = self.config.cate_evaluation.propensity_clip
        folds = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.config.cate_evaluation.random_state,
        )
        gamma_by_arm = {level: np.zeros(len(outcome)) for level in levels}

        for train_idx, test_idx in folds.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = outcome[train_idx]
            t_train = treatment_encoded[train_idx]
            propensity_learner = build_sklearn_learner(
                self.config.diagnostics.propensity_learner,
            )
            propensity_learner.fit(X_train, t_train)
            propensity = propensity_learner.predict_proba(X_test)
            propensity = np.clip(propensity, clip_bounds[0], clip_bounds[1])

            for arm_index, arm in enumerate(levels):
                outcome_learner = build_sklearn_learner(
                    self.config.cate_evaluation.outcome_learner,
                )
                mask = t_train == arm_index
                outcome_learner.fit(X_train.iloc[mask], y_train[mask])
                mu = outcome_learner.predict(X_test)
                arm_propensity = propensity[:, arm_index]
                observed = treatment_encoded[test_idx] == arm_index
                gamma = mu + observed * (outcome[test_idx] - mu) / arm_propensity
                gamma_by_arm[arm][test_idx] = gamma

        control = dataset.control_value
        contrasts = {}
        for arm in levels:
            if arm == control:
                continue
            column = f"{arm}_vs_{control}"
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
                    n_bins=self.config.cate_evaluation.calibration_bins,
                )
                calibration_path = f"cate/{estimator_id}/{contrast}/calibration.png"
                calibration_file = results_root / calibration_path
                df_calibration = self.plot_calibration(
                    tau_hat=tau_hat,
                    gamma=gamma,
                    n_bins=self.config.cate_evaluation.calibration_bins,
                    save_path=str(calibration_file),
                )
                df_calibration.to_csv(
                    calibration_file.with_suffix(".csv"),
                    index=False,
                )
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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return pd.DataFrame(rows)

    def plot_toc(self, tau_hat: np.ndarray, gamma: np.ndarray, save_path: str) -> None:
        df_rate = pd.DataFrame({"tau_hat": tau_hat, "robust_proxy": gamma})
        toc = get_toc(df_rate, treatment_effect_col="robust_proxy", outcome_col="tau_hat")
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(toc["quantile"], toc["toc"])
        axis.set_xlabel("Top fraction ranked by CATE")
        axis.set_ylabel("TOC")
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def rate_with_bootstrap(
        self,
        tau_hat: np.ndarray,
        gamma: np.ndarray,
        weighting: str,
    ) -> tuple[float, float]:
        df_rate = pd.DataFrame({"tau_hat": tau_hat, "robust_proxy": gamma})
        point = float(
            rate_score(
                df_rate,
                treatment_effect_col="robust_proxy",
                outcome_col="tau_hat",
                weighting=weighting,
                return_ci=False,
            )
        )
        bootstrap_values = []
        n_samples = self.config.cate_evaluation.rate_bootstrap_samples
        rng = np.random.default_rng(self.config.cate_evaluation.random_state)
        for replicate in range(n_samples):
            indices = rng.choice(len(df_rate), size=len(df_rate), replace=True)
            resampled = df_rate.iloc[indices].reset_index(drop=True)
            resampled_gamma = self.build_robust_scores_from_arrays(
                tau_hat=resampled["tau_hat"].values,
                gamma=resampled["robust_proxy"].values,
            )
            bootstrap_values.append(
                float(
                    rate_score(
                        pd.DataFrame(
                            {
                                "tau_hat": resampled["tau_hat"].values,
                                "robust_proxy": resampled_gamma,
                            }
                        ),
                        treatment_effect_col="robust_proxy",
                        outcome_col="tau_hat",
                        weighting=weighting,
                        return_ci=False,
                    )
                )
            )
        standard_error = float(np.std(bootstrap_values, ddof=1))
        if standard_error == 0.0:
            pvalue = 1.0
        else:
            z_score = point / standard_error
            pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
        return point, pvalue

    def build_robust_scores_from_arrays(
        self,
        tau_hat: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        return peer_mean_gamma_leave_one_out(
            tau_hat=tau_hat,
            gamma=gamma,
            n_bins=self.config.cate_evaluation.calibration_bins,
        )


def peer_mean_gamma_leave_one_out(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    order = np.argsort(tau_hat)
    bins = np.array_split(order, n_bins)
    bin_lookup = np.empty(len(tau_hat), dtype=int)
    for bin_index, indices in enumerate(bins):
        bin_lookup[indices] = bin_index
    peer_means = np.empty(len(tau_hat), dtype=float)
    for index in range(len(tau_hat)):
        bin_id = bin_lookup[index]
        peers = bins[bin_id]
        peers_without_self = peers[peers != index]
        if len(peers_without_self) == 0:
            peer_means[index] = float(np.mean(gamma[peers]))
        else:
            peer_means[index] = float(np.mean(gamma[peers_without_self]))
    return peer_means


def compute_eceth(
    tau_hat: np.ndarray,
    gamma: np.ndarray,
    n_bins: int,
) -> float:
    peer_means = peer_mean_gamma_leave_one_out(
        tau_hat=tau_hat,
        gamma=gamma,
        n_bins=n_bins,
    )
    return float(np.mean((gamma - tau_hat) * (peer_means - tau_hat)))

"""Propensity overlap and covariate balance diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import LabelEncoder

from causal_pipeline.config import (
    DiagnosticConfig,
    JsonValue,
    PipelineConfig,
    TreatmentMode,
    build_sklearn_learner,
)
from causal_pipeline.data import CausalDataset, ipw_weights_binary, ipw_weights_multi

logger = logging.getLogger(__name__)


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    propensity_overlap_path: str | None
    covariate_balance: pd.DataFrame


class DiagnosticsRunner:
    """Run training-set propensity and balance diagnostics."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        train: CausalDataset,
        results_root: str | None = None,
    ) -> DiagnosticResult:
        logger.info("Running causal diagnostics on training data.")
        diagnostic_config = self.config.diagnostics

        propensity_model = build_sklearn_learner(diagnostic_config.propensity_learner)
        treatment_encoded = self.encode_treatment(train)
        X = train.X_confounders

        propensity_model.fit(X, treatment_encoded)
        overlap_path = None
        balance = pd.DataFrame()

        if train.treatment_mode == TreatmentMode.BINARY:
            propensity = propensity_model.predict_proba(X)[:, 1]
        else:
            propensity = propensity_model.predict_proba(X)

        clip_min, clip_max = diagnostic_config.propensity_clip
        propensity = np.clip(propensity, clip_min, clip_max)

        if diagnostic_config.plot_propensity_overlap:
            overlap_path = "diagnostics/propensity_overlap.png"
            save_path = None
            if results_root is not None:
                save_path = str(Path(results_root) / overlap_path)
            self.plot_overlap(
                treatment=train.treatment_series.values,
                propensity=propensity,
                treatment_mode=train.treatment_mode,
                treatment_values=train.treatment_values,
                save_path=save_path,
            )

        if diagnostic_config.calculate_smd:
            balance = self.covariate_balance(
                X=X,
                treatment=train.treatment_series.values,
                propensity=propensity,
                dataset=train,
                diagnostic_config=diagnostic_config,
            )

        return DiagnosticResult(
            propensity_overlap_path=overlap_path,
            covariate_balance=balance,
        )

    def encode_treatment(self, dataset: CausalDataset) -> np.ndarray:
        encoder = LabelEncoder()
        encoder.fit(dataset.treatment_values)
        return encoder.transform(dataset.treatment_series)

    def plot_overlap(
        self,
        treatment: np.ndarray,
        propensity: np.ndarray,
        treatment_mode: TreatmentMode,
        treatment_values: list[JsonValue],
        save_path: str | None = None,
    ) -> None:
        fig, axes = plt.subplots(
            nrows=1 if treatment_mode == TreatmentMode.BINARY else len(treatment_values),
            figsize=(8, 4 if treatment_mode == TreatmentMode.BINARY else 3 * len(treatment_values)),
            squeeze=False,
        )

        if treatment_mode == TreatmentMode.BINARY:
            axis = axes[0, 0]
            for arm in treatment_values:
                mask = treatment == arm
                axis.hist(
                    propensity[mask],
                    bins=30,
                    alpha=0.5,
                    label=str(arm),
                    density=True,
                )
            axis.set_title("Propensity overlap")
            axis.set_xlabel("P(T=1 | X)")
            axis.legend()
        else:
            for panel_index, arm in enumerate(treatment_values):
                axis = axes[panel_index, 0]
                arm_probs = propensity[:, panel_index]
                for observed_arm in treatment_values:
                    mask = treatment == observed_arm
                    axis.hist(
                        arm_probs[mask],
                        bins=30,
                        alpha=0.5,
                        label=str(observed_arm),
                        density=True,
                    )
                axis.set_title(f"P(T={arm} | X)")
                axis.legend()

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def covariate_balance(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        propensity: np.ndarray,
        dataset: CausalDataset,
        diagnostic_config: DiagnosticConfig,
    ) -> pd.DataFrame:
        if dataset.treatment_mode == TreatmentMode.BINARY:
            treated = (treatment != dataset.control_value).astype(float)
            weights = ipw_weights_binary(
                treated,
                propensity if propensity.ndim == 1 else propensity[:, 1],
                stabilized=diagnostic_config.stabilized_weights,
            )
            rows = []
            for column in X.columns:
                values = X[column].astype(float).values
                unweighted = smd_binary(values, treated)
                weighted = smd_binary(values, treated, sample_weight=weights)
                rows.append(
                    {
                        "covariate": column,
                        "unweighted_smd": unweighted,
                        "weighted_smd": weighted,
                    }
                )
            return pd.DataFrame(rows)

        rows = []
        levels = dataset.treatment_values
        weights = ipw_weights_multi(treatment, propensity, levels)
        for column in X.columns:
            values = X[column].astype(float).values
            max_unweighted = 0.0
            max_weighted = 0.0
            for index_a, arm_a in enumerate(levels):
                for index_b, arm_b in enumerate(levels):
                    if index_a >= index_b:
                        continue
                    mask_a = treatment == arm_a
                    mask_b = treatment == arm_b
                    unweighted = smd_two_groups(values[mask_a], values[mask_b])
                    weighted = smd_two_groups(
                        values[mask_a],
                        values[mask_b],
                        weight_a=weights[mask_a],
                        weight_b=weights[mask_b],
                    )
                    max_unweighted = max(max_unweighted, abs(unweighted))
                    max_weighted = max(max_weighted, abs(weighted))
                    rows.append(
                        {
                            "covariate": column,
                            "arm_a": arm_a,
                            "arm_b": arm_b,
                            "unweighted_smd": unweighted,
                            "weighted_smd": weighted,
                        }
                    )
        pairwise = pd.DataFrame(rows)
        aggregate = (
            pairwise.groupby("covariate", as_index=False)
            .agg(
                max_abs_unweighted_smd=("unweighted_smd", lambda s: float(np.max(np.abs(s)))),
                max_abs_weighted_smd=("weighted_smd", lambda s: float(np.max(np.abs(s)))),
            )
        )
        return pairwise.merge(aggregate, on="covariate", how="left")


def smd_binary(
    values: np.ndarray,
    treated: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    treated_values = values[treated == 1]
    control_values = values[treated == 0]
    if sample_weight is None:
        return smd_two_groups(treated_values, control_values)
    weight_treated = sample_weight[treated == 1]
    weight_control = sample_weight[treated == 0]
    return smd_two_groups(
        treated_values,
        control_values,
        weight_a=weight_treated,
        weight_b=weight_control,
    )


def smd_two_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    weight_a: np.ndarray | None = None,
    weight_b: np.ndarray | None = None,
) -> float:
    mean_a = weighted_mean(group_a, weight_a)
    mean_b = weighted_mean(group_b, weight_b)
    var_a = weighted_variance(group_a, weight_a)
    var_b = weighted_variance(group_b, weight_b)
    pooled = np.sqrt((var_a + var_b) / 2.0)
    if pooled == 0.0:
        return 0.0
    return float((mean_a - mean_b) / pooled)


def weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    total = np.sum(weights)
    if total == 0.0:
        return 0.0
    return float(np.sum(values * weights) / total)


def weighted_variance(values: np.ndarray, weights: np.ndarray | None) -> float:
    mean = weighted_mean(values, weights)
    if weights is None:
        return float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    total = np.sum(weights)
    if total == 0.0:
        return 0.0
    return float(np.sum(weights * (values - mean) ** 2) / total)

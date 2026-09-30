"""Propensity overlap and covariate balance diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causallib.estimation import IPW
from causallib.utils.stat_utils import calc_weighted_standardized_mean_differences
from pydantic import BaseModel, ConfigDict

from causal_pipeline.config import (
    DiagnosticConfig,
    JsonValue,
    TreatmentMode,
    clone_estimator,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.utils import save_figure

AUSTIN_SMD_RESCALE = float(np.sqrt(2.0))
OVERLAP_HISTOGRAM_BINS = 30
OVERLAP_FIGURE_WIDTH = 8
OVERLAP_FIGURE_HEIGHT_BINARY = 4
OVERLAP_FIGURE_HEIGHT_PER_ARM = 3

logger = logging.getLogger(__name__)


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    propensity_overlap_path: str | None
    covariate_balance: pd.DataFrame


class DiagnosticsRunner:
    """Run training-set propensity and balance diagnostics."""

    def __init__(self, diagnostics: DiagnosticConfig) -> None:
        self.diagnostics = diagnostics

    def run(
        self,
        train: CausalDataset,
        results_root: str | None = None,
    ) -> DiagnosticResult:
        logger.info("Running causal diagnostics on training data.")
        diagnostic_config = self.diagnostics

        propensity_model = clone_estimator(diagnostic_config.propensity_learner)
        adjustment_covariates = train.X_adjustment.copy()
        propensity_model.fit(adjustment_covariates, train.treatment_series)
        probabilities = pd.DataFrame(
            propensity_model.predict_proba(adjustment_covariates),
            index=adjustment_covariates.index,
            columns=list(propensity_model.classes_),
        )
        overlap_path = "diagnostics/propensity_overlap.png"
        save_path = None
        if results_root is not None:
            save_path = str(Path(results_root) / overlap_path)

        self.plot_overlap(
            treatment=train.treatment_series.to_numpy(),
            probabilities=probabilities,
            treatment_mode=train.treatment_mode,
            treatment_values=train.treatment_values,
            control_value=train.control_value,
            save_path=save_path,
        )

        balance = self.covariate_balance(
            covariates=adjustment_covariates,
            dataset=train,
            diagnostic_config=diagnostic_config,
        )

        return DiagnosticResult(
            propensity_overlap_path=overlap_path if save_path is not None else None,
            covariate_balance=balance,
        )

    def plot_overlap(
        self,
        treatment: np.ndarray,
        probabilities: pd.DataFrame,
        treatment_mode: TreatmentMode,
        treatment_values: list[JsonValue],
        control_value: JsonValue,
        save_path: str | None = None,
    ) -> None:
        fig, axes = plt.subplots(
            nrows=1 if treatment_mode == TreatmentMode.BINARY else len(treatment_values),
            figsize=(
                OVERLAP_FIGURE_WIDTH,
                OVERLAP_FIGURE_HEIGHT_BINARY
                if treatment_mode == TreatmentMode.BINARY
                else OVERLAP_FIGURE_HEIGHT_PER_ARM * len(treatment_values),
            ),
            squeeze=False,
        )

        if treatment_mode == TreatmentMode.BINARY:
            axis = axes[0, 0]
            non_control = next(arm for arm in treatment_values if arm != control_value)
            propensity = probability_of_arm(probabilities, non_control)
            for arm in treatment_values:
                mask = treatment == arm
                axis.hist(
                    propensity[mask],
                    bins=OVERLAP_HISTOGRAM_BINS,
                    alpha=0.5,
                    label=str(arm),
                    density=True,
                )
            axis.set_title("Propensity overlap")
            axis.set_xlabel(f"P(T={non_control} | X)")
            axis.legend()
        else:
            for panel_index, arm in enumerate(treatment_values):
                axis = axes[panel_index, 0]
                arm_probs = probability_of_arm(probabilities, arm)
                for observed_arm in treatment_values:
                    mask = treatment == observed_arm
                    axis.hist(
                        arm_probs[mask],
                        bins=OVERLAP_HISTOGRAM_BINS,
                        alpha=0.5,
                        label=str(observed_arm),
                        density=True,
                    )
                axis.set_title(f"P(T={arm} | X)")
                axis.legend()

        fig.tight_layout()
        if save_path is not None:
            save_figure(save_path, fig)
        plt.close(fig)

    def covariate_balance(
        self,
        covariates: pd.DataFrame,
        dataset: CausalDataset,
        diagnostic_config: DiagnosticConfig,
    ) -> pd.DataFrame:
        clip_min, clip_max = diagnostic_config.propensity_clip
        weight_model = IPW(
            learner=clone_estimator(diagnostic_config.propensity_learner),
            clip_min=clip_min,
            clip_max=clip_max,
            use_stabilized=diagnostic_config.stabilized_weights,
        )
        treatment = dataset.treatment_series
        weight_model.fit(covariates, treatment)
        weights = weight_model.compute_weights(covariates, treatment).to_numpy(dtype=float)
        treatment_values = treatment.to_numpy()
        if dataset.treatment_mode == TreatmentMode.BINARY:
            treated = treatment_values != dataset.control_value
            rows = []
            for column in covariates.columns:
                values = covariates[column].astype(float).to_numpy()
                rows.append(
                    {
                        "covariate": column,
                        "unweighted_smd": smd_two_groups(values[treated], values[~treated]),
                        "weighted_smd": smd_two_groups(
                            values[treated],
                            values[~treated],
                            weight_a=weights[treated],
                            weight_b=weights[~treated],
                        ),
                    }
                )
            return pd.DataFrame(rows)

        rows = []
        levels = dataset.treatment_values
        for column in covariates.columns:
            values = covariates[column].astype(float).to_numpy()
            for index_a, arm_a in enumerate(levels):
                for arm_b in levels[index_a + 1 :]:
                    mask_a = treatment_values == arm_a
                    mask_b = treatment_values == arm_b
                    rows.append(
                        {
                            "covariate": column,
                            "arm_a": arm_a,
                            "arm_b": arm_b,
                            "unweighted_smd": smd_two_groups(values[mask_a], values[mask_b]),
                            "weighted_smd": smd_two_groups(
                                values[mask_a],
                                values[mask_b],
                                weight_a=weights[mask_a],
                                weight_b=weights[mask_b],
                            ),
                        }
                    )
        pairwise = pd.DataFrame(rows)
        aggregate = pairwise.groupby("covariate", as_index=False).agg(
            max_abs_unweighted_smd=("unweighted_smd", lambda values: float(np.max(np.abs(values)))),
            max_abs_weighted_smd=("weighted_smd", lambda values: float(np.max(np.abs(values)))),
        )
        return pairwise.merge(aggregate, on="covariate", how="left")


def probability_of_arm(probabilities: pd.DataFrame, arm: JsonValue) -> np.ndarray:
    for column in probabilities.columns:
        if column == arm or str(column) == str(arm):
            return probabilities[column].to_numpy(dtype=float)
    raise KeyError(arm)


def smd_two_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    weight_a: np.ndarray | None = None,
    weight_b: np.ndarray | None = None,
) -> float:
    """Austin standardized difference, with the unweighted variance in the denominator."""
    weights_a = np.ones(len(group_a)) if weight_a is None else weight_a
    weights_b = np.ones(len(group_b)) if weight_b is None else weight_b
    difference = calc_weighted_standardized_mean_differences(
        group_a,
        group_b,
        weights_a,
        weights_b,
        weighted_var=False,
    )
    if not np.isfinite(difference):
        return 0.0
    return float(difference * AUSTIN_SMD_RESCALE)

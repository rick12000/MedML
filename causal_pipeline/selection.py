"""Non-destructive selection: transport when excluded covariates are seen, and bounds when they are not.

Destructive selection is omitted. When inclusion depends only on observed
covariates, the included-sample effect is valid for the included patients and
the question is how far it travels.

If covariates of the excluded patients are observed, the transported effect is
the doubly robust estimator of Dahabreh, Robertson, Steingrimsson, Stuart, and
Hernán (2020): standardize the outcome regression onto the excluded covariate
distribution, then add inverse-odds weighted residuals from the included
sample. That remains consistent if either the outcome regressions are correct
or the sampling and propensity models are correct. The pure inverse-odds
average of the outcome regression, which is the estimator in the reference
note, is reported alongside it.

If the excluded patients are unseen and only their share of the population is
known, the population effect is a mixture. The outcome range supplies a bound
with no further assumptions, and the tipping point is the excluded effect that
would cancel the included estimate.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator

from causal_pipeline.config import JsonValue, clone_estimator, predict_outcome_mean, require_classifier
from causal_pipeline.data import contrast_columns
from causal_pipeline.diagnostics import probability_of_arm
from causal_pipeline.utils import save_figure

logger = logging.getLogger(__name__)

TIPPING_GRID_SIZE = 200
TIPPING_FIGURE_SIZE = (6, 4)
INCLUDED_LABEL = 1


class UnseenPopulationBound(BaseModel):
    model_config = ConfigDict(frozen=True)

    included_effect: float
    inclusion_fraction: float
    interval_lower: float | None
    interval_upper: float | None
    tipping_point: float


def split_included_population(
    df: pd.DataFrame,
    selection_column: str,
    blocked_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy included and excluded rows. The analysis frame drops the inclusion flag."""
    if selection_column not in df.columns:
        raise ValueError(f"Selection column {selection_column} is missing.")
    if selection_column in blocked_columns:
        raise ValueError("selection_column cannot also be an outcome, treatment, or covariate.")
    labels = df[selection_column]
    if not set(np.unique(labels.to_numpy())).issubset({0, 1}):
        raise ValueError("selection_column must be coded as 0 or 1.")
    included_mask = labels.to_numpy() == INCLUDED_LABEL
    if not np.any(included_mask):
        raise ValueError("The selection column marks no included patients.")
    included = df.loc[included_mask].drop(columns=[selection_column]).copy()
    excluded = df.loc[~included_mask].drop(columns=[selection_column]).copy()
    return included, excluded


def unseen_population_bound(
    included_effect: float,
    inclusion_fraction: float,
    outcome_lower: float | None,
    outcome_upper: float | None,
) -> UnseenPopulationBound:
    """Bound the mixture τ = π τ_included + (1 − π) τ_excluded, and solve τ = 0 for τ_excluded."""
    if not 0.0 < inclusion_fraction < 1.0:
        raise ValueError("inclusion_fraction must lie strictly between 0 and 1.")
    interval_lower = None
    interval_upper = None
    if outcome_lower is not None or outcome_upper is not None:
        if outcome_lower is None or outcome_upper is None:
            raise ValueError("outcome_lower and outcome_upper must be supplied together.")
        if outcome_lower >= outcome_upper:
            raise ValueError("outcome_bounds must be ordered as (lower, upper).")
        effect_span = outcome_upper - outcome_lower
        weight = 1.0 - inclusion_fraction
        center = inclusion_fraction * included_effect
        interval_lower = center - weight * effect_span
        interval_upper = center + weight * effect_span
    tipping_point = -inclusion_fraction / (1.0 - inclusion_fraction) * included_effect
    return UnseenPopulationBound(
        included_effect=included_effect,
        inclusion_fraction=inclusion_fraction,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        tipping_point=tipping_point,
    )


def mixture_effect(
    included_effect: float,
    inclusion_fraction: float,
    excluded_effect: float,
) -> float:
    return inclusion_fraction * included_effect + (1.0 - inclusion_fraction) * excluded_effect


def transport_included_effects(
    included_covariates: pd.DataFrame,
    included_treatment: pd.Series,
    included_outcome: pd.Series,
    excluded_covariates: pd.DataFrame,
    outcome_learner: BaseEstimator,
    sampling_learner: BaseEstimator,
    propensity_learner: BaseEstimator,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    propensity_clip: tuple[float, float],
) -> pd.DataFrame:
    """Transport each arm-versus-control effect onto the excluded covariate distribution."""
    if len(excluded_covariates) == 0:
        raise ValueError("Transport needs at least one excluded patient with covariates.")
    require_classifier(sampling_learner, "sampling_learner")
    require_classifier(propensity_learner, "propensity_learner")
    included = included_covariates.reset_index(drop=True)
    excluded = excluded_covariates.reset_index(drop=True).loc[:, list(included.columns)]
    treatment = included_treatment.reset_index(drop=True)
    outcome = included_outcome.reset_index(drop=True).to_numpy(dtype=float)
    if len(included) != len(treatment) or len(included) != len(outcome):
        raise ValueError("Included covariates, treatment, and outcome must have the same length.")
    clip_min, clip_max = propensity_clip
    inclusion_probability = fit_inclusion_probability(
        included=included,
        excluded=excluded,
        sampling_learner=sampling_learner,
    )
    inclusion_probability = np.clip(inclusion_probability, clip_min, clip_max)
    odds = (1.0 - inclusion_probability) / inclusion_probability
    propensity = fit_propensity(
        included=included,
        treatment=treatment,
        propensity_learner=propensity_learner,
    )
    arm_doubly_robust: dict[JsonValue, float] = {}
    arm_standardized: dict[JsonValue, float] = {}
    included_means: dict[JsonValue, np.ndarray] = {}
    for arm in treatment_values:
        arm_mask = treatment.to_numpy() == arm
        if int(arm_mask.sum()) == 0:
            raise ValueError(f"No included patients received treatment {arm}.")
        outcome_model = clone_estimator(outcome_learner)
        outcome_model.fit(included.iloc[np.flatnonzero(arm_mask)].copy(), outcome[arm_mask])
        mu_excluded = predict_outcome_mean(outcome_model, excluded)
        mu_included = predict_outcome_mean(outcome_model, included)
        arm_propensity = np.clip(probability_of_arm(propensity, arm), clip_min, clip_max)
        residual = (treatment.to_numpy() == arm) * (outcome - mu_included) / arm_propensity
        standardized = float(np.mean(mu_excluded))
        arm_standardized[arm] = standardized
        arm_doubly_robust[arm] = standardized + float(np.sum(odds * residual) / len(excluded))
        included_means[arm] = mu_included
    names = contrast_columns(control_value, treatment_values)
    arms = [value for value in treatment_values if value != control_value]
    rows = []
    for name, arm in zip(names, arms, strict=True):
        included_contrast = included_means[arm] - included_means[control_value]
        inverse_odds = float(np.sum(odds * included_contrast) / np.sum(odds))
        rows.append(
            {
                "contrast": name,
                "doubly_robust": arm_doubly_robust[arm] - arm_doubly_robust[control_value],
                "outcome_standardization": arm_standardized[arm] - arm_standardized[control_value],
                "inverse_odds": inverse_odds,
            }
        )
    logger.info("Transported %s contrasts onto %s excluded patients.", len(rows), len(excluded))
    return pd.DataFrame(rows)


def fit_inclusion_probability(
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    sampling_learner: BaseEstimator,
) -> np.ndarray:
    stacked = pd.concat([included, excluded], axis=0, ignore_index=True)
    labels = np.concatenate(
        [
            np.ones(len(included), dtype=int),
            np.zeros(len(excluded), dtype=int),
        ]
    )
    model = clone_estimator(sampling_learner)
    model.fit(stacked, labels)
    probabilities = pd.DataFrame(
        model.predict_proba(included),
        columns=list(model.classes_),
    )
    return probability_of_arm(probabilities, INCLUDED_LABEL)


def fit_propensity(
    included: pd.DataFrame,
    treatment: pd.Series,
    propensity_learner: BaseEstimator,
) -> pd.DataFrame:
    model = clone_estimator(propensity_learner)
    model.fit(included, treatment)
    return pd.DataFrame(
        model.predict_proba(included),
        columns=list(model.classes_),
    )


def plot_tipping_point(
    bound: UnseenPopulationBound,
    outcome_lower: float,
    outcome_upper: float,
    save_path: str,
) -> None:
    """Plot the population effect against the unknown effect among excluded patients."""
    excluded_lower = outcome_lower - outcome_upper
    excluded_upper = outcome_upper - outcome_lower
    excluded_grid = np.linspace(excluded_lower, excluded_upper, TIPPING_GRID_SIZE)
    population_effect = mixture_effect(
        included_effect=bound.included_effect,
        inclusion_fraction=bound.inclusion_fraction,
        excluded_effect=excluded_grid,
    )
    figure, axis = plt.subplots(figsize=TIPPING_FIGURE_SIZE)
    axis.plot(excluded_grid, population_effect)
    axis.axhline(0.0, color="gray", linestyle="--")
    axis.axvline(bound.tipping_point, color="gray", linestyle=":")
    axis.set_xlabel("ATE among excluded patients")
    axis.set_ylabel("Implied population ATE")
    figure.tight_layout()
    save_figure(save_path, figure)
    plt.close(figure)

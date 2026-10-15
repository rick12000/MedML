"""Out-of-fold and in-sample metrics for propensity and outcome nuisances."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, roc_auc_score

from causal_pipeline.config import JsonValue, predict_outcome_mean

logger = logging.getLogger(__name__)

MINIMUM_CLASSES_FOR_AUC = 2


@dataclass(frozen=True)
class OutcomeMetric:
    name: str
    value: float


class DoubleMLSubmodel(Protocol):
    predictions: dict[str, np.ndarray]


class EconmlCrossfitCache(Protocol):
    Y: np.ndarray
    T: np.ndarray
    nuisances: tuple[np.ndarray, ...]


class EconmlCrossfitModel(Protocol):
    _cached_values: EconmlCrossfitCache | None


def factual_potential_outcome(
    potential_outcomes: pd.DataFrame,
    treatment: pd.Series,
) -> np.ndarray:
    row_index = np.arange(len(treatment))
    column_index = potential_outcomes.columns.get_indexer(treatment.to_numpy())
    if np.any(column_index < 0):
        raise KeyError("Observed treatment is missing from the potential-outcome table.")
    return potential_outcomes.to_numpy()[row_index, column_index]


def learner_name(learner: BaseEstimator) -> str:
    return type(learner).__name__


def log_nuisance_metric(role: str, learner: object, metric: str, value: float) -> None:
    logger.info("Nuisance %s (%s): %s=%.4f", role, learner_name(learner), metric, value)


def has_multiple_classes(values: np.ndarray) -> bool:
    return len(np.unique(values)) >= MINIMUM_CLASSES_FOR_AUC


def finite_outcome_prediction_pairs(
    outcome: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with non-finite outcomes or scores (e.g. propensity-trimmed holdouts)."""
    outcome_values = np.asarray(outcome, dtype=float)
    prediction_values = np.asarray(predicted, dtype=float)
    if outcome_values.shape != prediction_values.shape:
        raise ValueError("Outcome and prediction arrays must have the same shape.")
    mask = np.isfinite(outcome_values) & np.isfinite(prediction_values)
    return outcome_values[mask], prediction_values[mask]


def outcome_metric_from_predictions(
    outcome: np.ndarray,
    predicted: np.ndarray,
    binary_outcome: bool,
) -> OutcomeMetric | None:
    outcome_values, prediction_values = finite_outcome_prediction_pairs(outcome, predicted)
    if outcome_values.size == 0 or not has_multiple_classes(outcome_values):
        return None
    if binary_outcome:
        return OutcomeMetric("AUC", float(roc_auc_score(outcome_values, prediction_values)))
    return OutcomeMetric("R2", float(r2_score(outcome_values, prediction_values)))


def binary_column_auc(
    treatment: np.ndarray,
    positive_label: object,
    positive_scores: np.ndarray,
) -> float:
    return float(roc_auc_score(treatment == positive_label, positive_scores))


def propensity_auc(treatment: pd.Series | np.ndarray, propensity: pd.DataFrame) -> float | None:
    """Multiclass or binary AUC from a row-aligned propensity matrix."""
    arm_labels = list(propensity.columns)
    treatment_values = np.asarray(treatment)
    if not has_multiple_classes(treatment_values):
        return None
    score_matrix = propensity.loc[:, arm_labels].to_numpy(dtype=float)
    column_count = score_matrix.shape[1]
    if column_count == 1:
        return binary_column_auc(treatment_values, arm_labels[0], score_matrix[:, 0])
    if column_count == 2:
        return binary_column_auc(treatment_values, arm_labels[1], score_matrix[:, 1])
    return float(
        roc_auc_score(
            treatment_values,
            score_matrix,
            multi_class="ovr",
            labels=arm_labels,
            average="weighted",
        )
    )


def factual_outcome_metric(
    outcome: pd.Series | np.ndarray,
    treatment: pd.Series,
    potential_outcomes: pd.DataFrame,
    binary_outcome: bool,
) -> OutcomeMetric | None:
    """Metric for μ(Y | X, T_observed) using out-of-fold potential outcomes."""
    predicted = factual_potential_outcome(potential_outcomes, treatment)
    return outcome_metric_from_predictions(
        outcome=np.asarray(outcome, dtype=float),
        predicted=predicted,
        binary_outcome=binary_outcome,
    )


def in_sample_classification_auc(
    learner: BaseEstimator,
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
) -> float | None:
    label_values = np.asarray(labels)
    if not has_multiple_classes(label_values):
        return None
    probabilities = pd.DataFrame(
        learner.predict_proba(features),
        columns=list(learner.classes_),
    )
    return propensity_auc(label_values, probabilities)


def in_sample_outcome_metric(
    learner: BaseEstimator,
    features: pd.DataFrame,
    outcome: pd.Series | np.ndarray,
    binary_outcome: bool,
) -> OutcomeMetric | None:
    predicted = predict_outcome_mean(learner, features)
    return outcome_metric_from_predictions(
        outcome=np.asarray(outcome, dtype=float),
        predicted=predicted,
        binary_outcome=binary_outcome,
    )


def log_crossfit_nuisances(
    scope: str,
    treatment: pd.Series,
    outcome: pd.Series,
    propensity: pd.DataFrame,
    propensity_learner: BaseEstimator,
    potential_outcomes: pd.DataFrame | None,
    outcome_learner: BaseEstimator | None,
    binary_outcome: bool,
) -> None:
    propensity_auc_value = propensity_auc(treatment, propensity)
    if propensity_auc_value is not None:
        log_nuisance_metric(f"{scope} propensity", propensity_learner, "OOF AUC", propensity_auc_value)
    if potential_outcomes is None or outcome_learner is None:
        return
    outcome_metric = factual_outcome_metric(outcome, treatment, potential_outcomes, binary_outcome)
    if outcome_metric is not None:
        log_nuisance_metric(
            f"{scope} outcome",
            outcome_learner,
            f"OOF {outcome_metric.name}",
            outcome_metric.value,
        )


def log_in_sample_propensity(
    scope: str,
    learner: BaseEstimator,
    features: pd.DataFrame,
    treatment: pd.Series | np.ndarray,
) -> None:
    auc = in_sample_classification_auc(learner, features, treatment)
    if auc is not None:
        log_nuisance_metric(f"{scope} propensity", learner, "AUC", auc)


def log_in_sample_outcome(
    scope: str,
    learner: BaseEstimator,
    features: pd.DataFrame,
    outcome: pd.Series | np.ndarray,
    binary_outcome: bool,
) -> None:
    outcome_metric = in_sample_outcome_metric(learner, features, outcome, binary_outcome)
    if outcome_metric is not None:
        log_nuisance_metric(f"{scope} outcome", learner, outcome_metric.name, outcome_metric.value)


def log_doubleml_irm_plr_nuisances(
    scope: str,
    treatment: np.ndarray,
    outcome: np.ndarray,
    predictions: dict[str, np.ndarray],
    outcome_learner: BaseEstimator,
    propensity_learner: BaseEstimator,
    binary_outcome: bool,
    plr: bool,
) -> None:
    propensity_scores = np.ravel(predictions["ml_m"])
    treated = treatment.astype(int) == 1
    if has_multiple_classes(treated.astype(int)):
        log_nuisance_metric(
            f"{scope} propensity",
            propensity_learner,
            "OOF AUC",
            float(roc_auc_score(treated, propensity_scores)),
        )
    if plr:
        predicted_outcome = np.ravel(predictions["ml_l"])
    else:
        control_outcome = np.ravel(predictions["ml_g0"])
        treated_outcome = np.ravel(predictions["ml_g1"])
        predicted_outcome = np.where(treated, treated_outcome, control_outcome)
    outcome_metric = outcome_metric_from_predictions(outcome, predicted_outcome, binary_outcome)
    if outcome_metric is not None:
        log_nuisance_metric(
            f"{scope} outcome",
            outcome_learner,
            f"OOF {outcome_metric.name}",
            outcome_metric.value,
        )


def log_doubleml_apos_nuisances(
    scope: str,
    treatment: np.ndarray,
    models: list[DoubleMLSubmodel],
    treatment_levels: list[JsonValue],
    propensity_learner: BaseEstimator,
) -> None:
    for level, submodel in zip(treatment_levels, models, strict=True):
        propensity_scores = np.ravel(submodel.predictions["ml_m"])
        arm_indicator = (treatment == level).astype(int)
        if has_multiple_classes(arm_indicator):
            log_nuisance_metric(
                f"{scope} propensity arm={level}",
                propensity_learner,
                "OOF AUC",
                float(roc_auc_score(arm_indicator, propensity_scores)),
            )


def log_econml_crossfit_nuisances(
    scope: str,
    model: EconmlCrossfitModel,
    outcome_learner: BaseEstimator,
    propensity_learner: BaseEstimator,
    binary_outcome: bool,
) -> None:
    """Score econml's cross-fit residuals. Those residuals are held-out fold predictions."""
    cached = model._cached_values
    if cached is None:
        return
    outcome = np.ravel(cached.Y)
    outcome_prediction = outcome - np.ravel(cached.nuisances[0])
    outcome_metric = outcome_metric_from_predictions(outcome, outcome_prediction, binary_outcome)
    if outcome_metric is not None:
        log_nuisance_metric(
            f"{scope} outcome",
            outcome_learner,
            f"OOF {outcome_metric.name}",
            outcome_metric.value,
        )
    treatment = np.ravel(cached.T)
    treatment_residual = np.asarray(cached.nuisances[1], dtype=float)
    if treatment_residual.ndim == 2 and treatment_residual.shape[1] == 1 and has_multiple_classes(treatment):
        propensity = treatment - np.ravel(treatment_residual)
        log_nuisance_metric(
            f"{scope} propensity",
            propensity_learner,
            "OOF AUC",
            float(roc_auc_score(treatment, propensity)),
        )

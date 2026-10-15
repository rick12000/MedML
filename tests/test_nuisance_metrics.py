"""Tests for nuisance model metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_pipeline.nuisance_metrics import (
    factual_outcome_metric,
    outcome_metric_from_predictions,
    propensity_auc,
)


def test_propensity_auc_binary() -> None:
    treatment = pd.Series([0, 0, 1, 1])
    propensity = pd.DataFrame({0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]})
    auc = propensity_auc(treatment, propensity)
    assert auc is not None
    assert auc > 0.99


def test_outcome_metric_ignores_nan_from_propensity_trim() -> None:
    outcome = np.array([0.0, 1.0, 0.0, 1.0])
    predicted = np.array([0.1, 0.9, np.nan, 0.8])
    metric = outcome_metric_from_predictions(outcome, predicted, binary_outcome=True)
    assert metric is not None
    assert metric.name == "AUC"
    assert metric.value > 0.99


def test_factual_outcome_metric_r2() -> None:
    treatment = pd.Series([0, 1, 0, 1])
    outcome = pd.Series([1.0, 3.0, 1.0, 3.0])
    potential = pd.DataFrame({0: [1.0, 2.0, 1.0, 2.0], 1: [2.0, 3.0, 2.0, 3.0]})
    metric = factual_outcome_metric(outcome, treatment, potential, binary_outcome=False)
    assert metric is not None
    assert metric.name == "R2"
    assert metric.value == 1.0

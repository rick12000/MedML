"""Tests for nuisance model metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_pipeline.nuisance_metrics import factual_outcome_metric, propensity_auc


def test_propensity_auc_binary() -> None:
    treatment = pd.Series([0, 0, 1, 1])
    propensity = pd.DataFrame({0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]})
    auc = propensity_auc(treatment, propensity)
    assert auc is not None
    assert auc > 0.99


def test_factual_outcome_metric_r2() -> None:
    treatment = pd.Series([0, 1, 0, 1])
    outcome = pd.Series([1.0, 3.0, 1.0, 3.0])
    potential = pd.DataFrame({0: [1.0, 2.0, 1.0, 2.0], 1: [2.0, 3.0, 2.0, 3.0]})
    metric = factual_outcome_metric(outcome, treatment, potential, binary_outcome=False)
    assert metric is not None
    assert metric.name == "R2"
    assert metric.value == 1.0

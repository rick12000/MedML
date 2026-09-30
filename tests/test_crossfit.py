from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_pipeline.crossfit import (
    clip_propensity,
    cross_fit_splits,
    one_versus_control_propensity,
    target_potential_outcomes,
)


def test_three_arm_clip_stays_inside_the_configured_bounds() -> None:
    propensity = pd.DataFrame([[0.01, 0.01, 0.98]], columns=[0, 1, 2])
    clipped = clip_propensity(propensity, (0.02, 0.98))
    assert clipped.shape == propensity.shape
    assert np.allclose(clipped.sum(axis=1), 1.0)
    assert float(clipped.min().min()) >= 0.02 - 1e-8
    assert float(clipped.max().max()) <= 0.98 + 1e-8


def test_cross_fit_refuses_fewer_rows_than_folds() -> None:
    treatment = pd.Series([0, 0, 0, 1, 1, 1])
    features = np.zeros((len(treatment), 1))
    with pytest.raises(ValueError):
        cross_fit_splits(
            features=features,
            treatment=treatment,
            n_folds=5,
            random_state=0,
        )


def test_one_versus_control_propensity_conditions_on_the_pair() -> None:
    propensity = pd.DataFrame({0: [0.2, 0.5], 1: [0.3, 0.25], 2: [0.5, 0.25]})
    conditional = one_versus_control_propensity(propensity=propensity, control_value=0)
    assert set(conditional) == {1, 2}
    assert np.allclose(conditional[1], [0.3 / 0.5, 0.25 / 0.75])
    assert np.allclose(conditional[2], [0.5 / 0.7, 0.25 / 0.75])


def test_continuous_tmle_scales_before_the_logit_clip() -> None:
    outcome = pd.Series(np.linspace(10.0, 20.0, 20))
    treatment = pd.Series([0, 1] * 10)
    potential_outcomes = pd.DataFrame({0: np.full(20, 12.0), 1: np.full(20, 18.0)})
    propensity = pd.DataFrame({0: np.full(20, 0.5), 1: np.full(20, 0.5)})
    updated = target_potential_outcomes(
        outcome=outcome,
        treatment=treatment,
        potential_outcomes=potential_outcomes,
        propensity=propensity,
        treatment_values=[0, 1],
        control_value=0,
        reduced=False,
    )
    assert updated.shape == potential_outcomes.shape
    assert float(updated[0].median()) > 11.0
    assert float(updated[1].median()) > 15.0

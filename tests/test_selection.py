from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import DEFAULT_PROPENSITY_CLIP
from causal_pipeline.selection import (
    mixture_effect,
    split_included_population,
    transport_included_effects,
    unseen_population_bound,
)


def test_unseen_selection_matches_the_age_cutoff_example() -> None:
    bound = unseen_population_bound(
        included_effect=1.2,
        inclusion_fraction=0.4,
        outcome_lower=0.0,
        outcome_upper=10.0,
    )
    assert np.isclose(bound.interval_lower, -5.52)
    assert np.isclose(bound.interval_upper, 6.48)
    assert np.isclose(bound.tipping_point, -0.8)
    assert np.isclose(
        mixture_effect(
            included_effect=bound.included_effect,
            inclusion_fraction=bound.inclusion_fraction,
            excluded_effect=bound.tipping_point,
        ),
        0.0,
    )


def test_split_included_population_keeps_only_selected_rows() -> None:
    frame = pd.DataFrame(
        {
            "y": [1, 0, 1],
            "t": [1, 0, 0],
            "x": [0.2, 0.4, 0.6],
            "selected": [1, 0, 1],
        }
    )
    included, excluded = split_included_population(
        df=frame,
        selection_column="selected",
        blocked_columns=["y", "t", "x"],
    )
    assert len(included) == 2
    assert len(excluded) == 1
    assert "selected" not in included.columns
    assert list(frame.columns) == ["y", "t", "x", "selected"]


def test_transport_moves_the_effect_toward_the_excluded_covariate_distribution() -> None:
    generator = np.random.default_rng(0)
    sample_size = 900
    covariate = generator.normal(size=sample_size)
    included = generator.random(sample_size) < 1.0 / (1.0 + np.exp(-1.2 * covariate))
    propensity = 0.35 + 0.2 * (covariate > 0)
    treatment = generator.random(sample_size) < propensity
    true_effect = 0.8 * covariate
    outcome = 0.1 * covariate + treatment * true_effect + generator.normal(scale=0.4, size=sample_size)
    included_frame = pd.DataFrame(
        {
            "x": covariate[included],
            "t": treatment[included].astype(int),
            "y": outcome[included],
        }
    )
    excluded_frame = pd.DataFrame({"x": covariate[~included]})
    transported = transport_included_effects(
        included_covariates=included_frame[["x"]],
        included_treatment=included_frame["t"],
        included_outcome=included_frame["y"],
        excluded_covariates=excluded_frame,
        outcome_learner=LinearRegression(),
        sampling_learner=LogisticRegression(max_iter=500),
        propensity_learner=LogisticRegression(max_iter=500),
        treatment_values=[0, 1],
        control_value=0,
        propensity_clip=DEFAULT_PROPENSITY_CLIP,
    )
    truth = float(np.mean(true_effect[~included]))
    included_truth = float(np.mean(true_effect[included]))
    estimate = float(transported["doubly_robust"].iloc[0])
    assert transported.shape == (1, 4)
    assert abs(estimate - truth) < abs(included_truth - truth)
    assert abs(estimate - truth) < 0.15

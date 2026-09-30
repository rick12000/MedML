from __future__ import annotations

import numpy as np
import pandas as pd

from causal_pipeline.config import OutcomeFavorability
from causal_pipeline.policy import (
    INTERVAL_STATUS_BOOTSTRAP,
    INTERVAL_STATUS_DEFINITIONAL,
    PolicyService,
    best_treatment_arm,
    reward_direction,
    reward_matrix,
)


def test_reward_matrix_places_each_contrast_on_its_arm() -> None:
    predictions = pd.DataFrame({"1_vs_0": [0.2, -0.4], "2_vs_0": [0.5, 0.1]})
    reward = reward_matrix(
        predictions=predictions,
        treatment_values=[1, 0, 2],
        control_value=0,
    )
    assert reward.shape == (2, 3)
    assert np.allclose(reward[:, 0], [0.2, -0.4])
    assert np.allclose(reward[:, 1], [0.0, 0.0])
    assert np.allclose(reward[:, 2], [0.5, 0.1])


def test_virtual_twin_leaf_recommends_control_when_contrasts_are_negative() -> None:
    effects = pd.Series({"1_vs_0": -0.3})
    recommended = best_treatment_arm(
        mean_effects=effects,
        treatment_values=[0, 1],
        control_value=0,
    )
    assert recommended == 0


def test_virtual_twin_leaf_recommends_the_positive_contrast() -> None:
    effects = pd.Series({"1_vs_0": 0.2, "2_vs_0": 0.5})
    recommended = best_treatment_arm(
        mean_effects=effects,
        treatment_values=[0, 1, 2],
        control_value=0,
    )
    assert recommended == 2


def test_lower_outcome_prefers_the_negative_contrast() -> None:
    effects = pd.Series({"1_vs_0": -0.4, "2_vs_0": 0.2})
    recommended = best_treatment_arm(
        mean_effects=effects,
        treatment_values=[0, 1, 2],
        control_value=0,
        direction=reward_direction(OutcomeFavorability.LOWER_IS_BETTER),
    )
    assert recommended == 1


def test_control_recommendation_does_not_report_a_zero_width_interval() -> None:
    service = PolicyService(methods=[], bootstrap_samples=30, random_state=0)
    scores = pd.DataFrame({"1_vs_0": [0.2, -0.1, 0.4, 0.0]})
    definitional = service.bootstrap_subgroup_effect(
        subgroup_scores=scores,
        recommended_treatment=0,
        control_value=0,
    )
    sampled = service.bootstrap_subgroup_effect(
        subgroup_scores=scores,
        recommended_treatment=1,
        control_value=0,
    )
    assert definitional.estimate == 0.0
    assert np.isnan(definitional.ci_lower)
    assert np.isnan(definitional.ci_upper)
    assert definitional.interval_status == INTERVAL_STATUS_DEFINITIONAL
    assert sampled.ci_lower <= sampled.estimate <= sampled.ci_upper
    assert sampled.interval_status == INTERVAL_STATUS_BOOTSTRAP

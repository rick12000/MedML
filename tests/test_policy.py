import numpy as np
import pandas as pd

from causal_pipeline.policy import _best_arm, _reward_matrix


def test_reward_matrix_places_each_contrast_on_its_arm() -> None:
    predictions = pd.DataFrame({"1_vs_0": [0.2, -0.4], "2_vs_0": [0.5, 0.1]})
    reward = _reward_matrix(predictions, treatment_values=[1, 0, 2], control_value=0)
    assert reward.shape == (2, 3)
    assert np.allclose(reward[:, 0], [0.2, -0.4])
    assert np.allclose(reward[:, 1], [0.0, 0.0])
    assert np.allclose(reward[:, 2], [0.5, 0.1])


def test_virtual_twin_leaf_recommends_control_when_contrasts_are_negative() -> None:
    effects = pd.Series({"1_vs_0": -0.3})
    assert _best_arm(effects, treatment_values=[0, 1], control_value=0) == 0


def test_virtual_twin_leaf_recommends_the_positive_contrast() -> None:
    effects = pd.Series({"1_vs_0": 0.2, "2_vs_0": 0.5})
    assert _best_arm(effects, treatment_values=[0, 1, 2], control_value=0) == 2

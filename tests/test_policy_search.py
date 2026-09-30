from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_pipeline.config import (
    DataConfig,
    ExactPolicyTreeMethodSpec,
    OutcomeType,
    PolicyKind,
    PolicySearchMode,
    TreatmentMode,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.policy import PolicyService
from causal_pipeline.policy_tree import (
    ExactPolicyTree,
    capital_inclusion_rewards,
    capital_reward_matrix,
    offset_treatment_rewards,
)


def test_depth_two_exact_tree_recovers_an_interaction_depth_one_misses() -> None:
    features = pd.DataFrame(
        {
            "first": [0, 0, 0, 0, 1, 1, 1, 1],
            "second": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    treat = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
    rewards = np.column_stack([np.zeros(len(treat)), treat])
    shallow = ExactPolicyTree(
        mode=PolicySearchMode.WELFARE,
        depth=1,
        min_node_size=1,
        split_step=1,
        feature_names=list(features.columns),
    )
    shallow.fit(features=features, rewards=rewards, control_action=0)
    deep = ExactPolicyTree(
        mode=PolicySearchMode.WELFARE,
        depth=2,
        min_node_size=1,
        split_step=1,
        feature_names=list(features.columns),
    )
    deep.fit(features=features, rewards=rewards, control_action=0)
    assert shallow.welfare == 0.0
    assert deep.welfare == 4.0
    assert np.array_equal(deep.predict(features), np.array([1, 1, 0, 0, 0, 0, 1, 1]))


def test_minimum_effect_offset_withholds_treatment_below_the_threshold() -> None:
    rewards = np.column_stack([np.zeros(4), np.full(4, 0.2)])
    adjusted = offset_treatment_rewards(rewards=rewards, minimum_effect=0.5, control_action=0)
    features = pd.DataFrame({"age": [1.0, 2.0, 3.0, 4.0]})
    tree = ExactPolicyTree(
        mode=PolicySearchMode.WELFARE,
        depth=1,
        min_node_size=1,
        split_step=1,
        feature_names=["age"],
    )
    tree.fit(features=features, rewards=adjusted, control_action=0)
    assert np.all(adjusted[:, 1] < 0.0)
    assert np.array_equal(tree.predict(features), np.zeros(4, dtype=int))


def test_capital_prefix_keeps_patients_who_clear_the_cumulative_mean() -> None:
    inclusion = capital_inclusion_rewards(
        effects=np.array([3.0, 0.5, 0.0, -4.0]),
        clinical_threshold=1.0,
        negative_effect_penalty=0.0,
    )
    assert np.array_equal(inclusion, np.array([1.0, 1.0, 1.0, -1.0]))


def test_capital_penalty_lowers_the_reward_of_a_harmful_included_patient() -> None:
    inclusion = capital_inclusion_rewards(
        effects=np.array([5.0, -0.5]),
        clinical_threshold=0.0,
        negative_effect_penalty=10.0,
    )
    assert inclusion[0] == 1.0
    assert inclusion[1] == 1.0 + 10.0 * -0.5


def test_capital_tree_treats_the_cumulative_prefix() -> None:
    features = pd.DataFrame({"marker": [0.0, 1.0, 2.0, 3.0]})
    effects = np.array([3.0, 0.5, 0.0, -4.0])
    rewards = capital_reward_matrix(
        effects=effects,
        treatment_values=[0, 1],
        control_value=0,
        clinical_threshold=1.0,
        negative_effect_penalty=0.0,
    )
    tree = ExactPolicyTree(
        mode=PolicySearchMode.CAPITAL,
        depth=1,
        min_node_size=1,
        split_step=1,
        feature_names=["marker"],
    )
    tree.fit(features=features, rewards=rewards, control_action=0)
    assert np.array_equal(tree.predict(features), np.array([1, 1, 1, 0]))


@pytest.mark.parametrize(
    ("mode", "expected_treatment"),
    [
        (PolicySearchMode.WELFARE, 1),
        (PolicySearchMode.CAPITAL, 0),
    ],
)
def test_policy_service_switches_exact_tree_mode(
    mode: PolicySearchMode,
    expected_treatment: int,
) -> None:
    data = DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1", "x2"],
        effect_modifiers=["x1"],
        outcome_type=OutcomeType.BINARY,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    frame = pd.DataFrame(
        {
            "y": np.zeros(12, dtype=int),
            "t": np.tile([0, 1], 6),
            "x1": np.linspace(0.0, 1.0, 12),
            "x2": np.zeros(12),
        }
    )
    dataset = CausalDataset(data=data, df=frame)
    service = PolicyService(
        methods=[
            ExactPolicyTreeMethodSpec(
                kind=PolicyKind.EXACT_POLICY_TREE,
                mode=mode,
                depth=1,
                clinical_threshold=1.0,
            )
        ],
        bootstrap_samples=5,
        random_state=0,
    )
    fitted = service.fit_policy_methods(
        estimation=dataset,
        predictions={"s_learner": pd.DataFrame({"1_vs_0": np.full(12, 0.4)})},
    )
    assert len(fitted) == 1
    assert len(fitted[0].rules) >= 1
    assert {rule.recommended_treatment for rule in fitted[0].rules} == {expected_treatment}

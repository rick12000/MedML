from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from econml.policy import PolicyTree

from causal_pipeline.config import (
    DataConfig,
    OutcomeFavorability,
    OutcomeType,
    PolicyKind,
    PolicyTreeMethodSpec,
    TreatmentMode,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.policy import (
    PolicyService,
    dr_policy_outcome,
    reward_direction,
    reward_matrix,
)
from causal_pipeline.policy_figures import build_tree_nodes


@pytest.mark.parametrize(
    "favorability",
    [OutcomeFavorability.HIGHER_IS_BETTER, OutcomeFavorability.LOWER_IS_BETTER],
)
def test_uniform_positive_cate_policy_tree_arm(favorability: OutcomeFavorability) -> None:
    sample_size = 12
    scores = pd.DataFrame({"1_vs_0": np.full(sample_size, 0.6)})
    frame = pd.DataFrame(
        {
            "y": np.zeros(sample_size, dtype=int),
            "t": np.zeros(sample_size, dtype=int),
            "x": np.linspace(0.0, 1.0, sample_size),
        }
    )
    data = DataConfig(
        outcome="y",
        treatment="t",
        confounders=[],
        effect_modifiers=["x"],
        outcome_type=OutcomeType.BINARY,
        outcome_favorability=favorability,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    dataset = CausalDataset(data=data, df=frame)
    reward = reward_matrix(
        predictions=scores,
        treatment_values=[0, 1],
        control_value=0,
        direction=reward_direction(favorability),
    )
    tree = PolicyTree(max_depth=1, min_samples_leaf=2)
    tree.fit(dataset.X_effect_modifiers, reward)
    expected_action = 1 if favorability == OutcomeFavorability.HIGHER_IS_BETTER else 0
    assert np.all(tree.predict(dataset.X_effect_modifiers) == expected_action)


def test_dr_policy_outcome_binary_uses_complement_when_lower_is_better() -> None:
    outcome = pd.Series([0.0, 1.0])
    unchanged = dr_policy_outcome(
        outcome=outcome,
        outcome_type=OutcomeType.BINARY,
        favorability=OutcomeFavorability.HIGHER_IS_BETTER,
    )
    assert unchanged.tolist() == [0.0, 1.0]
    flipped = dr_policy_outcome(
        outcome=outcome,
        outcome_type=OutcomeType.BINARY,
        favorability=OutcomeFavorability.LOWER_IS_BETTER,
    )
    assert flipped.tolist() == [1.0, 0.0]


def test_dr_policy_outcome_continuous_uses_negation_when_lower_is_better() -> None:
    outcome = pd.Series([1.0, 2.5])
    flipped = dr_policy_outcome(
        outcome=outcome,
        outcome_type=OutcomeType.CONTINUOUS,
        favorability=OutcomeFavorability.LOWER_IS_BETTER,
    )
    assert flipped.tolist() == [-1.0, -2.5]


def test_policy_tree_figure_decision_matches_fitted_policy_on_heterogeneous_cate() -> None:
    sample_size = 40
    covariate = np.linspace(0.0, 1.0, sample_size)
    scores = pd.DataFrame({"1_vs_0": np.where(covariate < 0.5, -0.8, 0.8)})
    frame = pd.DataFrame({"y": 0, "t": 0, "x": covariate})
    data = DataConfig(
        outcome="y",
        treatment="t",
        confounders=[],
        effect_modifiers=["x"],
        outcome_type=OutcomeType.BINARY,
        outcome_favorability=OutcomeFavorability.LOWER_IS_BETTER,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    dataset = CausalDataset(data=data, df=frame)
    service = PolicyService(methods=[], bootstrap_samples=10, random_state=0)
    fitted = service.fit_standard_policy_tree(
        spec=PolicyTreeMethodSpec(kind=PolicyKind.POLICY_TREE),
        estimator_id="s_learner",
        dataset=dataset,
        X_mod=dataset.X_effect_modifiers,
        robust_scores=scores,
    )
    nodes = build_tree_nodes(
        policy=fitted,
        estimation=dataset,
        train_scores=scores,
        test=dataset,
        test_scores=scores,
    )
    fitted_actions = fitted.model.predict(dataset.X_effect_modifiers)
    for node in nodes:
        if not node.is_leaf:
            continue
        row_mask = np.zeros(sample_size, dtype=bool)
        leaf_ids = fitted.model.apply(dataset.X_effect_modifiers)
        node_leaf = int(node.key)
        row_mask[leaf_ids == node_leaf] = True
        if not np.any(row_mask):
            continue
        majority_action = int(np.bincount(fitted_actions[row_mask]).argmax())
        assert node.decision == dataset.treatment_values[majority_action]
        if node.is_treat:
            assert node.train_benefit > 0.0
        else:
            assert node.train_benefit <= 0.0

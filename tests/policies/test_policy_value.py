from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from causal_pipeline.core.config import (
    DataConfig,
    OutcomeFavorability,
    OutcomeType,
    TreatmentMode,
)
from causal_pipeline.core.data import CausalDataset
from causal_pipeline.policies.policy import FittedPolicy, PolicyRule, assigned_treatments
from causal_pipeline.policies.policy_value import (
    baseline_assignments,
    benefit_matrix,
    make_assignment,
    policy_value_table,
)


def two_arm_dataset(
    frame: pd.DataFrame,
    favorability: OutcomeFavorability = OutcomeFavorability.HIGHER_IS_BETTER,
) -> CausalDataset:
    config = DataConfig(
        outcome="y",
        treatment="t",
        confounders=[],
        effect_modifiers=["x"],
        outcome_type=OutcomeType.CONTINUOUS,
        outcome_favorability=favorability,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    return CausalDataset(data=config, df=frame)


def test_benefit_matrix_is_zero_for_control_and_flips_lower_is_better() -> None:
    frame = pd.DataFrame({"y": [0.0, 1.0], "t": [0, 1], "x": [0.0, 1.0]})
    dataset = two_arm_dataset(frame, OutcomeFavorability.LOWER_IS_BETTER)
    gains = benefit_matrix(scores=pd.DataFrame({"1_vs_0": [0.4, -0.2]}), dataset=dataset)
    assert gains[0].tolist() == [0.0, 0.0]
    assert np.allclose(gains[1], [-0.4, 0.2])


def test_unrecommended_terminal_node_gets_control() -> None:
    frame = pd.DataFrame(
        {
            "y": [0.0, 0.0, 1.0, 1.0],
            "t": [0, 1, 0, 1],
            "x": [-2.0, -1.0, 1.0, 2.0],
        }
    )
    dataset = two_arm_dataset(frame)
    tree = DecisionTreeRegressor(max_depth=1).fit(
        dataset.X_effect_modifiers,
        np.array([0.0, 0.0, 1.0, 1.0]),
    )
    high_leaf = int(tree.apply(pd.DataFrame({"x": [1.0]}))[0])
    leaf_rule = PolicyRule(
        method="virtual_twins",
        source_cate_model="s_learner",
        rule_id=0,
        rule="x > 0",
        recommended_treatment=1,
        learning_n=2,
        subgroup_leaf=high_leaf,
    )
    union_rule = PolicyRule(
        method="virtual_twins",
        source_cate_model="s_learner",
        rule_id=1,
        rule="qualifying_union",
        recommended_treatment=1,
        learning_n=2,
        included_leaves=(high_leaf,),
    )
    policy = FittedPolicy(
        method="virtual_twins",
        source_cate_model="s_learner",
        model=tree,
        rules=[leaf_rule, union_rule],
        feature_names=["x"],
    )
    assert assigned_treatments(policy=policy, dataset=dataset).tolist() == [0, 0, 1, 1]


def test_policy_differences_are_paired_mean_gain_differences() -> None:
    frame = pd.DataFrame({"y": [0.0] * 4, "t": [1, 0, 1, 0], "x": [-1.0, -2.0, 1.0, 2.0]})
    dataset = two_arm_dataset(frame)
    gains = benefit_matrix(
        scores=pd.DataFrame({"1_vs_0": [1.0, -1.0, 2.0, 0.0]}),
        dataset=dataset,
    )
    baselines = baseline_assignments(
        dataset=dataset, gains=gains, control_outcome_scores=np.zeros(4),
    )
    learned = make_assignment(
        name="policy_tree_s_learner",
        method="policy_tree",
        source_cate_model="s_learner",
        treatment=np.array([0, 0, 1, 1], dtype=object),
        gains=gains,
        control_value=0,
    )
    table = policy_value_table(
        assignments=[*baselines, learned],
        comparators=baselines,
        groups=None,
        bootstrap_samples=50,
        random_state=0,
    ).set_index("policy")

    assert table.loc["never_treat", "vs_never_treat"] == 0.0
    assert table.loc["never_treat", ["vs_never_treat_ci_lower", "vs_never_treat_ci_upper"]].tolist() == [0.0, 0.0]
    assert np.isclose(table.loc["policy_tree_s_learner", "vs_never_treat"], 0.5)
    assert np.isclose(table.loc["policy_tree_s_learner", "vs_always_treat_1"], 0.0)
    assert np.isclose(table.loc["policy_tree_s_learner", "vs_observed"], 0.5)
    assert table.loc["observed", "vs_observed"] == 0.0
    assert table.loc["observed", ["vs_observed_ci_lower", "vs_observed_ci_upper"]].tolist() == [0.0, 0.0]


def test_each_non_control_arm_gets_its_own_always_treat_assignment() -> None:
    frame = pd.DataFrame({"y": [0.0, 0.0, 0.0], "t": [0, 1, 2], "x": [0.0, 1.0, 2.0]})
    config = DataConfig(
        outcome="y",
        treatment="t",
        confounders=[],
        effect_modifiers=["x"],
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.MULTI,
        control_value=0,
        treatment_values=[0, 1, 2],
    )
    dataset = CausalDataset(data=config, df=frame)
    gains = benefit_matrix(
        scores=pd.DataFrame({"1_vs_0": [0.1, 0.2, 0.3], "2_vs_0": [0.4, 0.5, 0.6]}),
        dataset=dataset,
    )
    assignments = baseline_assignments(
        dataset=dataset, gains=gains, control_outcome_scores=np.zeros(3),
    )
    assert [assignment.name for assignment in assignments] == [
        "never_treat",
        "always_treat_1",
        "always_treat_2",
        "observed",
    ]
    always_second_arm = assignments[2]
    assert np.allclose(always_second_arm.gain, [0.4, 0.5, 0.6])
    assert always_second_arm.share_treated == 1.0
    assert assignments[0].share_treated == 0.0


@pytest.mark.parametrize("direction", [1.0, -1.0])
@pytest.mark.parametrize("outcome_model_correct", [True, False])
def test_observed_gain_is_doubly_robust(direction: float, outcome_model_correct: bool) -> None:
    frame = pd.DataFrame({"y": [5.0, 6.0], "t": [0, 1], "x": [0.0, 0.0]})
    favorability = (
        OutcomeFavorability.HIGHER_IS_BETTER if direction == 1.0
        else OutcomeFavorability.LOWER_IS_BETTER
    )
    dataset = two_arm_dataset(frame, favorability)
    # Either correct outcome means with wrong propensity, or vice versa.
    propensity = 0.8 if outcome_model_correct else 0.5
    mean_control = 5.0 if outcome_model_correct else 0.0
    mean_treated = 6.0 if outcome_model_correct else 0.0
    treatment = frame["t"].to_numpy()
    gamma_control = mean_control + (1 - treatment) * (frame["y"].to_numpy() - mean_control) / (1 - propensity)
    gamma_treated = mean_treated + treatment * (frame["y"].to_numpy() - mean_treated) / propensity
    gains = benefit_matrix(
        scores=pd.DataFrame({"1_vs_0": gamma_treated - gamma_control}), dataset=dataset,
    )
    assignments = baseline_assignments(
        dataset=dataset, gains=gains, control_outcome_scores=gamma_control,
    )
    assert len(assignments) == 3
    assert assignments[-1].gain.shape == (2,)
    assert assignments[-1].gain.mean() == pytest.approx(direction * 0.5)

"""Exact depth-limited policy trees and CAPITAL inclusion rewards.

The search follows Sverdrup, Kanodia, Zhou, Athey, and Wager (2020) and Zhou,
Athey, and Wager (2023): among trees of depth at most two, pick the one that
maximizes the sum of action rewards. Depth two is the exact regime used by the
policytree package; deeper trees are a different, hybrid search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_pipeline.config import JsonValue, PolicySearchMode

POLICY_VALUE_TOLERANCE = 1e-8
EXACT_POLICY_WELFARE_METHOD = "exact_policy_welfare"
EXACT_POLICY_CAPITAL_METHOD = "exact_policy_capital"


@dataclass(frozen=True)
class PolicySplitNode:
    feature_index: int | None
    threshold: float | None
    action: int | None
    left: PolicySplitNode | None
    right: PolicySplitNode | None
    value: float
    leaf_id: int | None


@dataclass(frozen=True)
class RewardSearch:
    features: np.ndarray
    rewards: np.ndarray
    min_node_size: int
    split_step: int
    control_action: int


class ExactPolicyTree:
    """Globally optimal treat-or-not tree on a fixed reward matrix."""

    def __init__(
        self,
        mode: PolicySearchMode,
        depth: int,
        min_node_size: int,
        split_step: int,
        feature_names: list[str],
    ) -> None:
        self.mode = mode
        self.depth = depth
        self.min_node_size = min_node_size
        self.split_step = split_step
        self.feature_names = list(feature_names)
        self.node: PolicySplitNode | None = None
        self.welfare: float | None = None

    def fit(
        self,
        features: pd.DataFrame,
        rewards: np.ndarray,
        control_action: int,
    ) -> ExactPolicyTree:
        feature_matrix = np.asarray(features.to_numpy(), dtype=float)
        reward_matrix = np.asarray(rewards, dtype=float)
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
            raise ValueError("Exact policy search needs a non-empty covariate matrix.")
        if np.isnan(feature_matrix).any():
            raise ValueError("Exact policy trees do not accept missing covariate values.")
        if reward_matrix.ndim != 2 or reward_matrix.shape[0] != feature_matrix.shape[0]:
            raise ValueError("Rewards must have one row per patient and one column per action.")
        if np.isnan(reward_matrix).any():
            raise ValueError("Rewards contain missing values.")
        if control_action < 0 or control_action >= reward_matrix.shape[1]:
            raise ValueError("control_action is outside the reward matrix.")
        if self.depth not in (1, 2):
            raise ValueError("Exact policy search is implemented for depth 1 and 2.")
        problem = RewardSearch(
            features=feature_matrix,
            rewards=reward_matrix,
            min_node_size=self.min_node_size,
            split_step=self.split_step,
            control_action=control_action,
        )
        indices = np.arange(feature_matrix.shape[0])
        if self.depth == 1:
            fitted = fit_depth_one(problem=problem, indices=indices)
        else:
            fitted = fit_depth_two(problem=problem, indices=indices)
        numbered, leaf_count = number_leaves(node=fitted, next_leaf=0)
        if leaf_count < 1:
            raise RuntimeError("Exact policy search did not produce a leaf.")
        self.node = numbered
        self.welfare = float(numbered.value)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.assign(features=features)[0]

    def apply(self, features: pd.DataFrame) -> np.ndarray:
        return self.assign(features=features)[1]

    def assign(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.node is None:
            raise RuntimeError("Exact policy tree must be fit before prediction.")
        feature_matrix = np.asarray(features.to_numpy(), dtype=float)
        return terminal_assignments(node=self.node, features=feature_matrix)

    def leaf_descriptions(self) -> dict[int, str]:
        if self.node is None:
            raise RuntimeError("Exact policy tree must be fit before describing leaves.")
        return describe_leaves(node=self.node, feature_names=self.feature_names)


def method_name(mode: PolicySearchMode) -> str:
    if mode == PolicySearchMode.WELFARE:
        return EXACT_POLICY_WELFARE_METHOD
    return EXACT_POLICY_CAPITAL_METHOD


def offset_treatment_rewards(
    rewards: np.ndarray,
    minimum_effect: float,
    control_action: int,
) -> np.ndarray:
    """Subtract a minimum benefit from every non-control action.

    A leaf then selects treatment only when its mean reward clears that
    benefit. This is the known treatment-cost offset used by policytree,
    applied to the contrast rewards already stored against control.
    """
    adjusted = np.array(rewards, dtype=float, copy=True)
    if minimum_effect == 0.0:
        return adjusted
    for action in range(adjusted.shape[1]):
        if action != control_action:
            adjusted[:, action] = adjusted[:, action] - minimum_effect
    return adjusted


def capital_inclusion_rewards(
    effects: np.ndarray,
    clinical_threshold: float,
    negative_effect_penalty: float,
) -> np.ndarray:
    """Map estimated effects to the ±1 rewards used by CAPITAL.

    Cai et al. rank patients by excess effect τ − δ and keep the longest
    prefix whose mean excess is still non-negative. Patients inside that
    prefix get reward +1 for inclusion and the rest get −1, so an ordinary
    welfare tree approximates the maximum-coverage subgroup. The optional
    penalty is added afterwards, as in the authors' implementation: it does
    not change who is in the prefix, but it can make a harmful included
    patient costly enough that the tree leaves them out.
    """
    effect_values = np.asarray(effects, dtype=float)
    excess = effect_values - clinical_threshold
    order = np.argsort(-excess, kind="mergesort")
    cumulative_mean = np.cumsum(excess[order]) / np.arange(1, len(effect_values) + 1)
    included = np.zeros(len(effect_values), dtype=bool)
    included[order] = cumulative_mean >= 0.0
    inclusion = np.where(included, 1.0, -1.0)
    if negative_effect_penalty != 0.0:
        harmful = effect_values < 0.0
        inclusion = inclusion + negative_effect_penalty * harmful * effect_values
    return inclusion


def capital_reward_matrix(
    effects: np.ndarray,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    clinical_threshold: float,
    negative_effect_penalty: float,
) -> np.ndarray:
    treated = [value for value in treatment_values if value != control_value]
    if len(treated) != 1:
        raise NotImplementedError("CAPITAL is implemented for binary treatment.")
    rewards = np.zeros((len(effects), len(treatment_values)))
    treated_action = treatment_values.index(treated[0])
    rewards[:, treated_action] = capital_inclusion_rewards(
        effects=effects,
        clinical_threshold=clinical_threshold,
        negative_effect_penalty=negative_effect_penalty,
    )
    return rewards


def fit_depth_one(problem: RewardSearch, indices: np.ndarray) -> PolicySplitNode:
    totals = problem.rewards[indices].sum(axis=0)
    leaf_action, leaf_value = best_action(action_sums=totals, control_action=problem.control_action)
    best = PolicySplitNode(
        feature_index=None,
        threshold=None,
        action=leaf_action,
        left=None,
        right=None,
        value=leaf_value,
        leaf_id=None,
    )
    sample_count = len(indices)
    if sample_count < 2 * problem.min_node_size:
        return best
    indexed_rewards = problem.rewards[indices]
    for feature_index in range(problem.features.shape[1]):
        order = np.argsort(problem.features[indices, feature_index], kind="mergesort")
        ordered_values = problem.features[indices[order], feature_index]
        prefix = np.cumsum(indexed_rewards[order], axis=0)
        cut_positions = np.arange(
            problem.min_node_size,
            sample_count - problem.min_node_size + 1,
            problem.split_step,
        )
        if len(cut_positions) == 0:
            continue
        left_edge = ordered_values[cut_positions - 1]
        right_edge = ordered_values[cut_positions]
        separable = left_edge != right_edge
        left_sums = prefix[cut_positions - 1]
        right_sums = totals - left_sums
        left_actions, left_values = action_values(left_sums, problem.control_action)
        right_actions, right_values = action_values(right_sums, problem.control_action)
        split_values = np.where(separable, left_values + right_values, -np.inf)
        choice = int(np.argmax(split_values))
        chosen_value = float(split_values[choice])
        if chosen_value <= best.value + POLICY_VALUE_TOLERANCE:
            continue
        cut = int(cut_positions[choice])
        best = PolicySplitNode(
            feature_index=feature_index,
            threshold=float(left_edge[choice]),
            action=None,
            left=PolicySplitNode(
                feature_index=None,
                threshold=None,
                action=int(left_actions[choice]),
                left=None,
                right=None,
                value=float(left_values[choice]),
                leaf_id=None,
            ),
            right=PolicySplitNode(
                feature_index=None,
                threshold=None,
                action=int(right_actions[choice]),
                left=None,
                right=None,
                value=float(right_values[choice]),
                leaf_id=None,
            ),
            value=chosen_value,
            leaf_id=None,
        )
    return best


def fit_depth_two(problem: RewardSearch, indices: np.ndarray) -> PolicySplitNode:
    best = fit_depth_one(problem=problem, indices=indices)
    sample_count = len(indices)
    if sample_count < 2 * problem.min_node_size:
        return best
    for feature_index in range(problem.features.shape[1]):
        order = np.argsort(problem.features[indices, feature_index], kind="mergesort")
        ordered_indices = indices[order]
        ordered_values = problem.features[ordered_indices, feature_index]
        cut_positions = np.arange(
            problem.min_node_size,
            sample_count - problem.min_node_size + 1,
            problem.split_step,
        )
        for cut in cut_positions:
            cut = int(cut)
            if ordered_values[cut - 1] == ordered_values[cut]:
                continue
            left_node = fit_depth_one(problem=problem, indices=ordered_indices[:cut])
            right_node = fit_depth_one(problem=problem, indices=ordered_indices[cut:])
            split_value = left_node.value + right_node.value
            if split_value <= best.value + POLICY_VALUE_TOLERANCE:
                continue
            best = PolicySplitNode(
                feature_index=feature_index,
                threshold=float(ordered_values[cut - 1]),
                action=None,
                left=left_node,
                right=right_node,
                value=float(split_value),
                leaf_id=None,
            )
    return best


def best_action(action_sums: np.ndarray, control_action: int) -> tuple[int, float]:
    chosen, value = action_values(
        action_sums=action_sums.reshape(1, -1),
        control_action=control_action,
    )
    return int(chosen[0]), float(value[0])


def action_values(action_sums: np.ndarray, control_action: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose the action with the largest total reward, and control when tied."""
    chosen = np.argmax(action_sums, axis=1)
    row_index = np.arange(action_sums.shape[0])
    chosen_value = action_sums[row_index, chosen]
    control_value = action_sums[:, control_action]
    prefer_control = chosen_value <= control_value + POLICY_VALUE_TOLERANCE
    chosen = np.where(prefer_control, control_action, chosen)
    chosen_value = np.where(prefer_control, control_value, chosen_value)
    return chosen.astype(int), chosen_value.astype(float)


def number_leaves(node: PolicySplitNode, next_leaf: int) -> tuple[PolicySplitNode, int]:
    if node.left is None or node.right is None:
        leaf = PolicySplitNode(
            feature_index=None,
            threshold=None,
            action=node.action,
            left=None,
            right=None,
            value=node.value,
            leaf_id=next_leaf,
        )
        return leaf, next_leaf + 1
    left, next_leaf = number_leaves(node=node.left, next_leaf=next_leaf)
    right, next_leaf = number_leaves(node=node.right, next_leaf=next_leaf)
    numbered = PolicySplitNode(
        feature_index=node.feature_index,
        threshold=node.threshold,
        action=None,
        left=left,
        right=right,
        value=node.value,
        leaf_id=None,
    )
    return numbered, next_leaf


def terminal_assignments(node: PolicySplitNode, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = features.shape[0]
    actions = np.full(count, -1, dtype=int)
    leaves = np.full(count, -1, dtype=int)
    pending: list[tuple[PolicySplitNode, np.ndarray]] = [(node, np.arange(count))]
    while pending:
        current, positions = pending.pop()
        if len(positions) == 0:
            continue
        if current.leaf_id is not None:
            actions[positions] = int(current.action)
            leaves[positions] = int(current.leaf_id)
            continue
        if current.feature_index is None or current.threshold is None:
            raise RuntimeError("Internal policy node is missing its split.")
        if current.left is None or current.right is None:
            raise RuntimeError("Internal policy node is missing a child.")
        values = features[positions, current.feature_index]
        go_left = values <= current.threshold
        pending.append((current.left, positions[go_left]))
        pending.append((current.right, positions[~go_left]))
    if np.any(actions < 0):
        raise RuntimeError("Policy tree did not assign every row.")
    return actions, leaves


def describe_leaves(node: PolicySplitNode, feature_names: list[str]) -> dict[int, str]:
    descriptions: dict[int, str] = {}
    pending: list[tuple[PolicySplitNode, list[str]]] = [(node, [])]
    while pending:
        current, clauses = pending.pop()
        if current.leaf_id is not None:
            descriptions[int(current.leaf_id)] = " and ".join(clauses) if clauses else "all patients"
            continue
        if current.feature_index is None or current.threshold is None:
            raise RuntimeError("Internal policy node is missing its split.")
        if current.left is None or current.right is None:
            raise RuntimeError("Internal policy node is missing a child.")
        feature_name = feature_names[current.feature_index]
        threshold = format(current.threshold, ".6g")
        pending.append((current.left, clauses + [f"{feature_name} <= {threshold}"]))
        pending.append((current.right, clauses + [f"{feature_name} > {threshold}"]))
    return descriptions

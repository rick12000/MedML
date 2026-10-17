"""Decision diagrams for fitted policy trees on learning and held-out data."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.policy import PolicyTree
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeRegressor, _tree

from causal_pipeline.data import CausalDataset
from causal_pipeline.estimator_identity import normalize_estimator_alias
from causal_pipeline.figures import INK, PUBLICATION_RC, TITLE_SIZE
from causal_pipeline.policy import (
    FittedPolicy,
    best_treatment_arm,
    contrast_column_name,
    reward_direction,
)
from causal_pipeline.policy_tree import ExactPolicyTree, PolicySplitNode
from causal_pipeline.utils import save_figure

NODE_WIDTH = 0.38
NODE_HEIGHT = 0.17
FONT_SIZE = 6.5
LINE_WIDTH = 0.9
POLICY_TREE_FIGURE_NAME = "policy_tree.png"


@dataclass(frozen=True)
class TreeNodeLayout:
    node_key: int | str
    center_x: float
    center_y: float
    label: str
    is_leaf: bool
    children: tuple[int | str, int | str] | None


def policy_slice_slug(policy: FittedPolicy) -> str:
    if policy.source_cate_model:
        return normalize_estimator_alias(f"{policy.method}__{policy.source_cate_model}")
    return normalize_estimator_alias(policy.method)


def policy_slice_title(policy: FittedPolicy) -> str:
    if policy.source_cate_model:
        return f"{policy.method} · {policy.source_cate_model}"
    return policy.method


def policy_supports_tree_figure(policy: FittedPolicy) -> bool:
    return policy.method in {
        "policy_tree",
        "dr_policy_tree",
        "exact_policy_welfare",
        "exact_policy_capital",
        "virtual_twins",
    }


def save_policy_slice_figure(
    policy: FittedPolicy,
    estimation: CausalDataset,
    train_scores: pd.DataFrame,
    test: CausalDataset,
    test_scores: pd.DataFrame,
    save_path: str,
) -> None:
    """One trained tree; train and test effects annotated on the same nodes."""
    if not policy_supports_tree_figure(policy):
        return
    title = policy_slice_title(policy)
    with plt.rc_context(PUBLICATION_RC):
        figure, axis = plt.subplots(figsize=(5.8, 5.4))
        draw_frozen_policy_tree(
            policy=policy,
            estimation=estimation,
            train_scores=train_scores,
            test=test,
            test_scores=test_scores,
            ax=axis,
        )
        figure.suptitle(title, fontsize=TITLE_SIZE, y=1.02)
        save_figure(save_path, figure)
        plt.close(figure)


def draw_frozen_policy_tree(
    policy: FittedPolicy,
    estimation: CausalDataset,
    train_scores: pd.DataFrame,
    test: CausalDataset,
    test_scores: pd.DataFrame,
    ax: Axes,
) -> None:
    ax.axis("off")
    train_features = estimation.X_effect_modifiers
    test_features = test.X_effect_modifiers
    if policy.method in {"exact_policy_welfare", "exact_policy_capital"}:
        if not isinstance(policy.model, ExactPolicyTree) or policy.model.node is None:
            return
        layouts = layout_exact_policy_tree(
            root=policy.model.node,
            feature_names=policy.feature_names,
            train_features=train_features,
            test_features=test_features,
            train_scores=train_scores,
            test_scores=test_scores,
            dataset=estimation,
            policy=policy,
        )
    else:
        sklearn_model = sklearn_policy_tree(policy)
        if sklearn_model is None:
            return
        layouts = layout_sklearn_policy_tree(
            tree=sklearn_model,
            feature_names=policy.feature_names,
            train_features=train_features,
            test_features=test_features,
            train_scores=train_scores,
            test_scores=test_scores,
            dataset=estimation,
            policy=policy,
        )
    if not layouts:
        return
    render_tree_layout(ax=ax, layouts=layouts)


def sklearn_policy_tree(policy: FittedPolicy) -> PolicyTree | DecisionTreeRegressor | None:
    if policy.method == "policy_tree" and isinstance(policy.model, PolicyTree):
        return policy.model
    if policy.method == "dr_policy_tree":
        policy_model = getattr(policy.model, "policy_model_", None)
        if isinstance(policy_model, PolicyTree):
            return policy_model
    if policy.method == "virtual_twins" and isinstance(policy.model, DecisionTreeRegressor):
        return policy.model
    return None


def layout_sklearn_policy_tree(
    tree: PolicyTree | DecisionTreeRegressor,
    feature_names: list[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    dataset: CausalDataset,
    policy: FittedPolicy,
) -> list[TreeNodeLayout]:
    sklearn_tree = tree.tree_
    train_path = tree.decision_path(train_features).toarray().astype(bool)
    test_path = tree.decision_path(test_features).toarray().astype(bool)
    positions, children = sklearn_uniform_layout(sklearn_tree=sklearn_tree)
    layouts: list[TreeNodeLayout] = []
    for node_id, (center_x, center_y) in positions.items():
        train_mask = train_path[:, node_id]
        test_mask = test_path[:, node_id]
        feature_index = int(sklearn_tree.feature[node_id])
        is_leaf = sklearn_tree.children_left[node_id] == _tree.TREE_LEAF
        if is_leaf:
            split_text = "leaf"
        else:
            feature_name = feature_names[feature_index]
            threshold = format(float(sklearn_tree.threshold[node_id]), ".6g")
            split_text = f"{feature_name} ≤ {threshold}"
        label = dual_split_node_label(
            split_text=split_text,
            train_mask=train_mask,
            test_mask=test_mask,
            train_scores=train_scores,
            test_scores=test_scores,
            train_features=train_features,
            dataset=dataset,
            policy=policy,
        )
        layouts.append(
            TreeNodeLayout(
                node_key=int(node_id),
                center_x=center_x,
                center_y=center_y,
                label=label,
                is_leaf=is_leaf,
                children=children.get(node_id),
            )
        )
    return layouts


def layout_exact_policy_tree(
    root: PolicySplitNode,
    feature_names: list[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    dataset: CausalDataset,
    policy: FittedPolicy,
) -> list[TreeNodeLayout]:
    train_matrix = np.asarray(train_features.to_numpy(), dtype=float)
    test_matrix = np.asarray(test_features.to_numpy(), dtype=float)
    train_info = exact_node_metadata(node=root, features=train_matrix, feature_names=feature_names)
    test_info = exact_node_metadata(node=root, features=test_matrix, feature_names=feature_names)
    positions, children = exact_uniform_layout(node=root)
    layouts: list[TreeNodeLayout] = []
    for node_key, (center_x, center_y) in positions.items():
        train_mask = np.asarray(train_info[node_key]["mask"], dtype=bool)
        test_mask = np.asarray(test_info[node_key]["mask"], dtype=bool)
        label = dual_split_node_label(
            split_text=str(train_info[node_key]["split_text"]),
            train_mask=train_mask,
            test_mask=test_mask,
            train_scores=train_scores,
            test_scores=test_scores,
            train_features=train_features,
            dataset=dataset,
            policy=policy,
        )
        layouts.append(
            TreeNodeLayout(
                node_key=node_key,
                center_x=center_x,
                center_y=center_y,
                label=label,
                is_leaf=bool(train_info[node_key]["is_leaf"]),
                children=children.get(node_key),
            )
        )
    return layouts


def dual_split_node_label(
    split_text: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    train_features: pd.DataFrame,
    dataset: CausalDataset,
    policy: FittedPolicy,
) -> str:
    _, treatment = subgroup_effect_and_treatment(
        mask=train_mask,
        scores=train_scores,
        dataset=dataset,
        policy=policy,
        features=train_features,
    )
    train_effect = effect_for_recommended_treatment(
        mask=train_mask,
        scores=train_scores,
        dataset=dataset,
        recommended_treatment=treatment,
    )
    test_effect = effect_for_recommended_treatment(
        mask=test_mask,
        scores=test_scores,
        dataset=dataset,
        recommended_treatment=treatment,
    )
    return (
        f"{split_text}\n"
        f"treat = {treatment}\n"
        f"train: n = {int(train_mask.sum())}, effect = {format_effect(train_effect)}\n"
        f"test: n = {int(test_mask.sum())}, effect = {format_effect(test_effect)}"
    )


def subgroup_effect_and_treatment(
    mask: np.ndarray,
    scores: pd.DataFrame,
    dataset: CausalDataset,
    policy: FittedPolicy,
    features: pd.DataFrame,
) -> tuple[float, object]:
    if not np.any(mask):
        return float("nan"), dataset.control_value
    masked_features = features.iloc[mask]
    sklearn_model = sklearn_policy_tree(policy)
    if policy.method in {"exact_policy_welfare", "exact_policy_capital"}:
        if isinstance(policy.model, ExactPolicyTree):
            action = int(policy.model.predict(masked_features)[0])
            treatment = dataset.treatment_values[action]
        else:
            treatment = dataset.control_value
    elif sklearn_model is not None and isinstance(sklearn_model, PolicyTree):
        action = int(sklearn_model.predict(masked_features)[0])
        treatment = dataset.treatment_values[action]
    elif isinstance(sklearn_model, DecisionTreeRegressor):
        direction = reward_direction(dataset.data.outcome_favorability)
        mean_effects = scores.iloc[mask].mean(axis=0)
        treatment = best_treatment_arm(
            mean_effects=mean_effects,
            treatment_values=dataset.treatment_values,
            control_value=dataset.control_value,
            direction=direction,
        )
    else:
        treatment = dataset.control_value
    if treatment == dataset.control_value:
        return 0.0, treatment
    column = contrast_column_name(
        recommended_treatment=treatment,
        control_value=dataset.control_value,
        columns=list(scores.columns),
    )
    return float(scores.loc[mask, column].mean()), treatment


def effect_for_recommended_treatment(
    mask: np.ndarray,
    scores: pd.DataFrame,
    dataset: CausalDataset,
    recommended_treatment: object,
) -> float:
    if not np.any(mask):
        return float("nan")
    if recommended_treatment == dataset.control_value:
        return 0.0
    column = contrast_column_name(
        recommended_treatment=recommended_treatment,
        control_value=dataset.control_value,
        columns=list(scores.columns),
    )
    return float(scores.loc[mask, column].mean())


def sklearn_uniform_layout(
    sklearn_tree: object,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[int, int]]]:
    leaf_counter = 0
    positions: dict[int, tuple[float, float]] = {}
    children: dict[int, tuple[int, int]] = {}

    def visit(node_id: int, depth: int) -> float:
        nonlocal leaf_counter
        if sklearn_tree.children_left[node_id] == _tree.TREE_LEAF:
            center_x = float(leaf_counter)
            leaf_counter += 1
            positions[node_id] = (center_x, -float(depth))
            return center_x
        left_x = visit(int(sklearn_tree.children_left[node_id]), depth + 1)
        right_x = visit(int(sklearn_tree.children_right[node_id]), depth + 1)
        center_x = (left_x + right_x) / 2.0
        positions[node_id] = (center_x, -float(depth))
        children[node_id] = (
            int(sklearn_tree.children_left[node_id]),
            int(sklearn_tree.children_right[node_id]),
        )
        return center_x

    visit(0, 0)
    if not positions:
        return {}, {}
    xs = [position[0] for position in positions.values()]
    span = max(xs) - min(xs) if len(xs) > 1 else 1.0
    scale = 1.0 / span if span > 0 else 1.0
    return (
        {
            node_id: ((position[0] - min(xs)) * scale, position[1])
            for node_id, position in positions.items()
        },
        children,
    )


def exact_uniform_layout(
    node: PolicySplitNode,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[str, str]]]:
    leaf_counter = 0
    positions: dict[str, tuple[float, float]] = {}
    children: dict[str, tuple[str, str]] = {}

    def visit(current: PolicySplitNode, depth: int, node_key: str) -> float:
        nonlocal leaf_counter
        if current.leaf_id is not None:
            center_x = float(leaf_counter)
            leaf_counter += 1
            positions[node_key] = (center_x, -float(depth))
            return center_x
        if current.left is None or current.right is None:
            raise RuntimeError("Exact policy node is missing a child.")
        left_key = f"{node_key}:L"
        right_key = f"{node_key}:R"
        children[node_key] = (left_key, right_key)
        left_x = visit(current.left, depth + 1, left_key)
        right_x = visit(current.right, depth + 1, right_key)
        center_x = (left_x + right_x) / 2.0
        positions[node_key] = (center_x, -float(depth))
        return center_x

    visit(node, 0, "root")
    xs = [position[0] for position in positions.values()]
    span = max(xs) - min(xs) if len(xs) > 1 else 1.0
    scale = 1.0 / span if span > 0 else 1.0
    return (
        {
            node_key: ((position[0] - min(xs)) * scale, position[1])
            for node_key, position in positions.items()
        },
        children,
    )


def exact_node_metadata(
    node: PolicySplitNode,
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, dict[str, object]]:
    count = features.shape[0]
    metadata: dict[str, dict[str, object]] = {}

    def visit(current: PolicySplitNode, positions: np.ndarray, node_key: str, path: list[str]) -> None:
        if current.leaf_id is not None:
            metadata[node_key] = {
                "mask": positions.copy(),
                "split_text": " and ".join(path) if path else "all patients",
                "is_leaf": True,
            }
            return
        if current.feature_index is None or current.threshold is None:
            raise RuntimeError("Internal policy node is missing its split.")
        if current.left is None or current.right is None:
            raise RuntimeError("Internal policy node is missing a child.")
        feature_name = feature_names[current.feature_index]
        threshold = format(current.threshold, ".6g")
        metadata[node_key] = {
            "mask": positions.copy(),
            "split_text": f"{feature_name} ≤ {threshold}",
            "is_leaf": False,
        }
        values = features[positions, current.feature_index]
        go_left = values <= current.threshold
        left_positions = positions[go_left]
        right_positions = positions[~go_left]
        visit(
            current.left,
            left_positions,
            f"{node_key}:L",
            path + [f"{feature_name} ≤ {threshold}"],
        )
        visit(
            current.right,
            right_positions,
            f"{node_key}:R",
            path + [f"{feature_name} > {threshold}"],
        )

    visit(node, np.arange(count), "root", [])
    return metadata


def render_tree_layout(ax: Axes, layouts: list[TreeNodeLayout]) -> None:
    layout_by_key = {layout.node_key: layout for layout in layouts}
    min_y = min(layout.center_y for layout in layouts)
    max_y = max(layout.center_y for layout in layouts)
    y_span = max(max_y - min_y, 1.0)
    for layout in layouts:
        if layout.children is None:
            continue
        for child_key in layout.children:
            child = layout_by_key.get(child_key)
            if child is None:
                continue
            ax.plot(
                [layout.center_x, child.center_x],
                [layout.center_y, child.center_y],
                color=INK,
                linewidth=LINE_WIDTH,
                zorder=1,
            )
    for layout in layouts:
        normalized_y = (layout.center_y - min_y) / y_span
        box = FancyBboxPatch(
            (layout.center_x - NODE_WIDTH / 2.0, normalized_y - NODE_HEIGHT / 2.0),
            NODE_WIDTH,
            NODE_HEIGHT,
            boxstyle="round,pad=0.01",
            linewidth=LINE_WIDTH,
            edgecolor=INK,
            facecolor="white",
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            layout.center_x,
            normalized_y,
            layout.label,
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            color=INK,
            zorder=3,
        )
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.1)


def format_effect(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return format(value, ".4g")

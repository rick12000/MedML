"""Decision diagrams for fitted policy trees, with training and held-out effects per node."""

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

from causal_pipeline.config import JsonValue
from causal_pipeline.data import CausalDataset
from causal_pipeline.estimator_identity import normalize_estimator_alias
from causal_pipeline.figures import INK, PUBLICATION_RC, REFERENCE_COLOR
from causal_pipeline.policy import FittedPolicy, best_treatment_arm, reward_direction, subgroup_benefit
from causal_pipeline.policy_tree import ExactPolicyTree, PolicySplitNode
from causal_pipeline.utils import save_figure

POLICY_TREE_FIGURE_NAME = "policy_tree.png"
TREAT_FILL = "#E3EEF8"
TREAT_EDGE = "#0072B2"
CONTROL_FILL = "#F4F4F4"
CONTROL_EDGE = "#7A7A7A"
BOX_WIDTH = 0.94
BOX_HEIGHT = 0.72
BODY_FONT_SIZE = 8.0
HEADER_FONT_SIZE = 8.6
LEVEL_SPACING_INCHES = 1.25
LEAF_SPACING_INCHES = 1.55
EDGE_LINE_WIDTH = 0.9
BOX_LINE_WIDTH = 0.9
FOOTNOTE = (
    "Δ = benefit of treatment vs control in the node (sign-adjusted; positive favours treatment). "
    "Decisions come from the fitted tree; Δ is computed separately on training and held-out rows."
)

@dataclass(frozen=True)
class DrawnNode:
    key: str
    header: str
    is_leaf: bool
    decision: JsonValue
    train_n: int
    train_benefit: float
    test_n: int
    test_benefit: float
    children: tuple[str, str] | None
    branch_labels: tuple[str, str] | None
    is_treat: bool
    depth: int = 0
    x: float = 0.0


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
    """One fitted tree. Structure and decisions come from training; Δ is shown on both samples."""
    if not policy_supports_tree_figure(policy):
        return
    nodes = build_tree_nodes(
        policy=policy,
        estimation=estimation,
        train_scores=train_scores,
        test=test,
        test_scores=test_scores,
    )
    if not nodes:
        return
    with plt.rc_context(PUBLICATION_RC):
        figure = render_policy_tree(nodes=nodes, title=policy_slice_title(policy))
        save_figure(save_path, figure)
        plt.close(figure)


def build_tree_nodes(
    policy: FittedPolicy,
    estimation: CausalDataset,
    train_scores: pd.DataFrame,
    test: CausalDataset,
    test_scores: pd.DataFrame,
) -> list[DrawnNode]:
    """Collect every node with its train/test masks and decision, using each method's own tree."""
    if policy.method in {"exact_policy_welfare", "exact_policy_capital"}:
        if not isinstance(policy.model, ExactPolicyTree) or policy.model.node is None:
            return []
        specs = exact_tree_specs(
            root=policy.model.node,
            feature_names=policy.feature_names,
            train_features=estimation.X_effect_modifiers,
            test_features=test.X_effect_modifiers,
            train_scores=train_scores,
            dataset=estimation,
        )
    else:
        sklearn_model = sklearn_policy_tree(policy)
        if sklearn_model is None:
            return []
        specs = sklearn_tree_specs(
            tree=sklearn_model,
            feature_names=policy.feature_names,
            train_features=estimation.X_effect_modifiers,
            test_features=test.X_effect_modifiers,
            train_scores=train_scores,
            dataset=estimation,
            policy=policy,
        )
    return assemble_nodes(
        specs=specs,
        estimation=estimation,
        train_scores=train_scores,
        test=test,
        test_scores=test_scores,
    )


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


@dataclass(frozen=True)
class NodeSpec:
    key: str
    header: str
    is_leaf: bool
    children: tuple[str, str] | None
    branch_labels: tuple[str, str] | None
    train_mask: np.ndarray
    test_mask: np.ndarray
    decision: JsonValue
    depth: int


def sklearn_tree_specs(
    tree: PolicyTree | DecisionTreeRegressor,
    feature_names: list[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_scores: pd.DataFrame,
    dataset: CausalDataset,
    policy: FittedPolicy,
) -> list[NodeSpec]:
    sklearn_tree = tree.tree_
    train_path = tree.decision_path(train_features).toarray().astype(bool)
    test_path = tree.decision_path(test_features).toarray().astype(bool)
    levels = dataset.treatment_values
    if policy.method == "dr_policy_tree":
        levels = _control_first_levels(dataset)
    specs: list[NodeSpec] = []

    def visit(node_id: int, depth: int) -> None:
        is_leaf = sklearn_tree.children_left[node_id] == _tree.TREE_LEAF
        if is_leaf:
            header = "Leaf"
            children = None
            branch_labels = None
        else:
            feature = feature_names[int(sklearn_tree.feature[node_id])]
            threshold = format(float(sklearn_tree.threshold[node_id]), ".4g")
            header = f"{feature} ≤ {threshold}"
            children = (str(sklearn_tree.children_left[node_id]), str(sklearn_tree.children_right[node_id]))
            branch_labels = ("≤", ">")
        if policy.method == "virtual_twins":
            decision = _best_arm_for_mask(
                mask=train_path[:, node_id],
                scores=train_scores,
                dataset=dataset,
            )
        else:
            action = int(np.argmax(sklearn_tree.value[node_id][:, 0]))
            decision = levels[action]
        specs.append(
            NodeSpec(
                key=str(node_id),
                header=header,
                is_leaf=is_leaf,
                children=children,
                branch_labels=branch_labels,
                train_mask=train_path[:, node_id],
                test_mask=test_path[:, node_id],
                decision=decision,
                depth=depth,
            )
        )
        if not is_leaf:
            visit(int(sklearn_tree.children_left[node_id]), depth + 1)
            visit(int(sklearn_tree.children_right[node_id]), depth + 1)

    visit(0, 0)
    return specs


def exact_tree_specs(
    root: PolicySplitNode,
    feature_names: list[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_scores: pd.DataFrame,
    dataset: CausalDataset,
) -> list[NodeSpec]:
    train_matrix = np.asarray(train_features.to_numpy(), dtype=float)
    test_matrix = np.asarray(test_features.to_numpy(), dtype=float)
    specs: list[NodeSpec] = []

    def visit(
        current: PolicySplitNode,
        key: str,
        depth: int,
        train_positions: np.ndarray,
        test_positions: np.ndarray,
    ) -> None:
        train_mask = np.zeros(len(train_matrix), dtype=bool)
        train_mask[train_positions] = True
        test_mask = np.zeros(len(test_matrix), dtype=bool)
        test_mask[test_positions] = True
        if current.leaf_id is not None:
            decision = dataset.treatment_values[int(current.action)]
            specs.append(
                NodeSpec(key, "Leaf", True, None, None, train_mask, test_mask, decision, depth)
            )
            return
        if current.feature_index is None or current.threshold is None:
            raise RuntimeError("Internal policy node is missing its split.")
        if current.left is None or current.right is None:
            raise RuntimeError("Internal policy node is missing a child.")
        feature = feature_names[current.feature_index]
        header = f"{feature} ≤ {format(current.threshold, '.4g')}"
        decision = _best_arm_for_mask(mask=train_mask, scores=train_scores, dataset=dataset)
        specs.append(
            NodeSpec(
                key,
                header,
                False,
                (f"{key}L", f"{key}R"),
                ("≤", ">"),
                train_mask,
                test_mask,
                decision,
                depth,
            )
        )
        train_values = train_matrix[train_positions, current.feature_index]
        test_values = test_matrix[test_positions, current.feature_index]
        visit(
            current.left,
            f"{key}L",
            depth + 1,
            train_positions[train_values <= current.threshold],
            test_positions[test_values <= current.threshold],
        )
        visit(
            current.right,
            f"{key}R",
            depth + 1,
            train_positions[train_values > current.threshold],
            test_positions[test_values > current.threshold],
        )

    visit(root, "0", 0, np.arange(len(train_matrix)), np.arange(len(test_matrix)))
    return specs


def assemble_nodes(
    specs: list[NodeSpec],
    estimation: CausalDataset,
    train_scores: pd.DataFrame,
    test: CausalDataset,
    test_scores: pd.DataFrame,
) -> list[DrawnNode]:
    """Attach benefit values from the shared subgroup definition and x-positions for drawing."""
    positions = _leaf_x_positions({spec.key: spec for spec in specs})
    drawn: list[DrawnNode] = []
    for spec in specs:
        train_n = int(spec.train_mask.sum())
        test_n = int(spec.test_mask.sum())
        train_benefit = (
            subgroup_benefit(spec.train_mask, train_scores, estimation, spec.decision).benefit
            if train_n
            else float("nan")
        )
        test_benefit = (
            subgroup_benefit(spec.test_mask, test_scores, test, spec.decision).benefit
            if test_n
            else float("nan")
        )
        drawn.append(
            DrawnNode(
                key=spec.key,
                header=spec.header,
                is_leaf=spec.is_leaf,
                decision=spec.decision,
                train_n=train_n,
                train_benefit=train_benefit,
                test_n=test_n,
                test_benefit=test_benefit,
                children=spec.children,
                branch_labels=spec.branch_labels,
                is_treat=spec.decision != estimation.control_value,
                depth=spec.depth,
                x=positions[spec.key],
            )
        )
    return drawn


def _leaf_x_positions(specs_by_key: dict[str, NodeSpec]) -> dict[str, float]:
    """Leaves get consecutive x-slots; each internal node sits above the midpoint of its children."""
    positions: dict[str, float] = {}
    counter = 0

    def visit(key: str) -> float:
        nonlocal counter
        spec = specs_by_key[key]
        if spec.is_leaf or spec.children is None:
            positions[key] = float(counter)
            counter += 1
            return positions[key]
        left_x = visit(spec.children[0])
        right_x = visit(spec.children[1])
        positions[key] = (left_x + right_x) / 2.0
        return positions[key]

    visit("0")
    return positions


def _control_first_levels(dataset: CausalDataset) -> list[JsonValue]:
    return [dataset.control_value] + [
        value for value in dataset.treatment_values if value != dataset.control_value
    ]


def _best_arm_for_mask(mask: np.ndarray, scores: pd.DataFrame, dataset: CausalDataset) -> JsonValue:
    if not np.any(mask):
        return dataset.control_value
    return best_treatment_arm(
        mean_effects=scores.iloc[mask].mean(axis=0),
        treatment_values=dataset.treatment_values,
        control_value=dataset.control_value,
        direction=reward_direction(dataset.data.outcome_favorability),
    )


def render_policy_tree(nodes: list[DrawnNode], title: str) -> plt.Figure:
    leaf_count = sum(1 for node in nodes if node.is_leaf)
    max_depth = max(node.depth for node in nodes)
    width = max(7.0, LEAF_SPACING_INCHES * leaf_count + 1.6)
    height = LEVEL_SPACING_INCHES * (max_depth + 1) + 1.5
    figure, ax = plt.subplots(figsize=(width, height))
    by_key = {node.key: node for node in nodes}
    xs = [node.x for node in nodes]
    ax.set_xlim(min(xs) - 0.75, max(xs) + 0.75)
    ax.set_ylim(-max_depth - 0.75, 0.75)
    ax.axis("off")

    for node in nodes:
        if node.children is None:
            continue
        parent_bottom = -node.depth - BOX_HEIGHT / 2
        mid_y = -node.depth - 0.5
        for child_key, label in zip(node.children, node.branch_labels, strict=True):
            child = by_key[child_key]
            child_top = -child.depth + BOX_HEIGHT / 2
            ax.plot(
                [node.x, node.x, child.x, child.x],
                [parent_bottom, mid_y, mid_y, child_top],
                color=REFERENCE_COLOR,
                linewidth=EDGE_LINE_WIDTH,
                zorder=1,
                solid_capstyle="round",
            )
            ax.text(
                child.x + (0.1 if child.x >= node.x else -0.1),
                mid_y + 0.08,
                label,
                ha="left" if child.x >= node.x else "right",
                va="bottom",
                fontsize=6.5,
                color=REFERENCE_COLOR,
            )

    for node in nodes:
        draw_node(ax=ax, node=node)

    figure.suptitle(title, fontsize=9.5, color=INK, y=0.985)
    figure.text(0.5, 0.012, FOOTNOTE, ha="center", va="bottom", fontsize=6.2, color=REFERENCE_COLOR, wrap=True)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    return figure


def draw_node(ax: Axes, node: DrawnNode) -> None:
    y = -node.depth
    fill, edge = (TREAT_FILL, TREAT_EDGE) if node.is_treat else (CONTROL_FILL, CONTROL_EDGE)
    box = FancyBboxPatch(
        (node.x - BOX_WIDTH / 2, y - BOX_HEIGHT / 2),
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=BOX_LINE_WIDTH,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)
    decision_text = f"Treat: {node.decision}" if node.is_treat else "Control"
    lines = [
        (node.header, HEADER_FONT_SIZE, "bold"),
        (decision_text, BODY_FONT_SIZE, "normal"),
        (f"Train  n={node.train_n}  Δ={format_benefit(node.train_benefit)}", BODY_FONT_SIZE, "normal"),
        (f"Test   n={node.test_n}  Δ={format_benefit(node.test_benefit)}", BODY_FONT_SIZE, "normal"),
    ]
    spacing = 0.165
    top = y + (len(lines) - 1) * spacing / 2
    for index, (text, size, weight) in enumerate(lines):
        ax.text(
            node.x,
            top - index * spacing,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=INK,
            zorder=3,
        )


def format_benefit(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:+.2f}"

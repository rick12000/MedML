"""Virtual twins stage 2: a regression tree on estimated effects, with debiasing.

Stage 1 is whatever CATE model produced the effects. Foster, Taylor, and
Ruberg (2011) then grow a tree on those effects. Leaves whose absolute mean
effect does not clear a prespecified minimum are dropped. The bootstrap
correction below follows that paper's selection-optimism correction, applied
to the subgroup average treatment effect and estimated with doubly robust
scores rather than a randomized difference in means.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

VIRTUAL_TWINS_METHOD = "virtual_twins"
QUALIFYING_UNION_RULE = "qualifying_union"


class VirtualTwins:
    """Fit a transparent tree to estimated treatment effects and debias its subgroup."""

    def __init__(
        self,
        tree: DecisionTreeRegressor,
        minimum_effect: float,
        bootstrap_samples: int,
        random_state: int,
    ) -> None:
        self.tree = tree
        self.minimum_effect = minimum_effect
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.model: DecisionTreeRegressor | None = None
        self.qualifying_leaves: tuple[int, ...] = ()
        self.debiasing: pd.DataFrame | None = None

    def fit(
        self,
        features: pd.DataFrame,
        effects: pd.DataFrame,
        robust_scores: pd.DataFrame | None,
    ) -> VirtualTwins:
        effect_frame = effects.copy()
        model = clone(self.tree)
        model.fit(features, effect_frame.to_numpy(dtype=float))
        qualifying = qualifying_leaf_ids(
            model=model,
            features=features,
            effects=effect_frame.to_numpy(dtype=float),
            minimum_effect=self.minimum_effect,
        )
        self.model = model
        self.qualifying_leaves = tuple(int(leaf) for leaf in qualifying)
        if robust_scores is None or len(self.qualifying_leaves) == 0:
            self.debiasing = None
            return self
        self.debiasing = debias_qualifying_subgroup(
            fitted=model,
            tree=self.tree,
            features=features,
            effects=effect_frame,
            robust_scores=robust_scores,
            minimum_effect=self.minimum_effect,
            bootstrap_samples=self.bootstrap_samples,
            random_state=self.random_state,
        )
        return self


def qualifying_leaf_ids(
    model: DecisionTreeRegressor,
    features: pd.DataFrame,
    effects: np.ndarray,
    minimum_effect: float,
) -> np.ndarray:
    """Leaves whose largest absolute mean contrast exceeds the minimum effect."""
    leaf_ids = np.asarray(model.apply(features))
    effect_matrix = np.asarray(effects, dtype=float)
    if effect_matrix.ndim == 1:
        effect_matrix = effect_matrix.reshape(-1, 1)
    retained: list[int] = []
    for leaf in np.unique(leaf_ids):
        mean_effects = effect_matrix[leaf_ids == leaf].mean(axis=0)
        if float(np.max(np.abs(mean_effects))) > minimum_effect:
            retained.append(int(leaf))
    return np.asarray(retained, dtype=int)


def debias_qualifying_subgroup(
    fitted: DecisionTreeRegressor,
    tree: DecisionTreeRegressor,
    features: pd.DataFrame,
    effects: pd.DataFrame,
    robust_scores: pd.DataFrame,
    minimum_effect: float,
    bootstrap_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """Subtract bootstrap selection optimism from the doubly robust subgroup effect.

    Each replicate refits only the tree. The optimistic score is the doubly
    robust mean inside the subgroup that replicate just selected. The same
    frozen leaf rule, scored on the original sample, does not contain that
    replicate's selection step. Their difference estimates the optimism.
    """
    if bootstrap_samples < 1:
        raise ValueError("Virtual twins debiasing needs at least one bootstrap replicate.")
    effect_matrix = effects.to_numpy(dtype=float)
    feature_frame = features.reset_index(drop=True)
    original_leaves = qualifying_leaf_ids(
        model=fitted,
        features=feature_frame,
        effects=effect_matrix,
        minimum_effect=minimum_effect,
    )
    original_mask = np.isin(np.asarray(fitted.apply(feature_frame)), original_leaves)
    if not np.any(original_mask):
        return pd.DataFrame(columns=["contrast", "naive_effect", "bootstrap_bias", "debiased_effect"])
    generator = np.random.default_rng(random_state)
    sample_size = len(feature_frame)
    rows: list[dict[str, float | str]] = []
    for contrast in effects.columns:
        if contrast not in robust_scores.columns:
            continue
        scores = robust_scores[contrast].to_numpy(dtype=float)
        naive_effect = float(np.mean(scores[original_mask]))
        optimism: list[float] = []
        for replicate in range(bootstrap_samples):
            drawn = generator.choice(sample_size, size=sample_size, replace=True)
            bootstrap_model = clone(tree)
            bootstrap_model.fit(feature_frame.iloc[drawn], effect_matrix[drawn])
            bootstrap_leaves = qualifying_leaf_ids(
                model=bootstrap_model,
                features=feature_frame.iloc[drawn],
                effects=effect_matrix[drawn],
                minimum_effect=minimum_effect,
            )
            if len(bootstrap_leaves) == 0:
                continue
            bootstrap_membership = np.asarray(bootstrap_model.apply(feature_frame.iloc[drawn]))
            bootstrap_mask = np.isin(bootstrap_membership, bootstrap_leaves)
            original_membership = np.asarray(bootstrap_model.apply(feature_frame))
            evaluation_mask = np.isin(original_membership, bootstrap_leaves)
            if not np.any(bootstrap_mask) or not np.any(evaluation_mask):
                continue
            optimistic = float(np.mean(scores[drawn][bootstrap_mask]))
            evaluated = float(np.mean(scores[evaluation_mask]))
            optimism.append(optimistic - evaluated)
        if optimism:
            bias = float(np.mean(optimism))
            corrected = naive_effect - bias
        else:
            bias = float("nan")
            corrected = float("nan")
        rows.append(
            {
                "contrast": contrast,
                "naive_effect": naive_effect,
                "bootstrap_bias": bias,
                "debiased_effect": corrected,
            }
        )
        logger.info(
            "Virtual twins debiasing for %s used %s replicates. Bias %.4f.",
            contrast,
            len(optimism),
            bias,
        )
    return pd.DataFrame(rows)

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from causal_pipeline.virtual_twins import VirtualTwins, qualifying_leaf_ids


def test_minimum_effect_filter_drops_a_null_leaf() -> None:
    features = pd.DataFrame({"age": [0, 0, 0, 1, 1, 1]})
    effects = pd.DataFrame({"1_vs_0": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]})
    tree = DecisionTreeRegressor(max_depth=1, random_state=0)
    tree.fit(features, effects.to_numpy())
    retained = qualifying_leaf_ids(
        model=tree,
        features=features,
        effects=effects.to_numpy(),
        minimum_effect=0.2,
    )
    leaf_ids = np.asarray(tree.apply(features))
    retained_effects = effects.to_numpy()[np.isin(leaf_ids, retained)]
    assert len(retained) == 1
    assert np.all(np.abs(retained_effects) > 0.2)


def test_bootstrap_debiasing_is_zero_when_the_subgroup_cannot_change() -> None:
    sample_size = 20
    features = pd.DataFrame({"age": np.arange(sample_size, dtype=float)})
    effects = pd.DataFrame({"1_vs_0": np.full(sample_size, 0.4)})
    scores = pd.DataFrame({"1_vs_0": np.full(sample_size, 0.4)})
    twins = VirtualTwins(
        tree=DecisionTreeRegressor(min_samples_split=sample_size + 1, random_state=0),
        minimum_effect=0.0,
        bootstrap_samples=15,
        random_state=0,
    )
    twins.fit(features=features, effects=effects, robust_scores=scores)
    assert twins.debiasing is not None
    assert twins.debiasing.shape[0] == 1
    row = twins.debiasing.iloc[0]
    assert np.isclose(row["naive_effect"], 0.4)
    assert np.isclose(row["bootstrap_bias"], 0.0)
    assert np.isclose(row["debiased_effect"], 0.4)

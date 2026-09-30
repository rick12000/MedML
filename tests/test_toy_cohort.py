from __future__ import annotations

import pytest

from settings import (
    CONFOUNDERS,
    EFFECT_MODIFIERS,
    GROUP_COLUMN,
    OUTCOME_COLUMN,
    SELECTION_COLUMN,
    TREATMENT_COLUMN,
)
from generate_toy_cohort import generate_observational_cohort

TOY_COHORT_RANDOM_STATE = 0


@pytest.mark.parametrize("n_observations", [10, 1000])
def test_toy_cohort_matches_pipeline_column_contract(n_observations: int) -> None:
    df = generate_observational_cohort(
        n_observations,
        random_state=TOY_COHORT_RANDOM_STATE,
    )
    required = [
        OUTCOME_COLUMN,
        TREATMENT_COLUMN,
        SELECTION_COLUMN,
        GROUP_COLUMN,
        *CONFOUNDERS,
        *EFFECT_MODIFIERS,
    ]
    assert df.shape[0] == n_observations
    assert len(set(required)) == 15
    assert set(required).issubset(df.columns)
    assert not df[required].isna().any().any()
    assert set(df[TREATMENT_COLUMN].unique()).issubset({0, 1})
    assert set(df[OUTCOME_COLUMN].unique()).issubset({0, 1})
    assert set(df[SELECTION_COLUMN].unique()).issubset({0, 1})

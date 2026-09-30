import pytest

from causal_pipeline.settings import (
    DATA_CONFOUNDERS,
    DATA_EFFECT_MODIFIERS,
    DATA_OUTCOME_COLUMN,
    DATA_TREATMENT_COLUMN,
)
from generate_toy_cohort import generate_observational_cohort


@pytest.mark.parametrize("n_observations", [10, 1000])
def test_toy_cohort_matches_pipeline_column_contract(n_observations: int) -> None:
    df = generate_observational_cohort(n_observations, random_state=0)
    required = [
        DATA_OUTCOME_COLUMN,
        DATA_TREATMENT_COLUMN,
        *DATA_CONFOUNDERS,
        *DATA_EFFECT_MODIFIERS,
    ]
    assert df.shape[0] == n_observations
    assert set(required).issubset(df.columns)
    assert not df[required].isna().any().any()
    assert set(df[DATA_TREATMENT_COLUMN].unique()).issubset({0, 1})
    assert set(df[DATA_OUTCOME_COLUMN].unique()).issubset({0, 1})

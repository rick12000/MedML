from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from causallib.estimation import Standardization
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from causal_pipeline.ate import (
    contrasts_from_population_outcomes,
    initialize_ate_estimator,
    doubleml_dataframe,
    factual_outcome_predictions,
    treated_propensity,
)
from causal_pipeline.config import (
    ATEKind,
    DataConfig,
    DoublyRobustATEEstimatorSpec,
    IPWATEEstimatorSpec,
    PipelineConfig,
)
from causal_pipeline.data import CausalDataset, DataSplitter


def test_doubleml_dataframe_aligns_non_contiguous_indexes() -> None:
    index = pd.Index([10, 20, 30, 40])
    outcome = pd.Series([0, 1, 0, 1], index=index)
    treatment = pd.Series([0, 1, 1, 0], index=index)
    confounders = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4]}, index=index)
    table = doubleml_dataframe(
        outcome=outcome,
        treatment=treatment,
        confounders=confounders,
    )
    assert table.shape == (4, 3)
    assert list(table.columns) == ["y", "d", "x1"]
    assert not table.isna().any().any()


def test_treated_propensity_selects_treated_column_from_matrix() -> None:
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    selected = treated_propensity(matrix)
    assert selected.shape == (2,)
    assert np.allclose(selected, [0.2, 0.7])


def test_treated_propensity_passes_through_one_dimensional_scores() -> None:
    scores = np.array([0.2, 0.7, 0.4])
    selected = treated_propensity(scores)
    assert selected.shape == scores.shape
    assert np.allclose(selected, scores)


def test_factual_outcome_predictions_match_observed_arm() -> None:
    n_observations = 40
    rng = np.random.default_rng(0)
    covariates = pd.DataFrame({"x1": rng.normal(size=n_observations)})
    treatment = pd.Series(rng.integers(0, 2, size=n_observations))
    outcome = pd.Series(rng.normal(size=n_observations))
    model = Standardization(LinearRegression(), encode_treatment=True)
    model.fit(covariates, treatment, outcome)
    factual = factual_outcome_predictions(
        outcome_model=model,
        X=covariates,
        treatment=treatment,
        treatment_values=[0, 1],
    )
    potential = np.asarray(model.estimate_individual_outcome(covariates, treatment))
    expected = potential[np.arange(n_observations), treatment.to_numpy()]
    assert factual.shape == (n_observations,)
    assert np.allclose(factual, expected)


def test_ipw_estimate_is_finite_after_split(
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    df_synthetic_binary: pd.DataFrame,
    pipeline_config_no_policy: PipelineConfig,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    partitions = DataSplitter().split(dataset=dataset, config=pipeline_config_no_policy)
    spec = IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner)
    estimator = initialize_ate_estimator(spec=spec, data=partitions.estimation)
    estimator.fit(data=partitions.estimation)
    table = estimator.estimate()
    assert table.shape[0] == 1
    assert list(table.columns) == ["contrast", "estimand", "estimate", "ci_lower", "ci_upper"]
    assert np.isfinite(table["estimate"].iloc[0])
    assert table["ci_lower"].iloc[0] <= table["estimate"].iloc[0] <= table["ci_upper"].iloc[0]


def test_aipw_estimate_is_finite_on_binary_outcome(
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    forest_classifier: BaseEstimator,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    spec = DoublyRobustATEEstimatorSpec(
        kind=ATEKind.AIPW,
        outcome_learner=forest_classifier,
        propensity_learner=logistic_learner,
    )
    estimator = initialize_ate_estimator(spec=spec, data=dataset)
    estimator.fit(data=dataset)
    table = estimator.estimate()
    assert table.shape[0] == 1
    assert np.isfinite(table["estimate"].iloc[0])
    assert estimator.outcome_predictions is not None
    assert estimator.outcome_predictions.shape == (len(df_synthetic_binary),)


@pytest.mark.parametrize("index_key", [1, "1"])
def test_population_contrasts_accept_integer_or_string_index(index_key: int | str) -> None:
    population = pd.Series({0: 0.2, index_key: 0.5})
    table = contrasts_from_population_outcomes(
        population=population,
        control_value=0,
        treatment_values=[0, 1],
    )
    assert table.shape[0] == 1
    assert list(table["contrast"]) == ["1_vs_0"]
    assert np.isclose(table["estimate"].iloc[0], 0.3)

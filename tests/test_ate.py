from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from causallib.estimation import Standardization
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from causal_pipeline.ate import (
    DoubleMLIRMAdapter,
    contrasts_from_population_outcomes,
    doubleml_dataframe,
    factual_outcome_predictions,
    initialize_ate_estimator,
    treated_propensity,
)
from causal_pipeline.config import (
    ATEKind,
    DataConfig,
    DoubleMLATEEstimatorSpec,
    DoublyRobustATEEstimatorSpec,
    IPWATEEstimatorSpec,
    OutcomeType,
    TreatmentMode,
)
from causal_pipeline.data import CausalDataset


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
        treatment_arms=[0, 1],
    )
    potential = np.asarray(model.estimate_individual_outcome(covariates, treatment))
    expected = potential[np.arange(n_observations), treatment.to_numpy()]
    assert factual.shape == (n_observations,)
    assert np.allclose(factual, expected)


def test_doubleml_irm_rejects_multi_arm_treatment(
    logistic_learner: BaseEstimator,
) -> None:
    config = DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1"],
        effect_modifiers=["x1"],
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.MULTI,
        control_value=0,
        treatment_values=[0, 1, 2],
    )
    frame = pd.DataFrame(
        {
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "t": [0, 1, 2, 0, 1, 2],
            "x1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    dataset = CausalDataset(data=config, df=frame)
    estimator = DoubleMLIRMAdapter(
        spec=DoubleMLATEEstimatorSpec(
            kind=ATEKind.DML_IRM,
            outcome_learner=LinearRegression(),
            propensity_learner=logistic_learner,
        ),
        data=dataset,
    )
    with pytest.raises(ValueError):
        estimator.fit(data=dataset)


ATE_RECOVERY_TOLERANCE = 0.05
RECOVERY_FOLDS = 5


@pytest.mark.parametrize("architecture", ["linear", "forest"])
@pytest.mark.parametrize("kind", list(ATEKind))
def test_ate_recovers_a_known_constant_effect(
    kind: ATEKind,
    architecture: str,
    recovery_data_config: DataConfig,
    constant_effect_cohort: tuple[pd.DataFrame, float],
    linear_learner: BaseEstimator,
    logistic_learner: BaseEstimator,
    recovery_regressor: BaseEstimator,
    recovery_classifier: BaseEstimator,
) -> None:
    frame, truth = constant_effect_cohort
    dataset = CausalDataset(data=recovery_data_config, df=frame)
    outcome_learner = linear_learner if architecture == "linear" else recovery_regressor
    propensity_learner = logistic_learner if architecture == "linear" else recovery_classifier
    if kind == ATEKind.IPW:
        spec = IPWATEEstimatorSpec(
            kind=kind,
            propensity_learner=propensity_learner,
            n_folds=RECOVERY_FOLDS,
        )
    elif kind in {ATEKind.AIPW, ATEKind.TMLE}:
        spec = DoublyRobustATEEstimatorSpec(
            kind=kind,
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            n_folds=RECOVERY_FOLDS,
        )
    else:
        spec = DoubleMLATEEstimatorSpec(
            kind=kind,
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            n_folds=RECOVERY_FOLDS,
        )
    estimator = initialize_ate_estimator(spec=spec, data=dataset)
    # DoubleML draws RepeatedKFold splits from NumPy's global RNG.
    np.random.seed(0)
    estimator.fit(data=dataset)
    table = estimator.estimate()
    estimate = float(table["estimate"].iloc[0])
    assert table.shape[0] == 1
    assert np.isfinite(estimate)
    assert table["ci_lower"].iloc[0] <= estimate <= table["ci_upper"].iloc[0]
    assert abs(estimate - truth) < ATE_RECOVERY_TOLERANCE


@pytest.mark.parametrize("index_key", [1, "1"])
def test_population_contrasts_accept_integer_or_string_index(index_key: int | str) -> None:
    population = pd.Series({0: 0.2, index_key: 0.5})
    table = contrasts_from_population_outcomes(
        population=population,
        control_value=0,
        treatment_values=[0, 1],
        estimand="ate",
    )
    assert table.shape[0] == 1
    assert list(table["contrast"]) == ["1_vs_0"]
    assert np.isclose(table["estimate"].iloc[0], 0.3)

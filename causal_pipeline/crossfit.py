"""Cross-fit nuisances and influence-function intervals for ATE estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from causallib.estimation import IPW, Standardization
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from causal_pipeline.config import JsonValue, clone_estimator

PROBABILITY_CLIP_FLOOR = 1e-6
LOGIT_CLIP = 1e-6
MINIMUM_CROSSFIT_FOLDS = 2


def normal_interval_from_scores(scores: np.ndarray, level: float) -> tuple[float, float, float]:
    """Mean of an influence function and its normal confidence interval."""
    estimate = float(np.mean(scores))
    standard_error = float(np.std(scores, ddof=1) / np.sqrt(scores.size))
    quantile = float(stats.norm.ppf(0.5 + level / 2.0))
    return estimate, estimate - quantile * standard_error, estimate + quantile * standard_error


def cross_fit_splits(
    treatment: pd.Series,
    n_folds: int,
    random_state: int,
) -> StratifiedKFold:
    arm_counts = treatment.value_counts()
    n_splits = min(n_folds, int(arm_counts.min()))
    if n_splits < MINIMUM_CROSSFIT_FOLDS:
        raise ValueError(
            "Cross-fitting requires at least two observations in every treatment arm."
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def learner_random_state(learner: object) -> int:
    state = getattr(learner, "random_state", None)
    if isinstance(state, int):
        return state
    return 0


def cross_fit_nuisances(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    treatment_values: list[JsonValue],
    propensity_learner: object,
    outcome_learner: object | None,
    binary_outcome: bool,
    n_folds: int,
    random_state: int,
    clip_bounds: tuple[float, float],
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Out-of-fold potential outcomes and propensity, columns aligned to treatment_values."""
    splitter = cross_fit_splits(treatment, n_folds, random_state)
    propensity = pd.DataFrame(np.nan, index=X.index, columns=treatment_values)
    outcome_predictions = None
    if outcome_learner is not None:
        outcome_predictions = pd.DataFrame(np.nan, index=X.index, columns=treatment_values)

    for train_index, test_index in splitter.split(X, treatment):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        treatment_train = treatment.iloc[train_index]
        treatment_test = treatment.iloc[test_index]
        propensity.iloc[test_index] = _fold_propensity(
            X_train=X_train,
            treatment_train=treatment_train,
            X_test=X_test,
            propensity_learner=propensity_learner,
            treatment_values=treatment_values,
            clip_bounds=clip_bounds,
        ).to_numpy()
        if outcome_learner is None or outcome_predictions is None:
            continue
        outcome_predictions.iloc[test_index] = _fold_potential_outcomes(
            X_train=X_train,
            treatment_train=treatment_train,
            outcome_train=outcome.iloc[train_index],
            X_test=X_test,
            treatment_test=treatment_test,
            outcome_learner=outcome_learner,
            treatment_values=treatment_values,
            binary_outcome=binary_outcome,
        ).to_numpy()
    return outcome_predictions, propensity


def _fold_propensity(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    X_test: pd.DataFrame,
    propensity_learner: object,
    treatment_values: list[JsonValue],
    clip_bounds: tuple[float, float],
) -> pd.DataFrame:
    model = IPW(
        learner=clone_estimator(propensity_learner),
        clip_min=clip_bounds[0],
        clip_max=clip_bounds[1],
    )
    model.fit(X_train, treatment_train)
    matrix = model.compute_propensity_matrix(X_test)
    aligned = _align_treatment_columns(matrix, treatment_values)
    return aligned.div(aligned.sum(axis=1), axis=0)


def _fold_potential_outcomes(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    outcome_train: pd.Series,
    X_test: pd.DataFrame,
    treatment_test: pd.Series,
    outcome_learner: object,
    treatment_values: list[JsonValue],
    binary_outcome: bool,
) -> pd.DataFrame:
    model = Standardization(
        clone_estimator(outcome_learner),
        encode_treatment=True,
        predict_proba=binary_outcome,
    )
    model.fit(X_train, treatment_train, outcome_train)
    raw = model.estimate_individual_outcome(
        X_test,
        treatment_test,
        treatment_values=treatment_values,
        predict_proba=binary_outcome,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        outcome_level = raw.columns.get_level_values(-1).max()
        raw = raw.xs(outcome_level, axis="columns", level=-1)
    return _align_treatment_columns(raw, treatment_values)


def _align_treatment_columns(frame: pd.DataFrame, treatment_values: list[JsonValue]) -> pd.DataFrame:
    renamed = frame.copy()
    if not set(treatment_values).issubset(set(renamed.columns)):
        renamed.columns = [_match_treatment_label(column, treatment_values) for column in renamed.columns]
    missing = [value for value in treatment_values if value not in renamed.columns]
    if missing:
        raise KeyError(f"Nuisance predictions are missing treatment arms: {missing}")
    return renamed.loc[:, treatment_values].reset_index(drop=True)


def _match_treatment_label(column: object, treatment_values: list[JsonValue]) -> JsonValue:
    for value in treatment_values:
        if column == value or str(column) == str(value):
            return value
    return column


def factual_predictions(
    potential_outcomes: pd.DataFrame,
    treatment: pd.Series,
) -> np.ndarray:
    rows = np.arange(len(treatment))
    columns = potential_outcomes.columns.get_indexer(treatment.to_numpy())
    if np.any(columns < 0):
        raise KeyError("Observed treatment is missing from the potential-outcome table.")
    return potential_outcomes.to_numpy()[rows, columns]


def hajek_influence(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    arm: JsonValue,
) -> np.ndarray:
    """Influence function of the Hajek mean of Y under arm assignment."""
    observed = treatment == arm
    weight = observed.astype(float) / propensity
    weight_sum = float(np.sum(weight))
    if weight_sum == 0.0:
        raise ValueError(f"Hajek weights for arm {arm} sum to zero.")
    arm_mean = float(np.sum(weight * outcome) / weight_sum)
    return arm_mean + outcome.size * weight * (outcome - arm_mean) / weight_sum


def aipw_arm_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    potential_outcome: np.ndarray,
    propensity: np.ndarray,
    arm: JsonValue,
) -> np.ndarray:
    observed = treatment == arm
    return potential_outcome + observed.astype(float) * (outcome - potential_outcome) / propensity


def clip_propensity(propensity: pd.DataFrame, clip_bounds: tuple[float, float]) -> pd.DataFrame:
    clipped = propensity.clip(lower=clip_bounds[0], upper=clip_bounds[1])
    if clipped.shape[1] >= 2:
        clipped = clipped.div(clipped.sum(axis=1), axis=0)
        clipped = clipped.clip(lower=PROBABILITY_CLIP_FLOOR)
        clipped = clipped.div(clipped.sum(axis=1), axis=0)
    return clipped


def target_potential_outcomes(
    outcome: pd.Series,
    treatment: pd.Series,
    potential_outcomes: pd.DataFrame,
    propensity: pd.DataFrame,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    reduced: bool,
) -> pd.DataFrame:
    """Cross-fit TMLE update. Initial Q and g are already out-of-fold."""
    if reduced and len(treatment_values) != 2:
        raise ValueError("Reduced TMLE requires binary treatment.")
    outcome_values = outcome.to_numpy(dtype=float)
    scale_min = float(np.min(outcome_values))
    scale_max = float(np.max(outcome_values))
    scale = scale_max - scale_min
    if scale == 0.0:
        return potential_outcomes.copy()
    outcome_scaled = (outcome_values - scale_min) / scale
    initial = potential_outcomes.clip(lower=LOGIT_CLIP, upper=1.0 - LOGIT_CLIP)
    initial_scaled = (initial - scale_min) / scale
    initial_scaled = initial_scaled.clip(lower=LOGIT_CLIP, upper=1.0 - LOGIT_CLIP)
    observed_q = factual_predictions(initial_scaled, treatment)
    offset = _logit(observed_q)
    clever_fit = _clever_covariate(
        treatment=treatment.to_numpy(),
        propensity=propensity,
        treatment_values=treatment_values,
        control_value=control_value,
        reduced=reduced,
        arm=None,
    )
    fluctuation = sm.GLM(
        outcome_scaled,
        clever_fit,
        offset=offset,
        family=sm.families.Binomial(),
    ).fit(maxiter=100)
    updated = pd.DataFrame(index=potential_outcomes.index, columns=treatment_values, dtype=float)
    for arm in treatment_values:
        clever_arm = _clever_covariate(
            treatment=treatment.to_numpy(),
            propensity=propensity,
            treatment_values=treatment_values,
            control_value=control_value,
            reduced=reduced,
            arm=arm,
        )
        linear = _logit(initial_scaled[arm].to_numpy()) + clever_arm.to_numpy() @ fluctuation.params.to_numpy()
        updated[arm] = scale_min + scale * _expit(linear)
    return updated


def _clever_covariate(
    treatment: np.ndarray,
    propensity: pd.DataFrame,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    reduced: bool,
    arm: JsonValue | None,
) -> pd.DataFrame:
    if reduced:
        treated = next(value for value in treatment_values if value != control_value)
        treated_weight = 1.0 / propensity[treated].to_numpy()
        control_weight = 1.0 / propensity[control_value].to_numpy()
        if arm is None:
            observed_treated = treatment == treated
            values = np.where(observed_treated, treated_weight, -control_weight)
        elif arm == treated:
            values = treated_weight
        else:
            values = -control_weight
        return pd.DataFrame({"clever": values})
    columns = {}
    for value in treatment_values:
        weight = 1.0 / propensity[value].to_numpy()
        if arm is None:
            columns[str(value)] = (treatment == value).astype(float) * weight
        elif value == arm:
            columns[str(value)] = weight
        else:
            columns[str(value)] = np.zeros(len(treatment))
    return pd.DataFrame(columns)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(clipped / (1.0 - clipped))


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def cross_fit_propensity_map(
    X: pd.DataFrame,
    treatment: pd.Series,
    learner: object,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    n_folds: int,
    random_state: int,
    clip_bounds: tuple[float, float],
) -> dict[JsonValue, np.ndarray]:
    """Out-of-fold P(T=a | X) for each non-control arm, clipped to (0, 1)."""
    _, propensity = cross_fit_nuisances(
        X=X,
        treatment=treatment,
        outcome=treatment,
        treatment_values=treatment_values,
        propensity_learner=learner,
        outcome_learner=None,
        binary_outcome=False,
        n_folds=n_folds,
        random_state=random_state,
        clip_bounds=clip_bounds,
    )
    clipped = clip_propensity(propensity, clip_bounds)
    arms = [arm for arm in clipped.columns if arm != control_value]
    return {arm: clipped[arm].to_numpy(dtype=float) for arm in arms}

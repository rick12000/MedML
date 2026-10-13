"""Default sklearn and XGBoost learners for causal nuisance and effect models."""

from __future__ import annotations

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from causal_pipeline.config import OutcomeType, clone_estimator

DEFAULT_LOGISTIC_MAX_ITER: int = 10_000
DEFAULT_LOGISTIC_C: float = 1.0
DEFAULT_RIDGE_ALPHA: float = 1.0
DEFAULT_XGBOOST_N_ESTIMATORS: int = 300
DEFAULT_XGBOOST_MAX_DEPTH: int = 4
DEFAULT_XGBOOST_LEARNING_RATE: float = 0.08
DEFAULT_XGBOOST_SUBSAMPLE: float = 0.85
DEFAULT_XGBOOST_COLSAMPLE: float = 0.85
DEFAULT_XGBOOST_MIN_CHILD_WEIGHT: int = 5
DEFAULT_XGBOOST_REG_LAMBDA: float = 1.0
DEFAULT_XGBOOST_N_JOBS: int = 1


def default_propensity_classifier(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=DEFAULT_XGBOOST_N_ESTIMATORS,
        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
        learning_rate=DEFAULT_XGBOOST_LEARNING_RATE,
        subsample=DEFAULT_XGBOOST_SUBSAMPLE,
        colsample_bytree=DEFAULT_XGBOOST_COLSAMPLE,
        min_child_weight=DEFAULT_XGBOOST_MIN_CHILD_WEIGHT,
        reg_lambda=DEFAULT_XGBOOST_REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
    )


def default_outcome_regressor(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=DEFAULT_XGBOOST_N_ESTIMATORS,
        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
        learning_rate=DEFAULT_XGBOOST_LEARNING_RATE,
        subsample=DEFAULT_XGBOOST_SUBSAMPLE,
        colsample_bytree=DEFAULT_XGBOOST_COLSAMPLE,
        min_child_weight=DEFAULT_XGBOOST_MIN_CHILD_WEIGHT,
        reg_lambda=DEFAULT_XGBOOST_REG_LAMBDA,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
    )


def default_outcome_classifier(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=DEFAULT_XGBOOST_N_ESTIMATORS,
        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
        learning_rate=DEFAULT_XGBOOST_LEARNING_RATE,
        subsample=DEFAULT_XGBOOST_SUBSAMPLE,
        colsample_bytree=DEFAULT_XGBOOST_COLSAMPLE,
        min_child_weight=DEFAULT_XGBOOST_MIN_CHILD_WEIGHT,
        reg_lambda=DEFAULT_XGBOOST_REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
    )


def default_effect_regressor(random_state: int) -> XGBRegressor:
    return default_outcome_regressor(random_state=random_state)


def apply_defaults(estimator: BaseEstimator) -> BaseEstimator:
    """Return a copy with stronger defaults for small-sample causal workloads."""
    model = clone(estimator)
    name = type(model).__name__
    if name == "LogisticRegression":
        params: dict[str, int | float | str] = {"max_iter": DEFAULT_LOGISTIC_MAX_ITER}
        if model.get_params().get("C") == 1.0:
            params["C"] = DEFAULT_LOGISTIC_C
        model.set_params(**params)
        return model
    if name == "Ridge":
        if model.get_params().get("alpha") == 1.0:
            model.set_params(alpha=DEFAULT_RIDGE_ALPHA)
        return model
    if name in {"XGBClassifier", "XGBRegressor"}:
        return model
    if name == "RandomForestClassifier":
        return default_propensity_classifier(
            random_state=int(model.get_params().get("random_state") or 0),
        )
    if name == "RandomForestRegressor":
        return default_outcome_regressor(
            random_state=int(model.get_params().get("random_state") or 0),
        )
    if name == "LinearRegression":
        return Ridge(alpha=DEFAULT_RIDGE_ALPHA)
    return model


def materialize_propensity_learner(
    learner: BaseEstimator | None,
    random_state: int,
) -> BaseEstimator:
    if learner is None:
        return default_propensity_classifier(random_state=random_state)
    return apply_defaults(clone_estimator(learner))


def materialize_outcome_learner(
    learner: BaseEstimator | None,
    outcome_type: OutcomeType,
    random_state: int,
) -> BaseEstimator:
    if learner is None:
        if outcome_type == OutcomeType.BINARY:
            return default_outcome_classifier(random_state=random_state)
        return default_outcome_regressor(random_state=random_state)
    return apply_defaults(clone_estimator(learner))


def materialize_effect_learner(
    learner: BaseEstimator | None,
    random_state: int,
) -> BaseEstimator:
    if learner is None:
        return default_effect_regressor(random_state=random_state)
    return apply_defaults(clone_estimator(learner))

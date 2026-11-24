"""Pre-approved learner architectures. Users name an architecture; this module builds the estimator.

Presets are single-threaded on purpose. Parallelism happens at the process-pool level
(see parallel.py), so estimator-level threads would only oversubscribe the machine.
"""

from __future__ import annotations

from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

LearnerArchitecture = Literal["linear", "random_forest", "xgboost"]
LearnerTask = Literal["classification", "regression"]

DEFAULT_LOGISTIC_MAX_ITER: int = 2_000
DEFAULT_LOGISTIC_C: float = 0.5
DEFAULT_RIDGE_ALPHA: float = 1.0
DEFAULT_FOREST_N_ESTIMATORS: int = 300
DEFAULT_FOREST_MAX_DEPTH: int = 8
DEFAULT_FOREST_MIN_SAMPLES_LEAF: int = 5
DEFAULT_XGBOOST_N_ESTIMATORS: int = 300
DEFAULT_XGBOOST_MAX_DEPTH: int = 3
DEFAULT_XGBOOST_LEARNING_RATE: float = 0.05
DEFAULT_XGBOOST_SUBSAMPLE: float = 0.85
DEFAULT_XGBOOST_COLSAMPLE: float = 0.85
DEFAULT_XGBOOST_MIN_CHILD_WEIGHT: int = 5
DEFAULT_XGBOOST_REG_LAMBDA: float = 5.0
SINGLE_THREAD: int = 1


def outcome_task(binary: bool) -> LearnerTask:
    """Binary outcomes get classifiers; continuous outcomes get regressors."""
    return "classification" if binary else "regression"


def build_learner(
    architecture: LearnerArchitecture,
    task: LearnerTask,
    random_state: int,
) -> BaseEstimator:
    """Return a fresh, unfitted estimator for an approved architecture and task."""
    classification = task == "classification"
    if architecture == "linear":
        if classification:
            return LogisticRegression(
                C=DEFAULT_LOGISTIC_C,
                max_iter=DEFAULT_LOGISTIC_MAX_ITER,
                random_state=random_state,
            )
        return Ridge(alpha=DEFAULT_RIDGE_ALPHA)
    if architecture == "random_forest":
        forest = RandomForestClassifier if classification else RandomForestRegressor
        return forest(
            n_estimators=DEFAULT_FOREST_N_ESTIMATORS,
            max_depth=DEFAULT_FOREST_MAX_DEPTH,
            min_samples_leaf=DEFAULT_FOREST_MIN_SAMPLES_LEAF,
            random_state=random_state,
            n_jobs=SINGLE_THREAD,
        )
    if architecture == "xgboost":
        shared: dict[str, float | int | None] = {
            "n_estimators": DEFAULT_XGBOOST_N_ESTIMATORS,
            "max_depth": DEFAULT_XGBOOST_MAX_DEPTH,
            "learning_rate": DEFAULT_XGBOOST_LEARNING_RATE,
            "subsample": DEFAULT_XGBOOST_SUBSAMPLE,
            "colsample_bytree": DEFAULT_XGBOOST_COLSAMPLE,
            "min_child_weight": DEFAULT_XGBOOST_MIN_CHILD_WEIGHT,
            "reg_lambda": DEFAULT_XGBOOST_REG_LAMBDA,
            "random_state": random_state,
            "n_jobs": SINGLE_THREAD,
        }
        if classification:
            return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **shared)
        return XGBRegressor(objective="reg:squarederror", **shared)
    raise ValueError(f"Unknown learner architecture: {architecture!r}.")

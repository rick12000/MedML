"""Optuna hyperparameter search for learners fit during cross-fitting."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from causallib.estimation import Standardization
from optuna.samplers import GPSampler
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from causal_pipeline.config import (
    CATEEstimatorSpec,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    clone_estimator,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.learner_defaults import (
    apply_defaults,
    materialize_outcome_learner,
    materialize_propensity_learner,
)
from causal_pipeline.learner_resolve import fill_cate_spec_learners
from causal_pipeline.nuisance_metrics import outcome_metric_from_predictions, propensity_auc
from causal_pipeline.scaling import clone_scaled_estimator

logger = logging.getLogger(__name__)

TUNING_N_TRIALS: int = 30
TUNING_N_STARTUP_TRIALS: int = 15
TUNING_VALIDATION_FRACTION: float = 0.2
MINIMUM_TUNING_ROWS: int = 40
MINIMUM_CLASS_COUNT: int = 2


class TuningTask(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def prepare_learner_for_outer_train(
    template: BaseEstimator,
    features: pd.DataFrame,
    labels: pd.Series,
    task: TuningTask,
    tune: bool,
    random_state: int,
    stratify: pd.Series | None = None,
    score_outcome: Callable[
        [BaseEstimator, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, bool],
        float | None,
    ]
    | None = None,
    outcome_treatment: pd.Series | None = None,
    binary_outcome: bool = False,
) -> BaseEstimator:
    """Pick hyperparameters on outer-train only, then return an unfitted learner for that fold."""
    base = apply_defaults(clone_estimator(template))
    if not tune or len(features) < MINIMUM_TUNING_ROWS:
        return base
    inner_train, inner_val = inner_holdout(
        features=features,
        labels=labels,
        stratify=stratify if stratify is not None else labels,
        validation_fraction=TUNING_VALIDATION_FRACTION,
        random_state=random_state,
    )
    if score_outcome is not None and outcome_treatment is not None:
        train_treatment = outcome_treatment.loc[inner_train.index]
        validation_treatment = outcome_treatment.loc[inner_val.index]

        def objective(trial: optuna.Trial) -> float:
            return score_outcome_trial(
                trial=trial,
                template=base,
                train_features=inner_train,
                train_treatment=train_treatment,
                train_outcome=labels.loc[inner_train.index],
                validation_features=inner_val,
                validation_treatment=validation_treatment,
                validation_outcome=labels.loc[inner_val.index],
                binary_outcome=binary_outcome,
                score=score_outcome,
            )
    else:

        def objective(trial: optuna.Trial) -> float:
            return score_simple_trial(
                trial=trial,
                template=base,
                task=task,
                train_features=inner_train,
                train_labels=labels.loc[inner_train.index],
                validation_features=inner_val,
                validation_labels=labels.loc[inner_val.index],
            )
    best_params = run_study(objective=objective, random_state=random_state)
    tuned = clone_estimator(base)
    tuned.set_params(**best_params)
    return tuned


def resolve_cate_spec_for_outer_train(
    train: CausalDataset,
    spec: CATEEstimatorSpec,
    tune: bool,
) -> CATEEstimatorSpec:
    """Apply defaults or fold-local tuning to learner templates before CATE fit."""
    random_state = getattr(spec, "random_state", 0) if hasattr(spec, "random_state") else 0
    binary_outcome = train.outcome_type == OutcomeType.BINARY
    features = train.X_adjustment.copy()
    treatment = train.treatment_series
    outcome = train.outcome_series

    if spec.kind in {CATEKind.TARNET, CATEKind.CFRNET, CATEKind.DRAGONNET}:
        return spec

    spec = fill_cate_spec_learners(spec=spec, outcome_type=train.outcome_type)
    if not tune:
        return spec

    if isinstance(spec, MetaCATEEstimatorSpec):
        outcome_learner = prepare_learner_for_outer_train(
            template=spec.outcome_learner,
            features=features,
            labels=outcome,
            task=TuningTask.CLASSIFICATION if binary_outcome else TuningTask.REGRESSION,
            tune=tune,
            random_state=random_state,
            stratify=treatment,
            score_outcome=score_standardized_outcome,
            outcome_treatment=treatment,
            binary_outcome=binary_outcome,
        )
        effect_learner = spec.effect_learner
        if effect_learner is not None:
            effect_learner = prepare_learner_for_outer_train(
                template=effect_learner,
                features=features,
                labels=outcome,
                task=TuningTask.REGRESSION,
                tune=True,
                random_state=random_state,
                stratify=treatment,
            )
        propensity_learner = spec.propensity_learner
        if propensity_learner is not None:
            propensity_learner = prepare_learner_for_outer_train(
                template=propensity_learner,
                features=features,
                labels=treatment,
                task=TuningTask.CLASSIFICATION,
                tune=True,
                random_state=random_state,
                stratify=treatment,
                score_outcome=None,
            )
        return spec.model_copy(
            update={
                "outcome_learner": outcome_learner,
                "effect_learner": effect_learner,
                "propensity_learner": propensity_learner,
            },
        )

    if isinstance(spec, CausalForestCATEEstimatorSpec):
        outcome_task = TuningTask.CLASSIFICATION if binary_outcome else TuningTask.REGRESSION
        return spec.model_copy(
            update={
                "outcome_learner": prepare_learner_for_outer_train(
                    template=spec.outcome_learner,
                    features=features,
                    labels=outcome,
                    task=outcome_task,
                    tune=tune,
                    random_state=0,
                    stratify=treatment,
                    score_outcome=score_standardized_outcome,
                    outcome_treatment=treatment,
                    binary_outcome=binary_outcome,
                ),
                "propensity_learner": prepare_learner_for_outer_train(
                    template=spec.propensity_learner,
                    features=features,
                    labels=treatment,
                    task=TuningTask.CLASSIFICATION,
                    tune=tune,
                    random_state=0,
                    stratify=treatment,
                ),
            },
        )

    return spec


def inner_holdout(
    features: pd.DataFrame,
    labels: pd.Series,
    stratify: pd.Series,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify.nunique() < MINIMUM_CLASS_COUNT:
        train_index, validation_index = train_test_split(
            features.index,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
        )
    else:
        train_index, validation_index = train_test_split(
            features.index,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    return features.loc[train_index], features.loc[validation_index]


def run_study(
    objective: Callable[[optuna.Trial], float],
    random_state: int,
) -> dict[str, object]:
    sampler = GPSampler(seed=random_state, n_startup_trials=TUNING_N_STARTUP_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=TUNING_N_TRIALS, show_progress_bar=False)
    return study.best_params


def score_simple_trial(
    trial: optuna.Trial,
    template: BaseEstimator,
    task: TuningTask,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
) -> float:
    estimator = build_trial_estimator(trial=trial, template=template, task=task)
    model = clone_scaled_estimator(estimator)
    model.fit(train_features, train_labels)
    if task == TuningTask.CLASSIFICATION:
        probabilities = pd.DataFrame(
            model.predict_proba(validation_features),
            columns=list(model.classes_),
        )
        score = propensity_auc(validation_labels, probabilities)
        if score is None:
            return float("-inf")
        return score
    metric = outcome_metric_from_predictions(
        outcome=validation_labels.to_numpy(dtype=float),
        predicted=model.predict(validation_features),
        binary_outcome=False,
    )
    if metric is None:
        return float("-inf")
    return metric.value


def score_outcome_trial(
    trial: optuna.Trial,
    template: BaseEstimator,
    train_features: pd.DataFrame,
    train_treatment: pd.Series,
    train_outcome: pd.Series,
    validation_features: pd.DataFrame,
    validation_treatment: pd.Series,
    validation_outcome: pd.Series,
    binary_outcome: bool,
    score: Callable[
        [BaseEstimator, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, bool],
        float | None,
    ],
) -> float:
    task = TuningTask.CLASSIFICATION if binary_outcome else TuningTask.REGRESSION
    estimator = build_trial_estimator(trial=trial, template=template, task=task)
    value = score(
        estimator,
        train_features,
        train_treatment,
        train_outcome,
        validation_features,
        validation_treatment,
        validation_outcome,
        binary_outcome,
    )
    if value is None:
        return float("-inf")
    return value


def score_standardized_outcome(
    estimator: BaseEstimator,
    train_features: pd.DataFrame,
    train_treatment: pd.Series,
    train_outcome: pd.Series,
    validation_features: pd.DataFrame,
    validation_treatment: pd.Series,
    validation_outcome: pd.Series,
    binary_outcome: bool,
) -> float | None:
    model = Standardization(
        clone_scaled_estimator(estimator),
        encode_treatment=True,
        predict_proba=binary_outcome,
    )
    model.fit(train_features, train_treatment, train_outcome)
    raw = model.estimate_individual_outcome(
        validation_features,
        validation_treatment,
        treatment_values=sorted(train_treatment.unique()),
        predict_proba=binary_outcome,
    )
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        outcome_level = raw.columns.get_level_values(-1).max()
        raw = raw.xs(outcome_level, axis="columns", level=-1)
    row_index = np.arange(len(validation_treatment))
    column_index = raw.columns.get_indexer(validation_treatment.to_numpy())
    if np.any(column_index < 0):
        return None
    factual = raw.to_numpy()[row_index, column_index]
    if binary_outcome:
        clipped = np.clip(factual, 1e-6, 1.0 - 1e-6)
        return float(-log_loss(validation_outcome, clipped))
    metric = outcome_metric_from_predictions(
        outcome=validation_outcome.to_numpy(dtype=float),
        predicted=factual,
        binary_outcome=False,
    )
    if metric is None:
        return None
    return metric.value


def build_trial_estimator(
    trial: optuna.Trial,
    template: BaseEstimator,
    task: TuningTask,
) -> BaseEstimator:
    name = type(template).__name__
    estimator = clone(template)
    if name == "LogisticRegression":
        estimator.set_params(
            C=trial.suggest_float("C", 0.05, 20.0, log=True),
            max_iter=10_000,
        )
        return estimator
    if name == "Ridge":
        estimator.set_params(alpha=trial.suggest_float("alpha", 0.1, 50.0, log=True))
        return estimator
    if name in {"XGBClassifier", "XGBRegressor"}:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        }
        estimator.set_params(**params)
        return estimator
    if name in {"RandomForestClassifier", "RandomForestRegressor"}:
        estimator.set_params(
            n_estimators=trial.suggest_int("n_estimators", 200, 500, step=50),
            max_depth=trial.suggest_categorical("max_depth", [3, 5, 8, None]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
        )
        return estimator
    logger.warning("No tuning space for %s; using defaults only.", name)
    return apply_defaults(estimator)


def propensity_learner_for_fold(
    template: BaseEstimator | None,
    train_features: pd.DataFrame,
    train_treatment: pd.Series,
    tune: bool,
    random_state: int,
) -> BaseEstimator:
    resolved = materialize_propensity_learner(template, random_state=random_state)
    if not tune:
        return resolved
    return prepare_learner_for_outer_train(
        template=resolved,
        features=train_features,
        labels=train_treatment,
        task=TuningTask.CLASSIFICATION,
        tune=True,
        random_state=random_state,
        stratify=train_treatment,
    )


def outcome_learner_for_fold(
    template: BaseEstimator | None,
    train_features: pd.DataFrame,
    train_treatment: pd.Series,
    train_outcome: pd.Series,
    binary_outcome: bool,
    tune: bool,
    random_state: int,
) -> BaseEstimator:
    task = TuningTask.CLASSIFICATION if binary_outcome else TuningTask.REGRESSION
    outcome_type = OutcomeType.BINARY if binary_outcome else OutcomeType.CONTINUOUS
    resolved = materialize_outcome_learner(
        template,
        outcome_type=outcome_type,
        random_state=random_state,
    )
    if not tune:
        return resolved
    return prepare_learner_for_outer_train(
        template=resolved,
        features=train_features,
        labels=train_outcome,
        task=task,
        tune=True,
        random_state=random_state,
        stratify=train_treatment,
        score_outcome=score_standardized_outcome,
        outcome_treatment=train_treatment,
        binary_outcome=binary_outcome,
    )

"""Held-out logistic and gradient-boosted tree screens for propensity and outcome."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from causal_pipeline.config import (
    NuisanceScreenConfig,
    OutcomeType,
    predict_outcome_mean,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.learner_defaults import (
    DEFAULT_XGBOOST_COLSAMPLE,
    DEFAULT_XGBOOST_LEARNING_RATE,
    DEFAULT_XGBOOST_N_ESTIMATORS,
    DEFAULT_XGBOOST_N_JOBS,
    DEFAULT_XGBOOST_REG_LAMBDA,
    DEFAULT_XGBOOST_SUBSAMPLE,
)
from causal_pipeline.nuisance_metrics import (
    log_nuisance_metric,
    outcome_metric_from_predictions,
    propensity_auc,
)
from causal_pipeline.scaling import ScaledClassifier, ScaledLearner, clone_scaled_estimator
from causal_pipeline.utils import save_figure, write_dataframe

logger = logging.getLogger(__name__)

LOGISTIC_C_VALUES = (0.1, 1.0, 10.0)
LOGISTIC_MAX_ITER = 500
XGBOOST_MAX_DEPTHS: tuple[int, ...] = (3, 6)
XGBOOST_MIN_CHILD_WEIGHTS: tuple[int, ...] = (5, 12)
SCREEN_SPLIT_ATTEMPTS = 20
SHAP_ROW_LIMIT = 200
CORRELATION_FIGURE_SIZE = 6.0
LOWEST_SELECTION_SCORE = -1.0
XGBOOST_ARCHITECTURE = "xgboost"


@dataclass(frozen=True)
class ScreenSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class XgboostChoice:
    max_depth: int
    min_child_weight: int


def run_nuisance_screen(
    dataset: CausalDataset,
    screen: NuisanceScreenConfig,
    results_root: Path,
) -> None:
    """Tune on a validation slice, refit on train, and score the held-out test rows."""
    root = results_root / "nuisance_screen"
    covariates = dataset.X_adjustment.copy()
    treatment = dataset.treatment_series.copy()
    outcome = dataset.outcome_series.copy()
    binary_outcome = dataset.outcome_type == OutcomeType.BINARY
    split = screen_split(
        frame=screen_frame(dataset),
        group_id=dataset.group_id,
        stratify_column=dataset.treatment,
        test_fraction=screen.test_fraction,
        validation_fraction=screen.validation_fraction,
        random_state=screen.random_state,
    )
    write_correlation(covariates=covariates, outcome=outcome, destination=root / "correlation")
    fit_role(
        role="propensity",
        split=split,
        feature_columns=list(covariates.columns),
        label_column=dataset.treatment,
        binary_outcome=False,
        classification=True,
        random_state=screen.random_state,
        destination=root / "propensity",
    )
    outcome_frame = outcome_design(
        covariates=covariates,
        treatment=treatment,
        outcome=outcome,
        treatment_name=dataset.treatment,
    )
    outcome_split = align_outcome_split(split=split, design=outcome_frame)
    fit_role(
        role="outcome",
        split=outcome_split,
        feature_columns=[column for column in outcome_frame.columns if column != dataset.outcome],
        label_column=dataset.outcome,
        binary_outcome=binary_outcome,
        classification=binary_outcome,
        random_state=screen.random_state,
        destination=root / "outcome",
    )
    logger.info("Wrote nuisance screens under %s", root)


def screen_frame(dataset: CausalDataset) -> pd.DataFrame:
    """Covariates, treatment, outcome, and the group id when rows are clustered."""
    pieces: list[pd.DataFrame | pd.Series] = [
        dataset.X_adjustment,
        dataset.treatment_series,
        dataset.outcome_series,
    ]
    group_id = dataset.group_id
    if group_id is not None and group_id not in dataset.X_adjustment.columns:
        pieces.append(dataset.df[group_id])
    return pd.concat(pieces, axis=1)


def screen_split(
    frame: pd.DataFrame,
    group_id: str | None,
    stratify_column: str,
    test_fraction: float,
    validation_fraction: float,
    random_state: int,
) -> ScreenSplit:
    train_validation, test = holdout_frame(
        frame=frame,
        group_id=group_id,
        stratify_column=stratify_column,
        holdout_fraction=test_fraction,
        random_state=random_state,
    )
    train, validation = holdout_frame(
        frame=train_validation,
        group_id=group_id,
        stratify_column=stratify_column,
        holdout_fraction=validation_fraction,
        random_state=random_state,
    )
    return ScreenSplit(train=train, validation=validation, test=test)


def holdout_frame(
    frame: pd.DataFrame,
    group_id: str | None,
    stratify_column: str,
    holdout_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = frame[stratify_column]
    if group_id is None:
        train, held_out = train_test_split(
            frame,
            test_size=holdout_fraction,
            random_state=random_state,
            stratify=stratify,
            shuffle=True,
        )
        return train, held_out
    groups = frame[group_id]
    for attempt in range(SCREEN_SPLIT_ATTEMPTS):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=holdout_fraction,
            random_state=random_state + attempt,
        )
        train_index, held_index = next(splitter.split(frame, groups=groups))
        train = frame.iloc[train_index]
        held_out = frame.iloc[held_index]
        train_classes = stratify.iloc[train_index].nunique()
        held_classes = stratify.iloc[held_index].nunique()
        if train_classes > 1 and held_classes > 1:
            return train, held_out
    raise ValueError("Could not hold out groups while keeping more than one class on each side.")


def outcome_design(
    covariates: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    treatment_name: str,
) -> pd.DataFrame:
    """One outcome regression on covariates plus treatment indicators."""
    indicators = pd.get_dummies(treatment.astype(str), prefix=treatment_name, drop_first=True)
    indicators.index = covariates.index
    return pd.concat([covariates, indicators, outcome.rename(outcome.name)], axis=1)


def align_outcome_split(split: ScreenSplit, design: pd.DataFrame) -> ScreenSplit:
    return ScreenSplit(
        train=design.loc[split.train.index],
        validation=design.loc[split.validation.index],
        test=design.loc[split.test.index],
    )


def write_correlation(covariates: pd.DataFrame, outcome: pd.Series, destination: Path) -> None:
    numeric = pd.concat([covariates, outcome], axis=1).select_dtypes(include="number")
    matrix = numeric.corr(numeric_only=True)
    write_dataframe(destination / "correlation.csv", matrix.reset_index(names="feature"))
    figure, axis = plt.subplots(figsize=(CORRELATION_FIGURE_SIZE, CORRELATION_FIGURE_SIZE))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(matrix.shape[1]), list(matrix.columns), rotation=90)
    axis.set_yticks(range(matrix.shape[0]), list(matrix.index))
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure(destination / "correlation.png", figure)
    plt.close(figure)


def fit_role(
    role: str,
    split: ScreenSplit,
    feature_columns: list[str],
    label_column: str,
    binary_outcome: bool,
    classification: bool,
    random_state: int,
    destination: Path,
) -> None:
    train_features = split.train.loc[:, feature_columns]
    validation_features = split.validation.loc[:, feature_columns]
    test_features = split.test.loc[:, feature_columns]
    train_labels = split.train[label_column]
    validation_labels = split.validation[label_column]
    test_labels = split.test[label_column]
    if classification:
        logistic_penalty = choose_logistic_penalty(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            binary_outcome=binary_outcome,
            random_state=random_state,
        )
        logistic_parameters: dict[str, float | int | None] = {"C": logistic_penalty}
    else:
        logistic_penalty = None
        logistic_parameters = {}
    logistic = fit_logistic(
        features=pd.concat([train_features, validation_features], axis=0),
        labels=pd.concat([train_labels, validation_labels], axis=0),
        penalty_c=logistic_penalty,
        classification=classification,
        random_state=random_state,
    )
    xgboost_choice = choose_xgboost(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        classification=classification,
        binary_outcome=binary_outcome,
        random_state=random_state,
    )
    boosted = fit_xgboost(
        features=pd.concat([train_features, validation_features], axis=0),
        labels=pd.concat([train_labels, validation_labels], axis=0),
        choice=xgboost_choice,
        classification=classification,
        random_state=random_state,
    )
    write_model_screen(
        role=role,
        architecture="logistic_regression",
        model=logistic,
        features=test_features,
        labels=test_labels,
        classification=classification,
        binary_outcome=binary_outcome,
        parameters=logistic_parameters,
        destination=destination / "logistic_regression",
        random_state=random_state,
    )
    write_model_screen(
        role=role,
        architecture=XGBOOST_ARCHITECTURE,
        model=boosted,
        features=test_features,
        labels=test_labels,
        classification=classification,
        binary_outcome=binary_outcome,
        parameters={
            "max_depth": xgboost_choice.max_depth,
            "min_child_weight": xgboost_choice.min_child_weight,
        },
        destination=destination / XGBOOST_ARCHITECTURE,
        random_state=random_state,
    )


def choose_logistic_penalty(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
    binary_outcome: bool,
    random_state: int,
) -> float:
    best_penalty = LOGISTIC_C_VALUES[0]
    best_score = LOWEST_SELECTION_SCORE
    for penalty_c in LOGISTIC_C_VALUES:
        model = fit_logistic(
            features=train_features,
            labels=train_labels,
            penalty_c=penalty_c,
            classification=True,
            random_state=random_state,
        )
        score = validation_score(
            model=model,
            features=validation_features,
            labels=validation_labels,
            classification=True,
            binary_outcome=binary_outcome,
        )
        if score > best_score:
            best_score = score
            best_penalty = penalty_c
    return best_penalty


def choose_xgboost(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
    classification: bool,
    binary_outcome: bool,
    random_state: int,
) -> XgboostChoice:
    best = XgboostChoice(
        max_depth=XGBOOST_MAX_DEPTHS[0],
        min_child_weight=XGBOOST_MIN_CHILD_WEIGHTS[0],
    )
    best_score = LOWEST_SELECTION_SCORE
    for max_depth in XGBOOST_MAX_DEPTHS:
        for min_child_weight in XGBOOST_MIN_CHILD_WEIGHTS:
            choice = XgboostChoice(max_depth=max_depth, min_child_weight=min_child_weight)
            model = fit_xgboost(
                features=train_features,
                labels=train_labels,
                choice=choice,
                classification=classification,
                random_state=random_state,
            )
            score = validation_score(
                model=model,
                features=validation_features,
                labels=validation_labels,
                classification=classification,
                binary_outcome=binary_outcome,
            )
            if score > best_score:
                best_score = score
                best = choice
    return best


def fit_logistic(
    features: pd.DataFrame,
    labels: pd.Series,
    penalty_c: float | None,
    classification: bool,
    random_state: int,
) -> ScaledLearner:
    if classification:
        if penalty_c is None:
            raise ValueError("Logistic screens require a penalty C.")
        estimator = LogisticRegression(C=penalty_c, max_iter=LOGISTIC_MAX_ITER, random_state=random_state)
    else:
        estimator = LinearRegression()
    model = clone_scaled_estimator(estimator)
    model.fit(features, labels)
    return model


def fit_xgboost(
    features: pd.DataFrame,
    labels: pd.Series,
    choice: XgboostChoice,
    classification: bool,
    random_state: int,
) -> XGBClassifier | XGBRegressor:
    shared = {
        "n_estimators": DEFAULT_XGBOOST_N_ESTIMATORS,
        "max_depth": choice.max_depth,
        "min_child_weight": choice.min_child_weight,
        "learning_rate": DEFAULT_XGBOOST_LEARNING_RATE,
        "subsample": DEFAULT_XGBOOST_SUBSAMPLE,
        "colsample_bytree": DEFAULT_XGBOOST_COLSAMPLE,
        "reg_lambda": DEFAULT_XGBOOST_REG_LAMBDA,
        "random_state": random_state,
        "n_jobs": DEFAULT_XGBOOST_N_JOBS,
    }
    if classification:
        objective = "binary:logistic" if labels.nunique() <= 2 else "multi:softprob"
        model = XGBClassifier(
            **shared,
            objective=objective,
            eval_metric="mlogloss",
        )
    else:
        model = XGBRegressor(
            **shared,
            objective="reg:squarederror",
        )
    model.fit(features, labels)
    return model


def validation_score(
    model: ScaledLearner | XGBClassifier | XGBRegressor,
    features: pd.DataFrame,
    labels: pd.Series,
    classification: bool,
    binary_outcome: bool,
) -> float:
    if classification:
        probabilities = pd.DataFrame(model.predict_proba(features), columns=list(model.classes_))
        auc = propensity_auc(labels, probabilities)
        if auc is None:
            return LOWEST_SELECTION_SCORE
        return auc
    metric = outcome_metric_from_predictions(
        outcome=labels.to_numpy(dtype=float),
        predicted=predict_outcome_mean(model, features),
        binary_outcome=binary_outcome,
    )
    if metric is None:
        return LOWEST_SELECTION_SCORE
    return metric.value


def write_model_screen(
    role: str,
    architecture: str,
    model: ScaledLearner | XGBClassifier | XGBRegressor,
    features: pd.DataFrame,
    labels: pd.Series,
    classification: bool,
    binary_outcome: bool,
    parameters: dict[str, float | int | None],
    destination: Path,
    random_state: int,
) -> None:
    score = validation_score(
        model=model,
        features=features,
        labels=labels,
        classification=classification,
        binary_outcome=binary_outcome,
    )
    metric_name = "AUC" if classification else "R2"
    log_nuisance_metric(f"screen {role} {architecture}", model, f"test {metric_name}", score)
    rows = [{"parameter": name, "value": value} for name, value in parameters.items()]
    rows.append({"parameter": f"test_{metric_name}", "value": score})
    write_dataframe(destination / "metrics.csv", pd.DataFrame(rows))
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        importance = pd.DataFrame(
            {"feature": list(features.columns), "gain_importance": model.feature_importances_}
        )
        write_dataframe(
            destination / "gain_importance.csv",
            importance.sort_values("gain_importance", ascending=False),
        )
    write_shap(
        model=model,
        features=features,
        destination=destination,
        random_state=random_state,
    )


def write_shap(
    model: ScaledLearner | XGBClassifier | XGBRegressor,
    features: pd.DataFrame,
    destination: Path,
    random_state: int,
) -> None:
    sample = features.sample(n=min(SHAP_ROW_LIMIT, len(features)), random_state=random_state)
    values = shap_matrix(model=model, sample=sample)
    table = pd.DataFrame(values, columns=list(sample.columns))
    write_dataframe(destination / "shap_values.csv", table)
    shap.summary_plot(values, sample, show=False)
    save_figure(destination / "shap_summary.png", plt.gcf())
    plt.close("all")
    shap.summary_plot(values, sample, plot_type="bar", show=False)
    save_figure(destination / "shap_bar.png", plt.gcf())
    plt.close("all")


def shap_matrix(
    model: ScaledLearner | XGBClassifier | XGBRegressor,
    sample: pd.DataFrame,
) -> np.ndarray:
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        explanation = shap.TreeExplainer(model).shap_values(sample)
        return first_shap_matrix(explanation)
    if isinstance(model, (ScaledClassifier, ScaledLearner)):
        transformed = np.asarray(model.scaler_.transform(sample), dtype=float)
        explanation = shap.LinearExplainer(model.model_, transformed).shap_values(transformed)
        return first_shap_matrix(explanation)
    raise TypeError(f"No SHAP path for {type(model).__name__}.")


def first_shap_matrix(explanation: object) -> np.ndarray:
    """Binary classifiers return one matrix per class. Keep the positive class."""
    if isinstance(explanation, list):
        return np.asarray(explanation[-1], dtype=float)
    values = np.asarray(explanation, dtype=float)
    if values.ndim == 3:
        return values[:, :, -1]
    return values

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
COLUMN_DROP_FIGURE_WIDTH = 8.0
COLUMN_DROP_FIGURE_ROW_HEIGHT = 0.35
COLUMN_DROP_CI_MULTIPLIER = 1.96
LOWEST_SELECTION_SCORE = -1.0
XGBOOST_ARCHITECTURE = "xgboost"
LOGISTIC_ARCHITECTURE = "logistic_regression"


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
        column_drop_repeats=screen.column_drop_repeats,
        column_drop_top_features=screen.column_drop_top_features,
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
        column_drop_repeats=screen.column_drop_repeats,
        column_drop_top_features=screen.column_drop_top_features,
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
    column_drop_repeats: int,
    column_drop_top_features: int,
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
    training_features = pd.concat([train_features, validation_features], axis=0)
    training_labels = pd.concat([train_labels, validation_labels], axis=0)
    write_model_screen(
        role=role,
        architecture=LOGISTIC_ARCHITECTURE,
        model=logistic,
        features=test_features,
        labels=test_labels,
        training_features=training_features,
        training_labels=training_labels,
        classification=classification,
        binary_outcome=binary_outcome,
        parameters=logistic_parameters,
        destination=destination / LOGISTIC_ARCHITECTURE,
        random_state=random_state,
        column_drop_repeats=column_drop_repeats,
        column_drop_top_features=column_drop_top_features,
    )
    write_model_screen(
        role=role,
        architecture=XGBOOST_ARCHITECTURE,
        model=boosted,
        features=test_features,
        labels=test_labels,
        training_features=training_features,
        training_labels=training_labels,
        classification=classification,
        binary_outcome=binary_outcome,
        parameters={
            "max_depth": xgboost_choice.max_depth,
            "min_child_weight": xgboost_choice.min_child_weight,
        },
        destination=destination / XGBOOST_ARCHITECTURE,
        random_state=random_state,
        column_drop_repeats=column_drop_repeats,
        column_drop_top_features=column_drop_top_features,
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
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    classification: bool,
    binary_outcome: bool,
    parameters: dict[str, float | int | None],
    destination: Path,
    random_state: int,
    column_drop_repeats: int,
    column_drop_top_features: int,
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
    write_column_drop_importance(
        training_features=training_features,
        training_labels=training_labels,
        evaluation_features=features,
        evaluation_labels=labels,
        architecture=architecture,
        parameters=parameters,
        classification=classification,
        binary_outcome=binary_outcome,
        repeats=column_drop_repeats,
        top_features=column_drop_top_features,
        destination=destination,
        random_state=random_state,
    )


def refit_screen_model(
    architecture: str,
    features: pd.DataFrame,
    labels: pd.Series,
    parameters: dict[str, float | int | None],
    classification: bool,
    random_state: int,
) -> ScaledLearner | XGBClassifier | XGBRegressor:
    if architecture == LOGISTIC_ARCHITECTURE:
        penalty_c = parameters.get("C")
        if classification and penalty_c is None:
            raise ValueError("Logistic column-drop screens require a penalty C.")
        return fit_logistic(
            features=features,
            labels=labels,
            penalty_c=penalty_c if classification else None,
            classification=classification,
            random_state=random_state,
        )
    if architecture == XGBOOST_ARCHITECTURE:
        max_depth = parameters.get("max_depth")
        min_child_weight = parameters.get("min_child_weight")
        if max_depth is None or min_child_weight is None:
            raise ValueError("XGBoost column-drop screens require max_depth and min_child_weight.")
        return fit_xgboost(
            features=features,
            labels=labels,
            choice=XgboostChoice(max_depth=int(max_depth), min_child_weight=int(min_child_weight)),
            classification=classification,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported screen architecture for column drop: {architecture}")


def performance_drop_metric_name(classification: bool) -> str:
    return "AUC" if classification else "R2"


def column_drop_repeat_seed(random_state: int, repeat_index: int) -> int:
    return random_state + repeat_index


def summarize_column_drops(drops: np.ndarray) -> tuple[float, float, float, float, float]:
    """Mean drop with sample standard deviation, SEM, and a normal 95% interval."""
    values = np.asarray(drops, dtype=float)
    repeat_count = values.size
    mean_drop = float(np.mean(values))
    if repeat_count <= 1:
        return mean_drop, 0.0, 0.0, mean_drop, mean_drop
    std_drop = float(np.std(values, ddof=1))
    sem_drop = std_drop / np.sqrt(repeat_count)
    half_width = COLUMN_DROP_CI_MULTIPLIER * sem_drop
    return mean_drop, std_drop, float(sem_drop), mean_drop - half_width, mean_drop + half_width


def column_drop_importance_table(
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    evaluation_features: pd.DataFrame,
    evaluation_labels: pd.Series,
    architecture: str,
    parameters: dict[str, float | int | None],
    classification: bool,
    binary_outcome: bool,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Retrain without each column and measure held-out performance loss versus the full model."""
    if repeats < 1:
        raise ValueError("column_drop_repeats must be at least 1.")
    feature_columns = list(training_features.columns)
    metric_name = performance_drop_metric_name(classification=classification)
    drop_samples: dict[str, list[float]] = {column: [] for column in feature_columns}
    for repeat_index in range(repeats):
        repeat_seed = column_drop_repeat_seed(random_state=random_state, repeat_index=repeat_index)
        baseline_model = refit_screen_model(
            architecture=architecture,
            features=training_features,
            labels=training_labels,
            parameters=parameters,
            classification=classification,
            random_state=repeat_seed,
        )
        baseline_score = validation_score(
            model=baseline_model,
            features=evaluation_features,
            labels=evaluation_labels,
            classification=classification,
            binary_outcome=binary_outcome,
        )
        for column in feature_columns:
            remaining_columns = [name for name in feature_columns if name != column]
            reduced_training = training_features.loc[:, remaining_columns]
            reduced_evaluation = evaluation_features.loc[:, remaining_columns]
            reduced_model = refit_screen_model(
                architecture=architecture,
                features=reduced_training,
                labels=training_labels,
                parameters=parameters,
                classification=classification,
                random_state=repeat_seed,
            )
            reduced_score = validation_score(
                model=reduced_model,
                features=reduced_evaluation,
                labels=evaluation_labels,
                classification=classification,
                binary_outcome=binary_outcome,
            )
            drop_samples[column].append(baseline_score - reduced_score)
    rows: list[dict[str, float | int | str]] = []
    for column in feature_columns:
        mean_drop, std_drop, sem_drop, ci_low, ci_high = summarize_column_drops(
            np.asarray(drop_samples[column], dtype=float)
        )
        rows.append(
            {
                "feature": column,
                f"mean_{metric_name}_drop": mean_drop,
                f"std_{metric_name}_drop": std_drop,
                f"sem_{metric_name}_drop": sem_drop,
                f"ci95_lower_{metric_name}_drop": ci_low,
                f"ci95_upper_{metric_name}_drop": ci_high,
                "n_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def write_column_drop_importance(
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    evaluation_features: pd.DataFrame,
    evaluation_labels: pd.Series,
    architecture: str,
    parameters: dict[str, float | int | None],
    classification: bool,
    binary_outcome: bool,
    repeats: int,
    top_features: int,
    destination: Path,
    random_state: int,
) -> None:
    table = column_drop_importance_table(
        training_features=training_features,
        training_labels=training_labels,
        evaluation_features=evaluation_features,
        evaluation_labels=evaluation_labels,
        architecture=architecture,
        parameters=parameters,
        classification=classification,
        binary_outcome=binary_outcome,
        repeats=repeats,
        random_state=random_state,
    )
    metric_name = performance_drop_metric_name(classification=classification)
    mean_column = f"mean_{metric_name}_drop"
    sem_column = f"sem_{metric_name}_drop"
    ranked = table.sort_values(mean_column, ascending=False)
    write_dataframe(destination / "column_drop_importance.csv", ranked)
    plot_column_drop_importance(
        table=ranked,
        metric_name=metric_name,
        top_features=top_features,
        destination=destination / "column_drop_importance.png",
    )


def plot_column_drop_importance(
    table: pd.DataFrame,
    metric_name: str,
    top_features: int,
    destination: Path,
) -> None:
    mean_column = f"mean_{metric_name}_drop"
    sem_column = f"sem_{metric_name}_drop"
    panel = table.head(top_features).iloc[::-1]
    figure_height = max(3.0, COLUMN_DROP_FIGURE_ROW_HEIGHT * len(panel))
    figure, axis = plt.subplots(figsize=(COLUMN_DROP_FIGURE_WIDTH, figure_height))
    positions = np.arange(len(panel))
    axis.barh(
        y=positions,
        width=panel[mean_column].to_numpy(dtype=float),
        xerr=panel[sem_column].to_numpy(dtype=float),
        color="steelblue",
        capsize=3.0,
        zorder=2,
    )
    axis.set_yticks(positions, panel["feature"].tolist())
    axis.set_xlabel(f"Held-out {metric_name} drop when column is removed")
    axis.set_title(f"Top {len(panel)} features by column-drop importance")
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    figure.tight_layout()
    save_figure(destination, figure)
    plt.close(figure)


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

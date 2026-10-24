"""Held-out screens for the propensity and outcome learners used in a run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from causal_pipeline.config import (
    NuisanceScreenConfig,
    OutcomeType,
    predict_outcome_mean,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.figures import publication_style
from causal_pipeline.nuisance_metrics import (
    log_nuisance_metric,
    outcome_metric_from_predictions,
    propensity_auc,
)
from causal_pipeline.scaling import ScaledClassifier, ScaledLearner
from causal_pipeline.utils import save_figure, write_dataframe

logger = logging.getLogger(__name__)

SCREEN_SPLIT_ATTEMPTS = 20
SHAP_ROW_LIMIT = 200
CORRELATION_MIN_INCHES = 4.4
CORRELATION_MAX_INCHES = 11.0
CORRELATION_CELL_INCHES = 0.36
CORRELATION_ANNOTATE_LIMIT = 12
CORRELATION_FULL_TICK_LIMIT = 30
CORRELATION_TARGET_TICKS = 30
CORRELATION_LEGEND_ROWS = 22
CORRELATION_LEGEND_ROW_INCHES = 0.18
CORRELATION_LEGEND_FONT = 7.0
CORRELATION_TICK_FONT = 7.0
CORRELATION_LOW = "#2F5D8C"
CORRELATION_MID = "#F6F4F1"
CORRELATION_HIGH = "#B23A48"
CORRELATION_MISSING = "#E4E1DC"
COLUMN_DROP_FIGURE_WIDTH = 8.0
COLUMN_DROP_FIGURE_ROW_HEIGHT = 0.35
COLUMN_DROP_CI_MULTIPLIER = 1.96
LOWEST_SELECTION_SCORE = -1.0


@dataclass(frozen=True)
class ScreenSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def run_nuisance_screen(
    dataset: CausalDataset,
    screen: NuisanceScreenConfig,
    results_root: Path,
    propensity_learner: BaseEstimator,
    outcome_learners: list[BaseEstimator],
) -> None:
    """Fit configured nuisances on an internal split of the estimation sample."""
    root = results_root / "nuisance_screen"
    covariates = dataset.X_adjustment.copy()
    treatment = dataset.treatment_series.copy()
    outcome = dataset.outcome_series.copy()
    binary_outcome = dataset.outcome_type == OutcomeType.BINARY
    write_correlation(variables=correlation_variables(dataset), destination=root / "correlation")
    split = screen_split(
        frame=screen_frame(dataset),
        group_id=dataset.group_id,
        stratify_column=dataset.treatment,
        test_fraction=screen.test_fraction,
        validation_fraction=screen.validation_fraction,
        random_state=screen.random_state,
    )
    fit_role(
        role="propensity",
        learner=propensity_learner,
        split=split,
        feature_columns=list(covariates.columns),
        label_column=dataset.treatment,
        binary_outcome=False,
        classification=True,
        random_state=screen.random_state,
        column_drop_repeats=screen.column_drop_repeats,
        column_drop_top_features=screen.column_drop_top_features,
        destination=root / "propensity" / learner_directory_name(propensity_learner),
    )
    outcome_frame = outcome_design(
        covariates=covariates,
        treatment=treatment,
        outcome=outcome,
        treatment_name=dataset.treatment,
    )
    outcome_split = align_outcome_split(split=split, design=outcome_frame)
    used_names: set[str] = set()
    for learner in outcome_learners:
        directory_name = learner_directory_name(learner, used_names)
        fit_role(
            role="outcome",
            learner=learner,
            split=outcome_split,
            feature_columns=[column for column in outcome_frame.columns if column != dataset.outcome],
            label_column=dataset.outcome,
            binary_outcome=binary_outcome,
            classification=binary_outcome,
            random_state=screen.random_state,
            column_drop_repeats=screen.column_drop_repeats,
            column_drop_top_features=screen.column_drop_top_features,
            destination=root / "outcome" / directory_name,
        )
    logger.info("Wrote nuisance screens under %s", root)


def learner_directory_name(learner: BaseEstimator, used: set[str] | None = None) -> str:
    base = type(learner).__name__
    if used is None:
        return base
    name = base
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


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


def correlation_variables(dataset: CausalDataset) -> pd.DataFrame:
    """Numeric analysis columns on the full estimation sample, outcome last."""
    ordered = list(dict.fromkeys([*dataset.X_adjustment.columns, dataset.treatment, dataset.outcome]))
    numeric = dataset.df.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    kept = [column for column in ordered if numeric[column].notna().any()]
    if len(kept) < 2:
        raise ValueError("Correlation needs at least two numeric analysis columns.")
    return numeric.loc[:, kept].copy()


def correlation_matrix(variables: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations. Rows and columns stay in the input order."""
    return variables.corr(method="pearson", numeric_only=True)


def correlation_export(matrix: pd.DataFrame) -> pd.DataFrame:
    """Matrix keyed by 1-based aliases, with the original column name beside each row."""
    names = [str(column) for column in matrix.columns]
    aliases = [str(index) for index in range(1, len(names) + 1)]
    export = matrix.to_numpy(dtype=float)
    table = pd.DataFrame(export, columns=aliases)
    table.insert(0, "variable", names)
    table.insert(0, "alias", aliases)
    return table


def write_correlation(variables: pd.DataFrame, destination: Path) -> None:
    matrix = correlation_matrix(variables)
    write_dataframe(destination / "correlation.csv", correlation_export(matrix))
    figure = plot_correlation(matrix)
    save_figure(destination / "correlation.png", figure)
    plt.close(figure)


def plot_correlation(matrix: pd.DataFrame) -> plt.Figure:
    """Heatmap labeled by alias. The index below maps each alias to its column."""
    count = matrix.shape[0]
    names = [str(column) for column in matrix.columns]
    heat_inches = min(CORRELATION_MAX_INCHES, max(CORRELATION_MIN_INCHES, count * CORRELATION_CELL_INCHES))
    legend_columns = max(1, int(np.ceil(count / CORRELATION_LEGEND_ROWS)))
    legend_rows = int(np.ceil(count / legend_columns))
    legend_inches = max(1.2, legend_rows * CORRELATION_LEGEND_ROW_INCHES + 0.45)
    longest = max(len(f"{index + 1}  {name}") for index, name in enumerate(names))
    figure_width = max(heat_inches + 1.05, legend_columns * max(1.7, longest * 0.072))
    figure_width = min(figure_width, 22.0)
    with publication_style():
        figure = plt.figure(figsize=(figure_width, heat_inches + legend_inches))
        grid = figure.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[heat_inches, legend_inches],
            hspace=0.08,
        )
        heat_axis = figure.add_subplot(grid[0, 0])
        legend_axis = figure.add_subplot(grid[1, 0])
        image = heat_axis.imshow(
            matrix.to_numpy(dtype=float),
            vmin=-1.0,
            vmax=1.0,
            cmap=correlation_colormap(),
            interpolation="nearest",
            aspect="equal",
        )
        positions = correlation_tick_positions(count)
        labels = [str(position + 1) for position in positions]
        heat_axis.set_xticks(positions, labels)
        heat_axis.set_yticks(positions, labels)
        heat_axis.tick_params(length=0, labelsize=CORRELATION_TICK_FONT, colors="#1B1B1B")
        heat_axis.set_title("Pearson correlation", loc="left", pad=8, color="#1B1B1B")
        for spine in heat_axis.spines.values():
            spine.set_visible(False)
        if count <= CORRELATION_ANNOTATE_LIMIT:
            annotate_correlation(heat_axis, matrix.to_numpy(dtype=float))
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.03)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=CORRELATION_TICK_FONT)
        colorbar.set_label("Correlation", color="#1B1B1B")
        draw_correlation_legend(legend_axis, names, legend_columns, legend_rows)
        figure.subplots_adjust(left=0.06, right=0.92, top=0.94, bottom=0.03)
    return figure


def correlation_colormap() -> Colormap:
    gradient = LinearSegmentedColormap.from_list(
        "correlation",
        [CORRELATION_LOW, CORRELATION_MID, CORRELATION_HIGH],
    )
    return gradient.with_extremes(bad=CORRELATION_MISSING)


def correlation_tick_positions(count: int) -> list[int]:
    if count <= CORRELATION_FULL_TICK_LIMIT:
        return list(range(count))
    step = int(np.ceil(count / CORRELATION_TARGET_TICKS))
    positions = list(range(0, count, step))
    if positions[-1] != count - 1:
        positions.append(count - 1)
    return positions


def annotate_correlation(axis: plt.Axes, values: np.ndarray) -> None:
    count = values.shape[0]
    for row in range(count):
        for column in range(count):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            ink = "#1B1B1B" if abs(value) < 0.55 else "#FFFFFF"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=ink)


def draw_correlation_legend(
    axis: plt.Axes,
    names: list[str],
    legend_columns: int,
    legend_rows: int,
) -> None:
    axis.axis("off")
    axis.set_title("Index", loc="left", pad=2, color="#1B1B1B", fontsize=8)
    for index, name in enumerate(names):
        column = index // legend_rows
        row = index % legend_rows
        axis.text(
            (column + 0.02) / legend_columns,
            1.0 - (row + 0.5) / legend_rows,
            f"{index + 1}  {name}",
            ha="left",
            va="center",
            fontsize=CORRELATION_LEGEND_FONT,
            color="#1B1B1B",
            fontfamily="monospace",
            transform=axis.transAxes,
            clip_on=False,
        )


def fit_role(
    role: str,
    learner: BaseEstimator,
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
    training_features = pd.concat([train_features, validation_features], axis=0)
    training_labels = pd.concat(
        [split.train[label_column], split.validation[label_column]],
        axis=0,
    )
    model = fit_screen_learner(
        learner=learner,
        features=training_features,
        labels=training_labels,
        random_state=random_state,
    )
    write_model_screen(
        role=role,
        learner=learner,
        model=model,
        features=test_features,
        labels=split.test[label_column],
        training_features=training_features,
        training_labels=training_labels,
        classification=classification,
        binary_outcome=binary_outcome,
        destination=destination,
        random_state=random_state,
        column_drop_repeats=column_drop_repeats,
        column_drop_top_features=column_drop_top_features,
    )


def fit_screen_learner(
    learner: BaseEstimator,
    features: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> BaseEstimator:
    model = clone_screen_learner(learner, random_state)
    model.fit(features, labels)
    return model


def clone_screen_learner(learner: BaseEstimator, random_state: int) -> BaseEstimator:
    cloned = clone(learner)
    if "random_state" in cloned.get_params():
        cloned.set_params(random_state=random_state)
    return cloned


def validation_score(
    model: BaseEstimator,
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
    learner: BaseEstimator,
    model: BaseEstimator,
    features: pd.DataFrame,
    labels: pd.Series,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    classification: bool,
    binary_outcome: bool,
    destination: Path,
    random_state: int,
    column_drop_repeats: int,
    column_drop_top_features: int,
) -> None:
    del role
    score = validation_score(
        model=model,
        features=features,
        labels=labels,
        classification=classification,
        binary_outcome=binary_outcome,
    )
    metric_name = "AUC" if classification else "R2"
    log_nuisance_metric(f"screen {type(learner).__name__}", model, f"test {metric_name}", score)
    write_dataframe(
        destination / "metrics.csv",
        pd.DataFrame([{"parameter": f"test_{metric_name}", "value": score}]),
    )
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": list(features.columns), "gain_importance": model.feature_importances_}
        )
        write_dataframe(
            destination / "gain_importance.csv",
            importance.sort_values("gain_importance", ascending=False),
        )
    write_shap(model=model, features=features, destination=destination, random_state=random_state)
    write_column_drop_importance(
        learner=learner,
        training_features=training_features,
        training_labels=training_labels,
        evaluation_features=features,
        evaluation_labels=labels,
        classification=classification,
        binary_outcome=binary_outcome,
        repeats=column_drop_repeats,
        top_features=column_drop_top_features,
        destination=destination,
        random_state=random_state,
    )



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
    learner: BaseEstimator,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    evaluation_features: pd.DataFrame,
    evaluation_labels: pd.Series,
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
        baseline_model = fit_screen_learner(
            learner=learner,
            features=training_features,
            labels=training_labels,
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
            reduced_model = fit_screen_learner(
                learner=learner,
                features=reduced_training,
                labels=training_labels,
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
    learner: BaseEstimator,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    evaluation_features: pd.DataFrame,
    evaluation_labels: pd.Series,
    classification: bool,
    binary_outcome: bool,
    repeats: int,
    top_features: int,
    destination: Path,
    random_state: int,
) -> None:
    table = column_drop_importance_table(
        learner=learner,
        training_features=training_features,
        training_labels=training_labels,
        evaluation_features=evaluation_features,
        evaluation_labels=evaluation_labels,
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
    model: BaseEstimator,
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


def shap_matrix(model: BaseEstimator, sample: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (ScaledClassifier, ScaledLearner)):
        transformed = np.asarray(model.scaler_.transform(sample), dtype=float)
        explanation = shap.LinearExplainer(model.model_, transformed).shap_values(transformed)
        return first_shap_matrix(explanation)
    if isinstance(model, (XGBClassifier, XGBRegressor)) or hasattr(model, "estimators_"):
        explanation = shap.TreeExplainer(model).shap_values(sample)
        return first_shap_matrix(explanation)
    if isinstance(model, (LogisticRegression, LinearRegression, Ridge)):
        explanation = shap.LinearExplainer(model, np.asarray(sample, dtype=float)).shap_values(
            np.asarray(sample, dtype=float)
        )
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

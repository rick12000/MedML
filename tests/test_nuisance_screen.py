"""Held-out nuisance screen writes metrics and importance artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from causal_pipeline.config import NuisanceScreenConfig
from causal_pipeline.data import CausalDataset
from causal_pipeline.nuisance_screen import (
    column_drop_importance_table,
    correlation_export,
    correlation_matrix,
    correlation_tick_positions,
    run_nuisance_screen,
    screen_frame,
    screen_split,
    summarize_column_drops,
    write_correlation,
)


def test_nuisance_screen_writes_test_metrics(
    df_synthetic_binary,
    binary_data_config,
    test_output_dir: Path,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    propensity_learner = LogisticRegression(max_iter=200)
    outcome_learner = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        random_state=0,
        n_jobs=1,
        eval_metric="logloss",
    )
    run_nuisance_screen(
        dataset=dataset,
        screen=NuisanceScreenConfig(
            test_fraction=0.2,
            validation_fraction=0.25,
            random_state=0,
            column_drop_repeats=2,
        ),
        results_root=test_output_dir,
        propensity_learner=propensity_learner,
        outcome_learners=[outcome_learner],
    )
    propensity = test_output_dir / "nuisance_screen" / "propensity" / "LogisticRegression" / "metrics.csv"
    outcome = test_output_dir / "nuisance_screen" / "outcome" / "XGBClassifier" / "metrics.csv"
    importance = test_output_dir / "nuisance_screen" / "outcome" / "XGBClassifier" / "gain_importance.csv"
    column_drop = test_output_dir / "nuisance_screen" / "outcome" / "XGBClassifier" / "column_drop_importance.csv"
    column_drop_plot = (
        test_output_dir / "nuisance_screen" / "propensity" / "LogisticRegression" / "column_drop_importance.png"
    )
    shap_plot = test_output_dir / "nuisance_screen" / "outcome" / "XGBClassifier" / "shap_summary.png"
    correlation = test_output_dir / "nuisance_screen" / "correlation" / "correlation.csv"
    correlation_plot = test_output_dir / "nuisance_screen" / "correlation" / "correlation.png"
    assert propensity.exists()
    assert outcome.exists()
    assert importance.exists()
    assert column_drop.exists()
    assert column_drop_plot.exists()
    assert shap_plot.exists()
    assert correlation.exists()
    assert correlation_plot.exists()
    table = pd.read_csv(correlation)
    assert list(table["alias"]) == list(range(1, len(table) + 1))
    assert {"x1", "t", "y"}.issubset(set(table["variable"]))
    outcome_alias = str(int(table.loc[table["variable"] == "y", "alias"].iloc[0]))
    assert float(table.loc[table["variable"] == "y", outcome_alias].iloc[0]) == 1.0


def test_correlation_export_aliases_a_wide_matrix(tmp_path: Path) -> None:
    columns = [f"feature_{index}" for index in range(45)]
    variables = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 45)), columns=columns)
    matrix = correlation_matrix(variables)
    exported = correlation_export(matrix)
    assert list(exported["alias"]) == [str(index) for index in range(1, 46)]
    assert list(exported["variable"]) == columns
    assert exported.shape == (45, 47)
    write_correlation(variables=variables, destination=tmp_path)
    assert (tmp_path / "correlation.csv").is_file()
    assert (tmp_path / "correlation.png").is_file()
    assert len(correlation_tick_positions(45)) < 45


def test_summarize_column_drops_reports_uncertainty() -> None:
    mean_drop, std_drop, sem_drop, ci_low, ci_high = summarize_column_drops(
        np.array([0.10, 0.14, 0.12, 0.08, 0.16], dtype=float)
    )
    assert mean_drop == 0.12
    assert std_drop > 0.0
    assert sem_drop == std_drop / np.sqrt(5)
    assert ci_low < mean_drop < ci_high


def test_column_drop_importance_ranks_informative_features(
    df_synthetic_binary,
    binary_data_config,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    covariates = dataset.X_adjustment.copy()
    split = screen_split(
        frame=screen_frame(dataset),
        group_id=None,
        stratify_column=dataset.treatment,
        test_fraction=0.2,
        validation_fraction=0.25,
        random_state=0,
    )
    training_features = pd.concat(
        [split.train.loc[:, covariates.columns], split.validation.loc[:, covariates.columns]],
        axis=0,
    )
    training_labels = pd.concat(
        [split.train[dataset.treatment], split.validation[dataset.treatment]],
        axis=0,
    )
    table = column_drop_importance_table(
        learner=LogisticRegression(max_iter=200),
        training_features=training_features,
        training_labels=training_labels,
        evaluation_features=split.test.loc[:, covariates.columns],
        evaluation_labels=split.test[dataset.treatment],
        classification=True,
        binary_outcome=False,
        repeats=2,
        random_state=0,
    )
    assert "mean_AUC_drop" in table.columns
    assert len(table) == covariates.shape[1]


def test_screen_split_keeps_the_group_column(
    df_synthetic_grouped,
    binary_data_config,
) -> None:
    config = binary_data_config.model_copy(update={"group_id": "patient_id"})
    dataset = CausalDataset(data=config, df=df_synthetic_grouped)
    split = screen_split(
        frame=screen_frame(dataset),
        group_id="patient_id",
        stratify_column="t",
        test_fraction=0.2,
        validation_fraction=0.25,
        random_state=0,
    )
    assert "patient_id" in split.train.columns
    assert "patient_id" in split.validation.columns
    assert set(split.train["patient_id"]).isdisjoint(split.validation["patient_id"])
    assert set(split.train["patient_id"]).isdisjoint(split.test["patient_id"])

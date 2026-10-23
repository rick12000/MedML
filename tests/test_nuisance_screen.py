"""Held-out nuisance screen writes metrics and importance artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.config import NuisanceScreenConfig
from causal_pipeline.data import CausalDataset
from causal_pipeline.nuisance_screen import (
    column_drop_importance_table,
    run_nuisance_screen,
    screen_frame,
    screen_split,
    summarize_column_drops,
)


def test_nuisance_screen_writes_test_metrics(
    df_synthetic_binary,
    binary_data_config,
    test_output_dir: Path,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    run_nuisance_screen(
        dataset=dataset,
        screen=NuisanceScreenConfig(
            test_fraction=0.2,
            validation_fraction=0.25,
            random_state=0,
            column_drop_repeats=2,
        ),
        results_root=test_output_dir,
    )
    propensity = test_output_dir / "nuisance_screen" / "propensity" / "logistic_regression" / "metrics.csv"
    outcome = test_output_dir / "nuisance_screen" / "outcome" / "xgboost" / "metrics.csv"
    importance = test_output_dir / "nuisance_screen" / "outcome" / "xgboost" / "gain_importance.csv"
    column_drop = test_output_dir / "nuisance_screen" / "outcome" / "xgboost" / "column_drop_importance.csv"
    column_drop_plot = (
        test_output_dir / "nuisance_screen" / "propensity" / "logistic_regression" / "column_drop_importance.png"
    )
    shap_plot = test_output_dir / "nuisance_screen" / "propensity" / "xgboost" / "shap_summary.png"
    correlation = test_output_dir / "nuisance_screen" / "correlation" / "correlation.csv"
    assert propensity.exists()
    assert outcome.exists()
    assert importance.exists()
    assert column_drop.exists()
    assert column_drop_plot.exists()
    assert shap_plot.exists()
    assert correlation.exists()


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
        training_features=training_features,
        training_labels=training_labels,
        evaluation_features=split.test.loc[:, covariates.columns],
        evaluation_labels=split.test[dataset.treatment],
        architecture="logistic_regression",
        parameters={"C": 1.0},
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

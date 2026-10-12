"""Held-out nuisance screen writes metrics and importance artifacts."""

from __future__ import annotations

from pathlib import Path

from causal_pipeline.config import NuisanceScreenConfig
from causal_pipeline.data import CausalDataset
from causal_pipeline.nuisance_screen import run_nuisance_screen, screen_frame, screen_split


def test_nuisance_screen_writes_test_metrics(
    df_synthetic_binary,
    binary_data_config,
    tmp_path: Path,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    run_nuisance_screen(
        dataset=dataset,
        screen=NuisanceScreenConfig(test_fraction=0.2, validation_fraction=0.25, random_state=0),
        results_root=tmp_path,
    )
    propensity = tmp_path / "nuisance_screen" / "propensity" / "logistic_regression" / "metrics.csv"
    outcome = tmp_path / "nuisance_screen" / "outcome" / "random_forest" / "metrics.csv"
    importance = tmp_path / "nuisance_screen" / "outcome" / "random_forest" / "impurity_importance.csv"
    shap_plot = tmp_path / "nuisance_screen" / "propensity" / "random_forest" / "shap_summary.png"
    correlation = tmp_path / "nuisance_screen" / "correlation" / "correlation.csv"
    assert propensity.exists()
    assert outcome.exists()
    assert importance.exists()
    assert shap_plot.exists()
    assert correlation.exists()


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

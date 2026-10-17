from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from causal_pipeline.ate import TreatmentContrastScores
from causal_pipeline.config import DataConfig, OutcomeType, SubgroupSplit, TreatmentMode
from causal_pipeline.data import CausalDataset
from causal_pipeline.subgroup import (
    SUBGROUP_SPLIT_COLUMNS,
    available_subgroup_splits,
    subgroup_split_table,
)
from causal_pipeline.subgroup_figures import plot_subgroup_splits_table

CONFIDENCE_LEVEL = 0.95


def test_subgroup_split_reports_effect_and_outcome_gaps() -> None:
    dataset = CausalDataset(
        data=DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.CONTINUOUS,
            treatment_mode=TreatmentMode.BINARY,
            control_value=0,
            treatment_values=[0, 1],
        ),
        df=pd.DataFrame(
            {
                "y": [4.0, 6.0, 10.0, 10.0, 10.0, 10.0],
                "t": [0, 1, 0, 1, 0, 1],
                "x1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "age": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            }
        ),
    )
    scores = TreatmentContrastScores(
        values=pd.DataFrame({"1_vs_0": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]}),
        positions=np.arange(len(dataset.df)),
        groups=None,
        confidence_level=CONFIDENCE_LEVEL,
    )
    table = subgroup_split_table(
        dataset=dataset,
        splits=[
            SubgroupSplit(column="age", cutoff=60.0),
            SubgroupSplit(column="x1", cutoff=0.5),
        ],
        scores=scores,
        estimator_id="ipw",
    )
    assert table.shape == (2, len(SUBGROUP_SPLIT_COLUMNS))
    age = table.iloc[0]
    assert age["above_rule"] == "greater_or_equal"
    assert age["n_above"] == 4
    assert age["n_below"] == 2
    assert age["effect_above"] == 2.0
    assert age["effect_below"] == 1.0
    assert age["effect_difference"] == 1.0
    assert age["effect_se"] == 0.0
    assert age["effect_pvalue"] == 0.0
    assert age["effect_ci_lower"] == 1.0
    assert age["effect_ci_upper"] == 1.0
    assert age["outcome_above"] == 10.0
    assert age["outcome_below"] == 5.0
    assert age["outcome_difference"] == 5.0
    assert age["outcome_se"] == pytest.approx(1.0)
    quantile = float(stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
    assert age["outcome_ci_lower"] == pytest.approx(5.0 - quantile)
    assert age["outcome_ci_upper"] == pytest.approx(5.0 + quantile)
    modifier = table.iloc[1]
    assert modifier["n_above"] == 3
    assert modifier["n_below"] == 3
    assert modifier["effect_difference"] == pytest.approx(2.0 / 3.0)


def test_clustered_effect_difference_uses_the_cluster_scores() -> None:
    dataset = CausalDataset(
        data=DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.CONTINUOUS,
            treatment_mode=TreatmentMode.BINARY,
            control_value=0,
            treatment_values=[0, 1],
        ),
        df=pd.DataFrame(
            {
                "y": [0.0, 0.0, 1.0, 1.0],
                "t": [0, 1, 0, 1],
                "x1": [1.0, 1.0, 0.0, 0.0],
            }
        ),
    )
    scores = TreatmentContrastScores(
        values=pd.DataFrame({"1_vs_0": [1.0, 3.0, 0.0, 2.0]}),
        positions=np.arange(4),
        groups=np.array([1, 2, 3, 3]),
        confidence_level=CONFIDENCE_LEVEL,
    )
    table = subgroup_split_table(
        dataset=dataset,
        splits=[SubgroupSplit(column="x1", cutoff=1.0)],
        scores=scores,
        estimator_id="aipw",
    )
    assert table.shape == (1, len(SUBGROUP_SPLIT_COLUMNS))
    assert table.loc[0, "effect_difference"] == pytest.approx(1.0)
    assert table.loc[0, "effect_se"] == pytest.approx(np.sqrt(0.75))


def test_missing_subgroup_columns_are_skipped(caplog: pytest.LogCaptureFixture) -> None:
    columns = pd.Index(["age", "y"])
    with caplog.at_level(logging.WARNING):
        kept = available_subgroup_splits(
            columns,
            [
                SubgroupSplit(column="age", cutoff=60.0),
                SubgroupSplit(column="bmi", cutoff=30.0),
            ],
        )
    assert [split.column for split in kept] == ["age"]
    assert kept[0].cutoff == 60.0
    assert "bmi" in caplog.text


def test_subgroup_splits_are_skipped_when_every_column_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        kept = available_subgroup_splits(
            pd.Index(["y", "t"]),
            [SubgroupSplit(column="age", cutoff=60.0)],
        )
    assert kept == []
    assert "none of the columns" in caplog.text


def test_subgroup_split_figure_renders(tmp_path) -> None:
    dataset = CausalDataset(
        data=DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.CONTINUOUS,
            treatment_mode=TreatmentMode.BINARY,
            control_value=0,
            treatment_values=[0, 1],
        ),
        df=pd.DataFrame(
            {
                "y": [4.0, 6.0, 10.0, 10.0, 10.0, 10.0],
                "t": [0, 1, 0, 1, 0, 1],
                "x1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "age": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            }
        ),
    )
    scores = TreatmentContrastScores(
        values=pd.DataFrame({"1_vs_0": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]}),
        positions=np.arange(len(dataset.df)),
        groups=None,
        confidence_level=CONFIDENCE_LEVEL,
    )
    table = subgroup_split_table(
        dataset=dataset,
        splits=[SubgroupSplit(column="age", cutoff=60.0)],
        scores=scores,
        estimator_id="ipw",
    )
    output = tmp_path / "subgroup_splits.png"
    plot_subgroup_splits_table(
        table=table,
        split_order=[("age", 60.0)],
        save_path=str(output),
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_subgroup_split_rejects_a_one_sided_cutoff() -> None:
    dataset = CausalDataset(
        data=DataConfig(
            outcome="y",
            treatment="t",
            confounders=["x1"],
            effect_modifiers=["x1"],
            outcome_type=OutcomeType.CONTINUOUS,
            treatment_mode=TreatmentMode.BINARY,
            control_value=0,
            treatment_values=[0, 1],
        ),
        df=pd.DataFrame(
            {
                "y": [0.0, 1.0, 2.0, 3.0],
                "t": [0, 1, 0, 1],
                "x1": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    )
    scores = TreatmentContrastScores(
        values=pd.DataFrame({"1_vs_0": [0.0, 0.0, 0.0, 0.0]}),
        positions=np.arange(4),
        groups=None,
        confidence_level=CONFIDENCE_LEVEL,
    )
    with pytest.raises(ValueError):
        subgroup_split_table(
            dataset=dataset,
            splits=[SubgroupSplit(column="x1", cutoff=0.0)],
            scores=scores,
            estimator_id="ipw",
        )

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
    ABOVE_CUTOFF_RULE,
    OVERALL_LABEL,
    SUBGROUP_SPLIT_COLUMNS,
    available_subgroup_splits,
    is_binary_values,
    subgroup_split_table,
)
from causal_pipeline.subgroup_figures import (
    RAW_BLOCK_KEY,
    EstimateCell,
    Interval,
    build_table_model,
    estimate_cell,
    format_p,
    plot_subgroup_splits_table,
    split_text,
    table_blocks,
)

CONFIDENCE_LEVEL = 0.95


def _two_group_dataset() -> CausalDataset:
    return CausalDataset(
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


def _estimates(contrasts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contrast": contrasts,
            "estimand": ["ATE"] * len(contrasts),
            "estimate": [1.5] * len(contrasts),
            "standard_error": [0.25] * len(contrasts),
            "ci_lower": [1.0] * len(contrasts),
            "ci_upper": [2.0] * len(contrasts),
        }
    )


def _two_estimator_table() -> pd.DataFrame:
    dataset = _two_group_dataset()
    scores = TreatmentContrastScores(
        values=pd.DataFrame({"1_vs_0": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]}),
        positions=np.arange(len(dataset.df)),
        groups=None,
        confidence_level=CONFIDENCE_LEVEL,
    )
    splits = [SubgroupSplit(column="age", cutoff=60.0), SubgroupSplit(column="x1", cutoff=0.5)]
    tables = [
        subgroup_split_table(
            dataset=dataset,
            splits=splits,
            scores=scores,
            estimator_id=estimator_id,
            estimates=_estimates(["1_vs_0"]),
        )
        for estimator_id in ("ipw", "aipw")
    ]
    return pd.concat(tables, ignore_index=True)


def test_figure_model_puts_overall_row_first_and_shares_the_raw_difference() -> None:
    table = _two_estimator_table()
    model = build_table_model(table=table, split_order=[("age", 60.0), ("x1", 0.5)])
    assert [block.key for block in model.blocks] == ["ipw", "aipw", RAW_BLOCK_KEY]
    assert len(model.groups) == 1
    rows = model.groups[0].rows
    assert [row.group for row in rows] == [OVERALL_LABEL, "age", "x1"]
    assert [row.splits for row in rows] == ["", "≥ 60 / < 60", "≥ 0.5 / < 0.5"]
    assert rows[0].is_overall
    assert rows[0].patients == "6"
    assert set(rows[0].intervals) == {"ipw", "aipw"}
    assert set(rows[1].intervals) == {"ipw", "aipw", RAW_BLOCK_KEY}
    assert rows[1].patients == "4 / 2"
    assert rows[1].intervals[RAW_BLOCK_KEY].estimate == pytest.approx(5.0)


def test_table_blocks_accepts_stacked_scenarios_with_repeated_splits() -> None:
    table = _two_estimator_table()
    stacked = pd.concat([table, table], ignore_index=True)
    blocks = table_blocks(stacked)
    assert [block.key for block in blocks] == ["ipw", "aipw", RAW_BLOCK_KEY]
    with pytest.raises(ValueError, match="more than one row"):
        build_table_model(table=stacked, split_order=[("age", 60.0)])


def test_binary_split_is_flagged_only_when_every_value_is_zero_or_one() -> None:
    assert is_binary_values(np.array([0.0, 1.0, 1.0]))
    assert not is_binary_values(np.array([0.0, 1.0, 2.0]))
    assert not is_binary_values(np.array([0.0, 0.5]))


def test_binary_split_text_is_one_slash_zero() -> None:
    assert split_text(cutoff=1.0, above_rule=ABOVE_CUTOFF_RULE, is_binary=True) == "1 / 0"
    assert split_text(cutoff=60.0, above_rule=ABOVE_CUTOFF_RULE, is_binary=False) == "≥ 60 / < 60"


def test_figure_model_rejects_raw_differences_that_disagree_across_estimators() -> None:
    table = _two_estimator_table()
    aipw_rows = table["estimator"] == "aipw"
    table.loc[aipw_rows & (table["cutoff"] == 60.0), "outcome_difference"] += 1.0
    with pytest.raises(ValueError, match="must match across estimators"):
        build_table_model(table=table, split_order=[("age", 60.0)])


def test_figure_model_groups_each_contrast_with_its_own_overall_row() -> None:
    dataset = _two_group_dataset()
    scores = TreatmentContrastScores(
        values=pd.DataFrame(
            {"1_vs_0": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0], "2_vs_0": [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]}
        ),
        positions=np.arange(len(dataset.df)),
        groups=None,
        confidence_level=CONFIDENCE_LEVEL,
    )
    table = subgroup_split_table(
        dataset=dataset,
        splits=[SubgroupSplit(column="age", cutoff=60.0)],
        scores=scores,
        estimator_id="ipw",
        estimates=_estimates(["1_vs_0", "2_vs_0"]),
    )
    model = build_table_model(table=table, split_order=[("age", 60.0)])
    assert [group.title for group in model.groups] == ["1 vs 0", "2 vs 0"]
    for group in model.groups:
        assert group.rows[0].is_overall
        assert [row.group for row in group.rows] == [OVERALL_LABEL, "age"]


def test_estimate_cell_splits_estimate_p_and_interval() -> None:
    cell = estimate_cell(Interval(estimate=1.234, lower=0.8, upper=1.66, pvalue=0.02))
    assert cell == EstimateCell(estimate="1.23", pvalue="0.02", bounds="[0.80, 1.66]")
    assert estimate_cell(Interval(estimate=1.0, lower=0.0, upper=2.0, pvalue=0.004)).pvalue == "<0.01"
    assert format_p(float("nan")) == "NA"


def test_subgroup_split_figure_renders_multiple_estimators(test_output_dir) -> None:
    output = test_output_dir / "subgroup" / "subgroup_splits_two_estimators.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_subgroup_splits_table(
        table=_two_estimator_table(),
        split_order=[("age", 60.0), ("x1", 0.5)],
        save_path=str(output),
    )
    assert output.is_file()
    assert output.stat().st_size > 0


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


def test_subgroup_population_total_row_matches_ate_estimate() -> None:
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
    estimates = pd.DataFrame(
        {
            "contrast": ["1_vs_0"],
            "estimand": ["ATE"],
            "estimate": [1.5],
            "standard_error": [0.25],
            "ci_lower": [1.0],
            "ci_upper": [2.0],
        }
    )
    table = subgroup_split_table(
        dataset=dataset,
        splits=[SubgroupSplit(column="age", cutoff=60.0)],
        scores=scores,
        estimator_id="ipw",
        estimates=estimates,
    )
    assert len(table) == 2
    total = table.iloc[1]
    assert pd.isna(total["column"])
    assert pd.isna(total["cutoff"])
    assert pd.isna(total["n_above"])
    assert total["effect_difference"] == 1.5
    assert total["effect_se"] == 0.25
    assert total["effect_ci_lower"] == 1.0
    assert total["effect_ci_upper"] == 2.0
    assert pd.isna(total["outcome_difference"])


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


def test_subgroup_split_figure_renders(test_output_dir) -> None:
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
        estimates=pd.DataFrame(
            {
                "contrast": ["1_vs_0"],
                "estimand": ["ATE"],
                "estimate": [1.5],
                "standard_error": [0.25],
                "ci_lower": [1.0],
                "ci_upper": [2.0],
            }
        ),
    )
    output = test_output_dir / "subgroup" / "subgroup_splits.png"
    output.parent.mkdir(parents=True, exist_ok=True)
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

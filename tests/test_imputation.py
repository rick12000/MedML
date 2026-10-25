"""Missingness contract, Rubin pooling through the pipeline, and skipped stages."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    DRPolicyTreeMethodSpec,
    DataConfig,
    ImputationConfig,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    NuisanceScreenConfig,
    OverlapConfig,
    PolicyKind,
    PipelineConfig,
    SensitivityConfig,
    SplitConfig,
    SubgroupSplit,
    TRIM_WITH_IMPUTATION_MESSAGE,
)
from causal_pipeline.imputation import (
    TREATMENT_OUTCOME_MISSING,
    imputation_warnings,
    miceforest_impute,
    validate_missingness,
)
from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.pooling import pool_nuisance_screens
from causal_pipeline.utils import read_dataframe, write_dataframe

IMPUTATION_COUNT = 2
PIPELINE_ROWS = 80
PIPELINE_FOLDS = 2
INCLUSION_FRACTION = 0.8


def test_miceforest_fills_covariates_and_leaves_outcome_unchanged() -> None:
    pytest.importorskip("miceforest")
    generator = np.random.default_rng(1)
    row_count = 40
    frame = pd.DataFrame(
        {
            "y": generator.normal(size=row_count),
            "t": generator.integers(0, 2, size=row_count),
            "x1": generator.normal(size=row_count),
            "x2": generator.normal(size=row_count),
        }
    )
    frame.loc[frame.index[::4], "x2"] = np.nan
    observed = frame["x2"].notna()
    completed = miceforest_impute(
        frame=frame,
        covariates=["x1", "x2"],
        predictors=["t", "y"],
        datasets=IMPUTATION_COUNT,
        iterations=2,
        random_state=1,
    )
    assert len(completed) == IMPUTATION_COUNT
    assert completed[0].shape == frame.shape
    assert int(completed[0]["x2"].isna().sum()) == 0
    assert completed[0].loc[observed, "x2"].to_numpy().tolist() == frame.loc[observed, "x2"].to_numpy().tolist()
    assert completed[0]["y"].equals(frame["y"])
    assert completed[0]["t"].equals(frame["t"])


def test_disabled_imputation_rejects_missing_covariates(recovery_data_config: DataConfig) -> None:
    frame = pd.DataFrame({"y": [0.0, 1.0], "t": [0, 1], "x1": [0.2, np.nan], "x2": [0.1, 0.4]})
    config = PipelineConfig(
        data=recovery_data_config,
        split=SplitConfig(),
        ate_estimators=[IPWATEEstimatorSpec(kind=ATEKind.IPW)],
        imputation=ImputationConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="Multiple imputation is disabled"):
        validate_missingness(analysis_frame=frame, excluded_frame=None, config=config)


def test_enabled_imputation_rejects_a_missing_outcome(recovery_data_config: DataConfig) -> None:
    frame = pd.DataFrame({"y": [0.0, np.nan], "t": [0, 1], "x1": [0.2, 0.3], "x2": [0.1, np.nan]})
    config = PipelineConfig(
        data=recovery_data_config,
        split=SplitConfig(),
        ate_estimators=[IPWATEEstimatorSpec(kind=ATEKind.IPW)],
        imputation=ImputationConfig(enabled=True, datasets=IMPUTATION_COUNT),
    )
    with pytest.raises(ValueError, match=TREATMENT_OUTCOME_MISSING):
        validate_missingness(analysis_frame=frame, excluded_frame=None, config=config)


def test_trimming_with_imputation_is_rejected(recovery_data_config: DataConfig) -> None:
    with pytest.raises(ValueError, match=TRIM_WITH_IMPUTATION_MESSAGE):
        PipelineConfig(
            data=recovery_data_config,
            split=SplitConfig(),
            ate_estimators=[IPWATEEstimatorSpec(kind=ATEKind.IPW)],
            overlap=OverlapConfig(trim_bounds=(0.1, 0.9)),
            imputation=ImputationConfig(enabled=True, datasets=IMPUTATION_COUNT),
        )


def test_skip_warnings_name_policy_sensitivity_and_unit_intervals(
    recovery_data_config: DataConfig,
) -> None:
    config = PipelineConfig(
        data=recovery_data_config,
        split=SplitConfig(test_fraction=0.25),
        ate_estimators=[IPWATEEstimatorSpec(kind=ATEKind.IPW)],
        cate_estimators=[],
        policy=[DRPolicyTreeMethodSpec(kind=PolicyKind.DR_POLICY_TREE)],
        cate_evaluation=CATEEvaluationConfig(),
        sensitivity=SensitivityConfig(),
        imputation=ImputationConfig(enabled=True, datasets=IMPUTATION_COUNT),
    )
    messages = imputation_warnings(config)
    assert any("Policy methods are skipped" in message for message in messages)
    assert any("Omitted-variable sensitivity is skipped" in message for message in messages)
    assert all("Unit-level" not in message for message in messages)

    cate_config = config.model_copy(
        update={"cate_estimators": [MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER)]}
    )
    assert any(
        "Unit-level conditional treatment effect intervals are skipped" in message
        for message in imputation_warnings(cate_config)
    )


def test_imputed_pipeline_pools_the_ate_and_skips_policy(
    recovery_data_config: DataConfig,
    test_output_dir,
    logistic_learner,
    shift_covariate_imputer,
) -> None:
    generator = np.random.default_rng(0)
    covariate_one = generator.normal(size=PIPELINE_ROWS)
    covariate_two = generator.normal(size=PIPELINE_ROWS)
    propensity = 1.0 / (1.0 + np.exp(-(0.4 * covariate_one)))
    treatment = generator.binomial(1, propensity)
    outcome = 1.2 * treatment + 0.5 * covariate_one + generator.normal(scale=0.3, size=PIPELINE_ROWS)
    frame = pd.DataFrame(
        {"y": outcome, "t": treatment, "x1": covariate_one, "x2": covariate_two}
    )
    frame.loc[frame.index[::5], "x2"] = np.nan
    config = PipelineConfig(
        data=recovery_data_config,
        split=SplitConfig(test_fraction=0.25),
        overlap=OverlapConfig(n_folds=PIPELINE_FOLDS, propensity_learner=logistic_learner),
        ate_estimators=[
            IPWATEEstimatorSpec(
                kind=ATEKind.IPW,
                n_folds=PIPELINE_FOLDS,
                propensity_learner=LogisticRegression(max_iter=200),
            )
        ],
        subgroup_splits=[SubgroupSplit(column="x1", cutoff=0.0)],
        policy=[DRPolicyTreeMethodSpec(kind=PolicyKind.DR_POLICY_TREE)],
        cate_evaluation=CATEEvaluationConfig(dr_crossfit_folds=PIPELINE_FOLDS),
        sensitivity=SensitivityConfig(inclusion_fraction=INCLUSION_FRACTION),
        nuisance_screen=NuisanceScreenConfig(enabled=False),
        imputation=ImputationConfig(enabled=True, datasets=IMPUTATION_COUNT, iterations=1),
        results_dir=str(test_output_dir),
    )
    pipeline = CausalPipeline(config=config, imputer=shift_covariate_imputer)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.run(frame)
    messages = [str(item.message) for item in caught]
    assert any("Policy methods are skipped" in message for message in messages)
    assert any("Omitted-variable sensitivity is skipped" in message for message in messages)
    estimate = read_dataframe(test_output_dir / "ate" / "summary.csv")
    assert len(estimate) == 1
    assert estimate.loc[0, "imputations"] == IMPUTATION_COUNT
    inflation = 1.0 + 1.0 / IMPUTATION_COUNT
    total = estimate.loc[0, "within_variance"] + inflation * estimate.loc[0, "between_variance"]
    assert estimate.loc[0, "total_variance"] == pytest.approx(total)
    completed = read_dataframe(test_output_dir / "imputation" / "0" / "data" / "estimation.parquet")
    assert len(completed) == PIPELINE_ROWS
    assert int(completed["x2"].isna().sum()) == 0
    assert not (test_output_dir / "policy").exists()
    sensitivity_dirs = list((test_output_dir / "imputation").rglob("sensitivity"))
    assert sensitivity_dirs == []
    unseen = read_dataframe(
        next((test_output_dir / "ate").rglob("unseen.csv"))
    )
    pooled_effect = float(estimate.loc[0, "estimate"])
    expected_tipping = -INCLUSION_FRACTION / (1.0 - INCLUSION_FRACTION) * pooled_effect
    assert unseen.loc[0, "tipping_point"] == pytest.approx(expected_tipping)
    subgroup = read_dataframe(test_output_dir / "ate" / "subgroup_splits.csv")
    assert len(subgroup) == 1
    assert subgroup.loc[0, "imputations"] == IMPUTATION_COUNT


def test_nuisance_screen_metrics_are_averaged(test_output_dir) -> None:
    first = test_output_dir / "imputation" / "0"
    second = test_output_dir / "imputation" / "1"
    metric = pd.DataFrame({"parameter": ["test_AUC"], "value": [0.6]})
    other = pd.DataFrame({"parameter": ["test_AUC"], "value": [0.8]})
    write_dataframe(first / "nuisance_screen" / "propensity" / "logit" / "metrics.csv", metric)
    write_dataframe(second / "nuisance_screen" / "propensity" / "logit" / "metrics.csv", other)
    drop = pd.DataFrame(
        {
            "feature": ["x1"],
            "mean_AUC_drop": [0.1],
            "std_AUC_drop": [0.02],
            "sem_AUC_drop": [0.01],
            "ci95_lower_AUC_drop": [0.08],
            "ci95_upper_AUC_drop": [0.12],
            "n_repeats": [5],
        }
    )
    second_drop = drop.copy()
    second_drop["mean_AUC_drop"] = 0.3
    write_dataframe(first / "nuisance_screen" / "propensity" / "logit" / "column_drop_importance.csv", drop)
    write_dataframe(
        second / "nuisance_screen" / "propensity" / "logit" / "column_drop_importance.csv",
        second_drop,
    )
    pool_nuisance_screens(
        imputation_roots=[first, second],
        destination=test_output_dir / "pooled",
    )
    averaged = read_dataframe(
        test_output_dir / "pooled" / "nuisance_screen" / "propensity" / "logit" / "metrics.csv"
    )
    assert len(averaged) == 1
    assert averaged.loc[0, "estimate"] == pytest.approx(0.7)
    pooled_drop = read_dataframe(
        test_output_dir
        / "pooled"
        / "nuisance_screen"
        / "propensity"
        / "logit"
        / "column_drop_importance.csv"
    )
    assert pooled_drop.loc[0, "mean_AUC_drop"] == pytest.approx(0.2)
    assert pooled_drop.loc[0, "std_AUC_drop"] == pytest.approx(0.02)
    assert np.isnan(pooled_drop.loc[0, "sem_AUC_drop"])
    assert np.isnan(pooled_drop.loc[0, "ci95_lower_AUC_drop"])
    assert np.isnan(pooled_drop.loc[0, "ci95_upper_AUC_drop"])
    summary = read_dataframe(test_output_dir / "pooled" / "nuisance_screen" / "metrics_summary.csv")
    assert len(summary) == 1

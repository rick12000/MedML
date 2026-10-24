from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
    DEFAULT_SPLIT_RANDOM_STATE,
    OverlapConfig,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    PipelineConfig,
    PolicyKind,
    PolicyTreeMethodSpec,
    SensitivityConfig,
    SplitConfig,
)
from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.utils import read_dataframe

INTEGRATION_FOLDS = 2
INTEGRATION_BOOTSTRAP_SAMPLES = 8
INTEGRATION_TEST_FRACTION = 0.25
INTEGRATION_ATE_TOLERANCE = 0.05
INTEGRATION_ATE_KINDS = [
    ATEKind.IPW,
    ATEKind.AIPW,
    ATEKind.DML_IRM,
    ATEKind.DML_PLR,
]


def pipeline_config(
    ate_kind: ATEKind,
    hold_out_for_policy: bool,
    data_config: DataConfig,
    results_dir: Path,
    logistic_learner: BaseEstimator,
    linear_learner: BaseEstimator,
    sensitivity: SensitivityConfig | None,
) -> PipelineConfig:
    if ate_kind == ATEKind.IPW:
        ate_spec = IPWATEEstimatorSpec(
            kind=ate_kind,
            propensity_learner=logistic_learner,
            n_folds=INTEGRATION_FOLDS,
        )
    elif ate_kind == ATEKind.AIPW:
        ate_spec = DoublyRobustATEEstimatorSpec(
            kind=ate_kind,
            outcome_learner=linear_learner,
            propensity_learner=logistic_learner,
            n_folds=INTEGRATION_FOLDS,
        )
    else:
        ate_spec = DoubleMLATEEstimatorSpec(
            kind=ate_kind,
            outcome_learner=linear_learner,
            propensity_learner=logistic_learner,
            n_folds=INTEGRATION_FOLDS,
        )
    policy = None
    test_fraction = 0.0
    if hold_out_for_policy:
        policy = [PolicyTreeMethodSpec(kind=PolicyKind.POLICY_TREE)]
        test_fraction = INTEGRATION_TEST_FRACTION
    return PipelineConfig(
        data=data_config,
        split=SplitConfig(
            test_fraction=test_fraction,
            random_state=DEFAULT_SPLIT_RANDOM_STATE,
        ),
        overlap=OverlapConfig(propensity_learner=logistic_learner, n_folds=INTEGRATION_FOLDS),
        ate_estimators=[ate_spec],
        cate_estimators=[
            MetaCATEEstimatorSpec(kind=CATEKind.S_LEARNER, outcome_learner=linear_learner),
            MetaCATEEstimatorSpec(kind=CATEKind.T_LEARNER, outcome_learner=linear_learner),
        ],
        sensitivity=sensitivity,
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=logistic_learner,
            outcome_learner=linear_learner,
            dr_crossfit_folds=INTEGRATION_FOLDS,
            rate_bootstrap_samples=INTEGRATION_BOOTSTRAP_SAMPLES,
            eceth_bootstrap_samples=INTEGRATION_BOOTSTRAP_SAMPLES,
        ),
        policy=policy,
        cate_crossfit_folds=INTEGRATION_FOLDS,
        policy_bootstrap_samples=INTEGRATION_BOOTSTRAP_SAMPLES,
        results_dir=str(results_dir),
    )


@pytest.mark.integration
@pytest.mark.parametrize("hold_out_for_policy", [False, True])
@pytest.mark.parametrize("ate_kind", INTEGRATION_ATE_KINDS)
def test_pipeline_recovers_constant_effect(
    ate_kind: ATEKind,
    hold_out_for_policy: bool,
    test_output_dir: Path,
    recovery_data_config: DataConfig,
    constant_effect_cohort: tuple[pd.DataFrame, float],
    logistic_learner: BaseEstimator,
    linear_learner: BaseEstimator,
) -> None:
    frame, truth = constant_effect_cohort
    config = pipeline_config(
        ate_kind=ate_kind,
        hold_out_for_policy=hold_out_for_policy,
        data_config=recovery_data_config,
        results_dir=test_output_dir,
        logistic_learner=logistic_learner,
        linear_learner=linear_learner,
        sensitivity=None,
    )
    CausalPipeline(config=config).run(df_input=frame)

    ate_summary = read_dataframe(test_output_dir / "summary" / "ate_estimators.csv")
    cate_summary = read_dataframe(test_output_dir / "summary" / "cate_estimators.csv")
    assert ate_summary.shape[0] == 1
    assert cate_summary.shape[0] == 2
    assert list(ate_summary["estimator_kind"]) == [ate_kind.value]
    assert set(cate_summary["estimator_kind"]) == {"s_learner", "t_learner"}
    assert abs(float(ate_summary["estimate"].iloc[0]) - truth) < INTEGRATION_ATE_TOLERANCE
    assert np.isfinite(cate_summary["mean_crossfit_cate"]).all()
    assert np.isfinite(cate_summary["eceth"]).all()
    contrast_dir = test_output_dir / "cate" / "comparison" / "1_vs_0"
    assert (test_output_dir / "diagnostics" / "propensity_overlap.png").is_file()
    assert (test_output_dir / "diagnostics" / "propensity_overlap.csv").is_file()
    assert (test_output_dir / "data" / "estimation.parquet").is_file()
    assert (test_output_dir / "summary" / "ate_intervals.png").is_file()
    assert (test_output_dir / "summary" / "cate_mean_effects.png").is_file()
    assert (contrast_dir / "toc.png").is_file()
    assert (contrast_dir / "toc.csv").is_file()
    assert (contrast_dir / "calibration.png").is_file()
    assert (contrast_dir / "calibration.csv").is_file()
    s_learner_id = cate_summary.loc[cate_summary["estimator_kind"] == "s_learner", "estimator"].iloc[0]
    assert (test_output_dir / "cate" / s_learner_id / "1_vs_0" / "toc.png").is_file()
    assert (test_output_dir / "cate" / s_learner_id / "1_vs_0" / "toc.csv").is_file()
    assert (test_output_dir / "cate" / s_learner_id / "1_vs_0" / "calibration.png").is_file()
    policy_summary = test_output_dir / "summary" / "policy_methods.csv"
    if hold_out_for_policy:
        policy_table = read_dataframe(policy_summary)
        assert policy_table.shape[0] >= 1
        assert policy_table["effect_estimate"].notna().all()
        assert set(policy_table["data_split"]) == {"train", "test"}
        assert policy_table["rule"].str.startswith("leaf_").sum() == 0
        slice_dirs = list((test_output_dir / "policy" / "slices").glob("*"))
        assert slice_dirs
        assert slice_dirs[0].joinpath("policy_tree.png").is_file()
        assert slice_dirs[0].joinpath("rule_performance.csv").is_file()
        assert (contrast_dir / "test_toc.png").is_file()
        assert (contrast_dir / "test_toc.csv").is_file()
        assert (contrast_dir / "test_calibration.png").is_file()
        assert (contrast_dir / "test_calibration.csv").is_file()
    else:
        assert not policy_summary.exists()


@pytest.mark.integration
def test_pipeline_reports_selection_and_doubleml_sensitivity(
    test_output_dir: Path,
    recovery_data_config: DataConfig,
    constant_effect_cohort: tuple[pd.DataFrame, float],
    logistic_learner: BaseEstimator,
    linear_learner: BaseEstimator,
) -> None:
    frame, truth = constant_effect_cohort
    selected = frame.copy()
    selected["included"] = (selected["x1"] > 0.0).astype(int)
    sensitivity = SensitivityConfig(
        outcome_learner=LinearRegression(),
        sampling_learner=LogisticRegression(max_iter=500),
        propensity_learner=LogisticRegression(max_iter=500),
        selection_column="included",
        inclusion_fraction=0.5,
    )
    config = pipeline_config(
        ate_kind=ATEKind.DML_IRM,
        hold_out_for_policy=False,
        data_config=recovery_data_config,
        results_dir=test_output_dir,
        logistic_learner=logistic_learner,
        linear_learner=linear_learner,
        sensitivity=sensitivity,
    )
    CausalPipeline(config=config).run(df_input=selected)

    ate_summary = read_dataframe(test_output_dir / "summary" / "ate_estimators.csv")
    transported = read_dataframe(test_output_dir / "selection" / "transport.csv")
    assert ate_summary.shape[0] == 1
    assert abs(float(ate_summary["estimate"].iloc[0]) - truth) < INTEGRATION_ATE_TOLERANCE
    assert ate_summary["robustness_value_point"].notna().all()
    assert transported.shape[0] == 1
    assert np.isfinite(transported["doubly_robust"]).all()
    dml_id = ate_summary["estimator"].iloc[0]
    assert (test_output_dir / "ate" / dml_id / "sensitivity" / "contour.html").is_file()
    assert (test_output_dir / "ate" / dml_id / "sensitivity" / "contour.csv").is_file()
    assert (test_output_dir / "ate" / dml_id / "selection" / "unseen.csv").is_file()

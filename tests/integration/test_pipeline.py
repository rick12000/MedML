from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from causal_pipeline.learner_defaults import DEFAULT_LOGISTIC_MAX_ITER, DEFAULT_XGBOOST_MAX_DEPTH
from causal_pipeline.parallel import available_cores

from causal_pipeline.config import (
    ATEKind,
    CATEEvaluationConfig,
    CATEEstimatorSpec,
    CATEKind,
    CausalForestCATEEstimatorSpec,
    DataConfig,
    DEFAULT_SPLIT_RANDOM_STATE,
    OverlapConfig,
    DoublyRobustATEEstimatorSpec,
    DoubleMLATEEstimatorSpec,
    IPWATEEstimatorSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PipelineConfig,
    PolicyKind,
    PolicyTreeMethodSpec,
    SensitivityConfig,
    SplitConfig,
    TreatmentMode,
)
from causal_pipeline.pipeline import CausalPipeline
from causal_pipeline.utils import read_dataframe

INTEGRATION_FOLDS = 2
INTEGRATION_BOOTSTRAP_SAMPLES = 8
INTEGRATION_TEST_FRACTION = 0.25
INTEGRATION_ATE_TOLERANCE = 0.05
INTEGRATION_LEARNER_OBSERVATIONS = 800
INTEGRATION_BOOST_TREES = 20
INTEGRATION_ATE_KINDS = [
    ATEKind.IPW,
    ATEKind.AIPW,
    ATEKind.DML_IRM,
    ATEKind.DML_PLR,
]
INTEGRATION_CATE_KINDS = [
    CATEKind.S_LEARNER,
    CATEKind.T_LEARNER,
    CATEKind.X_LEARNER,
    CATEKind.R_LEARNER,
    CATEKind.DR_LEARNER,
    CATEKind.CAUSAL_FOREST,
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
    assert (contrast_dir / "evaluation.png").is_file()
    assert (contrast_dir / "toc.csv").is_file()
    assert (contrast_dir / "calibration.csv").is_file()
    s_learner_id = cate_summary.loc[cate_summary["estimator_kind"] == "s_learner", "estimator"].iloc[0]
    assert (test_output_dir / "cate" / s_learner_id / "1_vs_0" / "toc.csv").is_file()
    assert (test_output_dir / "cate" / s_learner_id / "1_vs_0" / "calibration.csv").is_file()
    policy_summary = test_output_dir / "summary" / "policy_methods.csv"
    assert "test_eceth" not in cate_summary.columns
    if hold_out_for_policy:
        policy_table = read_dataframe(policy_summary)
        assert policy_table.shape[0] >= 1
        assert policy_table["effect_estimate"].notna().all()
        assert set(policy_table["data_split"]) == {"test"}
        assert policy_table["rule"].str.startswith("leaf_").sum() == 0
        slice_dirs = list((test_output_dir / "policy" / "slices").glob("*"))
        assert slice_dirs
        assert slice_dirs[0].joinpath("policy_tree.png").is_file()
        assert slice_dirs[0].joinpath("rule_performance.csv").is_file()
        assert not (contrast_dir / "test_evaluation.png").exists()
        assert not (contrast_dir / "test_toc.csv").exists()
        assert not (contrast_dir / "test_calibration.csv").exists()
        assert not (contrast_dir / "test_evaluation.csv").exists()
        for estimator_id in cate_summary["estimator"].unique():
            contrast_path = test_output_dir / "cate" / estimator_id / "1_vs_0"
            assert (contrast_path / "evaluation.csv").is_file()
            assert not (contrast_path / "test_evaluation.csv").exists()
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
        sampling_learner=LogisticRegression(max_iter=DEFAULT_LOGISTIC_MAX_ITER),
        propensity_learner=LogisticRegression(max_iter=DEFAULT_LOGISTIC_MAX_ITER),
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


def integration_cohort(outcome_type: OutcomeType) -> tuple[pd.DataFrame, DataConfig]:
    generator = np.random.default_rng(DEFAULT_SPLIT_RANDOM_STATE)
    covariate_one = generator.normal(size=INTEGRATION_LEARNER_OBSERVATIONS)
    covariate_two = generator.normal(size=INTEGRATION_LEARNER_OBSERVATIONS)
    propensity = 1.0 / (1.0 + np.exp(-(0.5 * covariate_one - 0.4 * covariate_two)))
    treatment = generator.binomial(1, propensity)
    if outcome_type == OutcomeType.BINARY:
        outcome_probability = 1.0 / (1.0 + np.exp(-(0.4 * covariate_one + treatment)))
        outcome = generator.binomial(1, outcome_probability)
    else:
        outcome = 1.5 * treatment + 0.8 * covariate_one + generator.normal(
            scale=0.5,
            size=INTEGRATION_LEARNER_OBSERVATIONS,
        )
    frame = pd.DataFrame({"y": outcome, "t": treatment, "x1": covariate_one, "x2": covariate_two})
    config = DataConfig(
        outcome="y",
        treatment="t",
        confounders=["x1", "x2"],
        effect_modifiers=["x1"],
        outcome_type=outcome_type,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    return frame, config


def integration_cate_spec(
    kind: CATEKind,
    outcome_learner: BaseEstimator,
    effect_learner: BaseEstimator,
    propensity_learner: BaseEstimator,
) -> CATEEstimatorSpec:
    if kind == CATEKind.CAUSAL_FOREST:
        return CausalForestCATEEstimatorSpec(
            kind=kind,
            alias=kind.value,
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
        )
    return MetaCATEEstimatorSpec(
        kind=kind,
        alias=kind.value,
        outcome_learner=outcome_learner,
        effect_learner=effect_learner,
        propensity_learner=propensity_learner,
        n_folds=INTEGRATION_FOLDS,
        random_state=DEFAULT_SPLIT_RANDOM_STATE,
    )


@pytest.mark.integration
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("outcome_type", [OutcomeType.CONTINUOUS, OutcomeType.BINARY])
@pytest.mark.parametrize("cate_kind", INTEGRATION_CATE_KINDS)
def test_pipeline_fits_each_learner_end_to_end(
    cate_kind: CATEKind,
    outcome_type: OutcomeType,
    parallel: bool,
    test_output_dir: Path,
) -> None:
    if parallel and available_cores() < 2:
        pytest.skip("Process parallelism needs at least two cores.")
    frame, data_config = integration_cohort(outcome_type)
    propensity_learner = XGBClassifier(
        n_estimators=INTEGRATION_BOOST_TREES,
        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
        n_jobs=1,
        random_state=DEFAULT_SPLIT_RANDOM_STATE,
        eval_metric="logloss",
    )
    effect_learner = XGBRegressor(
        n_estimators=INTEGRATION_BOOST_TREES,
        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
        n_jobs=1,
        random_state=DEFAULT_SPLIT_RANDOM_STATE + 1,
    )
    if outcome_type == OutcomeType.BINARY:
        outcome_learner = XGBClassifier(
            n_estimators=INTEGRATION_BOOST_TREES,
            max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
            n_jobs=1,
            random_state=DEFAULT_SPLIT_RANDOM_STATE + 2,
            eval_metric="logloss",
        )
    else:
        outcome_learner = XGBRegressor(
            n_estimators=INTEGRATION_BOOST_TREES,
            max_depth=DEFAULT_XGBOOST_MAX_DEPTH,
            n_jobs=1,
            random_state=DEFAULT_SPLIT_RANDOM_STATE + 2,
        )
    config = PipelineConfig(
        data=data_config,
        split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
        overlap=OverlapConfig(propensity_learner=propensity_learner, n_folds=INTEGRATION_FOLDS),
        ate_estimators=[
            IPWATEEstimatorSpec(
                kind=ATEKind.IPW,
                alias="ipw",
                propensity_learner=propensity_learner,
                n_folds=INTEGRATION_FOLDS,
            )
        ],
        cate_estimators=[
            integration_cate_spec(
                kind=cate_kind,
                outcome_learner=outcome_learner,
                effect_learner=effect_learner,
                propensity_learner=propensity_learner,
            )
        ],
        cate_evaluation=CATEEvaluationConfig(
            propensity_learner=propensity_learner,
            outcome_learner=outcome_learner,
            dr_crossfit_folds=INTEGRATION_FOLDS,
            rate_bootstrap_samples=INTEGRATION_BOOTSTRAP_SAMPLES,
            eceth_bootstrap_samples=INTEGRATION_BOOTSTRAP_SAMPLES,
        ),
        cate_crossfit_folds=INTEGRATION_FOLDS,
        parallel=parallel,
        results_dir=str(test_output_dir),
    )
    CausalPipeline(config=config).run(df_input=frame)

    ate_summary = read_dataframe(test_output_dir / "summary" / "ate_estimators.csv")
    cate_summary = read_dataframe(test_output_dir / "summary" / "cate_estimators.csv")
    assert ate_summary.shape[0] == 1
    assert cate_summary.shape[0] == 1
    assert list(ate_summary["estimator_kind"]) == [ATEKind.IPW.value]
    assert list(cate_summary["estimator_kind"]) == [cate_kind.value]
    assert np.isfinite(ate_summary["estimate"]).all()
    assert np.isfinite(cate_summary["mean_crossfit_cate"]).all()
    assert (test_output_dir / "diagnostics" / "propensity_overlap.csv").is_file()
    assert (test_output_dir / "data" / "estimation.parquet").is_file()
    assert (test_output_dir / "cate" / cate_kind.value / "crossfit_predictions.parquet").is_file()
    assert (test_output_dir / "cate" / "comparison" / "1_vs_0" / "toc.csv").is_file()

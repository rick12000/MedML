from pathlib import Path

from causal_pipeline.ate import IPWAdapter, estimator_supports_sensitivity
from causal_pipeline.config import (
    ATEKind,
    IPWATEEstimatorSpec,
    PipelineConfig,
    SensitivityConfig,
    SplitConfig,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.pipeline import CausalPipeline


def test_ipw_does_not_publish_a_hand_rolled_sensitivity_bound(
    binary_data_config,
    logistic_learner,
    df_synthetic_binary,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = IPWAdapter(
        spec=IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        data=dataset,
    )
    assert estimator_supports_sensitivity(estimator) is False


def test_skipped_sensitivity_removes_a_previous_bound(
    tmp_path: Path,
    binary_data_config,
    logistic_learner,
    df_synthetic_binary,
) -> None:
    config = PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(
            train_fraction=0.7,
            validation_fraction=0.3,
            test_fraction=0.0,
            random_state=0,
        ),
        ate_estimators=[
            IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        ],
        sensitivity=SensitivityConfig(outcome_learner=logistic_learner),
        results_dir=str(tmp_path),
    )
    pipeline = CausalPipeline(config)
    stale = pipeline.results.ate_dir("ipw") / "sensitivity"
    stale.mkdir(parents=True)
    (stale / "summary.csv").write_text("old\n", encoding="utf-8")
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = IPWAdapter(
        spec=IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        data=dataset,
    )
    pipeline.run_ate_sensitivity({"ipw": estimator}, dataset)
    assert not stale.exists()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from causal_pipeline.ate import (
    IPWAdapter,
    confounding_multiple_to_null,
    estimator_supports_sensitivity,
)
from causal_pipeline.config import (
    ATEKind,
    DataConfig,
    IPWATEEstimatorSpec,
    PipelineConfig,
    SensitivityConfig,
    SplitConfig,
    DEFAULT_SPLIT_RANDOM_STATE,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.pipeline import CausalPipeline


def test_ipw_does_not_publish_a_hand_rolled_sensitivity_bound(
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = IPWAdapter(
        spec=IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        data=dataset,
    )
    assert estimator_supports_sensitivity(estimator) is False


def test_skipped_sensitivity_removes_a_previous_bound(
    tmp_path: Path,
    binary_data_config: DataConfig,
    logistic_learner: BaseEstimator,
    df_synthetic_binary: pd.DataFrame,
) -> None:
    config = PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(test_fraction=0.0, random_state=DEFAULT_SPLIT_RANDOM_STATE),
        ate_estimators=[
            IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        ],
        sensitivity=SensitivityConfig(outcome_learner=logistic_learner),
        results_dir=str(tmp_path),
    )
    pipeline = CausalPipeline(config=config)
    stale = pipeline.results.ate_dir("ipw") / "sensitivity"
    stale.mkdir(parents=True)
    (stale / "summary.csv").write_text("old\n", encoding="utf-8")
    dataset = CausalDataset(data=binary_data_config, df=df_synthetic_binary)
    estimator = IPWAdapter(
        spec=IPWATEEstimatorSpec(kind=ATEKind.IPW, propensity_learner=logistic_learner),
        data=dataset,
    )
    pipeline.run_ate_sensitivity(ate_models={"ipw": estimator})
    assert not stale.exists()


class RecordingSensitivityModel:
    def __init__(self, estimate: float, interval_half_width: float) -> None:
        self.estimate = estimate
        self.interval_half_width = interval_half_width
        self.calls: list[tuple[float, float]] = []
        self._framework = self

    def _calc_sensitivity_analysis(
        self,
        cf_y: float,
        cf_d: float,
        rho: float,
        level: float,
    ) -> dict[str, dict[str, np.ndarray]]:
        self.calls.append((cf_y, cf_d))
        bias = cf_y + cf_d
        return {
            "theta": {
                "lower": np.array([self.estimate - bias]),
                "upper": np.array([self.estimate + bias]),
            },
            "ci": {
                "lower": np.array([self.estimate - self.interval_half_width - bias]),
                "upper": np.array([self.estimate + self.interval_half_width + bias]),
            },
        }


def test_confounding_multiple_follows_the_benchmark_ray() -> None:
    weak = RecordingSensitivityModel(estimate=1.0, interval_half_width=0.5)
    point = confounding_multiple_to_null(
        model=weak,
        treatment_strength=0.01,
        outcome_strength=0.01,
        null_effect=0.0,
        level=0.95,
        bound="theta",
    )
    wide = RecordingSensitivityModel(estimate=1.0, interval_half_width=0.5)
    interval = confounding_multiple_to_null(
        model=wide,
        treatment_strength=0.2,
        outcome_strength=0.2,
        null_effect=0.0,
        level=0.95,
        bound="ci",
    )
    narrow = RecordingSensitivityModel(estimate=1.0, interval_half_width=0.5)
    point_again = confounding_multiple_to_null(
        model=narrow,
        treatment_strength=0.2,
        outcome_strength=0.2,
        null_effect=0.0,
        level=0.95,
        bound="theta",
    )
    assert 49.0 < point < 51.0
    assert interval < point_again
    for outcome_strength, treatment_strength in weak.calls:
        assert np.isclose(outcome_strength, treatment_strength)

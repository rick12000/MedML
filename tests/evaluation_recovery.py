"""Fit S/T learners and run CATE evaluation with pipeline-style figure output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_pipeline.cate import create_cate_estimator
from causal_pipeline.config import (
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
    MetaCATEEstimatorSpec,
    OutcomeFavorability,
)
from causal_pipeline.data import CausalDataset
from causal_pipeline.evaluation import CATEEvaluator

RECOVERY_EVAL_FOLDS = 5
RECOVERY_RATE_BOOTSTRAP = 20
RECOVERY_ECETH_BOOTSTRAP = 20
RECOVERY_RANDOM_STATE = 0


def write_linear_meta_learner_evaluation_figures(
    frame: pd.DataFrame,
    data_config: DataConfig,
    results_root: Path,
) -> dict[str, list]:
    """Cross-fit DR scores, fit S/T learners, write TOC and calibration like the pipeline."""
    dataset = CausalDataset(data=data_config, df=frame)
    evaluation = CATEEvaluationConfig(
        propensity_learner=LogisticRegression(max_iter=500),
        outcome_learner=LinearRegression(),
        dr_crossfit_folds=RECOVERY_EVAL_FOLDS,
        calibration_bins=5,
        rate_bootstrap_samples=RECOVERY_RATE_BOOTSTRAP,
        eceth_bootstrap_samples=RECOVERY_ECETH_BOOTSTRAP,
        random_state=RECOVERY_RANDOM_STATE,
    )
    evaluator = CATEEvaluator(evaluation=evaluation)
    robust_scores = evaluator.build_robust_scores(dataset=dataset)
    predictions: dict[str, pd.DataFrame] = {}
    for kind, label in ((CATEKind.S_LEARNER, "s_learner"), (CATEKind.T_LEARNER, "t_learner")):
        estimator = create_cate_estimator(
            spec=MetaCATEEstimatorSpec(kind=kind, outcome_learner=LinearRegression()),
            data=dataset,
        )
        estimator.fit(data=dataset)
        predictions[label] = estimator.predict_effects(data=dataset)
    return evaluator.evaluate_cate_models(
        predictions=predictions,
        robust_scores=robust_scores,
        results_root=results_root,
        outcome_favorability=data_config.outcome_favorability,
    )

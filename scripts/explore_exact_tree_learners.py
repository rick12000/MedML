"""Exploration (not a test): do R, X and T learner exact welfare trees collapse to one policy?

Runs the real pipeline pieces on synthetic data with known heterogeneity:
cross-fit CATEs -> PolicyService exact welfare tree -> held-out assignment -> value table.
Prints, per search setting, how the three learners' CATEs, trees and assignments compare.

Run: uv run python scripts/explore_exact_tree_learners.py
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from causal_pipeline.core.config import (
    CATEEvaluationConfig,
    CATEKind,
    DataConfig,
    ExactPolicyTreeMethodSpec,
    MetaCATEEstimatorSpec,
    OutcomeType,
    PolicyKind,
    PolicySearchMode,
    TreatmentMode,
)
from causal_pipeline.estimation.cate import create_cate_estimator
from causal_pipeline.nuisance.crossfit import cross_fit_predictions
from causal_pipeline.core.data import CausalDataset
from causal_pipeline.estimation.evaluation import CATEEvaluator
from causal_pipeline.policies.policy import PolicyService, assigned_treatments
from causal_pipeline.policies.policy_value import evaluate_policy_values

logging.disable(logging.CRITICAL)

LEARNERS = (CATEKind.R_LEARNER, CATEKind.X_LEARNER, CATEKind.T_LEARNER)
BASE_LEARNER = "linear"
FEATURES = ["x1", "x2", "x3", "x4"]


def simulate(n: int, seed: int, treated_share: float, effect_scale: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    propensity = 1.0 / (1.0 + np.exp(-(np.log(treated_share / (1 - treated_share)) + 0.5 * x[:, 0])))
    treatment = rng.binomial(1, propensity)
    tau = effect_scale * (x[:, 1] + 0.5 * x[:, 2])  # heterogeneous, mean zero
    outcome = x[:, 0] + 0.5 * x[:, 3] + treatment * tau + rng.normal(size=n)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["t"] = treatment
    frame["y"] = outcome
    return frame


def data_config() -> DataConfig:
    return DataConfig(
        outcome="y",
        treatment="t",
        confounders=FEATURES,
        effect_modifiers=FEATURES,
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )


def cate_predictions(estimation: CausalDataset, kind: CATEKind) -> pd.DataFrame:
    spec = MetaCATEEstimatorSpec(
        kind=kind,
        outcome_learner=BASE_LEARNER,
        effect_learner=BASE_LEARNER,
        propensity_learner=BASE_LEARNER,
    )

    def predict(train: CausalDataset, held_out: CausalDataset) -> pd.DataFrame:
        estimator = create_cate_estimator(spec=spec, data=train, forest_n_jobs=1, tune_estimators=False)
        estimator.fit(data=train)
        return estimator.predict_effects(data=held_out)

    return cross_fit_predictions(
        dataset=estimation, n_folds=3, random_state=0, predict_fold=predict
    )


def run_setting(
    n: int,
    seed: int,
    treated_share: float,
    effect_scale: float,
    min_node_size: int,
    split_step: int,
    depth: int,
) -> None:
    config = data_config()
    frame = simulate(2 * n, seed, treated_share, effect_scale)
    estimation = CausalDataset(data=config, df=frame.iloc[:n].reset_index(drop=True))
    test = CausalDataset(data=config, df=frame.iloc[n:].reset_index(drop=True))
    predictions = {kind.value: cate_predictions(estimation, kind) for kind in LEARNERS}

    cate_corr = pd.DataFrame({name: frame_["1_vs_0"] for name, frame_ in predictions.items()}).corr()
    service = PolicyService(
        methods=[
            ExactPolicyTreeMethodSpec(
                kind=PolicyKind.EXACT_POLICY_TREE,
                mode=PolicySearchMode.WELFARE,
                depth=depth,
                min_node_size=min_node_size,
                split_step=split_step,
            )
        ],
        bootstrap_samples=200,
        random_state=0,
    )
    policies = service.fit_policy_methods(estimation=estimation, predictions=predictions)
    evaluator = CATEEvaluator(
        CATEEvaluationConfig(propensity_learner=BASE_LEARNER, outcome_learner=BASE_LEARNER)
    )
    scores = evaluator.build_test_validation_scores(
        estimation=estimation, test=test, signal=_signal(), include_control_outcome=True,
    )
    report = evaluate_policy_values(
        policies=policies,
        test=test,
        test_scores=scores,
        bootstrap_samples=200,
        random_state=0,
    )

    print(
        f"\n=== n={n} seed={seed} treated={treated_share:.2f} tau_scale={effect_scale} "
        f"min_node={min_node_size} step={split_step} depth={depth} ==="
    )
    print("CATE correlation across learners:")
    print(cate_corr.round(3).to_string())
    assigned = {}
    for policy in policies:
        arm = assigned_treatments(policy=policy, dataset=test).astype(int)
        assigned[policy.source_cate_model] = arm
        print(
            f"  {policy.source_cate_model:10s} welfare={policy.model.welfare:9.3f} "
            f"test treated={arm.mean():.3f}  rules: "
            + "; ".join(f"[{r.rule}] -> {r.recommended_treatment}" for r in policy.rules)
        )
    names = list(assigned)
    for left, right in itertools.combinations(names, 2):
        same = np.array_equal(assigned[left], assigned[right])
        print(f"  assignments {left} vs {right}: identical={same}, "
              f"disagree={int((assigned[left] != assigned[right]).sum())} / {len(test.df)}")
    table = report.table.loc[
        report.table["method"] == "exact_policy_welfare",
        ["source_cate_model", "share_treated", "vs_never_treat", "vs_never_treat_ci_lower", "vs_never_treat_ci_upper"],
    ]
    print(table.round(3).to_string(index=False))


def _signal():
    from causal_pipeline.core.config import TestValidationSignal

    return TestValidationSignal.ESTIMATION_FULL_FIT


if __name__ == "__main__":
    # Defaults the pipeline uses (min_node_size=1, split_step=1, depth=2), then coarser searches.
    settings = [
        dict(n=400, seed=1, treated_share=0.15, effect_scale=1.0, min_node_size=1, split_step=1, depth=2),
        dict(n=400, seed=1, treated_share=0.15, effect_scale=0.05, min_node_size=1, split_step=1, depth=2),
        dict(n=400, seed=1, treated_share=0.15, effect_scale=1.0, min_node_size=50, split_step=25, depth=2),
        dict(n=400, seed=1, treated_share=0.15, effect_scale=1.0, min_node_size=150, split_step=25, depth=2),
        dict(n=400, seed=1, treated_share=0.15, effect_scale=1.0, min_node_size=1, split_step=1, depth=1),
    ]
    for setting in settings:
        run_setting(**setting)

"""Exploration (not a test): value of the *observed* assignment, current vs proposed estimator.

The observed policy's gain for patient i is Y_i(T_i) - Y_i(0). With simulated potential outcomes
the realized truth is known, so each estimator's error can be measured over repeated draws.

Estimators of mean[T_i * (Y_i(1) - Y_i(0))] on the held-out set:
  current   : the pipeline's DR contrast of the realized arm (T * (g1 - g0))
  proposed  : mean[ T (Y - mu0) - (1 - T) * e/(1-e) * (Y - mu0) ]   (ATT-style AIPW, fixed T_i)
  plug-in   : mean[ T (Y - mu0) ]                                    (no control correction)

Run: uv run python scripts/explore_observed_policy_value.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from causal_pipeline.core.config import (
    CATEEvaluationConfig,
    DataConfig,
    OutcomeType,
    TreatmentMode,
)
from causal_pipeline.core.data import CausalDataset
from causal_pipeline.estimation.evaluation import CATEEvaluator

logging.disable(logging.CRITICAL)

FEATURES = ["x1", "x2", "x3"]
LEARNER = "linear"  # deliberately misspecified for the nonlinear outcomes below
CLIP = (0.02, 0.98)


def simulate(n: int, rng: np.random.Generator, treated_share: float) -> pd.DataFrame:
    x = rng.normal(size=(n, len(FEATURES)))
    logit = np.log(treated_share / (1 - treated_share)) + 0.8 * x[:, 0]
    treatment = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    mu0 = 1.5 * np.sin(1.5 * x[:, 0]) + x[:, 1] ** 2
    mu1 = mu0 - 0.3 + 0.8 * x[:, 2] + 0.6 * np.cos(x[:, 1])
    y0 = mu0 + rng.normal(size=n)
    y1 = mu1 + rng.normal(size=n)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["t"] = treatment
    frame["y"] = np.where(treatment == 1, y1, y0)
    frame["y0"] = y0
    frame["y1"] = y1
    return frame


def run(n_estimation: int, n_test: int, treated_share: float, repeats: int) -> None:
    config = DataConfig(
        outcome="y",
        treatment="t",
        confounders=FEATURES,
        effect_modifiers=FEATURES,
        outcome_type=OutcomeType.CONTINUOUS,
        treatment_mode=TreatmentMode.BINARY,
        control_value=0,
        treatment_values=[0, 1],
    )
    evaluator = CATEEvaluator(
        CATEEvaluationConfig(propensity_learner=LEARNER, outcome_learner=LEARNER)
    )
    propensity_template, outcome_template = (
        __import__("causal_pipeline.estimation.evaluation", fromlist=["x"]).evaluation_learner_templates(
            evaluator.evaluation, binary=False
        )
    )
    errors: dict[str, list[float]] = {"current": [], "proposed": [], "plug-in": []}
    truths: list[float] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(repeat)
        estimation_frame = simulate(n_estimation, rng, treated_share)
        test_frame = simulate(n_test, rng, treated_share)
        estimation = CausalDataset(data=config, df=estimation_frame.drop(columns=["y0", "y1"]))
        test = CausalDataset(data=config, df=test_frame.drop(columns=["y0", "y1"]))
        contrasts, propensity, outcomes = evaluator.doubly_robust_contrasts(
            train=estimation,
            score_rows=test,
            propensity_template=propensity_template,
            outcome_template=outcome_template,
        )
        t = test_frame["t"].to_numpy(dtype=float)
        y = test_frame["y"].to_numpy()
        mu0 = outcomes[0].to_numpy(dtype=float)
        e1 = np.clip(propensity[1].to_numpy(dtype=float), *CLIP)
        truth = float(np.mean(t * (test_frame["y1"] - test_frame["y0"])))
        current = float(np.mean(t * contrasts["1_vs_0"].to_numpy()))
        weight = e1 / (1.0 - e1)
        proposed = float(np.mean(t * (y - mu0) - (1.0 - t) * weight * (y - mu0)))
        plug_in = float(np.mean(t * (y - mu0)))
        truths.append(truth)
        errors["current"].append(current - truth)
        errors["proposed"].append(proposed - truth)
        errors["plug-in"].append(plug_in - truth)

    print(
        f"\n=== n_est={n_estimation} n_test={n_test} treated~{treated_share:.2f} repeats={repeats} "
        f"(mean truth {np.mean(truths):+.3f}, sd of truth {np.std(truths):.3f}) ==="
    )
    rows = [
        {
            "estimator": name,
            "mean error (bias)": np.mean(values),
            "sd": np.std(values, ddof=1),
            "RMSE": float(np.sqrt(np.mean(np.square(values)))),
            "max |error|": np.max(np.abs(values)),
        }
        for name, values in errors.items()
    ]
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    run(n_estimation=600, n_test=208, treated_share=0.13, repeats=60)
    run(n_estimation=600, n_test=208, treated_share=0.40, repeats=60)

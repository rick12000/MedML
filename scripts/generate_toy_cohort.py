"""Simulate an observational cohort for the MedML causal pipeline.

Generates ``data/analysis.parquet`` with columns matching
``causal_pipeline.settings`` (binary treatment/outcome, confounding,
and heterogeneous treatment effects on the log-odds scale).

Usage (from repository root)::

    .\\.venv\\Scripts\\python.exe scripts/generate_toy_cohort.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.settings import (
    DATA_CONFOUNDERS,
    DATA_EFFECT_MODIFIERS,
    DATA_OUTCOME_COLUMN,
    DATA_TREATMENT_COLUMN,
)
from causal_pipeline.utils import write_dataframe

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "analysis.parquet"


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_observational_cohort(
    n_observations: int,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Draw a synthetic chronic-care cohort with confounded treatment and CATE."""
    rng = np.random.default_rng(random_state)

    age = rng.uniform(45.0, 85.0, n_observations)
    baseline_score = rng.beta(2.2, 4.5, n_observations) * 100.0
    biomarker = rng.normal(0.0, 1.0, n_observations) + 0.35 * (
        baseline_score / 100.0 - 0.35
    )
    comorbidity_risk = np.clip(
        0.12 + 0.38 * (age - 45.0) / 40.0 + 0.22 * (baseline_score > 55.0),
        0.05,
        0.85,
    )
    comorbidity = (rng.random(n_observations) < comorbidity_risk).astype(np.int64)

    # Confounded treatment: higher severity and comorbidity increase treatment uptake.
    logit_propensity = (
        -1.15
        + 0.032 * (age - 65.0)
        + 0.028 * baseline_score
        + 0.62 * comorbidity
        + 0.42 * biomarker
        + 0.18 * (baseline_score / 100.0) * biomarker
    )
    propensity = _logistic(logit_propensity)
    treatment = (rng.random(n_observations) < propensity).astype(np.int64)

    # Heterogeneous benefit: younger, higher baseline severity, higher biomarker.
    tau_log_odds = (
        0.38
        - 0.014 * (age - 65.0)
        + 0.52 * (baseline_score / 100.0 - 0.38)
        + 0.28 * biomarker
        - 0.15 * comorbidity
    )

    logit_control_outcome = (
        -0.75
        + 0.022 * baseline_score
        + 0.48 * comorbidity
        + 0.17 * biomarker
        + 0.012 * (age - 65.0)
        + 0.09 * (baseline_score / 100.0) * comorbidity
    )
    logit_treated_outcome = logit_control_outcome + tau_log_odds

    p_control = _logistic(logit_control_outcome)
    p_treated = _logistic(logit_treated_outcome)
    p_observed = np.where(treatment == 1, p_treated, p_control)
    outcome = (rng.random(n_observations) < p_observed).astype(np.int64)

    df = pd.DataFrame(
        {
            DATA_OUTCOME_COLUMN: outcome,
            DATA_TREATMENT_COLUMN: treatment,
            "age": np.round(age, 1),
            "baseline_score": np.round(baseline_score, 2),
            "biomarker": np.round(biomarker, 4),
            "comorbidity": comorbidity,
        },
    )

    required = [
        DATA_OUTCOME_COLUMN,
        DATA_TREATMENT_COLUMN,
        *DATA_CONFOUNDERS,
        *DATA_EFFECT_MODIFIERS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Generated data missing columns: {missing}")
    if df[required].isna().any().any():
        raise RuntimeError("Generated data contains missing values.")

    true_ate_risk_diff = float(np.mean(p_treated - p_control))
    treated_share = float(treatment.mean())
    outcome_rate = float(outcome.mean())
    logger.info(
        "Simulated n=%s | P(T=1)=%.3f | P(Y=1)=%.3f | true ATE (risk diff)=%.4f",
        n_observations,
        treated_share,
        outcome_rate,
        true_ate_risk_diff,
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=8_000,
        help="Number of observational rows (default: 8000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.n < 500:
        raise ValueError("Use at least 500 observations for stable pipeline splits.")

    df = generate_observational_cohort(args.n, random_state=args.seed)
    write_dataframe(args.output, df)
    logger.info("Wrote %s rows to %s", len(df), args.output.resolve())


if __name__ == "__main__":
    main()

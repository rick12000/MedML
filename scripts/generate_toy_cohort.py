"""Simulate the binary observational cohort used by the example analysis.

Writes ``data/analysis.parquet``. Patients younger than 65 stay in the file
with their covariates, marked excluded, so the run can transport the included
effect onto them. Treatment is binary. The outcome is a binary risk.

Usage (from the repository root)::

    python scripts/generate_toy_cohort.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from causal_pipeline.utils import write_dataframe

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "analysis.parquet"
TOY_COHORT_SIZE = 4000
TOY_COHORT_RANDOM_STATE = 42
MINIMUM_COHORT_SIZE = 10
AGE_INCLUSION_CUTOFF = 65.0
N_SITES = 8
SITE_EFFECT_SCALE = 0.25


def logistic_probability(linear_predictor: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-linear_predictor))


def generate_observational_cohort(
    n_observations: int,
    random_state: int = TOY_COHORT_RANDOM_STATE,
) -> pd.DataFrame:
    """Draw a chronic-care cohort with confounding, effect modification, and age selection."""
    generator = np.random.default_rng(random_state)
    site_outcome = generator.normal(scale=SITE_EFFECT_SCALE, size=N_SITES)
    site_treatment = generator.normal(scale=SITE_EFFECT_SCALE, size=N_SITES)

    age = generator.uniform(45.0, 85.0, n_observations)
    sex = generator.binomial(1, 0.48, n_observations)
    smoker_probability = np.clip(0.18 + 0.004 * (age - 60.0), 0.05, 0.45)
    smoker = (generator.random(n_observations) < smoker_probability).astype(np.int64)
    bmi = np.clip(generator.normal(28.0, 5.0, n_observations), 16.0, 45.0)
    egfr = np.clip(
        generator.normal(78.0 - 0.45 * (age - 65.0), 12.0, n_observations),
        15.0,
        130.0,
    )
    comorbidity_probability = logistic_probability(
        -1.4
        + 0.035 * (age - 65.0)
        + 0.06 * (bmi - 28.0)
        + 0.55 * smoker
        - 0.015 * (egfr - 75.0)
    )
    comorbidity = (generator.random(n_observations) < comorbidity_probability).astype(np.int64)
    baseline_score = np.clip(
        generator.beta(2.2, 4.5, n_observations) * 80.0
        + 12.0 * comorbidity
        + 0.08 * (80.0 - egfr),
        0.0,
        100.0,
    )
    biomarker = generator.normal(0.0, 1.0, n_observations) + 0.35 * (baseline_score / 100.0 - 0.4)
    frailty = generator.normal(
        0.15 * (age - 65.0) / 10.0 + 0.4 * comorbidity,
        0.8,
        n_observations,
    )
    symptom_years = np.clip(generator.gamma(2.0, 2.5, n_observations), 0.0, 30.0)
    site = generator.integers(0, N_SITES, n_observations)
    lab_noise = generator.normal(0.0, 1.0, n_observations)
    included = (age >= AGE_INCLUSION_CUTOFF).astype(np.int64)

    treatment_linear = (
        -0.35
        + site_treatment[site]
        + 0.028 * (age - 65.0)
        + 0.022 * baseline_score
        + 0.55 * comorbidity
        + 0.35 * biomarker
        + 0.25 * sex
        + 0.03 * (bmi - 28.0)
        - 0.012 * (egfr - 75.0)
        + 0.30 * smoker
    )
    propensity = logistic_probability(treatment_linear)
    treatment = (generator.random(n_observations) < propensity).astype(np.int64)

    # Benefit is larger at younger ages, higher baseline severity, and higher frailty.
    effect_log_odds = (
        0.55
        - 0.018 * (age - 65.0)
        + 0.70 * (baseline_score / 100.0 - 0.40)
        + 0.22 * biomarker
        + 0.18 * frailty
        - 0.035 * symptom_years
        + 0.08 * (baseline_score / 100.0) * biomarker
    )
    control_linear = (
        -0.85
        + site_outcome[site]
        + 0.018 * baseline_score
        + 0.45 * comorbidity
        + 0.15 * biomarker
        + 0.010 * (age - 65.0)
        + 0.02 * (bmi - 28.0)
        - 0.008 * (egfr - 75.0)
        + 0.20 * smoker
        + 0.12 * sex
    )
    control_risk = logistic_probability(control_linear)
    treated_risk = logistic_probability(control_linear + effect_log_odds)
    observed_risk = np.where(treatment == 1, treated_risk, control_risk)
    outcome = (generator.random(n_observations) < observed_risk).astype(np.int64)

    cohort = pd.DataFrame(
        {
            "outcome": outcome,
            "treatment": treatment,
            "included": included,
            "age": np.round(age, 1),
            "baseline_score": np.round(baseline_score, 2),
            "biomarker": np.round(biomarker, 4),
            "comorbidity": comorbidity,
            "sex": sex,
            "bmi": np.round(bmi, 2),
            "egfr": np.round(egfr, 2),
            "smoker": smoker,
            "frailty": np.round(frailty, 4),
            "symptom_years": np.round(symptom_years, 2),
            "site": site,
            "lab_noise": np.round(lab_noise, 4),
        }
    )
    risk_difference = treated_risk - control_risk
    logger.info(
        "Simulated n=%s | included=%.3f | P(T=1)=%.3f | P(Y=1)=%.3f | "
        "risk difference included=%.4f excluded=%.4f",
        n_observations,
        float(included.mean()),
        float(treatment.mean()),
        float(outcome.mean()),
        float(np.mean(risk_difference[included == 1])) if np.any(included == 1) else float("nan"),
        float(np.mean(risk_difference[included == 0])) if np.any(included == 0) else float("nan"),
    )
    return cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=TOY_COHORT_SIZE,
        help=f"Number of rows, included and excluded together (default: {TOY_COHORT_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TOY_COHORT_RANDOM_STATE,
        help="Random seed.",
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
    if args.n < MINIMUM_COHORT_SIZE:
        raise ValueError(f"Use at least {MINIMUM_COHORT_SIZE} observations.")
    if args.n > 5000:
        raise ValueError("Keep the toy cohort at 5000 rows or fewer.")
    cohort = generate_observational_cohort(args.n, random_state=args.seed)
    write_dataframe(args.output, cohort)
    logger.info("Wrote %s rows to %s", len(cohort), args.output.resolve())


if __name__ == "__main__":
    main()

# MedML

Modular observational causal inference pipeline built as a thin orchestration layer over **causallib**, **DoubleML**, **EconML**, and **CausalML**.

## Layout

```text
causal_pipeline/
  config.py        # Pydantic configuration models and estimator cloning
  data.py          # CausalDataset, splitting, contrast utilities
  diagnostics.py   # Propensity overlap and covariate balance (SMD)
  ate.py           # ATE adapters and omitted-variable sensitivity
  cate.py          # CATE adapters (meta-learners, forest, neural optional)
  evaluation.py    # Robust proxy scores, calibration, ECETH, TOC, RATE
  policy.py        # Policy trees, virtual twins, MOB (R), held-out policy scoring
  results.py       # ResultStore persistence
  pipeline.py      # CausalPipeline orchestration
tests/               # Unit tests
tests/integration/   # End-to-end pipeline tests
```

## Quick start

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`. From the repository root:

```bash
uv sync
uv run pytest
```

Optional extras: `uv sync --extra neural`, `--extra r`, or `--extra impute`.

`CausalPipeline` takes a `PipelineConfig` and a cohort dataframe. Unit tests check estimator recovery on simulated data with a known treatment effect. Integration tests run the full pipeline and write artifacts to a temporary directory.

## Optional dependencies

- `uv sync --extra neural` — TARNet, CFRNet (CATENets), DragonNet (JAX)
- `uv sync --extra r` — MOB via `partykit` through `rpy2` (requires R)
- `uv sync --extra impute` — MICE imputation via `miceforest`

## Policy workflow

Pass instantiated sklearn estimators (for example `RandomForestRegressor(...)`) on estimator specs. The pipeline clones them before fitting so shared config objects stay unfitted.

ATE and CATE estimator lists are optional, but at least one must be non-empty. Diagnostics, sensitivity, CATE evaluation, and policy are optional. CATE calibration and ranking metrics use cross-fitted predictions on the estimation sample only. With policy off, every estimator is cross-fit on the full sample. Pass `policy=[PolicyTreeMethodSpec(...), ...]` with `test_fraction` strictly between 0 and 1 to hold out a test set: rules are trained on out-of-fold learning-sample rewards and scored only on that test set. Omit `policy` or pass an empty list to skip policy fitting.

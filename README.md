# MedML

Modular observational causal inference pipeline built as a thin orchestration layer over **causallib**, **DoubleML**, **EconML**, and **CausalML**.

## Layout

```text
causal_pipeline/
  pipeline.py          # CausalPipeline orchestration
  diagnostics.py       # propensity overlap and covariate balance
  subgroup.py          # pre-specified subgroup splits
  core/                # config, datasets, learners, scaling, runtime, artifacts
  nuisance/            # cross-fitting, tuning, nuisance metrics and screens
  estimation/          # ATE, CATE, evaluation, selection
  policies/            # policy trees, virtual twins, held-out policy value
  missing_data/        # imputation and Rubin's rules pooling
  plotting/            # publication figures for the estimates above
tests/                 # unit tests, grouped by the same scopes
tests/integration/     # end-to-end pipeline tests
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

Learner fields on estimator specs take an approved architecture name: `"linear"`, `"random_forest"`, or `"xgboost"` (default). Pass a name, not an estimator object. The pipeline builds the estimator from the name, picks classifier or regressor from the outcome type, and keeps it single-threaded so parallelism stays at the process-pool level.

ATE and CATE estimator lists are optional, but at least one must be non-empty. Diagnostics, sensitivity, CATE evaluation, and policy are optional. CATE calibration and ranking metrics use cross-fitted predictions on the estimation sample only. With policy off, every estimator is cross-fit on the full sample. Pass `policy=[PolicyTreeMethodSpec(...), ...]` with `test_fraction` strictly between 0 and 1 to hold out a test set: rules are trained on out-of-fold learning-sample rewards and scored only on that test set. Omit `policy` or pass an empty list to skip policy fitting.

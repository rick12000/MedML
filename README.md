# MedML

Modular observational causal inference pipeline built as a thin orchestration layer over **causallib**, **DoubleML**, **EconML**, and **CausalML**.

## Layout

```text
causal_pipeline/
  config.py        # Pydantic configuration models and estimator cloning
  settings.py      # Project constants and assembled PipelineConfig
  data.py          # CausalDataset, splitting, contrast utilities
  diagnostics.py   # Propensity overlap and covariate balance (SMD)
  ate.py           # ATE adapters and omitted-variable sensitivity
  cate.py          # CATE adapters (meta-learners, forest, neural optional)
  evaluation.py    # Robust proxy scores, calibration, ECETH, TOC, RATE
  policy.py        # Policy trees, virtual twins, MOB (R), test evaluation
  results.py       # ResultStore persistence
  pipeline.py      # CausalPipeline orchestration
run_analysis.py    # Example entrypoint
tests/
```

## Quick start

```bash
pip install -e ".[dev]"
pytest tests
```

Place a prepared cohort at `data/analysis.parquet`, then:

```bash
python run_analysis.py
```

Outputs are written under `results/` (partitions, diagnostics, ATE/CATE summaries, optional policy artifacts).

## Optional dependencies

- `pip install -e ".[neural]"` — TARNet, CFRNet (CATENets), DragonNet (JAX)
- `pip install -e ".[r]"` — MOB via `partykit` through `rpy2` (requires R)

## Policy workflow

Pass instantiated sklearn estimators (for example `RandomForestRegressor(...)`) on estimator specs. The pipeline clones them before fitting so shared config objects stay unfitted.

ATE and CATE estimator lists are optional, but at least one must be non-empty. Diagnostics, sensitivity, CATE evaluation, and policy are optional. Pass `policy=[PolicyTreeMethodSpec(...), ...]` with `test_fraction > 0`; omit `policy` or pass an empty list to skip policy fitting. Policy methods require a held-out test split.

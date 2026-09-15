# MedML

Modular observational causal inference pipeline built as a thin orchestration layer over **causallib**, **DoubleML**, **EconML**, and **CausalML**.

## Layout

```text
causal_pipeline/
  config.py        # Pydantic configuration models and learner registry
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

Set `policy=PolicyConfig(...)` on `PipelineConfig` and use `train_fraction + validation_fraction + test_fraction = 1` with `test_fraction > 0`. Omit `policy` (or set it to `None`) to skip policy fitting; the test partition is then an empty `DataFrame` with the input schema. Omit `sensitivity` to skip ATE sensitivity analysis.

---
name: benchmark-variant-optimizer
description: >
  Use when a user asks to optimize code against a clear metric such as runtime, latency, memory, accuracy, precision, recall, cost, throughput, or quality score. Handles benchmark-driven comparison of implementation variants and post-review integration of the selected variant. Do NOT use when the optimization metric is missing or unclear.
---

# Benchmark Variant Optimizer

## Workflow

1. Identify the target optimization metric. If the user did not provide a clear metric and it cannot be determined from the prompt, stop using this skill.

2. Identify optional secondary metrics. Track them only if the user provides them or they add useful context. Treat them as adjunct metrics or trade-off metrics, e.g. accuracy vs runtime, memory vs throughput, precision vs recall.

3. Analyze the codebase in context. Understand the target function, method, object, or workflow; its inputs and outputs; core data structures; module interactions; and current tests.

4. Define candidate variants:
   - `baseline`: current codebase behavior
   - additional variants: plausible alternatives based on the prompt, codebase structure, the optimization metric, and any secondary metrics if present

5. Inspect the target data. Identify its shape, schema, domain, variability, and edge cases. Build a toy data fixture that is small enough to run quickly but realistic enough to preserve the main variability of the production data.

6. Identify optional breakout dimensions. Use them only when they add insight into how metrics change across contexts, e.g. number of observations in the toy data fixture.

7. Choose the benchmark integration strategy:
   - Determine whether the target can be benchmarked in isolation or whether part of the surrounding workflow must be exercised.
   - Prefer importing and reusing existing code paths from the codebase rather than copying implementation code.
   - If exercising the real workflow is important, add temporary variant-selection logic to the codebase so the benchmark can evaluate multiple implementations through the same entry points.
   - Otherwise, define candidate variants directly in the benchmark script and compare them against the baseline implementation.
   - Replicate only the minimum surrounding workflow necessary to reach the target code under realistic conditions.
   - Mock external APIs, network calls, and third-party services with representative outputs matching real data shapes and behavior.

8. Write a benchmark script. It must run all variants against the toy fixture, across breakout dimensions if any, and report the target metric plus secondary metrics if any.

9. Reduce measurement noise. Use repeated measurements and robust aggregation. Use direct averages when metrics are comparable. Use relative scores or average ranks when results span different domains, datasets, or scales.

10. Save outputs under `<repo-root>/cache/`. Create the folder if missing. Save full tabular results as CSV. If Matplotlib is available, also save visual summaries.

11. Report to the user before changing the main implementation:
    - target metric
    - secondary metrics, if any
    - variants tested
    - most salient table snippet
    - interpretation of full tabular results
    - result artifact paths
    - recommended variant
    - ask which variant the user wants implemented

12. After the user confirms a variant, implement only that variant in the main codebase. Remove temporary variant flags, benchmark-only branches, unused candidate code, and the benchmark script. Preserve result artifacts unless the user asks to delete them.

13. Run tests, linting, and type checks. If any fail, fix them before reporting completion.

## Final report format

1. `Metric optimized:`
2. `Secondary metrics:`
3. `Variants tested:`
4. `Results summary:`
5. `Artifacts:`
6. `Recommendation:`
7. `Please confirm which variant to implement.`
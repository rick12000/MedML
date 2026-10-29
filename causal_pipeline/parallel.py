"""Process pool for independent pipeline fits.

One worker runs one CATE fold, one whole in-process learner, one ATE fit,
one imputation draw, or one sensitivity refit. Nested pools are avoided by
giving imputation draws the cores only when there are at least as many draws
as cores, and by leaving each worker on a single thread.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import TypeVar

from joblib.externals.loky import get_reusable_executor

from causal_pipeline.logging_config import configure_package_runtime

logger = logging.getLogger(__name__)

MINIMUM_PARALLEL_TASKS: int = 2
ALL_CORES_N_JOBS: int = -1
SINGLE_CORE_N_JOBS: int = 1
WORKER_POOL_TIMEOUT_SECONDS: int = 3_600
WORKER_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

Job = TypeVar("Job")
Result = TypeVar("Result")


def available_cores() -> int:
    return os.cpu_count() or 1


def pool_workers(parallel: bool, n_tasks: int) -> int:
    if not parallel or n_tasks < MINIMUM_PARALLEL_TASKS:
        return 1
    cores = available_cores()
    if cores < MINIMUM_PARALLEL_TASKS:
        return 1
    return min(n_tasks, cores)


def worker_library_jobs(parallel: bool, n_tasks: int) -> int:
    """Core count a library may use inside one task.

    A shared pool already occupies one process per task, so the library stays
    on one core. A single task keeps the library default of every core.
    """
    if pool_workers(parallel=parallel, n_tasks=n_tasks) > 1:
        return SINGLE_CORE_N_JOBS
    return ALL_CORES_N_JOBS


def imputation_draws_are_parallel(parallel: bool, n_draws: int) -> bool:
    """Parallelize draws only when they cover the machine.

    Fewer draws than cores leave the sample-level estimator map on the parent,
    where one draw can use every core. At least as many draws as cores fills
    the machine with draws, and each draw then runs its own fits serially.
    """
    if pool_workers(parallel=parallel, n_tasks=n_draws) == 1:
        return False
    return n_draws >= available_cores()


def run_map(function: Callable[[Job], Result], items: Sequence[Job], parallel: bool) -> list[Result]:
    configure_package_runtime()

    def run_item(item: Job) -> Result:
        configure_package_runtime()
        return function(item)

    workers = pool_workers(parallel=parallel, n_tasks=len(items))
    if workers == 1:
        return [run_item(item) for item in items]
    logger.info("Running %s independent fits on %s workers...", len(items), workers)
    executor = get_reusable_executor(
        max_workers=available_cores(),
        timeout=WORKER_POOL_TIMEOUT_SECONDS,
        env=WORKER_THREAD_ENV,
    )
    results = list(executor.map(run_item, items))
    logger.info("...Finished %s independent fits on %s workers", len(items), workers)
    return results


def load_parallel_worker(worker_index: int) -> int:
    """Import the fit stack so a later timed pool does not pay that cost."""
    import causal_pipeline.pipeline  # noqa: F401
    import xgboost  # noqa: F401

    return worker_index


def warm_process_pool() -> None:
    cores = available_cores()
    if cores < MINIMUM_PARALLEL_TASKS:
        return
    run_map(load_parallel_worker, list(range(cores)), parallel=True)

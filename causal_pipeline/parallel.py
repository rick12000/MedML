"""Process pool for independent pipeline fits.

One worker runs one CATE fold, one whole in-process learner, one ATE fit,
one imputation draw, or one sensitivity refit.

Core budget: one logical core is always left idle, so the machine keeps a
buffer. Every library call inside a worker is pinned to one thread, both by
environment variables (BLAS, OpenMP) and by explicit library arguments
(XGBoost, econml/grf forests). A single task in the parent may use the whole
budget. Nested process pools are avoided: imputation draws get the pool only
when there are at least as many draws as budgeted cores.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

from joblib.externals.loky import get_reusable_executor

from causal_pipeline.logging_config import configure_package_runtime

logger = logging.getLogger(__name__)

MINIMUM_PARALLEL_TASKS: int = 2
SINGLE_CORE_N_JOBS: int = 1
WORKER_POOL_TIMEOUT_SECONDS: int = 3_600
BACKGROUND_NICE_INCREMENT: int = 10
WINDOWS_BELOW_NORMAL_PRIORITY: int = 0x00004000
WINDOWS_NORMAL_PRIORITY: int = 0x00000020
WORKER_MARKER_ENV: str = "CAUSAL_PIPELINE_PROCESS_WORKER"
WORKER_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    WORKER_MARKER_ENV: "1",
}

Job = TypeVar("Job")
Result = TypeVar("Result")


def in_worker_process() -> bool:
    """True inside a process started by the pool, where libraries must stay single-threaded."""
    return os.environ.get(WORKER_MARKER_ENV) == "1"


def _windows_kernel32():  # type: ignore[no-untyped-def]
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint32
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    return kernel32


def lower_process_priority() -> None:
    """Let interactive work preempt this process. Used for pool workers, and it is not undone.

    Windows lowers the priority class. POSIX raises the nice value, which only root can reverse.
    """
    if sys.platform == "win32":
        kernel32 = _windows_kernel32()
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), WINDOWS_BELOW_NORMAL_PRIORITY)
        return
    os.nice(BACKGROUND_NICE_INCREMENT)


@contextmanager
def background_priority() -> Iterator[None]:
    """Run a block at lower CPU priority and restore the previous class afterwards.

    On POSIX the block runs unchanged. The nice value cannot be lowered back without privileges.
    """
    if sys.platform != "win32":
        yield
        return
    kernel32 = _windows_kernel32()
    handle = kernel32.GetCurrentProcess()
    previous = kernel32.GetPriorityClass(handle) or WINDOWS_NORMAL_PRIORITY
    kernel32.SetPriorityClass(handle, WINDOWS_BELOW_NORMAL_PRIORITY)
    try:
        yield
    finally:
        kernel32.SetPriorityClass(handle, previous)


def available_cores() -> int:
    """Cores the pipeline may use: every logical core except one, never fewer than one."""
    return max(1, (os.cpu_count() or 1) - 1)


def pool_workers(parallel: bool, n_tasks: int) -> int:
    if not parallel or n_tasks < MINIMUM_PARALLEL_TASKS or in_worker_process():
        return 1
    cores = available_cores()
    if cores < MINIMUM_PARALLEL_TASKS:
        return 1
    return min(n_tasks, cores)


def worker_library_jobs(parallel: bool, n_tasks: int) -> int:
    """Thread count a library may use inside one task.

    A shared pool occupies one process per task, and so does any worker, so
    the library stays on one thread there. A single task in the parent may use
    the whole budget.
    """
    if in_worker_process() or pool_workers(parallel=parallel, n_tasks=n_tasks) > 1:
        return SINGLE_CORE_N_JOBS
    return available_cores()


def imputation_draws_are_parallel(parallel: bool, n_draws: int) -> bool:
    """Parallelize draws only when they cover the core budget.

    Fewer draws than cores leave the sample-level estimator map on the parent,
    where one draw can use every budgeted core. At least as many draws as cores
    fills the budget with draws, and each draw then runs its own fits serially.
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
        initializer=lower_process_priority,
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

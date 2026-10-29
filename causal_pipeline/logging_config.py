"""Logging format and warning filters for causal_pipeline execution."""

from __future__ import annotations

import logging
import warnings

from sklearn.exceptions import ConvergenceWarning

configured = False
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure_package_runtime() -> None:
    """Apply once: quiet known nuisance warnings and a default console handler."""
    global configured
    if configured:
        return
    configured = True
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=r".*[Ll]bfgs failed to converge.*")
    warnings.filterwarnings("ignore", message=r".*[Mm]aximum number of iterations.*")
    warnings.filterwarnings("ignore", message=r".*failed to converge.*")
    warnings.filterwarnings("ignore", category=Warning, module=r"statsmodels\..*")
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    logging.getLogger("optuna").setLevel(logging.WARNING)

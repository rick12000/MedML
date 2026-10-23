"""Paths for pytest artifacts under ``cache/pytest/`` at the repository root."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_CACHE_ROOT = REPO_ROOT / "cache" / "pytest"


def sanitize_cache_segment(name: str) -> str:
    return re.sub(r"[^\w.\-\[\]]+", "_", name).strip("_")


def module_cache_key(module_file: Path) -> str:
    relative = module_file.resolve().relative_to(REPO_ROOT / "tests")
    return "/".join(relative.with_suffix("").parts)


def artifact_path(output_dir: Path, *relative_parts: str) -> Path:
    destination = output_dir.joinpath(*relative_parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination

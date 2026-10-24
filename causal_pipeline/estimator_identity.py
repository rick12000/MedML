"""Stable estimator ids and configuration columns for pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from causal_pipeline.config import ATEEstimatorSpec, CATEEstimatorSpec

EstimatorSpec = ATEEstimatorSpec | CATEEstimatorSpec
SpecT = TypeVar("SpecT", bound=BaseModel)

logger = logging.getLogger(__name__)

LEARNER_FIELD_NAMES = frozenset(
    {
        "propensity_learner",
        "outcome_learner",
        "effect_learner",
    }
)


def _serialize_config_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, BaseEstimator):
        return type(value).__name__
    return value


def estimator_config_columns(spec: EstimatorSpec) -> dict[str, object]:
    """Flat configuration fields for summary and per-estimator result tables."""
    columns: dict[str, object] = {"estimator_kind": spec.kind.value}
    for name in type(spec).model_fields:
        if name == "kind":
            continue
        value = getattr(spec, name)
        if name in LEARNER_FIELD_NAMES:
            columns[name] = type(value).__name__ if value is not None else None
            continue
        if name == "clip_bounds":
            columns["propensity_clip_lower"] = value[0] if value is not None else None
            columns["propensity_clip_upper"] = value[1] if value is not None else None
            continue
        columns[name] = _serialize_config_value(value)
    return columns


def overlap_slug_part(spec: EstimatorSpec) -> str:
    clip = getattr(spec, "clip_bounds", None)
    if clip is not None:
        return f"clip_{clip[0]:g}_{clip[1]:g}"
    return "unclipped"


def normalize_estimator_alias(alias: str) -> str:
    """Filesystem-safe alias for directories and plot labels."""
    text = alias.strip()
    if not text:
        raise ValueError("alias must be non-empty when provided.")
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)


def resolve_estimator_id(spec: EstimatorSpec) -> str:
    alias = getattr(spec, "alias", None)
    if alias is not None:
        return normalize_estimator_alias(alias)
    return build_estimator_slug(spec)


def build_estimator_slug(spec: EstimatorSpec) -> str:
    """Short filesystem-safe id; full configuration lives in summary table columns."""
    config = estimator_config_columns(spec)
    payload = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    overlap = overlap_slug_part(spec).replace(".", "p")
    overlap = re.sub(r"[^a-zA-Z0-9_]+", "_", overlap)
    return f"{spec.kind.value}__{overlap}__{digest}"


def assign_estimator_ids(specs: Sequence[SpecT]) -> list[tuple[str, SpecT]]:
    """Return unique (estimator_id, spec) pairs for a configured estimator list."""
    alias_counts = Counter(
        normalize_estimator_alias(spec.alias)
        for spec in specs
        if getattr(spec, "alias", None) is not None
    )
    for alias, count in alias_counts.items():
        if count > 1:
            logger.warning(
                "Estimator alias '%s' appears %s times; later runs use suffixed ids.",
                alias,
                count,
            )

    assigned: list[tuple[str, SpecT]] = []
    used: set[str] = set()
    for spec in specs:
        base = resolve_estimator_id(spec)
        estimator_id = base
        suffix = 2
        while estimator_id in used:
            estimator_id = f"{base}__{suffix}"
            suffix += 1
        used.add(estimator_id)
        assigned.append((estimator_id, spec))
    return assigned


def attach_estimator_metadata(
    frame: pd.DataFrame,
    estimator_id: str,
    spec: EstimatorSpec,
) -> pd.DataFrame:
    """Add estimator id and one column per configured spec parameter."""
    annotated = frame.copy()
    metadata = estimator_config_columns(spec)
    metadata["estimator"] = estimator_id
    for key, value in metadata.items():
        annotated[key] = value
    return annotated

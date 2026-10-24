from __future__ import annotations

import pytest

from causal_pipeline.config import ATEKind, IPWATEEstimatorSpec
from causal_pipeline.estimator_identity import (
    assign_estimator_ids,
    build_estimator_slug,
    estimator_config_columns,
    normalize_estimator_alias,
    resolve_estimator_id,
)


def test_distinct_ipw_specs_get_distinct_ids() -> None:
    clipped = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=(0.02, 0.98))
    unclipped = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=None)
    wider = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=(0.05, 0.95))

    ids = [estimator_id for estimator_id, _ in assign_estimator_ids([clipped, unclipped, wider])]
    assert len(set(ids)) == 3


def test_normalize_estimator_alias_sanitizes_whitespace() -> None:
    assert normalize_estimator_alias("  my model  ") == "my_model"


def test_duplicate_specs_receive_numeric_suffix() -> None:
    spec = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=None)
    ids = [estimator_id for estimator_id, _ in assign_estimator_ids([spec, spec])]
    assert ids[0] != ids[1]
    assert ids[1].endswith("__2")


def test_config_columns_surface_clip_bounds() -> None:
    clipped = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=(0.02, 0.98))
    columns = estimator_config_columns(clipped)
    assert columns["estimator_kind"] == ATEKind.IPW.value
    assert columns["propensity_clip_lower"] == 0.02
    assert columns["propensity_clip_upper"] == 0.98
    assert "clip_bounds" not in columns

    unclipped = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=None)
    assert estimator_config_columns(unclipped)["propensity_clip_lower"] is None


def test_alias_becomes_estimator_id() -> None:
    spec = IPWATEEstimatorSpec(kind=ATEKind.IPW, alias="ipw_clipped")
    assert resolve_estimator_id(spec) == "ipw_clipped"
    estimator_id, _ = assign_estimator_ids([spec])[0]
    assert estimator_id == "ipw_clipped"


def test_duplicate_alias_warns_and_suffixes(caplog) -> None:
    import logging

    first = IPWATEEstimatorSpec(kind=ATEKind.IPW, alias="same_name")
    second = IPWATEEstimatorSpec(kind=ATEKind.IPW, alias="same_name", clip_bounds=None)
    with caplog.at_level(logging.WARNING):
        ids = [estimator_id for estimator_id, _ in assign_estimator_ids([first, second])]
    assert ids[0] == "same_name"
    assert ids[1] == "same_name__2"
    assert any("same_name" in record.message for record in caplog.records)


def test_build_estimator_slug_includes_overlap_tokens() -> None:
    clipped = IPWATEEstimatorSpec(kind=ATEKind.IPW, clip_bounds=(0.02, 0.98))
    slug = build_estimator_slug(clipped)
    assert slug.startswith("ipw__clip_0p02_0p98__")
    assert len(slug) < 80

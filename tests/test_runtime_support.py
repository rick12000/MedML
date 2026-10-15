from __future__ import annotations

import importlib.util

import pytest

from causal_pipeline.cate import ensure_cate_spec_supported
from causal_pipeline.config import (
    CATEKind,
    DataConfig,
    DragonNetCATEEstimatorSpec,
    PipelineConfig,
    SplitConfig,
)
from causal_pipeline.pipeline import CausalPipeline, ensure_pipeline_runtime_support


def test_ensure_cate_spec_supported_reports_missing_neural_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str) -> object | None:
        if name == "jax":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    spec = DragonNetCATEEstimatorSpec(kind=CATEKind.DRAGONNET)
    with pytest.raises(ImportError, match=r'medml\[neural\]'):
        ensure_cate_spec_supported(spec=spec)


def test_pipeline_init_fails_before_run_when_dragonnet_unavailable(
    binary_data_config: DataConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causal_pipeline.cate._ensure_modules",
        lambda _names, _label: (_ for _ in ()).throw(
            ImportError('Install with: pip install "medml[neural]"')
        ),
    )
    config = PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(test_fraction=0.0, random_state=0),
        cate_estimators=[DragonNetCATEEstimatorSpec(kind=CATEKind.DRAGONNET)],
    )
    with pytest.raises(ImportError, match=r'medml\[neural\]'):
        CausalPipeline(config=config)


def test_ensure_pipeline_runtime_support_checks_cate_specs(
    binary_data_config: DataConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CATEKind] = []

    def record(spec: DragonNetCATEEstimatorSpec) -> None:
        calls.append(spec.kind)

    monkeypatch.setattr("causal_pipeline.pipeline.ensure_cate_spec_supported", record)
    config = PipelineConfig(
        data=binary_data_config,
        split=SplitConfig(test_fraction=0.0, random_state=0),
        cate_estimators=[DragonNetCATEEstimatorSpec(kind=CATEKind.DRAGONNET)],
    )
    ensure_pipeline_runtime_support(config=config)
    assert calls == [CATEKind.DRAGONNET]

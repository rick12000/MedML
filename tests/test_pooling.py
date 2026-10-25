"""Rubin's rules against the mice::pool.scalar numeric example."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from causal_pipeline.pooling import (
    average_aligned_frames,
    pool_contrast_evaluation,
    pool_keyed_estimates,
    pool_numeric_mean,
    pool_scalar,
    pool_shap,
    wald_variance,
)


def test_pool_scalar_matches_mice_pool_scalar() -> None:
    pooled = pool_scalar(
        estimates=np.array([-1.5457, -1.428]),
        variances=np.array([0.9723**2, 1.041**2]),
        sample_size=25,
        level=0.95,
        parameter_count=2,
    )
    assert pooled.imputations == 2
    assert pooled.estimate == pytest.approx(-1.48685, abs=1e-6)
    assert pooled.within_variance == pytest.approx(1.014524145, abs=1e-8)
    assert pooled.between_variance == pytest.approx(0.006926645, abs=1e-8)
    assert pooled.total_variance == pytest.approx(1.0249141125, abs=1e-8)
    assert pooled.degrees_of_freedom == pytest.approx(20.970254838, abs=1e-6)
    assert pooled.relative_increase == pytest.approx(0.0102412225, abs=1e-8)
    assert pooled.fraction_missing_information == pytest.approx(0.0927283145, abs=1e-6)


def test_zero_between_variance_uses_the_within_variance() -> None:
    pooled = pool_scalar(
        estimates=np.array([1.5, 1.5]),
        variances=np.array([0.25, 0.25]),
        sample_size=100,
        level=0.95,
    )
    assert pooled.between_variance == 0.0
    assert pooled.total_variance == pytest.approx(0.25)
    assert pooled.relative_increase == 0.0
    assert pooled.fraction_missing_information == 0.0
    assert pooled.degrees_of_freedom == float("inf")
    half_width = float(norm.ppf(0.975)) * 0.5
    assert pooled.ci_lower == pytest.approx(1.5 - half_width)
    assert pooled.ci_upper == pytest.approx(1.5 + half_width)


def test_zero_within_variance_uses_rubin_degrees_of_freedom() -> None:
    pooled = pool_scalar(
        estimates=np.array([0.0, 2.0]),
        variances=np.array([0.0, 0.0]),
        sample_size=80,
        level=0.95,
    )
    assert pooled.degrees_of_freedom == 1.0
    assert pooled.relative_increase == float("inf")
    assert pooled.fraction_missing_information == 1.0
    assert pooled.total_variance == pytest.approx(3.0)
    half_width = float(t.ppf(0.975, 1)) * np.sqrt(3.0)
    assert pooled.ci_lower == pytest.approx(1.0 - half_width)


def test_wald_variance_rejects_an_asymmetric_interval() -> None:
    with pytest.raises(ValueError, match="not symmetric"):
        wald_variance(estimate=1.0, lower=0.0, upper=3.0, level=0.95)


def test_pooled_estimate_table_combines_symmetric_intervals() -> None:
    quantile = float(norm.ppf(0.975))
    first = pd.DataFrame(
        {
            "contrast": ["1_vs_0"],
            "estimand": ["ate"],
            "estimate": [0.0],
            "ci_lower": [-quantile],
            "ci_upper": [quantile],
        }
    )
    second = pd.DataFrame(
        {
            "contrast": ["1_vs_0"],
            "estimand": ["ate"],
            "estimate": [2.0],
            "ci_lower": [2.0 - quantile],
            "ci_upper": [2.0 + quantile],
        }
    )
    pooled = pool_keyed_estimates(
        frames=[first, second],
        keys=["contrast"],
        level=0.95,
        sample_size=80,
    )
    assert len(pooled) == 1
    assert pooled.loc[0, "estimate"] == pytest.approx(1.0)
    assert pooled.loc[0, "within_variance"] == pytest.approx(1.0)
    assert pooled.loc[0, "total_variance"] == pytest.approx(4.0)
    assert list(pooled.columns).count("contrast") == 1


def test_published_standard_error_overrides_an_asymmetric_interval() -> None:
    first = pd.DataFrame(
        {
            "contrast": ["1_vs_0"],
            "estimate": [0.0],
            "standard_error": [1.0],
            "ci_lower": [-5.0],
            "ci_upper": [1.0],
        }
    )
    second = pd.DataFrame(
        {
            "contrast": ["1_vs_0"],
            "estimate": [2.0],
            "standard_error": [1.0],
            "ci_lower": [0.0],
            "ci_upper": [4.0],
        }
    )
    pooled = pool_keyed_estimates(
        frames=[first, second],
        keys=["contrast"],
        level=0.95,
        sample_size=80,
    )
    assert pooled.loc[0, "within_variance"] == pytest.approx(1.0)
    assert pooled.loc[0, "total_variance"] == pytest.approx(4.0)
    assert "standard_error" not in pooled.columns


def test_averaged_frames_keep_the_shared_index() -> None:
    first = pd.DataFrame({"effect": [1.0, 3.0]}, index=pd.Index([4, 9], name="row"))
    second = pd.DataFrame({"effect": [3.0, 5.0]}, index=pd.Index([4, 9], name="row"))
    mean, between = average_aligned_frames([first, second])
    assert mean.index.equals(first.index)
    assert mean["effect"].tolist() == [2.0, 4.0]
    assert between["effect"].tolist() == pytest.approx([np.sqrt(2.0), np.sqrt(2.0)])


def test_pooled_eceth_uses_barnard_rubin_and_rate_intervals_stay_missing(contrast_evaluation) -> None:
    sample_size = 80
    evaluations = [
        contrast_evaluation(eceth=-2.0, rate_autoc=0.0),
        contrast_evaluation(eceth=0.0, rate_autoc=2.0),
    ]
    pooled = pool_contrast_evaluation(
        evaluations=evaluations,
        sample_size=sample_size,
        level=0.95,
        tolerance=0.01,
    )
    scalar = pool_scalar(
        estimates=np.array([-2.0, 0.0]),
        variances=np.array([1.0, 1.0]),
        sample_size=sample_size,
        level=0.95,
    )
    statistic = (scalar.estimate - 0.01) / np.sqrt(scalar.total_variance)
    assert pooled.eceth_pvalue == pytest.approx(float(t.cdf(statistic, scalar.degrees_of_freedom)))
    assert pooled.eceth_pvalue > float(norm.cdf(statistic))
    assert pooled.rate_autoc == pytest.approx(1.0)
    assert np.isnan(pooled.rate_autoc_standard_error)
    assert np.isnan(pooled.rate_autoc_pvalue)
    assert np.isnan(pooled.rate_qini_pvalue)


def test_mismatched_shap_tables_are_left_unpooled(caplog: pytest.LogCaptureFixture) -> None:
    first = pd.DataFrame({"row": [0], "value": [1.0]})
    second = pd.DataFrame({"row": [0, 1], "value": [1.0, 2.0]})
    with caplog.at_level("WARNING"):
        pooled = pool_shap([first, second])
    assert pooled is None
    assert "left unpooled" in caplog.text


def test_numeric_tables_without_a_key_are_averaged() -> None:
    first = pd.DataFrame({"value": [1.0, 3.0]})
    second = pd.DataFrame({"value": [3.0, 5.0]})
    pooled = pool_numeric_mean([first, second])
    assert pooled is not None
    assert pooled.shape == first.shape
    assert pooled["value"].tolist() == [2.0, 4.0]

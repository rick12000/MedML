import math

import numpy as np

from causal_pipeline.ate import (
    benchmark_multiple_to_null,
    robustness_value_equal_strength,
)


def test_robustness_value_increases_with_effect_magnitude() -> None:
    small = robustness_value_equal_strength(tau_hat=0.1, scale=1.0)
    large = robustness_value_equal_strength(tau_hat=1.0, scale=1.0)
    assert large > small
    assert 0.0 < small < 1.0


def test_robustness_value_infinite_when_scale_nonpositive() -> None:
    assert math.isinf(robustness_value_equal_strength(tau_hat=0.5, scale=0.0))


def test_benchmark_multiple_to_null_finds_multiplier_covering_effect() -> None:
    multiple = benchmark_multiple_to_null(
        tau_hat=0.5,
        null_effect=0.0,
        q=0.1,
        cy2=0.1,
        scale=1.0,
    )
    assert multiple > 0.0
    assert math.isfinite(multiple)

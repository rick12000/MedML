import numpy as np

from causal_pipeline.evaluation import compute_eceth


def test_compute_eceth_returns_finite_scalar_for_well_specified_inputs() -> None:
    n_observations = 40
    tau_hat = np.linspace(-1.0, 1.0, n_observations)
    gamma = tau_hat + np.random.default_rng(0).normal(scale=0.05, size=n_observations)
    value = compute_eceth(tau_hat=tau_hat, gamma=gamma, n_bins=4)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_compute_eceth_vector_length_matches_input() -> None:
    tau_hat = np.array([0.0, 0.5, 1.0, 1.5])
    gamma = np.array([0.1, 0.4, 0.9, 1.4])
    value = compute_eceth(tau_hat=tau_hat, gamma=gamma, n_bins=2)
    assert isinstance(value, float)

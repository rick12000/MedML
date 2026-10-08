from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causal_pipeline.scaling import CovariateScaler, clone_scaled_estimator

TRAIN_ROWS = 40


def test_scaler_standardizes_continuous_columns_from_training_rows_only() -> None:
    index = np.arange(TRAIN_ROWS)
    train = pd.DataFrame(
        {
            "binary": index % 2,
            "site": index % 20,
            "continuous": np.linspace(10.0, 40.0, TRAIN_ROWS),
            "two_point": (index % 2) * 1000.0,
            "coded": (index % 2) + 1,
            "category": index % 3,
            "wide_integer": index % 21,
            "age_band": 60 + (index % 16),
            "near_integer": 100000.2 + (index % 3) * 0.3,
        }
    )
    held_out = pd.DataFrame(
        {
            "binary": [1, 0],
            "site": [19, 0],
            "continuous": [100.0, 0.0],
            "two_point": [1000.0, 0.0],
            "coded": [2, 1],
            "category": [2, 0],
            "wide_integer": [21, 0],
            "age_band": [75, 60],
            "near_integer": [100000.2, 100000.8],
        }
    )
    untouched = train.copy()
    scaler = CovariateScaler()
    transformed_train = scaler.fit_transform(train)
    transformed_held_out = scaler.transform(held_out)
    assert train.equals(untouched)

    assert transformed_train.shape == train.shape
    assert list(transformed_train.columns) == list(train.columns)
    assert np.allclose(transformed_train["binary"], train["binary"])
    assert np.allclose(transformed_train["site"], train["site"])
    assert np.allclose(transformed_train["coded"], train["coded"])
    assert np.allclose(transformed_train["category"], train["category"])
    assert np.isclose(transformed_train["two_point"].mean(), 0.0)
    assert np.isclose(transformed_train["two_point"].std(ddof=0), 1.0)
    assert np.isclose(transformed_train["continuous"].mean(), 0.0)
    assert np.isclose(transformed_train["continuous"].std(ddof=0), 1.0)
    assert np.isclose(transformed_train["wide_integer"].mean(), 0.0)
    assert np.isclose(transformed_train["wide_integer"].std(ddof=0), 1.0)
    assert np.isclose(transformed_train["age_band"].mean(), 0.0)
    assert np.isclose(transformed_train["age_band"].std(ddof=0), 1.0)
    assert np.isclose(transformed_train["near_integer"].mean(), 0.0)
    assert np.isclose(transformed_train["near_integer"].std(ddof=0), 1.0)

    continuous = train["continuous"].to_numpy(dtype=float)
    center = float(continuous.mean())
    scale = float(continuous.std(ddof=0))
    expected = (held_out["continuous"].to_numpy(dtype=float) - center) / scale
    assert np.allclose(transformed_held_out["continuous"], expected)
    assert np.allclose(transformed_held_out["binary"], held_out["binary"])
    assert np.allclose(transformed_held_out["site"], held_out["site"])
    assert np.allclose(transformed_held_out["coded"], held_out["coded"])
    assert np.allclose(transformed_held_out["category"], held_out["category"])
    two_point = train["two_point"].to_numpy(dtype=float)
    two_point_center = float(two_point.mean())
    two_point_scale = float(two_point.std(ddof=0))
    expected_two_point = (held_out["two_point"].to_numpy(dtype=float) - two_point_center) / two_point_scale
    assert np.allclose(transformed_held_out["two_point"], expected_two_point)

    transformed_array = CovariateScaler().fit(train.to_numpy()).transform(train.to_numpy())
    assert isinstance(transformed_array, np.ndarray)
    assert transformed_array.shape == train.shape


def test_scaler_moments_follow_row_weights() -> None:
    features = pd.DataFrame({"x": [0.0, 10.0, 1000.0]})
    scaler = CovariateScaler()
    scaler.fit(features, sample_weight=np.array([1.0, 1.0, 0.0]))
    transformed = scaler.transform(features)
    assert transformed.shape == features.shape
    assert np.allclose(transformed["x"], [-1.0, 1.0, 199.0])


def test_scaled_learner_returns_one_probability_row_per_input() -> None:
    features = pd.DataFrame(
        {
            "binary": [0, 1, 0, 1, 0, 1, 0, 1],
            "continuous": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    model = clone_scaled_estimator(LogisticRegression(max_iter=200))
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(features), 2)

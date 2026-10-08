"""Z-score continuous covariates from the rows a learner is fit on."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

# Indicators are the codings {0, 1} and {-1, 1}. A category is a short integer
# code that starts at 0 or 1, such as site 0..19. Integer age 60..75 is a
# measurement: its values sit far above the code labels, so it is standardized.
INDICATOR_CODES = ((0.0, 1.0), (-1.0, 1.0))
CODE_ORIGINS = (0.0, 1.0)
CATEGORICAL_MAX_LEVELS = 20
MINIMUM_DISTINCT_VALUES = 2
INTEGER_TOLERANCE = 1e-8


class CovariateScaler(BaseEstimator, TransformerMixin):
    """Standardize continuous columns. Leave binary and categorical columns unchanged."""

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        y: object = None,
        sample_weight: np.ndarray | None = None,
    ) -> CovariateScaler:
        # Labels stay out of the scaler so a fold's outcome cannot set the covariate scale.
        del y
        frame = covariate_frame(features)
        continuous = [
            position
            for position in range(frame.shape[1])
            if is_continuous(frame.iloc[:, position].to_numpy())
        ]
        self.continuous_positions_ = continuous
        self.n_features_in_ = frame.shape[1]
        if continuous:
            values = frame.iloc[:, continuous].to_numpy(dtype=float)
            self.center_, self.scale_ = center_and_scale(values, sample_weight)
        return self

    def transform(self, features: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        check_is_fitted(self, "continuous_positions_")
        frame = covariate_frame(features)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} covariate columns, received {frame.shape[1]}."
            )
        scaled = frame.copy().astype(float)
        if self.continuous_positions_:
            values = scaled.iloc[:, self.continuous_positions_].to_numpy(dtype=float)
            scaled.iloc[:, self.continuous_positions_] = (values - self.center_) / self.scale_
        if isinstance(features, pd.DataFrame):
            return scaled
        return scaled.to_numpy(dtype=float)


def covariate_frame(features: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features
    array = np.asarray(features)
    if array.ndim != 2:
        raise ValueError("Covariates must be a two-dimensional table.")
    return pd.DataFrame(array)


def observed_unique(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    return np.unique(numeric[np.isfinite(numeric)])


def within_tolerance(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(np.abs(left - right) <= INTEGER_TOLERANCE))


def is_integer_valued(unique: np.ndarray) -> bool:
    # Absolute tolerance. A relative check treats 100000.2 as the integer 100000.
    return within_tolerance(unique, np.round(unique))


def is_binary_indicator(unique: np.ndarray) -> bool:
    ordered = np.sort(unique)
    for codes in INDICATOR_CODES:
        if ordered.size == len(codes) and within_tolerance(ordered, np.asarray(codes)):
            return True
    return False


def is_compact_integer_code(unique: np.ndarray) -> bool:
    """True for site-like ids: few integer levels, starting at 0 or 1."""
    if unique.size > CATEGORICAL_MAX_LEVELS or not is_integer_valued(unique):
        return False
    span = float(np.max(unique) - np.min(unique))
    if span > CATEGORICAL_MAX_LEVELS - 1:
        return False
    origin = float(np.round(np.min(unique)))
    return bool(np.any(np.abs(origin - np.asarray(CODE_ORIGINS)) <= INTEGER_TOLERANCE))


def is_continuous(values: np.ndarray) -> bool:
    unique = observed_unique(values)
    if unique.size < MINIMUM_DISTINCT_VALUES:
        return False
    if is_binary_indicator(unique):
        return False
    if is_compact_integer_code(unique):
        return False
    return True


def center_and_scale(
    values: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and population standard deviation. Zero-variance columns keep scale 1."""
    if sample_weight is None:
        center = values.mean(axis=0)
        scale = values.std(axis=0)
    else:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.shape[0] != values.shape[0]:
            raise ValueError(
                f"sample_weight has length {weights.shape[0]} for {values.shape[0]} rows."
            )
        if np.any(weights < 0.0):
            raise ValueError("sample_weight must be non-negative.")
        total = float(weights.sum())
        if total == 0.0:
            raise ValueError("sample_weight must have a positive sum.")
        center = (weights[:, None] * values).sum(axis=0) / total
        deviations = values - center
        scale = np.sqrt((weights[:, None] * np.square(deviations)).sum(axis=0) / total)
    scale = np.where(scale == 0.0, 1.0, scale)
    return np.asarray(center, dtype=float), np.asarray(scale, dtype=float)


class ScaledLearner(BaseEstimator):
    """Learner whose continuous covariates are scaled from the rows passed to fit."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ScaledLearner:
        self.scaler_ = CovariateScaler()
        self.scaler_.fit(features, sample_weight=sample_weight)
        transformed = self.scaler_.transform(features)
        self.model_ = clone(self.estimator)
        if sample_weight is None:
            self.model_.fit(transformed, y)
        else:
            self.model_.fit(transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(self.scaler_.transform(features))

    def __sklearn_tags__(self) -> object:
        tags = super().__sklearn_tags__()
        inner = get_tags(self.estimator)
        tags.estimator_type = inner.estimator_type
        tags.target_tags.multi_output = inner.target_tags.multi_output
        tags.classifier_tags = deepcopy(inner.classifier_tags)
        tags.regressor_tags = deepcopy(inner.regressor_tags)
        tags.transformer_tags = deepcopy(inner.transformer_tags)
        return tags


class ScaledClassifier(ScaledLearner):
    """Scaled learner that exposes class probabilities."""

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(self.scaler_.transform(features))

    @property
    def classes_(self) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.classes_


def clone_scaled_estimator(estimator: BaseEstimator) -> ScaledLearner:
    """Unfitted learner that scales continuous covariates from its own training rows."""
    cloned = clone(estimator)
    if hasattr(estimator, "predict_proba"):
        return ScaledClassifier(estimator=cloned)
    return ScaledLearner(estimator=cloned)


def fit_transform_covariates(features: pd.DataFrame) -> tuple[CovariateScaler, np.ndarray]:
    """Fit on these rows only. Callers must omit held-out rows."""
    scaler = CovariateScaler()
    transformed = scaler.fit_transform(features)
    return scaler, np.asarray(transformed, dtype=float)


def transform_covariates(scaler: CovariateScaler, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(scaler.transform(features), dtype=float)

"""Tests for backend abstraction layer."""

from __future__ import annotations

import numpy as np
import pytest

from insurance_thin_data.tabpfn.backends import (
    BackendNotAvailableError,
    MockBackend,
    get_backend,
)


def test_mock_backend_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.uniform(0.05, 0.15, size=50)

    backend = MockBackend(random_state=0)
    backend.fit(X, y)
    preds = backend.predict(X[:10])

    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))


def test_mock_backend_predict_before_fit():
    backend = MockBackend()
    X = np.ones((5, 3))
    with pytest.raises(RuntimeError, match="not fitted"):
        backend.predict(X)


def test_mock_backend_name():
    backend = MockBackend()
    assert backend.name == "mock"


def test_mock_backend_predict_quantiles():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.uniform(0.05, 0.2, size=30)

    backend = MockBackend()
    backend.fit(X, y)
    quantiles = backend.predict_quantiles(X[:5], [0.1, 0.5, 0.9])
    assert quantiles.shape == (5, 3)
    # Lower quantile <= median <= upper quantile
    assert np.all(quantiles[:, 0] <= quantiles[:, 1] + 1e-6)
    assert np.all(quantiles[:, 1] <= quantiles[:, 2] + 1e-6)


def test_get_backend_mock():
    backend = get_backend("mock")
    assert backend.name == "mock"


def test_get_backend_unknown():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent_backend_xyz")


def test_get_backend_tabicl_not_installed():
    """tabicl is not installed in test environment — should raise BackendNotAvailableError."""
    with pytest.raises(BackendNotAvailableError):
        get_backend("tabicl")


def test_get_backend_tabpfn_not_installed():
    """tabpfn is not installed in test environment — should raise BackendNotAvailableError."""
    with pytest.raises(BackendNotAvailableError):
        get_backend("tabpfn")


def test_get_backend_auto_falls_back_to_error():
    """auto with no backends installed should raise BackendNotAvailableError with helpful message."""
    with pytest.raises(BackendNotAvailableError, match="No backend available"):
        get_backend("auto")


def test_mock_predictions_near_training_mean():
    """Mock predictions should be close to training mean."""
    rng = np.random.default_rng(0)
    y = np.full(100, 0.1)
    X = rng.normal(size=(100, 3))

    backend = MockBackend(random_state=0)
    backend.fit(X, y)
    preds = backend.predict(X[:20])

    assert abs(np.mean(preds) - 0.1) < 0.05

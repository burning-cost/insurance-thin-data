"""Tests for InsuranceTabPFN main class."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_thin_data.tabpfn import InsuranceTabPFN
from insurance_thin_data.tabpfn.validators import ThinSegmentWarning


# ---------------------------------------------------------------------------
# Fit / predict basics
# ---------------------------------------------------------------------------


def test_fit_predict_with_exposure(small_df, small_y, small_exposure):
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(small_df, small_y, exposure=small_exposure)
    preds = model.predict(small_df, exposure=small_exposure)

    assert preds.shape == (len(small_df),)
    assert np.all(preds >= 0), "Predictions must be non-negative claim counts"
    assert np.all(np.isfinite(preds))


def test_fit_predict_no_exposure(small_df, small_y):
    loss_ratio = small_y / 0.5
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(small_df, loss_ratio)
    preds = model.predict(small_df)
    assert preds.shape == (len(small_df),)
    assert np.all(np.isfinite(preds))


def test_fit_predict_numpy_arrays(small_df, small_y, small_exposure):
    # Use only numeric columns — numpy arrays cannot hold mixed types
    X_numeric = small_df.select_dtypes(include=["number"])
    X_arr = X_numeric.values.astype(float)
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(X_arr, small_y, exposure=small_exposure)
    preds = model.predict(X_arr, exposure=small_exposure)
    assert preds.shape == (len(small_df),)


def test_predict_returns_claims_not_rates(small_df, small_y, small_exposure):
    """With exposure, predict() returns expected claims (rate * exposure)."""
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(small_df, small_y, exposure=small_exposure)

    # Predict with exposure=1 should give raw rate
    unit_exposure = np.ones(len(small_df))
    rate_preds = model.predict(small_df, exposure=unit_exposure)

    # Predict with 2x exposure should give 2x claims
    double_exposure = 2.0 * unit_exposure
    double_preds = model.predict(small_df, exposure=double_exposure)

    # Due to mock noise, ratio won't be exactly 2 — but directionally correct
    mean_ratio = np.mean(double_preds) / (np.mean(rate_preds) + 1e-10)
    assert 1.5 < mean_ratio < 2.5, f"Expected ~2x, got {mean_ratio:.2f}"


def test_predict_before_fit_raises():
    model = InsuranceTabPFN(backend="mock")
    X = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


def test_predict_missing_exposure_raises(fitted_model, small_df):
    """Exposure-fitted model must require exposure at predict time."""
    with pytest.raises(ValueError, match="must pass exposure"):
        fitted_model.predict(small_df)


def test_predict_extra_exposure_warns(fitted_model_no_exposure, small_df, small_exposure):
    """Model fitted without exposure warns if exposure provided at predict time."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fitted_model_no_exposure.predict(small_df, exposure=small_exposure)
    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert any("ignored" in str(x.message).lower() for x in user_warns)


# ---------------------------------------------------------------------------
# Exposure transform correctness
# ---------------------------------------------------------------------------

def test_exposure_transform_log_appended():
    """log(exposure) must be appended as an additional feature column."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.poisson(0.08, size=n).astype(float)
    exposure = rng.uniform(0.1, 1.0, size=n)

    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(X, y, exposure=exposure)

    # With 2 features + log(exposure), internal feature count should be 3
    assert model._n_features_in == 3


def test_no_exposure_transform():
    """Without exposure, feature count stays as-is."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.uniform(0.1, 1.0, size=n)

    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(X, y)

    assert model._n_features_in == 2


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def test_categorical_columns_encoded(small_df, small_y, small_exposure):
    """Categorical columns (object dtype) are label-encoded automatically."""
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(small_df, small_y, exposure=small_exposure)
    # If this doesn't raise, encoding worked
    preds = model.predict(small_df, exposure=small_exposure)
    assert np.all(np.isfinite(preds))


def test_unseen_category_handled(small_df, small_y, small_exposure):
    """Unseen categories at predict time should not raise — return category 0."""
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(small_df, small_y, exposure=small_exposure)

    X_pred = small_df.copy()
    X_pred.loc[0, "region"] = "NewRegionXYZ"
    # Should not raise
    preds = model.predict(X_pred, exposure=small_exposure)
    assert np.isfinite(preds[0])


# ---------------------------------------------------------------------------
# Conformal prediction intervals
# ---------------------------------------------------------------------------

def test_predict_interval_shape(fitted_model, small_df, small_exposure):
    lower, point, upper = fitted_model.predict_interval(
        small_df, exposure=small_exposure, alpha=0.1
    )
    n = len(small_df)
    assert lower.shape == (n,)
    assert point.shape == (n,)
    assert upper.shape == (n,)


def test_predict_interval_ordering(fitted_model, small_df, small_exposure):
    lower, point, upper = fitted_model.predict_interval(
        small_df, exposure=small_exposure
    )
    assert np.all(lower <= point + 1e-9), "lower must be <= point"
    assert np.all(point <= upper + 1e-9), "point must be <= upper"


def test_predict_interval_non_negative(fitted_model, small_df, small_exposure):
    lower, _, _ = fitted_model.predict_interval(
        small_df, exposure=small_exposure
    )
    assert np.all(lower >= 0), "Lower bound must be non-negative (claim counts >= 0)"


def test_predict_interval_no_conformal_raises():
    """Model with conformal_test_size=0 should raise on predict_interval."""
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = rng.poisson(0.1, size=n).astype(float)

    model = InsuranceTabPFN(backend="mock", conformal_test_size=0.0, random_state=0)
    model.fit(X, y)

    with pytest.raises(RuntimeError, match="No conformal calibration"):
        model.predict_interval(X)


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------

def test_get_params_set_params():
    """Sklearn API: get_params and set_params must work."""
    model = InsuranceTabPFN(backend="mock", n_estimators=4)
    params = model.get_params()
    assert "backend" in params
    assert params["backend"] == "mock"

    model.set_params(n_estimators=8)
    assert model.n_estimators == 8


def test_feature_names_out(fitted_model, small_df):
    names = fitted_model.get_feature_names_out()
    # Should return training feature names (not including log_exposure)
    assert set(names) == set(small_df.columns)

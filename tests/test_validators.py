"""Tests for input validation and thin-segment warnings."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_thin_data.tabpfn.validators import (
    ExposureWarning,
    ThinSegmentWarning,
    ValidationError,
    validate_inputs,
    validate_feature_names,
    THIN_SEGMENT_RECOMMENDED_MAX,
    TABPFN_HARD_LIMIT,
    THIN_SEGMENT_MINIMUM,
)


def make_valid_inputs(n=100):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.poisson(0.1, size=n).astype(float)
    exposure = rng.uniform(0.1, 1.0, size=n)
    return X, y, exposure


def test_valid_inputs_pass():
    X, y, exposure = make_valid_inputs()
    X_arr, y_arr, exp_arr = validate_inputs(X, y, exposure)
    assert X_arr.shape == (100, 2)
    assert y_arr.shape == (100,)
    assert exp_arr is not None and exp_arr.shape == (100,)


def test_valid_inputs_no_exposure():
    X, y, _ = make_valid_inputs()
    X_arr, y_arr, exp_arr = validate_inputs(X, y, None)
    assert exp_arr is None


def test_numpy_array_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.poisson(0.1, size=50).astype(float)
    X_arr, y_arr, _ = validate_inputs(X, y)
    assert X_arr.shape == (50, 3)


def test_y_negative_raises():
    X, y, _ = make_valid_inputs()
    y[5] = -1.0
    with pytest.raises(ValidationError, match="negative"):
        validate_inputs(X, y)


def test_nan_in_X_raises():
    X, y, _ = make_valid_inputs()
    X.iloc[3, 0] = np.nan
    with pytest.raises(ValidationError, match="non-finite"):
        validate_inputs(X, y)


def test_inf_in_X_raises():
    X, y, _ = make_valid_inputs()
    X.iloc[0, 1] = np.inf
    with pytest.raises(ValidationError, match="non-finite"):
        validate_inputs(X, y)


def test_shape_mismatch_raises():
    X, y, _ = make_valid_inputs()
    with pytest.raises(ValidationError, match="100 rows but y has 99"):
        validate_inputs(X, y[:99])


def test_exposure_shape_mismatch_raises():
    X, y, exp = make_valid_inputs()
    with pytest.raises(ValidationError, match="exposure has 99"):
        validate_inputs(X, y, exp[:99])


def test_zero_exposure_raises():
    X, y, exp = make_valid_inputs()
    exp[10] = 0.0
    with pytest.raises(ValidationError, match="zero or negative"):
        validate_inputs(X, y, exp)


def test_negative_exposure_raises():
    X, y, exp = make_valid_inputs()
    exp[5] = -0.1
    with pytest.raises(ValidationError, match="zero or negative"):
        validate_inputs(X, y, exp)


def test_large_exposure_warns():
    X, y, exp = make_valid_inputs()
    exp[0] = 2.0  # > 1.5 threshold
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_inputs(X, y, exp, check_size=False)
    exposure_warns = [x for x in w if issubclass(x.category, ExposureWarning)]
    assert len(exposure_warns) > 0
    assert "days" in str(exposure_warns[0].message).lower() or "months" in str(exposure_warns[0].message).lower()


def test_thin_segment_below_minimum_warns():
    rng = np.random.default_rng(0)
    n = THIN_SEGMENT_MINIMUM - 5
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = rng.poisson(0.1, size=n).astype(float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_inputs(X, y)
    thin_warns = [x for x in w if issubclass(x.category, ThinSegmentWarning)]
    assert len(thin_warns) > 0


def test_thin_segment_above_recommended_warns():
    rng = np.random.default_rng(0)
    n = THIN_SEGMENT_RECOMMENDED_MAX + 100
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = rng.poisson(0.1, size=n).astype(float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_inputs(X, y)
    thin_warns = [x for x in w if issubclass(x.category, ThinSegmentWarning)]
    assert len(thin_warns) > 0


def test_above_hard_limit_warns():
    rng = np.random.default_rng(0)
    n = TABPFN_HARD_LIMIT + 100
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = rng.poisson(0.1, size=n).astype(float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_inputs(X, y)
    thin_warns = [x for x in w if issubclass(x.category, ThinSegmentWarning)]
    assert len(thin_warns) > 0
    assert "hard limit" in str(thin_warns[0].message).lower()


def test_validate_feature_names_missing_raises():
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_pred = pd.DataFrame({"a": [5, 6]})  # missing 'b'
    with pytest.raises(ValidationError, match="missing"):
        validate_feature_names(X_train, X_pred)


def test_validate_feature_names_extra_warns():
    X_train = pd.DataFrame({"a": [1, 2]})
    X_pred = pd.DataFrame({"a": [5, 6], "extra_col": [7, 8]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_feature_names(X_train, X_pred)
    assert len(w) > 0
    assert "extra" in str(w[0].message).lower()

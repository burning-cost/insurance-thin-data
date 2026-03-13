"""Tests for PDP-based relativities extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_thin_data.tabpfn import InsuranceTabPFN, RelativitiesExtractor


@pytest.fixture
def fitted_model_for_relat():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "age": rng.uniform(18, 80, size=n),
        "region": rng.choice(["North", "South", "London"], size=n),
        "mileage": rng.uniform(1000, 20000, size=n),
    })
    y = rng.poisson(0.08, size=n).astype(float)
    exposure = rng.uniform(0.1, 1.0, size=n)

    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(X, y, exposure=exposure)
    return model, X, exposure


def test_extract_single_numeric_feature(fitted_model_for_relat):
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=10, n_sample_rows=50)
    df = extractor.extract(X, "age")

    assert isinstance(df, pd.DataFrame)
    assert "feature_value" in df.columns
    assert "mean_prediction" in df.columns
    assert "relativity" in df.columns
    assert len(df) <= 10
    assert np.all(np.isfinite(df["relativity"].values))


def test_extract_single_categorical_feature(fitted_model_for_relat):
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=10, n_sample_rows=50)
    df = extractor.extract(X, "region")

    # Categorical: one row per unique value
    assert set(df["feature_value"]) <= {"North", "South", "London"}
    assert np.all(df["relativity"].values > 0)


def test_extract_by_index(fitted_model_for_relat):
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=5, n_sample_rows=40)
    df = extractor.extract(X, 0)  # 'age' by index
    assert len(df) <= 5


def test_relativities_mean_near_one(fitted_model_for_relat):
    """Mean relativity across grid should be close to 1.0 by definition."""
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=10, n_sample_rows=60)
    df = extractor.extract(X, "mileage")
    # By construction: mean(relativity) = mean(pred) / grand_mean = 1.0 when evenly weighted
    # This is approximate due to non-uniform grid spacing
    assert 0.5 < df["relativity"].mean() < 2.0


def test_extract_all_returns_dict(fitted_model_for_relat):
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=5, n_sample_rows=40)
    tables = extractor.extract_all(X)
    assert isinstance(tables, dict)
    assert set(tables.keys()) == {"age", "region", "mileage"}


def test_extract_subset(fitted_model_for_relat):
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=5, n_sample_rows=40)
    tables = extractor.extract_all(X, features=["age", "mileage"])
    assert "region" not in tables
    assert "age" in tables
    assert "mileage" in tables


def test_to_factor_table_shape(fitted_model_for_relat):
    model, X, exposure = fitted_model_for_relat
    extractor = RelativitiesExtractor(model, n_grid_points=5, n_sample_rows=40)
    ft = extractor.to_factor_table(X)
    assert isinstance(ft, pd.DataFrame)
    assert list(ft.columns) == ["feature", "feature_value", "relativity"]
    assert len(ft) > 0


def test_numpy_array_input(fitted_model_for_relat):
    """RelativitiesExtractor should work with numpy arrays (by index)."""
    model, X, exposure = fitted_model_for_relat
    X_arr = X[["age", "mileage"]].values

    rng = np.random.default_rng(0)
    model2 = InsuranceTabPFN(backend="mock", random_state=0)
    y = rng.poisson(0.08, size=len(X_arr)).astype(float)
    model2.fit(X_arr, y)

    extractor = RelativitiesExtractor(model2, n_grid_points=5, n_sample_rows=40)
    df = extractor.extract(X_arr, 0)
    assert "feature_value" in df.columns

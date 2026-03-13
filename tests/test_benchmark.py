"""Tests for GLMBenchmark and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_thin_data.tabpfn.benchmark import (
    BenchmarkResult,
    ComparisonResult,
    GLMBenchmark,
    _gini,
    _poisson_deviance,
    _double_lift,
)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def test_gini_perfect_ranking():
    """Perfect ranking should give Gini close to 1."""
    y = np.array([0.0, 0.0, 0.1, 0.2, 0.5])
    yhat = np.array([0.01, 0.02, 0.08, 0.15, 0.45])
    g = _gini(y, yhat)
    assert g > 0.4


def test_gini_random_ranking():
    """Random ranking should give Gini near 0."""
    rng = np.random.default_rng(0)
    y = np.ones(1000) * 0.1
    yhat = rng.uniform(0, 1, size=1000)
    g = _gini(y, yhat)
    assert abs(g) < 0.15


def test_gini_empty():
    assert _gini(np.array([]), np.array([])) == 0.0


def test_gini_with_exposure():
    y = np.array([0.1, 0.2, 0.05, 0.3])
    yhat = np.array([0.09, 0.18, 0.06, 0.28])
    exp = np.array([1.0, 2.0, 0.5, 1.5])
    g = _gini(y, yhat, exp)
    assert isinstance(g, float)
    assert -1 <= g <= 1


def test_poisson_deviance_perfect():
    """If predictions equal actuals, deviance should be near 0."""
    y = np.array([0.1, 0.2, 0.05])
    yhat = np.array([0.1, 0.2, 0.05])
    d = _poisson_deviance(y, yhat)
    assert abs(d) < 1e-8


def test_poisson_deviance_positive():
    y = np.array([0.1, 0.5, 0.05])
    yhat = np.array([0.2, 0.3, 0.08])
    d = _poisson_deviance(y, yhat)
    assert d > 0


def test_double_lift_shape():
    rng = np.random.default_rng(0)
    n = 100
    y = rng.poisson(0.1, size=n).astype(float)
    tabpfn = rng.uniform(0.05, 0.2, size=n)
    glm = rng.uniform(0.05, 0.2, size=n)
    exp = rng.uniform(0.5, 1.0, size=n)
    df = _double_lift(y, tabpfn, glm, exp, n_deciles=10)
    assert len(df) == 10
    assert "decile" in df.columns
    assert "actual_rate" in df.columns
    assert "tabpfn_rate" in df.columns
    assert "glm_rate" in df.columns


def test_double_lift_no_glm():
    rng = np.random.default_rng(0)
    n = 100
    y = rng.poisson(0.1, size=n).astype(float)
    tabpfn = rng.uniform(0.05, 0.2, size=n)
    df = _double_lift(y, tabpfn, None, None, n_deciles=5)
    assert len(df) == 5
    assert "glm_rate" not in df.columns


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

def test_benchmark_result_to_series():
    rng = np.random.default_rng(0)
    dl = _double_lift(
        np.array([0.1, 0.2]),
        np.array([0.09, 0.18]),
        None, None, n_deciles=2
    )
    result = BenchmarkResult(
        model_name="TestModel",
        gini=0.45,
        poisson_deviance=0.03,
        rmse=0.02,
        double_lift=dl,
        n_samples=100,
    )
    s = result.to_series()
    assert s["model"] == "TestModel"
    assert s["gini"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# GLMBenchmark
# ---------------------------------------------------------------------------

def make_glm_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.uniform(18, 80, size=n),
        "mileage": rng.uniform(1000, 20000, size=n),
    })
    exposure = rng.uniform(0.1, 1.0, size=n)
    rate = 0.05 + 0.002 * (X["age"] - 30)
    y = rng.poisson(rate * exposure)
    return X, y.astype(float), exposure


def test_glm_benchmark_fit_no_statsmodels():
    """GLMBenchmark raises helpful error if statsmodels not installed."""
    try:
        import statsmodels  # noqa: F401
        pytest.skip("statsmodels is installed — skipping no-statsmodels test")
    except ImportError:
        pass

    X, y, exposure = make_glm_data()
    bench = GLMBenchmark()
    with pytest.raises(ImportError, match="statsmodels"):
        bench.fit(X, y, exposure=exposure)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("statsmodels"),
    reason="statsmodels not installed",
)
def test_glm_benchmark_fit_predict():
    X, y, exposure = make_glm_data()
    bench = GLMBenchmark()
    bench.fit(X, y, exposure=exposure)
    preds = bench.predict(X, exposure=exposure)
    assert preds.shape == (len(X),)
    assert np.all(preds > 0)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("statsmodels"),
    reason="statsmodels not installed",
)
def test_glm_benchmark_compare():
    from insurance_thin_data.tabpfn import InsuranceTabPFN

    X, y, exposure = make_glm_data(n=300)
    split = 200

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    exp_train, exp_test = exposure[:split], exposure[split:]

    # Fit GLM benchmark
    bench = GLMBenchmark()
    bench.fit(X_train, y_train, exposure=exp_train)

    # Fit TabPFN (mock)
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(X_train, y_train, exposure=exp_train)
    tabpfn_preds = model.predict(X_test, exposure=exp_test)

    # Compare
    result = bench.compare(X_test, y_test, tabpfn_preds, exposure_test=exp_test)
    assert isinstance(result, ComparisonResult)
    assert result.tabpfn.n_samples == len(y_test)
    assert result.glm is not None
    assert result.glm.n_samples == len(y_test)

    # Summary table
    df = result.to_dataframe()
    assert len(df) == 2
    assert "gini" in df.columns

    # winner() should return a valid model name
    winner = result.winner()
    assert winner in ("InsuranceTabPFN", "Poisson GLM")

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


# ---------------------------------------------------------------------------
# Regression tests for P0-4: Poisson deviance zero-count handling
# ---------------------------------------------------------------------------

class TestPoissonDevianceRegression:
    """Regression tests for the Poisson deviance zero-observation bug (P0-4).

    Before the fix, the deviance mixed y_actual (unclipped) in the linear term
    with y=clip(y_actual, eps) in the log term. For zero-count observations
    this caused:
        deviance_i = 2 * (eps * log(eps / yhat) - (0 - yhat))
    instead of the correct:
        deviance_i = 2 * (0 - (0 - yhat)) = 2 * yhat
    """

    def test_zero_actual_gives_correct_deviance(self):
        """Poisson deviance with y=0 should equal 2*yhat (log term vanishes)."""
        y_actual = np.array([0.0, 0.0, 0.0])
        y_predicted = np.array([0.5, 1.0, 2.0])
        d = _poisson_deviance(y_actual, y_predicted)
        # D = mean(2 * (0 - (0 - yhat))) = mean(2 * yhat)
        expected = float(np.mean(2.0 * y_predicted))
        assert abs(d - expected) < 1e-8, f"Zero-actual deviance: got {d}, expected {expected}"

    def test_deviance_consistent_nonneg_actuals(self):
        """Mixed zeros and positives should be consistent with per-element calculation."""
        rng = np.random.default_rng(200)
        n = 200
        y_actual = rng.poisson(0.3, size=n).astype(float)  # many zeros
        y_predicted = rng.uniform(0.1, 0.8, size=n)

        d = _poisson_deviance(y_actual, y_predicted)

        # Manually compute the correct formula
        yhat = np.maximum(y_predicted, 1e-10)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(y_actual > 0, y_actual * np.log(y_actual / yhat), 0.0)
        expected = float(np.mean(2.0 * (log_term - (y_actual - yhat))))

        assert abs(d - expected) < 1e-10, (
            f"Deviance mismatch: got {d}, expected {expected}"
        )

    def test_perfect_predictions_zero_deviance(self):
        """With zero actual counts, perfect prediction (yhat=0+eps) gives near-zero deviance."""
        # The old code would give a non-zero deviance even when yhat matches y_actual
        # for zero counts because the linear term used unclipped y_actual=0 while
        # the log term used clipped y=eps.
        y_actual = np.array([0.0, 1.0, 2.0, 0.0])
        y_predicted = np.array([1e-9, 1.0, 2.0, 1e-9])  # near-perfect
        d = _poisson_deviance(y_actual, y_predicted)
        # Should be very close to zero for near-perfect predictions
        assert d < 1e-5, f"Near-perfect deviance too large: {d}"

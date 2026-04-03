"""Extended tests for NegativeTransferDiagnostic and deviance metrics."""

import numpy as np
import pytest

from insurance_thin_data.transfer.diagnostic import (
    NegativeTransferDiagnostic,
    TransferDiagnosticResult,
    poisson_deviance,
    gamma_deviance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConstModel:
    def __init__(self, val: float):
        self.val = val

    def predict(self, X, exposure=None):
        return np.full(X.shape[0], self.val)


class ScaledPredictor:
    """Predicts from stored predictions array."""
    def __init__(self, preds: np.ndarray):
        self.preds = preds

    def predict(self, X, exposure=None):
        return self.preds


# ---------------------------------------------------------------------------
# poisson_deviance extended
# ---------------------------------------------------------------------------

class TestPoissonDevianceExtended:
    def test_perfect_prediction_zero(self):
        y = np.array([2.0, 3.0, 1.0, 0.0])
        assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-8)

    def test_zero_y_no_nan(self):
        y = np.zeros(10)
        mu = np.ones(10)
        d = poisson_deviance(y, mu)
        assert np.isfinite(d)

    def test_returns_float(self):
        d = poisson_deviance(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert isinstance(d, float)

    def test_non_negative_for_any_predictions(self):
        rng = np.random.default_rng(100)
        y = rng.poisson(1.0, 200).astype(float)
        mu = rng.uniform(0.1, 5.0, 200)
        d = poisson_deviance(y, mu)
        assert d >= 0

    def test_mu_clipped_to_avoid_log_zero(self):
        # mu=0 should not produce nan/inf
        y = np.array([1.0, 0.0])
        mu = np.array([0.0, 0.0])
        d = poisson_deviance(y, mu)
        assert np.isfinite(d)

    def test_deviance_higher_for_bad_model(self):
        y = np.ones(100) * 2.0
        mu_good = np.ones(100) * 2.0
        mu_bad = np.ones(100) * 0.1
        assert poisson_deviance(y, mu_bad) > poisson_deviance(y, mu_good)

    def test_single_observation(self):
        d = poisson_deviance(np.array([1.0]), np.array([1.0]))
        assert d == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# gamma_deviance extended
# ---------------------------------------------------------------------------

class TestGammaDevianceExtended:
    def test_perfect_prediction_zero(self):
        y = np.array([100.0, 200.0, 50.0])
        d = gamma_deviance(y, y)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_returns_float(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        d = gamma_deviance(y, mu)
        assert isinstance(d, float)

    def test_non_negative(self):
        rng = np.random.default_rng(101)
        y = rng.gamma(2.0, 500.0, 100)
        mu = rng.gamma(2.0, 500.0, 100)
        d = gamma_deviance(y, mu)
        assert d >= 0

    def test_deviance_higher_for_bad_model(self):
        y = np.ones(100) * 1000.0
        mu_good = np.ones(100) * 1000.0
        mu_bad = np.ones(100) * 10.0
        assert gamma_deviance(y, mu_bad) > gamma_deviance(y, mu_good)

    def test_near_zero_mu_handled(self):
        y = np.array([100.0, 200.0])
        mu = np.array([0.0, 0.0])  # clipped internally
        d = gamma_deviance(y, mu)
        assert np.isfinite(d)


# ---------------------------------------------------------------------------
# TransferDiagnosticResult
# ---------------------------------------------------------------------------

class TestTransferDiagnosticResult:
    def _make(self, ntg=0.1, **kw):
        defaults = dict(
            poisson_deviance_transfer=1.2,
            poisson_deviance_target_only=1.1,
            poisson_deviance_source_only=1.5,
            ntg=ntg,
            ntg_relative=ntg / 1.1 * 100,
            transfer_is_beneficial=(ntg < 0),
            n_test=100,
        )
        defaults.update(kw)
        return TransferDiagnosticResult(**defaults)

    def test_ntg_positive_means_harmful(self):
        r = self._make(ntg=0.2)
        assert not r.transfer_is_beneficial

    def test_ntg_negative_means_beneficial(self):
        r = self._make(ntg=-0.1, transfer_is_beneficial=True)
        assert r.transfer_is_beneficial

    def test_repr_is_string(self):
        r = self._make()
        text = repr(r)
        assert isinstance(text, str)
        assert "NTG" in text or "ntg" in text.lower()

    def test_repr_harmful(self):
        r = self._make(ntg=0.5)
        text = repr(r)
        assert "HARMFUL" in text or "harmful" in text.lower()

    def test_per_feature_analysis_default_empty(self):
        r = self._make()
        assert isinstance(r.per_feature_analysis, dict)

    def test_n_test_stored(self):
        r = self._make(n_test=250)
        assert r.n_test == 250

    def test_source_only_deviance_optional_none(self):
        r = self._make(poisson_deviance_source_only=None)
        assert r.poisson_deviance_source_only is None


# ---------------------------------------------------------------------------
# NegativeTransferDiagnostic
# ---------------------------------------------------------------------------

class TestNegativeTransferDiagnosticExtended:
    def _make_data(self, seed=50, n=200):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 4))
        y = rng.poisson(np.exp(0.2 * X[:, 0])).astype(float)
        exp = np.ones(n)
        return X, y, exp

    def test_basic_evaluation(self):
        X, y, exp = self._make_data()
        transfer = ConstModel(0.5)
        target_only = ConstModel(0.6)
        diag = NegativeTransferDiagnostic()
        result = diag.evaluate(X, y, exp, transfer_model=transfer, target_only_model=target_only)
        assert isinstance(result, TransferDiagnosticResult)

    def test_ntg_is_float(self):
        X, y, exp = self._make_data(seed=51)
        result = NegativeTransferDiagnostic().evaluate(
            X, y, exp,
            transfer_model=ConstModel(0.4),
            target_only_model=ConstModel(0.5),
        )
        assert isinstance(result.ntg, float)

    def test_ntg_equals_deviance_difference(self):
        X, y, exp = self._make_data(seed=52)
        m_transfer = ConstModel(1.0)
        m_target = ConstModel(0.8)
        result = NegativeTransferDiagnostic().evaluate(
            X, y, exp,
            transfer_model=m_transfer,
            target_only_model=m_target,
        )
        expected_ntg = result.poisson_deviance_transfer - result.poisson_deviance_target_only
        assert result.ntg == pytest.approx(expected_ntg, abs=1e-8)

    def test_beneficial_when_transfer_better(self):
        rng = np.random.default_rng(53)
        X = rng.standard_normal((200, 4))
        true_mu = np.exp(0.3 * X[:, 0])
        y = rng.poisson(true_mu).astype(float)
        exp = np.ones(200)

        # Transfer model much better than target-only
        m_transfer = ScaledPredictor(true_mu)
        m_target = ConstModel(y.mean() * 2)  # bad model

        result = NegativeTransferDiagnostic().evaluate(
            X, y, exp,
            transfer_model=m_transfer,
            target_only_model=m_target,
        )
        assert result.transfer_is_beneficial

    def test_gamma_deviance_metric(self):
        rng = np.random.default_rng(54)
        X = rng.standard_normal((200, 3))
        y = rng.gamma(2.0, 500.0, 200)
        exp = np.ones(200)
        m1 = ConstModel(1000.0)
        m2 = ConstModel(900.0)
        diag = NegativeTransferDiagnostic(metric="gamma_deviance")
        result = diag.evaluate(X, y, exp, transfer_model=m1, target_only_model=m2)
        assert isinstance(result.ntg, float)
        assert np.isfinite(result.ntg)

    def test_n_test_recorded(self):
        X, y, exp = self._make_data(seed=55, n=150)
        result = NegativeTransferDiagnostic().evaluate(
            X, y, exp,
            transfer_model=ConstModel(0.5),
            target_only_model=ConstModel(0.5),
        )
        assert result.n_test == 150

    def test_source_only_model_optional(self):
        X, y, exp = self._make_data(seed=56)
        result = NegativeTransferDiagnostic().evaluate(
            X, y, exp,
            transfer_model=ConstModel(0.5),
            target_only_model=ConstModel(0.6),
            source_only_model=ConstModel(0.7),
        )
        assert result.poisson_deviance_source_only is not None

    def test_ntg_relative_computed(self):
        X, y, exp = self._make_data(seed=57)
        result = NegativeTransferDiagnostic().evaluate(
            X, y, exp,
            transfer_model=ConstModel(0.5),
            target_only_model=ConstModel(0.6),
        )
        assert isinstance(result.ntg_relative, float)
        assert np.isfinite(result.ntg_relative)

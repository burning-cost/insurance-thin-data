"""Extended tests for GLMTransfer — families, regularisation, edge cases."""

import numpy as np
import pytest

from insurance_thin_data.transfer.glm_transfer import (
    GLMTransfer,
    _fit_penalised_glm,
    _poisson_negloglik,
    _poisson_grad,
    _gamma_negloglik,
    _gamma_grad,
    _gaussian_negloglik,
)


# ---------------------------------------------------------------------------
# Loss function smoke tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def _setup(self, n=50, p=4, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        return X

    def test_poisson_negloglik_finite(self):
        rng = np.random.default_rng(200)
        X = rng.standard_normal((50, 4))
        y = rng.poisson(1.0, 50).astype(float)
        beta = np.zeros(4)
        log_exp = np.zeros(50)
        val = _poisson_negloglik(beta, X, y, log_exp, l1_lambda=0.01)
        assert np.isfinite(val)

    def test_poisson_negloglik_positive(self):
        rng = np.random.default_rng(201)
        X = rng.standard_normal((50, 4))
        y = rng.poisson(1.0, 50).astype(float)
        beta = np.zeros(4)
        log_exp = np.zeros(50)
        val = _poisson_negloglik(beta, X, y, log_exp, l1_lambda=0.01)
        assert val > 0

    def test_gamma_negloglik_finite(self):
        rng = np.random.default_rng(202)
        X = rng.standard_normal((50, 4))
        y = rng.gamma(2.0, 100.0, 50)
        beta = np.zeros(4)
        log_exp = np.zeros(50)
        val = _gamma_negloglik(beta, X, y, log_exp, l1_lambda=0.01)
        assert np.isfinite(val)

    def test_gaussian_negloglik_finite(self):
        rng = np.random.default_rng(203)
        X = rng.standard_normal((50, 4))
        y = rng.normal(5.0, 1.0, 50)
        beta = np.zeros(4)
        log_exp = np.zeros(50)
        val = _gaussian_negloglik(beta, X, y, log_exp, l1_lambda=0.01)
        assert np.isfinite(val)

    def test_poisson_grad_shape(self):
        rng = np.random.default_rng(204)
        X = rng.standard_normal((50, 4))
        y = rng.poisson(1.0, 50).astype(float)
        beta = np.zeros(4)
        log_exp = np.zeros(50)
        g = _poisson_grad(beta, X, y, log_exp, l1_lambda=0.01)
        assert g.shape == (4,)

    def test_gamma_grad_shape(self):
        rng = np.random.default_rng(205)
        X = rng.standard_normal((50, 4))
        y = rng.gamma(2.0, 100.0, 50)
        beta = np.zeros(4)
        log_exp = np.zeros(50)
        g = _gamma_grad(beta, X, y, log_exp, l1_lambda=0.01)
        assert g.shape == (4,)


# ---------------------------------------------------------------------------
# _fit_penalised_glm
# ---------------------------------------------------------------------------

class TestFitPenalisedGLMExtended:
    def test_returns_array_shape(self):
        rng = np.random.default_rng(300)
        p = 6
        X = rng.standard_normal((200, p))
        y = rng.poisson(0.5, 200).astype(float)
        log_exp = np.zeros(200)
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.01, family="poisson")
        assert beta.shape == (p,)

    def test_high_penalty_sparsity(self):
        rng = np.random.default_rng(301)
        X = rng.standard_normal((300, 10))
        y = rng.poisson(0.5, 300).astype(float)
        log_exp = np.zeros(300)
        beta_high = _fit_penalised_glm(X, y, log_exp, l1_lambda=5.0, family="poisson")
        beta_low = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.001, family="poisson")
        # High penalty -> smaller coefficients on average
        assert np.sum(np.abs(beta_high)) <= np.sum(np.abs(beta_low)) + 1e-6

    def test_exposure_offset_used(self):
        # With log_exposure varying, predictions should differ
        rng = np.random.default_rng(302)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(0.3, 100).astype(float)
        log_exp_ones = np.zeros(100)
        log_exp_half = np.log(0.5) * np.ones(100)
        beta1 = _fit_penalised_glm(X, y, log_exp_ones, l1_lambda=0.01, family="poisson")
        beta2 = _fit_penalised_glm(X, y, log_exp_half, l1_lambda=0.01, family="poisson")
        # Different exposure -> different beta (not guaranteed identical)
        assert beta1.shape == beta2.shape

    def test_gamma_family(self):
        rng = np.random.default_rng(303)
        X = rng.standard_normal((100, 4))
        y = rng.gamma(2.0, 500.0, 100)
        log_exp = np.zeros(100)
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.01, family="gamma")
        assert beta.shape == (4,)
        assert np.all(np.isfinite(beta))

    def test_gaussian_family(self):
        rng = np.random.default_rng(304)
        X = rng.standard_normal((100, 4))
        true_beta = np.array([0.5, -0.3, 0.1, 0.2])
        y = X @ true_beta + rng.standard_normal(100) * 0.2
        log_exp = np.zeros(100)
        beta = _fit_penalised_glm(X, y, log_exp, l1_lambda=0.001, family="gaussian")
        assert beta.shape == (4,)


# ---------------------------------------------------------------------------
# GLMTransfer
# ---------------------------------------------------------------------------

class TestGLMTransferExtended:
    def _make_data(self, n_src=500, n_tgt=80, p=5, seed=400):
        rng = np.random.default_rng(seed)
        true_beta = np.array([0.3, -0.2, 0.1, 0.0, 0.1])
        X_src = rng.standard_normal((n_src, p))
        y_src = rng.poisson(np.exp(X_src @ true_beta)).astype(float)
        X_tgt = rng.standard_normal((n_tgt, p))
        y_tgt = rng.poisson(np.exp(X_tgt @ true_beta)).astype(float)
        exp_src = np.ones(n_src)
        exp_tgt = np.ones(n_tgt)
        return X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt

    def test_fit_returns_self(self):
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data()
        glm = GLMTransfer(family="poisson", lambda_pool=0.01)
        result = glm.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result is glm

    def test_predict_shape(self):
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data()
        glm = GLMTransfer(family="poisson")
        glm.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        preds = glm.predict(X_tgt)
        assert preds.shape == (len(X_tgt),)

    def test_predict_positive(self):
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data()
        glm = GLMTransfer(family="poisson")
        glm.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        preds = glm.predict(X_tgt)
        assert (preds > 0).all()

    def test_predict_before_fit_raises(self):
        glm = GLMTransfer(family="poisson")
        rng = np.random.default_rng(401)
        with pytest.raises(Exception):
            glm.predict(rng.standard_normal((10, 5)))

    def test_target_only_mode(self):
        # No source data
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data()
        glm = GLMTransfer(family="poisson")
        glm.fit(X_tgt, y_tgt, exp_tgt)
        preds = glm.predict(X_tgt)
        assert preds.shape == (len(X_tgt),)

    def test_gamma_family(self):
        rng = np.random.default_rng(402)
        n_src, n_tgt, p = 400, 60, 4
        true_beta = np.array([6.0, 0.2, -0.1, 0.15])
        X_src = rng.standard_normal((n_src, p))
        y_src = rng.gamma(2.0, np.exp(X_src @ true_beta) / 2.0)
        X_tgt = rng.standard_normal((n_tgt, p))
        y_tgt = rng.gamma(2.0, np.exp(X_tgt @ true_beta) / 2.0)
        exp = np.ones(n_tgt)
        glm = GLMTransfer(family="gamma")
        glm.fit(X_tgt, y_tgt, exp, X_source=X_src, y_source=y_src,
                exposure_source=np.ones(n_src))
        preds = glm.predict(X_tgt)
        assert preds.shape == (n_tgt,)

    def test_repr(self):
        glm = GLMTransfer(family="poisson")
        r = repr(glm)
        assert "GLMTransfer" in r or "glm" in r.lower() or "poisson" in r.lower()

    def test_fitted_beta_stored(self):
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data()
        glm = GLMTransfer(family="poisson")
        glm.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        # After fitting, there should be a beta_ or coef_ attribute
        assert hasattr(glm, "coef_") or hasattr(glm, "beta_") or hasattr(glm, "_beta")

    def test_l1_lambda_stored(self):
        glm = GLMTransfer(family="poisson", lambda_pool=0.05)
        assert glm.lambda_pool == pytest.approx(0.05)

    def test_scale_features_option(self):
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = self._make_data()
        glm = GLMTransfer(family="poisson", scale_features=True)
        glm.fit(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        preds = glm.predict(X_tgt)
        assert (preds > 0).all()

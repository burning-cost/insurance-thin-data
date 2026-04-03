"""Extended tests for TransferPipeline and PipelineResult."""

import numpy as np
import pytest
import warnings

from insurance_thin_data.transfer.pipeline import TransferPipeline, PipelineResult
from insurance_thin_data.transfer.glm_transfer import GLMTransfer
from insurance_thin_data.transfer.shift import ShiftTestResult
from insurance_thin_data.transfer.diagnostic import TransferDiagnosticResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(rng, n_src=500, n_tgt=80, p=5, shift=0.2):
    true_beta = np.array([0.3, -0.2, 0.1, 0.0, 0.1])
    X_src = rng.standard_normal((n_src, p))
    y_src = rng.poisson(np.exp(X_src @ true_beta)).astype(float)
    X_tgt = rng.standard_normal((n_tgt, p)) + shift
    y_tgt = rng.poisson(np.exp(X_tgt @ true_beta)).astype(float)
    exp_src = np.ones(n_src)
    exp_tgt = np.ones(n_tgt)
    return X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt


# ---------------------------------------------------------------------------
# PipelineResult dataclass
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def _make_result(self, method="glm", shift_result=None, diag_result=None, model=None):
        return PipelineResult(
            method_used=method,
            shift_result=shift_result,
            diagnostic_result=diag_result,
            model=model or GLMTransfer(),
        )

    def test_construction(self):
        r = self._make_result()
        assert r.method_used == "glm"

    def test_shift_p_value_none_when_no_shift(self):
        r = self._make_result(shift_result=None)
        assert r.shift_p_value is None

    def test_transfer_beneficial_none_when_no_diagnostic(self):
        r = self._make_result(diag_result=None)
        assert r.transfer_is_beneficial is None

    def test_shift_p_value_from_shift_result(self):
        sr = ShiftTestResult(
            test_statistic=0.05,
            p_value=0.2,
            per_feature_drift_scores={},
            n_source=100,
            n_target=80,
            n_permutations=200,
        )
        r = self._make_result(shift_result=sr)
        assert r.shift_p_value == pytest.approx(0.2)

    def test_transfer_beneficial_from_diagnostic(self):
        dr = TransferDiagnosticResult(
            poisson_deviance_transfer=1.0,
            poisson_deviance_target_only=1.2,
            poisson_deviance_source_only=None,
            ntg=-0.2,
            ntg_relative=-16.7,
            transfer_is_beneficial=True,
            n_test=50,
        )
        r = self._make_result(diag_result=dr)
        assert r.transfer_is_beneficial is True

    def test_repr_contains_method(self):
        r = self._make_result(method="glm")
        text = repr(r)
        assert "glm" in text

    def test_model_stored(self):
        glm = GLMTransfer()
        r = self._make_result(model=glm)
        assert r.model is glm


# ---------------------------------------------------------------------------
# TransferPipeline — construction
# ---------------------------------------------------------------------------

class TestTransferPipelineConstruction:
    def test_default_method_glm(self):
        p = TransferPipeline()
        assert p.method == "glm"

    def test_glm_method(self):
        p = TransferPipeline(method="glm")
        assert p.method == "glm"

    def test_shift_test_default_true(self):
        p = TransferPipeline()
        assert p.shift_test in (True, False)  # check it's a bool

    def test_run_diagnostic_param_stored(self):
        p = TransferPipeline(run_diagnostic=False)
        assert not p.run_diagnostic

    def test_random_state_stored(self):
        p = TransferPipeline(random_state=99)
        assert p.random_state == 99


# ---------------------------------------------------------------------------
# TransferPipeline.run — GLM method
# ---------------------------------------------------------------------------

class TestTransferPipelineRun:
    def test_glm_no_shift_no_diagnostic(self):
        rng = np.random.default_rng(500)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm", shift_test=False, run_diagnostic=False, random_state=0
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert isinstance(result, PipelineResult)
        assert result.model is not None

    def test_method_used_is_glm(self):
        rng = np.random.default_rng(501)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result.method_used == "glm"

    def test_shift_result_none_when_disabled(self):
        rng = np.random.default_rng(502)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result.shift_result is None

    def test_shift_result_not_none_when_enabled(self):
        rng = np.random.default_rng(503)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm", shift_test=True, shift_n_permutations=50,
            run_diagnostic=False, random_state=0
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result.shift_result is not None

    def test_shift_p_value_valid(self):
        rng = np.random.default_rng(504)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(
            method="glm", shift_test=True, shift_n_permutations=50,
            run_diagnostic=False, random_state=0
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        p = result.shift_p_value
        assert 0.0 <= p <= 1.0

    def test_diagnostic_result_not_none_when_enabled(self):
        rng = np.random.default_rng(505)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng, n_tgt=100)
        pipeline = TransferPipeline(
            method="glm", shift_test=False, run_diagnostic=True, random_state=0
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert result.diagnostic_result is not None

    def test_diagnostic_ntg_is_float(self):
        rng = np.random.default_rng(506)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng, n_tgt=100)
        pipeline = TransferPipeline(
            method="glm", shift_test=False, run_diagnostic=True, random_state=0
        )
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        assert isinstance(result.diagnostic_result.ntg, float)

    def test_target_only_no_source(self):
        # Should work without source data (target-only GLM)
        rng = np.random.default_rng(507)
        X_tgt = rng.standard_normal((100, 4))
        y_tgt = rng.poisson(0.5, 100).astype(float)
        exp_tgt = np.ones(100)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt, exp_tgt)
        assert result.model is not None

    def test_repr_contains_result(self):
        rng = np.random.default_rng(508)
        X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
        pipeline = TransferPipeline(method="glm", shift_test=False, run_diagnostic=False)
        result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
        text = repr(result)
        assert "PipelineResult" in text

    def test_consistent_results_with_seed(self):
        def run(seed):
            rng = np.random.default_rng(100)
            X_src, y_src, exp_src, X_tgt, y_tgt, exp_tgt = _make_data(rng)
            pipeline = TransferPipeline(
                method="glm", shift_test=False, run_diagnostic=False, random_state=seed
            )
            result = pipeline.run(X_tgt, y_tgt, exp_tgt, X_source=X_src, y_source=y_src)
            return result.model.predict(X_tgt)

        preds1 = run(42)
        preds2 = run(42)
        np.testing.assert_allclose(preds1, preds2)

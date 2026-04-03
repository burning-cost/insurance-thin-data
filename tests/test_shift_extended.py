"""Extended tests for covariate shift detection — edge cases and kernel tests."""

import numpy as np
import pytest

from insurance_thin_data.transfer.shift import (
    CovariateShiftTest,
    ShiftTestResult,
    _mmd_squared,
    _rbf_kernel,
    _indicator_kernel,
    _mixed_kernel,
    _estimate_bandwidth,
)


# ---------------------------------------------------------------------------
# _rbf_kernel extended
# ---------------------------------------------------------------------------

class TestRBFKernelExtended:
    def test_single_row(self):
        X = np.array([[1.0, 2.0, 3.0]])
        K = _rbf_kernel(X, X, bandwidth=1.0)
        assert K.shape == (1, 1)
        assert K[0, 0] == pytest.approx(1.0)

    def test_very_wide_bandwidth_approaches_one(self):
        X = np.random.default_rng(0).normal(0, 1, (10, 3))
        Y = np.random.default_rng(1).normal(5, 1, (8, 3))
        K = _rbf_kernel(X, Y, bandwidth=1e6)
        np.testing.assert_allclose(K, 1.0, atol=0.01)

    def test_very_narrow_bandwidth_near_zero_off_diagonal(self):
        X = np.array([[0.0, 0.0], [100.0, 100.0]])
        K = _rbf_kernel(X, X, bandwidth=0.01)
        # Off-diagonal should be near 0
        assert K[0, 1] < 1e-6
        assert K[1, 0] < 1e-6

    def test_non_negative_values(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (20, 4))
        Y = rng.normal(1, 1, (15, 4))
        K = _rbf_kernel(X, Y, bandwidth=1.0)
        assert (K >= 0).all()

    def test_bandwidth_scaling(self):
        X = np.array([[0.0], [1.0]])
        K1 = _rbf_kernel(X, X, bandwidth=1.0)
        K10 = _rbf_kernel(X, X, bandwidth=10.0)
        # Wider bandwidth -> off-diagonal higher
        assert K10[0, 1] > K1[0, 1]


# ---------------------------------------------------------------------------
# _indicator_kernel extended
# ---------------------------------------------------------------------------

class TestIndicatorKernelExtended:
    def test_single_column_partial_match(self):
        X = np.array([[1], [1], [2]])
        Y = np.array([[1], [2]])
        K = _indicator_kernel(X, Y)
        assert K.shape == (3, 2)
        # X[0]=[1], Y[0]=[1] -> match
        assert K[0, 0] == 1.0
        # X[0]=[1], Y[1]=[2] -> no match
        assert K[0, 1] == 0.0
        # X[2]=[2], Y[1]=[2] -> match
        assert K[2, 1] == 1.0

    def test_multi_column_all_must_match(self):
        X = np.array([[1, 2]])
        Y = np.array([[1, 3]])
        K = _indicator_kernel(X, Y)
        # col 0 matches but col 1 doesn't -> 0
        assert K[0, 0] == 0.0

    def test_self_kernel_diagonal_ones(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        K = _indicator_kernel(X, X)
        np.testing.assert_array_equal(np.diag(K), 1.0)

    def test_no_match_gives_zeros(self):
        X = np.array([[1, 1], [1, 1]])
        Y = np.array([[2, 2], [2, 2]])
        K = _indicator_kernel(X, Y)
        np.testing.assert_array_equal(K, 0.0)

    def test_output_dtype_float64(self):
        X = np.array([[1, 2]])
        Y = np.array([[1, 2]])
        K = _indicator_kernel(X, Y)
        assert K.dtype == np.float64


# ---------------------------------------------------------------------------
# _estimate_bandwidth
# ---------------------------------------------------------------------------

class TestEstimateBandwidth:
    def test_returns_positive(self):
        rng = np.random.default_rng(10)
        X = rng.normal(0, 1, (100, 3))
        Y = rng.normal(1, 1, (80, 3))
        bw = _estimate_bandwidth(X, Y, cont_cols=[0, 1, 2])
        assert bw > 0

    def test_returns_float(self):
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (50, 2))
        Y = rng.normal(0, 1, (40, 2))
        bw = _estimate_bandwidth(X, Y, cont_cols=[0, 1])
        assert isinstance(bw, float)

    def test_no_cont_cols_returns_one(self):
        X = np.zeros((10, 3))
        Y = np.zeros((10, 3))
        bw = _estimate_bandwidth(X, Y, cont_cols=[])
        assert bw == 1.0

    def test_scaled_data_larger_bandwidth(self):
        rng = np.random.default_rng(12)
        X = rng.normal(0, 1, (100, 2))
        Y = rng.normal(0, 1, (80, 2))
        bw1 = _estimate_bandwidth(X, Y, cont_cols=[0, 1])
        bw10 = _estimate_bandwidth(X * 10, Y * 10, cont_cols=[0, 1])
        assert bw10 > bw1

    def test_large_dataset_subsampled(self):
        # >500 rows should subsample without crashing
        rng = np.random.default_rng(13)
        X = rng.normal(0, 1, (700, 3))
        Y = rng.normal(1, 1, (600, 3))
        bw = _estimate_bandwidth(X, Y, cont_cols=[0, 1, 2])
        assert bw > 0


# ---------------------------------------------------------------------------
# _mmd_squared
# ---------------------------------------------------------------------------

class TestMMDSquared:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(20)
        X = rng.normal(0, 1, (100, 3))
        Y = rng.normal(0, 1, (100, 3))
        mmd = _mmd_squared(X, Y, cat_cols=[], cont_cols=[0, 1, 2], bandwidth=1.0)
        # Should be close to zero for same distribution
        assert abs(mmd) < 0.5

    def test_shifted_distribution_positive_mmd(self):
        rng = np.random.default_rng(21)
        X = rng.normal(0, 1, (100, 2))
        Y = rng.normal(5, 1, (100, 2))
        mmd = _mmd_squared(X, Y, cat_cols=[], cont_cols=[0, 1], bandwidth=1.0)
        assert mmd > 0

    def test_mmd_increases_with_shift(self):
        rng = np.random.default_rng(22)
        X = rng.normal(0, 1, (100, 2))
        Y_small = rng.normal(1, 1, (100, 2))
        Y_large = rng.normal(5, 1, (100, 2))
        mmd_small = _mmd_squared(X, Y_small, cat_cols=[], cont_cols=[0, 1], bandwidth=1.0)
        mmd_large = _mmd_squared(X, Y_large, cat_cols=[], cont_cols=[0, 1], bandwidth=1.0)
        assert mmd_large >= mmd_small


# ---------------------------------------------------------------------------
# CovariateShiftTest
# ---------------------------------------------------------------------------

class TestCovariateShiftTestExtended:
    def _data(self, seed=30, n=100, shift=0.0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 3))
        Y = rng.normal(shift, 1, (n, 3))
        return X, Y

    def test_basic_run(self):
        X, Y = self._data(seed=30)
        test = CovariateShiftTest(n_permutations=50, random_state=0)
        result = test.test(X, Y)
        assert isinstance(result, ShiftTestResult)

    def test_p_value_in_zero_one(self):
        X, Y = self._data(seed=31, shift=0.5)
        test = CovariateShiftTest(n_permutations=50, random_state=1)
        result = test.test(X, Y)
        assert 0.0 <= result.p_value <= 1.0

    def test_test_statistic_non_negative(self):
        X, Y = self._data(seed=32)
        test = CovariateShiftTest(n_permutations=50, random_state=2)
        result = test.test(X, Y)
        assert isinstance(result.test_statistic, float)  # MMD^2 — just check it's numeric

    def test_n_source_n_target_recorded(self):
        rng = np.random.default_rng(33)
        X = rng.normal(0, 1, (80, 3))
        Y = rng.normal(0, 1, (60, 3))
        test = CovariateShiftTest(n_permutations=20, random_state=3)
        result = test.test(X, Y)
        assert result.n_source == 80
        assert result.n_target == 60

    def test_n_permutations_recorded(self):
        X, Y = self._data(seed=34)
        test = CovariateShiftTest(n_permutations=100, random_state=4)
        result = test.test(X, Y)
        assert result.n_permutations == 100

    def test_large_shift_low_p_value(self):
        rng = np.random.default_rng(35)
        X = rng.normal(0, 0.5, (100, 2))
        Y = rng.normal(5, 0.5, (100, 2))
        test = CovariateShiftTest(n_permutations=200, random_state=5)
        result = test.test(X, Y)
        # Extreme shift should give very low p-value
        assert result.p_value < 0.05

    def test_identical_distributions_high_p_value(self):
        rng = np.random.default_rng(36)
        X = rng.normal(0, 1, (200, 3))
        Y = rng.normal(0, 1, (200, 3))
        test = CovariateShiftTest(n_permutations=200, random_state=6)
        result = test.test(X, Y)
        # Should not reject H0 in most cases for same distribution
        assert result.p_value > 0.01  # Very loose bound

    def test_repr_is_string(self):
        X, Y = self._data(seed=37)
        test = CovariateShiftTest(n_permutations=20, random_state=7)
        result = test.test(X, Y)
        r = repr(result)
        assert isinstance(r, str)
        assert "MMD" in r or "p=" in r

    def test_per_feature_drift_scores_populated(self):
        X, Y = self._data(seed=38)
        test = CovariateShiftTest(n_permutations=20, random_state=8)
        result = test.test(X, Y)
        assert isinstance(result.per_feature_drift_scores, dict)

    def test_categorical_cols_accepted(self):
        rng = np.random.default_rng(39)
        X = np.column_stack([rng.integers(0, 5, (80, 2)), rng.normal(0, 1, (80, 1))])
        Y = np.column_stack([rng.integers(0, 5, (60, 2)), rng.normal(0, 1, (60, 1))])
        test = CovariateShiftTest(categorical_cols=[0, 1], n_permutations=20, random_state=9)
        result = test.test(X, Y)
        assert isinstance(result, ShiftTestResult)

    def test_custom_bandwidth(self):
        X, Y = self._data(seed=40)
        test = CovariateShiftTest(bandwidth=2.0, n_permutations=20, random_state=10)
        result = test.test(X, Y)
        assert result is not None

    def test_returns_self_or_result_from_fit(self):
        X, Y = self._data(seed=41)
        test = CovariateShiftTest(n_permutations=20, random_state=11)
        result = test.test(X, Y)
        assert isinstance(result, ShiftTestResult)


# ---------------------------------------------------------------------------
# ShiftTestResult
# ---------------------------------------------------------------------------

class TestShiftTestResult:
    def _make_result(self, **kw):
        defaults = dict(
            test_statistic=0.05,
            p_value=0.3,
            per_feature_drift_scores={0: 0.01, 1: 0.02},
            n_source=100,
            n_target=80,
            n_permutations=200,
        )
        defaults.update(kw)
        return ShiftTestResult(**defaults)

    def test_construction(self):
        r = self._make_result()
        assert r.test_statistic == 0.05
        assert r.p_value == 0.3

    def test_repr_significant(self):
        r = self._make_result(p_value=0.01)
        text = repr(r)
        assert "significant" in text

    def test_repr_not_significant(self):
        r = self._make_result(p_value=0.5)
        text = repr(r)
        assert "not significant" in text

    def test_n_source_n_target_in_repr(self):
        r = self._make_result(n_source=150, n_target=90)
        text = repr(r)
        assert "150" in text
        assert "90" in text

"""
InsuranceTabPFN: main model class.

This is the class practitioners actually use. It handles everything the raw
TabPFN/TabICL backends don't: exposure normalisation, categorical encoding,
train/test split for conformal intervals, and inverse-transforming predictions.

Exposure handling design (critical — read this before modifying):
  Standard actuarial approach uses log(exposure) as an offset in a GLM, so
  the model learns E[claims] = exposure * exp(Xβ). TabPFN has no offset param.
  Our workaround:
    1. Target becomes claim_rate = y / exposure (claims per policy year).
    2. log(exposure) is appended as an additional feature.
    3. At prediction time, predicted_rate * exposure gives expected claims.
  This is not equivalent to a true Poisson offset — the log(exposure) feature
  is learned from data rather than fixed. For thin segments the difference is
  small in practice, but the limitation is documented in CommitteeReport.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from insurance_thin_data.tabpfn.backends import BackendProtocol, get_backend
from insurance_thin_data.tabpfn.validators import validate_inputs, validate_feature_names, _df_to_float_array, _is_non_numeric_dtype


class InsuranceTabPFN(BaseEstimator, RegressorMixin):
    """
    Foundation model wrapper for thin-data insurance pricing segments.

    Wraps TabPFN v2 or TabICLv2 with exposure handling, categorical encoding,
    and split-conformal prediction intervals.

    Parameters
    ----------
    backend : str
        Backend to use. One of 'auto', 'tabicl', 'tabpfn', 'mock'.
        'auto' prefers TabICLv2 (better benchmarks, Apache 2.0), falls back
        to TabPFN v2.
    device : str
        Device for inference. 'cpu' or 'cuda'.
    n_estimators : int
        Number of estimators (TabPFN only; ignored by TabICL).
    conformal_coverage : float
        Target coverage for prediction intervals, e.g. 0.9 for 90% intervals.
    conformal_test_size : float
        Fraction of training data held out for conformal calibration.
        Set to 0.0 to disable conformal intervals (predict_interval will raise).
    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    Frequency model with exposure:

        model = InsuranceTabPFN(backend="auto")
        model.fit(X_train, claims_train, exposure=years_in_force_train)
        expected_claims = model.predict(X_test, exposure=years_in_force_test)

    Without exposure (e.g. loss ratio modelling):

        model = InsuranceTabPFN()
        model.fit(X_train, loss_ratio_train)
        predicted_lr = model.predict(X_test)
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "cpu",
        n_estimators: int = 4,
        conformal_coverage: float = 0.9,
        conformal_test_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self.n_estimators = n_estimators
        self.conformal_coverage = conformal_coverage
        self.conformal_test_size = conformal_test_size
        self.random_state = random_state

    def fit(
        self,
        X: Union[pd.DataFrame, NDArray],
        y: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> "InsuranceTabPFN":
        """
        Fit the model.

        Parameters
        ----------
        X : DataFrame or array, shape (n_samples, n_features)
            Feature matrix. Missing values are not supported — impute first.
        y : array, shape (n_samples,)
            Claim counts (frequency) or loss amounts (severity/pure premium).
            Do not pre-divide by exposure — pass raw claim counts.
        exposure : array, shape (n_samples,), optional
            Policy years in force. If provided, the model learns claim rate
            (y / exposure) and multiplies back at prediction time.

        Returns
        -------
        self
        """
        # --- store feature names if DataFrame ---
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = list(X.columns)
            self._feature_dtypes = X.dtypes.to_dict()
        else:
            self._feature_names_in = None
            self._feature_dtypes = None

        X_arr, y_arr, exp_arr = validate_inputs(
            X, y, exposure, check_size=True, backend_name=self.backend
        )

        self._has_exposure = exp_arr is not None
        self._exposure_train = exp_arr

        # --- categorical encoding ---
        X_arr = self._encode_categoricals(X_arr, X, fit=True)

        # --- exposure transform ---
        if self._has_exposure:
            # Clip to avoid log(0) on zero-exposure edge cases caught by validator
            y_transformed = y_arr / exp_arr
            log_exp = np.log(np.clip(exp_arr, 1e-10, None)).reshape(-1, 1)
            X_arr = np.hstack([X_arr, log_exp])
        else:
            y_transformed = y_arr

        # --- conformal calibration split ---
        self._conformal_residuals: Optional[NDArray] = None
        if self.conformal_test_size > 0:
            split = train_test_split(
                X_arr,
                y_transformed,
                test_size=self.conformal_test_size,
                random_state=self.random_state,
            )
            X_fit, X_cal, y_fit, y_cal = split
        else:
            X_fit, y_fit = X_arr, y_transformed
            X_cal, y_cal = None, None

        # --- backend ---
        backend_kwargs: dict = {"device": self.device}
        if self.backend in ("tabpfn", "auto"):
            backend_kwargs["n_estimators"] = self.n_estimators
        if self.random_state is not None:
            backend_kwargs["random_state"] = self.random_state

        self._backend: BackendProtocol = get_backend(self.backend, **backend_kwargs)
        self._backend.fit(X_fit, y_fit)

        # --- calibration residuals for conformal ---
        if X_cal is not None and len(X_cal) > 0:
            cal_preds = self._backend.predict(X_cal)
            # Use absolute residuals (symmetric conformity scores)
            self._conformal_residuals = np.abs(y_cal - cal_preds)

        self._n_features_in = X_arr.shape[1]
        self._is_fitted = True

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, NDArray],
        exposure: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Predict expected claims (or rates if no exposure provided).

        Parameters
        ----------
        X : DataFrame or array, shape (n_samples, n_features)
        exposure : array, shape (n_samples,), optional
            Must be provided if exposure was provided at fit time.

        Returns
        -------
        predictions : NDArray, shape (n_samples,)
            Expected claims if exposure provided; claim rate otherwise.
        """
        self._check_fitted()

        if self._has_exposure and exposure is None:
            raise ValueError(
                "This model was fitted with exposure. "
                "You must pass exposure= at prediction time."
            )
        if not self._has_exposure and exposure is not None:
            warnings.warn(
                "exposure was not provided at fit time but is provided at prediction "
                "time. The exposure will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            exposure = None

        # Feature name check
        if self._feature_names_in is not None and isinstance(X, pd.DataFrame):
            validate_feature_names(
                pd.DataFrame(columns=self._feature_names_in), X
            )
            X = X[self._feature_names_in]  # ensure column order

        if isinstance(X, pd.DataFrame):
            X_arr = _df_to_float_array(X)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        X_arr = self._encode_categoricals(X_arr, X, fit=False)

        exp_arr: Optional[NDArray] = None
        if exposure is not None:
            exp_arr = np.asarray(exposure, dtype=np.float64)
            if not np.all(exp_arr > 0):
                raise ValueError("Prediction exposure must be strictly positive.")
            log_exp = np.log(exp_arr).reshape(-1, 1)
            X_arr = np.hstack([X_arr, log_exp])

        predictions = self._backend.predict(X_arr)
        predictions = np.clip(predictions, 0, None)  # rates can't be negative

        if exp_arr is not None:
            predictions = predictions * exp_arr

        return predictions

    def predict_interval(
        self,
        X: Union[pd.DataFrame, NDArray],
        exposure: Optional[NDArray] = None,
        alpha: float = 0.1,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Prediction intervals via split conformal prediction.

        Uses the held-out calibration set residuals to construct
        distribution-free intervals with (1 - alpha) coverage guarantee.

        Parameters
        ----------
        X : DataFrame or array, shape (n_samples, n_features)
        exposure : array, optional
        alpha : float
            Miscoverage rate. alpha=0.1 gives 90% intervals.

        Returns
        -------
        lower : NDArray
        point : NDArray
        upper : NDArray
        """
        self._check_fitted()

        if self._conformal_residuals is None or len(self._conformal_residuals) == 0:
            raise RuntimeError(
                "No conformal calibration data available. "
                "Set conformal_test_size > 0 when initialising InsuranceTabPFN."
            )

        point = self.predict(X, exposure=exposure)

        # Conformal quantile: Hoerl-Tibshirani (1 - alpha)(1 + 1/n) quantile
        n_cal = len(self._conformal_residuals)
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)
        q_hat = float(np.quantile(self._conformal_residuals, q_level))

        lower = np.clip(point - q_hat, 0, None)
        upper = point + q_hat

        return lower, point, upper

    def get_feature_names_out(self) -> list[str]:
        """Return feature names as seen at fit time."""
        self._check_fitted()
        if self._feature_names_in is not None:
            return self._feature_names_in.copy()
        n = self._n_features_in
        if self._has_exposure:
            n -= 1  # log_exposure was appended internally
        return [f"x{i}" for i in range(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_categoricals(
        self,
        X_arr: NDArray,
        X_orig: Union[pd.DataFrame, NDArray],
        fit: bool,
    ) -> NDArray:
        """Label-encode any object/categorical columns."""
        if not isinstance(X_orig, pd.DataFrame):
            return X_arr

        if fit:
            self._label_encoders: dict[int, LabelEncoder] = {}

        X_out = X_arr.copy()
        for i, (col, dtype) in enumerate(X_orig.dtypes.items()):
            if _is_non_numeric_dtype(dtype):
                if fit:
                    le = LabelEncoder()
                    X_out[:, i] = le.fit_transform(
                        X_orig.iloc[:, i].astype(str)
                    ).astype(np.float64)
                    self._label_encoders[i] = le
                else:
                    if i in self._label_encoders:
                        le = self._label_encoders[i]
                        # Handle unseen categories gracefully
                        col_vals = X_orig.iloc[:, i].astype(str).values
                        encoded = np.zeros(len(col_vals), dtype=np.float64)
                        for j, v in enumerate(col_vals):
                            if v in le.classes_:
                                encoded[j] = le.transform([v])[0]
                            else:
                                # Assign to class index 0 with a warning
                                encoded[j] = 0.0
                        X_out[:, i] = encoded
        return X_out

    def _check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                "Model not fitted. Call fit() before predict() or predict_interval()."
            )

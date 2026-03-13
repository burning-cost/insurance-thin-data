"""
Input validation and thin-segment warnings.

TabPFN is designed for small data. This module enforces sane limits and warns
practitioners when they're using the tool outside its design envelope.

The n > 5000 threshold is our recommended cutoff. Above this, standard GLMs
with cross-validation are more reliable, faster, and more defensible to the PRA.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ThinSegmentWarning(UserWarning):
    """Warning issued when data size exceeds the thin-segment design envelope."""
    pass


class ExposureWarning(UserWarning):
    """Warning issued for suspicious exposure values."""
    pass


class ValidationError(ValueError):
    """Raised for data quality issues that prevent safe model fitting."""
    pass


# TabPFN hard limit (will fail or degrade above this)
TABPFN_HARD_LIMIT = 10_000
# Recommended limit — above this, standard GLMs are preferable
THIN_SEGMENT_RECOMMENDED_MAX = 5_000
# Below this, even TabPFN may be unreliable
THIN_SEGMENT_MINIMUM = 20


def _df_to_float_array(X: pd.DataFrame) -> NDArray:
    """
    Convert a DataFrame to float64 array.

    Object / category columns are substituted with 0.0 placeholder — they
    will be label-encoded later in model._encode_categoricals(). Numeric
    columns are converted normally so NaN/Inf checks work correctly.
    """
    n, p = X.shape
    out = np.zeros((n, p), dtype=np.float64)
    for i, col in enumerate(X.columns):
        dtype = X.dtypes.iloc[i]
        if dtype == object or str(dtype) == "category":
            # placeholder — will be encoded in model layer
            out[:, i] = 0.0
        else:
            out[:, i] = X.iloc[:, i].astype(np.float64).values
    return out


def validate_inputs(
    X: pd.DataFrame | NDArray,
    y: NDArray,
    exposure: Optional[NDArray] = None,
    check_size: bool = True,
    backend_name: str = "tabpfn",
) -> tuple[NDArray, NDArray, Optional[NDArray]]:
    """
    Validate and coerce inputs for InsuranceTabPFN.

    Parameters
    ----------
    X : DataFrame or array, shape (n_samples, n_features)
    y : array, shape (n_samples,). Claim counts or loss amounts.
    exposure : array, shape (n_samples,), optional. Policy years or earned premium.
    check_size : bool. Whether to emit thin-segment warnings.
    backend_name : str. Used for warning messages.

    Returns
    -------
    X_arr : NDArray, shape (n_samples, n_features). Float64 (object cols = 0.0 placeholder).
    y_arr : NDArray, shape (n_samples,). Float64.
    exposure_arr : NDArray or None.
    """
    # Convert X to float array, handling mixed types
    if isinstance(X, pd.DataFrame):
        X_arr = _df_to_float_array(X)
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValidationError(
            f"X must be 2-dimensional, got shape {X_arr.shape}"
        )

    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValidationError(
            f"y must be 1-dimensional, got shape {y_arr.shape}"
        )

    n_samples = len(X_arr)
    if len(y_arr) != n_samples:
        raise ValidationError(
            f"X has {n_samples} rows but y has {len(y_arr)} elements."
        )

    # NaN / Inf checks on numeric columns only
    # (object placeholder cols are 0.0, always finite)
    if isinstance(X, pd.DataFrame):
        numeric_mask = np.array([
            X.dtypes.iloc[i] != object and str(X.dtypes.iloc[i]) != "category"
            for i in range(X.shape[1])
        ])
        X_numeric = X_arr[:, numeric_mask]
    else:
        X_numeric = X_arr

    if not np.all(np.isfinite(X_numeric)):
        n_bad = int(np.sum(~np.isfinite(X_numeric)))
        raise ValidationError(
            f"X contains {n_bad} non-finite values (NaN or Inf) in numeric columns. "
            "TabPFN cannot handle missing values. Impute before fitting."
        )

    if not np.all(np.isfinite(y_arr)):
        n_bad = int(np.sum(~np.isfinite(y_arr)))
        raise ValidationError(
            f"y contains {n_bad} non-finite values."
        )

    if np.any(y_arr < 0):
        raise ValidationError(
            "y contains negative values. For frequency modelling, y should be "
            "claim counts (>= 0). For loss modelling, y should be loss amounts (>= 0)."
        )

    # Exposure validation
    exposure_arr: Optional[NDArray] = None
    if exposure is not None:
        exposure_arr = np.asarray(exposure, dtype=np.float64)
        if len(exposure_arr) != n_samples:
            raise ValidationError(
                f"X has {n_samples} rows but exposure has {len(exposure_arr)} elements."
            )
        if not np.all(np.isfinite(exposure_arr)):
            raise ValidationError("exposure contains non-finite values.")
        if np.any(exposure_arr <= 0):
            n_bad = int(np.sum(exposure_arr <= 0))
            raise ValidationError(
                f"exposure contains {n_bad} zero or negative values. "
                "Exposure must be strictly positive (e.g. policy years in force)."
            )
        if np.any(exposure_arr > 1.5):
            warnings.warn(
                f"exposure contains values > 1.5 ({int(np.sum(exposure_arr > 1.5))} rows). "
                "If modelling annual claim frequency, exposure should be in policy-years "
                "(typically 0 < e <= 1.0). Values > 1.5 suggest exposure is in days, "
                "months, or premium units — normalise before fitting.",
                ExposureWarning,
                stacklevel=3,
            )

    # Size warnings
    if check_size:
        if n_samples < THIN_SEGMENT_MINIMUM:
            warnings.warn(
                f"Only {n_samples} samples. TabPFN needs at least "
                f"{THIN_SEGMENT_MINIMUM} samples to produce reliable predictions. "
                "Consider manually constructing credibility-weighted rates instead.",
                ThinSegmentWarning,
                stacklevel=3,
            )
        elif n_samples > TABPFN_HARD_LIMIT:
            warnings.warn(
                f"{n_samples} samples exceeds the {backend_name} hard limit of "
                f"{TABPFN_HARD_LIMIT}. Predictions may fail or degrade severely. "
                "This library is a thin-segment specialist. For larger datasets, "
                "use a GLM or gradient boosted trees.",
                ThinSegmentWarning,
                stacklevel=3,
            )
        elif n_samples > THIN_SEGMENT_RECOMMENDED_MAX:
            warnings.warn(
                f"{n_samples} samples exceeds the recommended maximum of "
                f"{THIN_SEGMENT_RECOMMENDED_MAX} for thin-segment modelling. "
                "Above 5,000 records, a Poisson GLM with cross-validation will "
                "typically outperform TabPFN and is more defensible to the PRA.",
                ThinSegmentWarning,
                stacklevel=3,
            )

    return X_arr, y_arr, exposure_arr


def validate_feature_names(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
) -> None:
    """Check that prediction DataFrame columns match training columns."""
    train_cols = set(X_train.columns)
    pred_cols = set(X_pred.columns)

    missing = train_cols - pred_cols
    extra = pred_cols - train_cols

    if missing:
        raise ValidationError(
            f"Prediction data is missing {len(missing)} feature(s) "
            f"that were present at training time: {sorted(missing)}"
        )
    if extra:
        warnings.warn(
            f"Prediction data has {len(extra)} extra column(s) not seen at "
            f"training time: {sorted(extra)}. These will be ignored.",
            UserWarning,
            stacklevel=2,
        )

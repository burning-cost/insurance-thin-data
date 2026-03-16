"""
Backend abstraction for TabPFN v2 and TabICLv2.

Both backends are optional dependencies. The library works without either
installed — you just can't train or predict. The abstraction exists because
TabICLv2 (INRIA/SODA, Feb 2026, Apache 2.0) now outperforms TabPFN v2.5 on
all TabArena benchmarks, and we expect users to migrate as TabICL matures.

Key design choice: we expose a minimal Protocol rather than inheriting from
sklearn BaseEstimator. The backends are thin shims; the actuarial logic lives
in model.py, not here.
"""

from __future__ import annotations

import warnings
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class BackendNotAvailableError(ImportError):
    """Raised when neither TabPFN nor TabICL is installed."""
    pass


@runtime_checkable
class BackendProtocol(Protocol):
    """Minimal interface that all backends must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable backend name, e.g. 'tabpfn-v2' or 'tabicl-v2'."""
        ...

    def fit(self, X: NDArray, y: NDArray) -> "BackendProtocol":
        """Fit on training data. Returns self."""
        ...

    def predict(self, X: NDArray) -> NDArray:
        """Point predictions. Shape: (n_samples,)."""
        ...

    def predict_quantiles(
        self, X: NDArray, quantiles: list[float]
    ) -> NDArray:
        """Quantile predictions. Shape: (n_samples, n_quantiles).

        Not all backends support this natively. Backends that don't should
        raise NotImplementedError so the caller can fall back to empirical
        quantiles.
        """
        ...


class TabPFNBackend:
    """
    Wraps TabPFN v2 (Prior Labs).

    Uses ModelVersion.V2 (not V2.5) by default — V2 is Apache 2.0 licensed
    and safe for commercial deployment. V2.5 is non-commercial only.

    TabPFN treats regression as Gaussian; it has no Poisson/Gamma native
    support. For insurance frequency we handle this at the model layer by
    transforming the target (claim rate = claims / exposure) before fitting.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_version: str = "v2",
        n_estimators: int = 4,
        random_state: Optional[int] = None,
    ) -> None:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore[import-untyped]
        except ImportError as e:
            raise BackendNotAvailableError(
                "TabPFN is not installed. Run: pip install insurance-thin-data[tabpfn]"
            ) from e

        # V2.5 weights are non-commercial — warn loudly if user requests them
        if model_version == "v2.5":
            warnings.warn(
                "TabPFN v2.5 model weights are non-commercial only (Prior Labs "
                "Non-Commercial License). Do not use in production pricing systems "
                "without a commercial license from sales@priorlabs.ai. "
                "Use model_version='v2' for commercial deployment.",
                UserWarning,
                stacklevel=2,
            )

        from tabpfn import TabPFNRegressor  # noqa: F811

        self._model = TabPFNRegressor(
            device=device,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._model_version = model_version
        self._fitted = False

    @property
    def name(self) -> str:
        return f"tabpfn-{self._model_version}"

    def fit(self, X: NDArray, y: NDArray) -> "TabPFNBackend":
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        if not self._fitted:
            raise RuntimeError("Backend not fitted. Call fit() first.")
        return np.asarray(self._model.predict(X))

    def predict_quantiles(
        self, X: NDArray, quantiles: list[float]
    ) -> NDArray:
        """
        TabPFNRegressor exposes a full predictive distribution. We can
        extract quantiles from it using predict() with output_type parameter
        if available, otherwise fall back to prediction_intervals.
        """
        if not self._fitted:
            raise RuntimeError("Backend not fitted. Call fit() first.")

        try:
            # TabPFN >= 6.x exposes output_type parameter
            results = []
            for q in quantiles:
                # TabPFN returns distribution; use quantile output
                pred = self._model.predict(X, output_type="quantiles", quantiles=[q])
                results.append(pred[:, 0])
            return np.column_stack(results)
        except (TypeError, AttributeError):
            raise NotImplementedError(
                "This version of TabPFN does not support quantile output. "
                "Falling back to empirical quantiles from point predictions."
            )


class TabICLBackend:
    """
    Wraps TabICLv2 (INRIA/SODA, February 2026).

    TabICL (Tabular In-Context Learning) uses a transformer pre-trained on
    real tabular datasets (not synthetic SCMs like TabPFN). Apache 2.0 licensed
    with no commercial restrictions.

    As of March 2026, TabICLv2 outperforms TabPFN v2.5 on all TabArena
    benchmarks. Preferred backend when available.
    """

    def __init__(
        self,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ) -> None:
        try:
            from tabicl import TabICLRegressor  # type: ignore[import-untyped]
        except ImportError as e:
            raise BackendNotAvailableError(
                "TabICL is not installed. Run: pip install insurance-thin-data[tabicl]"
            ) from e

        from tabicl import TabICLRegressor  # noqa: F811

        kwargs: dict = {}
        if random_state is not None:
            kwargs["random_state"] = random_state

        self._model = TabICLRegressor(device=device, **kwargs)
        self._fitted = False

    @property
    def name(self) -> str:
        return "tabicl-v2"

    def fit(self, X: NDArray, y: NDArray) -> "TabICLBackend":
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        if not self._fitted:
            raise RuntimeError("Backend not fitted. Call fit() first.")
        return np.asarray(self._model.predict(X))

    def predict_quantiles(
        self, X: NDArray, quantiles: list[float]
    ) -> NDArray:
        raise NotImplementedError(
            "TabICLv2 does not expose native quantile predictions. "
            "Falling back to empirical quantiles."
        )


class MockBackend:
    """
    Drop-in backend for testing without any ML installation.

    Returns simple mean predictions and linear interpolation for quantiles.
    Never use in production.
    """

    def __init__(self, random_state: Optional[int] = 42, **kwargs) -> None:
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._rng = np.random.default_rng(random_state)
        self._fitted = False

    @property
    def name(self) -> str:
        return "mock"

    def fit(self, X: NDArray, y: NDArray) -> "MockBackend":
        self._mean = float(np.mean(y))
        self._std = float(np.std(y)) or 1.0
        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        if not self._fitted:
            raise RuntimeError("Backend not fitted. Call fit() first.")
        n = len(X)
        noise = self._rng.normal(0, self._std * 0.1, size=n)
        return np.full(n, self._mean) + noise

    def predict_quantiles(
        self, X: NDArray, quantiles: list[float]
    ) -> NDArray:
        if not self._fitted:
            raise RuntimeError("Backend not fitted. Call fit() first.")
        from scipy.stats import norm  # type: ignore[import-untyped]

        results = []
        for q in quantiles:
            results.append(
                np.full(len(X), norm.ppf(q, loc=self._mean, scale=self._std))
            )
        return np.column_stack(results)


def get_backend(backend: str = "auto", **kwargs) -> BackendProtocol:
    """
    Resolve backend name to a concrete backend instance.

    Parameters
    ----------
    backend : str
        One of: 'auto', 'tabicl', 'tabpfn', 'mock'.
        'auto' tries TabICLv2 first (better benchmarks, Apache 2.0),
        falls back to TabPFN v2, raises if neither available.
    **kwargs
        Passed to the backend constructor.

    Returns
    -------
    BackendProtocol
        Concrete backend instance ready for .fit() / .predict().
    """
    if backend == "mock":
        return MockBackend(**kwargs)

    if backend == "tabicl":
        return TabICLBackend(**kwargs)

    if backend == "tabpfn":
        return TabPFNBackend(**kwargs)

    if backend == "auto":
        # Prefer TabICLv2: better benchmarks, Apache 2.0, no commercial risk
        try:
            return TabICLBackend(**kwargs)
        except BackendNotAvailableError:
            pass

        try:
            return TabPFNBackend(**kwargs)
        except BackendNotAvailableError:
            pass

        raise BackendNotAvailableError(
            "No backend available. Install one of:\n"
            "  pip install insurance-thin-data[tabicl]   # TabICLv2 (preferred)\n"
            "  pip install insurance-thin-data[tabpfn]   # TabPFN v2\n"
            "\n"
            "For testing without ML backends: use backend='mock'."
        )

    raise ValueError(
        f"Unknown backend '{backend}'. Choose from: 'auto', 'tabicl', 'tabpfn', 'mock'."
    )

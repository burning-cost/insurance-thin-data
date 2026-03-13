"""
GLMBenchmark: side-by-side comparison of InsuranceTabPFN vs Poisson GLM.

The benchmark is not optional vanity — it's a regulatory necessity. The PRA
expects validation against a benchmark model that uses 'generally accepted
market practice'. A Poisson GLM with log link is that benchmark for UK motor
frequency pricing. Without this comparison, TabPFN results can't go in a
committee paper.

Metrics computed:
  - Normalised Gini coefficient (the industry standard discrimination measure)
  - Poisson deviance (proper scoring rule for frequency)
  - RMSE on claim counts
  - Double-lift chart (actuals vs predicted by decile — the standard QA plot)

Dependencies: statsmodels (optional). If not installed, GLM is unavailable
but BenchmarkResult objects can still be created for TabPFN-only reporting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BenchmarkResult:
    """
    Metrics container for a single model's performance.

    Attributes
    ----------
    model_name : str
    gini : float. Normalised Gini coefficient (0 = random, 1 = perfect).
    poisson_deviance : float. Mean Poisson deviance (lower = better).
    rmse : float. Root mean squared error on claim counts.
    double_lift : DataFrame. Decile lift table (actual vs predicted rate).
    n_samples : int. Number of test samples.
    exposure_total : float. Total exposure in test set.
    """

    model_name: str
    gini: float
    poisson_deviance: float
    rmse: float
    double_lift: pd.DataFrame
    n_samples: int
    exposure_total: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        """Single-row summary for comparison tables."""
        return pd.Series(
            {
                "model": self.model_name,
                "gini": round(self.gini, 4),
                "poisson_deviance": round(self.poisson_deviance, 4),
                "rmse": round(self.rmse, 4),
                "n_test": self.n_samples,
            }
        )


def _gini(y_actual: NDArray, y_predicted: NDArray, exposure: Optional[NDArray] = None) -> float:
    """
    Normalised Gini coefficient.

    Computed as 2 * AUC(Lorenz curve) - 1, using exposure as weights where
    provided. The Lorenz curve plots cumulative actual claims against cumulative
    exposure when policies are sorted by predicted rate (ascending).

    This is the standard actuarial discrimination metric — equivalent to the
    one used in CMP/CIROS models but computed directly rather than via ROC.
    """
    n = len(y_actual)
    if n == 0:
        return 0.0

    if exposure is None:
        exposure = np.ones(n)

    # Sort by predicted rate ascending
    order = np.argsort(y_predicted)
    y_sorted = y_actual[order]
    e_sorted = exposure[order]

    # Cumulative proportions
    cum_exposure = np.cumsum(e_sorted) / np.sum(e_sorted)
    cum_claims = np.cumsum(y_sorted) / np.sum(y_sorted)

    # Insert (0, 0)
    cum_exposure = np.concatenate([[0.0], cum_exposure])
    cum_claims = np.concatenate([[0.0], cum_claims])

    # Area under Lorenz curve via trapezoid rule
    _trapezoid = getattr(np, "trapezoid", np.trapz) if hasattr(np, "trapz") else np.trapezoid
    lorenz_area = float(_trapezoid(cum_claims, cum_exposure))
    # Perfect model area = 0.5; gini = 2 * (lorenz_area - 0.5) ... wait
    # Standard: Gini = 2 * (AUC - 0.5) if AUC is ROC; for Lorenz it's different
    # For Lorenz: Gini = 1 - 2 * lorenz_area would give concentration, but
    # for predictive Gini we want: area between Lorenz and line of equality
    # Normalised Gini = (0.5 - lorenz_area) / 0.5 * 2 = (lorenz_area - 0.5) * 2
    # (if sorted by predicted, well-calibrated model => lorenz_area > 0.5)
    # Lorenz area < 0.5 for good model (low-risk policies first accumulate fewer claims)
    # Normalised Gini = 1 - 2 * lorenz_area
    return float(1.0 - 2.0 * lorenz_area)


def _poisson_deviance(
    y_actual: NDArray,
    y_predicted: NDArray,
    exposure: Optional[NDArray] = None,
) -> float:
    """
    Mean Poisson deviance: D = 2 * mean(y*log(y/yhat) - (y - yhat))
    where y and yhat are claim counts (not rates).
    """
    eps = 1e-10
    y = np.clip(y_actual, eps, None)
    yhat = np.clip(y_predicted, eps, None)

    deviance_i = 2.0 * (y * np.log(y / yhat) - (y_actual - y_predicted))

    if exposure is not None:
        # Weight by exposure
        return float(np.average(deviance_i, weights=exposure))
    return float(np.mean(deviance_i))


def _double_lift(
    y_actual: NDArray,
    y_tabpfn: NDArray,
    y_glm: Optional[NDArray],
    exposure: Optional[NDArray],
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Double-lift table. Policies sorted by TabPFN/GLM ratio, binned into deciles.
    Shows actual rate vs both model predictions per decile.
    """
    n = len(y_actual)
    if exposure is None:
        exposure = np.ones(n)

    rows = []
    if y_glm is not None:
        # Sort by log(tabpfn/glm) ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                (y_glm > 0) & (y_tabpfn > 0),
                np.log(y_tabpfn / y_glm),
                0.0,
            )
        sort_key = ratio
    else:
        sort_key = y_tabpfn

    order = np.argsort(sort_key)
    bins = np.array_split(order, n_deciles)

    for decile_idx, idx in enumerate(bins):
        exp_bin = exposure[idx].sum()
        actual_rate = y_actual[idx].sum() / exp_bin if exp_bin > 0 else 0.0
        tabpfn_rate = y_tabpfn[idx].sum() / exp_bin if exp_bin > 0 else 0.0
        row: dict = {
            "decile": decile_idx + 1,
            "exposure": round(exp_bin, 2),
            "actual_rate": round(actual_rate, 5),
            "tabpfn_rate": round(tabpfn_rate, 5),
        }
        if y_glm is not None:
            glm_rate = y_glm[idx].sum() / exp_bin if exp_bin > 0 else 0.0
            row["glm_rate"] = round(glm_rate, 5)
        rows.append(row)

    return pd.DataFrame(rows)


class GLMBenchmark:
    """
    Fit a Poisson GLM and produce a side-by-side comparison with InsuranceTabPFN.

    The GLM uses log link with formula-style feature handling. No interaction
    terms are added automatically — this is an intentionally naive GLM so the
    comparison is honest. If you have domain-knowledge interactions in your
    GLM spec, add them via the formula parameter.

    Parameters
    ----------
    formula : str, optional
        Patsy formula string, e.g. 'y ~ age + C(region) + mileage'.
        If None, all features are used as linear terms.
    max_iter : int
        Maximum IRLS iterations.

    Examples
    --------
        bench = GLMBenchmark()
        bench.fit(X_train, claims_train, exposure=exposure_train)
        result = bench.compare(
            X_test, claims_test, exposure=exposure_test,
            tabpfn_predictions=tabpfn_model.predict(X_test, exposure_test)
        )
        print(result.to_dataframe())
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        max_iter: int = 100,
    ) -> None:
        self.formula = formula
        self.max_iter = max_iter
        self._glm = None
        self._feature_names: Optional[list[str]] = None

    def fit(
        self,
        X: pd.DataFrame | NDArray,
        y: NDArray,
        exposure: Optional[NDArray] = None,
    ) -> "GLMBenchmark":
        """
        Fit a Poisson GLM.

        Parameters
        ----------
        X : DataFrame or array
        y : array. Claim counts.
        exposure : array, optional. Offsets via log-link exposure offset.
        """
        try:
            import statsmodels.api as sm  # type: ignore[import-untyped]
            import statsmodels.formula.api as smf  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "statsmodels is required for GLMBenchmark. "
                "Run: pip install insurance-tabpfn[glm]"
            ) from e

        y_arr = np.asarray(y, dtype=np.float64)

        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self._feature_names = list(X.columns)
        else:
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            self._feature_names = list(df.columns)

        df["_y"] = y_arr
        if exposure is not None:
            df["_log_exposure"] = np.log(np.clip(np.asarray(exposure), 1e-10, None))
            offset_col = "_log_exposure"
        else:
            offset_col = None

        # Sanitise column names for patsy (no spaces, no special chars)
        rename = {c: c.replace(" ", "_").replace("-", "_") for c in df.columns}
        df = df.rename(columns=rename)
        self._feature_names = [rename.get(f, f) for f in self._feature_names]

        if self.formula is not None:
            formula_str = self.formula
        else:
            safe_features = [
                f for f in self._feature_names if f not in ("_y", "_log_exposure")
            ]
            formula_str = "_y ~ " + " + ".join(safe_features)

        kwargs: dict = {"family": sm.families.Poisson()}
        if offset_col and offset_col in df.columns:
            kwargs["offset"] = df[offset_col].values

        result = smf.glm(formula=formula_str, data=df, **kwargs).fit(disp=False, maxiter=self.max_iter)
        self._glm = result
        self._df_columns = list(df.columns)
        self._rename_map = rename
        self._offset_col = offset_col
        self._fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame | NDArray,
        exposure: Optional[NDArray] = None,
    ) -> NDArray:
        """Predict expected claim counts from fitted GLM."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("GLMBenchmark not fitted.")

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(
                X, columns=[f"x{i}" for i in range(X.shape[1])]
            )

        # Apply same rename
        df = df.rename(columns=self._rename_map)

        if exposure is not None and self._offset_col:
            df[self._offset_col] = np.log(
                np.clip(np.asarray(exposure), 1e-10, None)
            )
            preds = self._glm.predict(df, offset=df[self._offset_col].values)
        else:
            preds = self._glm.predict(df)

        return np.asarray(preds, dtype=np.float64)

    def compare(
        self,
        X_test: pd.DataFrame | NDArray,
        y_test: NDArray,
        tabpfn_predictions: NDArray,
        exposure_test: Optional[NDArray] = None,
        n_deciles: int = 10,
    ) -> "ComparisonResult":
        """
        Compare TabPFN predictions against the GLM on the same test set.

        Parameters
        ----------
        X_test : test features
        y_test : actual claim counts
        tabpfn_predictions : array of predicted claim counts from InsuranceTabPFN
        exposure_test : optional exposure for the test set
        n_deciles : number of deciles for double-lift chart

        Returns
        -------
        ComparisonResult
        """
        y_arr = np.asarray(y_test, dtype=np.float64)
        tfpfn_arr = np.asarray(tabpfn_predictions, dtype=np.float64)

        exp_arr: Optional[NDArray] = None
        if exposure_test is not None:
            exp_arr = np.asarray(exposure_test, dtype=np.float64)

        # GLM predictions
        if getattr(self, "_fitted", False):
            glm_preds = self.predict(X_test, exposure_test)
        else:
            glm_preds = None

        # TabPFN metrics
        tabpfn_result = BenchmarkResult(
            model_name="InsuranceTabPFN",
            gini=_gini(y_arr, tfpfn_arr, exp_arr),
            poisson_deviance=_poisson_deviance(y_arr, tfpfn_arr, exp_arr),
            rmse=float(np.sqrt(np.mean((y_arr - tfpfn_arr) ** 2))),
            double_lift=_double_lift(y_arr, tfpfn_arr, glm_preds, exp_arr, n_deciles),
            n_samples=len(y_arr),
            exposure_total=float(exp_arr.sum()) if exp_arr is not None else float(len(y_arr)),
        )

        # GLM metrics
        glm_result: Optional[BenchmarkResult] = None
        if glm_preds is not None:
            glm_result = BenchmarkResult(
                model_name="Poisson GLM",
                gini=_gini(y_arr, glm_preds, exp_arr),
                poisson_deviance=_poisson_deviance(y_arr, glm_preds, exp_arr),
                rmse=float(np.sqrt(np.mean((y_arr - glm_preds) ** 2))),
                double_lift=_double_lift(
                    y_arr, tfpfn_arr, glm_preds, exp_arr, n_deciles
                ),
                n_samples=len(y_arr),
                exposure_total=float(exp_arr.sum()) if exp_arr is not None else float(len(y_arr)),
            )

        return ComparisonResult(tabpfn=tabpfn_result, glm=glm_result)


@dataclass
class ComparisonResult:
    """Container for side-by-side model comparison."""

    tabpfn: BenchmarkResult
    glm: Optional[BenchmarkResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Summary table with one row per model."""
        rows = [self.tabpfn.to_series()]
        if self.glm is not None:
            rows.append(self.glm.to_series())
        return pd.DataFrame(rows).reset_index(drop=True)

    def winner(self) -> str:
        """
        Which model has higher Gini? Returns model name.
        Gini is the primary discrimination metric used in UK pricing submissions.
        """
        if self.glm is None:
            return self.tabpfn.model_name
        if self.tabpfn.gini >= self.glm.gini:
            return self.tabpfn.model_name
        return self.glm.model_name

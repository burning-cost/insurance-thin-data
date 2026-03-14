"""
PDP-based pseudo-relativities extraction.

Why PDP not SHAP: TabPFN and TabICL are in-context learning models. They have
no gradients at inference time — the 'model' is the entire training set passed
as context, not a set of learned weights. Gradient-based attribution (SHAP,
integrated gradients) is undefined. Permutation importance is defined but does
not give directional relativities.

Partial dependence plots (PDPs) give us what pricing teams actually need:
"what does the model say about a 30-year-old vs a 40-year-old, all else equal?"
That's a relativity, computed by marginalising over the joint distribution of
other features.

The output format deliberately mirrors shap-relativities (our SHAP library)
so committee papers can show both approaches side by side.

Reference:
  Friedman, J.H. (2001). Greedy function approximation: A gradient boosting
  machine. Annals of Statistics, 29(5):1189-1232. (Section 8.1)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class RelativitiesExtractor:
    """
    Extract pseudo-relativities from a fitted InsuranceTabPFN model.

    Uses partial dependence (PDP) to estimate the marginal effect of each
    feature on predicted claim rate, expressed as a relativity table (factor
    relative to the base level / overall mean).

    Parameters
    ----------
    model : fitted InsuranceTabPFN
        Must have been fitted before passing here.
    n_grid_points : int
        Resolution for continuous features. 20 is sufficient for relativities
        tables; 100 gives smoother PDPs for charts.
    n_sample_rows : int
        Background sample size for marginalisation. Larger = more accurate
        but slower. 200 is a reasonable default for <5000-row datasets.
    random_state : int, optional

    Examples
    --------
        extractor = RelativitiesExtractor(model, n_grid_points=20)
        relats = extractor.extract(X_train, feature="age")
        print(relats)
        # Returns DataFrame: age | mean_prediction | relativity
    """

    def __init__(
        self,
        model,
        n_grid_points: int = 20,
        n_sample_rows: int = 200,
        random_state: Optional[int] = None,
    ) -> None:
        self.model = model
        self.n_grid_points = n_grid_points
        self.n_sample_rows = n_sample_rows
        self.random_state = random_state

    def extract(
        self,
        X: Union[pd.DataFrame, NDArray],
        feature: Union[str, int],
        exposure: Optional[NDArray] = None,
    ) -> pd.DataFrame:
        """
        Compute PDP-based relativity table for a single feature.

        Parameters
        ----------
        X : background data used for marginalisation.
        feature : column name (if X is DataFrame) or integer index.
        exposure : background exposure for rate prediction. If None, uses
                   uniform exposure = 1.0.

        Returns
        -------
        DataFrame with columns:
            - feature_value: grid of feature values
            - mean_prediction: average predicted rate at this value
            - relativity: mean_prediction / grand_mean (i.e. base = 1.0)
        """
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            if isinstance(feature, str):
                feat_idx = col_names.index(feature)
                feat_name = feature
            else:
                feat_idx = feature
                feat_name = col_names[feat_idx] if feat_idx < len(col_names) else f"x{feat_idx}"
            X_bg = X.copy()
        else:
            X_bg = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            feat_idx = feature if isinstance(feature, int) else int(feature)
            feat_name = f"x{feat_idx}"

        rng = np.random.default_rng(self.random_state)
        n_bg = min(self.n_sample_rows, len(X_bg))
        bg_idx = rng.choice(len(X_bg), size=n_bg, replace=False)
        X_sample = X_bg.iloc[bg_idx].reset_index(drop=True)

        if exposure is not None:
            exp_sample = np.asarray(exposure)[bg_idx]
        else:
            exp_sample = np.ones(n_bg)

        # Build grid for this feature.
        # Use pd.api.types to handle all non-numeric dtypes robustly —
        # object, StringDtype (pandas 2.0+), CategoricalDtype all count as
        # categorical for relativity purposes.
        col = X_bg.iloc[:, feat_idx]
        if not pd.api.types.is_numeric_dtype(col):
            grid_values = list(col.unique())
            is_categorical = True
        else:
            col_numeric = col.astype(float)
            grid_values = list(np.unique(
                np.percentile(
                    col_numeric.dropna(),
                    np.linspace(0, 100, self.n_grid_points),
                )
            ))
            is_categorical = False

        # For each grid value, substitute into background data and predict
        mean_preds = []
        for val in grid_values:
            X_temp = X_sample.copy()
            X_temp.iloc[:, feat_idx] = val

            # No exposure at prediction time — we want rates, not counts
            # Multiply by mean background exposure for correct units
            # If model was fitted with exposure, pass unit exposure to get pure rates
            _has_exp = getattr(self.model, "_has_exposure", False)
            if _has_exp:
                _unit_exp = np.ones(len(X_temp))
                raw_preds = self.model.predict(X_temp, exposure=_unit_exp)
            else:
                raw_preds = self.model.predict(X_temp)
            mean_preds.append(float(np.mean(raw_preds)))

        mean_preds_arr = np.array(mean_preds)
        grand_mean = float(np.mean(mean_preds_arr))

        if grand_mean == 0:
            relativities = np.ones(len(mean_preds_arr))
        else:
            relativities = mean_preds_arr / grand_mean

        return pd.DataFrame(
            {
                "feature": feat_name,
                "feature_value": grid_values,
                "mean_prediction": mean_preds_arr.round(6),
                "relativity": relativities.round(4),
            }
        )

    def extract_all(
        self,
        X: Union[pd.DataFrame, NDArray],
        exposure: Optional[NDArray] = None,
        features: Optional[list] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Extract relativities for all features (or a specified subset).

        Returns
        -------
        dict mapping feature name -> relativity DataFrame
        """
        if isinstance(X, pd.DataFrame):
            all_features = list(X.columns)
        else:
            all_features = [f"x{i}" for i in range(np.asarray(X).shape[1])]

        if features is not None:
            target_features = features
        else:
            target_features = all_features

        return {
            feat: self.extract(X, feat, exposure)
            for feat in target_features
        }

    def to_factor_table(
        self,
        X: Union[pd.DataFrame, NDArray],
        exposure: Optional[NDArray] = None,
        features: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Combine all relativities into a single long-format factor table.

        This format is compatible with the shap-relativities output so both
        can be compared in a committee paper.

        Columns: feature | feature_value | relativity
        """
        tables = self.extract_all(X, exposure, features)
        return pd.concat(
            [df[["feature", "feature_value", "relativity"]] for df in tables.values()],
            ignore_index=True,
        )

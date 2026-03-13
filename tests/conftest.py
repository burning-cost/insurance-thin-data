"""
Shared fixtures for insurance-thin-data tests.

Merged from insurance-tabpfn and insurance-transfer conftest files.
All fixtures use MockBackend — no TabPFN/TabICL installation required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_df(rng):
    """150-row DataFrame with mixed-type features and exposure."""
    n = 150
    return pd.DataFrame(
        {
            "age": rng.integers(17, 80, size=n).astype(float),
            "region": rng.choice(["North", "South", "London", "Midlands"], size=n),
            "mileage": rng.uniform(1000, 30000, size=n),
            "vehicle_age": rng.integers(0, 15, size=n).astype(float),
        }
    )


@pytest.fixture
def small_y(rng, small_df):
    """Claim counts (roughly Poisson)."""
    n = len(small_df)
    return rng.poisson(lam=0.08 * (small_df["age"].values / 40), size=n).astype(float)


@pytest.fixture
def small_exposure(rng, small_df):
    """Policy years in force (between 0.1 and 1.0)."""
    return rng.uniform(0.1, 1.0, size=len(small_df))


@pytest.fixture
def fitted_model(small_df, small_y, small_exposure):
    """Pre-fitted InsuranceTabPFN (mock backend)."""
    from insurance_thin_data.tabpfn import InsuranceTabPFN

    model = InsuranceTabPFN(backend="mock", random_state=42)
    model.fit(small_df, small_y, exposure=small_exposure)
    return model


@pytest.fixture
def fitted_model_no_exposure(small_df, small_y):
    """Pre-fitted InsuranceTabPFN without exposure (loss ratio style)."""
    from insurance_thin_data.tabpfn import InsuranceTabPFN

    loss_ratio = small_y / 0.5  # synthetic loss ratio
    model = InsuranceTabPFN(backend="mock", random_state=42)
    model.fit(small_df, loss_ratio)
    return model

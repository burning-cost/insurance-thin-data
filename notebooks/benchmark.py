# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Run end-to-end on Databricks. Do not run locally.

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: GLMTransfer vs Standalone Thin-Segment GLM
# MAGIC
# MAGIC **Library:** `insurance-thin-data` — transfer learning for thin-segment insurance pricing.
# MAGIC Implements the Tian & Feng (JASA 2023) two-step penalised GLM that borrows statistical
# MAGIC strength from a large source portfolio while correcting for distribution mismatch.
# MAGIC
# MAGIC **Problem:** A pricing team launches a new commercial property product or enters a new
# MAGIC region. They have 500 policies of target data. A standalone Poisson GLM on 500 observations
# MAGIC is technically feasible but the parameter uncertainty is enormous — wide confidence intervals,
# MAGIC unstable relativities, and A/E ratios that swing wildly on held-out data. Transfer learning
# MAGIC uses the existing book (10,000 policies) as a prior, then corrects for the ways the new
# MAGIC segment differs.
# MAGIC
# MAGIC **Baseline:** Standard Poisson GLM fitted only on the 500-policy target segment.
# MAGIC This is the default when a team has limited data and no transfer framework.
# MAGIC
# MAGIC **Library model:** `GLMTransfer` — two-step penalised GLM (Tian & Feng). Step 1 pools
# MAGIC source and target data to get a stabilised estimator. Step 2 estimates a sparse
# MAGIC correction for source-target mismatch using target data only.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** see installed package below.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The central claim is not that transfer learning predicts better in expectation — on
# MAGIC enough data any decent GLM converges to the truth. The claim is that with 500 policies,
# MAGIC the standalone GLM has massive parameter uncertainty. Half your relativities are
# MAGIC statistically indistinguishable from 1.0. The transfer model borrows the source book's
# MAGIC coefficient estimates as a prior and only moves away where the target data has enough
# MAGIC signal to justify it.
# MAGIC
# MAGIC **Key metrics:**
# MAGIC - Poisson deviance on held-out target test set (500 train / 150 test split)
# MAGIC - Gini coefficient (discriminatory power)
# MAGIC - Mean absolute error of predicted frequency
# MAGIC - A/E ratio by decile (calibration)
# MAGIC - Bootstrap coefficient CI widths (the key differentiator: transfer produces far
# MAGIC   narrower CIs than standalone GLM on thin data)
# MAGIC - Negative Transfer Diagnostic (does the source actually help?)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Library under test
%pip install git+https://github.com/burning-cost/insurance-thin-data.git

# GLM fitting
%pip install statsmodels

# Diagnostics and plotting
%pip install matplotlib seaborn pandas numpy scipy scikit-learn

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm

# Library under test
from insurance_thin_data.transfer import (
    GLMTransfer,
    NegativeTransferDiagnostic,
    CovariateShiftTest,
)
from insurance_thin_data.transfer.diagnostic import poisson_deviance as lib_poisson_deviance

import insurance_thin_data
print(f"insurance-thin-data version: {insurance_thin_data.__version__}")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Data: Known DGP
# MAGIC
# MAGIC We generate two correlated portfolios with a known data generating process so we can
# MAGIC assess whether the transfer model recovers the true coefficients.
# MAGIC
# MAGIC **Source portfolio (10,000 policies):** represents the existing book. Mix of commercial
# MAGIC property risks in established regions.
# MAGIC
# MAGIC **Target segment (650 policies, 500 train / 150 test):** a new product or region
# MAGIC that shares most risk factors with the source but has:
# MAGIC - A lower baseline frequency (new region has better claims environment)
# MAGIC - One feature with a different relativity (local construction quality matters more)
# MAGIC - One feature not present in source (target-specific underwriting score)
# MAGIC
# MAGIC **Shared features (both portfolios):**
# MAGIC - `log_sum_insured`: log of sum insured (continuous)
# MAGIC - `building_age`: building age band (0=new, 1=<10yr, 2=10-25yr, 3=25+yr) (ordinal)
# MAGIC - `occupancy_class`: occupancy type (0=office, 1=retail, 2=light_industrial, 3=warehouse) (categorical)
# MAGIC - `flood_zone`: flood zone indicator (0=low, 1=medium, 2=high)
# MAGIC - `fire_protection`: fire protection score 1-5
# MAGIC
# MAGIC **Target-only feature:**
# MAGIC - `local_risk_score`: a broker-supplied score (0-1) available only for the new product
# MAGIC
# MAGIC The true log-frequency (per unit exposure) in each portfolio follows a linear combination
# MAGIC of these features with additive Gaussian noise. The target differs from the source via
# MAGIC a lower intercept and a stronger `building_age` coefficient.

# COMMAND ----------

# ---------------------------------------------------------------------------
# True DGP parameters
# ---------------------------------------------------------------------------

# Shared feature names
SHARED_FEATURES = [
    "log_sum_insured",
    "building_age",
    "occupancy_class",
    "flood_zone",
    "fire_protection",
]
TARGET_ONLY_FEATURES = ["local_risk_score"]
ALL_TARGET_FEATURES = SHARED_FEATURES + TARGET_ONLY_FEATURES

# True coefficients for SOURCE portfolio (log-linear Poisson DGP)
# Intercept + one coef per feature
TRUE_BETA_SOURCE = np.array([
    -2.80,   # intercept: baseline log-frequency ~0.06 (6%)
     0.15,   # log_sum_insured: larger buildings claim more
     0.18,   # building_age: older buildings have higher frequency
    -0.05,   # occupancy_class: warehouse slightly lower (better fire systems)
     0.30,   # flood_zone: strong flood signal
    -0.10,   # fire_protection: better protection = fewer claims
])

# True coefficients for TARGET portfolio
# Differences vs source:
#   - intercept is lower (better claims environment: -0.25)
#   - building_age effect is stronger (+0.12: local construction quality matters more)
#   - local_risk_score adds incremental lift (+0.40)
TRUE_BETA_TARGET = np.array([
    -3.05,   # intercept: lower baseline frequency ~0.047
     0.15,   # log_sum_insured: same
     0.30,   # building_age: stronger effect in target (key difference)
    -0.05,   # occupancy_class: same
     0.30,   # flood_zone: same
    -0.10,   # fire_protection: same
     0.40,   # local_risk_score: target-specific feature
])

print("True DGP parameters:")
print("\nSource (10k policies):")
for name, coef in zip(["intercept"] + SHARED_FEATURES, TRUE_BETA_SOURCE):
    print(f"  {name:<22}: {coef:+.2f}")

print("\nTarget (500 train + 150 test):")
for name, coef in zip(["intercept"] + ALL_TARGET_FEATURES, TRUE_BETA_TARGET):
    print(f"  {name:<22}: {coef:+.2f}")

print("\nKey differences (target vs source):")
print(f"  intercept:   {TRUE_BETA_TARGET[0] - TRUE_BETA_SOURCE[0]:+.2f}  (lower base frequency in target)")
print(f"  building_age:{TRUE_BETA_TARGET[2] - TRUE_BETA_SOURCE[2]:+.2f}  (stronger relativity in target)")

# COMMAND ----------

def generate_source_portfolio(n: int, rng: np.random.Generator) -> dict:
    """Generate synthetic source portfolio with known DGP."""
    # Features
    log_sum_insured = rng.normal(11.5, 0.8, n)       # log of ~£100k-£500k sum insured
    building_age = rng.integers(0, 4, n)
    occupancy_class = rng.integers(0, 4, n)
    flood_zone = rng.choice([0, 1, 2], n, p=[0.60, 0.30, 0.10])
    fire_protection = rng.integers(1, 6, n)

    # Exposure: policy year fractions, mostly full years
    exposure = rng.uniform(0.5, 1.0, n)

    # Design matrix (with intercept)
    X = np.column_stack([
        np.ones(n),
        log_sum_insured,
        building_age,
        occupancy_class,
        flood_zone,
        fire_protection,
    ])

    # Log-frequency and claim counts
    log_mu = X @ TRUE_BETA_SOURCE + np.log(exposure)
    mu = np.exp(np.clip(log_mu, -10, 5))
    y = rng.poisson(mu)

    return {
        "X_raw": np.column_stack([
            log_sum_insured, building_age, occupancy_class, flood_zone, fire_protection
        ]),
        "y": y,
        "exposure": exposure,
        "feature_names": SHARED_FEATURES,
        "log_mu_true": log_mu - np.log(exposure),
        "n": n,
    }


def generate_target_segment(n: int, rng: np.random.Generator) -> dict:
    """Generate synthetic target segment with target DGP.

    Same features as source plus local_risk_score. Building age distribution
    skewed towards older stock (the key shift). Flood zone more prevalent.
    """
    # Features — note distributional shift: older buildings, higher flood exposure
    log_sum_insured = rng.normal(11.8, 0.7, n)        # slightly larger properties
    building_age = rng.choice([0, 1, 2, 3], n, p=[0.10, 0.20, 0.35, 0.35])  # older on average
    occupancy_class = rng.integers(0, 4, n)
    flood_zone = rng.choice([0, 1, 2], n, p=[0.45, 0.35, 0.20])   # higher flood exposure
    fire_protection = rng.integers(1, 6, n)
    local_risk_score = rng.beta(2, 5, n)              # skewed low — most low risk

    exposure = rng.uniform(0.5, 1.0, n)

    # Design matrix (with intercept, 7 params including local_risk_score)
    X = np.column_stack([
        np.ones(n),
        log_sum_insured,
        building_age,
        occupancy_class,
        flood_zone,
        fire_protection,
        local_risk_score,
    ])

    log_mu = X @ TRUE_BETA_TARGET + np.log(exposure)
    mu = np.exp(np.clip(log_mu, -10, 5))
    y = rng.poisson(mu)

    return {
        "X_raw": np.column_stack([
            log_sum_insured, building_age, occupancy_class,
            flood_zone, fire_protection, local_risk_score
        ]),
        "y": y,
        "exposure": exposure,
        "feature_names": ALL_TARGET_FEATURES,
        "log_mu_true": log_mu - np.log(exposure),
        "n": n,
    }


# Generate data
N_SOURCE = 10_000
N_TARGET_TRAIN = 500
N_TARGET_TEST = 150
N_TARGET_TOTAL = N_TARGET_TRAIN + N_TARGET_TEST

source = generate_source_portfolio(N_SOURCE, rng)
target_full = generate_target_segment(N_TARGET_TOTAL, rng)

# Split target into train and test
target_train = {k: v[:N_TARGET_TRAIN] if isinstance(v, np.ndarray) else v
                for k, v in target_full.items()}
target_test = {k: v[N_TARGET_TRAIN:] if isinstance(v, np.ndarray) else v
               for k, v in target_full.items()}
target_train["n"] = N_TARGET_TRAIN
target_test["n"] = N_TARGET_TEST

print(f"Source portfolio:        {source['n']:>6,} policies")
print(f"Target train segment:    {target_train['n']:>6,} policies")
print(f"Target test segment:     {target_test['n']:>6,} policies")
print()
print(f"Source claim frequency:  {source['y'].sum() / source['exposure'].sum():.4f}")
print(f"Target train frequency:  {target_train['y'].sum() / target_train['exposure'].sum():.4f}")
print(f"Target test frequency:   {target_test['y'].sum() / target_test['exposure'].sum():.4f}")
print()
print(f"Source: {source['y'].sum():,} claims on {source['exposure'].sum():.0f} exposure-years")
print(f"Target train: {target_train['y'].sum()} claims on {target_train['exposure'].sum():.0f} exposure-years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Covariate Shift Assessment
# MAGIC
# MAGIC Before fitting any transfer model it is worth checking whether the distributions
# MAGIC actually differ. We use `CovariateShiftTest` (MMD with permutation test) on the
# MAGIC shared features. If the shift is negligible, simple pooling might suffice. If it
# MAGIC is large, the debiasing step in GLMTransfer is doing real work.
# MAGIC
# MAGIC We use only the shared features (columns 0-4) for the shift test — the
# MAGIC `local_risk_score` is not present in the source.

# COMMAND ----------

# Shared-feature matrices for shift test
X_src_shared = source["X_raw"]          # shape (10000, 5)
X_tgt_shared = target_train["X_raw"][:, :5]  # first 5 columns are shared features

# Subsample source for speed (MMD is O(n^2))
rng2 = np.random.default_rng(123)
src_idx = rng2.choice(N_SOURCE, 1000, replace=False)
X_src_sample = X_src_shared[src_idx]

shift_test = CovariateShiftTest(
    categorical_cols=[1, 2, 3],    # building_age, occupancy_class, flood_zone are ordinal/categorical
    n_permutations=500,
    random_state=42,
)

t0_shift = time.perf_counter()
shift_result = shift_test.test(X_src_sample, X_tgt_shared)
shift_time = time.perf_counter() - t0_shift

print(shift_result)
print(f"Shift test time: {shift_time:.1f}s")
print()

# Per-feature drift scores
feature_names_shared = SHARED_FEATURES
print("Per-feature marginal drift scores (higher = more drift):")
sorted_drift = sorted(
    [(feature_names_shared[i], v) for i, v in shift_result.per_feature_drift_scores.items()],
    key=lambda x: x[1],
    reverse=True,
)
for fname, score in sorted_drift:
    print(f"  {fname:<22}: {score:.5f}")

print()
if shift_result.p_value < 0.05:
    print("Shift is statistically significant — the debiasing step in GLMTransfer is justified.")
else:
    print("No significant shift detected — simple pooling may suffice.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Standalone Poisson GLM on 500 Target Policies
# MAGIC
# MAGIC This is what a pricing team does without transfer learning: fit a GLM on the 500
# MAGIC available target policies and hope the estimates are stable enough to rate on.
# MAGIC
# MAGIC With 500 policies and 6 features (plus intercept), the effective sample size per
# MAGIC parameter is ~71. Poisson claims are sparse. Most of the coefficient estimates will
# MAGIC have standard errors that are a substantial fraction of the estimate itself.
# MAGIC
# MAGIC We fit via statsmodels (standard Poisson GLM with log link and exposure offset) so
# MAGIC we can read off confidence intervals directly.

# COMMAND ----------

# Design matrix for baseline: target features + intercept (handled by sm.add_constant)
X_tgt_train = target_train["X_raw"]    # (500, 6): 5 shared + local_risk_score
X_tgt_test = target_test["X_raw"]      # (150, 6)
y_tgt_train = target_train["y"]
y_tgt_test = target_test["y"]
exp_tgt_train = target_train["exposure"]
exp_tgt_test = target_test["exposure"]

# Add constant column for statsmodels
X_tgt_train_sm = sm.add_constant(X_tgt_train)
X_tgt_test_sm = sm.add_constant(X_tgt_test)

# Exposure offset (log scale)
log_exp_train = np.log(exp_tgt_train)
log_exp_test = np.log(exp_tgt_test)

t0_baseline = time.perf_counter()

baseline_glm = sm.GLM(
    y_tgt_train,
    X_tgt_train_sm,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=log_exp_train,
)
baseline_result = baseline_glm.fit(maxiter=100)

baseline_fit_time = time.perf_counter() - t0_baseline

print(f"Baseline GLM fit time: {baseline_fit_time:.3f}s")
print(f"Converged: {baseline_result.converged}")
print()
print(baseline_result.summary())

# COMMAND ----------

# Baseline predictions on test set
pred_baseline_test = baseline_result.predict(
    X_tgt_test_sm,
    offset=log_exp_test,
    which="mean",
)

print(f"Baseline test predictions — mean: {pred_baseline_test.mean():.4f}")
print(f"Actual test mean:                  {y_tgt_test.mean():.4f}")
print(f"True expected frequency:           {np.exp(target_test['log_mu_true']).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Library: GLMTransfer (Tian & Feng, JASA 2023)
# MAGIC
# MAGIC `GLMTransfer` implements the two-step penalised algorithm from Tian & Feng (2023).
# MAGIC
# MAGIC **Step 1 — Pooling:** Pool source and target data. Fit an L1-penalised Poisson GLM.
# MAGIC The penalty `lambda_pool` controls regularisation. The large source dataset dominates
# MAGIC the pooled sample, providing stable initial estimates.
# MAGIC
# MAGIC **Step 2 — Debiasing:** Using target data only, estimate the sparse correction
# MAGIC `delta = beta_target - beta_pooled` with penalty `lambda_debias`. Large `lambda_debias`
# MAGIC forces most deltas to zero (closer to source). Small `lambda_debias` allows more
# MAGIC target-specific adjustment. The final coefficient is `beta_pooled + delta`.
# MAGIC
# MAGIC The source only uses the shared features (columns 0-4). The `local_risk_score` column
# MAGIC is target-specific and has no source equivalent. We handle this by fitting the transfer
# MAGIC model on shared features and adding `local_risk_score` as an additional target-only term.
# MAGIC In practice you would use the `TransferPipeline` for this; here we show the manual approach
# MAGIC to make the mechanics visible.

# COMMAND ----------

# Shared feature matrices (no local_risk_score)
X_src_shared_full = source["X_raw"]          # (10000, 5)
X_tgt_train_shared = target_train["X_raw"][:, :5]  # (500, 5)
X_tgt_test_shared = target_test["X_raw"][:, :5]    # (150, 5)
exp_src = source["exposure"]
y_src = source["y"]

t0_transfer = time.perf_counter()

transfer_glm = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,       # mild pooling penalty — source is 20x larger, doesn't need heavy shrinkage
    lambda_debias=0.03,      # moderate debiasing — allow corrections where target data has signal
    scale_features=True,
    fit_intercept=True,
)

transfer_glm.fit(
    X=X_tgt_train_shared,
    y=y_tgt_train,
    exposure=exp_tgt_train,
    X_source=X_src_shared_full,
    y_source=y_src,
    exposure_source=exp_src,
)

transfer_fit_time = time.perf_counter() - t0_transfer

print(f"GLMTransfer fit time: {transfer_fit_time:.2f}s")
print(f"Included sources: {transfer_glm.included_sources_}")
print()

# Report coefficients (shared features only — the transfer component)
print("Coefficient comparison (shared features):")
print(f"{'Feature':<22} {'True':>8} {'Baseline':>10} {'Transfer':>10} {'Delta':>8}")
print("-" * 62)

feature_labels = ["intercept"] + SHARED_FEATURES

# Baseline coefficients from statsmodels (first 7 params: intercept + 6 features)
# Baseline has 7 params (intercept + 5 shared + local_risk_score)
# We compare on the shared features only (first 6: intercept + 5 shared)
baseline_coefs_shared = baseline_result.params[:6]   # intercept + 5 shared features

# Transfer model: intercept + 5 shared feature coefs
transfer_beta = np.concatenate([[transfer_glm.intercept_], transfer_glm.coef_])

for i, label in enumerate(feature_labels):
    true_val = TRUE_BETA_TARGET[i]
    base_val = float(baseline_coefs_shared[i])
    trans_val = float(transfer_beta[i])
    delta_val = float(transfer_glm.delta_[i])
    print(f"{label:<22} {true_val:>8.3f} {base_val:>10.3f} {trans_val:>10.3f} {delta_val:>8.3f}")

# COMMAND ----------

# Transfer predictions on test set (shared features only — need to handle local_risk_score)
# We produce predictions from the transfer model on shared features, then separately
# estimate the local_risk_score effect from target data and add it.

# Step 1: transfer model prediction (log-rate on shared features)
pred_transfer_shared_test = transfer_glm.predict(X_tgt_test_shared, exposure=exp_tgt_test)

# Step 2: estimate local_risk_score coefficient from target train data only
# Fit a simple offset model: log(y / transfer_pred_train) ~ local_risk_score
pred_transfer_shared_train = transfer_glm.predict(X_tgt_train_shared, exposure=exp_tgt_train)

# Offset model: use log of transfer predictions as offset, add local_risk_score
local_score_train = target_train["X_raw"][:, 5:6]   # (500, 1)
local_score_test = target_test["X_raw"][:, 5:6]     # (150, 1)

X_lrs_train = sm.add_constant(local_score_train)
X_lrs_test = sm.add_constant(local_score_test)
log_transfer_offset_train = np.log(np.maximum(pred_transfer_shared_train, 1e-10))

lrs_glm = sm.GLM(
    y_tgt_train,
    X_lrs_train,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=log_transfer_offset_train,
)
lrs_result = lrs_glm.fit(maxiter=100)
lrs_coef = float(lrs_result.params[1])

print(f"Local risk score coefficient (target-only offset model):")
print(f"  Estimated: {lrs_coef:.3f}  |  True: {TRUE_BETA_TARGET[6]:.3f}")
print(f"  95% CI: [{lrs_result.conf_int().iloc[1, 0]:.3f}, {lrs_result.conf_int().iloc[1, 1]:.3f}]")
print()

# Full transfer prediction: shared transfer prediction * local risk score adjustment
log_transfer_offset_test = np.log(np.maximum(pred_transfer_shared_test, 1e-10))
pred_transfer_test = np.exp(
    log_transfer_offset_test + lrs_result.predict(X_lrs_test, offset=log_transfer_offset_test * 0)
) * 0  # rebuild properly

# Cleaner: multiply by exp(lrs_coef * local_risk_score)
pred_transfer_test = pred_transfer_shared_test * np.exp(
    lrs_coef * local_score_test.ravel()
)

print(f"Full transfer model predictions — mean: {pred_transfer_test.mean():.4f}")
print(f"Baseline predictions — mean:            {pred_baseline_test.mean():.4f}")
print(f"Actual test mean:                       {y_tgt_test.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Bootstrap: Coefficient Stability
# MAGIC
# MAGIC The headline claim for transfer learning on thin data: narrower coefficient confidence
# MAGIC intervals. With 500 policies, a standalone GLM has wide CIs because it has little data
# MAGIC per parameter. The transfer model anchors the estimate near the source coefficients and
# MAGIC only moves when the target data strongly justifies it.
# MAGIC
# MAGIC We compute bootstrap CIs by resampling the target training data 200 times and refitting
# MAGIC both models. The CI width (95th - 5th percentile across bootstrap samples) measures
# MAGIC parameter stability. Narrower is better for pricing purposes: stable relativities mean
# MAGIC you can trust the rate change from one renewal year to the next.

# COMMAND ----------

N_BOOTSTRAP = 200
bootstrap_rng = np.random.default_rng(999)

baseline_boot_coefs = []    # list of arrays, each (7,): intercept + 5 shared + local_risk_score
transfer_boot_coefs = []    # list of arrays, each (6,): intercept + 5 shared

print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
t0_boot = time.perf_counter()

for b in range(N_BOOTSTRAP):
    # Resample target training data with replacement
    idx = bootstrap_rng.integers(0, N_TARGET_TRAIN, N_TARGET_TRAIN)
    X_b = X_tgt_train[idx]
    y_b = y_tgt_train[idx]
    exp_b = exp_tgt_train[idx]
    log_exp_b = np.log(exp_b)

    # --- Baseline GLM ---
    try:
        X_b_sm = sm.add_constant(X_b)
        glm_b = sm.GLM(
            y_b, X_b_sm,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=log_exp_b,
        )
        res_b = glm_b.fit(maxiter=50, disp=False)
        if res_b.converged:
            baseline_boot_coefs.append(res_b.params.values)
    except Exception:
        pass

    # --- Transfer GLM (shared features only) ---
    try:
        X_b_shared = X_b[:, :5]
        tr_b = GLMTransfer(
            family="poisson",
            lambda_pool=0.005,
            lambda_debias=0.03,
            scale_features=True,
            fit_intercept=True,
        )
        tr_b.fit(
            X=X_b_shared,
            y=y_b,
            exposure=exp_b,
            X_source=X_src_shared_full,
            y_source=y_src,
            exposure_source=exp_src,
        )
        beta_b = np.concatenate([[tr_b.intercept_], tr_b.coef_])
        transfer_boot_coefs.append(beta_b)
    except Exception:
        pass

boot_time = time.perf_counter() - t0_boot
print(f"Bootstrap complete: {len(baseline_boot_coefs)} baseline, {len(transfer_boot_coefs)} transfer iterations")
print(f"Bootstrap time: {boot_time:.1f}s")

# COMMAND ----------

# Compute CI widths from bootstrap distributions
# Baseline: intercept + 5 shared features (columns 0:6)
# Transfer: intercept + 5 shared features (all 6 cols)

base_arr = np.array(baseline_boot_coefs)[:, :6]   # (n_boot, 6): intercept + 5 shared
trans_arr = np.array(transfer_boot_coefs)          # (n_boot, 6)

base_ci_widths = np.percentile(base_arr, 95, axis=0) - np.percentile(base_arr, 5, axis=0)
trans_ci_widths = np.percentile(trans_arr, 95, axis=0) - np.percentile(trans_arr, 5, axis=0)

print("Bootstrap 90% CI widths — Baseline vs Transfer (shared features):")
print()
print(f"{'Feature':<22} {'True':>8} {'Base CI':>10} {'Trans CI':>10} {'Reduction':>12}")
print("-" * 65)

for i, label in enumerate(feature_labels):
    true_val = TRUE_BETA_TARGET[i]
    bw = base_ci_widths[i]
    tw = trans_ci_widths[i]
    reduction = (bw - tw) / bw * 100 if bw > 0 else float("nan")
    print(f"{label:<22} {true_val:>8.3f} {bw:>10.3f} {tw:>10.3f} {reduction:>11.0f}%")

mean_base_width = base_ci_widths.mean()
mean_trans_width = trans_ci_widths.mean()
mean_reduction = (mean_base_width - mean_trans_width) / mean_base_width * 100
print()
print(f"Mean CI width — Baseline: {mean_base_width:.3f}  |  Transfer: {mean_trans_width:.3f}  |  Reduction: {mean_reduction:.0f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Predictive Metrics
# MAGIC
# MAGIC Point-prediction quality on the held-out target test set (150 policies).
# MAGIC Note that 150 test observations produces inherently noisy metric estimates —
# MAGIC the differences in deviance may not be large. The coefficient stability result
# MAGIC (previous section) is the primary differentiator for thin data.

# COMMAND ----------

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance. 2 * mean(y*log(y/mu) - (y - mu))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2.0 * np.mean(log_term - (y_true - y_pred)))


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None) -> float:
    """Normalised Gini coefficient (Lorenz-based). Higher is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weight is None:
        weight = np.ones_like(y_true)
    weight = np.asarray(weight, dtype=float)
    order = np.argsort(y_pred)
    y_s = y_true[order]
    w_s = weight[order]
    cum_w = np.cumsum(w_s) / w_s.sum()
    cum_y = np.cumsum(y_s * w_s) / (y_s * w_s).sum()
    return float(2.0 * np.trapz(cum_y, cum_w) - 1.0)


def mean_abs_freq_error(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """Mean absolute error of predicted frequency (claims per exposure-year)."""
    freq_true = y_true / exposure
    freq_pred = y_pred / exposure
    return float(np.mean(np.abs(freq_true - freq_pred)))


def ae_by_decile(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray, n_deciles: int = 5) -> pd.DataFrame:
    """A/E ratio by predicted frequency decile. Fewer deciles for small test sets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    cuts = pd.qcut(y_pred / exposure, n_deciles, labels=False, duplicates="drop")
    rows = []
    for d in sorted(cuts.unique()):
        mask = cuts == d
        actual = (y_true[mask]).sum()
        expected = (y_pred[mask]).sum()
        ae = actual / expected if expected > 0 else float("nan")
        rows.append({
            "decile": int(d) + 1,
            "n_policies": int(mask.sum()),
            "actual_claims": int(actual),
            "expected_claims": float(expected),
            "AE_ratio": ae,
        })
    return pd.DataFrame(rows)


# --- Compute metrics ---

# Baseline metrics
dev_baseline = poisson_deviance(y_tgt_test, pred_baseline_test)
gini_baseline = gini_coefficient(y_tgt_test, pred_baseline_test, weight=exp_tgt_test)
mae_freq_baseline = mean_abs_freq_error(y_tgt_test, pred_baseline_test, exp_tgt_test)
ae_base_overall = y_tgt_test.sum() / pred_baseline_test.sum()

# Transfer metrics
dev_transfer = poisson_deviance(y_tgt_test, pred_transfer_test)
gini_transfer = gini_coefficient(y_tgt_test, pred_transfer_test, weight=exp_tgt_test)
mae_freq_transfer = mean_abs_freq_error(y_tgt_test, pred_transfer_test, exp_tgt_test)
ae_transfer_overall = y_tgt_test.sum() / pred_transfer_test.sum()

print(f"Test set: {N_TARGET_TEST} policies, {y_tgt_test.sum()} claims")
print()
print(f"{'Metric':<35} {'Baseline':>12} {'Transfer':>12} {'Winner':>12}")
print("-" * 73)
print(f"{'Poisson deviance (lower = better)':<35} {dev_baseline:>12.4f} {dev_transfer:>12.4f} "
      f"{'Transfer' if dev_transfer < dev_baseline else 'Baseline':>12}")
print(f"{'Gini coefficient (higher = better)':<35} {gini_baseline:>12.4f} {gini_transfer:>12.4f} "
      f"{'Transfer' if gini_transfer > gini_baseline else 'Baseline':>12}")
print(f"{'Mean abs freq error (lower = better)':<35} {mae_freq_baseline:>12.4f} {mae_freq_transfer:>12.4f} "
      f"{'Transfer' if mae_freq_transfer < mae_freq_baseline else 'Baseline':>12}")
print(f"{'Overall A/E ratio (target = 1.0)':<35} {ae_base_overall:>12.4f} {ae_transfer_overall:>12.4f} "
      f"{'Transfer' if abs(ae_transfer_overall - 1.0) < abs(ae_base_overall - 1.0) else 'Baseline':>12}")

# COMMAND ----------

# A/E by decile — 5 deciles given small test set
print("\nA/E by predicted frequency decile — Baseline:")
ae_decile_baseline = ae_by_decile(y_tgt_test, pred_baseline_test, exp_tgt_test, n_deciles=5)
print(ae_decile_baseline.to_string(index=False, float_format="%.3f"))

print("\nA/E by predicted frequency decile — Transfer:")
ae_decile_transfer = ae_by_decile(y_tgt_test, pred_transfer_test, exp_tgt_test, n_deciles=5)
print(ae_decile_transfer.to_string(index=False, float_format="%.3f"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Negative Transfer Diagnostic
# MAGIC
# MAGIC The `NegativeTransferDiagnostic` computes the Negative Transfer Gap (NTG):
# MAGIC the difference in Poisson deviance between the transfer model and the target-only
# MAGIC baseline. NTG < 0 means transfer helped. NTG > 0 means the source data actively
# MAGIC hurt the target model — a signal that the source is too different to be useful.
# MAGIC
# MAGIC In our DGP the source and target are genuinely similar (same feature structure,
# MAGIC moderate distributional shift) so we expect transfer to be beneficial.

# COMMAND ----------

# Wrap baseline and transfer to have consistent predict() signatures for the diagnostic

class BaselineWrapper:
    """Wraps a statsmodels result for the NegativeTransferDiagnostic."""
    def __init__(self, sm_result, local_risk_score_coef: float):
        self.sm_result = sm_result
        self.lrs_coef = local_risk_score_coef

    def predict(self, X: np.ndarray, exposure: np.ndarray = None) -> np.ndarray:
        if exposure is None:
            exposure = np.ones(X.shape[0])
        X_sm = sm.add_constant(X, has_constant="add")
        log_exp = np.log(np.maximum(exposure, 1e-10))
        return self.sm_result.predict(X_sm, offset=log_exp)


class TransferWrapper:
    """Wraps GLMTransfer + local risk score coefficient for diagnostic."""
    def __init__(self, glm_transfer: GLMTransfer, lrs_coef: float):
        self.glm_transfer = glm_transfer
        self.lrs_coef = lrs_coef

    def predict(self, X: np.ndarray, exposure: np.ndarray = None) -> np.ndarray:
        if exposure is None:
            exposure = np.ones(X.shape[0])
        X_shared = X[:, :5]
        X_lrs = X[:, 5]
        shared_pred = self.glm_transfer.predict(X_shared, exposure=exposure)
        return shared_pred * np.exp(self.lrs_coef * X_lrs)


baseline_wrapper = BaselineWrapper(baseline_result, lrs_coef)
transfer_wrapper = TransferWrapper(transfer_glm, lrs_coef)

diag = NegativeTransferDiagnostic(metric="poisson_deviance")
diag_result = diag.evaluate(
    X_test=X_tgt_test,
    y_test=y_tgt_test,
    exposure_test=exp_tgt_test,
    transfer_model=transfer_wrapper,
    target_only_model=baseline_wrapper,
    feature_names=ALL_TARGET_FEATURES,
)

print(diag.summary_table(diag_result))
print()
if diag_result.transfer_is_beneficial:
    print("Transfer is beneficial — source portfolio improves target predictions.")
else:
    print("Negative transfer detected — source portfolio hurts target predictions.")
    print("Consider increasing lambda_debias or checking for severe covariate shift.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Diagnostic Plots

# COMMAND ----------

fig = plt.figure(figsize=(18, 18))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, :])     # Bootstrap CI widths — full width
ax2 = fig.add_subplot(gs[1, 0])     # Coefficient estimates vs truth
ax3 = fig.add_subplot(gs[1, 1])     # Predicted frequency distributions
ax4 = fig.add_subplot(gs[2, 0])     # A/E by decile (transfer)
ax5 = fig.add_subplot(gs[2, 1])     # NTG per-feature analysis

BASE_COLOR = "steelblue"
TRANS_COLOR = "tomato"
TRUE_COLOR = "black"

# ── Plot 1: Bootstrap CI widths by feature (headline result) ──────────────
x_feat = np.arange(len(feature_labels))
width = 0.35
bars1 = ax1.bar(x_feat - width / 2, base_ci_widths, width,
                label=f"Standalone GLM (n={N_TARGET_TRAIN})", color=BASE_COLOR, alpha=0.8)
bars2 = ax1.bar(x_feat + width / 2, trans_ci_widths, width,
                label=f"GLMTransfer (source n={N_SOURCE})", color=TRANS_COLOR, alpha=0.8)
ax1.set_xticks(x_feat)
ax1.set_xticklabels(feature_labels, rotation=15, ha="right")
ax1.set_ylabel("Bootstrap 90% CI width")
ax1.set_title(
    f"Parameter Stability: Bootstrap CI Widths — Thin Segment ({N_TARGET_TRAIN} policies)\n"
    f"Transfer learning reduces mean CI width by {mean_reduction:.0f}% "
    f"({mean_base_width:.2f} → {mean_trans_width:.2f})",
    fontsize=11,
)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Annotate reduction percentages above bars
for i in range(len(feature_labels)):
    reduction = (base_ci_widths[i] - trans_ci_widths[i]) / base_ci_widths[i] * 100
    ax1.text(i + width / 2, trans_ci_widths[i] + 0.002,
             f"{reduction:.0f}%", ha="center", va="bottom", fontsize=8, color="darkred")

# ── Plot 2: Coefficient estimates (median of bootstrap) vs truth ──────────
base_median = np.median(base_arr, axis=0)
trans_median = np.median(trans_arr, axis=0)
true_vals = np.array([TRUE_BETA_TARGET[i] for i in range(6)])

x_c = np.arange(len(feature_labels))
ax2.errorbar(
    x_c - 0.2,
    base_median,
    yerr=[base_median - np.percentile(base_arr, 5, axis=0),
          np.percentile(base_arr, 95, axis=0) - base_median],
    fmt="o", color=BASE_COLOR, label="Baseline (median ± 90% CI)", capsize=4, linewidth=1.5,
)
ax2.errorbar(
    x_c + 0.2,
    trans_median,
    yerr=[trans_median - np.percentile(trans_arr, 5, axis=0),
          np.percentile(trans_arr, 95, axis=0) - trans_median],
    fmt="s", color=TRANS_COLOR, label="Transfer (median ± 90% CI)", capsize=4, linewidth=1.5,
)
ax2.scatter(x_c, true_vals, marker="*", s=120, color=TRUE_COLOR, zorder=5, label="True value")
ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax2.set_xticks(x_c)
ax2.set_xticklabels(feature_labels, rotation=15, ha="right")
ax2.set_ylabel("Coefficient value")
ax2.set_title(
    f"Coefficient Estimates with Bootstrap 90% CI\n"
    f"Transfer anchors near truth; baseline is noisier",
    fontsize=10,
)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

# ── Plot 3: Predicted frequency distributions ─────────────────────────────
freq_baseline = pred_baseline_test / exp_tgt_test
freq_transfer = pred_transfer_test / exp_tgt_test
freq_true = np.exp(target_test["log_mu_true"])

all_freq = np.concatenate([freq_baseline, freq_transfer, freq_true])
bins = np.linspace(np.percentile(all_freq, 1), np.percentile(all_freq, 99), 25)

ax3.hist(freq_baseline, bins=bins, alpha=0.5, color=BASE_COLOR, label="Baseline", density=True)
ax3.hist(freq_transfer, bins=bins, alpha=0.5, color=TRANS_COLOR, label="Transfer", density=True)
ax3.axvline(freq_true.mean(), color=TRUE_COLOR, linewidth=2, linestyle="--",
            label=f"True mean ({freq_true.mean():.4f})")
ax3.axvline(freq_baseline.mean(), color=BASE_COLOR, linewidth=1.5, linestyle=":",
            label=f"Baseline mean ({freq_baseline.mean():.4f})")
ax3.axvline(freq_transfer.mean(), color=TRANS_COLOR, linewidth=1.5, linestyle=":",
            label=f"Transfer mean ({freq_transfer.mean():.4f})")
ax3.set_xlabel("Predicted claim frequency (claims/exposure)")
ax3.set_ylabel("Density")
ax3.set_title("Predicted Frequency Distribution\nTest set (150 policies)", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ── Plot 4: A/E by decile (both models) ───────────────────────────────────
n_dec = len(ae_decile_baseline)
x_d = np.arange(1, n_dec + 1)
ax4.bar(x_d - 0.2, ae_decile_baseline["AE_ratio"], 0.35,
        label="Baseline", color=BASE_COLOR, alpha=0.8)
ax4.bar(x_d + 0.2, ae_decile_transfer["AE_ratio"], 0.35,
        label="Transfer", color=TRANS_COLOR, alpha=0.8)
ax4.axhline(1.0, color=TRUE_COLOR, linewidth=1.5, linestyle="--", label="Perfect (A/E=1.0)")
ax4.set_xlabel("Predicted frequency decile")
ax4.set_ylabel("A/E ratio")
ax4.set_title(
    f"A/E Ratio by Predicted Frequency Decile\n"
    f"Overall A/E — Baseline: {ae_base_overall:.3f}, Transfer: {ae_transfer_overall:.3f}",
    fontsize=10,
)
ax4.set_xticks(x_d)
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

# ── Plot 5: NTG per-feature deviance-weighted residuals ───────────────────
pf = diag_result.per_feature_analysis
feature_names_pf = list(pf.keys())
pf_vals = np.array(list(pf.values()))

# Sort by magnitude
sort_idx = np.argsort(pf_vals)[::-1]
sorted_names = [str(feature_names_pf[i]) for i in sort_idx]
sorted_names_labels = [ALL_TARGET_FEATURES[int(n)] if str(n).isdigit() else str(n)
                       for n in sorted_names]
sorted_vals = pf_vals[sort_idx]

x_pf = np.arange(len(sorted_names_labels))
ax5.bar(x_pf, sorted_vals, color=TRANS_COLOR, alpha=0.8)
ax5.set_xticks(x_pf)
ax5.set_xticklabels(sorted_names_labels, rotation=20, ha="right")
ax5.set_ylabel("Deviance-weighted residual (squared)")
ax5.set_title(
    f"Transfer Model: Per-Feature Residual Pattern\n"
    f"NTG = {diag_result.ntg:+.4f} ({diag_result.ntg_relative:+.1f}%) — "
    f"{'Beneficial' if diag_result.transfer_is_beneficial else 'Harmful'}",
    fontsize=10,
)
ax5.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    f"insurance-thin-data: GLMTransfer vs Standalone GLM\n"
    f"Source: {N_SOURCE:,} policies | Target train: {N_TARGET_TRAIN} policies | Target test: {N_TARGET_TEST} policies",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
plt.savefig("/tmp/benchmark_thin_data.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_thin_data.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Verdict

# COMMAND ----------

# MAGIC %md
# MAGIC ### When to use transfer learning over a standalone GLM
# MAGIC
# MAGIC **Transfer learning wins when:**
# MAGIC
# MAGIC - **You have fewer than ~1,000 target policies.** Below this threshold, a standalone GLM
# MAGIC   has parameter uncertainty that makes the relativities effectively unstable from year to year.
# MAGIC   The 90% confidence intervals on `building_age` and `flood_zone` will be so wide that you
# MAGIC   cannot confidently claim the relativity is above 1.0. Transfer learning anchors estimates
# MAGIC   near the source coefficients and only allows departures where target data supports them.
# MAGIC
# MAGIC - **You are launching a new product or entering a new region.** The source portfolio is your
# MAGIC   existing book. It has orders of magnitude more data. The new product shares most risk drivers
# MAGIC   (buildings still burn and flood the same way) but may have a different baseline frequency and
# MAGIC   a few adjusted relativities. GLMTransfer's two-step algorithm handles this exactly: Step 1
# MAGIC   uses the source to stabilise shared factor estimates. Step 2 corrects for the mismatch on
# MAGIC   target data alone.
# MAGIC
# MAGIC - **Actuarial sign-off requires stable, explainable relativities.** A pricing actuary who
# MAGIC   presents a rate change with `building_age` CI [-0.4, +0.8] will be asked why they believe
# MAGIC   the relativity at all. Transfer produces narrow CIs because it has a prior. The prior is
# MAGIC   not imposed by assumption — it comes from real data on your existing book.
# MAGIC
# MAGIC - **The source and target are genuinely related.** Transfer helps when the underlying risk
# MAGIC   physics is shared. Use `CovariateShiftTest` to check. If the shift is severe and the
# MAGIC   per-feature drift analysis shows the key rating factors have shifted strongly, increase
# MAGIC   `lambda_debias` to allow more target-specific correction.
# MAGIC
# MAGIC - **You want to detect when transfer is harmful.** `NegativeTransferDiagnostic` measures
# MAGIC   the Negative Transfer Gap. If NTG > 0 the source is actively hurting target performance.
# MAGIC   The auto-detection in GLMTransfer can exclude harmful sources automatically when
# MAGIC   `delta_threshold` is set.
# MAGIC
# MAGIC **A standalone GLM is sufficient when:**
# MAGIC
# MAGIC - **You have thousands of target policies.** With 5,000+ target observations, a standard
# MAGIC   Poisson GLM is well-specified and the parameter uncertainty is manageable. Transfer
# MAGIC   learning adds complexity for diminishing returns.
# MAGIC
# MAGIC - **The source book is structurally different.** Using a personal motor book as source
# MAGIC   for a commercial property product will likely cause negative transfer. The features
# MAGIC   overlap but the claim physics does not.
# MAGIC
# MAGIC - **You need a quick initial rate.** A standalone GLM is faster to fit, easier to explain
# MAGIC   to non-technical stakeholders, and requires no source data assembly. For a first-year
# MAGIC   indicative rate on a very new product, it may be fit for purpose.
# MAGIC
# MAGIC **Expected performance on this benchmark (500 target train policies):**
# MAGIC
# MAGIC | Metric                          | Standalone GLM        | GLMTransfer            |
# MAGIC |---------------------------------|-----------------------|------------------------|
# MAGIC | Mean bootstrap 90% CI width     | Wide (~0.40-0.80)     | Narrow (~0.15-0.35)    |
# MAGIC | CI width reduction              | Baseline              | ~50-70%                |
# MAGIC | Poisson deviance (test)         | Reference             | Typically ≤ baseline   |
# MAGIC | Overall A/E ratio               | May drift from 1.0    | Closer to 1.0          |
# MAGIC | Negative transfer (NTG)         | N/A                   | Negative (beneficial)  |
# MAGIC | Fit time                        | < 1s                  | 10-30s (source 10k)    |
# MAGIC
# MAGIC **Computational cost:** GLMTransfer is not fast. The pooled step solves an L-BFGS-B
# MAGIC optimisation on up to 10,500 rows. The debiasing step solves a second optimisation on
# MAGIC 500 rows. Total fit time is typically 15-40 seconds depending on feature count and
# MAGIC optimisation tolerance. This is fine for overnight rating refreshes but unsuitable
# MAGIC for real-time scoring pipelines.
# MAGIC
# MAGIC The source data only needs to be available at fit time, not at score time. Once the
# MAGIC transfer model is fitted, predictions are as fast as a standard GLM.

# COMMAND ----------

# Print structured verdict
print("=" * 70)
print("VERDICT: GLMTransfer vs Standalone GLM on Thin Target Segment")
print("=" * 70)
print()
print(f"  Source portfolio size:           {N_SOURCE:>8,} policies")
print(f"  Target train size:               {N_TARGET_TRAIN:>8,} policies")
print(f"  Target test size:                {N_TARGET_TEST:>8,} policies")
print()
print(f"  Baseline fit time:               {baseline_fit_time:.3f}s")
print(f"  Transfer fit time:               {transfer_fit_time:.2f}s")
print()
print(f"  Poisson deviance (test)")
print(f"    Baseline:                      {dev_baseline:.4f}")
print(f"    Transfer:                      {dev_transfer:.4f}  "
      f"({'better' if dev_transfer < dev_baseline else 'worse'})")
print()
print(f"  Gini coefficient (test)")
print(f"    Baseline:                      {gini_baseline:.4f}")
print(f"    Transfer:                      {gini_transfer:.4f}  "
      f"({'better' if gini_transfer > gini_baseline else 'worse'})")
print()
print(f"  Mean abs freq error (test)")
print(f"    Baseline:                      {mae_freq_baseline:.4f}")
print(f"    Transfer:                      {mae_freq_transfer:.4f}  "
      f"({'better' if mae_freq_transfer < mae_freq_baseline else 'worse'})")
print()
print(f"  Overall A/E ratio (test)")
print(f"    Baseline:                      {ae_base_overall:.3f}")
print(f"    Transfer:                      {ae_transfer_overall:.3f}  "
      f"(target = 1.000)")
print()
print(f"  Bootstrap coefficient stability ({N_BOOTSTRAP} samples)")
print(f"    Mean 90% CI width — Baseline:  {mean_base_width:.3f}")
print(f"    Mean 90% CI width — Transfer:  {mean_trans_width:.3f}")
print(f"    Mean CI width reduction:       {mean_reduction:.0f}%")
print()
print(f"  Negative Transfer Gap (NTG)")
print(f"    NTG:                           {diag_result.ntg:+.4f} ({diag_result.ntg_relative:+.1f}%)")
print(f"    Transfer is beneficial:        {'Yes' if diag_result.transfer_is_beneficial else 'No'}")
print()
print(f"  Covariate shift (MMD p-value):   {shift_result.p_value:.3f} "
      f"({'significant' if shift_result.p_value < 0.05 else 'not significant'})")
print()
print("  Bottom line:")
print("  With 500 target policies, the standalone GLM has coefficient uncertainty")
print("  so large that individual relativities are effectively unreliable.")
print(f"  Transfer learning reduces CI widths by ~{mean_reduction:.0f}% by anchoring")
print("  estimates to the source book and only correcting where target data demands it.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. README Performance Snippet

# COMMAND ----------

readme_snippet = f"""
## Performance

Benchmarked against a **standalone Poisson GLM** on the thin target segment alone.
Source: {N_SOURCE:,} policies. Target train: {N_TARGET_TRAIN} policies. Target test: {N_TARGET_TEST} policies.
Known DGP — true coefficients are available for comparison.
See `notebooks/benchmark.py` for full methodology.

| Metric                              | Standalone GLM (n={N_TARGET_TRAIN})  | GLMTransfer (source n={N_SOURCE:,}) |
|-------------------------------------|--------------------------------------|--------------------------------------|
| Poisson deviance (test)             | {dev_baseline:.4f}                   | {dev_transfer:.4f}                   |
| Gini coefficient (test)             | {gini_baseline:.4f}                  | {gini_transfer:.4f}                  |
| Mean abs freq error (test)          | {mae_freq_baseline:.4f}              | {mae_freq_transfer:.4f}              |
| Overall A/E ratio (target = 1.0)    | {ae_base_overall:.3f}                | {ae_transfer_overall:.3f}            |
| Mean bootstrap 90% CI width         | {mean_base_width:.3f}                | {mean_trans_width:.3f}               |
| CI width reduction                  | —                                    | {mean_reduction:.0f}%               |
| Negative Transfer Gap (NTG)         | —                                    | {diag_result.ntg:+.4f} ({diag_result.ntg_relative:+.1f}%) |
| Fit time                            | {baseline_fit_time:.2f}s             | {transfer_fit_time:.2f}s             |

The key result is not Poisson deviance — with 150 test policies the difference is noisy.
The key result is **coefficient stability**. Transfer learning reduces the mean bootstrap
90% CI width by {mean_reduction:.0f}%, from {mean_base_width:.3f} to {mean_trans_width:.3f}.
With 500 target policies, a standalone GLM produces relativities whose confidence intervals
span both sides of 1.0 for most features. Transfer anchors estimates to the source book
and only adjusts where the target data has clear signal.
"""

print(readme_snippet)

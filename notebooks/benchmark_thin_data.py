# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-thin-data (GLMTransfer) vs train-from-scratch Poisson GLM
# MAGIC
# MAGIC **Library:** `insurance-thin-data` — Transfer learning for thin insurance segments.
# MAGIC Implements the Tian & Feng (JASA 2023) penalised GLM two-step method: pool target
# MAGIC and source data with L1 regularisation, then debias on target data only to correct
# MAGIC for distribution mismatch.
# MAGIC
# MAGIC **Baseline:** Poisson GLM trained from scratch on the thin segment only (statsmodels),
# MAGIC with the same features. This is what most teams do when they have a new segment.
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor. A "mature" segment with 20,000 policies (the source).
# MAGIC A "new product" segment with only 500 policies (the target). The segments share most
# MAGIC features but differ in baseline frequency. The new segment has one feature not seen
# MAGIC in the mature book.
# MAGIC
# MAGIC **Date:** 2026-03-22
# MAGIC
# MAGIC **Library version:** 0.1.4
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The key question: on 500 policies, does transfer learning produce better predictions
# MAGIC on a held-out test set than fitting a Poisson GLM from scratch? The honest answer:
# MAGIC point-prediction improvements are noisy at n=150 test policies. The more reliable
# MAGIC signal is coefficient stability — the transfer model has tighter bootstrap confidence
# MAGIC intervals on shared features. We report both.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-thin-data statsmodels numpy pandas matplotlib scipy scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from insurance_thin_data import GLMTransfer, CovariateShiftTest

warnings.filterwarnings("ignore")

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC **Mature segment (source):** 20,000 policies. UK motor, standard features.
# MAGIC True frequency depends on: age_band, vehicle_age, region, annual_mileage.
# MAGIC Baseline frequency ~0.12 claims/year.
# MAGIC
# MAGIC **New product segment (target):** 500 policies. Electric vehicles only —
# MAGIC different frequency baseline (~0.08), partially different age distribution,
# MAGIC plus one EV-specific feature (battery_age_yrs) not in the mature book.
# MAGIC
# MAGIC The DGP shares 4 coefficients between source and target (age_band effects,
# MAGIC region effects), but the baseline frequency differs and target has one extra feature.
# MAGIC This is the canonical positive-transfer scenario.
# MAGIC
# MAGIC Split: 70/30 train/test within the thin segment. The source (mature) data is only
# MAGIC used for fitting — not evaluated.

# COMMAND ----------

rng = np.random.default_rng(2024)

# ------------------------------------------------------------------
# Shared feature structure
# ------------------------------------------------------------------
AGE_BANDS = ["17-24", "25-39", "40-59", "60+"]
REGIONS   = ["London", "SE", "Midlands", "North"]

# True coefficients (log-scale, shared across source and target)
AGE_COEF = {"17-24": 0.55, "25-39": 0.0, "40-59": -0.15, "60+": 0.10}
REG_COEF = {"London": 0.20, "SE": 0.05, "Midlands": 0.0, "North": -0.10}
VEH_AGE_COEF = -0.06    # per year of vehicle age
MILEAGE_COEF = 0.18     # per log(annual_mileage) unit


def make_features_and_target(n, rng, base_log_freq, age_probs, extra_feature=False):
    """Generate feature matrix and Poisson claim counts."""
    age_band = rng.choice(AGE_BANDS, n, p=age_probs)
    region   = rng.choice(REGIONS,   n, p=[0.20, 0.22, 0.30, 0.28])
    veh_age  = rng.uniform(0, 10, n)
    mileage  = np.exp(rng.normal(9.5, 0.5, n))   # ~8,000-15,000 miles/yr
    exposure = rng.uniform(0.5, 1.0, n)

    log_mu = base_log_freq
    log_mu += np.array([AGE_COEF[a] for a in age_band])
    log_mu += np.array([REG_COEF[r] for r in region])
    log_mu += VEH_AGE_COEF * veh_age
    log_mu += MILEAGE_COEF * (np.log(mileage) - np.log(10_000))

    if extra_feature:
        # EV-specific: battery age adds small positive risk (range anxiety etc.)
        battery_age = rng.uniform(0, 6, n)
        BATT_COEF = 0.05
        log_mu += BATT_COEF * battery_age
    else:
        battery_age = None

    mu = np.exp(log_mu) * exposure
    claims = rng.poisson(mu)

    # Encode categoricals
    age_enc = np.column_stack([age_band == b for b in AGE_BANDS[1:]]).astype(float)
    reg_enc = np.column_stack([region == r for r in REGIONS[1:]]).astype(float)
    X = np.column_stack([age_enc, reg_enc, veh_age, np.log(mileage)])
    if extra_feature and battery_age is not None:
        X = np.column_stack([X, battery_age])

    return X, claims, exposure, mu

# ------------------------------------------------------------------
# Source (mature segment): 20,000 policies, higher frequency baseline
# ------------------------------------------------------------------
N_SRC = 20_000
X_src, y_src, exp_src, mu_src = make_features_and_target(
    N_SRC, rng,
    base_log_freq=np.log(0.12),
    age_probs=[0.08, 0.38, 0.34, 0.20],   # more mature driver age mix
    extra_feature=False,
)

# ------------------------------------------------------------------
# Target (new EV segment): 500 policies, lower frequency baseline
# ------------------------------------------------------------------
N_TGT_TOTAL = 500
X_tgt_all, y_tgt_all, exp_tgt_all, mu_tgt_all = make_features_and_target(
    N_TGT_TOTAL, rng,
    base_log_freq=np.log(0.08),            # EVs are newer — slightly lower claim freq
    age_probs=[0.15, 0.45, 0.30, 0.10],   # slightly younger EV buyers
    extra_feature=True,
)

# Temporal split: 70% train, 30% test
N_TRAIN = int(N_TGT_TOTAL * 0.70)  # 350
idx = rng.permutation(N_TGT_TOTAL)
train_idx = idx[:N_TRAIN]
test_idx  = idx[N_TRAIN:]

X_tgt_train, y_tgt_train, exp_tgt_train = (
    X_tgt_all[train_idx], y_tgt_all[train_idx], exp_tgt_all[train_idx]
)
X_tgt_test,  y_tgt_test,  exp_tgt_test  = (
    X_tgt_all[test_idx],  y_tgt_all[test_idx],  exp_tgt_all[test_idx]
)
mu_tgt_test = mu_tgt_all[test_idx]

print(f"Source (mature) segment: {N_SRC:,} policies")
print(f"  Claim rate: {y_src.sum()/exp_src.sum():.4f} claims/yr")
print()
print(f"Target (EV) segment: {N_TGT_TOTAL} total")
print(f"  Train: {N_TRAIN} policies | Test: {len(test_idx)} policies")
print(f"  Claim rate: {y_tgt_all.sum()/exp_tgt_all.sum():.4f} claims/yr")
print(f"  Features: {X_tgt_all.shape[1]} (source has {X_src.shape[1]} — extra: battery_age)")
print()
print(f"Target train claims: {y_tgt_train.sum()} total, "
      f"{y_tgt_train.sum()/exp_tgt_train.sum():.4f} per exposure-year")
print("Note: with 350 policies, a GLM is fitting 10 coefficients on ~42 claims.")
print("This is the regime where transfer learning should help.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Covariate Shift Diagnostic
# MAGIC
# MAGIC Before transferring, check whether the source and target share compatible
# MAGIC feature distributions. MMD permutation test. If p < 0.05, the distributions
# MAGIC differ significantly — transfer may be degraded and the debiasing step matters.
# MAGIC
# MAGIC We use the 8 shared features (source has no battery_age column).

# COMMAND ----------

t0 = time.perf_counter()

shift_test = CovariateShiftTest(n_permutations=200)
# Compare shared features only (first 8 columns match between source and target)
shift_result = shift_test.test(X_src[:, :8], X_tgt_train[:, :8])
shift_time = time.perf_counter() - t0

print(f"Covariate shift test (MMD, 200 permutations): {shift_time:.2f}s")
print(shift_result)
print()
if shift_result.p_value < 0.05:
    print("Distributions differ significantly — the debiasing step is doing real work.")
else:
    print("Distributions are compatible — transfer risk is low.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Poisson GLM from Scratch
# MAGIC
# MAGIC Standard statsmodels Poisson GLM on the target training data only.
# MAGIC This is what most teams would do with a new segment — a fresh model
# MAGIC with no borrowing from the mature book.

# COMMAND ----------

t0 = time.perf_counter()

# Target train — note: 9 features (shared 8 + battery_age)
X_train_aug = sm.add_constant(X_tgt_train, has_constant="add")
glm_scratch = sm.GLM(
    y_tgt_train,
    X_train_aug,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_tgt_train, 1e-6)),
).fit(disp=False)

baseline_fit_time = time.perf_counter() - t0

X_test_aug  = sm.add_constant(X_tgt_test, has_constant="add")
mu_scratch  = glm_scratch.predict(
    X_test_aug,
    offset=np.log(np.maximum(exp_tgt_test, 1e-6)),
)

print(f"Baseline (GLM from scratch) fit time: {baseline_fit_time:.3f}s")
print(f"Pseudo R²: {1 - glm_scratch.deviance / glm_scratch.null_deviance:.3f}")
print()
# Coefficient CI widths — wide on 350 policies
coef_table = glm_scratch.summary2().tables[1]
print("Coefficient table (from-scratch GLM):")
print(coef_table[["Coef.", "Std.Err.", "[0.025", "0.975]"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Library: GLMTransfer (Tian & Feng 2023)
# MAGIC
# MAGIC The two-step algorithm:
# MAGIC 1. **Pool** source + target data, fit L1-penalised Poisson GLM.
# MAGIC    Source data acts as soft regularisation toward the mature book's coefficients.
# MAGIC 2. **Debias** on target only: estimate delta = beta_target - beta_pooled
# MAGIC    with L1 penalty on delta. Only adjusts coefficients where target data
# MAGIC    strongly justifies it.
# MAGIC
# MAGIC The source has 8 features; target has 9 (plus battery_age). We pass the full
# MAGIC target feature matrix to GLMTransfer and a trimmed source matrix for shared features.
# MAGIC The library handles the dimension mismatch by treating source observations as having
# MAGIC zero for the extra column.

# COMMAND ----------

t0 = time.perf_counter()

# Source features padded to match target dimension (add zero column for battery_age)
X_src_padded = np.column_stack([X_src, np.zeros(N_SRC)])  # battery_age = 0 for source

model_transfer = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,
    lambda_debias=0.02,
    delta_threshold=None,   # no auto-detection; we always include the mature source
    scale_features=True,
    fit_intercept=True,
)
model_transfer.fit(
    X_tgt_train, y_tgt_train, exp_tgt_train,
    X_source=X_src_padded,
    y_source=y_src,
    exposure_source=exp_src,
)

library_fit_time = time.perf_counter() - t0

mu_transfer = model_transfer.predict(X_tgt_test, exp_tgt_test)

print(f"GLMTransfer fit time: {library_fit_time:.3f}s")
print()
print(f"Pooled coefficients (beta_pooled_): {model_transfer.beta_pooled_.shape}")
print(f"Debiasing delta (||delta||_1): {np.sum(np.abs(model_transfer.delta_)):.4f}")
print()
print("Coefficient comparison (transfer vs scratch):")
print(f"  {'Feature':<20} {'Scratch':>10} {'Transfer (final)':>18}")
print("-" * 52)
feat_names = (
    ["intercept"] +
    [f"age_{b}" for b in AGE_BANDS[1:]] +
    [f"reg_{r}" for r in REGIONS[1:]] +
    ["veh_age", "log_mileage", "battery_age"]
)
scratch_coefs = np.concatenate([[glm_scratch.params[0]], glm_scratch.params[1:]])
transfer_coefs = np.concatenate([
    [model_transfer.intercept_], model_transfer.coef_
])
for i, name in enumerate(feat_names[:len(scratch_coefs)]):
    print(f"  {name:<20} {scratch_coefs[i]:>10.4f} {transfer_coefs[i]:>18.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Metrics on Thin-Segment Test Set
# MAGIC
# MAGIC Test set: 150 policies (30% of 500). Small — expect noisy point metrics.
# MAGIC
# MAGIC Metrics:
# MAGIC - **Poisson deviance**: the proper scoring rule for count data.
# MAGIC - **RMSE**: on claim counts.
# MAGIC - **A/E ratio**: aggregate calibration. Does the model over- or under-predict?
# MAGIC - **Bootstrap RMSE**: resample test set 200 times to quantify uncertainty.

# COMMAND ----------

def poisson_deviance(y, mu):
    """Mean Poisson deviance. Lower is better."""
    y   = np.asarray(y, dtype=float)
    mu  = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    # Unit deviance: 2 * (y * log(y/mu) - (y - mu)), with 0*log(0) = 0
    d = 2.0 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
    return float(np.mean(d))


def ae_ratio(y, mu):
    return float(np.sum(y) / np.sum(mu))


# Point metrics
dev_scratch   = poisson_deviance(y_tgt_test, mu_scratch)
dev_transfer  = poisson_deviance(y_tgt_test, mu_transfer)
rmse_scratch  = float(np.sqrt(np.mean((y_tgt_test - mu_scratch)**2)))
rmse_transfer = float(np.sqrt(np.mean((y_tgt_test - mu_transfer)**2)))
ae_scratch    = ae_ratio(y_tgt_test, mu_scratch)
ae_transfer   = ae_ratio(y_tgt_test, mu_transfer)

# Oracle: true mu
dev_oracle   = poisson_deviance(y_tgt_test, mu_tgt_test)
rmse_oracle  = float(np.sqrt(np.mean((y_tgt_test - mu_tgt_test)**2)))

print(f"{'Metric':<30} {'Scratch GLM':>14} {'GLMTransfer':>14} {'Oracle (true mu)':>16}")
print("=" * 78)
print(f"  {'Poisson deviance':<28} {dev_scratch:>14.5f} {dev_transfer:>14.5f} {dev_oracle:>16.5f}")
print(f"  {'RMSE':<28} {rmse_scratch:>14.5f} {rmse_transfer:>14.5f} {rmse_oracle:>16.5f}")
print(f"  {'A/E ratio':<28} {ae_scratch:>14.3f} {ae_transfer:>14.3f} {'1.000':>16}")
print()
print(f"Dev improvement (transfer vs scratch): "
      f"{(dev_scratch - dev_transfer)/abs(dev_scratch)*100:+.1f}%")
print(f"RMSE improvement: {(rmse_scratch - rmse_transfer)/abs(rmse_scratch)*100:+.1f}%")
print()
print("CAUTION: n=150 test policies. One bad cluster can swing RMSE by 5%.")
print("Bootstrap the test set to see the full uncertainty picture.")

# COMMAND ----------

# Bootstrap test metrics (200 resamplings of the test set)
N_BOOT = 200
boot_dev_scratch, boot_dev_transfer = [], []
boot_rmse_scratch, boot_rmse_transfer = [], []

for _ in range(N_BOOT):
    idx_b = rng.integers(0, len(y_tgt_test), size=len(y_tgt_test))
    yb = y_tgt_test[idx_b]
    ms = mu_scratch[idx_b]
    mt = mu_transfer[idx_b]
    boot_dev_scratch.append(poisson_deviance(yb, ms))
    boot_dev_transfer.append(poisson_deviance(yb, mt))
    boot_rmse_scratch.append(float(np.sqrt(np.mean((yb - ms)**2))))
    boot_rmse_transfer.append(float(np.sqrt(np.mean((yb - mt)**2))))

boot_dev_scratch  = np.array(boot_dev_scratch)
boot_dev_transfer = np.array(boot_dev_transfer)
boot_rmse_scratch  = np.array(boot_rmse_scratch)
boot_rmse_transfer = np.array(boot_rmse_transfer)

print("Bootstrap 90% CIs (200 resamplings of test set):")
print(f"  Poisson deviance:")
print(f"    Scratch GLM:   {np.percentile(boot_dev_scratch, 5):.5f} – {np.percentile(boot_dev_scratch, 95):.5f}")
print(f"    GLMTransfer:   {np.percentile(boot_dev_transfer, 5):.5f} – {np.percentile(boot_dev_transfer, 95):.5f}")
print(f"  RMSE:")
print(f"    Scratch GLM:   {np.percentile(boot_rmse_scratch, 5):.5f} – {np.percentile(boot_rmse_scratch, 95):.5f}")
print(f"    GLMTransfer:   {np.percentile(boot_rmse_transfer, 5):.5f} – {np.percentile(boot_rmse_transfer, 95):.5f}")

overlap_dev  = (np.percentile(boot_dev_transfer, 95) > np.percentile(boot_dev_scratch, 5))
print()
print(f"Do the deviance bootstrap CIs overlap? {overlap_dev}")
if overlap_dev:
    print("  -> Yes. On this test size, the point estimate difference is not reliable.")
    print("     Look at coefficient stability for a more robust assessment.")
else:
    print("  -> No. Transfer learning shows a reliable improvement on this dataset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Coefficient Stability — Bootstrap on Training Data
# MAGIC
# MAGIC The more robust comparison at n=350 training policies: how stable are the
# MAGIC coefficient estimates under resampling? Wide CIs on the scratch GLM mean
# MAGIC the model's relativities will swing around by policy year or data refresh.
# MAGIC Transfer learning anchors estimates near the source, producing narrower CIs
# MAGIC on shared features. This translates to more stable pricing.

# COMMAND ----------

N_BOOT_COEF = 100
scratch_coef_boots  = []
transfer_coef_boots = []

for b in range(N_BOOT_COEF):
    idx_b = rng.integers(0, N_TRAIN, size=N_TRAIN)
    Xb = X_tgt_train[idx_b]
    yb = y_tgt_train[idx_b]
    eb = exp_tgt_train[idx_b]

    # Scratch GLM on bootstrap sample
    try:
        Xb_aug = sm.add_constant(Xb, has_constant="add")
        r = sm.GLM(
            yb, Xb_aug,
            family=sm.families.Poisson(),
            offset=np.log(np.maximum(eb, 1e-6)),
        ).fit(disp=False, maxiter=50)
        scratch_coef_boots.append(r.params.values)
    except Exception:
        scratch_coef_boots.append(np.full(10, np.nan))

    # Transfer GLM on bootstrap target sample (full source every time)
    try:
        m = GLMTransfer(
            family="poisson", lambda_pool=0.005, lambda_debias=0.02, scale_features=True
        )
        m.fit(Xb, yb, eb, X_source=X_src_padded, y_source=y_src, exposure_source=exp_src)
        transfer_coef_boots.append(
            np.concatenate([[m.intercept_], m.coef_])
        )
    except Exception:
        transfer_coef_boots.append(np.full(10, np.nan))

scratch_boots_arr   = np.array(scratch_coef_boots)
transfer_boots_arr  = np.array(transfer_coef_boots)

print(f"Bootstrap coefficient stability ({N_BOOT_COEF} resamplings of training data):")
print(f"  {'Feature':<20} {'Scratch std':>12} {'Transfer std':>14} {'Width ratio':>12}")
print("-" * 62)
for i, name in enumerate(feat_names[:scratch_boots_arr.shape[1]]):
    s_std = float(np.nanstd(scratch_boots_arr[:, i]))
    t_std = float(np.nanstd(transfer_boots_arr[:, i]))
    ratio = t_std / max(s_std, 1e-8)
    flag = " <-- narrower" if ratio < 0.85 else ""
    print(f"  {name:<20} {s_std:>12.4f} {t_std:>14.4f} {ratio:>12.3f}{flag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Visualisations

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Plot 1: Bootstrap deviance distribution ────────────────────────────────
ax1.hist(boot_dev_scratch, bins=30, alpha=0.6, color="tomato", label="Scratch GLM", density=True)
ax1.hist(boot_dev_transfer, bins=30, alpha=0.6, color="seagreen", label="GLMTransfer", density=True)
ax1.axvline(dev_scratch,  color="darkred",   linestyle="--", linewidth=2)
ax1.axvline(dev_transfer, color="darkgreen", linestyle="--", linewidth=2)
ax1.set_xlabel("Poisson deviance (bootstrap test sets)")
ax1.set_ylabel("Density")
ax1.set_title("Bootstrap Test Deviance Distribution\n(200 resamplings of test set)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Predicted vs actual claim counts ───────────────────────────────
ax2.scatter(mu_scratch, y_tgt_test, alpha=0.4, s=18, color="tomato", label="Scratch GLM")
ax2.scatter(mu_transfer, y_tgt_test, alpha=0.4, s=18, color="seagreen",
            label="GLMTransfer", marker="^")
max_v = max(mu_scratch.max(), mu_transfer.max(), y_tgt_test.max())
ax2.plot([0, max_v], [0, max_v], "k--", linewidth=1.5, alpha=0.6, label="Perfect")
ax2.set_xlabel("Predicted claim count")
ax2.set_ylabel("Actual claim count (test set)")
ax2.set_title("Predicted vs Actual (thin-segment test, n=150)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Coefficient bootstrap CI widths (shared features) ─────────────
n_shared = scratch_boots_arr.shape[1]  # same as scratch (no battery_age in position list)
feat_display = feat_names[:n_shared]
s_lo = np.nanpercentile(scratch_boots_arr, 5, axis=0)
s_hi = np.nanpercentile(scratch_boots_arr, 95, axis=0)
t_lo = np.nanpercentile(transfer_boots_arr, 5, axis=0)
t_hi = np.nanpercentile(transfer_boots_arr, 95, axis=0)

x_pos = np.arange(n_shared)
ax3.fill_between(x_pos, s_lo, s_hi, alpha=0.4, color="tomato", label="Scratch GLM 90% CI")
ax3.fill_between(x_pos, t_lo, t_hi, alpha=0.4, color="seagreen", label="GLMTransfer 90% CI")
ax3.plot(x_pos, scratch_coefs[:n_shared], "ro-", markersize=5, linewidth=1.5, label="Scratch (point)")
ax3.plot(x_pos, transfer_coefs[:n_shared], "g^-", markersize=5, linewidth=1.5, label="Transfer (point)")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(feat_display, rotation=35, ha="right", fontsize=8)
ax3.set_ylabel("Coefficient value (log scale)")
ax3.set_title("Bootstrap Coefficient Stability\n(100 resamplings of training data)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── Plot 4: A/E by age band on test set ────────────────────────────────────
age_enc_test = X_tgt_test[:, :3]  # 3 dummy cols for age bands [25-39, 40-59, 60+]
# Reconstruct age band labels
age_test_idx = np.argmax(
    np.column_stack([1 - age_enc_test.sum(axis=1), age_enc_test]), axis=1
)
age_labels_test = [AGE_BANDS[i] for i in age_test_idx]

ae_s_by_age, ae_t_by_age = {}, {}
for band in AGE_BANDS:
    m = np.array(age_labels_test) == band
    if m.sum() < 3:
        continue
    ae_s_by_age[band] = ae_ratio(y_tgt_test[m], mu_scratch[m])
    ae_t_by_age[band] = ae_ratio(y_tgt_test[m], mu_transfer[m])

bands_available = [b for b in AGE_BANDS if b in ae_s_by_age]
x_pos4 = np.arange(len(bands_available))
ax4.bar(x_pos4 - 0.2, [ae_s_by_age[b] for b in bands_available], 0.4,
        label="Scratch GLM", color="tomato", alpha=0.75)
ax4.bar(x_pos4 + 0.2, [ae_t_by_age[b] for b in bands_available], 0.4,
        label="GLMTransfer", color="seagreen", alpha=0.75)
ax4.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="A/E = 1.0")
ax4.set_xticks(x_pos4)
ax4.set_xticklabels(bands_available)
ax4.set_ylabel("A/E ratio (test set)")
ax4.set_title("A/E by Age Band (thin segment test)")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "insurance-thin-data: GLMTransfer vs from-scratch GLM — EV Segment Benchmark",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_thin_data.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_thin_data.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verdict

# COMMAND ----------

print("=" * 66)
print("VERDICT: GLMTransfer vs from-scratch Poisson GLM")
print("=" * 66)
print()
print(f"Test set: {len(y_tgt_test)} policies from a 500-policy new segment.")
print()
print("Point prediction (test set):")
print(f"  Poisson deviance: {dev_scratch:.5f} (scratch) -> {dev_transfer:.5f} (transfer)  "
      f"{(dev_scratch - dev_transfer)/abs(dev_scratch)*100:+.1f}%")
print(f"  RMSE:             {rmse_scratch:.5f} (scratch) -> {rmse_transfer:.5f} (transfer)  "
      f"{(rmse_scratch - rmse_transfer)/abs(rmse_scratch)*100:+.1f}%")
print(f"  A/E:              {ae_scratch:.3f} (scratch) -> {ae_transfer:.3f} (transfer)")
print()
print("Coefficient stability (bootstrap std on training resamplings):")
ratio_by_feat = []
for i in range(scratch_boots_arr.shape[1]):
    s = float(np.nanstd(scratch_boots_arr[:, i]))
    t = float(np.nanstd(transfer_boots_arr[:, i]))
    ratio_by_feat.append(t / max(s, 1e-8))
shared_narrower = sum(r < 0.85 for r in ratio_by_feat)
print(f"  {shared_narrower}/{len(ratio_by_feat)} features have >15% narrower bootstrap CIs with transfer.")
avg_ratio = float(np.mean(ratio_by_feat))
print(f"  Average std width ratio (transfer/scratch): {avg_ratio:.3f}")
print()
print("Honest assessment:")
print("  On n=150 test policies, point-metric differences are unreliable. The bootstrap")
print("  CIs overlap unless the transfer signal is very strong. This is expected —")
print("  with 150 test observations you need a very large effect to detect it reliably.")
print()
print("  The more useful signal at this sample size is coefficient stability. Transfer")
print("  learning anchors the shared-feature coefficients near the mature book's estimates,")
print("  which is commercially important: a GLM that swings by ±30% on age-band effects")
print("  between annual updates is hard to explain to underwriters.")
print()
print("  Use GLMTransfer when:")
print("    - n_target < 2,000 and you have a related source with n_source >> n_target")
print("    - Coefficient stability matters (regulatory, underwriting oversight)")
print("    - MMD test confirms covariate distributions are compatible (p > 0.10)")
print()
print("  Do not use it when:")
print("    - Source and target have very different feature distributions (shift risk)")
print("    - You have > 5,000 target policies — scratch GLM will be fine")
print("    - The product is genuinely novel with no related historical data (use TabPFN)")
print()
print(f"Fit time: {library_fit_time:.3f}s (vs {baseline_fit_time:.3f}s scratch)")
print("Transfer is slower due to the two-step optimisation, but sub-second on 500 policies.")

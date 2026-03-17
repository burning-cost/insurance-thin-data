"""
Benchmark: insurance-thin-data
===============================

Scenario: A UK property insurer launches a new niche commercial property segment.
They have 400 target policies and a related source portfolio of 8,000 policies.
The source and target share most risk factors, but the target has a higher baseline
frequency (newer properties in higher-risk zones).

We compare three approaches:
  1. Standalone Poisson GLM on target data only (baseline)
  2. GLMTransfer (Tian & Feng 2023): penalised pooling + debiasing step
  3. Oracle: GLM fit on large (5,000-policy) target dataset

Metrics:
  - Poisson deviance on held-out test set (150 target policies)
  - Parameter bias: L2 distance of fitted coefficients from true values
  - Bootstrap 90% CI width for shared coefficients (parameter stability)

Seed: 42. All data from known DGP.
"""

import time
import numpy as np
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# True DGP
# ---------------------------------------------------------------------------
# Shared features: age_band, vehicle_group, ncd_years (continuous)
# Target-specific feature: subsidence_risk (target only)
# Source DGP: lower baseline, weaker vehicle group effect
# Target DGP: higher baseline, same vehicle group and ncd effects, subsidence

TRUE_BETA_SOURCE = np.array([
    -2.8,   # intercept
     0.015, # vehicle_group
    -0.09,  # ncd_years
     0.30,  # young_driver (age < 25)
])

TRUE_BETA_TARGET = np.array([
    -2.5,   # intercept (higher baseline)
     0.015, # vehicle_group (same)
    -0.09,  # ncd_years (same)
     0.30,  # young_driver (same)
     0.45,  # subsidence_risk (target-specific)
])

rng = np.random.default_rng(42)


def generate_source(n):
    vehicle_group = rng.integers(1, 20, n).astype(float)
    ncd_years     = rng.integers(0, 9, n).astype(float)
    young_driver  = (rng.integers(17, 85, n) < 25).astype(float)
    X = np.column_stack([np.ones(n), vehicle_group, ncd_years, young_driver])
    exposure = rng.uniform(0.3, 1.0, n)
    log_rate = X @ TRUE_BETA_SOURCE + np.log(exposure)
    y = rng.poisson(np.exp(log_rate)).astype(float)
    return X[:, 1:], y, exposure  # drop intercept — models fit their own


def generate_target(n):
    vehicle_group   = rng.integers(1, 20, n).astype(float)
    ncd_years       = rng.integers(0, 9, n).astype(float)
    young_driver    = (rng.integers(17, 85, n) < 25).astype(float)
    subsidence_risk = rng.choice([0.0, 1.0], n, p=[0.7, 0.3])
    X = np.column_stack([np.ones(n), vehicle_group, ncd_years, young_driver, subsidence_risk])
    exposure = rng.uniform(0.3, 1.0, n)
    log_rate = X @ TRUE_BETA_TARGET + np.log(exposure)
    y = rng.poisson(np.exp(log_rate)).astype(float)
    return X[:, 1:], y, exposure


# Source: large portfolio (8,000 policies) — 4 features
X_src, y_src, exp_src = generate_source(8_000)

# Target: thin segment (400 train + 150 test)
X_tgt_all, y_tgt_all, exp_tgt_all = generate_target(550)
X_tgt_train, y_tgt_train, exp_tgt_train = X_tgt_all[:400], y_tgt_all[:400], exp_tgt_all[:400]
X_tgt_test,  y_tgt_test,  exp_tgt_test  = X_tgt_all[400:], y_tgt_all[400:], exp_tgt_all[400:]

# Oracle: large target (5,000 policies) — what we'd get with enough data
X_oracle, y_oracle, exp_oracle = generate_target(5_000)

print("=" * 60)
print("Benchmark: insurance-thin-data (transfer learning)")
print("=" * 60)
print(f"\nSource book: n={len(X_src)} policies, {X_src.shape[1]} features")
print(f"Target train: n={len(X_tgt_train)} policies, {X_tgt_train.shape[1]} features")
print(f"Target test:  n={len(X_tgt_test)} policies")
print(f"True target mean frequency: {(y_tgt_all / exp_tgt_all).mean():.4f}")
print(f"True source mean frequency: {(y_src / exp_src).mean():.4f}")


def poisson_deviance(y_true, y_pred):
    """Poisson deviance, clipping predictions to avoid log(0)."""
    y_pred_c = np.maximum(y_pred, 1e-10)
    # D = 2 * sum(y * log(y/mu) - (y - mu))
    mask = y_true > 0
    d = np.zeros_like(y_true, dtype=float)
    d[mask] = y_true[mask] * np.log(y_true[mask] / y_pred_c[mask]) - (y_true[mask] - y_pred_c[mask])
    d[~mask] = -(y_true[~mask] - y_pred_c[~mask])
    return 2.0 * d.mean()


# ---------------------------------------------------------------------------
# 1. Baseline: standalone GLM on target data only
# ---------------------------------------------------------------------------
# Use statsmodels Poisson GLM for proper coefficient estimates
import statsmodels.api as sm

print("\n--- Fitting models ---")

t0 = time.time()
X_sm_tgt = sm.add_constant(X_tgt_train)
glm_standalone = sm.GLM(
    y_tgt_train, X_sm_tgt,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_tgt_train, 1e-10)),
).fit(disp=False)
t_standalone = time.time() - t0

# Predict on test
X_sm_test = sm.add_constant(X_tgt_test)
y_pred_standalone = glm_standalone.predict(X_sm_test, offset=np.log(np.maximum(exp_tgt_test, 1e-10)))
dev_standalone = poisson_deviance(y_tgt_test, y_pred_standalone)

print(f"  Standalone GLM fit:    {t_standalone:.2f}s | test deviance={dev_standalone:.4f}")

# ---------------------------------------------------------------------------
# 2. Transfer learning: GLMTransfer (shared features only)
# ---------------------------------------------------------------------------
from insurance_thin_data import GLMTransfer

# NOTE: source has 4 features (veh_group, ncd, young_driver, NO subsidence)
# Target has 5 features. For transfer we use only the 4 shared features.
X_tgt_shared = X_tgt_train[:, :4]  # drop subsidence (not in source)
X_src_shared = X_src                # source has 4 features

t0 = time.time()
transfer_model = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,
    lambda_debias=0.02,
    scale_features=True,
    fit_intercept=True,
)
transfer_model.fit(
    X_tgt_shared, y_tgt_train, exp_tgt_train,
    X_source=X_src_shared,
    y_source=y_src,
    exposure_source=exp_src,
)
t_transfer = time.time() - t0

# Predict on test (shared features only — ignores subsidence)
y_pred_transfer = transfer_model.predict(X_tgt_test[:, :4], exp_tgt_test)
dev_transfer = poisson_deviance(y_tgt_test, y_pred_transfer)

print(f"  GLMTransfer fit:       {t_transfer:.2f}s | test deviance={dev_transfer:.4f}")

# ---------------------------------------------------------------------------
# 3. Oracle: GLM on large target (5,000 policies)
# ---------------------------------------------------------------------------
X_sm_oracle = sm.add_constant(X_oracle)
glm_oracle = sm.GLM(
    y_oracle, X_sm_oracle,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_oracle, 1e-10)),
).fit(disp=False)

y_pred_oracle = glm_oracle.predict(X_sm_test, offset=np.log(np.maximum(exp_tgt_test, 1e-10)))
dev_oracle = poisson_deviance(y_tgt_test, y_pred_oracle)

print(f"  Oracle GLM (n=5000):   test deviance={dev_oracle:.4f}")

# ---------------------------------------------------------------------------
# Parameter recovery: how close are coefficient estimates to true values?
# ---------------------------------------------------------------------------
print("\n--- Parameter recovery (shared features only: veh_group, ncd, young_driver) ---")
print(f"{'Feature':<20} {'True':>8} {'Standalone':>12} {'Transfer':>12}")
print("-" * 56)
# True beta for shared features (no intercept, using standardised scale from transfer):
# Standalone and oracle use statsmodels (unstandardised)
feature_names = ["vehicle_group", "ncd_years", "young_driver"]
true_vals_shared = [TRUE_BETA_TARGET[1], TRUE_BETA_TARGET[2], TRUE_BETA_TARGET[3]]

# Standalone params (statsmodels)
sa_params = glm_standalone.params[1:5]  # skip intercept, take 4 shared features
for i, (name, true_v) in enumerate(zip(feature_names, true_vals_shared)):
    sa_val = sa_params[i] if i < len(sa_params) else float('nan')
    tr_val = transfer_model.coef_[i] if i < len(transfer_model.coef_) else float('nan')
    print(f"  {name:<18} {true_v:>8.4f} {sa_val:>12.4f} {tr_val:>12.4f}")

# L2 parameter bias (vs true target params)
def param_bias(coefs, true_coefs):
    n = min(len(coefs), len(true_coefs))
    return float(np.sqrt(np.mean((coefs[:n] - true_coefs[:n]) ** 2)))

sa_bias = param_bias(glm_standalone.params[1:4], np.array(true_vals_shared))
tr_bias = param_bias(transfer_model.coef_[:3], np.array(true_vals_shared))
print(f"\n  Coefficient RMSE vs true (shared features):")
print(f"    Standalone:  {sa_bias:.4f}")
print(f"    Transfer:    {tr_bias:.4f}")

# ---------------------------------------------------------------------------
# Bootstrap CI width (parameter stability)
# ---------------------------------------------------------------------------
print("\n--- Bootstrap 90% CI width for ncd_years coefficient (200 resamples) ---")
n_boot = 200
boot_sa_coefs = []
boot_tr_coefs = []

for b in range(n_boot):
    boot_rng = np.random.default_rng(b + 1000)
    idx = boot_rng.choice(len(X_tgt_train), len(X_tgt_train), replace=True)
    Xb = X_tgt_train[idx]
    yb = y_tgt_train[idx]
    eb = exp_tgt_train[idx]

    # Standalone bootstrap
    Xb_sm = sm.add_constant(Xb)
    try:
        gb = sm.GLM(yb, Xb_sm, family=sm.families.Poisson(),
                    offset=np.log(np.maximum(eb, 1e-10))).fit(disp=False)
        boot_sa_coefs.append(gb.params[2])  # ncd_years index 2 (after const + veh_group)
    except Exception:
        pass

    # Transfer bootstrap
    try:
        tmb = GLMTransfer(family="poisson", lambda_pool=0.005,
                          lambda_debias=0.02, scale_features=True)
        tmb.fit(Xb[:, :4], yb, eb, X_source=X_src_shared,
                y_source=y_src, exposure_source=exp_src)
        boot_tr_coefs.append(tmb.coef_[1])  # ncd_years is index 1 in coef_
    except Exception:
        pass

sa_ci_width = float(np.percentile(boot_sa_coefs, 95) - np.percentile(boot_sa_coefs, 5)) if boot_sa_coefs else float('nan')
tr_ci_width = float(np.percentile(boot_tr_coefs, 95) - np.percentile(boot_tr_coefs, 5)) if boot_tr_coefs else float('nan')

print(f"  Standalone 90% CI width for ncd_years: {sa_ci_width:.4f}")
print(f"  Transfer   90% CI width for ncd_years: {tr_ci_width:.4f}")
if sa_ci_width > 0 and tr_ci_width > 0:
    narrowing = (1 - tr_ci_width / sa_ci_width) * 100
    print(f"  Transfer CI narrowing: {narrowing:.1f}%")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Standalone GLM (n=400):  deviance={dev_standalone:.4f}  param RMSE={sa_bias:.4f}")
print(f"  GLMTransfer (n=400+8k):  deviance={dev_transfer:.4f}  param RMSE={tr_bias:.4f}")
print(f"  Oracle GLM (n=5000):     deviance={dev_oracle:.4f}")
if sa_ci_width > 0 and tr_ci_width > 0:
    print(f"  Bootstrap CI narrowing (ncd_years, 200 resamples): {narrowing:.1f}%")
print(f"\n  Note: Transfer uses only {X_tgt_shared.shape[1]} shared features.")
print("  The subsidence feature (target-specific) is not present in source")
print("  and cannot be improved by transfer — it must be estimated from target data.")

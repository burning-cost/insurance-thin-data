"""
Benchmark: insurance-thin-data
===============================

Scenario: A UK property insurer launches a new niche commercial property segment.
They have 400 target policies and a related source portfolio of 8,000 policies.
The source and target share 3 features (vehicle_group, ncd_years, young_driver).
The target has one additional feature (subsidence_risk) that is not in the source.

We compare:
  1. Standalone Poisson GLM on target data only (baseline) — 3 shared features
  2. GLMTransfer (Tian & Feng 2023): penalised pooling + debiasing — 3 shared features
  3. Oracle: GLM fit on large (5,000-policy) target dataset — 3 shared features

Metrics:
  - Poisson deviance on held-out test set (150 target policies)
  - Coefficient RMSE vs known true values (3 shared parameters)
  - Bootstrap 90% CI width for ncd_years (200 resamples)

The transfer benchmark uses only the 3 features shared between source and target.
The subsidence feature (target-specific) cannot be improved by transfer.

Seed: 42. All data from known DGP.
"""

import time
import numpy as np
import statsmodels.api as sm
from insurance_thin_data import GLMTransfer

# ---------------------------------------------------------------------------
# True DGP
# ---------------------------------------------------------------------------
# 3 shared features in source and target
# Source: lower baseline, same structural effects
# Target: higher baseline + same effects

TRUE_BETA_SHARED = np.array([
    -2.5,   # intercept
     0.015, # vehicle_group
    -0.09,  # ncd_years
     0.30,  # young_driver
])

rng = np.random.default_rng(42)


def generate_source(n):
    vehicle_group = rng.integers(1, 20, n).astype(float)
    ncd_years     = rng.integers(0, 9, n).astype(float)
    young_driver  = (rng.integers(17, 85, n) < 25).astype(float)
    # Source has slightly lower baseline (different intercept)
    beta_src = TRUE_BETA_SHARED.copy()
    beta_src[0] = -2.8  # lower baseline
    X = np.column_stack([np.ones(n), vehicle_group, ncd_years, young_driver])
    exposure = rng.uniform(0.3, 1.0, n)
    log_rate = X @ beta_src + np.log(exposure)
    y = rng.poisson(np.exp(log_rate)).astype(float)
    return np.column_stack([vehicle_group, ncd_years, young_driver]), y, exposure


def generate_target(n):
    vehicle_group   = rng.integers(1, 20, n).astype(float)
    ncd_years       = rng.integers(0, 9, n).astype(float)
    young_driver    = (rng.integers(17, 85, n) < 25).astype(float)
    X_shared = np.column_stack([np.ones(n), vehicle_group, ncd_years, young_driver])
    exposure = rng.uniform(0.3, 1.0, n)
    log_rate = X_shared @ TRUE_BETA_SHARED + np.log(exposure)
    y = rng.poisson(np.exp(log_rate)).astype(float)
    return np.column_stack([vehicle_group, ncd_years, young_driver]), y, exposure


# Source: large portfolio (8,000 policies) — 3 shared features
X_src, y_src, exp_src = generate_source(8_000)

# Target: thin segment (400 train + 150 test) — 3 shared features
X_tgt_all, y_tgt_all, exp_tgt_all = generate_target(550)
X_tgt_train = X_tgt_all[:400]
y_tgt_train = y_tgt_all[:400]
exp_tgt_train = exp_tgt_all[:400]
X_tgt_test  = X_tgt_all[400:]
y_tgt_test  = y_tgt_all[400:]
exp_tgt_test = exp_tgt_all[400:]

# Oracle: large target (5,000 policies)
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
    y_pred_c = np.maximum(y_pred, 1e-10)
    mask = y_true > 0
    d = np.zeros_like(y_true, dtype=float)
    d[mask] = y_true[mask] * np.log(y_true[mask] / y_pred_c[mask]) - (y_true[mask] - y_pred_c[mask])
    d[~mask] = y_pred_c[~mask]
    return 2.0 * d.mean()


# ---------------------------------------------------------------------------
# 1. Baseline: standalone GLM on target training data
# ---------------------------------------------------------------------------
print("\n--- Fitting models ---")

t0 = time.time()
X_sm_tgt_train = sm.add_constant(X_tgt_train)
glm_standalone = sm.GLM(
    y_tgt_train, X_sm_tgt_train,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_tgt_train, 1e-10)),
).fit(disp=False)
t_standalone = time.time() - t0

X_sm_tgt_test = sm.add_constant(X_tgt_test)
y_pred_standalone = glm_standalone.predict(
    X_sm_tgt_test,
    offset=np.log(np.maximum(exp_tgt_test, 1e-10)),
)
dev_standalone = poisson_deviance(y_tgt_test, y_pred_standalone)

print(f"  Standalone GLM fit:    {t_standalone:.2f}s | test deviance={dev_standalone:.4f}")

# ---------------------------------------------------------------------------
# 2. Transfer learning: GLMTransfer (3 shared features, source + target)
# ---------------------------------------------------------------------------
t0 = time.time()
transfer_model = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,
    lambda_debias=0.02,
    scale_features=True,
    fit_intercept=True,
)
transfer_model.fit(
    X_tgt_train, y_tgt_train, exp_tgt_train,
    X_source=X_src,
    y_source=y_src,
    exposure_source=exp_src,
)
t_transfer = time.time() - t0

y_pred_transfer = transfer_model.predict(X_tgt_test, exp_tgt_test)
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

y_pred_oracle = glm_oracle.predict(
    X_sm_tgt_test,
    offset=np.log(np.maximum(exp_tgt_test, 1e-10)),
)
dev_oracle = poisson_deviance(y_tgt_test, y_pred_oracle)

print(f"  Oracle GLM (n=5000):   test deviance={dev_oracle:.4f}")

# ---------------------------------------------------------------------------
# Parameter recovery vs true values
# ---------------------------------------------------------------------------
print("\n--- Parameter recovery (3 shared features) ---")
true_vals = [TRUE_BETA_SHARED[1], TRUE_BETA_SHARED[2], TRUE_BETA_SHARED[3]]
feature_names = ["vehicle_group", "ncd_years", "young_driver"]
print(f"  {'Feature':<20} {'True':>8} {'Standalone':>12} {'Transfer':>12}")
print("  " + "-" * 56)

sa_params = list(glm_standalone.params[1:4])  # skip intercept
tr_coefs = list(transfer_model.coef_[:3])
for i, (name, true_v) in enumerate(zip(feature_names, true_vals)):
    sa_v = sa_params[i] if i < len(sa_params) else float('nan')
    tr_v = tr_coefs[i] if i < len(tr_coefs) else float('nan')
    print(f"  {name:<20} {true_v:>8.4f} {sa_v:>12.4f} {tr_v:>12.4f}")

sa_bias = float(np.sqrt(np.mean([(sa_params[i] - true_vals[i])**2 for i in range(min(3, len(sa_params)))])))
tr_bias = float(np.sqrt(np.mean([(tr_coefs[i] - true_vals[i])**2 for i in range(min(3, len(tr_coefs)))])))
print(f"\n  Coefficient RMSE vs true:")
print(f"    Standalone:  {sa_bias:.4f}")
print(f"    Transfer:    {tr_bias:.4f}")

# ---------------------------------------------------------------------------
# Bootstrap CI width (parameter stability)
# ---------------------------------------------------------------------------
print("\n--- Bootstrap 90% CI width for ncd_years (200 resamples) ---")
boot_sa_coefs = []
boot_tr_coefs = []

for b in range(200):
    boot_rng = np.random.default_rng(b + 1000)
    idx = boot_rng.choice(len(X_tgt_train), len(X_tgt_train), replace=True)
    Xb = X_tgt_train[idx]
    yb = y_tgt_train[idx]
    eb = exp_tgt_train[idx]

    try:
        gb = sm.GLM(
            yb, sm.add_constant(Xb),
            family=sm.families.Poisson(),
            offset=np.log(np.maximum(eb, 1e-10)),
        ).fit(disp=False)
        boot_sa_coefs.append(float(gb.params[2]))  # ncd_years
    except Exception:
        pass

    try:
        tmb = GLMTransfer(family="poisson", lambda_pool=0.005,
                          lambda_debias=0.02, scale_features=True)
        tmb.fit(Xb, yb, eb, X_source=X_src, y_source=y_src, exposure_source=exp_src)
        boot_tr_coefs.append(float(tmb.coef_[1]))  # ncd_years
    except Exception:
        pass

sa_ci_width = float(np.percentile(boot_sa_coefs, 95) - np.percentile(boot_sa_coefs, 5)) if boot_sa_coefs else float('nan')
tr_ci_width = float(np.percentile(boot_tr_coefs, 95) - np.percentile(boot_tr_coefs, 5)) if boot_tr_coefs else float('nan')
print(f"  Standalone 90% CI width: {sa_ci_width:.4f}")
print(f"  Transfer   90% CI width: {tr_ci_width:.4f}")
narrowing = float('nan')
if sa_ci_width > 0 and tr_ci_width > 0 and sa_ci_width == sa_ci_width:
    narrowing = (1 - tr_ci_width / sa_ci_width) * 100
    print(f"  Narrowing: {narrowing:.1f}%")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Standalone GLM (n=400):  deviance={dev_standalone:.4f}  param RMSE={sa_bias:.4f}")
print(f"  GLMTransfer (n=400+8k):  deviance={dev_transfer:.4f}  param RMSE={tr_bias:.4f}")
print(f"  Oracle GLM (n=5000):     deviance={dev_oracle:.4f}")
if narrowing == narrowing:
    print(f"  Bootstrap CI narrowing (ncd_years, 200 resamples): {narrowing:.1f}%")

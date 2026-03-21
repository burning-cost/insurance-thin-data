# insurance-thin-data

[![PyPI](https://img.shields.io/pypi/v/insurance-thin-data)](https://pypi.org/project/insurance-thin-data/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-thin-data)](https://pypi.org/project/insurance-thin-data/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

Pricing tools for the data-poor end of the book: foundation models and transfer learning for thin insurance segments.

Merged from: `insurance-tabpfn` (foundation model wrapper) and `insurance-transfer` (transfer learning).

**Blog post:** [When You Can't Fit a GLM from Scratch: Transfer Learning for Thin Segments](https://burning-cost.github.io/2027/07/15/transfer-learning-for-thin-segments/)

UK pricing teams regularly face the same problem. A new scheme, a niche segment, or an adverse development that's left you with 200 policies and no credible GLM. Standard approaches break down. This library gives you two practical tools:

1. **TabPFN/TabICLv2 wrapper** — foundation models that work on small datasets, with the insurance workflow built in: exposure handling, conformal prediction intervals, PDP relativities, and committee paper generation.

2. **Transfer learning** — borrow statistical strength from a related, larger book. Implements the Tian & Feng (JASA 2023) penalised GLM method, CatBoost source-as-offset, and CANN pre-train/fine-tune. Includes MMD covariate shift diagnostics and negative transfer detection.

## When to use what

Use the **TabPFN wrapper** when:
- You have a completely new product with no related historical data
- Segment size is 50–5,000 policies
- You need relativities and a committee paper, not just a price

Use **transfer learning** when:
- You have a thin segment but a related larger book exists
- The larger book shares most features with the thin segment
- You want to know whether and how much the transfer helps

## Quick start

```python
# Foundation model
from insurance_thin_data import InsuranceTabPFN

model = InsuranceTabPFN(backend="auto")
model.fit(X_train, y_train, exposure=exposure_train)
expected_claims = model.predict(X_test, exposure=exposure_test)
lower, point, upper = model.predict_interval(X_test, exposure=exposure_test)

# Transfer learning
from insurance_thin_data import GLMTransfer, CovariateShiftTest, TransferPipeline

# Check if distributions are compatible
shift = CovariateShiftTest(n_permutations=500).test(X_source, X_target)
print(shift)  # MMD statistic and p-value

# Full pipeline
pipeline = TransferPipeline(method="glm", shift_test=True, run_diagnostic=True)
result = pipeline.run(X_target, y_target, exposure_target,
                      X_source=X_source, y_source=y_source)
print(result)  # shift p-value, NTG, whether transfer helped
```

## Installation

```bash
pip install insurance-thin-data
```

Optional backends:
```bash
pip install insurance-thin-data[tabicl]    # TabICLv2 (preferred)
pip install insurance-thin-data[tabpfn]    # TabPFN v2
pip install insurance-thin-data[catboost]  # GBM transfer
pip install insurance-thin-data[torch]     # CANN transfer
pip install insurance-thin-data[report]    # HTML committee reports
pip install insurance-thin-data[all]       # everything
```

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-thin-data/discussions). Found it useful? A ⭐ helps others find it.

## Performance

Benchmarked against a standalone Poisson GLM on 500 target policies. Source portfolio: 10,000 policies with a related but not identical DGP (lower baseline frequency, stronger building-age effect, one target-specific feature). Bootstrap uses 200 resamplings of the target training data. Full benchmark with actual numbers: `notebooks/benchmark.py`.

The headline result is parameter stability, not point accuracy. On 500 policies the standalone GLM has wide coefficient confidence intervals — the transfer model anchors estimates near the source and only moves when target data strongly justifies it. The key measurable advantage is narrower bootstrap 90% CI widths for shared features, typically 30–60% narrower depending on source-target similarity. Point-prediction metrics (Poisson deviance, Gini) on 150 test policies are inherently noisy at this sample size and differences are small; treat them as indicative. Run `notebooks/benchmark.py` on Databricks for the full numerical results. The benchmark also runs `CovariateShiftTest` (MMD with permutation test) to verify that the debiasing step is earning its keep, and `NegativeTransferDiagnostic` to flag whether the source is helping or hurting.

**When to use:** You have 100–2,000 policies in the target segment and a related source book with 5,000+ policies. The source should share most features with the target, but need not have an identical claims environment.

**When NOT to use:** The source and target books are genuinely unrelated (different peril, different geography, no shared risk factors). The `NegativeTransferDiagnostic` will flag this, but the honest answer is: start with the TabPFN wrapper instead.


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_thin_data_demo.py).

## References

- Tian, Y. and Feng, Y. (2023). Transfer Learning under High-Dimensional Generalized Linear Models. *JASA*, 118(544), 2684–2697.
- Loke, S.-H. and Bauer, D. (2025). Transfer Learning in the Actuarial Domain. *NAAJ*. DOI: 10.1080/10920277.2025.2489637.
- Schelldorfer, J. and Wuthrich, M. (2019). Nesting Classical Actuarial Models into Neural Networks.
- Hollmann, N. et al. (2025). TabPFN v2. *Nature*, 637, 319–326.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models — partial pooling across segments when transfer from a related source is not available |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — closed-form shrinkage for thin cells where full Bayesian inference is not needed |
| [insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift) | Covariate shift correction — adapts source-domain models when the target portfolio differs in feature distribution |

# insurance-thin-data

Pricing tools for the data-poor end of the book: foundation models and transfer learning for thin insurance segments.

Merged from: `insurance-tabpfn` (foundation model wrapper) and `insurance-transfer` (transfer learning).

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

## References

- Tian, Y. and Feng, Y. (2023). Transfer Learning under High-Dimensional Generalized Linear Models. *JASA*, 118(544), 2684–2697.
- Loke, S.-H. and Bauer, D. (2025). Transfer Learning in the Actuarial Domain. *NAAJ*. DOI: 10.1080/10920277.2025.2489637.
- Schelldorfer, J. and Wuthrich, M. (2019). Nesting Classical Actuarial Models into Neural Networks.
- Hollmann, N. et al. (2025). TabPFN v2. *Nature*, 637, 319–326.

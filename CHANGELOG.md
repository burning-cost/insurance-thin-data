# Changelog

## v0.1.4 (2026-03-22) [unreleased]
- feat: add Databricks benchmark notebook
- fix: add missing Issues and Documentation URLs to pyproject.toml
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.4 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- Add MIT license
- fix: handle pandas DataFrame input in GLMTransfer and CovariateShiftTest
- Fix benchmark: align source and target feature dimensions for GLMTransfer
- Add benchmark: GLMTransfer vs standalone GLM on thin segment (400 target policies)
- Fix P0/P1/P2 quality audit issues
- Add PyPI classifiers for financial/insurance audience
- Relax floating-point tolerance in Poisson deviance regression test
- Fix P0/P1 bugs and add regression tests (v0.1.3)
- pin statsmodels>=0.14.5 for scipy compat
- fix: use DataFrame.assign() instead of iloc setitem in RelativitiesExtractor
- fix: remove polars from required dependencies (0.1.1)
- Add shields.io badge row to README
- docs: add Databricks notebook link
- Add Related Libraries section to README
- fix: use pd.api.types.is_numeric_dtype for categorical detection


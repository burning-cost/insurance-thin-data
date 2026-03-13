"""Tests for CommitteeReport generation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from insurance_thin_data.tabpfn import InsuranceTabPFN, CommitteeReport
from insurance_thin_data.tabpfn.benchmark import ComparisonResult, BenchmarkResult, _double_lift
from insurance_thin_data.tabpfn.report import ReportConfig


@pytest.fixture
def sample_comparison():
    rng = np.random.default_rng(0)
    n = 50
    y = rng.poisson(0.1, size=n).astype(float)
    yhat = rng.uniform(0.05, 0.2, size=n)
    exp = rng.uniform(0.5, 1.0, size=n)
    dl = _double_lift(y, yhat, None, exp)
    tabpfn_result = BenchmarkResult(
        model_name="InsuranceTabPFN",
        gini=0.32,
        poisson_deviance=0.041,
        rmse=0.089,
        double_lift=dl,
        n_samples=n,
        exposure_total=float(exp.sum()),
    )
    glm_result = BenchmarkResult(
        model_name="Poisson GLM",
        gini=0.28,
        poisson_deviance=0.055,
        rmse=0.095,
        double_lift=dl,
        n_samples=n,
        exposure_total=float(exp.sum()),
    )
    return ComparisonResult(tabpfn=tabpfn_result, glm=glm_result)


@pytest.fixture
def simple_model():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"age": rng.uniform(18, 80, size=n)})
    y = rng.poisson(0.08, size=n).astype(float)
    model = InsuranceTabPFN(backend="mock", random_state=0)
    model.fit(X, y)
    return model


def test_to_json_basic(simple_model, sample_comparison):
    report = CommitteeReport(simple_model)
    report.add_benchmark(sample_comparison)
    json_str = report.to_json()
    data = json.loads(json_str)

    assert "title" in data
    assert "model_card" in data
    assert "limitations" in data
    assert len(data["limitations"]) > 0


def test_json_contains_benchmark(simple_model, sample_comparison):
    report = CommitteeReport(simple_model)
    report.add_benchmark(sample_comparison)
    data = json.loads(report.to_json())

    assert "benchmark" in data
    assert len(data["benchmark"]) == 2  # TabPFN + GLM rows


def test_json_winner_correct(simple_model, sample_comparison):
    report = CommitteeReport(simple_model)
    report.add_benchmark(sample_comparison)
    data = json.loads(report.to_json())
    # TabPFN gini=0.32 > GLM gini=0.28
    assert data["winner"] == "InsuranceTabPFN"


def test_json_no_benchmark(simple_model):
    report = CommitteeReport(simple_model)
    data = json.loads(report.to_json())
    assert "benchmark" not in data


def test_json_with_coverage(simple_model):
    rng = np.random.default_rng(0)
    n = 50
    y = rng.poisson(0.1, size=n).astype(float)
    lower = y * 0.5
    upper = y * 1.8
    point = y * 1.1

    report = CommitteeReport(simple_model)
    report.add_coverage(lower, point, upper, y)
    data = json.loads(report.to_json())

    assert "coverage" in data
    assert "empirical_coverage_%" in data["coverage"]


def test_json_limitations_mandatory(simple_model):
    """All 5 mandatory limitations must appear in output."""
    report = CommitteeReport(simple_model)
    data = json.loads(report.to_json())
    assert len(data["limitations"]) == 5
    # Check key phrases present
    full_text = " ".join(data["limitations"]).lower()
    assert "exposure offset" in full_text
    assert "gaussian" in full_text
    assert "black-box" in full_text
    assert "5,000" in full_text


def test_json_limitations_suppressed(simple_model):
    config = ReportConfig(include_limitations=False)
    report = CommitteeReport(simple_model, config=config)
    data = json.loads(report.to_json())
    assert data["limitations"] == []


def test_html_requires_jinja2(simple_model):
    """HTML generation raises ImportError when jinja2 is not installed."""
    try:
        import jinja2  # noqa: F401
        pytest.skip("jinja2 is installed — skipping no-jinja2 test")
    except ImportError:
        pass

    report = CommitteeReport(simple_model)
    with pytest.raises(ImportError, match="jinja2"):
        report.to_html()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("jinja2"),
    reason="jinja2 not installed",
)
def test_html_contains_key_sections(simple_model, sample_comparison):
    config = ReportConfig(title="Test Motor Segment Report")
    report = CommitteeReport(simple_model, config=config)
    report.add_benchmark(sample_comparison)
    html = report.to_html()

    assert "Test Motor Segment Report" in html
    assert "Limitations" in html
    assert "Benchmark" in html
    assert "Poisson GLM" in html
    assert "InsuranceTabPFN" in html
    assert "exposure offset" in html.lower()


def test_report_config_defaults():
    config = ReportConfig()
    assert config.include_limitations is True
    assert config.n_double_lift_deciles == 10


def test_report_with_relativities(simple_model):
    from insurance_thin_data.tabpfn import RelativitiesExtractor
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"age": rng.uniform(18, 80, size=n)})

    extractor = RelativitiesExtractor(simple_model, n_grid_points=5, n_sample_rows=30)
    ft = extractor.to_factor_table(X)

    report = CommitteeReport(simple_model)
    report.add_relativities(ft)
    data = json.loads(report.to_json())
    assert "relativities" in data
    assert len(data["relativities"]) > 0

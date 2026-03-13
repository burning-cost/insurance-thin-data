"""
CommitteeReport: generate HTML/JSON model card for pricing committee submissions.

The UK actuarial standard for a new segment pricing model requires:
  1. Model description and data sources
  2. Benchmark against standard approach (Poisson GLM)
  3. Key factor relativities
  4. Coverage/calibration check
  5. Explicit limitations section — this is not optional for model governance

The limitations section is particularly important here because TabPFN/TabICL:
  - Has no true Poisson offset (documented exposure workaround)
  - Uses Gaussian-equivalent regression (not Poisson/Gamma likelihood)
  - Is a black-box ICL model (no coefficient interpretation)
  - Has a hard limit on training data size

These limitations do not prevent use — the PRA accepts black-box models for
lower-materiality applications when accompanied by benchmark comparison and
explicit disclosure. But they must be disclosed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from insurance_thin_data.tabpfn.benchmark import ComparisonResult


# Jinja2 template (inline to keep the library self-contained)
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { color: #444; margin-top: 32px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th { background: #f0f0f0; text-align: left; padding: 8px 12px; border: 1px solid #ccc; }
  td { padding: 6px 12px; border: 1px solid #ddd; }
  tr:nth-child(even) { background: #fafafa; }
  .warning { background: #fff3cd; border: 1px solid #ffc107; padding: 12px; margin: 12px 0; border-radius: 4px; }
  .limitation { background: #f8d7da; border: 1px solid #f5c6cb; padding: 12px; margin: 8px 0; border-radius: 4px; }
  .meta { color: #666; font-size: 0.9em; }
  .winner { font-weight: bold; color: #2a6; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p class="meta">Generated: {{ generated_at }} | Version: {{ model_version }}</p>

<h2>Model Card</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  {% for k, v in model_card.items() %}
  <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
  {% endfor %}
</table>

{% if limitations %}
<h2>Limitations</h2>
<div class="warning"><strong>Required disclosure for pricing committee.</strong></div>
{% for lim in limitations %}
<div class="limitation">{{ lim }}</div>
{% endfor %}
{% endif %}

{% if benchmark_table %}
<h2>Benchmark vs Poisson GLM</h2>
{{ benchmark_table }}
{% if winner %}
<p>Best Gini: <span class="winner">{{ winner }}</span></p>
{% endif %}
{% endif %}

{% if double_lift_table %}
<h2>Double-Lift Chart (tabular)</h2>
{{ double_lift_table }}
{% endif %}

{% if relativities_table %}
<h2>Key Factor Relativities (PDP-based)</h2>
{{ relativities_table }}
{% endif %}

{% if coverage %}
<h2>Prediction Interval Coverage</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {% for k, v in coverage.items() %}
  <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
  {% endfor %}
</table>
{% endif %}

</body>
</html>"""


@dataclass
class ReportConfig:
    """Configuration for CommitteeReport generation."""

    title: str = "InsuranceTabPFN Model Report"
    segment_name: str = "Thin Segment"
    model_version: str = "0.1.0"
    analyst: str = "Pricing Team"
    notes: str = ""
    include_limitations: bool = True
    include_double_lift: bool = True
    n_double_lift_deciles: int = 10


class CommitteeReport:
    """
    Generate an HTML or JSON committee report for a fitted InsuranceTabPFN model.

    Parameters
    ----------
    model : fitted InsuranceTabPFN
    config : ReportConfig, optional

    Examples
    --------
        report = CommitteeReport(model, config=ReportConfig(
            title="Motor Telematics New Business — Thin Segment",
            segment_name="Young Drivers <25 (<300 policies)",
        ))
        report.add_benchmark(comparison_result)
        report.add_relativities(factor_table_df)
        report.add_coverage(lower, point, upper, y_actual)
        html = report.to_html()
        with open("committee_report.html", "w") as f:
            f.write(html)
    """

    # Limitations that must appear in all committee reports for TabPFN-based models
    MANDATORY_LIMITATIONS = [
        (
            "No true Poisson exposure offset: TabPFN has no exposure offset parameter. "
            "This implementation appends log(exposure) as a feature and fits claim rate "
            "(claims/exposure) as the target. This approximates but does not replicate "
            "the standard Poisson GLM log-offset. Calibration at extreme exposure values "
            "may be degraded."
        ),
        (
            "Gaussian regression assumption: TabPFN and TabICLv2 use Gaussian-equivalent "
            "regression priors. The model is not trained with Poisson or Gamma likelihood. "
            "For frequency modelling this means mean-variance relationship is not "
            "automatically handled. The Poisson deviance benchmark metric is computed "
            "post-hoc, not during training."
        ),
        (
            "Black-box in-context learning: The model produces predictions by performing "
            "a single forward pass over the training set as context. There are no learned "
            "coefficients. Factor relativities are computed via partial dependence (PDP), "
            "not coefficient extraction. PDPs are a marginal approximation and assume "
            "no interaction structure unless features are correlated."
        ),
        (
            "Thin-segment scope: This model is intended for segments with < 5,000 policies. "
            "Above this threshold, a Poisson GLM with cross-validation will typically "
            "achieve better Gini and is more defensible under PRA SS1/24 model validation "
            "requirements. Do not use this model as a replacement for a stable GLM on "
            "large, mature books."
        ),
        (
            "Training data limit: TabPFN has a hard limit of approximately 10,000 training "
            "rows. TabICLv2 has similar constraints. Exceeding this limit will cause "
            "degraded performance or runtime errors."
        ),
    ]

    def __init__(self, model, config: Optional[ReportConfig] = None) -> None:
        self.model = model
        self.config = config or ReportConfig()
        self._comparison: Optional[ComparisonResult] = None
        self._relativities: Optional[pd.DataFrame] = None
        self._coverage_metrics: Optional[dict] = None

    def add_benchmark(self, comparison: ComparisonResult) -> "CommitteeReport":
        """Add GLM benchmark comparison results."""
        self._comparison = comparison
        return self

    def add_relativities(self, factor_table: pd.DataFrame) -> "CommitteeReport":
        """Add PDP-based relativities table (output of RelativitiesExtractor)."""
        self._relativities = factor_table
        return self

    def add_coverage(
        self,
        lower: "np.ndarray",
        point: "np.ndarray",
        upper: "np.ndarray",
        y_actual: "np.ndarray",
    ) -> "CommitteeReport":
        """
        Add prediction interval coverage statistics.

        Computes empirical coverage (what fraction of actuals fall within intervals)
        and mean interval width.
        """
        y = np.asarray(y_actual, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)

        in_interval = (y >= lo) & (y <= hi)
        coverage_pct = float(np.mean(in_interval)) * 100.0
        mean_width = float(np.mean(hi - lo))
        target_coverage = (1.0 - 0.1) * 100.0  # default alpha=0.1

        self._coverage_metrics = {
            "target_coverage_%": f"{target_coverage:.0f}%",
            "empirical_coverage_%": f"{coverage_pct:.1f}%",
            "mean_interval_width": f"{mean_width:.4f}",
            "n_test_samples": len(y),
            "calibrated": "Yes" if abs(coverage_pct - target_coverage) < 5 else "No — check conformal calibration",
        }
        return self

    def to_html(self) -> str:
        """Render the report as an HTML string."""
        try:
            from jinja2 import Template  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "jinja2 is required for HTML reports. "
                "Run: pip install insurance-thin-data[report]"
            ) from e

        model_card = self._build_model_card()
        limitations = self.MANDATORY_LIMITATIONS if self.config.include_limitations else []

        benchmark_html: Optional[str] = None
        double_lift_html: Optional[str] = None
        winner: Optional[str] = None
        if self._comparison is not None:
            benchmark_html = self._comparison.to_dataframe().to_html(index=False)
            winner = self._comparison.winner()
            if self.config.include_double_lift:
                double_lift_html = self._comparison.tabpfn.double_lift.to_html(index=False)

        relat_html: Optional[str] = None
        if self._relativities is not None:
            relat_html = self._relativities.to_html(index=False)

        template = Template(_HTML_TEMPLATE)
        return template.render(
            title=self.config.title,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            model_version=self.config.model_version,
            model_card=model_card,
            limitations=limitations,
            benchmark_table=benchmark_html,
            double_lift_table=double_lift_html,
            winner=winner,
            relativities_table=relat_html,
            coverage=self._coverage_metrics,
        )

    def to_json(self) -> str:
        """Render the report as a JSON string for machine consumption."""
        model_card = self._build_model_card()

        payload: dict = {
            "title": self.config.title,
            "generated_at": datetime.utcnow().isoformat(),
            "model_card": model_card,
            "limitations": self.MANDATORY_LIMITATIONS if self.config.include_limitations else [],
        }

        if self._comparison is not None:
            payload["benchmark"] = self._comparison.to_dataframe().to_dict(orient="records")
            payload["winner"] = self._comparison.winner()
            if self.config.include_double_lift:
                payload["double_lift"] = self._comparison.tabpfn.double_lift.to_dict(
                    orient="records"
                )

        if self._relativities is not None:
            payload["relativities"] = self._relativities.to_dict(orient="records")

        if self._coverage_metrics is not None:
            payload["coverage"] = self._coverage_metrics

        return json.dumps(payload, indent=2, default=str)

    def _build_model_card(self) -> dict:
        backend_name = getattr(
            getattr(self.model, "_backend", None), "name", "unknown"
        )
        n_train = getattr(self.model, "_n_features_in", "unknown")
        has_exposure = getattr(self.model, "_has_exposure", False)

        return {
            "Segment": self.config.segment_name,
            "Model type": "In-Context Learning (foundation model)",
            "Backend": backend_name,
            "Features (incl. log-exposure)": n_train,
            "Exposure handling": "log(exposure) feature + rate target" if has_exposure else "None",
            "Prediction intervals": "Split conformal (distribution-free)",
            "Analyst": self.config.analyst,
            "Notes": self.config.notes or "—",
        }

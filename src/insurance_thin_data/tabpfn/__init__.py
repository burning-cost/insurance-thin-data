"""
insurance_thin_data.tabpfn: Foundation model wrapper for thin-data insurance pricing.

Wraps TabPFN v2 and TabICLv2 with insurance-specific workflow: exposure handling,
GLM benchmark comparison, PDP-based relativities extraction, and committee paper
generation.

Use case: small books, new products, schemes with <5,000 policies where you
don't have enough data for a stable GLM but still need defensible relativities.
"""

from insurance_thin_data.tabpfn.model import InsuranceTabPFN
from insurance_thin_data.tabpfn.benchmark import GLMBenchmark, BenchmarkResult
from insurance_thin_data.tabpfn.relativities import RelativitiesExtractor
from insurance_thin_data.tabpfn.report import CommitteeReport
from insurance_thin_data.tabpfn.validators import validate_inputs, ThinSegmentWarning

__all__ = [
    "InsuranceTabPFN",
    "GLMBenchmark",
    "BenchmarkResult",
    "RelativitiesExtractor",
    "CommitteeReport",
    "validate_inputs",
    "ThinSegmentWarning",
]

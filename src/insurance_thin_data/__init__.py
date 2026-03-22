"""
insurance-thin-data: Foundation models and transfer learning for thin-data insurance pricing.

Merges two specialist libraries into one package:

  tabpfn subpackage — Foundation model wrapper:
    TabPFN v2 / TabICLv2 backend abstraction with insurance-specific exposure
    handling, conformal prediction intervals, PDP-based relativities, and
    committee paper generation. For books with < 5,000 policies.

  transfer subpackage — Transfer learning:
    Covariate shift diagnostics (MMD permutation test), penalised GLM transfer
    (Tian & Feng, JASA 2023), GBM source-as-offset (CatBoost), and CANN
    pre-train / fine-tune (PyTorch). For thin segments that can borrow strength
    from a related, larger book.

Quick start:

    # Foundation model approach
    from insurance_thin_data import InsuranceTabPFN
    model = InsuranceTabPFN(backend="auto")
    model.fit(X_train, y_train, exposure=exposure_train)

    # Transfer learning approach
    from insurance_thin_data import GLMTransfer, CovariateShiftTest
    shift = CovariateShiftTest().test(X_source, X_target)
    transfer = GLMTransfer(family="poisson")
    transfer.fit(X_tgt, y_tgt, exposure_tgt, X_source=X_src, y_source=y_src)

    # Full pipeline
    from insurance_thin_data import TransferPipeline
    result = TransferPipeline(method="glm").run(X_tgt, y_tgt, exposure_tgt,
                                                 X_source=X_src, y_source=y_src)
"""

# TabPFN subpackage
from insurance_thin_data.tabpfn import (
    InsuranceTabPFN,
    GLMBenchmark,
    BenchmarkResult,
    RelativitiesExtractor,
    CommitteeReport,
    ThinSegmentWarning,
)

# Transfer subpackage
from insurance_thin_data.transfer import (
    CovariateShiftTest,
    ShiftTestResult,
    GLMTransfer,
    GBMTransfer,
    CANNTransfer,
    NegativeTransferDiagnostic,
    TransferDiagnosticResult,
    TransferPipeline,
)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-thin-data")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    # tabpfn
    "InsuranceTabPFN",
    "GLMBenchmark",
    "BenchmarkResult",
    "RelativitiesExtractor",
    "CommitteeReport",
    "ThinSegmentWarning",
    # transfer
    "CovariateShiftTest",
    "ShiftTestResult",
    "GLMTransfer",
    "GBMTransfer",
    "CANNTransfer",
    "NegativeTransferDiagnostic",
    "TransferDiagnosticResult",
    "TransferPipeline",
    "__version__",
]

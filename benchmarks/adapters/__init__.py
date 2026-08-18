"""Method adapters used by the benchmark protocol."""

from .base import (
    AdapterError,
    AdapterUnavailable,
    BenchmarkAdapter,
    FitArtifact,
    PreflightReport,
    UnsupportedPrediction,
)
from .external import (
    HHY2021Adapter,
    ZSY2026AuthorCodeAdapter,
    ZW2015Adapter,
    ZY2025Adapter,
    ZZW2020Adapter,
)
from .trace import TraceVCAMAdapter

__all__ = [
    "AdapterError",
    "AdapterUnavailable",
    "BenchmarkAdapter",
    "FitArtifact",
    "HHY2021Adapter",
    "PreflightReport",
    "TraceVCAMAdapter",
    "UnsupportedPrediction",
    "ZSY2026AuthorCodeAdapter",
    "ZW2015Adapter",
    "ZY2025Adapter",
    "ZZW2020Adapter",
]

"""Auditable benchmark infrastructure for the VCAM study.

The package deliberately separates a method's *scientific applicability* from
whether its software happens to be available on the current machine.  A method
that is out of scope is recorded as ``N/A by design``; an applicable method
whose dependency fails is recorded as an attempted failure and remains in the
failure-rate denominator.
"""

from .admission import AdmissionDecision, PublishedTarget, assess_reproduction
from .data import (
    SubjectDataset,
    SubjectSplit,
    make_repeated_subject_folds,
    read_exchange_bundle,
    write_exchange_bundle,
)
from .methods import (
    FIXED_METHOD_LABELS,
    METHOD_SPECS,
    Protocol,
    applicability_for,
)
from .protocol import (
    BenchmarkResult,
    CurveEstimate,
    FailureInfo,
    PredictionRecord,
    audit_replication_cohort,
    metric_summary,
    run_replication,
)

__all__ = [
    "AdmissionDecision",
    "BenchmarkResult",
    "CurveEstimate",
    "FailureInfo",
    "FIXED_METHOD_LABELS",
    "METHOD_SPECS",
    "PredictionRecord",
    "Protocol",
    "PublishedTarget",
    "SubjectDataset",
    "SubjectSplit",
    "applicability_for",
    "assess_reproduction",
    "audit_replication_cohort",
    "make_repeated_subject_folds",
    "metric_summary",
    "read_exchange_bundle",
    "run_replication",
    "write_exchange_bundle",
]

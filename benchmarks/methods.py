"""Fixed method labels, provenance, and design-level applicability decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MethodLabel(str, Enum):
    TRACE_VCAM = "TRACE-VCAM"
    ZW2015 = "ZW2015"
    ZZW2020 = "ZZW2020"
    HHY2021_HUBER = "HHY2021-Huber"
    ZSY2026_AUTHOR_CODE = "ZSY2026-author-code"
    ZY2025 = "ZY2025-paper-implementation"


FIXED_METHOD_LABELS: tuple[str, ...] = tuple(item.value for item in MethodLabel)


class Protocol(str, Enum):
    REPRO_ZW2015 = "reproduction/ZW2015"
    REPRO_ZZW2020 = "reproduction/ZZW2020"
    REPRO_HHY2021 = "reproduction/HHY2021-Huber"
    REPRO_ZSY2026 = "reproduction/ZSY2026-author-code"
    REPRO_ZY2025 = "reproduction/ZY2025-paper-implementation"
    EXAMPLE1_DENSE = "example-1/dense-functional"
    EXAMPLE2_GAUSSIAN = "example-2/sparse-longitudinal-gaussian"
    EXAMPLE2_HEAVY_TAIL = "example-2/sparse-longitudinal-heavy-tail"
    EXAMPLE2_CONTAMINATION = "example-2/sparse-longitudinal-contamination"
    EXAMPLE3_HIGH_DIMENSIONAL = "example-3/high-dimensional-p10"
    EXAMPLE3_SYMMETRIC_CONTAMINATION = "example-3/high-dimensional-p10-contamination"
    EXAMPLE4_ROBUST_NORMAL = "example-4/robust-normal"
    EXAMPLE4_ROBUST_HEAVY_TAIL = "example-4/robust-t2"
    EXAMPLE4_ROBUST_CONTAMINATION = "example-4/robust-mixed-normal"
    # Where the contamination enters, and whether the number of visits is
    # informative.  The covariate, cluster, and component construction is the
    # sparse-longitudinal Example 2 design in every one of these.
    SCOPE_CLEAN = "scope/clean"
    SCOPE_RESPONSE = "scope/response-contamination"
    SCOPE_SUBJECT = "scope/subject-contamination"
    SCOPE_TRAJECTORY = "scope/trajectory-contamination"
    SCOPE_LEVERAGE = "scope/leverage-contamination"
    SCOPE_INFORMATIVE_CLEAN = "scope/informative-cluster-size-clean"
    SCOPE_INFORMATIVE_SUBJECT = "scope/informative-cluster-size-subject"
    SCALING = "scaling/p10-p25-p50"
    MACS_CD4 = "application/MACS-CD4"


class Applicability(str, Enum):
    APPLICABLE = "applicable"
    N_A_BY_DESIGN = "N/A by design"
    UNAVAILABLE_NOT_EVALUATED = "unavailable/not evaluated"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    display_name: str
    version: str
    source: str
    implementation: str
    prediction_capability: str
    selection_capability: bool
    notes: str


@dataclass(frozen=True)
class ApplicabilityDecision:
    status: Applicability
    reason: str
    supported_metrics: tuple[str, ...] = ()

    @property
    def is_applicable(self) -> bool:
        return self.status is Applicability.APPLICABLE


METHOD_SPECS: dict[str, MethodSpec] = {
    MethodLabel.TRACE_VCAM.value: MethodSpec(
        label=MethodLabel.TRACE_VCAM.value,
        display_name="TRACE-VCAM",
        version="workspace-two-stage-v1",
        source="local audited implementation",
        implementation="Python",
        prediction_capability="new-subject population prediction",
        selection_capability=True,
        notes="Convex subject-balanced Huber/nuclear/roughness pilot followed by fixed-direction scalar refit.",
    ),
    MethodLabel.ZW2015.value: MethodSpec(
        label=MethodLabel.ZW2015.value,
        display_name="Two-step spline VCAM (Zhang & Wang, 2015)",
        version="fdapace::VCAM (runtime version recorded)",
        source="CRAN fdapace original implementation",
        implementation="R",
        prediction_capability="dense functional design supported by original API",
        selection_capability=False,
        notes="Used only for the dense, time-invariant-covariate design for which the original method was written.",
    ),
    MethodLabel.ZZW2020.value: MethodSpec(
        label=MethodLabel.ZZW2020.value,
        display_name="Backfitting VCAM (Zhang et al., 2020)",
        version="paper-Algorithm-1/2020",
        source="paper specification; no verified public author code",
        implementation="audited R/Python paper-aligned implementation",
        prediction_capability="population prediction",
        selection_capability=False,
        notes="Two initializations, inner/outer backfitting, normalization, and the registered source-paper tuning rule are audited explicitly.",
    ),
    MethodLabel.HHY2021_HUBER.value: MethodSpec(
        label=MethodLabel.HHY2021_HUBER.value,
        display_name="Three-step M-VCAM (Hu et al., 2021)",
        version="paper-three-stage/2021",
        source="paper and official supplement; no verified public author code",
        implementation="audited R/Python paper-aligned implementation",
        prediction_capability="population prediction",
        selection_capability=False,
        notes="Tensor pilot, additive M-step, varying-coefficient M-step, subject-balanced Huber loss, delta=1.345.",
    ),
    MethodLabel.ZSY2026_AUTHOR_CODE.value: MethodSpec(
        label=MethodLabel.ZSY2026_AUTHOR_CODE.value,
        display_name="VCAM-Lasso (Zhao et al., 2026)",
        version="VCAMLasso-0.1.0@27d857a71807de807761a022a4e334745737761e",
        source="https://github.com/yyh198841/vcampackage",
        implementation="unmodified vendored R source",
        prediction_capability="in-sample fitted rows only; author code exposes no out-of-sample predict method",
        selection_capability=True,
        notes="No silent fixes: internal row-level cv.glmnet, stopping behavior, normalization, and returned MSE are preserved and audited.",
    ),
    MethodLabel.ZY2025.value: MethodSpec(
        label=MethodLabel.ZY2025.value,
        display_name="Penalized robust VCAM (Zhao & Yang, 2025)",
        version="paper-equations/2025",
        source="Zhao and Yang (2025) full text supplied by the author; no author code or supplement",
        implementation="auditable paper implementation (never labelled author code)",
        prediction_capability="population prediction",
        selection_capability=True,
        notes="Squared loss with coefficientwise L1 regularization, paper normalization, and ten-fold minimum-error CV.",
    ),
}


_COMMON_METRICS = (
    "component_ise",
    "factor_ise",
    "noise_free_test_mspe",
    "test_mse",
    "runtime_seconds",
)

#: The robustness-scope protocols keep the sparse longitudinal, time-dependent
#: covariate design of Example 2 and move only the contamination channel and the
#: cluster-size mechanism, so every method that is applicable to Example 2 is
#: applicable to all of them by the same argument.
_SCOPE_PROTOCOLS = frozenset(
    {
        Protocol.SCOPE_CLEAN,
        Protocol.SCOPE_RESPONSE,
        Protocol.SCOPE_SUBJECT,
        Protocol.SCOPE_TRAJECTORY,
        Protocol.SCOPE_LEVERAGE,
        Protocol.SCOPE_INFORMATIVE_CLEAN,
        Protocol.SCOPE_INFORMATIVE_SUBJECT,
    }
)
_SPARSE_METRICS = _COMMON_METRICS + ("tpr", "fdr", "model_size")
_ZSY_AUTHOR_METRICS = ("component_ise", "factor_ise", "runtime_seconds", "tpr", "fdr", "model_size")


def _na(reason: str) -> ApplicabilityDecision:
    return ApplicabilityDecision(Applicability.N_A_BY_DESIGN, reason)


def applicability_for(method: str, protocol: Protocol | str) -> ApplicabilityDecision:
    """Return the pre-registered design decision, not a post-hoc software result."""

    method = method.value if isinstance(method, MethodLabel) else str(method)
    try:
        protocol = Protocol(protocol)
    except ValueError as error:
        raise ValueError(f"unknown benchmark protocol: {protocol}") from error
    if method not in METHOD_SPECS:
        raise ValueError(f"method label is not pre-registered: {method}")
    reproduction_owner = {
        Protocol.REPRO_ZW2015: MethodLabel.ZW2015.value,
        Protocol.REPRO_ZZW2020: MethodLabel.ZZW2020.value,
        Protocol.REPRO_HHY2021: MethodLabel.HHY2021_HUBER.value,
        Protocol.REPRO_ZSY2026: MethodLabel.ZSY2026_AUTHOR_CODE.value,
        Protocol.REPRO_ZY2025: MethodLabel.ZY2025.value,
    }
    if protocol in reproduction_owner:
        if method != reproduction_owner[protocol]:
            return _na("A reproduction protocol is reserved for its published method.")
        metrics = _ZSY_AUTHOR_METRICS if method == MethodLabel.ZSY2026_AUTHOR_CODE.value else _COMMON_METRICS
        return ApplicabilityDecision(Applicability.APPLICABLE, "Original-paper reproduction design.", metrics)

    if method == MethodLabel.TRACE_VCAM.value:
        return ApplicabilityDecision(Applicability.APPLICABLE, "Proposed estimator is evaluated in every common protocol.", _SPARSE_METRICS)

    if method == MethodLabel.ZW2015.value:
        if protocol is Protocol.EXAMPLE1_DENSE:
            return ApplicabilityDecision(
                Applicability.APPLICABLE,
                "Original dense functional, time-invariant-covariate setting.",
                _COMMON_METRICS,
            )
        return _na("The original fdapace::VCAM method requires the dense functional/time-invariant design.")

    if method == MethodLabel.ZZW2020.value:
        if protocol in {
            Protocol.EXAMPLE2_GAUSSIAN,
            Protocol.EXAMPLE2_HEAVY_TAIL,
            Protocol.EXAMPLE2_CONTAMINATION,
            Protocol.EXAMPLE3_HIGH_DIMENSIONAL,
            Protocol.EXAMPLE3_SYMMETRIC_CONTAMINATION,
            Protocol.EXAMPLE4_ROBUST_NORMAL,
            Protocol.EXAMPLE4_ROBUST_HEAVY_TAIL,
            Protocol.EXAMPLE4_ROBUST_CONTAMINATION,
            Protocol.MACS_CD4,
        } | _SCOPE_PROTOCOLS:
            return ApplicabilityDecision(
                Applicability.APPLICABLE,
                "Sparse longitudinal, time-dependent-covariate design supported by Algorithm 1.",
                _COMMON_METRICS,
            )
        return _na("Outside the sparse-longitudinal designs registered for this implementation; the p=25/50 scaling extension is excluded.")

    if method == MethodLabel.HHY2021_HUBER.value:
        if protocol in {
            Protocol.EXAMPLE2_GAUSSIAN,
            Protocol.EXAMPLE2_HEAVY_TAIL,
            Protocol.EXAMPLE2_CONTAMINATION,
            Protocol.EXAMPLE3_HIGH_DIMENSIONAL,
            Protocol.EXAMPLE3_SYMMETRIC_CONTAMINATION,
            Protocol.EXAMPLE4_ROBUST_NORMAL,
            Protocol.EXAMPLE4_ROBUST_HEAVY_TAIL,
            Protocol.EXAMPLE4_ROBUST_CONTAMINATION,
            Protocol.MACS_CD4,
        } | _SCOPE_PROTOCOLS:
            return ApplicabilityDecision(
                Applicability.APPLICABLE,
                "Longitudinal robust VCAM setting supported by the published three-stage method.",
                _COMMON_METRICS,
            )
        return _na("Outside the longitudinal designs registered for this implementation; the p=25/50 scaling extension is excluded.")

    if method == MethodLabel.ZSY2026_AUTHOR_CODE.value:
        if protocol in {
            Protocol.EXAMPLE3_HIGH_DIMENSIONAL,
            Protocol.EXAMPLE3_SYMMETRIC_CONTAMINATION,
            Protocol.SCALING,
        }:
            return ApplicabilityDecision(
                Applicability.APPLICABLE,
                "Sparse high-dimensional design accepted by the unmodified author function.",
                _ZSY_AUTHOR_METRICS,
            )
        return _na(
            "The unmodified author function has no subject-ID or out-of-sample prediction interface; it is not adapted silently."
        )

    if method == MethodLabel.ZY2025.value:
        if protocol in {
            Protocol.EXAMPLE2_GAUSSIAN,
            Protocol.EXAMPLE2_HEAVY_TAIL,
            Protocol.EXAMPLE2_CONTAMINATION,
            Protocol.EXAMPLE3_HIGH_DIMENSIONAL,
            Protocol.EXAMPLE3_SYMMETRIC_CONTAMINATION,
            Protocol.EXAMPLE4_ROBUST_NORMAL,
            Protocol.EXAMPLE4_ROBUST_HEAVY_TAIL,
            Protocol.EXAMPLE4_ROBUST_CONTAMINATION,
            Protocol.SCALING,
            Protocol.MACS_CD4,
        } | _SCOPE_PROTOCOLS:
            return ApplicabilityDecision(
                Applicability.APPLICABLE,
                "The supplied full text defines a coefficientwise-L1 estimator; the paper-aligned implementation enters the same-setting comparison, while optional source-value reproduction remains diagnostic because no author code is available.",
                _SPARSE_METRICS,
            )
        return _na("The paper implementation is registered for sparse longitudinal/high-dimensional designs, not the dense functional benchmark.")

    raise AssertionError("unreachable registered method")


def applicability_matrix() -> dict[str, dict[str, ApplicabilityDecision]]:
    return {
        protocol.value: {
            method: applicability_for(method, protocol) for method in FIXED_METHOD_LABELS
        }
        for protocol in Protocol
    }


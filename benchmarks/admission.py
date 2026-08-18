"""Published-scenario reproduction gate for admitting comparison methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t

from .methods import MethodLabel, Protocol


@dataclass(frozen=True)
class PublishedTarget:
    method: str
    protocol: str
    scenario_id: str
    metric: str
    published_value: float
    rounding_digits: int
    published_sd: float | None = None
    published_replications: int | None = None
    source_location: str = ""

    @property
    def rounding_tolerance(self) -> float:
        return 0.5 * 10.0 ** (-self.rounding_digits)


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    method: str
    scenario_id: str
    metric: str
    published_value: float
    reproduced_mean: float | None
    mc_lower: float | None
    mc_upper: float | None
    rounding_tolerance: float
    n_attempted: int
    n_finite: int
    n_failed: int
    failure_rate: float | None
    reproduced_sd: float | None
    published_sd: float | None


def assess_reproduction(
    results: Iterable["BenchmarkResult"],
    target: PublishedTarget,
    *,
    confidence: float = 0.95,
    minimum_finite_replications: int = 2,
) -> AdmissionDecision:
    """Admit iff the published point lies in the MC CI or rounding tolerance.

    Every applicable attempted replication remains in ``n_attempted`` and the
    failure-rate denominator.  Only finite successful metric values can define
    a Monte Carlo interval.
    """

    rows = [
        row
        for row in results
        if row.method == target.method
        and row.protocol == target.protocol
        and row.scenario_id == target.scenario_id
        and row.attempted
    ]
    values = np.asarray(
        [
            row.metrics[target.metric]
            for row in rows
            if row.successful and target.metric in row.metrics
        ],
        dtype=float,
    )
    n_attempted = len(rows)
    n_finite = len(values)
    n_failed = n_attempted - n_finite
    failure_rate = n_failed / n_attempted if n_attempted else None
    tolerance = target.rounding_tolerance
    if n_finite < minimum_finite_replications:
        return AdmissionDecision(
            admitted=False,
            reason=(
                f"Only {n_finite} finite successful replications; "
                f"at least {minimum_finite_replications} are required."
            ),
            method=target.method,
            scenario_id=target.scenario_id,
            metric=target.metric,
            published_value=target.published_value,
            reproduced_mean=None,
            mc_lower=None,
            mc_upper=None,
            rounding_tolerance=tolerance,
            n_attempted=n_attempted,
            n_finite=n_finite,
            n_failed=n_failed,
            failure_rate=failure_rate,
            reproduced_sd=None,
            published_sd=target.published_sd,
        )
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    standard_error = sd / np.sqrt(n_finite)
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=n_finite - 1))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    covered = lower - tolerance <= target.published_value <= upper + tolerance
    if covered:
        reason = (
            "Published point is inside the reproduced Monte Carlo interval "
            "after the registered rounding tolerance."
        )
    else:
        reason = (
            "Published point is outside the reproduced Monte Carlo interval "
            "and its registered rounding tolerance."
        )
    return AdmissionDecision(
        admitted=covered,
        reason=reason,
        method=target.method,
        scenario_id=target.scenario_id,
        metric=target.metric,
        published_value=target.published_value,
        reproduced_mean=mean,
        mc_lower=lower,
        mc_upper=upper,
        rounding_tolerance=tolerance,
        n_attempted=n_attempted,
        n_finite=n_finite,
        n_failed=n_failed,
        failure_rate=failure_rate,
        reproduced_sd=sd,
        published_sd=target.published_sd,
    )


# Zhao--Yang (2025), Table 1.  The paper calls the reported quantity MSPE and
# uses 300 Monte Carlo replications.  These are registration targets, never
# generated benchmark results.
_ZY2025_VALUES = {
    (50, 0.1): (0.1488, 0.1123),
    (50, 0.4): (0.1803, 0.1585),
    (50, 1.0): (0.3301, 0.1776),
    (200, 0.1): (0.0575, 0.0376),
    (200, 0.4): (0.0638, 0.0379),
    (200, 1.0): (0.0965, 0.0460),
}

ZY2025_TABLE1_TARGETS: dict[tuple[int, float], PublishedTarget] = {
    (n_subjects, sigma): PublishedTarget(
        method=MethodLabel.ZY2025.value,
        protocol=Protocol.REPRO_ZY2025.value,
        scenario_id=f"ZY2025-table1-n{n_subjects}-sigma{sigma:g}",
        metric="test_mse",
        published_value=mean,
        published_sd=sd,
        published_replications=300,
        rounding_digits=4,
        source_location="Zhao and Yang (2025), Table 1, Proposed method",
    )
    for (n_subjects, sigma), (mean, sd) in _ZY2025_VALUES.items()
}


from .protocol import BenchmarkResult  # noqa: E402  (type/runtime import after definitions)

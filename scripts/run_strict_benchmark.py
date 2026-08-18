"""Run the registered VCAM source diagnostics and strict common benchmark.

The runner enforces three separations that are easy to blur in a simulation
study:

1. scientific applicability is decided before fitting;
2. optional source-paper reproduction is diagnostic and never controls entry
   to the same-setting comparison; and
3. an attempted failure remains in the replication denominator.

No missing external result is imputed.  A method outside its published scope
is written as ``N/A by design``; every applicable original/author/paper-aligned
implementation is attempted on the common data, seed, and subject split.

Examples
--------
Small protocol smoke run::

    python scripts/run_strict_benchmark.py --quick --output results/strict_quick

Locked protocol (the published-target JSON is intentionally explicit)::

    python scripts/run_strict_benchmark.py --formal \
        --published-targets protocol/published_targets.json \
        --output results/strict_formal
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import time
import traceback
import tracemalloc
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.adapters import (  # noqa: E402
    HHY2021Adapter,
    TraceVCAMAdapter,
    ZSY2026AuthorCodeAdapter,
    ZW2015Adapter,
    ZY2025Adapter,
    ZZW2020Adapter,
)
from benchmarks.data import SubjectDataset, SubjectSplit  # noqa: E402
from benchmarks.methods import (  # noqa: E402
    FIXED_METHOD_LABELS,
    METHOD_SPECS,
    Applicability,
    MethodLabel,
    Protocol,
    applicability_for,
)
from experiments.dgp import (  # noqa: E402
    PublishedDataset,
    Truth,
    _assemble,
    _draw_errors,
    _fourier_random_effect,
    _gaussian_copula_uniforms,
    _spline_function,
    generate_block_sparse,
    generate_hhy2021,
    generate_zsy2026,
    generate_zw2015,
    generate_zzw2020,
    subject_split,
    zzw2020_truth,
)
from src.trace_tuning_protocol import load_trace_tuning_lock  # noqa: E402


SCHEMA_VERSION = "vcam-strict-benchmark/1"
PROGRESS_SCHEMA_VERSION = "vcam-strict-progress/2"
RUN_FINGERPRINT_SCHEMA_VERSION = "vcam-strict-run-fingerprint/1"
DEFAULT_ROOT_SEED = 20260810
ORDERED_TASK_SCHEDULER_VERSION = "bounded-ordered-prefetch/1"
ORDERED_TASK_PREFETCH_FACTOR = 3
RESULT_FIELDS = (
    "schema_version",
    "mode",
    "phase",
    "example",
    "protocol",
    "scenario",
    "replicate",
    "seed",
    "split_seed",
    "split_unit",
    "method",
    "method_display_name",
    "method_version",
    "applicability",
    "applicability_reason",
    "admission_status",
    "attempt_status",
    "converged",
    "failure_code",
    "failure_message",
    "design_id",
    "provenance",
    "n_subjects",
    "n_train_subjects",
    "n_test_subjects",
    "n_rows",
    "n_covariates",
    "n_active",
    "has_null_blocks",
    "data_hash",
    "train_subject_hash",
    "test_subject_hash",
    "tuning_json",
    "tuning_sha256",
    "realized_tuning_json",
    "realized_tuning_sha256",
    "runtime_seconds",
    "peak_python_memory_mb",
    "observed_test_mspe",
    "test_mse",
    "noise_free_test_mspe",
    "baseline_ise",
    "component_ise",
    "factor_ise",
    "paper_observed_factor_mse_total",
    "paper_training_function_mse_total",
    "tpr",
    "fdr",
    "model_size",
    "selected_blocks_json",
    "fit_metadata_json",
)
PREDICTION_FIELDS = (
    "schema_version",
    "scenario",
    "replicate",
    "seed",
    "method",
    "row_id",
    "subject_id",
    "observed_response",
    "noise_free_target",
    "prediction",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_seed(root_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (root_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (
        2**32 - 1
    )


def _enum_value(name: str, fallback: str) -> str:
    value = getattr(Protocol, name, None)
    return fallback if value is None else str(value.value)


def _method_value(name: str, fallback: str) -> str:
    value = getattr(MethodLabel, name, None)
    return fallback if value is None else str(value.value)


TRACE = _method_value("TRACE_VCAM", "TRACE-VCAM")
ZW = _method_value("ZW2015", "ZW2015")
ZZW = _method_value("ZZW2020", "ZZW2020")
HHY = _method_value("HHY2021_HUBER", "HHY2021-Huber")
ZSY = _method_value("ZSY2026_AUTHOR_CODE", "ZSY2026-author-code")
ZY = _method_value("ZY2025", "ZY2025-paper-implementation")


@dataclass(frozen=True)
class Scenario:
    scenario: str
    phase: str
    example: str
    protocol: str
    generator: str
    parameters: Mapping[str, object]
    formal_replications: int
    owner: str | None = None
    split_unit: str = "subject"

    def build(self, seed: int) -> PublishedDataset:
        parameters = dict(self.parameters)
        if self.generator == "zw2015":
            return generate_zw2015(seed, **parameters)
        if self.generator == "zzw2020":
            return generate_zzw2020(seed, **parameters)
        if self.generator == "hhy2021":
            return generate_hhy2021(seed, **parameters)
        if self.generator == "zsy2026":
            return generate_zsy2026(seed, **parameters)
        if self.generator == "block_sparse":
            return generate_block_sparse(seed, **parameters)
        if self.generator == "robustness_scope":
            from experiments.dgp import generate_robustness_scope

            return generate_robustness_scope(seed, **parameters)
        if self.generator == "zy2025_literal":
            return generate_zy2025_literal(seed, **parameters)
        raise ValueError(f"unregistered generator: {self.generator}")


def generate_zy2025_literal(
    seed: int, *, n_subjects: int, sigma: float
) -> PublishedDataset:
    """Literal Example 4.1 transcription, kept distinct from ZZW2020.

    The printed centering constants (0.4 and 2) do not center the functions
    under this open cubic B-spline convention.  That source inconsistency is
    recorded in the target registry and blocks formal admission; this generator
    is retained only so the literal specification can be audited rather than
    silently numerically re-centered.
    """

    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(2, 11, size=n_subjects)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time_values = rng.uniform(0.0, 2.0, size=subject.size)
    u = _gaussian_copula_uniforms(
        rng, n_subjects, np.array([[1.0, 0.6], [0.6, 1.0]])
    )
    v = _gaussian_copula_uniforms(
        rng, n_subjects, np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    covariates = np.column_stack(
        [
            0.5 * u[subject, 0] * (0.5 * time_values) ** 0.5 + 0.5 * v[subject, 0],
            0.5 * u[subject, 1] * (0.5 * time_values) ** (1.0 / 3.0)
            + 0.5 * v[subject, 1],
        ]
    )
    zzw_truth = zzw2020_truth()
    phi1_star = _spline_function(
        np.array([0.0, -1.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
        (0.0, 1.0),
        3,
    )
    phi2_star = _spline_function(
        np.array([0.0, 0.0, 4.0, 2.0, 0.0, 0.0]),
        (0.0, 1.0),
        2,
    )
    truth = Truth(
        beta0=zzw_truth.beta0,
        beta=zzw_truth.beta,
        phi=(lambda z: phi1_star(z) - 0.4, lambda z: phi2_star(z) - 2.0),
        active=(True, True),
    )
    return _assemble(
        time=time_values,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=_fourier_random_effect(rng, time_values, subject, n_subjects),
        errors=_draw_errors(rng, time_values.size, sigma, "gaussian"),
        design_id="ZY2025-Example4.1-literal-printed-centering",
        provenance="original-text-literal/source-centering-inconsistency",
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
    )


def registered_scenarios(
    *, quick: bool, include_reproduction_audit: bool = False
) -> tuple[Scenario, ...]:
    """Return the immutable protocol registry.

    Quick mode keeps one common scenario per example and one replication.  It
    is a software smoke test and can never authorize manuscript claims.
    Reproducing published table entries is an optional implementation audit;
    it is not an admission requirement for the same-setting comparison.
    """

    reproduction = [
        Scenario(
            "repro-zw2015-n100",
            "reproduction",
            "Reproduction",
            _enum_value("REPRO_ZW2015", "reproduction/ZW2015"),
            "zw2015",
            {"n_subjects": 100},
            500,
            ZW,
            "full_sample",
        ),
        Scenario(
            "repro-zzw2020-n200-sigma0.4",
            "reproduction",
            "Reproduction",
            _enum_value("REPRO_ZZW2020", "reproduction/ZZW2020"),
            "zzw2020",
            {"n_subjects": 200, "sigma": 0.4, "error_distribution": "gaussian"},
            300,
            ZZW,
        ),
        Scenario(
            "repro-hhy2021-n30-t2",
            "reproduction",
            "Reproduction",
            _enum_value("REPRO_HHY2021", "reproduction/HHY2021-Huber"),
            "hhy2021",
            {"error_distribution": "hhy-t2"},
            500,
            HHY,
            "full_sample",
        ),
        Scenario(
            "repro-zsy2026-n50-p10-sigma0.1",
            "reproduction",
            "Reproduction",
            _enum_value("REPRO_ZSY2026", "reproduction/ZSY2026-author-code"),
            "zsy2026",
            {
                "n_subjects": 50,
                "sigma": 0.1,
                "error_distribution": "gaussian",
                "n_covariates": 10,
            },
            100,
            ZSY,
        ),
        Scenario(
            "ZY2025-table1-n200-sigma0.1",
            "reproduction",
            "Reproduction",
            _enum_value("REPRO_ZY2025", "reproduction/ZY2025-paper-implementation"),
            "zy2025_literal",
            {"n_subjects": 200, "sigma": 0.1},
            300,
            ZY,
            "observation",
        ),
    ]

    example1 = [
        Scenario(
            "example1-zw2015-n100",
            "common",
            "Example 1",
            _enum_value("EXAMPLE1_DENSE", "example-1/dense-functional"),
            "zw2015",
            {"n_subjects": 100},
            500,
        )
    ]
    example2: list[Scenario] = []
    for n_subjects in (50, 100, 200):
        for sigma in (0.1, 0.4):
            example2.append(
                Scenario(
                    f"example2-gaussian-n{n_subjects}-sigma{sigma:g}",
                    "common",
                    "Example 2",
                    _enum_value(
                        "EXAMPLE2_GAUSSIAN", "example-2/sparse-longitudinal-gaussian"
                    ),
                    "zzw2020",
                    {
                        "n_subjects": n_subjects,
                        "sigma": sigma,
                        "error_distribution": "gaussian",
                    },
                    300,
                )
            )
        for error, protocol_name, suffix in (
            (
                "hhy-t2",
                "EXAMPLE2_HEAVY_TAIL",
                "t2",
            ),
            (
                "hhy-mixed-normal",
                "EXAMPLE2_CONTAMINATION",
                "mixed-normal",
            ),
        ):
            example2.append(
                Scenario(
                    f"example2-{suffix}-n{n_subjects}",
                    "common",
                    "Example 2",
                    _enum_value(protocol_name, f"example-2/{suffix}"),
                    "zzw2020",
                    {
                        "n_subjects": n_subjects,
                        "sigma": 0.4,
                        "error_distribution": error,
                    },
                    300,
                )
            )

    example3: list[Scenario] = []
    for n_subjects in (50, 200):
        for sigma in (0.1, 0.4):
            for error, protocol_name, suffix in (
                ("gaussian", "EXAMPLE3_HIGH_DIMENSIONAL", "gaussian"),
                (
                    "symmetric-contamination",
                    "EXAMPLE3_SYMMETRIC_CONTAMINATION",
                    "contamination",
                ),
            ):
                example3.append(
                    Scenario(
                        f"example3-{suffix}-n{n_subjects}-p10-sigma{sigma:g}",
                        "common",
                        "Example 3",
                        _enum_value(protocol_name, f"example-3/{suffix}"),
                        "zsy2026",
                        {
                            "n_subjects": n_subjects,
                            "sigma": sigma,
                            "error_distribution": error,
                            "n_covariates": 10,
                        },
                        100,
                    )
                )

    scaling = [
        Scenario(
            f"scaling-n200-p{p}",
            "common",
            "Scaling",
            _enum_value("SCALING", "scaling/p10-p25-p50"),
            "zsy2026",
            {
                "n_subjects": 200,
                "sigma": 0.4,
                "error_distribution": "gaussian",
                "n_covariates": p,
            },
            5,
        )
        for p in (10, 25, 50)
    ]
    if quick:
        selected = [
            example1[0],
            example2[1],
            example2[3],
            example3[0],
            example3[1],
            scaling[0],
        ]
        return tuple((reproduction if include_reproduction_audit else []) + selected)
    common = example1 + example2 + example3 + scaling
    return tuple((reproduction if include_reproduction_audit else []) + common)


# This is a transcription of the target supplied from Table 1 of Zhao--Yang
# (2025), not an estimated or filled-in benchmark result.  Other methods need
# an explicit target file because no unverified numbers are embedded here.
BUILTIN_PUBLISHED_TARGETS: dict[str, dict[str, object]] = {
    "ZY2025-table1-n200-sigma0.1": {
        "method": ZY,
        "metric": "noise_free_test_mspe",
        "reported_mean": 0.0575,
        "reported_sd": 0.0376,
        "reported_replications": 300,
        "rounding_tolerance": 0.00005,
        "source": "Zhao and Yang (2025), Table 1, n=200, sigma=0.1",
    }
}


def load_published_targets(path: Path | None) -> dict[str, dict[str, object]]:
    targets = {key: dict(value) for key, value in BUILTIN_PUBLISHED_TARGETS.items()}
    if path is None:
        return targets
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "targets" in payload:
        payload = payload["targets"]
    if isinstance(payload, list):
        payload = {str(item["scenario"]): item for item in payload}
    if not isinstance(payload, Mapping):
        raise ValueError("published targets must be a mapping or a list under 'targets'")
    for scenario, target in payload.items():
        if not isinstance(target, Mapping):
            raise ValueError(f"target for {scenario} must be an object")
        required = {"method", "metric", "reported_replications", "source"}
        missing = required - set(target)
        if "reported_value" not in target and "reported_mean" not in target:
            missing.add("reported_value")
        if missing:
            raise ValueError(f"target for {scenario} is missing {sorted(missing)}")
        targets[str(scenario)] = dict(target)
    return targets


def assess_reproduction_gate(
    rows: Sequence[Mapping[str, object]],
    scenario: Scenario,
    target: Mapping[str, object] | None,
    *,
    expected_replications: int,
) -> dict[str, object]:
    """Apply the registered 95% Monte Carlo/rounding admission rule."""

    selected = [
        row
        for row in rows
        if row.get("scenario") == scenario.scenario and row.get("method") == scenario.owner
    ]
    decision: dict[str, object] = {
        "scenario": scenario.scenario,
        "method": scenario.owner,
        "expected_replications": expected_replications,
        "observed_rows": len(selected),
        "passed": False,
    }
    if target is None:
        decision.update(
            status="pending_no_published_target",
            reason="No sourced published target was supplied; common-benchmark admission is blocked.",
        )
        return decision
    if str(target.get("method")) != str(scenario.owner):
        decision.update(status="invalid_target", reason="Published target method does not match owner.")
        return decision
    reported_replications = int(target.get("reported_replications", -1))
    if reported_replications != expected_replications:
        decision.update(
            status="invalid_target_replication_count",
            reason=(
                f"Published target declares Q={reported_replications}, but the locked "
                f"reproduction requires Q={expected_replications}."
            ),
        )
        return decision
    if target.get("source_gate_blocked_reason"):
        decision.update(
            status="blocked_source_protocol_inconsistency",
            reason=str(target["source_gate_blocked_reason"]),
            source=str(target["source"]),
        )
        return decision
    metric = str(target["metric"])
    statistic = str(target.get("statistic", "mean")).lower()
    if statistic not in {"mean", "median"}:
        decision.update(status="invalid_target_statistic", reason=f"Unknown statistic: {statistic}")
        return decision
    successful = [
        float(row[metric])
        for row in selected
        if row.get("attempt_status") == "success"
        and row.get("converged") is True
        and _is_finite(row.get(metric))
    ]
    decision.update(
        metric=metric,
        source=str(target["source"]),
        statistic=statistic,
        reported_value=float(target.get("reported_value", target.get("reported_mean"))),
        reported_sd=(
            None if target.get("reported_sd") is None else float(target["reported_sd"])
        ),
        reported_mad=(
            None if target.get("reported_mad") is None else float(target["reported_mad"])
        ),
        successful_replications=len(successful),
        failure_rate=(
            1.0 - len(successful) / len(selected) if selected else 1.0
        ),
    )
    if len(selected) != expected_replications:
        decision.update(
            status="incomplete_reproduction",
            reason="The reproduction row count does not equal the locked replication count.",
        )
        return decision
    if not successful:
        decision.update(status="no_successful_replications", reason="No finite converged result.")
        return decision
    values = np.asarray(successful, dtype=float)
    estimate = float(np.mean(values) if statistic == "mean" else np.median(values))
    sd = float(np.std(successful, ddof=1)) if len(successful) > 1 else float("nan")
    if statistic == "mean":
        mcse = sd / math.sqrt(len(successful)) if len(successful) > 1 else float("nan")
        lower = estimate - 1.96 * mcse if np.isfinite(mcse) else estimate
        upper = estimate + 1.96 * mcse if np.isfinite(mcse) else estimate
    else:
        draws = int(target.get("bootstrap_replications", 4000))
        rng = np.random.default_rng(_stable_seed(DEFAULT_ROOT_SEED, scenario.scenario, "median-gate"))
        bootstrap = np.median(
            values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1
        )
        lower, upper = (float(item) for item in np.quantile(bootstrap, [0.025, 0.975]))
        mcse = float(np.std(bootstrap, ddof=1))
    tolerance = float(target.get("rounding_tolerance", 0.0))
    reported = float(target.get("reported_value", target.get("reported_mean")))
    interval_pass = lower - tolerance <= reported <= upper + tolerance
    rounding_pass = abs(estimate - reported) <= tolerance
    passed = bool(interval_pass or rounding_pass)
    decision.update(
        estimated_statistic=estimate,
        estimated_mean=(estimate if statistic == "mean" else None),
        estimated_median=(estimate if statistic == "median" else None),
        estimated_sd=sd,
        mcse=mcse,
        ci95=[lower, upper],
        rounding_tolerance=tolerance,
        interval_pass=bool(interval_pass),
        rounding_pass=bool(rounding_pass),
        passed=passed,
        status="admitted" if passed else "reproduction_mismatch",
        reason=(
            "Published value is inside the reproduction 95% Monte Carlo interval or rounding tolerance."
            if passed
            else "Published value is outside both the reproduction interval and rounding tolerance."
        ),
    )
    return decision


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _subject_dataset(data: PublishedDataset, scenario: Scenario) -> SubjectDataset:
    return SubjectDataset(
        time=data.time,
        covariates=data.covariates,
        response=data.response,
        subject_id=np.asarray([str(item) for item in data.subject], dtype=str),
        row_id=np.asarray(
            [f"{scenario.scenario}-row-{index}" for index in range(data.time.size)], dtype=str
        ),
        noise_free_target=data.conditional_mean,
        covariate_names=tuple(f"x_{index + 1}" for index in range(data.covariates.shape[1])),
        metadata={
            "design_id": data.design_id,
            "provenance": data.provenance,
            "time_invariant_covariates": data.time_invariant_covariates,
            "time_domain": list(data.domain_time),
            "covariate_domains": [list(item) for item in data.domain_covariates],
            "active": list(data.truth.active),
        },
    )


def _split_dataset(
    dataset: SubjectDataset, raw: PublishedDataset, *, split_seed: int
) -> tuple[SubjectDataset, SubjectDataset, SubjectSplit]:
    train_rows, test_rows = subject_split(raw.subject, seed=split_seed, train_fraction=0.8)
    train_subjects = tuple(sorted(set(dataset.subject_id[train_rows].tolist())))
    test_subjects = tuple(sorted(set(dataset.subject_id[test_rows].tolist())))
    split = SubjectSplit(
        repeat=0,
        fold=0,
        seed=split_seed,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
    )
    split.validate_against(dataset)
    return dataset.subset_subjects(train_subjects), dataset.subset_subjects(test_subjects), split


@dataclass(frozen=True)
class SourceSplitAudit:
    repeat: int
    fold: int
    seed: int
    train_hash: str
    test_hash: str


def _row_subset(dataset: SubjectDataset, indices: np.ndarray) -> SubjectDataset:
    target = None if dataset.noise_free_target is None else dataset.noise_free_target[indices]
    return SubjectDataset(
        time=dataset.time[indices],
        covariates=dataset.covariates[indices],
        response=dataset.response[indices],
        subject_id=dataset.subject_id[indices],
        row_id=dataset.row_id[indices],
        noise_free_target=target,
        covariate_names=dataset.covariate_names,
        metadata=dataset.metadata,
    )


def _id_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(item) for item in values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _registered_split(
    dataset: SubjectDataset,
    raw: PublishedDataset,
    scenario: Scenario,
    *,
    split_seed: int,
) -> tuple[SubjectDataset, SubjectDataset, SubjectSplit | SourceSplitAudit]:
    if scenario.split_unit == "subject":
        return _split_dataset(dataset, raw, split_seed=split_seed)
    if scenario.split_unit == "full_sample":
        audit = SourceSplitAudit(
            repeat=0,
            fold=0,
            seed=split_seed,
            train_hash=_id_hash(dataset.row_id.tolist()),
            test_hash=_id_hash(dataset.row_id.tolist()),
        )
        return dataset, dataset, audit
    if scenario.split_unit == "observation":
        permutation = np.random.default_rng(split_seed).permutation(dataset.n_rows)
        cut = int(np.floor(0.8 * dataset.n_rows))
        train_index = np.asarray(permutation[:cut], dtype=int)
        test_index = np.asarray(permutation[cut:], dtype=int)
        audit = SourceSplitAudit(
            repeat=0,
            fold=0,
            seed=split_seed,
            train_hash=_id_hash(dataset.row_id[train_index].tolist()),
            test_hash=_id_hash(dataset.row_id[test_index].tolist()),
        )
        return _row_subset(dataset, train_index), _row_subset(dataset, test_index), audit
    raise ValueError(f"unknown split unit: {scenario.split_unit}")


def adapter_registry() -> dict[str, object]:
    """Instantiate the fixed adapters without substituting proxy methods."""

    candidates = [
        TraceVCAMAdapter(),
        ZW2015Adapter(),
        ZZW2020Adapter(),
        HHY2021Adapter(),
        ZSY2026AuthorCodeAdapter(),
        ZY2025Adapter(),
    ]
    registry = {str(adapter.label): adapter for adapter in candidates}
    missing = set(FIXED_METHOD_LABELS) - set(registry)
    if missing:
        raise RuntimeError(f"missing adapters for fixed methods: {sorted(missing)}")
    return registry


def _safe_applicability(method: str, scenario: Scenario) -> tuple[str, str]:
    if scenario.phase == "reproduction":
        if method == scenario.owner:
            return Applicability.APPLICABLE.value, "Owner of this original-paper reproduction."
        return Applicability.N_A_BY_DESIGN.value, "Reproduction is reserved for its source method."
    try:
        decision = applicability_for(method, scenario.protocol)
    except (ValueError, KeyError):
        # A newly registered ZY2025 protocol may reach the runner before an
        # older applicability enum.  Do not infer scientific applicability.
        return "unregistered", "No pre-registered applicability decision exists."
    return str(decision.status.value), str(decision.reason)


def _default_tuning(method: str, scenario: Scenario, *, quick: bool) -> dict[str, object]:
    domain_time = [0.0, 1.0] if scenario.generator in {"zw2015", "hhy2021"} else [0.0, 2.0]
    n_covariates = int(scenario.parameters.get("n_covariates", 2))
    tuning: dict[str, object] = {"protocol_mode": "quick" if quick else "formal"}
    if method == TRACE:
        trace_lock = load_trace_tuning_lock()
        tuning.update(
            q_time=4 if quick else 6,
            q_covariate=4 if quick else 6,
            delta_rule="mad",
            huber_multiplier=1.345,
            lambda_ratio=float(trace_lock["lambda_ratio"]),
            roughness=float(trace_lock["roughness"]),
            tuning_mode=str(trace_lock["tuning_mode"]),
            calibration_path="protocol/trace_tuning_v1.json",
            calibration_content_sha256=str(trace_lock["content_sha256"]),
            calibration_file_sha256=str(trace_lock["file_sha256"]),
            max_iter=300 if quick else 2000,
            tolerance=1e-4 if quick else 1e-7,
            postfit_max_iter=500 if quick else 1000,
            postfit_tolerance=2e-7,
        )
    elif method == ZW:
        # The public package interprets nKnot as total basis count and rejects
        # the 1--2 interior-knot values selected in the paper.  We therefore
        # run the unmodified package at its documented feasible default in
        # every protocol and record this source/package incompatibility.
        tuning.update(
            tuning_mode="CRAN_fdapace_0.6.0_documented_defaults",
            add_nknot=[10, 10],
            add_order=[3, 3],
            vc_nknot=[10, 10, 10],
            vc_order=[3, 3, 3],
            paper_knot_interface_compatible=False,
            paper_knot_departure=(
                "fdapace::VCAM converts nKnot to nIntKnot=nKnot-order-1; "
                "GenBSpline rejects the paper's 1--2 interior-knot choices"
            ),
        )
        tuning.update(
            grid_size=51 if quick else 201,
            time_domain=domain_time,
            timeout_seconds=60 if quick else 900,
        )
    elif method == ZZW:
        if n_covariates == 2:
            time_counts = [4, 1, 2]
            additive_counts = [3, 2]
            knot_vectors = [
                {"time": [4, 1, 2], "additive": [3, 2]},
                {"time": [4, 2, 2], "additive": [3, 2]},
                {"time": [3, 2, 2], "additive": [3, 2]},
            ]
            tuning.update(
                tuning_mode=(
                    "paper_design_counts" if quick else "paper_cv_registered_vectors"
                ),
                knot_candidate_vectors=knot_vectors,
                candidate_grid_source="registered_p2_source_design_grid",
                cv_folds=5,
                cv_max_inner=50,
                cv_max_outer=50,
                max_inner=20 if quick else 200,
                max_outer=20 if quick else 200,
            )
        else:
            # Zhao--Sun--Yang apply this published backfitting method in their
            # p=10 design but neither source supplies a high-dimensional joint
            # CV grid.  Lock the reported p=10 design-count vector rather than
            # inventing an expensive, undocumented grid search.  This is an
            # explicitly labelled paper-aligned extension, not source-original
            # Zhang--Zhong--Wang tuning.
            time_counts = [4] + [2] * n_covariates
            additive_counts = [2] * n_covariates
            tuning.update(
                tuning_mode="paper_aligned_fixed_p10_extension",
                fixed_knot_provenance=(
                    "reported Zhao--Sun--Yang p=10 design-count vector; "
                    "no high-dimensional Zhang--Zhong--Wang joint CV grid is claimed"
                ),
                high_dimensional_tuning_departure=(
                    "fixed paper-aligned p=10 counts replace an undocumented "
                    "high-dimensional CV search"
                ),
                high_dimensional_iteration_budget=(
                    "50 outer and 50 inner iterations; an unreached source "
                    "tolerance is recorded as an iteration-limit failure"
                ),
                max_inner=20 if quick else 50,
                max_outer=20 if quick else 50,
            )
        tuning.update(
            spline_order=4,
            time_interior_knots=time_counts,
            covariate_interior_knots=additive_counts,
            epsilon_inner=1e-2,
            epsilon_outer=1e-3,
            time_domain=domain_time,
        )
    elif method == HHY:
        tuning.update(
            tuning_mode="paper_locked" if quick else "paper_bic",
            spline_order=4,
            delta=1.345,
            pilot_time_interior_knots=2,
            pilot_covariate_interior_knots=2,
            final_time_interior_knots=2,
            final_additive_interior_knots=2,
            irls_tolerance=1e-8,
            irls_max_iter=100 if quick else 300,
            irls_objective_relative_tolerance=1e-9,
            irls_objective_stable_steps=3,
            bic_knot_candidates=[1, 2, 3],
            anchor_quantiles=[0.25, 0.5, 0.75],
            time_domain=domain_time,
        )
    elif method == ZY:
        if n_covariates == 2:
            zy_time_counts = [4, 1, 2]
            zy_additive_counts = [3, 2]
        else:
            zy_time_counts = [4] + [2] * n_covariates
            zy_additive_counts = [2] * n_covariates
        tuning.update(
            tuning_mode="paper_locked" if quick else "paper_cv",
            spline_order=4,
            time_interior_knots=zy_time_counts,
            additive_interior_knots=zy_additive_counts,
            inner_mrs_tolerance=1e-3 if quick else 1e-4,
            outer_mrs_tolerance=1e-3 if quick else 1e-4,
            max_inner=10 if quick else 100,
            max_outer=10 if quick else 100,
            lasso_tolerance=1e-5 if quick else 1e-8,
            lasso_max_iter=500 if quick else 5000,
            cv_solver=(
                "coordinate_fista_path"
                if scenario.generator == "zsy2026"
                else "fista_warm_path"
            ),
            cv_folds=10,
            cv_penalty_count=5 if quick else 10,
            cv_tolerance=1e-4 if quick else 1e-5,
            cv_lasso_max_iter=(
                300 if quick else (3000 if scenario.generator == "zsy2026" else 1000)
            ),
            timeout_seconds=(
                60
                if quick
                else (300 if scenario.example == "Scaling" else 180)
            ),
            time_domain=domain_time,
        )
        if quick:
            # Quick mode is explicitly non-admissible.  Fixed small penalties
            # exercise every paper-algorithm stage without rerunning hundreds
            # of ten-fold searches; formal mode above retains paper CV.
            tuning.update(
                lambda_initial_additive=0.01,
                lambda_initial_coefficient=0.01,
                lambda_additive=0.01,
                lambda_coefficient=0.01,
                lambda_baseline=0.0,
            )
    elif method == ZSY:
        tuning.update(
            tuning_mode="author_source_locked_spline_dimensions",
            df=[8] + [6] * (2 * n_covariates),
            df_source=(
                "cubic splines with K0,C=4 and all coefficient/additive K=2"
            ),
            workspace_root=str(ROOT),
            strict_no_silent_patch=True,
            timeout_seconds=60 if quick else 1800,
        )
    else:
        tuning.update(
            tuning_mode="original_method",
            workspace_root=str(ROOT),
            strict_no_silent_patch=True,
            timeout_seconds=60 if quick else 1800,
        )
    return tuning


def _curve_map(curves: Iterable[Mapping[str, object]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for curve in curves:
        name = str(curve.get("component", ""))
        grid = np.asarray(curve.get("grid", []), dtype=float)
        values = np.asarray(curve.get("values", []), dtype=float)
        if name and grid.ndim == values.ndim == 1 and len(grid) == len(values) and len(grid) >= 2:
            order = np.argsort(grid)
            mapped[name] = (grid[order], values[order])
    return mapped


def _integrated_mse(grid: np.ndarray, error: np.ndarray) -> float:
    length = float(grid[-1] - grid[0])
    return float(np.trapezoid(error**2, grid) / length) if length > 0 else float("nan")


def _common_identify_curves(
    curves: Iterable[Mapping[str, object]],
    *,
    n_covariates: int,
    time_domain: tuple[float, float] | None = None,
    covariate_domains: Sequence[tuple[float, float]] | None = None,
    tolerance: float = 1e-6,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Put every method's factor curves in the manuscript's common coordinates.

    For each available pair, subtract the Lebesgue mean of ``phi`` and absorb
    the removed constant into the baseline.  Then scale ``beta`` to have
    mapped-domain integral one (equivalently raw-domain mean one after a
    linear coordinate map) and apply the inverse scale to ``phi``.  This
    transformation preserves the fitted conditional-mean surface.  A
    near-zero beta mean is recorded as an identification failure;
    component-surface loss remains evaluable, while factor loss uses the
    registered missing-factor penalty.
    """

    records = [dict(curve) for curve in curves]
    positions = {
        str(record.get("component", "")): position
        for position, record in enumerate(records)
    }
    baseline_position = positions.get("baseline")
    baseline_grid: np.ndarray | None = None
    baseline_values: np.ndarray | None = None
    if baseline_position is not None:
        baseline_grid = np.asarray(records[baseline_position].get("grid", []), dtype=float)
        baseline_values = np.asarray(
            records[baseline_position].get("values", []), dtype=float
        ).copy()
        if (
            baseline_grid.ndim != 1
            or baseline_values.shape != baseline_grid.shape
            or baseline_grid.size < 2
        ):
            baseline_grid = None
            baseline_values = None
        elif time_domain is not None:
            order = np.argsort(baseline_grid)
            target_grid = np.linspace(time_domain[0], time_domain[1], 201)
            baseline_values = np.interp(
                target_grid, baseline_grid[order], baseline_values[order]
            )
            baseline_grid = target_grid

    invalid_blocks: list[int] = []
    transformed_blocks: list[int] = []
    diagnostics: list[dict[str, object]] = []
    for index in range(n_covariates):
        beta_key = f"beta_{index + 1}"
        phi_key = f"phi_{index + 1}"
        beta_position = positions.get(beta_key)
        phi_position = positions.get(phi_key)
        if beta_position is None and phi_position is None:
            continue
        if beta_position is None or phi_position is None:
            invalid_blocks.append(index)
            diagnostics.append(
                {
                    "block": index,
                    "status": "incomplete_factor_pair",
                }
            )
            continue
        beta_grid = np.asarray(records[beta_position].get("grid", []), dtype=float)
        beta_values = np.asarray(
            records[beta_position].get("values", []), dtype=float
        ).copy()
        phi_grid = np.asarray(records[phi_position].get("grid", []), dtype=float)
        phi_values = np.asarray(
            records[phi_position].get("values", []), dtype=float
        ).copy()
        valid_arrays = (
            beta_grid.ndim == phi_grid.ndim == 1
            and beta_values.shape == beta_grid.shape
            and phi_values.shape == phi_grid.shape
            and beta_grid.size >= 2
            and phi_grid.size >= 2
            and np.all(np.isfinite(beta_grid))
            and np.all(np.isfinite(beta_values))
            and np.all(np.isfinite(phi_grid))
            and np.all(np.isfinite(phi_values))
        )
        if not valid_arrays:
            invalid_blocks.append(index)
            diagnostics.append({"block": index, "status": "invalid_curve_arrays"})
            continue
        beta_order = np.argsort(beta_grid)
        phi_order = np.argsort(phi_grid)
        beta_grid, beta_values = beta_grid[beta_order], beta_values[beta_order]
        phi_grid, phi_values = phi_grid[phi_order], phi_values[phi_order]
        if time_domain is not None:
            target_beta_grid = np.linspace(time_domain[0], time_domain[1], 201)
            beta_values = np.interp(target_beta_grid, beta_grid, beta_values)
            beta_grid = target_beta_grid
        if covariate_domains is not None:
            target_phi_grid = np.linspace(
                covariate_domains[index][0], covariate_domains[index][1], 201
            )
            phi_values = np.interp(target_phi_grid, phi_grid, phi_values)
            phi_grid = target_phi_grid
        phi_length = float(phi_grid[-1] - phi_grid[0])
        if phi_length <= 0.0:
            invalid_blocks.append(index)
            diagnostics.append({"block": index, "status": "degenerate_phi_domain"})
            continue
        phi_mean = float(np.trapezoid(phi_values, phi_grid) / phi_length)
        phi_values -= phi_mean
        if baseline_grid is not None and baseline_values is not None:
            baseline_values += phi_mean * np.interp(
                baseline_grid, beta_grid, beta_values
            )
        elif abs(phi_mean) > tolerance:
            invalid_blocks.append(index)
            diagnostics.append({"block": index, "status": "missing_baseline"})

        beta_length = float(beta_grid[-1] - beta_grid[0])
        beta_mean = (
            float(np.trapezoid(beta_values, beta_grid) / beta_length)
            if beta_length > 0.0
            else float("nan")
        )
        if not np.isfinite(beta_mean) or abs(beta_mean) <= tolerance:
            if index not in invalid_blocks:
                invalid_blocks.append(index)
            status = "near_zero_beta_mean"
        else:
            beta_values /= beta_mean
            phi_values *= beta_mean
            transformed_blocks.append(index)
            status = "identified"
        records[beta_position]["grid"] = beta_grid.tolist()
        records[beta_position]["values"] = beta_values.tolist()
        records[phi_position]["grid"] = phi_grid.tolist()
        records[phi_position]["values"] = phi_values.tolist()
        diagnostics.append(
            {
                "block": index,
                "status": status,
                "removed_phi_mean": phi_mean,
                "beta_raw_domain_mean_before_scaling": beta_mean,
            }
        )

    if baseline_position is not None and baseline_grid is not None and baseline_values is not None:
        records[baseline_position]["grid"] = baseline_grid.tolist()
        records[baseline_position]["values"] = baseline_values.tolist()
    audit = {
        "rule": "phi Lebesgue centering, baseline absorption, beta raw-domain mean-one scaling (mapped-domain integral one)",
        "tolerance": float(tolerance),
        "transformed_blocks": transformed_blocks,
        "invalid_blocks": sorted(set(invalid_blocks)),
        "blocks": diagnostics,
    }
    return tuple(records), audit


def _truth_metrics(
    raw: PublishedDataset,
    curves: Iterable[Mapping[str, object]],
    *,
    invalid_factor_blocks: Iterable[int] = (),
) -> tuple[float, float, float]:
    mapped = _curve_map(curves)
    invalid = {int(index) for index in invalid_factor_blocks}
    t_grid = np.linspace(raw.domain_time[0], raw.domain_time[1], 201)
    if "baseline" in mapped:
        source_grid, source_values = mapped["baseline"]
        baseline = np.interp(t_grid, source_grid, source_values)
    else:
        baseline = np.zeros_like(t_grid)
    baseline_ise = _integrated_mse(t_grid, baseline - raw.truth.beta0(t_grid))

    surface_errors: list[float] = []
    factor_errors: list[float] = [baseline_ise]
    for index, active in enumerate(raw.truth.active):
        if not active:
            continue
        beta_key, phi_key = f"beta_{index + 1}", f"phi_{index + 1}"
        z_domain = raw.domain_covariates[index]
        z_grid = np.linspace(z_domain[0], z_domain[1], 201)
        beta_true = raw.truth.beta[index](t_grid)
        phi_true = raw.truth.phi[index](z_grid)
        pair_available = beta_key in mapped and phi_key in mapped
        if pair_available:
            beta_grid, beta_values = mapped[beta_key]
            phi_grid, phi_values = mapped[phi_key]
            beta_hat = np.interp(t_grid, beta_grid, beta_values)
            phi_hat = np.interp(z_grid, phi_grid, phi_values)
        else:
            beta_hat = np.zeros_like(t_grid)
            phi_hat = np.zeros_like(z_grid)
        # Missing or non-identifiable factors receive a fixed zero-estimate
        # penalty instead of disappearing from the Monte Carlo denominator.
        factor_beta = beta_hat if pair_available and index not in invalid else np.zeros_like(t_grid)
        factor_phi = phi_hat if pair_available and index not in invalid else np.zeros_like(z_grid)
        factor_errors.append(
            _integrated_mse(t_grid, factor_beta - beta_true)
            + _integrated_mse(z_grid, factor_phi - phi_true)
        )
        surface_hat = beta_hat[:, None] * phi_hat[None, :]
        surface_true = beta_true[:, None] * phi_true[None, :]
        surface_errors.append(float(np.mean((surface_hat - surface_true) ** 2)))
    return (
        baseline_ise,
        float(np.sum(surface_errors)) if surface_errors else float("nan"),
        float(np.sum(factor_errors)) if factor_errors else float("nan"),
    )


def _observed_factor_mse_total(
    raw: PublishedDataset,
    data: SubjectDataset,
    curves: Iterable[Mapping[str, object]],
    *,
    include_baseline: bool,
) -> float:
    """Paper-style pointwise function error on registered observed arguments."""

    mapped = _curve_map(curves)
    errors: list[float] = []
    if include_baseline:
        if "baseline" not in mapped:
            return float("nan")
        grid, values = mapped["baseline"]
        estimate = np.interp(data.time, grid, values)
        errors.append(float(np.mean((estimate - raw.truth.beta0(data.time)) ** 2)))
    for index, active in enumerate(raw.truth.active):
        if not active:
            continue
        beta_key, phi_key = f"beta_{index + 1}", f"phi_{index + 1}"
        if beta_key not in mapped or phi_key not in mapped:
            return float("nan")
        beta_grid, beta_values = mapped[beta_key]
        phi_grid, phi_values = mapped[phi_key]
        beta_hat = np.interp(data.time, beta_grid, beta_values)
        phi_hat = np.interp(data.covariates[:, index], phi_grid, phi_values)
        errors.append(
            float(np.mean((beta_hat - raw.truth.beta[index](data.time)) ** 2))
        )
        errors.append(
            float(
                np.mean(
                    (phi_hat - raw.truth.phi[index](data.covariates[:, index])) ** 2
                )
            )
        )
    return float(np.sum(errors)) if errors else float("nan")


def _selection_metrics(
    active: Sequence[bool], selected: Sequence[int], *, selection_capability: bool
) -> tuple[float, float, float]:
    truth = {index for index, flag in enumerate(active) if flag}
    null = set(range(len(active))) - truth
    if not selection_capability or not null:
        return float("nan"), float("nan"), float("nan")
    selected_set = {int(item) for item in selected}
    true_positive = len(selected_set & truth)
    false_positive = len(selected_set & null)
    tpr = true_positive / len(truth) if truth else float("nan")
    fdr = false_positive / len(selected_set) if selected_set else 0.0
    return float(tpr), float(fdr), float(len(selected_set))


def _base_row(
    scenario: Scenario,
    raw: PublishedDataset,
    dataset: SubjectDataset,
    train: SubjectDataset,
    test: SubjectDataset,
    split: SubjectSplit | SourceSplitAudit,
    *,
    mode: str,
    replicate: int,
    seed: int,
    split_seed: int,
    method: str,
    applicability: str,
    applicability_reason: str,
    admission_status: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "phase": scenario.phase,
        "example": scenario.example,
        "protocol": scenario.protocol,
        "scenario": scenario.scenario,
        "replicate": replicate,
        "seed": seed,
        "split_seed": split_seed,
        "split_unit": scenario.split_unit,
        "method": method,
        "method_display_name": (
            METHOD_SPECS[method].display_name
            if method in METHOD_SPECS
            else method
        ),
        "method_version": METHOD_SPECS.get(method).version if method in METHOD_SPECS else "unregistered",
        "applicability": applicability,
        "applicability_reason": applicability_reason,
        "admission_status": admission_status,
        "attempt_status": "not_attempted",
        "converged": False,
        "failure_code": "",
        "failure_message": "",
        "design_id": raw.design_id,
        "provenance": raw.provenance,
        "n_subjects": dataset.n_subjects,
        "n_train_subjects": train.n_subjects,
        "n_test_subjects": test.n_subjects,
        "n_rows": dataset.n_rows,
        "n_covariates": dataset.covariates.shape[1],
        "n_active": int(sum(raw.truth.active)),
        "has_null_blocks": bool(not all(raw.truth.active)),
        "data_hash": dataset.data_hash,
        "train_subject_hash": split.train_hash,
        "test_subject_hash": split.test_hash,
        "tuning_json": "{}",
        "tuning_sha256": _sha256_bytes(b"{}"),
        "realized_tuning_json": "{}",
        "realized_tuning_sha256": _sha256_bytes(b"{}"),
        "runtime_seconds": float("nan"),
        "peak_python_memory_mb": float("nan"),
        "observed_test_mspe": float("nan"),
        "test_mse": float("nan"),
        "noise_free_test_mspe": float("nan"),
        "baseline_ise": float("nan"),
        "component_ise": float("nan"),
        "factor_ise": float("nan"),
        "paper_observed_factor_mse_total": float("nan"),
        "paper_training_function_mse_total": float("nan"),
        "tpr": float("nan"),
        "fdr": float("nan"),
        "model_size": float("nan"),
        "selected_blocks_json": "[]",
        "fit_metadata_json": "{}",
    }


def _failure_row(row: dict[str, object], code: str, message: str) -> dict[str, object]:
    row["attempt_status"] = "failed"
    row["failure_code"] = code
    row["failure_message"] = message[:2000]
    return row


def run_one_method(
    adapter: object,
    scenario: Scenario,
    raw: PublishedDataset,
    dataset: SubjectDataset,
    train: SubjectDataset,
    test: SubjectDataset,
    split: SubjectSplit | SourceSplitAudit,
    *,
    mode: str,
    quick: bool,
    replicate: int,
    seed: int,
    split_seed: int,
    applicability: str,
    applicability_reason: str,
    admission_status: str,
    preflight_report: object | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
    method = str(adapter.label)
    row = _base_row(
        scenario,
        raw,
        dataset,
        train,
        test,
        split,
        mode=mode,
        replicate=replicate,
        seed=seed,
        split_seed=split_seed,
        method=method,
        applicability=applicability,
        applicability_reason=applicability_reason,
        admission_status=admission_status,
    )
    if applicability == Applicability.N_A_BY_DESIGN.value:
        row["attempt_status"] = "N/A by design"
        row["failure_code"] = "not_applicable"
        return row, [], None
    if applicability not in {Applicability.APPLICABLE.value, "applicable"}:
        row["attempt_status"] = "not_evaluated"
        row["failure_code"] = "applicability_not_registered"
        return row, [], None
    if scenario.phase == "common" and method != TRACE and admission_status != "admitted":
        row["attempt_status"] = "blocked_reproduction_gate"
        row["failure_code"] = "reproduction_not_verified"
        return row, [], None

    preflight = adapter.preflight() if preflight_report is None else preflight_report
    row["method_version"] = str(preflight.version)
    if not preflight.ready:
        return (
            _failure_row(row, str(preflight.code), str(preflight.message)),
            [],
            None,
        )

    tuning = _default_tuning(method, scenario, quick=quick)
    tuning_json = _canonical_json(tuning)
    row["tuning_json"] = tuning_json
    row["tuning_sha256"] = _sha256_bytes(tuning_json.encode("utf-8"))
    started = time.perf_counter()
    tracemalloc.start()
    try:
        artifact = adapter.fit(train, seed=seed, tuning=tuning)
        realized_tuning_json = _canonical_json(
            dict(getattr(artifact, "tuning", tuning))
        )
        row["realized_tuning_json"] = realized_tuning_json
        row["realized_tuning_sha256"] = _sha256_bytes(
            realized_tuning_json.encode("utf-8")
        )
        row["converged"] = bool(artifact.converged)
        row["attempt_status"] = "success" if artifact.converged else "failed"
        if not artifact.converged:
            row["failure_code"] = "nonconvergence"
            row["failure_message"] = "Adapter returned a finite fit but did not meet its stopping rule."
        raw_curves = tuple(adapter.factor_curves(artifact))
        curves, identification_audit = _common_identify_curves(
            raw_curves,
            n_covariates=dataset.covariates.shape[1],
            time_domain=raw.domain_time,
            covariate_domains=raw.domain_covariates,
        )
        baseline_ise, component_ise, factor_ise = _truth_metrics(
            raw,
            curves,
            invalid_factor_blocks=identification_audit["invalid_blocks"],
        )
        row["baseline_ise"] = baseline_ise
        row["component_ise"] = component_ise
        row["factor_ise"] = factor_ise
        row["paper_observed_factor_mse_total"] = _observed_factor_mse_total(
            raw, train, curves, include_baseline=True
        )
        row["paper_training_function_mse_total"] = _observed_factor_mse_total(
            raw, train, curves, include_baseline=True
        )
        selected = tuple(int(item) for item in artifact.selected_blocks)
        row["selected_blocks_json"] = _canonical_json(list(selected))
        spec = METHOD_SPECS.get(method)
        tpr, fdr, model_size = _selection_metrics(
            raw.truth.active,
            selected,
            selection_capability=bool(spec and spec.selection_capability),
        )
        row["tpr"], row["fdr"], row["model_size"] = tpr, fdr, model_size
        fit_metadata = dict(artifact.metadata)
        fit_metadata["common_factor_identification"] = identification_audit
        predictions: list[dict[str, object]] = []
        if method == ZSY:
            # The pinned function exposes fitted rows but no coefficients or
            # predict method.  Retain its estimable function/selection metrics;
            # held-out MSPE is N/A by capability, not a failed fit and not a
            # license to construct an extrapolation proxy.
            fit_metadata["held_out_prediction"] = "N/A by capability"
        else:
            prediction = np.asarray(adapter.predict(artifact, test), dtype=float)
            if prediction.shape != (test.n_rows,):
                raise ValueError(
                    f"prediction shape {prediction.shape} does not match {(test.n_rows,)}"
                )
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError("prediction contains a non-finite value")
            row["observed_test_mspe"] = float(np.mean((prediction - test.response) ** 2))
            row["test_mse"] = row["observed_test_mspe"]
            if test.noise_free_target is not None:
                row["noise_free_test_mspe"] = float(
                    np.mean((prediction - test.noise_free_target) ** 2)
                )
            predictions = [
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario": scenario.scenario,
                    "replicate": replicate,
                    "seed": seed,
                    "method": method,
                    "row_id": test.row_id[index],
                    "subject_id": test.subject_id[index],
                    "observed_response": float(test.response[index]),
                    "noise_free_target": (
                        float("nan")
                        if test.noise_free_target is None
                        else float(test.noise_free_target[index])
                    ),
                    "prediction": float(prediction[index]),
                }
                for index in range(test.n_rows)
            ]
        row["fit_metadata_json"] = _canonical_json(fit_metadata)
        row["runtime_seconds"] = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        row["peak_python_memory_mb"] = peak / (1024**2)
        curve_record = {
            "schema_version": SCHEMA_VERSION,
            "scenario": scenario.scenario,
            "replicate": replicate,
            "seed": seed,
            "method": method,
            "curves": curves,
        }
        return row, predictions, curve_record
    except Exception as error:  # every attempted failure stays in the denominator
        row["runtime_seconds"] = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        row["peak_python_memory_mb"] = peak / (1024**2)
        row["fit_metadata_json"] = _canonical_json(
            {"traceback": traceback.format_exc(limit=8)}
        )
        code = getattr(error, "code", type(error).__name__)
        return _failure_row(row, str(code), f"{type(error).__name__}: {error}"), [], None
    finally:
        tracemalloc.stop()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    include_header: bool,
) -> bytes:
    """Serialize a bounded CSV chunk with platform-independent newlines."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    if include_header:
        writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a small journal record durably, then atomically publish it."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_durable(path: Path, payload: bytes) -> int:
    """Append one already-validated cohort chunk and return the byte offset."""

    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return int(handle.tell())


def _truncate_to_offset(path: Path, offset: int) -> None:
    if offset < 0:
        raise RuntimeError(f"negative committed offset for {path.name}")
    if not path.exists():
        raise RuntimeError(f"committed output is missing: {path}")
    size = path.stat().st_size
    if size < offset:
        raise RuntimeError(
            f"{path.name} is shorter than its committed offset ({size} < {offset})"
        )
    with path.open("r+b") as handle:
        handle.truncate(offset)
        handle.flush()
        os.fsync(handle.fileno())


def _new_prefix_hasher(path: Path) -> "hashlib._Hash":
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest


def _source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "scripts" / "run_trace_tuning_calibration.py",
        ROOT / "scripts" / "run_reproduction_audit.py",
        ROOT / "scripts" / "run_macs_application.py",
        ROOT / "scripts" / "analyze_strict_results.py",
        ROOT / "experiments" / "dgp.py",
        ROOT / "src" / "trace_vcam.py",
        ROOT / "src" / "trace_tuning_protocol.py",
        ROOT / "protocol" / "trace_tuning_v1.json",
    ]
    paths.extend(sorted((ROOT / "benchmarks").rglob("*.py")))
    paths.extend(sorted((ROOT / "benchmarks" / "runners").glob("*.R")))
    paths.extend(sorted((ROOT / "benchmarks" / "vendor").rglob("*.R")))
    paths.extend(sorted((ROOT / "benchmarks" / "vendor").rglob("ORIGIN.json")))
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in paths
        if path.exists()
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


OUTPUT_STREAM_KEYS = ("results", "predictions", "curves")


def _run_contract(
    *,
    mode: str,
    root_seed: int,
    jobs: int,
    scenarios: Sequence[Scenario],
    replications: Mapping[str, int],
    targets: Mapping[str, object],
    trace_tuning_lock: Mapping[str, object],
    preflight: Mapping[str, object],
    include_reproduction_audit: bool,
) -> dict[str, object]:
    """Return every immutable input that permits a safe interrupted resume."""

    return {
        "schema_version": RUN_FINGERPRINT_SCHEMA_VERSION,
        "mode": mode,
        "root_seed": int(root_seed),
        "execution": {
            "jobs": int(jobs),
            "ordered_task_scheduler_version": ORDERED_TASK_SCHEDULER_VERSION,
            "ordered_task_prefetch_factor": ORDERED_TASK_PREFETCH_FACTOR,
            "max_outstanding_futures": _ordered_task_prefetch_limit(jobs),
        },
        "include_reproduction_audit": bool(include_reproduction_audit),
        "scenario_registry": [asdict(item) for item in scenarios],
        "replications": {str(key): int(value) for key, value in replications.items()},
        "method_order": list(FIXED_METHOD_LABELS),
        "source_sha256": _source_hashes(),
        "trace_calibration": {
            "content_sha256": str(trace_tuning_lock["content_sha256"]),
            "file_sha256": str(trace_tuning_lock["file_sha256"]),
            "lambda_ratio": float(trace_tuning_lock["lambda_ratio"]),
            "roughness": float(trace_tuning_lock["roughness"]),
            "tuning_mode": str(trace_tuning_lock["tuning_mode"]),
        },
        "published_targets_sha256": _sha256_bytes(
            _canonical_json(targets).encode("utf-8")
        ),
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": _package_versions(),
        "preflight": preflight,
    }


def _run_fingerprint(contract: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json(contract).encode("utf-8"))


def _ordered_registered_tasks(
    scenarios: Sequence[Scenario], replications: Mapping[str, int]
) -> list[tuple[Scenario, int]]:
    return [
        (scenario, replicate)
        for scenario in scenarios
        for replicate in range(int(replications[scenario.scenario]))
    ]


def _progress_template(
    *,
    contract: Mapping[str, object],
    fingerprint: str,
    expected_cohorts: int,
    paths: Mapping[str, Path],
) -> dict[str, object]:
    offsets = {key: int(paths[key].stat().st_size) for key in OUTPUT_STREAM_KEYS}
    hashes = {key: file_sha256(paths[key]) for key in OUTPUT_STREAM_KEYS}
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "run_contract": dict(contract),
        "expected_cohorts": int(expected_cohorts),
        "committed_cohorts": 0,
        "committed_offsets": offsets,
        "committed_sha256": hashes,
        "last_completed": None,
        "status": "running",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _initialize_or_restore_streams(
    paths: Mapping[str, Path],
    *,
    contract: Mapping[str, object],
    fingerprint: str,
    expected_cohorts: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Restore the last committed three-stream transaction boundary.

    Bytes written after the atomically published offsets belong to an
    interrupted cohort and are truncated before any task is skipped.
    """

    progress_path = paths["progress"]
    if not progress_path.exists():
        existing = [str(paths[key]) for key in OUTPUT_STREAM_KEYS if paths[key].exists()]
        if existing:
            raise RuntimeError(
                "strict outputs exist without a resumable progress journal; "
                f"use a new output directory or archive them first: {existing}"
            )
        _atomic_write_bytes(
            paths["results"], _csv_bytes(RESULT_FIELDS, [], include_header=True)
        )
        _atomic_write_bytes(
            paths["predictions"],
            _csv_bytes(PREDICTION_FIELDS, [], include_header=True),
        )
        _atomic_write_bytes(paths["curves"], b"")
        progress = _progress_template(
            contract=contract,
            fingerprint=fingerprint,
            expected_cohorts=expected_cohorts,
            paths=paths,
        )
        _atomic_write_json(progress_path, progress)
    else:
        with progress_path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if not isinstance(progress, Mapping):
            raise RuntimeError("strict progress journal is not a JSON object")
        if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise RuntimeError(
                "strict output uses a legacy/non-resumable progress schema; "
                "archive it and use a dedicated output directory"
            )
        if str(progress.get("run_fingerprint")) != fingerprint:
            raise RuntimeError(
                "strict resume fingerprint mismatch (mode/seed/registry/methods/"
                "source/calibration/environment/execution changed)"
            )
        if _canonical_json(progress.get("run_contract")) != _canonical_json(contract):
            raise RuntimeError("strict resume contract differs despite its fingerprint")
        if int(progress.get("expected_cohorts", -1)) != expected_cohorts:
            raise RuntimeError("strict resume expected-cohort count mismatch")
        committed = int(progress.get("committed_cohorts", -1))
        if not 0 <= committed <= expected_cohorts:
            raise RuntimeError("strict progress contains an invalid committed-cohort count")
        offsets = progress.get("committed_offsets")
        hashes = progress.get("committed_sha256")
        if not isinstance(offsets, Mapping) or not isinstance(hashes, Mapping):
            raise RuntimeError("strict progress lacks committed offsets or hashes")
        for key in OUTPUT_STREAM_KEYS:
            offset = int(offsets.get(key, -1))
            _truncate_to_offset(paths[key], offset)
            actual_hash = file_sha256(paths[key])
            if actual_hash != str(hashes.get(key, "")):
                raise RuntimeError(
                    f"committed {key} prefix hash mismatch; refusing to skip cohorts"
                )
        progress = dict(progress)

    hashers = {key: _new_prefix_hasher(paths[key]) for key in OUTPUT_STREAM_KEYS}
    return dict(progress), hashers


_RESULT_INTEGER_FIELDS = {
    "replicate",
    "seed",
    "split_seed",
    "n_subjects",
    "n_train_subjects",
    "n_test_subjects",
    "n_rows",
    "n_covariates",
    "n_active",
}
_RESULT_FLOAT_FIELDS = {
    "runtime_seconds",
    "peak_python_memory_mb",
    "observed_test_mspe",
    "test_mse",
    "noise_free_test_mspe",
    "baseline_ise",
    "component_ise",
    "factor_ise",
    "paper_observed_factor_mse_total",
    "paper_training_function_mse_total",
    "tpr",
    "fdr",
    "model_size",
}
_RESULT_BOOLEAN_FIELDS = {"converged", "has_null_blocks"}


def _coerce_result_row(row: Mapping[str, object]) -> dict[str, object]:
    result = dict(row)
    for field in _RESULT_INTEGER_FIELDS:
        value = result.get(field)
        if value not in {None, ""}:
            result[field] = int(str(value))
    for field in _RESULT_FLOAT_FIELDS:
        value = result.get(field)
        if value not in {None, ""}:
            result[field] = float(str(value))
    for field in _RESULT_BOOLEAN_FIELDS:
        value = result.get(field)
        if isinstance(value, bool):
            continue
        if str(value) == "True":
            result[field] = True
        elif str(value) == "False":
            result[field] = False
    return result


def _expected_methods(scenario: Scenario) -> tuple[str, ...]:
    return (
        (str(scenario.owner),)
        if scenario.phase == "reproduction"
        else tuple(FIXED_METHOD_LABELS)
    )


def _validate_cohort_rows(
    rows: Sequence[Mapping[str, object]],
    scenario: Scenario,
    replicate: int,
    *,
    mode: str,
    root_seed: int,
) -> None:
    expected_methods = _expected_methods(scenario)
    observed_methods = tuple(str(row.get("method")) for row in rows)
    if observed_methods != expected_methods:
        raise RuntimeError(
            f"{scenario.scenario}/{replicate}: method order {observed_methods} "
            f"does not match {expected_methods}"
        )
    expected_seed = _stable_seed(root_seed, scenario.scenario, replicate, "data")
    expected_split_seed = _stable_seed(
        root_seed, scenario.scenario, replicate, "subject-split"
    )
    for row in rows:
        expected_values = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "phase": scenario.phase,
            "example": scenario.example,
            "protocol": scenario.protocol,
            "scenario": scenario.scenario,
        }
        for field, expected in expected_values.items():
            if str(row.get(field)) != str(expected):
                raise RuntimeError(
                    f"{scenario.scenario}/{replicate}: invalid result {field}"
                )
        if int(row.get("replicate", -1)) != replicate:
            raise RuntimeError(f"{scenario.scenario}/{replicate}: invalid replicate")
        if int(row.get("seed", -1)) != expected_seed:
            raise RuntimeError(f"{scenario.scenario}/{replicate}: invalid data seed")
        if int(row.get("split_seed", -1)) != expected_split_seed:
            raise RuntimeError(f"{scenario.scenario}/{replicate}: invalid split seed")
    for field in (
        "data_hash",
        "train_subject_hash",
        "test_subject_hash",
        "seed",
        "split_seed",
    ):
        values = {str(row.get(field)) for row in rows}
        if len(values) != 1 or "" in values:
            raise RuntimeError(
                f"{scenario.scenario}/{replicate}: inconsistent shared {field}"
            )


def _load_and_validate_committed_results(
    path: Path,
    tasks: Sequence[tuple[Scenario, int]],
    committed_cohorts: int,
    *,
    mode: str,
    root_seed: int,
) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(RESULT_FIELDS):
            raise RuntimeError("strict result header differs from the registered schema")
        rows = [_coerce_result_row(row) for row in reader]
    cursor = 0
    for scenario, replicate in tasks[:committed_cohorts]:
        count = len(_expected_methods(scenario))
        cohort = rows[cursor : cursor + count]
        if len(cohort) != count:
            raise RuntimeError("committed result stream ends inside a cohort")
        _validate_cohort_rows(
            cohort, scenario, replicate, mode=mode, root_seed=root_seed
        )
        cursor += count
    if cursor != len(rows):
        raise RuntimeError("committed result stream contains unregistered trailing rows")
    return rows


def _result_groups(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], dict[str, Mapping[str, object]]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["scenario"]), int(row["replicate"]))
        grouped.setdefault(key, {})[str(row["method"])] = row
    return grouped


def _validate_committed_predictions(
    path: Path,
    tasks: Sequence[tuple[Scenario, int]],
    committed_cohorts: int,
    results: Sequence[Mapping[str, object]],
) -> None:
    registered = {
        (scenario.scenario, replicate): index
        for index, (scenario, replicate) in enumerate(tasks[:committed_cohorts])
    }
    result_groups = _result_groups(results)
    expected_success = {
        (key, method)
        for key, methods in result_groups.items()
        for method, row in methods.items()
        if str(row.get("attempt_status")) == "success" and method != ZSY
    }
    seen_methods: set[tuple[tuple[str, int], str]] = set()
    previous_order: tuple[int, int, str] | None = None
    current_group: tuple[tuple[str, int], str] | None = None
    current_row_ids: list[str] = []
    reference_key: tuple[str, int] | None = None
    reference_row_ids: tuple[str, ...] | None = None

    def finish_group() -> None:
        nonlocal current_group, current_row_ids, reference_key, reference_row_ids
        if current_group is None:
            return
        key, method = current_group
        row_ids = tuple(current_row_ids)
        if len(row_ids) != len(set(row_ids)):
            raise RuntimeError(f"duplicate prediction row IDs in {key}/{method}")
        if reference_key != key:
            reference_key = key
            reference_row_ids = row_ids
        elif reference_row_ids != row_ids:
            raise RuntimeError(f"prediction row IDs differ across methods in {key}")
        current_group = None
        current_row_ids = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(PREDICTION_FIELDS):
            raise RuntimeError("strict prediction header differs from the registered schema")
        for row in reader:
            key = (str(row["scenario"]), int(row["replicate"]))
            if key not in registered:
                raise RuntimeError(f"prediction belongs to an uncommitted cohort: {key}")
            method = str(row["method"])
            methods = result_groups.get(key, {})
            if method not in methods or method == ZSY:
                raise RuntimeError(f"unexpected prediction method in {key}: {method}")
            method_order = list(FIXED_METHOD_LABELS).index(method)
            order = (registered[key], method_order, str(row["row_id"]))
            if previous_order is not None and order < previous_order:
                raise RuntimeError("prediction stream is not in registered canonical order")
            previous_order = order
            group = (key, method)
            if group != current_group:
                finish_group()
                if group in seen_methods:
                    raise RuntimeError(f"noncontiguous prediction method group: {group}")
                seen_methods.add(group)
                current_group = group
            current_row_ids.append(str(row["row_id"]))
            if str(row.get("schema_version")) != SCHEMA_VERSION:
                raise RuntimeError("prediction schema version mismatch")
            result_seed = int(methods[method]["seed"])
            if int(row.get("seed", -1)) != result_seed:
                raise RuntimeError(f"prediction seed mismatch in {key}/{method}")
    finish_group()
    if not expected_success.issubset(seen_methods):
        missing = sorted(expected_success - seen_methods)
        raise RuntimeError(f"successful fits lack committed predictions: {missing[:5]}")


def _validate_committed_curves(
    path: Path,
    tasks: Sequence[tuple[Scenario, int]],
    committed_cohorts: int,
    results: Sequence[Mapping[str, object]],
) -> None:
    registered = {
        (scenario.scenario, replicate): index
        for index, (scenario, replicate) in enumerate(tasks[:committed_cohorts])
    }
    result_groups = _result_groups(results)
    expected_success = {
        (key, method)
        for key, methods in result_groups.items()
        for method, row in methods.items()
        if str(row.get("attempt_status")) == "success"
    }
    seen: set[tuple[tuple[str, int], str]] = set()
    previous_order: tuple[int, int] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (str(record["scenario"]), int(record["replicate"]))
            if key not in registered:
                raise RuntimeError(f"curve row belongs to an uncommitted cohort: {key}")
            method = str(record["method"])
            if method not in result_groups.get(key, {}):
                raise RuntimeError(f"unexpected curve method in {key}: {method}")
            order = (registered[key], list(FIXED_METHOD_LABELS).index(method))
            if previous_order is not None and order <= previous_order:
                raise RuntimeError("curve stream is duplicated or not canonically ordered")
            previous_order = order
            identity = (key, method)
            if identity in seen:
                raise RuntimeError(f"duplicate curve row: {identity}")
            seen.add(identity)
            if str(record.get("schema_version")) != SCHEMA_VERSION:
                raise RuntimeError("curve schema version mismatch")
            if int(record.get("seed", -1)) != int(result_groups[key][method]["seed"]):
                raise RuntimeError(f"curve seed mismatch in {key}/{method}")
            if not isinstance(record.get("curves"), (list, tuple)):
                raise RuntimeError(f"curve payload is missing in {key}/{method}")
    if not expected_success.issubset(seen):
        missing = sorted(expected_success - seen)
        raise RuntimeError(f"successful fits lack committed curves: {missing[:5]}")


def _normalize_cohort_output(
    scenario: Scenario,
    replicate: int,
    task_result: object,
    *,
    mode: str,
    root_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    method_index = {method: index for index, method in enumerate(FIXED_METHOD_LABELS)}
    triples = [task_result] if scenario.phase == "reproduction" else list(task_result)  # type: ignore[arg-type]
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for triple in triples:
        row, method_predictions, method_curves = triple  # type: ignore[misc]
        rows.append(dict(row))
        predictions.extend(dict(item) for item in method_predictions)
        if method_curves is not None:
            curves.append(dict(method_curves))
    rows.sort(key=lambda row: method_index[str(row["method"])])
    predictions.sort(
        key=lambda row: (method_index[str(row["method"])], str(row["row_id"]))
    )
    curves.sort(key=lambda row: method_index[str(row["method"])])
    _validate_cohort_rows(
        rows, scenario, replicate, mode=mode, root_seed=root_seed
    )
    result_methods = {str(row["method"]): row for row in rows}
    for prediction in predictions:
        if (
            str(prediction.get("scenario")) != scenario.scenario
            or int(prediction.get("replicate", -1)) != replicate
            or str(prediction.get("method")) not in result_methods
        ):
            raise RuntimeError(
                f"{scenario.scenario}/{replicate}: inconsistent prediction payload"
            )
    for curve in curves:
        if (
            str(curve.get("scenario")) != scenario.scenario
            or int(curve.get("replicate", -1)) != replicate
            or str(curve.get("method")) not in result_methods
        ):
            raise RuntimeError(
                f"{scenario.scenario}/{replicate}: inconsistent curve payload"
            )
    successful = {
        method
        for method, row in result_methods.items()
        if str(row.get("attempt_status")) == "success"
    }
    prediction_methods = {str(row["method"]) for row in predictions}
    curve_methods = {str(row["method"]) for row in curves}
    if not {method for method in successful if method != ZSY}.issubset(
        prediction_methods
    ):
        raise RuntimeError(
            f"{scenario.scenario}/{replicate}: successful method lacks predictions"
        )
    if not successful.issubset(curve_methods):
        raise RuntimeError(f"{scenario.scenario}/{replicate}: successful method lacks curves")
    return rows, predictions, curves


def _commit_cohort(
    paths: Mapping[str, Path],
    progress: dict[str, object],
    hashers: Mapping[str, object],
    *,
    scenario: Scenario,
    replicate: int,
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    curves: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    chunks = {
        "results": _csv_bytes(RESULT_FIELDS, rows, include_header=False),
        "predictions": _csv_bytes(
            PREDICTION_FIELDS, predictions, include_header=False
        ),
        "curves": _jsonl_bytes(curves),
    }
    offsets = dict(progress["committed_offsets"])  # type: ignore[arg-type]
    new_offsets: dict[str, int] = {}
    for key in OUTPUT_STREAM_KEYS:
        if paths[key].stat().st_size != int(offsets[key]):
            raise RuntimeError(f"{key} changed after its last committed offset")
        chunk = chunks[key]
        new_offsets[key] = (
            _append_durable(paths[key], chunk) if chunk else int(offsets[key])
        )
        hashers[key].update(chunk)  # type: ignore[union-attr]
    committed = int(progress["committed_cohorts"]) + 1
    updated = dict(progress)
    updated.update(
        committed_cohorts=committed,
        committed_offsets=new_offsets,
        committed_sha256={
            key: hashers[key].hexdigest() for key in OUTPUT_STREAM_KEYS  # type: ignore[union-attr]
        },
        last_completed={
            "phase": scenario.phase,
            "scenario": scenario.scenario,
            "replicate": replicate,
            "result_rows": len(rows),
            "prediction_rows": len(predictions),
            "curve_rows": len(curves),
            "cohort_chunks_sha256": {
                key: _sha256_bytes(chunks[key]) for key in OUTPUT_STREAM_KEYS
            },
        },
        status=(
            "complete"
            if committed == int(progress["expected_cohorts"])
            else "running"
        ),
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _atomic_write_json(paths["progress"], updated)
    return updated


def _validate_shared_rows(rows: Sequence[Mapping[str, object]], scenarios: Sequence[Scenario], reps: Mapping[str, int]) -> list[str]:
    issues: list[str] = []
    keys: set[tuple[object, object, object]] = set()
    method_order = tuple(FIXED_METHOD_LABELS)
    for row in rows:
        key = (row.get("scenario"), row.get("replicate"), row.get("method"))
        if key in keys:
            issues.append(f"duplicate result key: {key}")
        keys.add(key)
    for scenario in scenarios:
        expected_methods = (scenario.owner,) if scenario.phase == "reproduction" else method_order
        for replicate in range(reps[scenario.scenario]):
            cohort = [
                row
                for row in rows
                if row.get("scenario") == scenario.scenario
                and int(row.get("replicate", -1)) == replicate
            ]
            observed = {str(row.get("method")) for row in cohort}
            if observed != set(expected_methods):
                issues.append(
                    f"{scenario.scenario}/{replicate}: methods={sorted(observed)}, expected={sorted(expected_methods)}"
                )
            for field in ("data_hash", "train_subject_hash", "test_subject_hash", "seed", "split_seed"):
                values = {str(row.get(field)) for row in cohort}
                if len(values) != 1:
                    issues.append(f"{scenario.scenario}/{replicate}: non-common {field}")
    return issues


def _reproduction_task(
    scenario: Scenario,
    replicate: int,
    root_seed: int,
    mode: str,
    quick: bool,
    target: Mapping[str, object] | None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
    """Run one source-method reproduction in an isolated worker."""

    seed = _stable_seed(root_seed, scenario.scenario, replicate, "data")
    split_seed = _stable_seed(
        root_seed, scenario.scenario, replicate, "subject-split"
    )
    raw = scenario.build(seed)
    dataset = _subject_dataset(raw, scenario)
    train, test, split = _registered_split(
        dataset, raw, scenario, split_seed=split_seed
    )
    applicability, reason = _safe_applicability(str(scenario.owner), scenario)
    adapters = adapter_registry()
    adapter = adapters[str(scenario.owner)]

    source_block = None if target is None else target.get("source_gate_blocked_reason")
    if source_block:
        row = _base_row(
            scenario,
            raw,
            dataset,
            train,
            test,
            split,
            mode=mode,
            replicate=replicate,
            seed=seed,
            split_seed=split_seed,
            method=str(scenario.owner),
            applicability=applicability,
            applicability_reason=reason,
            admission_status="blocked_source_protocol_inconsistency",
        )
        row["attempt_status"] = "not_evaluated_source_incompatibility"
        row["failure_code"] = "source_protocol_incompatibility"
        row["failure_message"] = str(source_block)[:2000]
        row["method_version"] = str(adapter.preflight().version)
        return row, [], None

    return run_one_method(
        adapter,
        scenario,
        raw,
        dataset,
        train,
        test,
        split,
        mode=mode,
        quick=quick,
        replicate=replicate,
        seed=seed,
        split_seed=split_seed,
        applicability=applicability,
        applicability_reason=reason,
        admission_status="reproduction_under_review",
    )


def _common_task(
    scenario: Scenario,
    replicate: int,
    root_seed: int,
    mode: str,
    quick: bool,
    gates: Mapping[str, Mapping[str, object]],
) -> list[tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]]:
    """Run one common data cohort, keeping every method on the same split."""

    seed = _stable_seed(root_seed, scenario.scenario, replicate, "data")
    split_seed = _stable_seed(
        root_seed, scenario.scenario, replicate, "subject-split"
    )
    raw = scenario.build(seed)
    dataset = _subject_dataset(raw, scenario)
    train, test, split = _registered_split(
        dataset, raw, scenario, split_seed=split_seed
    )
    adapters = adapter_registry()
    results = []
    for method in FIXED_METHOD_LABELS:
        applicability, reason = _safe_applicability(method, scenario)
        admission = "not_required" if method == TRACE else str(
            gates.get(method, {}).get("status", "pending_no_reproduction")
        )
        results.append(
            run_one_method(
                adapters[method],
                scenario,
                raw,
                dataset,
                train,
                test,
                split,
                mode=mode,
                quick=quick,
                replicate=replicate,
                seed=seed,
                split_seed=split_seed,
                applicability=applicability,
                applicability_reason=reason,
                admission_status=admission,
            )
        )
    return results


def _ordered_task_prefetch_limit(jobs: int) -> int:
    """Return the fixed submitted-but-uncommitted cohort bound."""

    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs == 1:
        return 1
    return ORDERED_TASK_PREFETCH_FACTOR * jobs


def _ordered_task_results(
    futures: Sequence[tuple[object, Scenario, int]],
    *,
    jobs: int,
    worker: Callable[..., object],
    worker_arguments: Callable[[Scenario, int], tuple[object, ...]],
) -> Iterable[tuple[Scenario, int, object]]:
    """Yield in registry order with bounded eager task prefetch.

    Registry-order commits make the three output streams canonical without a
    terminal in-memory sort.  The submitted-but-not-yielded window is capped
    at ``ORDERED_TASK_PREFETCH_FACTOR * jobs`` (one in sequential mode).  This
    lets workers continue beyond the first batch when the registry head is
    slow, while placing the same explicit cohort-count bound on completed
    prediction/curve payloads retained by the parent process.
    """

    if jobs == 1:
        for _, scenario, replicate in futures:
            yield scenario, replicate, worker(*worker_arguments(scenario, replicate))
        return
    iterator = iter(futures)
    max_outstanding = _ordered_task_prefetch_limit(jobs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        pending: deque[tuple[Scenario, int, object]] = deque()

        def submit_one() -> bool:
            try:
                _, scenario, replicate = next(iterator)
            except StopIteration:
                return False
            future = executor.submit(
                worker, *worker_arguments(scenario, replicate)
            )
            pending.append((scenario, replicate, future))
            return True

        for _ in range(max_outstanding):
            if not submit_one():
                break
        while pending:
            scenario, replicate, future = pending.popleft()
            yield scenario, replicate, future.result()  # type: ignore[union-attr]
            submit_one()


def _existing_final_is_valid(
    paths: Mapping[str, Path], run_fingerprint: str
) -> bool:
    if not paths["metadata"].exists() or not paths["metadata_sha256"].exists():
        return False
    try:
        declared = paths["metadata_sha256"].read_text(encoding="ascii").split()[0]
        if declared != file_sha256(paths["metadata"]):
            return False
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if str(metadata.get("run_fingerprint")) != run_fingerprint:
            return False
        files = metadata.get("files", {})
        for key in OUTPUT_STREAM_KEYS:
            path = paths[key]
            audit = files.get(key, {})
            if (
                not path.exists()
                or int(audit.get("bytes", -1)) != path.stat().st_size
                or str(audit.get("sha256")) != file_sha256(path)
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def execute(args: argparse.Namespace) -> dict[str, Path]:
    quick = bool(args.quick)
    mode = "quick" if quick else "formal"
    trace_tuning_lock = load_trace_tuning_lock()
    scenarios = registered_scenarios(
        quick=quick,
        include_reproduction_audit=bool(args.include_reproduction_audit),
    )
    replications = {
        scenario.scenario: (
            int(args.quick_replications) if quick else scenario.formal_replications
        )
        for scenario in scenarios
    }
    targets = load_published_targets(args.published_targets)
    adapters = adapter_registry()
    preflight_reports: dict[str, object] = {}
    for method in FIXED_METHOD_LABELS:
        print(f"[preflight] {method}", flush=True)
        preflight_reports[method] = adapters[method].preflight()
    preflight = {method: asdict(report) for method, report in preflight_reports.items()}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "strict_results.csv",
        "predictions": output / "strict_predictions.csv",
        "curves": output / "strict_factor_curves.jsonl",
        "metadata": output / "strict_metadata.json",
        "metadata_sha256": output / "strict_metadata.sha256",
        "progress": output / "strict_progress.json",
    }
    contract = _run_contract(
        mode=mode,
        root_seed=int(args.seed),
        jobs=int(args.jobs),
        scenarios=scenarios,
        replications=replications,
        targets=targets,
        trace_tuning_lock=trace_tuning_lock,
        preflight=preflight,
        include_reproduction_audit=bool(args.include_reproduction_audit),
    )
    fingerprint = _run_fingerprint(contract)
    registered_tasks = _ordered_registered_tasks(scenarios, replications)
    progress, hashers = _initialize_or_restore_streams(
        paths,
        contract=contract,
        fingerprint=fingerprint,
        expected_cohorts=len(registered_tasks),
    )
    initially_committed = int(progress["committed_cohorts"])
    result_rows = _load_and_validate_committed_results(
        paths["results"],
        registered_tasks,
        initially_committed,
        mode=mode,
        root_seed=int(args.seed),
    )
    _validate_committed_predictions(
        paths["predictions"], registered_tasks, initially_committed, result_rows
    )
    _validate_committed_curves(
        paths["curves"], registered_tasks, initially_committed, result_rows
    )
    if initially_committed:
        print(
            f"[resume] committed_cohorts={initially_committed}/"
            f"{len(registered_tasks)}; uncommitted tails truncated",
            flush=True,
        )
    if initially_committed == len(registered_tasks) and _existing_final_is_valid(
        paths, fingerprint
    ):
        if progress.get("status") != "finalized":
            progress = dict(progress)
            progress.update(
                status="finalized",
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            _atomic_write_json(paths["progress"], progress)
        print("[resume] all cohorts and final hashes already valid; no-op", flush=True)
        return paths

    jobs = int(args.jobs)
    if jobs > 1:
        # Windows workers spawn after inheriting these limits.  Keeping BLAS at
        # one thread per worker avoids jobs-by-BLAS oversubscription.
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

    reproduction_scenarios = [
        item for item in scenarios if item.phase == "reproduction"
    ]
    reproduction_tasks = [
        (None, scenario, replicate)
        for scenario in reproduction_scenarios
        for replicate in range(replications[scenario.scenario])
    ]
    reproduction_count = len(reproduction_tasks)
    if initially_committed < reproduction_count:
        pending_reproduction = reproduction_tasks[initially_committed:]
        for scenario, replicate, task_result in _ordered_task_results(
            pending_reproduction,
            jobs=jobs,
            worker=_reproduction_task,
            worker_arguments=lambda item, index: (
                item,
                index,
                int(args.seed),
                mode,
                quick,
                targets.get(item.scenario),
            ),
        ):
            rows, predictions, curves = _normalize_cohort_output(
                scenario,
                replicate,
                task_result,
                mode=mode,
                root_seed=int(args.seed),
            )
            progress = _commit_cohort(
                paths,
                progress,
                hashers,
                scenario=scenario,
                replicate=replicate,
                rows=rows,
                predictions=predictions,
                curves=curves,
            )
            result_rows.extend(rows)
            print(
                f"[completed] phase=reproduction scenario={scenario.scenario} "
                f"replicate={replicate} method={scenario.owner} "
                f"status={rows[0]['attempt_status']}",
                flush=True,
            )

    reproduction_audit: dict[str, dict[str, object]] = {}
    for scenario in reproduction_scenarios:
        reproduction_audit[str(scenario.owner)] = assess_reproduction_gate(
            result_rows,
            scenario,
            targets.get(scenario.scenario),
            expected_replications=replications[scenario.scenario],
        )

    gates: dict[str, dict[str, object]] = {
        method: {
            "method": method,
            "passed": True,
            "status": "admitted",
            "admission_basis": "same_setting_original_method_comparison",
            "reason": (
                "The original/author/paper implementation is compared on the "
                "same generated data, seed, and subject split; agreement with "
                "a published table entry is not an admission requirement."
            ),
            "preflight_ready": bool(preflight_reports[method].ready),
            "implementation_version": str(preflight_reports[method].version),
        }
        for method in FIXED_METHOD_LABELS
        if method != TRACE
    }

    common_scenarios = [item for item in scenarios if item.phase == "common"]
    common_tasks = [
        (None, scenario, replicate)
        for scenario in common_scenarios
        for replicate in range(replications[scenario.scenario])
    ]
    common_skip = max(0, initially_committed - reproduction_count)
    pending_common = common_tasks[common_skip:]
    for scenario, replicate, cohort_results in _ordered_task_results(
        pending_common,
        jobs=jobs,
        worker=_common_task,
        worker_arguments=lambda item, index: (
            item,
            index,
            int(args.seed),
            mode,
            quick,
            gates,
        ),
    ):
        rows, predictions, curves = _normalize_cohort_output(
            scenario,
            replicate,
            cohort_results,
            mode=mode,
            root_seed=int(args.seed),
        )
        progress = _commit_cohort(
            paths,
            progress,
            hashers,
            scenario=scenario,
            replicate=replicate,
            rows=rows,
            predictions=predictions,
            curves=curves,
        )
        result_rows.extend(rows)
        statuses = ",".join(
            f"{row['method']}={row['attempt_status']}" for row in rows
        )
        print(
            f"[completed] phase=common scenario={scenario.scenario} "
            f"replicate={replicate} {statuses}",
            flush=True,
        )

    if int(progress["committed_cohorts"]) != len(registered_tasks):
        raise RuntimeError("strict execution ended before every cohort was committed")
    # Final validation streams predictions and curves one cohort at a time;
    # only the 29,490 compact result rows remain resident for metadata audits.
    result_rows = _load_and_validate_committed_results(
        paths["results"],
        registered_tasks,
        len(registered_tasks),
        mode=mode,
        root_seed=int(args.seed),
    )
    _validate_committed_predictions(
        paths["predictions"], registered_tasks, len(registered_tasks), result_rows
    )
    _validate_committed_curves(
        paths["curves"], registered_tasks, len(registered_tasks), result_rows
    )
    cohort_issues = _validate_shared_rows(result_rows, scenarios, replications)
    applicable_external = [
        method
        for method in FIXED_METHOD_LABELS
        if method != TRACE
        and any(
            _safe_applicability(method, scenario)[0] == Applicability.APPLICABLE.value
            for scenario in scenarios
            if scenario.phase == "common"
        )
    ]
    formal_protocol_complete = bool(not quick and not cohort_issues)
    formal_complete = bool(
        formal_protocol_complete
        and all(bool(gates.get(method, {}).get("passed")) for method in applicable_external)
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "run_fingerprint": fingerprint,
        "resume_audit": {
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "initially_committed_cohorts": initially_committed,
            "expected_cohorts": len(registered_tasks),
            "committed_offsets": dict(progress["committed_offsets"]),
            "committed_sha256": dict(progress["committed_sha256"]),
            "uncommitted_tail_policy": "truncate to the last atomically journaled byte offsets",
        },
        "formal_protocol_complete": formal_protocol_complete,
        "descriptive_results_eligible": formal_protocol_complete,
        "formal_claims_eligible": formal_complete,
        "formal_claims_reason": (
            "The locked same-setting cohort protocol passed its integrity audit."
            if formal_complete
            else "Quick mode or an incomplete same-setting cohort audit prevents claims."
        ),
        "benchmark_policy": {
            "primary_comparison": "same data, seed, and subject split",
            "published_value_reproduction_required_for_admission": False,
            "source_reproduction_audit_included": bool(
                args.include_reproduction_audit
            ),
            "failure_policy": "all registered attempts remain in the denominator",
        },
        "trace_tuning_lock": {
            **trace_tuning_lock,
            "path": "protocol/trace_tuning_v1.json",
        },
        "root_seed": args.seed,
        "execution": {
            "jobs": jobs,
            "ordered_task_scheduler_version": ORDERED_TASK_SCHEDULER_VERSION,
            "ordered_task_prefetch_factor": ORDERED_TASK_PREFETCH_FACTOR,
            "max_active_worker_tasks": jobs,
            "max_in_flight_futures": _ordered_task_prefetch_limit(jobs),
            "max_outstanding_futures": _ordered_task_prefetch_limit(jobs),
            "max_parent_buffered_cohort_payloads": _ordered_task_prefetch_limit(
                jobs
            ),
            "worker_blas_threads": 1 if jobs > 1 else None,
            "completion_order_canonicalized": True,
            "streaming_outputs": True,
            "prediction_curve_global_memory": False,
        },
        "scenario_registry": [asdict(item) for item in scenarios],
        "replications": replications,
        "method_order": list(FIXED_METHOD_LABELS),
        "method_display_names": {
            key: value.display_name for key, value in METHOD_SPECS.items()
        },
        "method_specs": {key: asdict(value) for key, value in METHOD_SPECS.items()},
        "preflight": preflight,
        "published_targets": targets,
        "admission_gates": gates,
        "reproduction_audit": reproduction_audit,
        "cohort_audit": {"passed": not cohort_issues, "issues": cohort_issues},
        "metric_definitions": {
            "observed_test_mspe": "mean squared prediction error against held-out observed response",
            "test_mse": "alias of observed_test_mspe used when an original paper labels its target MSPE",
            "noise_free_test_mspe": "mean squared prediction error against held-out conditional mean",
            "common_factor_coordinates": "for every method, phi is Lebesgue-centered with the removed beta-times-constant absorbed into the baseline, then beta is scaled to raw-domain mean one (equivalently integral one after mapping the domain to [0,1]) with inverse scaling of phi",
            "baseline_ise": "domain-average squared error after common factor centering and baseline absorption",
            "component_ise": "sum of grid-integrated squared errors of active centered component surfaces; a missing active block is the zero surface and remains in the denominator",
            "factor_ise": "baseline ISE plus active beta/phi domain-average squared errors in common identified coordinates; a missing or non-identifiable active factor receives the registered zero-estimate factor penalty",
            "paper_observed_factor_mse_total": "sum of baseline/beta/phi squared errors evaluated at the registered training arguments (HHY source statistic)",
            "paper_training_function_mse_total": "sum of baseline/beta/phi squared errors on author-code training arguments (ZSY source statistic)",
            "runtime_seconds": "wall time for adapter fit plus prediction",
            "peak_python_memory_mb": "tracemalloc peak; does not include child R process allocation",
            "tpr_fdr_rule": "reported only for a selection-capable method when the DGP contains a nonempty null set",
            "zsy_prediction": "N/A by capability: pinned author code exposes fitted rows but no out-of-sample predict method",
        },
        "failure_denominator": "every registered method-by-scenario-by-replication row; N/A-by-design is separate from attempted failure",
        "tuning_record_policy": (
            "tuning_json is the requested rule/configuration; realized_tuning_json "
            "is the adapter-returned configuration including selected CV/BIC values "
            "when fitting succeeds, and remains empty when no fit artifact exists"
        ),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": _package_versions(),
        "source_sha256": _source_hashes(),
        "files": {
            key: {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in paths.items()
            if key not in {"metadata", "metadata_sha256", "progress"}
        },
    }
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(paths["metadata"], metadata_bytes)
    metadata_hash = file_sha256(paths["metadata"])
    _atomic_write_bytes(
        paths["metadata_sha256"],
        f"{metadata_hash}  {paths['metadata'].name}\n".encode("ascii"),
    )
    progress = dict(progress)
    progress.update(
        status="finalized",
        metadata_sha256=metadata_hash,
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _atomic_write_json(paths["progress"], progress)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="smoke protocol; never claim-eligible")
    mode.add_argument("--formal", action="store_true", help="locked same-setting comparison counts")
    parser.add_argument("--output", type=Path, required=True, help="dedicated result directory")
    parser.add_argument(
        "--published-targets",
        type=Path,
        default=ROOT / "protocol" / "published_targets.json",
        help="sourced reproduction targets (defaults to the audited workspace registry)",
    )
    parser.add_argument(
        "--include-reproduction-audit",
        action="store_true",
        help=(
            "optionally append published-table reproduction scenarios; these "
            "are diagnostic and never control common-comparison admission"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="independent worker processes; final rows are canonically ordered",
    )
    parser.add_argument(
        "--quick-replications",
        type=int,
        default=1,
        help="replications per quick scenario (ignored in formal mode)",
    )
    args = parser.parse_args(argv)
    if args.quick_replications < 1:
        parser.error("--quick-replications must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = execute(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

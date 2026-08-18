"""Execution, result schema, and cohort-level audit for strict benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .adapters.base import AdapterError, BenchmarkAdapter
from .data import SubjectDataset, SubjectSplit
from .methods import (
    FIXED_METHOD_LABELS,
    METHOD_SPECS,
    Applicability,
    Protocol,
    applicability_for,
)


@dataclass(frozen=True)
class PredictionRecord:
    row_id: str
    subject_id: str
    observed: float
    predicted: float
    noise_free_target: float | None = None


@dataclass(frozen=True)
class CurveEstimate:
    component: str
    domain: str
    grid: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.component or not self.domain:
            raise ValueError("curve component and domain labels are required")
        if len(self.grid) != len(self.values) or not self.grid:
            raise ValueError("curve grid and values must have the same nonzero length")
        if not np.all(np.isfinite(self.grid)) or not np.all(np.isfinite(self.values)):
            raise ValueError("curve grids and values must be finite")


@dataclass(frozen=True)
class FailureInfo:
    code: str
    stage: str
    message: str
    exception_type: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: str
    method: str
    method_version: str
    method_display: str
    protocol: str
    scenario_id: str
    replication_id: int
    seed: int
    data_hash: str
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_subject_hash: str
    test_subject_hash: str
    applicability: str
    applicability_reason: str
    attempted: bool
    tuning: Mapping[str, object]
    converged: bool
    failure: FailureInfo | None
    runtime_seconds: float
    predictions: tuple[PredictionRecord, ...] = ()
    factor_curves: tuple[CurveEstimate, ...] = ()
    selected_blocks: tuple[int, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "vcam-benchmark-result/1":
            raise ValueError("unsupported benchmark result schema")
        if self.method not in FIXED_METHOD_LABELS:
            raise ValueError(f"unregistered method label: {self.method}")
        if set(self.train_subjects).intersection(self.test_subjects):
            raise ValueError("result contains overlapping train/test subjects")
        if self.runtime_seconds < 0 or not np.isfinite(self.runtime_seconds):
            raise ValueError("runtime_seconds must be finite and nonnegative")
        if self.attempted and self.applicability != Applicability.APPLICABLE.value:
            raise ValueError("only scientifically applicable methods may be attempted")
        if self.failure is None and self.attempted and not self.converged:
            raise ValueError("an attempted nonconverged result must carry failure details")
        if self.failure is not None and self.converged:
            raise ValueError("a failed result cannot be marked converged")
        if len(set(self.selected_blocks)) != len(self.selected_blocks):
            raise ValueError("selected_blocks must not contain duplicates")
        if any(index < 0 for index in self.selected_blocks):
            raise ValueError("selected block indices must be nonnegative")
        if any(not np.isfinite(value) for value in self.metrics.values()):
            raise ValueError("stored metrics must be finite; failures belong in failure metadata")

    @property
    def successful(self) -> bool:
        return bool(self.attempted and self.converged and self.failure is None)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def audit_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _curve_from_mapping(curve: Mapping[str, object]) -> CurveEstimate:
    return CurveEstimate(
        component=str(curve["component"]),
        domain=str(curve["domain"]),
        grid=tuple(float(item) for item in curve["grid"]),  # type: ignore[union-attr]
        values=tuple(float(item) for item in curve["values"]),  # type: ignore[union-attr]
    )


def _base_result_fields(
    *,
    adapter: BenchmarkAdapter,
    dataset: SubjectDataset,
    split: SubjectSplit,
    protocol: Protocol,
    scenario_id: str,
    replication_id: int,
    seed: int,
) -> dict[str, object]:
    spec = METHOD_SPECS[adapter.label]
    return {
        "schema_version": "vcam-benchmark-result/1",
        "method": adapter.label,
        "method_version": spec.version,
        "method_display": adapter.label,
        "protocol": protocol.value,
        "scenario_id": str(scenario_id),
        "replication_id": int(replication_id),
        "seed": int(seed),
        "data_hash": dataset.data_hash,
        "train_subjects": split.train_subjects,
        "test_subjects": split.test_subjects,
        "train_subject_hash": split.train_hash,
        "test_subject_hash": split.test_hash,
    }


def run_replication(
    adapter: BenchmarkAdapter,
    dataset: SubjectDataset,
    split: SubjectSplit,
    *,
    protocol: Protocol | str,
    scenario_id: str,
    replication_id: int,
    tuning: Mapping[str, object],
    seed: int | None = None,
) -> BenchmarkResult:
    """Run one method on one precomputed subject split.

    The caller passes the same ``dataset``, ``split``, and ``seed`` to every
    method.  This function never resamples rows and never drops a failed fit.
    """

    protocol = Protocol(protocol)
    split.validate_against(dataset)
    run_seed = split.seed if seed is None else int(seed)
    decision = applicability_for(adapter.label, protocol)
    base = _base_result_fields(
        adapter=adapter,
        dataset=dataset,
        split=split,
        protocol=protocol,
        scenario_id=scenario_id,
        replication_id=replication_id,
        seed=run_seed,
    )
    if not decision.is_applicable:
        code = (
            "n_a_by_design"
            if decision.status is Applicability.N_A_BY_DESIGN
            else "unavailable_not_evaluated"
        )
        return BenchmarkResult(
            **base,
            applicability=decision.status.value,
            applicability_reason=decision.reason,
            attempted=False,
            tuning=dict(tuning),
            converged=False,
            failure=FailureInfo(code, "applicability", decision.reason),
            runtime_seconds=0.0,
            metadata={"supported_metrics": list(decision.supported_metrics)},
        )

    preflight = adapter.preflight()
    if not preflight.ready:
        return BenchmarkResult(
            **{**base, "method_version": preflight.version},
            applicability=decision.status.value,
            applicability_reason=decision.reason,
            attempted=True,
            tuning=dict(tuning),
            converged=False,
            failure=FailureInfo(preflight.code, "preflight", preflight.message),
            runtime_seconds=0.0,
            metadata={
                "supported_metrics": list(decision.supported_metrics),
                "preflight_environment": dict(preflight.environment),
            },
        )

    train = dataset.subset_subjects(split.train_subjects)
    test = dataset.subset_subjects(split.test_subjects)
    start = perf_counter()
    try:
        artifact = adapter.fit(train, seed=run_seed, tuning=tuning)
        if artifact.method != adapter.label:
            raise ValueError("adapter returned an artifact under a different method label")
        if not artifact.converged:
            elapsed = perf_counter() - start
            return BenchmarkResult(
                **{**base, "method_version": artifact.version},
                applicability=decision.status.value,
                applicability_reason=decision.reason,
                attempted=True,
                tuning=artifact.tuning,
                converged=False,
                failure=FailureInfo(
                    "algorithm_nonconvergence",
                    "fit",
                    "The original stopping criterion was not met before the registered safety cap.",
                ),
                runtime_seconds=elapsed,
                selected_blocks=artifact.selected_blocks,
                metadata={
                    **artifact.metadata,
                    "supported_metrics": list(decision.supported_metrics),
                    "preflight_environment": dict(preflight.environment),
                },
            )

        needs_prediction = bool(
            {"noise_free_test_mspe", "test_mse"}.intersection(decision.supported_metrics)
        )
        prediction = (
            np.asarray(adapter.predict(artifact, test), dtype=float)
            if needs_prediction
            else np.empty(0, dtype=float)
        )
        if needs_prediction and (
            prediction.shape != (test.n_rows,) or not np.all(np.isfinite(prediction))
        ):
            raise ValueError("adapter returned nonfinite predictions or the wrong row count")
        prediction_records: tuple[PredictionRecord, ...] = ()
        metrics: dict[str, float] = {}
        if needs_prediction:
            prediction_records = tuple(
                PredictionRecord(
                    row_id=str(test.row_id[index]),
                    subject_id=str(test.subject_id[index]),
                    observed=float(test.response[index]),
                    predicted=float(prediction[index]),
                    noise_free_target=(
                        None
                        if test.noise_free_target is None
                        else float(test.noise_free_target[index])
                    ),
                )
                for index in range(test.n_rows)
            )
            metrics["test_mse"] = float(np.mean((test.response - prediction) ** 2))
            if test.noise_free_target is not None:
                metrics["noise_free_test_mspe"] = float(
                    np.mean((test.noise_free_target - prediction) ** 2)
                )
        elapsed = perf_counter() - start
        metrics["runtime_seconds"] = elapsed
        curves = tuple(_curve_from_mapping(item) for item in adapter.factor_curves(artifact))
        return BenchmarkResult(
            **{**base, "method_version": artifact.version},
            applicability=decision.status.value,
            applicability_reason=decision.reason,
            attempted=True,
            tuning=artifact.tuning,
            converged=True,
            failure=None,
            runtime_seconds=elapsed,
            predictions=prediction_records,
            factor_curves=curves,
            selected_blocks=artifact.selected_blocks,
            metrics=metrics,
            metadata={
                **artifact.metadata,
                "supported_metrics": list(decision.supported_metrics),
                "preflight_environment": dict(preflight.environment),
            },
        )
    except Exception as error:
        elapsed = perf_counter() - start
        code = error.code if isinstance(error, AdapterError) else "fit_or_predict_exception"
        return BenchmarkResult(
            **{**base, "method_version": preflight.version},
            applicability=decision.status.value,
            applicability_reason=decision.reason,
            attempted=True,
            tuning=dict(tuning),
            converged=False,
            failure=FailureInfo(code, "fit/predict", str(error), type(error).__name__),
            runtime_seconds=elapsed,
            metadata={
                "supported_metrics": list(decision.supported_metrics),
                "preflight_environment": dict(preflight.environment),
            },
        )


def audit_replication_cohort(
    results: Sequence[BenchmarkResult], *, require_all_methods: bool = True
) -> dict[str, object]:
    """Fail closed if methods did not receive one shared replication contract."""

    if not results:
        raise ValueError("a replication cohort is empty")
    reference = results[0]
    shared_fields = (
        "protocol",
        "scenario_id",
        "replication_id",
        "seed",
        "data_hash",
        "train_subject_hash",
        "test_subject_hash",
        "train_subjects",
        "test_subjects",
    )
    for result in results[1:]:
        mismatches = [
            field
            for field in shared_fields
            if getattr(result, field) != getattr(reference, field)
        ]
        if mismatches:
            raise ValueError(f"replication contract mismatch for {result.method}: {mismatches}")
    methods = [result.method for result in results]
    if len(methods) != len(set(methods)):
        raise ValueError("a method occurs more than once in the replication cohort")
    if require_all_methods and set(methods) != set(FIXED_METHOD_LABELS):
        missing = sorted(set(FIXED_METHOD_LABELS) - set(methods))
        extra = sorted(set(methods) - set(FIXED_METHOD_LABELS))
        raise ValueError(f"cohort method registry mismatch; missing={missing}, extra={extra}")
    for result in results:
        if result.attempted and not result.successful and result.failure is None:
            raise ValueError(f"failed attempt for {result.method} lacks a failure record")
    cohort_payload = "".join(sorted(result.audit_hash for result in results)).encode("ascii")
    return {
        "valid": True,
        "cohort_hash": hashlib.sha256(cohort_payload).hexdigest(),
        "n_methods": len(results),
        "n_attempted": sum(result.attempted for result in results),
        "n_failed": sum(result.attempted and not result.successful for result in results),
        "n_n_a": sum(not result.attempted for result in results),
    }


def metric_summary(
    results: Iterable[BenchmarkResult], metric: str
) -> dict[str, float | int]:
    """Summarize a metric while keeping every attempted failure in the denominator."""

    rows = list(results)
    attempted = [row for row in rows if row.attempted]
    successful_values = np.asarray(
        [row.metrics[metric] for row in attempted if row.successful and metric in row.metrics],
        dtype=float,
    )
    n_attempted = len(attempted)
    n_finite = len(successful_values)
    n_failed = n_attempted - n_finite
    return {
        "n_records": len(rows),
        "n_attempted": n_attempted,
        "n_finite": n_finite,
        "n_failed": n_failed,
        "failure_rate": (n_failed / n_attempted if n_attempted else math.nan),
        "mean": (float(np.mean(successful_values)) if n_finite else math.nan),
        "sd": (float(np.std(successful_values, ddof=1)) if n_finite > 1 else math.nan),
    }


def write_results_jsonl(path: str | Path, results: Iterable[BenchmarkResult]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

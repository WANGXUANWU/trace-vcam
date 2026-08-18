"""Run the subject-level MACS/CD4 application protocol.

The response is the raw CD4 count.  Time (years relative to seroconversion),
centered age, and time-varying CES-D are linearly mapped to [0, 1] using fixed
full-study support bounds.  The mapping is a declared coordinate convention,
not an outcome-dependent tuning step.  Every outer split is by person ID.

Formal mode runs five repeats of five folds for prediction evaluation and four
pre-registered sensitivities: removal of subjects containing an outer-fence
CD4 observation, 1%/99% response winsorization, and basis dimensions 5 and 8.
A separate TRACE-VCAM fit on the complete primary data set supplies the sole
set of jointly identified factors and component surfaces used in figures.  No
fold-level factor averaging or confidence interval is constructed.
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
from benchmarks.data import (  # noqa: E402
    SubjectDataset,
    SubjectSplit,
    make_repeated_subject_folds,
)
from benchmarks.methods import (  # noqa: E402
    FIXED_METHOD_LABELS,
    METHOD_SPECS,
    Applicability,
    MethodLabel,
    Protocol,
    applicability_for,
)
from scripts.run_strict_benchmark import _common_identify_curves  # noqa: E402
from src.trace_tuning_protocol import load_trace_tuning_lock  # noqa: E402


SCHEMA_VERSION = "vcam-macs-application/1"
PROGRESS_SCHEMA_VERSION = "vcam-macs-progress/1"
RUN_FINGERPRINT_SCHEMA_VERSION = "vcam-macs-run-fingerprint/1"
COHORT_AUDIT_SCHEMA_VERSION = "vcam-macs-cohort-audit/2"
METADATA_REFINALIZATION_SCHEMA_VERSION = "vcam-macs-metadata-refinalization/1"
DEFAULT_SEED = 20260810
ORDERED_TASK_SCHEDULER_VERSION = "bounded-ordered-prefetch/1"
ORDERED_TASK_PREFETCH_FACTOR = 3
RESULT_FIELDS = (
    "schema_version",
    "mode",
    "variant",
    "basis_dimension",
    "repeat",
    "fold",
    "fold_seed",
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
    "n_subjects",
    "n_rows",
    "n_train_subjects",
    "n_test_subjects",
    "data_hash",
    "train_subject_hash",
    "test_subject_hash",
    "tuning_json",
    "realized_tuning_json",
    "runtime_seconds",
    "peak_python_memory_mb",
    "test_mse",
    "subject_balanced_test_mse",
    "test_mae",
    "fit_metadata_json",
)
PREDICTION_FIELDS = (
    "schema_version",
    "variant",
    "basis_dimension",
    "repeat",
    "fold",
    "method",
    "row_id",
    "subject_id",
    "observed_cd4",
    "prediction",
)
OUTPUT_STREAM_KEYS = ("results", "predictions", "curves")


def _method_value(name: str, fallback: str) -> str:
    value = getattr(MethodLabel, name, None)
    return fallback if value is None else str(value.value)


TRACE = _method_value("TRACE_VCAM", "TRACE-VCAM")
ZW = _method_value("ZW2015", "ZW2015")
ZZW = _method_value("ZZW2020", "ZZW2020")
HHY = _method_value("HHY2021_HUBER", "HHY2021-Huber")
ZSY = _method_value("ZSY2026_AUTHOR_CODE", "ZSY2026-author-code")
ZY = _method_value("ZY2025", "ZY2025-paper-implementation")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_bytes(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    include_header: bool,
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    if include_header:
        writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably write a small artifact before atomically publishing it."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _append_durable(path: Path, payload: bytes) -> int:
    """Append a validated transaction chunk and return its durable offset."""

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


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _source_hashes() -> dict[str, str]:
    """Hash every executable input that can change a MACS resume result."""

    paths = [
        Path(__file__),
        ROOT / "scripts" / "run_strict_benchmark.py",
        ROOT / "scripts" / "analyze_strict_results.py",
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


def _minmax(values: np.ndarray, bounds: tuple[float, float] | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    lower, upper = (
        (float(np.min(values)), float(np.max(values))) if bounds is None else bounds
    )
    if not lower < upper:
        raise ValueError("a continuous domain must have positive range")
    transformed = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return transformed, (lower, upper)


@dataclass(frozen=True)
class RawMACS:
    cd4: np.ndarray
    time: np.ndarray
    age: np.ndarray
    cesd: np.ndarray
    person: np.ndarray
    row_id: np.ndarray


@dataclass(frozen=True)
class MACSCohort:
    """One complete method-by-fold transaction on a common subject split."""

    index: int
    variant: str
    basis_dimension: int
    split: SubjectSplit
    dataset: SubjectDataset


def read_macs_csv(path: Path) -> RawMACS:
    """Read only the variables registered in the Hu et al. analysis."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    required = {"cd4", "time", "age", "cesd", "person"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"MACS CSV must contain {sorted(required)}")
    arrays = {
        name: np.asarray([float(record[name]) for record in records], dtype=float)
        for name in ("cd4", "time", "age", "cesd")
    }
    person = np.asarray([str(record["person"]) for record in records], dtype=str)
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("MACS analysis variables must be finite")
    return RawMACS(
        cd4=arrays["cd4"],
        time=arrays["time"],
        age=arrays["age"],
        cesd=arrays["cesd"],
        person=person,
        row_id=np.asarray([f"macs-row-{index}" for index in range(len(records))], dtype=str),
    )


def _outer_fence_subjects(raw: RawMACS) -> tuple[str, ...]:
    q1, q3 = np.quantile(raw.cd4, [0.25, 0.75])
    iqr = q3 - q1
    outside = (raw.cd4 < q1 - 3.0 * iqr) | (raw.cd4 > q3 + 3.0 * iqr)
    return tuple(sorted(set(raw.person[outside].tolist())))


def prepare_macs_variant(
    raw: RawMACS,
    *,
    variant: str,
    global_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> SubjectDataset:
    keep = np.ones(len(raw.cd4), dtype=bool)
    response = raw.cd4.copy()
    metadata: dict[str, object] = {
        "response": "raw CD4 cell count",
        "time": "years relative to seroconversion",
        "age": "age centered around 30 as supplied by catdata::aids",
        "cesd": "time-varying CES-D score as supplied by catdata::aids",
        "variant": variant,
    }
    if variant == "delete_outer_fence_subjects":
        removed = _outer_fence_subjects(raw)
        keep = ~np.isin(raw.person, np.asarray(removed, dtype=str))
        metadata["removed_subjects"] = list(removed)
        metadata["removal_rule"] = (
            "remove an entire subject if any raw CD4 observation lies beyond a 3-IQR Tukey outer fence"
        )
    elif variant == "winsorize_response_1_99":
        lower, upper = np.quantile(response, [0.01, 0.99])
        response = np.clip(response, lower, upper)
        metadata["winsor_bounds"] = [float(lower), float(upper)]
    elif variant not in {"primary", "basis_5", "basis_8"}:
        raise ValueError(f"unknown MACS variant: {variant}")

    bounds = dict(global_bounds or {})
    time_scaled, time_bounds = _minmax(
        raw.time[keep], bounds.get("time")
    )
    age_scaled, age_bounds = _minmax(raw.age[keep], bounds.get("age"))
    cesd_scaled, cesd_bounds = _minmax(raw.cesd[keep], bounds.get("cesd"))
    metadata["coordinate_bounds"] = {
        "time": list(time_bounds),
        "age": list(age_bounds),
        "cesd": list(cesd_bounds),
    }
    metadata["time_domain"] = [0.0, 1.0]
    metadata["covariate_domains"] = [[0.0, 1.0], [0.0, 1.0]]
    metadata["time_invariant_covariates"] = False
    return SubjectDataset(
        time=time_scaled,
        covariates=np.column_stack([age_scaled, cesd_scaled]),
        response=response[keep],
        subject_id=raw.person[keep],
        row_id=raw.row_id[keep],
        noise_free_target=None,
        covariate_names=("centered_age_scaled", "cesd_scaled"),
        metadata=metadata,
    )


def adapter_registry() -> dict[str, object]:
    adapters = [
        TraceVCAMAdapter(),
        ZW2015Adapter(),
        ZZW2020Adapter(),
        HHY2021Adapter(),
        ZSY2026AuthorCodeAdapter(),
        ZY2025Adapter(),
    ]
    registry = {str(adapter.label): adapter for adapter in adapters}
    if set(registry) != set(FIXED_METHOD_LABELS):
        raise RuntimeError(
            f"adapter labels {sorted(registry)} do not equal registered labels {sorted(FIXED_METHOD_LABELS)}"
        )
    return registry


def _load_admissions(path: Path | None) -> tuple[dict[str, str], dict[str, object]]:
    if path is None:
        return {}, {"status": "no_strict_metadata_supplied"}
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    gates = metadata.get("admission_gates", {})
    admissions = {
        str(method): ("admitted" if bool(decision.get("passed")) else str(decision.get("status", "not_admitted")))
        for method, decision in gates.items()
    }
    return admissions, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "schema_version": metadata.get("schema_version"),
        "mode": metadata.get("mode"),
        "formal_protocol_complete": metadata.get(
            "formal_protocol_complete",
            bool(
                metadata.get("mode") == "formal"
                and isinstance(metadata.get("cohort_audit"), Mapping)
                and metadata.get("cohort_audit", {}).get("passed") is True
            ),
        ),
        "formal_claims_eligible": metadata.get("formal_claims_eligible", False),
    }


def _macs_applicability(method: str) -> tuple[str, str]:
    try:
        decision = applicability_for(method, Protocol.MACS_CD4)
        return str(decision.status.value), str(decision.reason)
    except (AttributeError, ValueError, KeyError):
        return "unregistered", "No pre-registered MACS applicability decision exists."


def _tuning(method: str, basis_dimension: int, *, quick: bool) -> dict[str, object]:
    common = {
        "time_domain": [0.0, 1.0],
        "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
        "application": "MACS-CD4",
    }
    if method == TRACE:
        trace_lock = load_trace_tuning_lock()
        tuning = {
            **common,
            "q_time": basis_dimension,
            "q_covariate": basis_dimension,
            "delta_rule": "mad",
            "huber_multiplier": 1.345,
            "lambda_ratio": float(trace_lock["lambda_ratio"]),
            "roughness": float(trace_lock["roughness"]),
            "tuning_mode": str(trace_lock["tuning_mode"]),
            "calibration_path": "protocol/trace_tuning_v1.json",
            "calibration_content_sha256": str(trace_lock["content_sha256"]),
            "calibration_file_sha256": str(trace_lock["file_sha256"]),
            "max_iter": 300 if quick else 2000,
            "tolerance": 1e-4 if quick else 1e-7,
        }
        if quick:
            return tuning
        # On the application every published competitor selects its own tuning
        # parameters from the analysed data.  The proposed estimator is given
        # exactly the same freedom, on training subjects only: a subject-level
        # five-fold cross-validation inside each training fold selects the
        # penalty level, roughness weight, Huber multiplier, and basis size.
        # The simulation-locked pair remains available as a sensitivity fit.
        tuning.update(
            {
                "tuning_mode": "application_subject_cv",
                "selection": "subject_cv",
                "cv_folds": 3,
                "cv_basis_grid": [basis_dimension - 1, basis_dimension],
                "cv_lambda_ratio_grid": [0.2, 0.6, 0.9],
                "cv_roughness_grid": [0.5],
                "cv_huber_multiplier_grid": [1.345, 3.0, 10.0],
                "locked_lambda_ratio": float(trace_lock["lambda_ratio"]),
                "locked_roughness": float(trace_lock["roughness"]),
            }
        )
        return tuning
    if method == ZZW:
        return {
            **common,
            "tuning_mode": (
                "paper_design_counts" if quick else "paper_cv_registered_vectors"
            ),
            "spline_order": 4,
            # The quick fit locks the source paper's wage-application vector.
            # Formal fits select over a registered grid containing that result
            # and the source simulation vectors; a common count is not imposed.
            "time_interior_knots": [2, 1, 2],
            "covariate_interior_knots": [3, 1],
            "knot_candidate_vectors": [
                {"time": [2, 1, 2], "additive": [3, 1]},
                {"time": [4, 1, 2], "additive": [3, 2]},
                {"time": [4, 2, 2], "additive": [3, 2]},
                {"time": [3, 2, 2], "additive": [3, 2]},
            ],
            "knot_grid_provenance": (
                "registered paper-aligned grid containing the ZZW2020 "
                "wage-application five-fold-CV result and simulation vectors"
            ),
            "cv_folds": 5,
            "epsilon_inner": 1e-2,
            "epsilon_outer": 1e-3,
            "max_inner": 20 if quick else 200,
            "max_outer": 20 if quick else 200,
            "cv_max_inner": 10 if quick else 50,
            "cv_max_outer": 10 if quick else 50,
        }
    if method == HHY:
        return {
            **common,
            "tuning_mode": "paper_locked" if quick else "paper_bic",
            "spline_order": 4,
            "delta": 1.345,
            "pilot_time_interior_knots": 2,
            "pilot_covariate_interior_knots": [2, 2],
            "final_time_interior_knots": 4,
            "final_additive_interior_knots": [3, 3],
            "irls_tolerance": 1e-8,
            "irls_max_iter": 30 if quick else 300,
            "irls_objective_relative_tolerance": 1e-9,
            "irls_objective_stable_steps": 3,
            "bic_knot_candidates": [1, 2, 3, 4],
            "knot_grid_provenance": (
                "includes the HHY2021 MACS BIC result: pilot K_C,K_A=(2,2), "
                "final K_C,K_A=(4,3)"
            ),
        }
    if method == ZY:
        tuning = {
            **common,
            "tuning_mode": "paper_locked" if quick else "paper_cv",
            "spline_order": 4,
            "time_interior_knots": [5, 1, 3],
            "additive_interior_knots": [2, 5],
            "knot_tuning_mode": (
                "published Zhao-Yang application five-fold-CV vector transported "
                "to MACS; penalties reselected by the paper's ten-fold rule"
            ),
            "inner_mrs_tolerance": 1e-4,
            "outer_mrs_tolerance": 1e-4,
            "max_inner": 10 if quick else 100,
            "max_outer": 10 if quick else 100,
            "lasso_tolerance": 1e-5 if quick else 1e-8,
            "lasso_max_iter": 500 if quick else 5000,
            "cv_solver": "fista_warm_path",
            "cv_folds": 10,
            "cv_penalty_count": 5 if quick else 10,
            "cv_tolerance": 1e-4 if quick else 1e-5,
            "cv_lasso_max_iter": 300 if quick else 1000,
            "timeout_seconds": 60 if quick else 180,
        }
        if quick:
            tuning.update(
                lambda_initial_additive=0.01,
                lambda_initial_coefficient=0.01,
                lambda_additive=0.01,
                lambda_coefficient=0.01,
                lambda_baseline=0.0,
            )
        return tuning
    return {
        **common,
        "tuning_mode": "original_method_application",
        "workspace_root": str(ROOT),
        "strict_no_silent_patch": True,
        "basis_dimension": basis_dimension,
        "timeout_seconds": 60 if quick else 1800,
    }


def _subject_balanced_mse(subject: np.ndarray, residual: np.ndarray) -> float:
    values = [float(np.mean(residual[subject == item] ** 2)) for item in np.unique(subject)]
    return float(np.mean(values))


def _empty_result(
    *,
    mode: str,
    variant: str,
    basis_dimension: int,
    split: object,
    method: str,
    dataset: SubjectDataset,
    train: SubjectDataset,
    test: SubjectDataset,
    applicability: str,
    reason: str,
    admission: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "variant": variant,
        "basis_dimension": basis_dimension,
        "repeat": split.repeat,
        "fold": split.fold,
        "fold_seed": split.seed,
        "method": method,
        "method_display_name": (
            METHOD_SPECS[method].display_name
            if method in METHOD_SPECS
            else method
        ),
        "method_version": METHOD_SPECS.get(method).version if method in METHOD_SPECS else "unregistered",
        "applicability": applicability,
        "applicability_reason": reason,
        "admission_status": admission,
        "attempt_status": "not_attempted",
        "converged": False,
        "failure_code": "",
        "failure_message": "",
        "n_subjects": dataset.n_subjects,
        "n_rows": dataset.n_rows,
        "n_train_subjects": train.n_subjects,
        "n_test_subjects": test.n_subjects,
        "data_hash": dataset.data_hash,
        "train_subject_hash": split.train_hash,
        "test_subject_hash": split.test_hash,
        "tuning_json": "{}",
        "realized_tuning_json": "{}",
        "runtime_seconds": float("nan"),
        "peak_python_memory_mb": float("nan"),
        "test_mse": float("nan"),
        "subject_balanced_test_mse": float("nan"),
        "test_mae": float("nan"),
        "fit_metadata_json": "{}",
    }


def _run_fold_method(
    adapter: object,
    *,
    mode: str,
    quick: bool,
    variant: str,
    basis_dimension: int,
    split: object,
    dataset: SubjectDataset,
    train: SubjectDataset,
    test: SubjectDataset,
    applicability: str,
    reason: str,
    admission: str,
    preflight_report: object | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
    method = str(adapter.label)
    row = _empty_result(
        mode=mode,
        variant=variant,
        basis_dimension=basis_dimension,
        split=split,
        method=method,
        dataset=dataset,
        train=train,
        test=test,
        applicability=applicability,
        reason=reason,
        admission=admission,
    )
    if applicability == Applicability.N_A_BY_DESIGN.value:
        row["attempt_status"] = "N/A by design"
        row["failure_code"] = "not_applicable"
        return row, [], None
    if applicability != Applicability.APPLICABLE.value:
        row["attempt_status"] = "not_evaluated"
        row["failure_code"] = "applicability_not_registered"
        return row, [], None
    if method != TRACE and admission != "admitted":
        row["attempt_status"] = "blocked_reproduction_gate"
        row["failure_code"] = "reproduction_not_verified"
        return row, [], None
    preflight = adapter.preflight() if preflight_report is None else preflight_report
    row["method_version"] = str(preflight.version)
    if not preflight.ready:
        row["attempt_status"] = "failed"
        row["failure_code"] = str(preflight.code)
        row["failure_message"] = str(preflight.message)[:2000]
        return row, [], None
    tuning = _tuning(method, basis_dimension, quick=quick)
    row["tuning_json"] = _canonical_json(tuning)
    started = time.perf_counter()
    tracemalloc.start()
    try:
        artifact = adapter.fit(train, seed=int(split.seed), tuning=tuning)
        row["realized_tuning_json"] = _canonical_json(
            dict(getattr(artifact, "tuning", tuning))
        )
        prediction = np.asarray(adapter.predict(artifact, test), dtype=float)
        if prediction.shape != (test.n_rows,) or not np.all(np.isfinite(prediction)):
            raise FloatingPointError("adapter returned non-finite or wrong-length prediction")
        residual = prediction - test.response
        row["runtime_seconds"] = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        row["peak_python_memory_mb"] = peak / (1024**2)
        row["converged"] = bool(artifact.converged)
        row["attempt_status"] = "success" if artifact.converged else "failed"
        if not artifact.converged:
            row["failure_code"] = "nonconvergence"
        row["test_mse"] = float(np.mean(residual**2))
        row["subject_balanced_test_mse"] = _subject_balanced_mse(test.subject_id, residual)
        row["test_mae"] = float(np.mean(np.abs(residual)))
        row["fit_metadata_json"] = _canonical_json(dict(artifact.metadata))
        predictions = [
            {
                "schema_version": SCHEMA_VERSION,
                "variant": variant,
                "basis_dimension": basis_dimension,
                "repeat": split.repeat,
                "fold": split.fold,
                "method": method,
                "row_id": test.row_id[index],
                "subject_id": test.subject_id[index],
                "observed_cd4": float(test.response[index]),
                "prediction": float(prediction[index]),
            }
            for index in range(test.n_rows)
        ]
        # Outer folds exist only to evaluate held-out prediction.  Factor
        # curves are fitted once on the complete primary data set below;
        # separately averaging beta and phi across folds is not invariant to
        # the multiplicative identification of a VCAM component.
        return row, predictions, None
    except Exception as error:
        row["runtime_seconds"] = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        row["peak_python_memory_mb"] = peak / (1024**2)
        row["attempt_status"] = "failed"
        row["failure_code"] = str(getattr(error, "code", type(error).__name__))
        row["failure_message"] = f"{type(error).__name__}: {error}"[:2000]
        row["fit_metadata_json"] = _canonical_json(
            {"traceback": traceback.format_exc(limit=8)}
        )
        return row, [], None
    finally:
        tracemalloc.stop()


def _fit_registered_full_data_trace_curves(
    adapter: object,
    *,
    mode: str,
    quick: bool,
    variant: str,
    basis_dimension: int,
    dataset: SubjectDataset,
    seed: int,
    preflight_report: object | None = None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Fit the sole registered curve model on all primary MACS subjects.

    Cross-validation estimates prediction error only.  This independent fit
    supplies one jointly identified set of TRACE-VCAM factors and component
    surfaces for the application figures; no fold-level factor is serialized
    or averaged.
    """

    method = str(adapter.label)
    if method != TRACE:
        raise ValueError("the registered MACS curve fit must use TRACE-VCAM")
    tuning = _tuning(method, basis_dimension, quick=quick)
    tuning_json = _canonical_json(tuning)
    audit: dict[str, object] = {
        "fit_id": "macs-primary-full-data-trace-v1",
        "fit_scope": "full primary data; all subjects and observations",
        "purpose": "identifiable factor and component-surface figures only",
        "prediction_evaluation": "excluded; outer subject CV is reported separately",
        "curve_aggregation": "none",
        "fold_curves_serialized": False,
        "variant": variant,
        "basis_dimension": basis_dimension,
        "method": method,
        "seed": int(seed),
        "n_subjects": dataset.n_subjects,
        "n_rows": dataset.n_rows,
        "data_hash": dataset.data_hash,
        "tuning": tuning,
        "tuning_sha256": _object_sha256(tuning),
        "attempt_status": "not_attempted",
        "converged": False,
        "failure_code": "",
        "failure_message": "",
    }
    preflight = adapter.preflight() if preflight_report is None else preflight_report
    audit["method_version"] = str(preflight.version)
    audit["preflight"] = {
        "ready": bool(preflight.ready),
        "code": str(preflight.code),
        "message": str(preflight.message),
        "environment": dict(preflight.environment),
    }
    if not preflight.ready:
        audit["attempt_status"] = "failed"
        audit["failure_code"] = str(preflight.code)
        audit["failure_message"] = str(preflight.message)[:2000]
        return None, audit

    started = time.perf_counter()
    tracemalloc.start()
    try:
        artifact = adapter.fit(dataset, seed=int(seed), tuning=tuning)
        audit["runtime_seconds"] = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        audit["peak_python_memory_mb"] = peak / (1024**2)
        audit["converged"] = bool(artifact.converged)
        audit["selected_blocks"] = [int(item) for item in artifact.selected_blocks]
        audit["fit_metadata"] = dict(artifact.metadata)
        if not artifact.converged:
            audit["attempt_status"] = "failed"
            audit["failure_code"] = "nonconvergence"
            audit["failure_message"] = (
                "The registered full-data TRACE-VCAM fit did not meet its stopping rule."
            )
            return None, audit

        raw_curves = tuple(adapter.factor_curves(artifact))
        identified_curves, identification_audit = _common_identify_curves(
            raw_curves, n_covariates=dataset.covariates.shape[1]
        )
        serialized_curves = [dict(curve) for curve in identified_curves]
        audit["attempt_status"] = "success"
        audit["common_factor_identification"] = identification_audit
        audit["raw_curves_sha256"] = _object_sha256(raw_curves)
        audit["identified_curves_sha256"] = _object_sha256(serialized_curves)
        curve_row: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "basis_dimension": basis_dimension,
            # These zero indices preserve compatibility with the current
            # plotting reader; they are explicitly not CV fold identifiers.
            "repeat": 0,
            "fold": 0,
            "repeat_fold_semantics": "compatibility selector; not a CV fit",
            "method": method,
            "fit_id": audit["fit_id"],
            "fit_scope": "registered_full_data",
            "seed": int(seed),
            "n_subjects": dataset.n_subjects,
            "n_rows": dataset.n_rows,
            "data_hash": dataset.data_hash,
            "method_version": str(preflight.version),
            "tuning_sha256": audit["tuning_sha256"],
            "identified_curves_sha256": audit["identified_curves_sha256"],
            "curves": serialized_curves,
        }
        audit["curve_row_sha256"] = _object_sha256(curve_row)
        return curve_row, audit
    except Exception as error:
        audit["runtime_seconds"] = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        audit["peak_python_memory_mb"] = peak / (1024**2)
        audit["attempt_status"] = "failed"
        audit["failure_code"] = str(getattr(error, "code", type(error).__name__))
        audit["failure_message"] = f"{type(error).__name__}: {error}"[:2000]
        audit["traceback"] = traceback.format_exc(limit=8)
        return None, audit
    finally:
        tracemalloc.stop()


def _macs_cohort_task(
    task: tuple[
        MACSCohort,
        str,
        bool,
        Mapping[str, str],
        Mapping[str, object],
    ]
) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    """Fit every registered method on one common subject-level fold.

    The fold, rather than an individual method, is the atomic resume unit.
    It makes the common train/test split and all six result rows durable in
    one journaled transaction while retaining at most one data set per worker.
    """

    cohort, mode, quick, admissions, preflight_reports = task
    adapters = adapter_registry()
    train = cohort.dataset.subset_subjects(cohort.split.train_subjects)
    test = cohort.dataset.subset_subjects(cohort.split.test_subjects)
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for method in FIXED_METHOD_LABELS:
        applicability, reason = _macs_applicability(method)
        admission = "not_required" if method == TRACE else admissions.get(
            method, "pending_no_reproduction"
        )
        row, method_predictions, curves = _run_fold_method(
            adapters[method],
            mode=mode,
            quick=quick,
            variant=cohort.variant,
            basis_dimension=cohort.basis_dimension,
            split=cohort.split,
            dataset=cohort.dataset,
            train=train,
            test=test,
            applicability=applicability,
            reason=reason,
            admission=admission,
            preflight_report=preflight_reports.get(method),
        )
        if curves is not None:
            raise RuntimeError("MACS outer folds must not serialize factor curves")
        rows.append(row)
        predictions.extend(method_predictions)
    method_index = {method: index for index, method in enumerate(FIXED_METHOD_LABELS)}
    rows.sort(key=lambda row: method_index[str(row["method"])])
    predictions.sort(
        key=lambda row: (method_index[str(row["method"])], str(row["row_id"]))
    )
    return cohort.index, rows, predictions


def _macs_prefetch_limit(jobs: int) -> int:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    return 1 if jobs == 1 else ORDERED_TASK_PREFETCH_FACTOR * jobs


def _ordered_macs_cohort_results(
    tasks: Sequence[
        tuple[
            MACSCohort,
            str,
            bool,
            Mapping[str, str],
            Mapping[str, object],
        ]
    ],
    *,
    jobs: int,
    worker: Callable[
        [
            tuple[
                MACSCohort,
                str,
                bool,
                Mapping[str, str],
                Mapping[str, object],
            ]
        ],
        tuple[int, list[dict[str, object]], list[dict[str, object]]],
    ] = _macs_cohort_task,
) -> Iterable[tuple[int, list[dict[str, object]], list[dict[str, object]]]]:
    """Yield deterministic fold transactions with bounded eager prefetch."""

    if jobs == 1:
        for task in tasks:
            yield worker(task)
        return
    iterator = iter(tasks)
    limit = _macs_prefetch_limit(jobs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        pending: deque[
            tuple[
                tuple[
                    MACSCohort,
                    str,
                    bool,
                    Mapping[str, str],
                    Mapping[str, object],
                ],
                object,
            ]
        ] = deque()

        def submit_one() -> bool:
            try:
                task = next(iterator)
            except StopIteration:
                return False
            pending.append((task, executor.submit(worker, task)))
            return True

        for _ in range(limit):
            if not submit_one():
                break
        while pending:
            _, future = pending.popleft()
            yield future.result()  # type: ignore[union-attr]
            submit_one()


def _cohort_registry(cohorts: Sequence[MACSCohort]) -> list[dict[str, object]]:
    return [
        {
            "index": cohort.index,
            "variant": cohort.variant,
            "basis_dimension": cohort.basis_dimension,
            "data_hash": cohort.dataset.data_hash,
            "n_rows": cohort.dataset.n_rows,
            "n_subjects": cohort.dataset.n_subjects,
            "repeat": cohort.split.repeat,
            "fold": cohort.split.fold,
            "fold_seed": cohort.split.seed,
            "train_subject_hash": cohort.split.train_hash,
            "test_subject_hash": cohort.split.test_hash,
        }
        for cohort in cohorts
    ]


def _cohort_key_from_record(
    record: Mapping[str, object], *, context: str
) -> tuple[str, int, int]:
    """Return a canonical MACS cohort key from CSV or registry metadata.

    ``csv.DictReader`` deliberately returns text for every field.  In particular,
    ``repeat`` and ``fold`` must be parsed before they can be compared with the
    integer-valued registered folds.  Keeping that conversion in one function
    prevents a final metadata audit from silently filtering every CSV cohort out.
    """

    variant = str(record.get("variant", ""))
    if not variant:
        raise RuntimeError(f"{context} has an empty variant")
    return (
        variant,
        _as_int(record.get("repeat"), field=f"{context}.repeat"),
        _as_int(record.get("fold"), field=f"{context}.fold"),
    )


def audit_macs_result_cohorts(
    result_rows: Sequence[Mapping[str, object]],
    *,
    cohort_registry: Sequence[Mapping[str, object]],
    method_order: Sequence[str] = FIXED_METHOD_LABELS,
) -> dict[str, object]:
    """Audit that every registered subject split is shared by all methods.

    This is intentionally independent of row order so it can also validate a
    frozen completed run during a metadata-only re-finalization.  It verifies the
    same common hashes as the original final audit and, additionally, checks each
    group against the registered cohort identity.
    """

    issues: list[str] = []
    expected: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for index, record in enumerate(cohort_registry):
        try:
            key = _cohort_key_from_record(record, context=f"cohort registry {index}")
        except RuntimeError as error:
            issues.append(str(error))
            continue
        if key in expected:
            issues.append(f"duplicate registered MACS cohort: {key[0]}/{key[1]}/{key[2]}")
            continue
        expected[key] = record

    observed: dict[tuple[str, int, int], list[Mapping[str, object]]] = {}
    for index, row in enumerate(result_rows):
        try:
            key = _cohort_key_from_record(row, context=f"result row {index}")
        except RuntimeError as error:
            issues.append(str(error))
            continue
        observed.setdefault(key, []).append(row)

    observed_cohort_count = len(observed)

    expected_methods = {str(method) for method in method_order}
    cohort_fields = ("data_hash", "train_subject_hash", "test_subject_hash")
    for key, registered in expected.items():
        cohort = observed.pop(key, [])
        label = f"{key[0]}/{key[1]}/{key[2]}"
        if len(cohort) != len(method_order):
            issues.append(
                f"{label}: result rows {len(cohort)} != expected {len(method_order)}"
            )
        if {str(row.get("method")) for row in cohort} != expected_methods:
            issues.append(f"{label}: method cohort mismatch")
        for field in cohort_fields:
            values = {str(row.get(field, "")) for row in cohort}
            if len(values) != 1:
                issues.append(f"{label}: non-common {field}")
            elif values and str(registered.get(field, "")) not in values:
                issues.append(f"{label}: {field} differs from the registered split")

    for key in sorted(observed):
        issues.append(f"unregistered MACS result cohort: {key[0]}/{key[1]}/{key[2]}")

    return {
        "schema_version": COHORT_AUDIT_SCHEMA_VERSION,
        "passed": not issues,
        "issues": issues,
        "expected_cohorts": len(expected),
        "observed_cohorts": observed_cohort_count,
        "expected_result_rows": len(expected) * len(method_order),
        "observed_result_rows": len(result_rows),
        "method_order": [str(method) for method in method_order],
        "key_normalization": "repeat and fold parsed as integers from CSV before cohort comparison",
    }


def _macs_run_contract(
    *,
    mode: str,
    seed: int,
    jobs: int,
    data_path: Path,
    data_sha256: str,
    variants: Sequence[tuple[str, int]],
    n_splits: int,
    n_repeats: int,
    cohorts: Sequence[MACSCohort],
    variant_metadata: Mapping[str, object],
    admissions: Mapping[str, str],
    admission_source: Mapping[str, object],
    trace_tuning_lock: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Record every immutable input needed to skip committed MACS folds."""

    return {
        "schema_version": RUN_FINGERPRINT_SCHEMA_VERSION,
        "mode": mode,
        "seed": int(seed),
        "execution": {
            "jobs": int(jobs),
            "ordered_task_scheduler_version": ORDERED_TASK_SCHEDULER_VERSION,
            "ordered_task_prefetch_factor": ORDERED_TASK_PREFETCH_FACTOR,
            "max_outstanding_futures": _macs_prefetch_limit(jobs),
        },
        "data_source": {
            "path": str(data_path.resolve()),
            "sha256": data_sha256,
        },
        "variants": list(variants),
        "fold_protocol": {"n_splits": int(n_splits), "n_repeats": int(n_repeats)},
        "cohort_registry": _cohort_registry(cohorts),
        "variant_metadata": dict(variant_metadata),
        "method_order": list(FIXED_METHOD_LABELS),
        "admission_status": dict(admissions),
        "admission_source": dict(admission_source),
        "trace_calibration": {
            "content_sha256": str(trace_tuning_lock["content_sha256"]),
            "file_sha256": str(trace_tuning_lock["file_sha256"]),
            "lambda_ratio": float(trace_tuning_lock["lambda_ratio"]),
            "roughness": float(trace_tuning_lock["roughness"]),
            "tuning_mode": str(trace_tuning_lock["tuning_mode"]),
        },
        "source_sha256": _source_hashes(),
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": _package_versions(),
        "preflight": dict(preflight),
    }


def _run_fingerprint(contract: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json(contract).encode("utf-8"))


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
        "curve_completion": {"state": "pending", "audit": None},
        "last_completed": None,
        "status": "running",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _curve_completion(progress: Mapping[str, object]) -> dict[str, object]:
    completion = progress.get("curve_completion")
    if not isinstance(completion, Mapping):
        raise RuntimeError("MACS progress lacks a registered-curve completion record")
    state = str(completion.get("state", ""))
    if state not in {"pending", "completed"}:
        raise RuntimeError("MACS progress has an invalid registered-curve state")
    audit = completion.get("audit")
    if state == "pending" and audit is not None:
        raise RuntimeError("pending MACS curve completion unexpectedly carries an audit")
    if state == "completed" and not isinstance(audit, Mapping):
        raise RuntimeError("completed MACS curve completion lacks its audit")
    return {"state": state, "audit": None if audit is None else dict(audit)}


def _initialize_or_restore_streams(
    paths: Mapping[str, Path],
    *,
    contract: Mapping[str, object],
    fingerprint: str,
    expected_cohorts: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Restore the last atomically committed fold boundary.

    Any bytes beyond journaled offsets arose after the last durable journal
    update and are discarded before a task is skipped.
    """

    progress_path = paths["progress"]
    if not progress_path.exists():
        existing = [
            str(paths[key])
            for key in (*OUTPUT_STREAM_KEYS, "metadata", "metadata_sha256")
            if paths[key].exists()
        ]
        if existing:
            raise RuntimeError(
                "MACS outputs exist without a resumable progress journal; use a new "
                f"output directory or archive them first: {existing}"
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
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"MACS progress journal is unreadable: {error}") from error
        if not isinstance(progress, Mapping):
            raise RuntimeError("MACS progress journal is not a JSON object")
        if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise RuntimeError(
                "MACS output uses a legacy/non-resumable progress schema; archive it "
                "and use a dedicated output directory"
            )
        if str(progress.get("run_fingerprint")) != fingerprint:
            raise RuntimeError(
                "MACS resume fingerprint mismatch (mode/seed/data/registry/methods/"
                "source/calibration/environment/execution changed)"
            )
        if _canonical_json(progress.get("run_contract")) != _canonical_json(contract):
            raise RuntimeError("MACS resume contract differs despite its fingerprint")
        if int(progress.get("expected_cohorts", -1)) != expected_cohorts:
            raise RuntimeError("MACS resume expected-cohort count mismatch")
        committed = int(progress.get("committed_cohorts", -1))
        if not 0 <= committed <= expected_cohorts:
            raise RuntimeError("MACS progress contains an invalid committed-cohort count")
        offsets = progress.get("committed_offsets")
        hashes = progress.get("committed_sha256")
        if not isinstance(offsets, Mapping) or not isinstance(hashes, Mapping):
            raise RuntimeError("MACS progress lacks committed offsets or hashes")
        _curve_completion(progress)
        for key in OUTPUT_STREAM_KEYS:
            offset = int(offsets.get(key, -1))
            _truncate_to_offset(paths[key], offset)
            actual_hash = file_sha256(paths[key])
            if actual_hash != str(hashes.get(key, "")):
                raise RuntimeError(
                    f"committed MACS {key} prefix hash mismatch; refusing to skip folds"
                )
        progress = dict(progress)

    hashers = {key: _new_prefix_hasher(paths[key]) for key in OUTPUT_STREAM_KEYS}
    return dict(progress), hashers


def _append_transaction(
    paths: Mapping[str, Path],
    progress: Mapping[str, object],
    hashers: Mapping[str, object],
    chunks: Mapping[str, bytes],
) -> tuple[dict[str, int], dict[str, str]]:
    offsets = progress.get("committed_offsets")
    if not isinstance(offsets, Mapping):
        raise RuntimeError("MACS progress lacks transaction offsets")
    for key in OUTPUT_STREAM_KEYS:
        if paths[key].stat().st_size != int(offsets.get(key, -1)):
            raise RuntimeError(f"MACS {key} changed after its last committed offset")
    new_offsets: dict[str, int] = {}
    for key in OUTPUT_STREAM_KEYS:
        chunk = chunks[key]
        new_offsets[key] = (
            _append_durable(paths[key], chunk)
            if chunk
            else int(offsets.get(key, -1))
        )
        hashers[key].update(chunk)  # type: ignore[union-attr]
    return new_offsets, {
        key: hashers[key].hexdigest() for key in OUTPUT_STREAM_KEYS  # type: ignore[union-attr]
    }


def _commit_fold(
    paths: Mapping[str, Path],
    progress: dict[str, object],
    hashers: Mapping[str, object],
    *,
    cohort: MACSCohort,
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if int(progress.get("committed_cohorts", -1)) != cohort.index:
        raise RuntimeError("MACS fold commit is out of registered order")
    if _curve_completion(progress)["state"] != "pending":
        raise RuntimeError("cannot append a fold after the MACS curve fit is committed")
    chunks = {
        "results": _csv_bytes(RESULT_FIELDS, rows, include_header=False),
        "predictions": _csv_bytes(PREDICTION_FIELDS, predictions, include_header=False),
        "curves": b"",
    }
    offsets, hashes = _append_transaction(paths, progress, hashers, chunks)
    committed = cohort.index + 1
    expected = int(progress["expected_cohorts"])
    updated = dict(progress)
    updated.update(
        committed_cohorts=committed,
        committed_offsets=offsets,
        committed_sha256=hashes,
        last_completed={
            "variant": cohort.variant,
            "repeat": cohort.split.repeat,
            "fold": cohort.split.fold,
            "result_rows": len(rows),
            "prediction_rows": len(predictions),
            "curve_rows": 0,
            "cohort_chunks_sha256": {
                key: _sha256_bytes(chunks[key]) for key in OUTPUT_STREAM_KEYS
            },
        },
        status="folds_complete" if committed == expected else "running",
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _atomic_write_json(paths["progress"], updated)
    return updated


def _commit_registered_curve(
    paths: Mapping[str, Path],
    progress: dict[str, object],
    hashers: Mapping[str, object],
    *,
    curve_row: Mapping[str, object] | None,
    curve_audit: Mapping[str, object],
) -> dict[str, object]:
    if int(progress.get("committed_cohorts", -1)) != int(progress["expected_cohorts"]):
        raise RuntimeError("cannot commit the MACS curve fit before every fold")
    if _curve_completion(progress)["state"] != "pending":
        raise RuntimeError("MACS curve fit is already committed")
    succeeded = (
        str(curve_audit.get("attempt_status")) == "success"
        and curve_audit.get("converged") is True
    )
    if succeeded != (curve_row is not None):
        raise RuntimeError("MACS curve audit and serialized curve row disagree")
    chunks = {
        "results": b"",
        "predictions": b"",
        "curves": _jsonl_bytes([curve_row]) if curve_row is not None else b"",
    }
    offsets, hashes = _append_transaction(paths, progress, hashers, chunks)
    updated = dict(progress)
    updated.update(
        committed_offsets=offsets,
        committed_sha256=hashes,
        curve_completion={"state": "completed", "audit": dict(curve_audit)},
        last_completed={
            "registered_curve_fit": str(curve_audit.get("fit_id", "")),
            "attempt_status": str(curve_audit.get("attempt_status", "")),
            "curve_rows": 1 if curve_row is not None else 0,
            "curve_chunks_sha256": _sha256_bytes(chunks["curves"]),
        },
        status="curve_committed",
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _atomic_write_json(paths["progress"], updated)
    return updated


def _as_int(value: object, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"MACS field {field} is not an integer: {value!r}") from error


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_cohort_rows(
    rows: Sequence[Mapping[str, object]],
    cohort: MACSCohort,
    *,
    mode: str,
    admissions: Mapping[str, str],
) -> None:
    if len(rows) != len(FIXED_METHOD_LABELS):
        raise RuntimeError("committed MACS fold does not contain every fixed method")
    if tuple(str(row.get("method")) for row in rows) != tuple(FIXED_METHOD_LABELS):
        raise RuntimeError("committed MACS fold is not in fixed method order")
    expected_common = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "variant": cohort.variant,
        "basis_dimension": cohort.basis_dimension,
        "repeat": cohort.split.repeat,
        "fold": cohort.split.fold,
        "fold_seed": cohort.split.seed,
        "n_subjects": cohort.dataset.n_subjects,
        "n_rows": cohort.dataset.n_rows,
        "n_train_subjects": len(cohort.split.train_subjects),
        "n_test_subjects": len(cohort.split.test_subjects),
        "data_hash": cohort.dataset.data_hash,
        "train_subject_hash": cohort.split.train_hash,
        "test_subject_hash": cohort.split.test_hash,
    }
    for row in rows:
        method = str(row.get("method"))
        for field, expected in expected_common.items():
            actual = row.get(field)
            if field in {
                "basis_dimension",
                "repeat",
                "fold",
                "fold_seed",
                "n_subjects",
                "n_rows",
                "n_train_subjects",
                "n_test_subjects",
            }:
                if _as_int(actual, field=field) != int(expected):
                    raise RuntimeError(
                        f"invalid MACS {field} in {cohort.variant}/{cohort.split.repeat}/"
                        f"{cohort.split.fold}/{method}"
                    )
            elif str(actual) != str(expected):
                raise RuntimeError(
                    f"invalid MACS {field} in {cohort.variant}/{cohort.split.repeat}/"
                    f"{cohort.split.fold}/{method}"
                )
        applicability, reason = _macs_applicability(method)
        if str(row.get("applicability")) != applicability or str(
            row.get("applicability_reason")
        ) != reason:
            raise RuntimeError(f"invalid registered applicability for MACS {method}")
        admission = "not_required" if method == TRACE else admissions.get(
            method, "pending_no_reproduction"
        )
        if str(row.get("admission_status")) != admission:
            raise RuntimeError(f"invalid MACS admission status for {method}")
        status = str(row.get("attempt_status"))
        if applicability == Applicability.N_A_BY_DESIGN.value:
            if status != "N/A by design":
                raise RuntimeError(f"N/A-by-design MACS method was unexpectedly attempted: {method}")
        elif applicability != Applicability.APPLICABLE.value:
            if status != "not_evaluated":
                raise RuntimeError(f"unregistered MACS method has an invalid status: {method}")
        elif method != TRACE and admission != "admitted":
            if status != "blocked_reproduction_gate":
                raise RuntimeError(f"blocked MACS method has an invalid status: {method}")
        elif status not in {"success", "failed"}:
            raise RuntimeError(f"attempted MACS method has an invalid status: {method}")
        if status == "success":
            for field in (
                "runtime_seconds",
                "test_mse",
                "subject_balanced_test_mse",
                "test_mae",
            ):
                if not _finite(row.get(field)):
                    raise RuntimeError(f"successful MACS fit lacks finite {field}: {method}")


def _load_and_validate_committed_results(
    path: Path,
    cohorts: Sequence[MACSCohort],
    committed_cohorts: int,
    *,
    mode: str,
    admissions: Mapping[str, str],
) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(RESULT_FIELDS):
            raise RuntimeError("MACS result header differs from the registered schema")
        rows = [dict(row) for row in reader]
    expected_rows = committed_cohorts * len(FIXED_METHOD_LABELS)
    if len(rows) != expected_rows:
        raise RuntimeError("committed MACS result stream ends inside a fold or has trailing rows")
    for index, cohort in enumerate(cohorts[:committed_cohorts]):
        start = index * len(FIXED_METHOD_LABELS)
        _validate_cohort_rows(
            rows[start : start + len(FIXED_METHOD_LABELS)],
            cohort,
            mode=mode,
            admissions=admissions,
        )
    return rows


def _result_groups(
    rows: Sequence[Mapping[str, object]], cohorts: Sequence[MACSCohort]
) -> dict[int, dict[str, Mapping[str, object]]]:
    groups: dict[int, dict[str, Mapping[str, object]]] = {}
    width = len(FIXED_METHOD_LABELS)
    for index, cohort in enumerate(cohorts):
        start = index * width
        groups[cohort.index] = {
            str(row["method"]): row for row in rows[start : start + width]
        }
    return groups


def _validate_committed_predictions(
    path: Path,
    cohorts: Sequence[MACSCohort],
    committed_cohorts: int,
    results: Sequence[Mapping[str, object]],
) -> None:
    registered = {
        (cohort.variant, cohort.split.repeat, cohort.split.fold): cohort
        for cohort in cohorts[:committed_cohorts]
    }
    groups = _result_groups(results, cohorts[:committed_cohorts])
    method_index = {method: index for index, method in enumerate(FIXED_METHOD_LABELS)}
    previous_order: tuple[int, int, str] | None = None
    seen: set[tuple[int, str]] = set()
    current: tuple[int, str] | None = None
    current_rows: list[dict[str, str]] = []

    def finish_group() -> None:
        nonlocal current, current_rows
        if current is None:
            return
        cohort_index, method = current
        cohort = cohorts[cohort_index]
        test = cohort.dataset.subset_subjects(cohort.split.test_subjects)
        expected = {
            str(test.row_id[index]): (str(test.subject_id[index]), float(test.response[index]))
            for index in range(test.n_rows)
        }
        observed = {str(row["row_id"]): row for row in current_rows}
        if len(observed) != len(current_rows) or set(observed) != set(expected):
            raise RuntimeError(f"MACS prediction rows do not match the held-out fold: {current}")
        for row_id, (subject_id, response) in expected.items():
            row = observed[row_id]
            if str(row["subject_id"]) != subject_id:
                raise RuntimeError(f"MACS prediction subject mismatch: {current}/{row_id}")
            if not _finite(row["prediction"]) or not math.isclose(
                float(row["observed_cd4"]), response, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise RuntimeError(f"invalid MACS prediction payload: {current}/{row_id}")
        seen.add(current)
        current = None
        current_rows = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(PREDICTION_FIELDS):
            raise RuntimeError("MACS prediction header differs from the registered schema")
        for row in reader:
            key = (
                str(row.get("variant")),
                _as_int(row.get("repeat"), field="repeat"),
                _as_int(row.get("fold"), field="fold"),
            )
            cohort = registered.get(key)
            if cohort is None:
                raise RuntimeError(f"MACS prediction belongs to an uncommitted fold: {key}")
            method = str(row.get("method"))
            if method not in method_index or method not in groups[cohort.index]:
                raise RuntimeError(f"unexpected MACS prediction method: {method}")
            result = groups[cohort.index][method]
            if str(result.get("attempt_status")) not in {"success", "failed"}:
                raise RuntimeError(f"unattempted MACS method emitted predictions: {method}")
            if (
                str(row.get("schema_version")) != SCHEMA_VERSION
                or _as_int(row.get("basis_dimension"), field="basis_dimension")
                != cohort.basis_dimension
            ):
                raise RuntimeError("MACS prediction schema or basis mismatch")
            order = (cohort.index, method_index[method], str(row.get("row_id")))
            if previous_order is not None and order <= previous_order:
                raise RuntimeError("MACS prediction stream is duplicated or not canonically ordered")
            previous_order = order
            group = (cohort.index, method)
            if group != current:
                finish_group()
                if group in seen:
                    raise RuntimeError(f"noncontiguous MACS prediction group: {group}")
                current = group
            current_rows.append(dict(row))
    finish_group()
    expected_success = {
        (cohort.index, method)
        for cohort in cohorts[:committed_cohorts]
        for method, row in groups[cohort.index].items()
        if str(row.get("attempt_status")) == "success"
    }
    if not expected_success.issubset(seen):
        missing = sorted(expected_success - seen)
        raise RuntimeError(f"successful MACS fits lack predictions: {missing[:5]}")


def _validate_fold_prediction_payload(
    predictions: Sequence[Mapping[str, object]],
    cohort: MACSCohort,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate a worker payload before its fold transaction becomes durable."""

    results = {str(row["method"]): row for row in rows}
    test = cohort.dataset.subset_subjects(cohort.split.test_subjects)
    expected = {
        str(test.row_id[index]): (str(test.subject_id[index]), float(test.response[index]))
        for index in range(test.n_rows)
    }
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in predictions:
        method = str(row.get("method"))
        if method not in results or str(results[method].get("attempt_status")) not in {
            "success",
            "failed",
        }:
            raise RuntimeError(f"unattempted MACS method emitted predictions: {method}")
        if (
            str(row.get("schema_version")) != SCHEMA_VERSION
            or str(row.get("variant")) != cohort.variant
            or _as_int(row.get("basis_dimension"), field="basis_dimension")
            != cohort.basis_dimension
            or _as_int(row.get("repeat"), field="repeat") != cohort.split.repeat
            or _as_int(row.get("fold"), field="fold") != cohort.split.fold
        ):
            raise RuntimeError("MACS prediction payload has inconsistent fold metadata")
        grouped.setdefault(method, []).append(row)
    for method, payload in grouped.items():
        observed = {str(row.get("row_id")): row for row in payload}
        if len(observed) != len(payload) or set(observed) != set(expected):
            raise RuntimeError(f"MACS prediction payload is incomplete for {method}")
        for row_id, (subject_id, response) in expected.items():
            row = observed[row_id]
            if str(row.get("subject_id")) != subject_id or not _finite(
                row.get("prediction")
            ):
                raise RuntimeError(f"invalid MACS prediction payload for {method}/{row_id}")
            try:
                matches_response = math.isclose(
                    float(row.get("observed_cd4")),
                    response,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches_response = False
            if not matches_response:
                raise RuntimeError(f"MACS observed response mismatch for {method}/{row_id}")
    missing = {
        str(row["method"])
        for row in rows
        if str(row.get("attempt_status")) == "success"
    } - set(grouped)
    if missing:
        raise RuntimeError(f"successful MACS fits lack prediction payloads: {sorted(missing)}")


def _validate_committed_curves(path: Path, progress: Mapping[str, object]) -> None:
    completion = _curve_completion(progress)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as error:
        raise RuntimeError(f"MACS factor-curve stream is unreadable: {error}") from error
    if completion["state"] == "pending":
        if rows:
            raise RuntimeError("MACS fold stream unexpectedly contains factor curves")
        return
    audit = completion["audit"]
    assert isinstance(audit, Mapping)
    succeeded = (
        str(audit.get("attempt_status")) == "success"
        and audit.get("converged") is True
    )
    if not succeeded:
        if rows:
            raise RuntimeError("failed registered MACS curve fit unexpectedly emitted curves")
        return
    if len(rows) != 1:
        raise RuntimeError("registered MACS curve stream must contain exactly one row")
    row = rows[0]
    if (
        str(row.get("schema_version")) != SCHEMA_VERSION
        or row.get("fit_scope") != "registered_full_data"
        or row.get("fit_id") != audit.get("fit_id")
    ):
        raise RuntimeError("registered MACS curve row does not match its audit")
    curves = row.get("curves")
    if not isinstance(curves, list):
        raise RuntimeError("registered MACS curve payload is missing")
    identified_hash = _object_sha256(curves)
    if (
        str(row.get("identified_curves_sha256")) != identified_hash
        or str(audit.get("identified_curves_sha256")) != identified_hash
        or str(audit.get("curve_row_sha256")) != _object_sha256(row)
    ):
        raise RuntimeError("registered MACS curve hashes do not match the audit")


def _existing_final_is_valid(paths: Mapping[str, Path], fingerprint: str) -> bool:
    if not paths["metadata"].exists() or not paths["metadata_sha256"].exists():
        return False
    try:
        declared = paths["metadata_sha256"].read_text(encoding="ascii").split()[0]
        if declared != file_sha256(paths["metadata"]):
            return False
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if str(metadata.get("run_fingerprint")) != fingerprint:
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


def _frozen_macs_stream_audit(
    paths: Mapping[str, Path],
    progress: Mapping[str, object],
    metadata: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Verify immutable result streams before a metadata-only repair.

    A metadata re-finalization is deliberately narrower than a normal resume:
    it may never truncate, append, or recompute a held-out fold.  Both the
    journal and the original final metadata must therefore agree with the bytes
    currently on disk.
    """

    offsets = progress.get("committed_offsets")
    hashes = progress.get("committed_sha256")
    files = metadata.get("files")
    if not isinstance(offsets, Mapping) or not isinstance(hashes, Mapping):
        raise RuntimeError("MACS progress lacks immutable stream offsets or hashes")
    if not isinstance(files, Mapping):
        raise RuntimeError("MACS metadata lacks final stream hashes")
    audit: dict[str, dict[str, object]] = {}
    for key in OUTPUT_STREAM_KEYS:
        path = paths[key]
        if not path.exists():
            raise RuntimeError(f"frozen MACS stream is missing: {path.name}")
        actual_bytes = int(path.stat().st_size)
        actual_hash = file_sha256(path)
        if actual_bytes != int(offsets.get(key, -1)) or actual_hash != str(
            hashes.get(key, "")
        ):
            raise RuntimeError(
                f"frozen MACS {key} stream differs from its committed journal hash"
            )
        declared = files.get(key)
        if not isinstance(declared, Mapping):
            raise RuntimeError(f"original MACS metadata lacks {key} stream audit")
        if (
            str(declared.get("path", "")) != path.name
            or int(declared.get("bytes", -1)) != actual_bytes
            or str(declared.get("sha256", "")) != actual_hash
        ):
            raise RuntimeError(
                f"frozen MACS {key} stream differs from the original final metadata"
            )
        audit[key] = {"path": path.name, "bytes": actual_bytes, "sha256": actual_hash}
    return audit


def refinalize_existing_macs_metadata(output: Path) -> dict[str, Path]:
    """Correct a metadata-only cohort-audit bug without rerunning MACS fits.

    This migration is intentionally guarded for the historical CSV type bug:
    all finalized raw streams must first match both their append journal and the
    original metadata.  It snapshots the pre-fix metadata, writes only a new
    metadata/hash pair and a sidecar audit, and is idempotent after success.
    """

    output = output.resolve()
    paths: dict[str, Path] = {
        "results": output / "macs_results.csv",
        "predictions": output / "macs_predictions.csv",
        "curves": output / "macs_factor_curves.jsonl",
        "metadata": output / "macs_metadata.json",
        "metadata_sha256": output / "macs_metadata.sha256",
        "progress": output / "macs_progress.json",
    }
    snapshot_path = output / "macs_metadata.before_csv_cohort_audit_fix.json"
    audit_path = output / "macs_metadata_refinalization_audit.json"
    if not output.is_dir():
        raise RuntimeError(f"MACS output directory does not exist: {output}")
    if not paths["metadata"].exists() or not paths["metadata_sha256"].exists():
        raise RuntimeError("metadata-only re-finalization requires an existing final metadata pair")
    if not paths["progress"].exists():
        raise RuntimeError("metadata-only re-finalization requires a MACS progress journal")

    metadata_bytes = paths["metadata"].read_bytes()
    metadata_hash = _sha256_bytes(metadata_bytes)
    try:
        declared_metadata_hash = paths["metadata_sha256"].read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as error:
        raise RuntimeError("MACS metadata checksum file is unreadable") from error
    if declared_metadata_hash != metadata_hash:
        raise RuntimeError("MACS metadata checksum does not match the frozen metadata")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"MACS final metadata or progress is unreadable: {error}") from error
    if not isinstance(metadata, Mapping) or not isinstance(progress, Mapping):
        raise RuntimeError("MACS final metadata or progress is not a JSON object")
    if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("MACS progress journal has an unsupported schema")
    if progress.get("status") != "finalized":
        raise RuntimeError("MACS output is not finalized; use the normal resumable runner")
    if int(progress.get("committed_cohorts", -1)) != int(
        progress.get("expected_cohorts", -2)
    ):
        raise RuntimeError("MACS output has uncommitted folds; use the normal resumable runner")
    completion = _curve_completion(progress)
    if completion["state"] != "completed":
        raise RuntimeError("MACS output lacks a committed registered curve fit")
    _validate_committed_curves(paths["curves"], progress)
    raw_streams_before = _frozen_macs_stream_audit(paths, progress, metadata)

    if audit_path.exists():
        try:
            prior_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing MACS re-finalization audit is unreadable") from error
        if not isinstance(prior_audit, Mapping) or prior_audit.get("state") != "completed":
            raise RuntimeError("existing MACS re-finalization audit is incomplete")
        if str(prior_audit.get("post_metadata_sha256", "")) != metadata_hash:
            raise RuntimeError("existing MACS re-finalization audit does not match metadata")
        if prior_audit.get("raw_streams_after") != raw_streams_before:
            raise RuntimeError("existing MACS re-finalization audit does not match frozen streams")
        return {**paths, "audit": audit_path, "snapshot": snapshot_path}

    run_contract = progress.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise RuntimeError("MACS progress lacks the immutable run contract")
    cohort_registry = run_contract.get("cohort_registry")
    method_order = run_contract.get("method_order")
    if not isinstance(cohort_registry, list) or not all(
        isinstance(item, Mapping) for item in cohort_registry
    ):
        raise RuntimeError("MACS run contract has no usable cohort registry")
    if not isinstance(method_order, list) or tuple(str(item) for item in method_order) != tuple(
        FIXED_METHOD_LABELS
    ):
        raise RuntimeError("MACS run contract method registry differs from the fixed protocol")
    if str(metadata.get("run_fingerprint")) != str(progress.get("run_fingerprint")):
        raise RuntimeError("MACS metadata and progress do not describe the same run")
    if str(metadata.get("mode")) != str(run_contract.get("mode")):
        raise RuntimeError("MACS metadata and progress disagree on the execution mode")
    with paths["results"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(RESULT_FIELDS):
            raise RuntimeError("MACS result header differs from the registered schema")
        result_rows = [dict(row) for row in reader]
    cohort_audit = audit_macs_result_cohorts(
        result_rows,
        cohort_registry=cohort_registry,
        method_order=method_order,
    )
    if not cohort_audit["passed"]:
        raise RuntimeError(
            "metadata-only re-finalization refuses a genuine cohort inconsistency: "
            f"{list(cohort_audit['issues'])[:3]}"
        )
    prior_cohort_audit = metadata.get("cohort_audit")
    if isinstance(prior_cohort_audit, Mapping) and prior_cohort_audit.get("passed") is True:
        raise RuntimeError("MACS metadata already reports a passing cohort audit")
    if snapshot_path.exists():
        if file_sha256(snapshot_path) != metadata_hash:
            raise RuntimeError(
                "pre-fix MACS metadata snapshot exists but does not match current metadata"
            )
    else:
        _atomic_write_bytes(snapshot_path, metadata_bytes)

    registered_curve_fit = completion["audit"]
    assert isinstance(registered_curve_fit, Mapping)
    admission_source = metadata.get("admission_source")
    if not isinstance(admission_source, Mapping):
        raise RuntimeError("MACS metadata lacks an admission-source audit")
    formal_protocol_complete = bool(
        metadata.get("mode") == "formal"
        and registered_curve_fit.get("attempt_status") == "success"
        and registered_curve_fit.get("converged") is True
        and admission_source.get("mode") == "formal"
        and admission_source.get("formal_protocol_complete") is True
    )
    formal_eligible = bool(
        formal_protocol_complete and admission_source.get("formal_claims_eligible") is True
    )
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    revision = {
        "schema_version": METADATA_REFINALIZATION_SCHEMA_VERSION,
        "timestamp_utc": timestamp,
        "reason": (
            "Corrected the final cohort audit's CSV repeat/fold type normalization; "
            "no raw result, prediction, or curve stream was rerun or edited."
        ),
        "previous_metadata_sha256": metadata_hash,
        "previous_cohort_audit_sha256": _object_sha256(prior_cohort_audit),
        "previous_cohort_audit_passed": bool(
            isinstance(prior_cohort_audit, Mapping)
            and prior_cohort_audit.get("passed") is True
        ),
        "cohort_audit_schema_version": COHORT_AUDIT_SCHEMA_VERSION,
        "pre_fix_metadata_snapshot": snapshot_path.name,
        "raw_streams_before": raw_streams_before,
        "runner_file_sha256": file_sha256(Path(__file__).resolve()),
    }
    corrected = dict(metadata)
    revisions = corrected.get("metadata_refinalizations", [])
    if not isinstance(revisions, list):
        raise RuntimeError("MACS metadata has a malformed metadata-refinalization history")
    corrected["metadata_refinalizations"] = [*revisions, revision]
    corrected["cohort_audit"] = cohort_audit
    corrected["formal_protocol_complete"] = formal_protocol_complete
    corrected["descriptive_results_eligible"] = formal_protocol_complete
    corrected["formal_claims_eligible"] = formal_eligible
    corrected["formal_claims_reason"] = (
        "Locked 5x5 subject CV and the registered full-data curve fit completed and inherited a claim-eligible strict benchmark audit."
        if formal_eligible
        else "Quick/incomplete application or the inherited strict benchmark audit prevents claims."
    )
    corrected["metadata_refinalized_utc"] = timestamp
    corrected_bytes = (
        json.dumps(corrected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    corrected_hash = _sha256_bytes(corrected_bytes)
    _atomic_write_bytes(paths["metadata"], corrected_bytes)
    _atomic_write_bytes(
        paths["metadata_sha256"],
        f"{corrected_hash}  {paths['metadata'].name}\n".encode("ascii"),
    )
    raw_streams_after = _frozen_macs_stream_audit(paths, progress, corrected)
    if raw_streams_after != raw_streams_before:
        raise RuntimeError("metadata-only re-finalization unexpectedly changed a raw stream")
    _atomic_write_json(
        audit_path,
        {
            "schema_version": METADATA_REFINALIZATION_SCHEMA_VERSION,
            "state": "completed",
            "timestamp_utc": timestamp,
            "previous_metadata": {
                "path": snapshot_path.name,
                "sha256": metadata_hash,
            },
            "post_metadata_sha256": corrected_hash,
            "progress_sha256": file_sha256(paths["progress"]),
            "raw_streams_before": raw_streams_before,
            "raw_streams_after": raw_streams_after,
            "corrected_cohort_audit": cohort_audit,
            "operation": "metadata-only; raw streams were verified frozen before and after",
        },
    )
    return {**paths, "audit": audit_path, "snapshot": snapshot_path}


def execute(args: argparse.Namespace) -> dict[str, Path]:
    quick = bool(args.quick)
    mode = "quick" if quick else "formal"
    trace_tuning_lock = load_trace_tuning_lock()
    raw = read_macs_csv(args.data.resolve())
    if not quick and (len(raw.cd4) != 2376 or len(np.unique(raw.person)) != 369):
        raise ValueError("formal MACS protocol requires exactly 2,376 rows and 369 subjects")
    global_bounds = {
        "time": (float(np.min(raw.time)), float(np.max(raw.time))),
        "age": (float(np.min(raw.age)), float(np.max(raw.age))),
        "cesd": (float(np.min(raw.cesd)), float(np.max(raw.cesd))),
    }
    variants = [("primary", 4)] if quick else [
        ("primary", 6),
        ("delete_outer_fence_subjects", 6),
        ("winsorize_response_1_99", 6),
        ("basis_5", 5),
        ("basis_8", 8),
    ]
    n_splits, n_repeats = ((2, 1) if quick else (5, 5))
    admissions, admission_source = _load_admissions(args.admission_metadata)
    adapters = adapter_registry()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output / "macs_results.csv",
        "predictions": output / "macs_predictions.csv",
        "curves": output / "macs_factor_curves.jsonl",
        "metadata": output / "macs_metadata.json",
        "metadata_sha256": output / "macs_metadata.sha256",
        "progress": output / "macs_progress.json",
    }
    variant_metadata: dict[str, object] = {}
    variant_datasets: dict[str, SubjectDataset] = {}
    cohorts: list[MACSCohort] = []
    for variant, basis_dimension in variants:
        dataset = prepare_macs_variant(raw, variant=variant, global_bounds=global_bounds)
        variant_datasets[variant] = dataset
        variant_metadata[variant] = {
            "basis_dimension": basis_dimension,
            "n_rows": dataset.n_rows,
            "n_subjects": dataset.n_subjects,
            "data_hash": dataset.data_hash,
            **dataset.metadata,
        }
        splits = make_repeated_subject_folds(
            dataset,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=args.seed,
        )
        for split in splits:
            cohorts.append(
                MACSCohort(
                    index=len(cohorts),
                    variant=variant,
                    basis_dimension=basis_dimension,
                    split=split,
                    dataset=dataset,
                )
            )

    preflight_reports: dict[str, object] = {}
    for method in FIXED_METHOD_LABELS:
        if method != TRACE and admissions.get(method) != "admitted":
            continue
        print(f"[preflight] {method}", flush=True)
        preflight_reports[method] = adapters[method].preflight()

    jobs = int(args.jobs)
    preflight_payload = {
        method: asdict(report) for method, report in preflight_reports.items()
    }
    contract = _macs_run_contract(
        mode=mode,
        seed=int(args.seed),
        jobs=jobs,
        data_path=args.data.resolve(),
        data_sha256=file_sha256(args.data.resolve()),
        variants=variants,
        n_splits=n_splits,
        n_repeats=n_repeats,
        cohorts=cohorts,
        variant_metadata=variant_metadata,
        admissions=admissions,
        admission_source=admission_source,
        trace_tuning_lock=trace_tuning_lock,
        preflight=preflight_payload,
    )
    fingerprint = _run_fingerprint(contract)
    progress, hashers = _initialize_or_restore_streams(
        paths,
        contract=contract,
        fingerprint=fingerprint,
        expected_cohorts=len(cohorts),
    )
    initially_committed = int(progress["committed_cohorts"])
    result_rows = _load_and_validate_committed_results(
        paths["results"],
        cohorts,
        initially_committed,
        mode=mode,
        admissions=admissions,
    )
    _validate_committed_predictions(
        paths["predictions"], cohorts, initially_committed, result_rows
    )
    _validate_committed_curves(paths["curves"], progress)
    completion = _curve_completion(progress)
    if (
        initially_committed == len(cohorts)
        and completion["state"] == "completed"
        and _existing_final_is_valid(paths, fingerprint)
    ):
        if progress.get("status") != "finalized":
            progress = dict(progress)
            progress.update(
                status="finalized",
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            _atomic_write_json(paths["progress"], progress)
        print("[resume] all MACS folds and final hashes are valid; no-op", flush=True)
        return paths
    if initially_committed:
        print(
            f"[resume] committed_folds={initially_committed}/{len(cohorts)}; "
            "uncommitted tails truncated",
            flush=True,
        )

    if jobs > 1:
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

    pending_tasks = [
        (cohort, mode, quick, admissions, preflight_reports)
        for cohort in cohorts[initially_committed:]
    ]
    for completed, (index, rows, predictions) in enumerate(
        _ordered_macs_cohort_results(
            pending_tasks, jobs=jobs, worker=_macs_cohort_task
        ),
        start=initially_committed + 1,
    ):
        cohort = cohorts[index]
        if index != int(progress["committed_cohorts"]):
            raise RuntimeError("MACS worker result is not in registered fold order")
        _validate_cohort_rows(rows, cohort, mode=mode, admissions=admissions)
        _validate_fold_prediction_payload(predictions, cohort, rows)
        progress = _commit_fold(
            paths,
            progress,
            hashers,
            cohort=cohort,
            rows=rows,
            predictions=predictions,
        )
        statuses = ",".join(
            f"{row['method']}={row['attempt_status']}" for row in rows
        )
        print(
            f"[macs] {completed}/{len(cohorts)} variant={cohort.variant} "
            f"repeat={cohort.split.repeat} fold={cohort.split.fold} {statuses}",
            flush=True,
        )

    if int(progress["committed_cohorts"]) != len(cohorts):
        raise RuntimeError("MACS execution ended before every registered fold committed")
    result_rows = _load_and_validate_committed_results(
        paths["results"],
        cohorts,
        len(cohorts),
        mode=mode,
        admissions=admissions,
    )
    _validate_committed_predictions(paths["predictions"], cohorts, len(cohorts), result_rows)

    completion = _curve_completion(progress)
    if completion["state"] == "pending":
        primary_basis_dimension = dict(variants)["primary"]
        registered_curve_row, registered_curve_fit = _fit_registered_full_data_trace_curves(
            adapters[TRACE],
            mode=mode,
            quick=quick,
            variant="primary",
            basis_dimension=primary_basis_dimension,
            dataset=variant_datasets["primary"],
            seed=args.seed,
            preflight_report=preflight_reports.get(TRACE),
        )
        progress = _commit_registered_curve(
            paths,
            progress,
            hashers,
            curve_row=registered_curve_row,
            curve_audit=registered_curve_fit,
        )
        print(
            "[macs-curve-fit] "
            f"fit_id={registered_curve_fit['fit_id']} "
            f"status={registered_curve_fit['attempt_status']}",
            flush=True,
        )
    completion = _curve_completion(progress)
    registered_curve_fit = completion["audit"]
    assert isinstance(registered_curve_fit, Mapping)
    _validate_committed_curves(paths["curves"], progress)

    cohort_audit = audit_macs_result_cohorts(
        result_rows,
        cohort_registry=_cohort_registry(cohorts),
        method_order=FIXED_METHOD_LABELS,
    )
    issues = list(cohort_audit["issues"])
    formal_protocol_complete = bool(
        not quick
        and not issues
        and registered_curve_fit.get("attempt_status") == "success"
        and registered_curve_fit.get("converged") is True
        and admission_source.get("mode") == "formal"
        and admission_source.get("formal_protocol_complete") is True
    )
    formal_eligible = bool(
        formal_protocol_complete
        and admission_source.get("formal_claims_eligible") is True
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "run_fingerprint": fingerprint,
        "resume_audit": {
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "initially_committed_folds": initially_committed,
            "expected_folds": len(cohorts),
            "committed_offsets": dict(progress["committed_offsets"]),
            "committed_sha256": dict(progress["committed_sha256"]),
            "uncommitted_tail_policy": (
                "truncate to the last atomically journaled byte offsets before "
                "skipping a fold"
            ),
        },
        "formal_protocol_complete": formal_protocol_complete,
        "descriptive_results_eligible": formal_protocol_complete,
        "formal_claims_eligible": formal_eligible,
        "formal_claims_reason": (
            "Locked 5x5 subject CV and the registered full-data curve fit completed and inherited a claim-eligible strict benchmark audit."
            if formal_eligible
            else "Quick/incomplete application or the inherited strict benchmark audit prevents claims."
        ),
        "data_source": {
            "path": str(args.data.resolve()),
            "sha256": file_sha256(args.data.resolve()),
            "package": "CRAN catdata::aids",
            "n_rows": len(raw.cd4),
            "n_subjects": int(len(np.unique(raw.person))),
        },
        "response_transform": "none (raw CD4)",
        "trace_tuning_lock": {
            **trace_tuning_lock,
            "path": "protocol/trace_tuning_v1.json",
        },
        "coordinate_mapping": {
            key: {"raw_min": value[0], "raw_max": value[1], "mapped_domain": [0.0, 1.0]}
            for key, value in global_bounds.items()
        },
        "fold_protocol": {
            "unit": "person/subject",
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "seed": args.seed,
            "role": "held-out prediction evaluation only; no factor curves are retained",
        },
        "curve_protocol": {
            "registered_fit_id": registered_curve_fit["fit_id"],
            "source": "one TRACE-VCAM fit on the complete primary MACS data set",
            "aggregation": "none",
            "fold_curves_serialized": False,
            "identification": "joint phi centering, baseline absorption, and beta integral-one scaling before serialization",
            "identification_implementation": {
                "path": "scripts/run_strict_benchmark.py::_common_identify_curves",
                "file_sha256": file_sha256(ROOT / "scripts" / "run_strict_benchmark.py"),
            },
            "runner_file_sha256": file_sha256(Path(__file__).resolve()),
        },
        "registered_curve_fit": registered_curve_fit,
        "execution": {
            "jobs": jobs,
            "worker_blas_threads": 1 if jobs > 1 else None,
            "completion_order_canonicalized": True,
            "ordered_task_scheduler_version": ORDERED_TASK_SCHEDULER_VERSION,
            "ordered_task_prefetch_factor": ORDERED_TASK_PREFETCH_FACTOR,
            "max_active_worker_tasks": jobs,
            "max_in_flight_futures": _macs_prefetch_limit(jobs),
            "max_parent_buffered_cohort_payloads": _macs_prefetch_limit(jobs),
        },
        "variants": variant_metadata,
        "admission_source": admission_source,
        "admission_status": admissions,
        "method_order": list(FIXED_METHOD_LABELS),
        "method_display_names": {
            key: value.display_name for key, value in METHOD_SPECS.items()
        },
        "method_specs": {key: asdict(value) for key, value in METHOD_SPECS.items()},
        "tuning_record_policy": (
            "tuning_json is the requested rule/configuration; realized_tuning_json "
            "is the adapter-returned configuration including selected CV/BIC values "
            "when fitting succeeds, and remains empty when no fit artifact exists"
        ),
        "cohort_audit": cohort_audit,
        "inference": "No confidence intervals are computed or claimed.",
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {
            package: (
                importlib.metadata.version(package)
                if _package_exists(package)
                else "not-installed"
            )
            for package in ("numpy", "scipy", "scikit-learn")
        },
        "source_sha256": _source_hashes(),
        "files": {
            key: {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
            if key not in {"metadata", "metadata_sha256", "progress"}
        },
    }
    _atomic_write_bytes(
        paths["metadata"],
        (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    metadata_hash = file_sha256(paths["metadata"])
    _atomic_write_bytes(
        paths["metadata_sha256"],
        f"{metadata_hash}  {paths['metadata'].name}\n".encode("ascii"),
    )
    progress = dict(progress)
    progress.update(
        status="finalized",
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _atomic_write_json(paths["progress"], progress)
    return paths


def _package_exists(package: str) -> bool:
    try:
        importlib.metadata.version(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "catdata_aids.csv")
    parser.add_argument("--admission-metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    paths = execute(parse_args(argv))
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

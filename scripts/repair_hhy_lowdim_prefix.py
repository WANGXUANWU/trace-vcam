"""Replace only the low-dimensional Three-step M-VCAM artifacts in a strict prefix.

This deliberately narrow repair tool exists because a verified Example-1/2 prefix
was copied before the audited final-normalisation/IRLS correction to the
Three-step M-VCAM adapter.  It accepts *only* a freshly migrated complete
Example-1/2 prefix, regenerates the deterministic data and subject split for
every Example-2 cohort, and reruns only that adapter.  All non-target records
are copied in canonical registry order and are checked for logical identity.

The source directory is never modified.  The destination is built in a sibling
temporary directory and is published atomically only after the result,
prediction, and curve streams pass the strict runner's validators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Works both as ``python scripts/...`` and in the unit-test package import.
    from scripts import run_strict_benchmark as strict
except ModuleNotFoundError:  # pragma: no cover - direct-script compatibility guard
    import run_strict_benchmark as strict  # type: ignore[no-redef]


REPAIR_SCHEMA_VERSION = "vcam-hhy-lowdim-prefix-repair/1"
REPAIR_LINEAGE_FILENAME = "hhy_lowdim_repair_lineage.json"
UPSTREAM_MIGRATION_FILENAME = "strict_prefix_migration.json"
COPIED_UPSTREAM_MIGRATION_FILENAME = "source_strict_prefix_migration.json"
TEMPORARY_MARKER_FILENAME = ".hhy_lowdim_repair_temporary.json"
STREAM_FILENAMES = {
    "results": "strict_results.csv",
    "predictions": "strict_predictions.csv",
    "curves": "strict_factor_curves.jsonl",
}
TARGET_EXAMPLE = "Example 2"
ALLOWED_PREFIX_EXAMPLES = {"Example 1", TARGET_EXAMPLE}


class _PrefixRawReader(io.RawIOBase):
    """Expose exactly a journaled byte prefix to a text reader."""

    def __init__(self, path: Path, limit: int) -> None:
        self._handle = path.open("rb")
        self._remaining = int(limit)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        if self._remaining <= 0:
            return 0
        payload = self._handle.read(min(len(buffer), self._remaining))
        count = len(payload)
        buffer[:count] = payload
        self._remaining -= count
        return count

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


class _RecordDigest:
    """Compact audit digest for one logical stream partition."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.records = 0
        self.bytes = 0

    def update(self, payload: bytes, *, records: int) -> None:
        self._digest.update(payload)
        self.records += int(records)
        self.bytes += len(payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self._digest.hexdigest(),
        }


def _prefix_sha256(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    remaining = int(limit)
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"{path} ended before its committed offset")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _limited_text(path: Path, limit: int) -> io.TextIOWrapper:
    return io.TextIOWrapper(
        io.BufferedReader(_PrefixRawReader(path, limit)), encoding="utf-8", newline=""
    )


def _source_paths(root: Path) -> dict[str, Path]:
    return {key: root / filename for key, filename in STREAM_FILENAMES.items()}


def _current_contract(*, jobs: int) -> tuple[dict[str, object], list[tuple[object, int]]]:
    """Build the exact formal runner contract used by the repaired prefix."""

    trace_lock = strict.load_trace_tuning_lock()
    scenarios = strict.registered_scenarios(quick=False, include_reproduction_audit=False)
    replications = {
        scenario.scenario: scenario.formal_replications for scenario in scenarios
    }
    targets = strict.load_published_targets(strict.ROOT / "protocol" / "published_targets.json")
    adapters = strict.adapter_registry()
    preflight = {
        method: asdict(adapters[method].preflight()) for method in strict.FIXED_METHOD_LABELS
    }
    contract = strict._run_contract(
        mode="formal",
        root_seed=strict.DEFAULT_ROOT_SEED,
        jobs=int(jobs),
        scenarios=scenarios,
        replications=replications,
        targets=targets,
        trace_tuning_lock=trace_lock,
        preflight=preflight,
        include_reproduction_audit=False,
    )
    return contract, strict._ordered_registered_tasks(scenarios, replications)


def _validate_source_progress(source: Path) -> tuple[dict[str, object], dict[str, Path]]:
    progress_path = source / "strict_progress.json"
    if not progress_path.is_file():
        raise RuntimeError(f"source has no strict progress journal: {progress_path}")
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("source strict progress is not valid JSON") from error
    if not isinstance(progress, Mapping) or progress.get("schema_version") != strict.PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("source progress has an unsupported schema")
    offsets = progress.get("committed_offsets")
    hashes = progress.get("committed_sha256")
    if not isinstance(offsets, Mapping) or not isinstance(hashes, Mapping):
        raise RuntimeError("source progress lacks committed offsets/hashes")
    paths = _source_paths(source)
    for key, path in paths.items():
        offset = int(offsets.get(key, -1))
        if offset < 0 or not path.is_file() or path.stat().st_size < offset:
            raise RuntimeError(f"source {key} does not contain its committed prefix")
        if _prefix_sha256(path, offset) != str(hashes.get(key, "")):
            raise RuntimeError(f"source committed {key} prefix hash mismatch")
    return dict(progress), paths


def _task_key(task: tuple[object, int]) -> tuple[str, int]:
    scenario, replicate = task
    return str(getattr(scenario, "scenario")), int(replicate)


def _validate_repair_scope(
    tasks: Sequence[tuple[object, int]], *, prefix: int, source_progress: Mapping[str, object]
) -> list[tuple[object, int]]:
    if int(source_progress.get("committed_cohorts", -1)) != int(prefix):
        raise RuntimeError(
            "source must be a fresh prefix with exactly --prefix-cohorts committed; "
            "do not repair a prefix after the formal continuation has started"
        )
    if not 1 <= prefix < len(tasks):
        raise RuntimeError("prefix must be nonempty and leave registered work after it")
    prefix_tasks = list(tasks[:prefix])
    if not prefix_tasks or any(
        str(getattr(scenario, "example")) not in ALLOWED_PREFIX_EXAMPLES
        for scenario, _ in prefix_tasks
    ):
        raise RuntimeError("repair prefix must contain only complete Example 1--2 cohorts")
    if any(
        str(getattr(scenario, "example")) in ALLOWED_PREFIX_EXAMPLES
        for scenario, _ in tasks[prefix:]
    ):
        raise RuntimeError("prefix stops before all registered Example 1--2 cohorts")
    if not any(str(getattr(scenario, "example")) == TARGET_EXAMPLE for scenario, _ in prefix_tasks):
        raise RuntimeError("repair prefix contains no Example-2 cohort")
    if _task_key(tasks[prefix - 1])[0] == _task_key(tasks[prefix])[0]:
        raise RuntimeError("prefix must end at a complete scenario boundary")
    last = source_progress.get("last_completed")
    if not isinstance(last, Mapping) or (
        str(last.get("scenario")) != _task_key(prefix_tasks[-1])[0]
        or int(last.get("replicate", -1)) != _task_key(prefix_tasks[-1])[1]
    ):
        raise RuntimeError("source progress last-completed task is inconsistent with the prefix")
    return prefix_tasks


def _read_exact_rows(reader: csv.DictReader, count: int, *, context: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _ in range(count):
        try:
            rows.append(dict(next(reader)))
        except StopIteration as error:
            raise RuntimeError(f"source result stream ends inside {context}") from error
    return rows


def _iter_csv_groups(
    handle: io.TextIOBase,
    fields: Sequence[str],
    *,
    stream_name: str,
) -> Iterator[tuple[tuple[str, int, str], list[dict[str, str]]]]:
    reader = csv.DictReader(handle)
    if tuple(reader.fieldnames or ()) != tuple(fields):
        raise RuntimeError(f"unexpected {stream_name} CSV header")
    current_key: tuple[str, int, str] | None = None
    group: list[dict[str, str]] = []
    for source_row in reader:
        row = dict(source_row)
        key = (str(row.get("scenario", "")), int(row.get("replicate", "-1")), str(row.get("method", "")))
        if current_key is not None and key != current_key:
            yield current_key, group
            group = []
        current_key = key
        group.append(row)
    if current_key is not None:
        yield current_key, group


def _iter_curve_groups(
    handle: io.TextIOBase, *, stream_name: str
) -> Iterator[tuple[tuple[str, int, str], list[dict[str, object]]]]:
    """Yield contiguous curve records grouped by their registered stream key."""

    current_key: tuple[str, int, str] | None = None
    group: list[dict[str, object]] = []
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid {stream_name} JSONL row {line_number}") from error
        if not isinstance(record, Mapping):
            raise RuntimeError(f"invalid {stream_name} record {line_number}")
        payload = dict(record)
        key = (
            str(payload.get("scenario", "")),
            int(payload.get("replicate", -1)),
            str(payload.get("method", "")),
        )
        if current_key is not None and key != current_key:
            yield current_key, group
            group = []
        current_key = key
        group.append(payload)
    if current_key is not None:
        yield current_key, group


def _next_group(
    iterator: Iterator[tuple[tuple[str, int, str], list[dict[str, str]]]],
    expected: tuple[str, int, str],
    *,
    stream_name: str,
) -> list[dict[str, str]]:
    try:
        observed, rows = next(iterator)
    except StopIteration as error:
        raise RuntimeError(f"{stream_name} is missing expected group {expected}") from error
    if observed != expected:
        raise RuntimeError(
            f"{stream_name} canonical order mismatch: observed {observed}, expected {expected}"
        )
    return rows


def _next_optional_group(
    iterator: Iterator[tuple[tuple[str, int, str], list[object]]],
) -> tuple[tuple[str, int, str], list[object]] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _canonical_stream_positions(
    prefix_tasks: Sequence[tuple[object, int]],
) -> dict[tuple[str, int, str], int]:
    """Map every registered stream key to its canonical sequence position."""

    positions: dict[tuple[str, int, str], int] = {}
    for cohort_index, (scenario, replicate) in enumerate(prefix_tasks):
        scenario_name = str(getattr(scenario, "scenario"))
        for method_index, method in enumerate(strict.FIXED_METHOD_LABELS):
            key = (scenario_name, int(replicate), method)
            if key in positions:
                raise RuntimeError(f"duplicate registered stream key: {key}")
            positions[key] = cohort_index * len(strict.FIXED_METHOD_LABELS) + method_index
    return positions


def _consume_source_group_if_present(
    pending: tuple[tuple[str, int, str], list[object]] | None,
    iterator: Iterator[tuple[tuple[str, int, str], list[object]]],
    *,
    expected: tuple[str, int, str],
    positions: Mapping[tuple[str, int, str], int],
    stream_name: str,
) -> tuple[list[object] | None, tuple[tuple[str, int, str], list[object]] | None]:
    """Consume one actual source group only when it belongs at ``expected``.

    Result status is deliberately not consulted here.  Historical strict
    prefixes can contain a prediction or curve payload from a failed method;
    the stream itself remains the authoritative record of artifacts that must
    survive this narrowly targeted HHY replacement.
    """

    if pending is None:
        return None, None
    observed, payload = pending
    observed_position = positions.get(observed)
    if observed_position is None:
        raise RuntimeError(f"{stream_name} contains an unregistered group: {observed}")
    expected_position = positions[expected]
    if observed_position < expected_position:
        raise RuntimeError(
            f"{stream_name} is not in canonical order: observed {observed} before {expected}"
        )
    if observed_position > expected_position:
        return None, pending
    return payload, _next_optional_group(iterator)


def _expect_exhausted(iterator: Iterator[object], *, stream_name: str) -> None:
    try:
        extra = next(iterator)
    except StopIteration:
        return
    raise RuntimeError(f"{stream_name} contains an unexpected trailing record: {extra!r}")


def _rebuild_cohort(
    scenario: object, replicate: int, root_seed: int
) -> tuple[object, object, object, object, object, int, int]:
    """Regenerate precisely the DGP and registered split for one repair task."""

    scenario_name = str(getattr(scenario, "scenario"))
    seed = strict._stable_seed(root_seed, scenario_name, replicate, "data")
    split_seed = strict._stable_seed(root_seed, scenario_name, replicate, "subject-split")
    raw = scenario.build(seed)
    dataset = strict._subject_dataset(raw, scenario)
    train, test, split = strict._registered_split(dataset, raw, scenario, split_seed=split_seed)
    return raw, dataset, train, test, split, seed, split_seed


def _source_shared_value(
    cohort: Sequence[Mapping[str, object]], field: str, *, context: str) -> str:
    values = {str(row.get(field, "")) for row in cohort}
    if len(values) != 1 or "" in values:
        raise RuntimeError(f"{context}: source cohort has inconsistent {field}")
    return next(iter(values))


def _regenerated_identity(
    *, raw: object, dataset: object, split: object, seed: int, split_seed: int
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "split_seed": int(split_seed),
        "data_hash": str(getattr(dataset, "data_hash")),
        "train_subject_hash": str(getattr(split, "train_hash")),
        "test_subject_hash": str(getattr(split, "test_hash")),
        "design_id": str(getattr(raw, "design_id")),
        "provenance": str(getattr(raw, "provenance")),
    }


def _validate_regenerated_identity(
    cohort: Sequence[Mapping[str, object]],
    *,
    identity: Mapping[str, object],
    context: str,
) -> dict[str, object]:
    expected = {field: str(identity.get(field, "")) for field in (
        "seed",
        "split_seed",
        "data_hash",
        "train_subject_hash",
        "test_subject_hash",
        "design_id",
        "provenance",
    )}
    observed = {
        field: _source_shared_value(cohort, field, context=context) for field in expected
    }
    mismatch = {
        field: {"source": observed[field], "regenerated": expected[field]}
        for field in expected
        if observed[field] != expected[field]
    }
    if mismatch:
        raise RuntimeError(f"{context}: regenerated DGP/split identity mismatch: {mismatch}")
    return {
        "seed": int(identity["seed"]),
        "split_seed": int(identity["split_seed"]),
        "data_hash": expected["data_hash"],
        "train_subject_hash": expected["train_subject_hash"],
        "test_subject_hash": expected["test_subject_hash"],
        "design_id": expected["design_id"],
        "provenance": expected["provenance"],
    }


def _validate_replacement(
    row: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    curve: Mapping[str, object] | None,
    *,
    scenario: object,
    replicate: int,
    source_cohort: Sequence[Mapping[str, object]],
    test_row_ids: Sequence[str],
    root_seed: int,
) -> None:
    scenario_name = str(getattr(scenario, "scenario"))
    if str(row.get("method")) != strict.HHY:
        raise RuntimeError(f"{scenario_name}/{replicate}: repair returned the wrong method")
    for field in ("scenario", "replicate", "seed", "split_seed", "data_hash", "train_subject_hash", "test_subject_hash"):
        value = str(row.get(field, ""))
        source_value = _source_shared_value(source_cohort, field, context=f"{scenario_name}/{replicate}")
        if value != source_value:
            raise RuntimeError(
                f"{scenario_name}/{replicate}: repaired HHY {field} differs from source cohort"
            )
    expected_seed = strict._stable_seed(root_seed, scenario_name, replicate, "data")
    expected_split_seed = strict._stable_seed(
        root_seed, scenario_name, replicate, "subject-split"
    )
    if int(row.get("seed", -1)) != expected_seed or int(row.get("split_seed", -1)) != expected_split_seed:
        raise RuntimeError(f"{scenario_name}/{replicate}: repaired HHY seed mismatch")
    success = str(row.get("attempt_status")) == "success"
    if success and (not predictions or curve is None):
        raise RuntimeError(f"{scenario_name}/{replicate}: successful HHY repair lacks predictions/curves")
    if not success and (predictions or curve is not None):
        raise RuntimeError(f"{scenario_name}/{replicate}: failed HHY repair emitted artifacts")
    expected_row_ids = tuple(str(item) for item in test_row_ids)
    observed_row_ids = tuple(str(item.get("row_id", "")) for item in predictions)
    if predictions and observed_row_ids != tuple(sorted(expected_row_ids)):
        raise RuntimeError(
            f"{scenario_name}/{replicate}: repaired HHY prediction row IDs differ from the registered test split"
        )
    for prediction in predictions:
        if (
            str(prediction.get("scenario")) != scenario_name
            or int(prediction.get("replicate", -1)) != replicate
            or str(prediction.get("method")) != strict.HHY
            or int(prediction.get("seed", -1)) != expected_seed
        ):
            raise RuntimeError(f"{scenario_name}/{replicate}: malformed repaired HHY prediction")
    if curve is not None and (
        str(curve.get("scenario")) != scenario_name
        or int(curve.get("replicate", -1)) != replicate
        or str(curve.get("method")) != strict.HHY
        or int(curve.get("seed", -1)) != expected_seed
    ):
        raise RuntimeError(f"{scenario_name}/{replicate}: malformed repaired HHY curve")


def _is_target(scenario: object, method: str) -> bool:
    return str(getattr(scenario, "example")) == TARGET_EXAMPLE and method == strict.HHY


def _csv_chunk(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    return strict._csv_bytes(fields, rows, include_header=False)


def _copy_upstream_manifest(source: Path, temporary: Path) -> dict[str, object]:
    source_manifest = source / UPSTREAM_MIGRATION_FILENAME
    if not source_manifest.is_file():
        raise RuntimeError(
            "source must be a freshly migrated prefix with strict_prefix_migration.json"
        )
    try:
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("upstream prefix-migration manifest is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("upstream prefix-migration manifest is not an object")
    destination = temporary / COPIED_UPSTREAM_MIGRATION_FILENAME
    shutil.copyfile(source_manifest, destination)
    return {
        "source_filename": source_manifest.name,
        "source_sha256": strict.file_sha256(source_manifest),
        "copied_filename": destination.name,
        "copied_sha256": strict.file_sha256(destination),
        "upstream_source_dependencies": dict(payload.get("source", {})).get("source_sha256"),
        "upstream_destination_dependencies": dict(payload.get("destination", {})).get(
            "source_sha256"
        ),
    }


def _repair_hhy_task(
    scenario: object, replicate: int, root_seed: int
) -> dict[str, object]:
    """Run one deterministic HHY repair task in a parent or worker process."""

    raw, dataset, train, test, split, seed, split_seed = _rebuild_cohort(
        scenario, replicate, root_seed
    )
    adapter = strict.adapter_registry().get(strict.HHY)
    if adapter is None:
        raise RuntimeError("worker adapter registry lacks Three-step M-VCAM")
    applicability, reason = strict._safe_applicability(strict.HHY, scenario)
    if applicability != "applicable":
        raise RuntimeError(
            f"{getattr(scenario, 'scenario')}/{replicate}: HHY is no longer registered as applicable"
        )
    row, predictions, curve = strict.run_one_method(
        adapter,
        scenario,
        raw,
        dataset,
        train,
        test,
        split,
        mode="formal",
        quick=False,
        replicate=replicate,
        seed=seed,
        split_seed=split_seed,
        applicability=applicability,
        applicability_reason=reason,
        admission_status="admitted",
    )
    return {
        "identity": _regenerated_identity(
            raw=raw, dataset=dataset, split=split, seed=seed, split_seed=split_seed
        ),
        "test_row_ids": [str(item) for item in getattr(test, "row_id")],
        "row": dict(row),
        "predictions": [dict(item) for item in predictions],
        "curve": None if curve is None else dict(curve),
    }


def _configure_worker_threads(jobs: int) -> None:
    """Match the strict runner's one-BLAS-thread-per-worker policy."""

    if int(jobs) > 1:
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"


def _load_source_cohorts(
    *,
    source_paths: Mapping[str, Path],
    source_offsets: Mapping[str, object],
    prefix_tasks: Sequence[tuple[object, int]],
    root_seed: int,
) -> tuple[list[tuple[object, int, list[dict[str, str]]]], dict[tuple[str, int, str], str]]:
    """Validate the source result prefix before any repair workers are submitted."""

    cohorts: list[tuple[object, int, list[dict[str, str]]]] = []
    source_status: dict[tuple[str, int, str], str] = {}
    with _limited_text(
        source_paths["results"], int(source_offsets["results"])
    ) as input_handle:
        reader = csv.DictReader(input_handle)
        if tuple(reader.fieldnames or ()) != tuple(strict.RESULT_FIELDS):
            raise RuntimeError("unexpected source result CSV header")
        for scenario, replicate in prefix_tasks:
            context = f"{getattr(scenario, 'scenario')}/{replicate}"
            cohort = _read_exact_rows(reader, len(strict.FIXED_METHOD_LABELS), context=context)
            strict._validate_cohort_rows(
                cohort,
                scenario,
                replicate,
                mode="formal",
                root_seed=root_seed,
            )
            source_by_method = {str(row["method"]): row for row in cohort}
            if set(source_by_method) != set(strict.FIXED_METHOD_LABELS):
                raise RuntimeError(f"{context}: malformed source method set")
            if str(getattr(scenario, "example")) == TARGET_EXAMPLE:
                target = source_by_method[strict.HHY]
                if str(target.get("admission_status")) != "admitted":
                    raise RuntimeError(f"{context}: source HHY row is not admitted")
                if str(target.get("applicability")) != "applicable":
                    raise RuntimeError(f"{context}: source HHY row is not applicable")
            for method, source_row in source_by_method.items():
                source_status[(str(getattr(scenario, "scenario")), replicate, method)] = str(
                    source_row.get("attempt_status", "")
                )
            cohorts.append((scenario, replicate, cohort))
        if next(reader, None) is not None:
            raise RuntimeError("source result stream contains trailing committed rows")
    return cohorts, source_status


def _rewrite_results_and_spools(
    *,
    source_paths: Mapping[str, Path],
    source_offsets: Mapping[str, object],
    temporary: Path,
    prefix_tasks: Sequence[tuple[object, int]],
    root_seed: int,
    jobs: int,
) -> tuple[
    dict[tuple[str, int, str], str],
    dict[tuple[str, int, str], str],
    dict[str, dict[str, _RecordDigest]],
    dict[str, object],
]:
    """Rewrite result rows and spool repaired HHY prediction/curve payloads."""

    output_results = temporary / STREAM_FILENAMES["results"]
    replacement_predictions = temporary / ".replacement_hhy_predictions.csv"
    replacement_curves = temporary / ".replacement_hhy_curves.jsonl"
    source_cohorts, source_status = _load_source_cohorts(
        source_paths=source_paths,
        source_offsets=source_offsets,
        prefix_tasks=prefix_tasks,
        root_seed=root_seed,
    )
    repaired_status: dict[tuple[str, int, str], str] = {}
    streams = {
        name: {
            "unmodified_source": _RecordDigest(),
            "unmodified_output": _RecordDigest(),
            "replaced_old": _RecordDigest(),
            "replaced_new": _RecordDigest(),
        }
        for name in strict.OUTPUT_STREAM_KEYS
    }
    cohort_digest = _RecordDigest()
    by_scenario: dict[str, int] = {}
    repaired_rows = 0
    repaired_successes = 0
    repaired_failures = 0

    repair_futures = [
        (None, scenario, replicate)
        for scenario, replicate, _ in source_cohorts
        if str(getattr(scenario, "example")) == TARGET_EXAMPLE
    ]
    if not repair_futures:
        raise RuntimeError("repair prefix contains no HHY task")
    _configure_worker_threads(jobs)
    repair_results = strict._ordered_task_results(
        repair_futures,
        jobs=int(jobs),
        worker=_repair_hhy_task,
        worker_arguments=lambda scenario, replicate: (scenario, replicate, root_seed),
    )

    with output_results.open("wb") as output_handle, replacement_predictions.open(
        "wb"
    ) as prediction_handle, replacement_curves.open("wb") as curve_handle:
        output_handle.write(strict._csv_bytes(strict.RESULT_FIELDS, [], include_header=True))
        prediction_handle.write(strict._csv_bytes(strict.PREDICTION_FIELDS, [], include_header=True))
        for scenario, replicate, source_cohort in source_cohorts:
            context = f"{getattr(scenario, 'scenario')}/{replicate}"
            source_by_method = {str(row["method"]): row for row in source_cohort}
            output_cohort: list[Mapping[str, object]] = []
            if str(getattr(scenario, "example")) == TARGET_EXAMPLE:
                try:
                    completed_scenario, completed_replicate, task_payload = next(repair_results)
                except StopIteration as error:
                    raise RuntimeError(f"{context}: ordered HHY scheduler ended early") from error
                if (
                    str(getattr(completed_scenario, "scenario")) != str(getattr(scenario, "scenario"))
                    or int(completed_replicate) != int(replicate)
                    or not isinstance(task_payload, Mapping)
                ):
                    raise RuntimeError(f"{context}: ordered HHY scheduler yielded an unexpected cohort")
                identity_payload = task_payload.get("identity")
                if not isinstance(identity_payload, Mapping):
                    raise RuntimeError(f"{context}: HHY worker omitted regenerated identity")
                identity = _validate_regenerated_identity(
                    source_cohort,
                    identity=identity_payload,
                    context=context,
                )
                cohort_payload = strict._canonical_json(
                    {"scenario": str(getattr(scenario, "scenario")), "replicate": replicate, **identity}
                ).encode("utf-8") + b"\n"
                cohort_digest.update(cohort_payload, records=1)
                repaired_row = dict(task_payload.get("row", {}))
                repaired_predictions = sorted(
                    (dict(item) for item in task_payload.get("predictions", [])),
                    key=lambda item: str(item["row_id"]),
                )
                worker_curve = task_payload.get("curve")
                repaired_curve = None if worker_curve is None else dict(worker_curve)
                worker_row_ids = task_payload.get("test_row_ids")
                if not isinstance(worker_row_ids, list):
                    raise RuntimeError(f"{context}: HHY worker omitted registered test row IDs")
                _validate_replacement(
                    repaired_row,
                    repaired_predictions,
                    repaired_curve,
                    scenario=scenario,
                    replicate=replicate,
                    source_cohort=source_cohort,
                    test_row_ids=[str(item) for item in worker_row_ids],
                    root_seed=root_seed,
                )
                target_key = (str(getattr(scenario, "scenario")), replicate, strict.HHY)
                repaired_status[target_key] = str(repaired_row.get("attempt_status", ""))
                repaired_rows += 1
                by_scenario[str(getattr(scenario, "scenario"))] = (
                    by_scenario.get(str(getattr(scenario, "scenario")), 0) + 1
                )
                if repaired_status[target_key] == "success":
                    repaired_successes += 1
                else:
                    repaired_failures += 1
                old_chunk = _csv_chunk([source_by_method[strict.HHY]], strict.RESULT_FIELDS)
                new_chunk = _csv_chunk([repaired_row], strict.RESULT_FIELDS)
                streams["results"]["replaced_old"].update(old_chunk, records=1)
                streams["results"]["replaced_new"].update(new_chunk, records=1)
                if repaired_predictions:
                    prediction_handle.write(
                        _csv_chunk(repaired_predictions, strict.PREDICTION_FIELDS)
                    )
                if repaired_curve is not None:
                    curve_handle.write(strict._jsonl_bytes([repaired_curve]))
            for source_row in source_cohort:
                method = str(source_row["method"])
                if _is_target(scenario, method):
                    if str(getattr(scenario, "example")) != TARGET_EXAMPLE:
                        raise AssertionError("unreachable target classification")
                    output_row = repaired_row
                else:
                    output_row = source_row
                    payload = _csv_chunk([source_row], strict.RESULT_FIELDS)
                    streams["results"]["unmodified_source"].update(payload, records=1)
                    streams["results"]["unmodified_output"].update(payload, records=1)
                output_cohort.append(output_row)
            strict._validate_cohort_rows(
                output_cohort,
                scenario,
                replicate,
                mode="formal",
                root_seed=root_seed,
            )
            output_handle.write(_csv_chunk(output_cohort, strict.RESULT_FIELDS))
        _expect_exhausted(repair_results, stream_name="ordered HHY scheduler")

    if repaired_rows != sum(
        1 for scenario, _ in prefix_tasks if str(getattr(scenario, "example")) == TARGET_EXAMPLE
    ):
        raise RuntimeError("did not repair exactly one HHY row per Example-2 cohort")
    summary = {
        "repaired_cohorts": repaired_rows,
        "successful_repaired_fits": repaired_successes,
        "failed_repaired_fits": repaired_failures,
        "cohort_identity_audit": cohort_digest.as_dict(),
        "repaired_cohorts_by_scenario": dict(sorted(by_scenario.items())),
        "execution": {
            "jobs": int(jobs),
            "ordered_task_scheduler_version": strict.ORDERED_TASK_SCHEDULER_VERSION,
            "max_outstanding_futures": strict._ordered_task_prefetch_limit(int(jobs)),
            "submitted_hhy_cohorts": len(repair_futures),
            "canonical_yield_order": True,
        },
        "replacement_prediction_spool": replacement_predictions.name,
        "replacement_curve_spool": replacement_curves.name,
    }
    return source_status, repaired_status, streams, summary


def _rewrite_predictions(
    *,
    source_paths: Mapping[str, Path],
    source_offsets: Mapping[str, object],
    temporary: Path,
    prefix_tasks: Sequence[tuple[object, int]],
    repaired_status: Mapping[tuple[str, int, str], str],
    streams: Mapping[str, Mapping[str, _RecordDigest]],
) -> None:
    destination = temporary / STREAM_FILENAMES["predictions"]
    replacement = temporary / ".replacement_hhy_predictions.csv"
    with _limited_text(
        source_paths["predictions"], int(source_offsets["predictions"])
    ) as source_handle, replacement.open("r", encoding="utf-8", newline="") as replacement_handle, destination.open(
        "wb"
    ) as output_handle:
        source_groups = _iter_csv_groups(
            source_handle, strict.PREDICTION_FIELDS, stream_name="source predictions"
        )
        replacement_groups = _iter_csv_groups(
            replacement_handle,
            strict.PREDICTION_FIELDS,
            stream_name="replacement HHY predictions",
        )
        positions = _canonical_stream_positions(prefix_tasks)
        source_pending = _next_optional_group(source_groups)
        output_handle.write(strict._csv_bytes(strict.PREDICTION_FIELDS, [], include_header=True))
        for scenario, replicate in prefix_tasks:
            scenario_name = str(getattr(scenario, "scenario"))
            for method in strict.FIXED_METHOD_LABELS:
                key = (scenario_name, replicate, method)
                source_payload, source_pending = _consume_source_group_if_present(
                    source_pending,
                    source_groups,
                    expected=key,
                    positions=positions,
                    stream_name="source predictions",
                )
                source_rows = (
                    []
                    if source_payload is None
                    else [dict(item) for item in source_payload]
                )
                if _is_target(scenario, method):
                    if source_rows:
                        streams["predictions"]["replaced_old"].update(
                            _csv_chunk(source_rows, strict.PREDICTION_FIELDS),
                            records=len(source_rows),
                        )
                    if repaired_status.get(key) == "success":
                        replacement_rows = _next_group(
                            replacement_groups,
                            key,
                            stream_name="replacement HHY predictions",
                        )
                        payload = _csv_chunk(replacement_rows, strict.PREDICTION_FIELDS)
                        output_handle.write(payload)
                        streams["predictions"]["replaced_new"].update(
                            payload, records=len(replacement_rows)
                        )
                elif source_rows:
                    payload = _csv_chunk(source_rows, strict.PREDICTION_FIELDS)
                    output_handle.write(payload)
                    streams["predictions"]["unmodified_source"].update(
                        payload, records=len(source_rows)
                    )
                    streams["predictions"]["unmodified_output"].update(
                        payload, records=len(source_rows)
                    )
        if source_pending is not None:
            raise RuntimeError(
                f"source predictions contains an unexpected trailing group: {source_pending[0]}"
            )
        _expect_exhausted(replacement_groups, stream_name="replacement HHY predictions")


def _rewrite_curves(
    *,
    source_paths: Mapping[str, Path],
    source_offsets: Mapping[str, object],
    temporary: Path,
    prefix_tasks: Sequence[tuple[object, int]],
    repaired_status: Mapping[tuple[str, int, str], str],
    streams: Mapping[str, Mapping[str, _RecordDigest]],
) -> None:
    destination = temporary / STREAM_FILENAMES["curves"]
    replacement = temporary / ".replacement_hhy_curves.jsonl"
    with _limited_text(
        source_paths["curves"], int(source_offsets["curves"])
    ) as source_handle, replacement.open("r", encoding="utf-8") as replacement_handle, destination.open(
        "wb"
    ) as output_handle:
        source_groups = _iter_curve_groups(source_handle, stream_name="source curves")
        replacement_groups = _iter_curve_groups(
            replacement_handle, stream_name="replacement HHY curves"
        )
        positions = _canonical_stream_positions(prefix_tasks)
        source_pending = _next_optional_group(source_groups)
        for scenario, replicate in prefix_tasks:
            scenario_name = str(getattr(scenario, "scenario"))
            for method in strict.FIXED_METHOD_LABELS:
                key = (scenario_name, replicate, method)
                source_payload, source_pending = _consume_source_group_if_present(
                    source_pending,
                    source_groups,
                    expected=key,
                    positions=positions,
                    stream_name="source curves",
                )
                source_records = (
                    []
                    if source_payload is None
                    else [dict(item) for item in source_payload]
                )
                if _is_target(scenario, method):
                    if source_records:
                        streams["curves"]["replaced_old"].update(
                            strict._jsonl_bytes(source_records), records=len(source_records)
                        )
                    if repaired_status.get(key) == "success":
                        replacement_records = _next_group(
                            replacement_groups,
                            key,
                            stream_name="replacement HHY curves",
                        )
                        if len(replacement_records) != 1:
                            raise RuntimeError(
                                f"replacement HHY curves has {len(replacement_records)} records for {key}"
                            )
                        payload = strict._jsonl_bytes(replacement_records)
                        output_handle.write(payload)
                        streams["curves"]["replaced_new"].update(
                            payload, records=len(replacement_records)
                        )
                elif source_records:
                    payload = strict._jsonl_bytes(source_records)
                    output_handle.write(payload)
                    streams["curves"]["unmodified_source"].update(
                        payload, records=len(source_records)
                    )
                    streams["curves"]["unmodified_output"].update(
                        payload, records=len(source_records)
                    )
        if source_pending is not None:
            raise RuntimeError(
                f"source curves contains an unexpected trailing group: {source_pending[0]}"
            )
        _expect_exhausted(replacement_groups, stream_name="replacement HHY curves")


def _validate_unmodified_digests(streams: Mapping[str, Mapping[str, _RecordDigest]]) -> dict[str, object]:
    audit: dict[str, object] = {}
    for stream, partitions in streams.items():
        source = partitions["unmodified_source"].as_dict()
        output = partitions["unmodified_output"].as_dict()
        equal = source == output
        if not equal:
            raise RuntimeError(f"non-HHY {stream} records changed during repair")
        audit[stream] = {
            "unmodified": {"source": source, "output": output, "verified_equal": True},
            "replaced_hhy": {
                "old": partitions["replaced_old"].as_dict(),
                "new": partitions["replaced_new"].as_dict(),
            },
        }
    return audit


def _prefix_replication_map(prefix_tasks: Sequence[tuple[object, int]]) -> tuple[list[object], dict[str, int]]:
    scenarios: list[object] = []
    counts: dict[str, int] = {}
    for scenario, replicate in prefix_tasks:
        name = str(getattr(scenario, "scenario"))
        if name not in counts:
            scenarios.append(scenario)
            counts[name] = 0
        if int(replicate) != counts[name]:
            raise RuntimeError("prefix tasks are not in canonical consecutive-replication order")
        counts[name] += 1
    return scenarios, counts


def _validate_output(
    *,
    temporary: Path,
    prefix_tasks: Sequence[tuple[object, int]],
    root_seed: int,
) -> list[dict[str, object]]:
    paths = _source_paths(temporary)
    rows = strict._load_and_validate_committed_results(
        paths["results"],
        prefix_tasks,
        len(prefix_tasks),
        mode="formal",
        root_seed=root_seed,
    )
    strict._validate_committed_predictions(
        paths["predictions"], prefix_tasks, len(prefix_tasks), rows
    )
    strict._validate_committed_curves(paths["curves"], prefix_tasks, len(prefix_tasks), rows)
    scenarios, replications = _prefix_replication_map(prefix_tasks)
    issues = strict._validate_shared_rows(rows, scenarios, replications)
    if issues:
        raise RuntimeError(f"repaired prefix shared-cohort validation failed: {issues[:5]}")
    return rows


def _output_stream_audit(temporary: Path) -> tuple[dict[str, int], dict[str, str]]:
    paths = _source_paths(temporary)
    offsets = {key: int(path.stat().st_size) for key, path in paths.items()}
    hashes = {key: strict.file_sha256(path) for key, path in paths.items()}
    return offsets, hashes


def _discard_explicit_temporary(*, source: Path, output: Path, temporary: Path) -> None:
    """Remove only the exact sibling temporary named by this invocation."""

    expected = output.with_name(output.name + ".hhy-repairing").resolve()
    resolved = temporary.resolve()
    if (
        resolved != expected
        or not temporary.is_dir()
        or resolved == ROOT.resolve()
        or source == resolved
        or source.is_relative_to(resolved)
    ):
        raise RuntimeError("refusing to remove a non-target repair temporary directory")
    shutil.rmtree(temporary)


def _write_temporary_marker(
    temporary: Path,
    output: Path,
    *,
    status: str,
    stage: str,
    error: BaseException | None = None,
) -> None:
    """Journal the state of an unpublished repair directory for safe recovery."""

    payload: dict[str, object] = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "output": str(output),
        "status": status,
        "stage": stage,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if error is not None:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
    strict._atomic_write_json(temporary / TEMPORARY_MARKER_FILENAME, payload)


def repair(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".hhy-repairing")
    if source == output or source == temporary:
        raise RuntimeError("source and destination must be distinct")
    if output.exists():
        raise RuntimeError("destination already exists")
    if temporary.exists():
        if not bool(args.discard_stale_temporary):
            raise RuntimeError(
                "destination temporary sibling already exists; inspect it, then rerun with "
                "--discard-stale-temporary to remove only that exact target"
            )
        _discard_explicit_temporary(source=source, output=output, temporary=temporary)

    source_progress, source_paths = _validate_source_progress(source)
    source_contract = source_progress.get("run_contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError("source progress lacks its run contract")
    contract, tasks = _current_contract(jobs=args.jobs)
    if strict._canonical_json(source_contract) != strict._canonical_json(contract):
        raise RuntimeError(
            "source prefix contract differs from the current formal source/configuration"
        )
    prefix_tasks = _validate_repair_scope(
        tasks, prefix=int(args.prefix_cohorts), source_progress=source_progress
    )
    source_offsets = source_progress["committed_offsets"]
    source_hashes = source_progress["committed_sha256"]
    if not isinstance(source_offsets, Mapping) or not isinstance(source_hashes, Mapping):
        raise RuntimeError("source progress lacks committed stream audit data")

    adapters = strict.adapter_registry()
    adapter = adapters.get(strict.HHY)
    if adapter is None:
        raise RuntimeError("current adapter registry lacks Three-step M-VCAM")
    preflight = adapter.preflight()
    if not preflight.ready:
        raise RuntimeError(
            f"Three-step M-VCAM preflight is not ready: {preflight.code}: {preflight.message}"
        )
    current_fingerprint = strict._run_fingerprint(contract)
    root_seed = int(contract["root_seed"])

    temporary.mkdir(parents=True)
    _write_temporary_marker(
        temporary,
        output,
        status="running",
        stage="before_hhy_repairs",
    )
    stage = "before_hhy_repairs"
    try:
        upstream = _copy_upstream_manifest(source, temporary)
        _source_status, repaired_status, stream_digests, repair_summary = _rewrite_results_and_spools(
            source_paths=source_paths,
            source_offsets=source_offsets,
            temporary=temporary,
            prefix_tasks=prefix_tasks,
            root_seed=root_seed,
            jobs=int(args.jobs),
        )
        # From here onward every expensive HHY fit has completed and both
        # replacement spools exist.  Preserve this exact directory if a
        # post-compute merge or validation error occurs instead of deleting
        # hundreds or thousands of already-audited repairs.
        stage = "replacement_spools_complete"
        _write_temporary_marker(
            temporary,
            output,
            status="postcompute_merge_pending",
            stage=stage,
        )
        _rewrite_predictions(
            source_paths=source_paths,
            source_offsets=source_offsets,
            temporary=temporary,
            prefix_tasks=prefix_tasks,
            repaired_status=repaired_status,
            streams=stream_digests,
        )
        stage = "predictions_merged"
        _rewrite_curves(
            source_paths=source_paths,
            source_offsets=source_offsets,
            temporary=temporary,
            prefix_tasks=prefix_tasks,
            repaired_status=repaired_status,
            streams=stream_digests,
        )
        stage = "curves_merged"
        result_rows = _validate_output(
            temporary=temporary, prefix_tasks=prefix_tasks, root_seed=root_seed
        )
        stream_audit = _validate_unmodified_digests(stream_digests)
        output_offsets, output_hashes = _output_stream_audit(temporary)
        paths_for_progress = {
            **_source_paths(temporary),
            "metadata": temporary / "strict_metadata.json",
            "metadata_sha256": temporary / "strict_metadata.sha256",
            "progress": temporary / "strict_progress.json",
        }
        progress = strict._progress_template(
            contract=contract,
            fingerprint=current_fingerprint,
            expected_cohorts=len(tasks),
            paths=paths_for_progress,
        )
        last_scenario, last_replicate = _task_key(prefix_tasks[-1])
        progress.update(
            committed_cohorts=len(prefix_tasks),
            committed_offsets=output_offsets,
            committed_sha256=output_hashes,
            last_completed={
                "phase": getattr(prefix_tasks[-1][0], "phase"),
                "scenario": last_scenario,
                "replicate": last_replicate,
                "repair": {
                    "schema_version": REPAIR_SCHEMA_VERSION,
                    "method": strict.HHY,
                    "example": TARGET_EXAMPLE,
                    "repaired_cohorts": repair_summary["repaired_cohorts"],
                },
            },
            status="running",
            updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        strict._atomic_write_json(paths_for_progress["progress"], progress)

        target_scenarios: dict[str, object] = {}
        for scenario, _ in prefix_tasks:
            if str(getattr(scenario, "example")) == TARGET_EXAMPLE:
                target_scenarios.setdefault(str(getattr(scenario, "scenario")), scenario)
        lineage = {
            "schema_version": REPAIR_SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repair_scope": {
                "method": strict.HHY,
                "method_display_name": strict.METHOD_SPECS[strict.HHY].display_name,
                "examples": [TARGET_EXAMPLE],
                "prefix_cohorts": len(prefix_tasks),
                "repair_policy": (
                    "Regenerate the registered DGP and subject split for every Example-2 "
                    "cohort and replace only Three-step M-VCAM result/prediction/curve records."
                ),
            },
            "source_progress": {
                "path": str(source),
                "progress_sha256": strict.file_sha256(source / "strict_progress.json"),
                "run_fingerprint": source_progress.get("run_fingerprint"),
                "committed_cohorts": source_progress.get("committed_cohorts"),
                "expected_cohorts": source_progress.get("expected_cohorts"),
                "committed_offsets": dict(source_offsets),
                "committed_sha256": dict(source_hashes),
                "run_contract_source_sha256": source_contract.get("source_sha256"),
            },
            "upstream_migration": upstream,
            "current_dependencies": {
                "run_fingerprint": current_fingerprint,
                "run_contract_source_sha256": contract.get("source_sha256"),
                "repair_script_sha256": strict.file_sha256(Path(__file__)),
                "adapter_preflight": asdict(preflight),
                "hhy_tuning_by_scenario": {
                    str(getattr(scenario, "scenario")): {
                        "tuning_sha256": strict._sha256_bytes(
                            strict._canonical_json(
                                strict._default_tuning(strict.HHY, scenario, quick=False)
                            ).encode("utf-8")
                        ),
                        "tuning": strict._default_tuning(strict.HHY, scenario, quick=False),
                    }
                    for scenario_name, scenario in sorted(target_scenarios.items())
                },
            },
            "cohort_identity_validation": repair_summary,
            "stream_record_audit": stream_audit,
            "output": {
                "path": str(output),
                "committed_offsets": output_offsets,
                "committed_sha256": output_hashes,
                "validated_result_rows": len(result_rows),
                "validated_prediction_curve_streams": True,
                "run_contract_source_sha256": contract.get("source_sha256"),
            },
        }
        lineage_path = temporary / REPAIR_LINEAGE_FILENAME
        strict._atomic_write_json(lineage_path, lineage)
        progress = dict(progress)
        progress.update(
            repair_lineage_filename=lineage_path.name,
            repair_lineage_sha256=strict.file_sha256(lineage_path),
            updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        strict._atomic_write_json(paths_for_progress["progress"], progress)
        stage = "validation_complete"
        (temporary / ".replacement_hhy_predictions.csv").unlink()
        (temporary / ".replacement_hhy_curves.jsonl").unlink()
        # This marker belongs only to the unpublished sibling directory.  Do
        # not leave it in an otherwise complete, atomically-published result.
        (temporary / TEMPORARY_MARKER_FILENAME).unlink()
        temporary.replace(output)
    except BaseException as error:
        if stage in {
            "replacement_spools_complete",
            "predictions_merged",
            "curves_merged",
            "validation_complete",
        }:
            try:
                _write_temporary_marker(
                    temporary,
                    output,
                    status="postcompute_failed_preserved",
                    stage=stage,
                    error=error,
                )
            except OSError:
                # Do not replace the original repair/merge failure with a
                # secondary diagnostic-write error.  The spool files remain
                # intact as long as this directory remains present.
                pass
            print(
                "[repair-preserved] "
                + json.dumps(
                    {
                        "temporary": str(temporary),
                        "stage": stage,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument(
        "--prefix-cohorts",
        type=int,
        default=4100,
        help="complete freshly migrated Example-1/2 prefix (formal default: 4100)",
    )
    parser.add_argument(
        "--discard-stale-temporary",
        action="store_true",
        help="remove only this output's exact stale .hhy-repairing sibling before starting",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.prefix_cohorts < 1:
        parser.error("--prefix-cohorts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    output = repair(parse_args(argv))
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

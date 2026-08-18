"""Seed a revised strict run with an auditable completed prefix.

This utility is deliberately narrow.  It copies only whole, canonically
registered cohorts from an interrupted strict run into a new output directory,
then writes a *new* progress journal made from the current scientific contract.
It is intended for a source change that is proved not to affect that prefix
(for example, an Example-3-only high-dimensional tuning correction).

The source remains untouched.  A JSON manifest in the destination records the
old fingerprint, committed byte hashes, selected cohort boundary, and the new
fingerprint.  The normal strict runner validates the destination again before
skipping any copied cohort.
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
from typing import Iterator, Mapping, Sequence

import run_strict_benchmark as strict


class _PrefixRawReader(io.RawIOBase):
    """Expose exactly the committed byte prefix of a file to TextIOWrapper."""

    def __init__(self, path: Path, limit: int) -> None:
        self._handle = path.open("rb")
        self._remaining = int(limit)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        if self._remaining <= 0:
            return 0
        data = self._handle.read(min(len(buffer), self._remaining))
        count = len(data)
        buffer[:count] = data
        self._remaining -= count
        return count

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


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


def _current_contract(*, jobs: int) -> tuple[dict[str, object], list[object]]:
    trace_lock = strict.load_trace_tuning_lock()
    scenarios = strict.registered_scenarios(quick=False, include_reproduction_audit=False)
    replications = {
        scenario.scenario: scenario.formal_replications for scenario in scenarios
    }
    targets = strict.load_published_targets(strict.ROOT / "protocol" / "published_targets.json")
    adapters = strict.adapter_registry()
    preflight = {
        method: asdict(adapters[method].preflight())
        for method in strict.FIXED_METHOD_LABELS
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


def _noncode_contract(contract: Mapping[str, object]) -> dict[str, object]:
    ignored = {"source_sha256"}
    return {key: value for key, value in contract.items() if key not in ignored}


def _task_key(task: tuple[object, int]) -> tuple[str, int]:
    scenario, replicate = task
    return str(getattr(scenario, "scenario")), int(replicate)


def _copy_csv_prefix(
    source: Path,
    limit: int,
    destination: Path,
    fields: Sequence[str],
    wanted: set[tuple[str, int]],
    *,
    collect_last: tuple[str, int],
) -> tuple[int, list[dict[str, str]]]:
    count = 0
    last: list[dict[str, str]] = []
    with _limited_text(source, limit) as input_handle, destination.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if list(reader.fieldnames or []) != list(fields):
            raise RuntimeError(f"unexpected CSV header in {source}")
        writer = csv.DictWriter(output_handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in reader:
            key = (str(row.get("scenario", "")), int(row.get("replicate", "-1")))
            if key not in wanted:
                continue
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
            if key == collect_last:
                last.append(dict(row))
    return count, last


def _copy_curves_prefix(
    source: Path,
    limit: int,
    destination: Path,
    wanted: set[tuple[str, int]],
    *,
    collect_last: tuple[str, int],
) -> tuple[int, list[dict[str, object]]]:
    count = 0
    last: list[dict[str, object]] = []
    with _limited_text(source, limit) as input_handle, destination.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (str(payload.get("scenario", "")), int(payload.get("replicate", -1)))
            if key not in wanted:
                continue
            output_handle.write(line if line.endswith("\n") else line + "\n")
            count += 1
            if key == collect_last:
                last.append(payload)
    return count, last


def _validate_source_progress(source: Path) -> dict[str, object]:
    progress_path = source / "strict_progress.json"
    if not progress_path.is_file():
        raise RuntimeError(f"source has no strict progress journal: {progress_path}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("schema_version") != strict.PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("source progress has an unsupported schema")
    offsets = progress.get("committed_offsets")
    hashes = progress.get("committed_sha256")
    if not isinstance(offsets, Mapping) or not isinstance(hashes, Mapping):
        raise RuntimeError("source progress lacks committed offsets/hashes")
    for key in strict.OUTPUT_STREAM_KEYS:
        path = source / {
            "results": "strict_results.csv",
            "predictions": "strict_predictions.csv",
            "curves": "strict_factor_curves.jsonl",
        }[key]
        offset = int(offsets.get(key, -1))
        if offset < 0 or not path.is_file() or path.stat().st_size < offset:
            raise RuntimeError(f"source {key} does not contain its committed prefix")
        observed = _prefix_sha256(path, offset)
        if observed != str(hashes.get(key, "")):
            raise RuntimeError(f"source committed {key} prefix hash mismatch")
    return dict(progress)


def migrate(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".migrating")
    if output.exists() or temporary.exists():
        raise RuntimeError("destination or its temporary sibling already exists")
    if source == output or source == temporary:
        raise RuntimeError("source and destination must be distinct")

    source_progress = _validate_source_progress(source)
    contract, registered_tasks = _current_contract(jobs=args.jobs)
    current_fingerprint = strict._run_fingerprint(contract)
    source_contract = source_progress.get("run_contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError("source progress lacks its run contract")
    if strict._canonical_json(_noncode_contract(source_contract)) != strict._canonical_json(
        _noncode_contract(contract)
    ):
        raise RuntimeError("the source and current contracts differ outside source hashes")

    prefix = int(args.prefix_cohorts)
    committed = int(source_progress.get("committed_cohorts", -1))
    if not 1 <= prefix <= committed <= len(registered_tasks):
        raise RuntimeError("prefix must be positive, committed, and registered")
    wanted_tasks = registered_tasks[:prefix]
    following = registered_tasks[prefix:]
    if not following:
        raise RuntimeError("the requested prefix leaves no work for the revised run")
    wanted = {_task_key(task) for task in wanted_tasks}
    if len(wanted) != prefix:
        raise RuntimeError("registered prefix has duplicate scenario/replication keys")
    last_key = _task_key(wanted_tasks[-1])
    # The migration is only meaningful when it stops at a scenario boundary.
    if _task_key(following[0])[0] == last_key[0]:
        raise RuntimeError("prefix must end at a whole scenario boundary")

    source_paths = {
        "results": source / "strict_results.csv",
        "predictions": source / "strict_predictions.csv",
        "curves": source / "strict_factor_curves.jsonl",
    }
    destination_paths = {
        "results": temporary / "strict_results.csv",
        "predictions": temporary / "strict_predictions.csv",
        "curves": temporary / "strict_factor_curves.jsonl",
        "progress": temporary / "strict_progress.json",
    }
    offsets = source_progress["committed_offsets"]

    temporary.mkdir(parents=True)
    try:
        result_count, last_results = _copy_csv_prefix(
            source_paths["results"],
            int(offsets["results"]),
            destination_paths["results"],
            strict.RESULT_FIELDS,
            wanted,
            collect_last=last_key,
        )
        prediction_count, last_predictions = _copy_csv_prefix(
            source_paths["predictions"],
            int(offsets["predictions"]),
            destination_paths["predictions"],
            strict.PREDICTION_FIELDS,
            wanted,
            collect_last=last_key,
        )
        curve_count, last_curves = _copy_curves_prefix(
            source_paths["curves"],
            int(offsets["curves"]),
            destination_paths["curves"],
            wanted,
            collect_last=last_key,
        )
        if result_count != prefix * len(strict.FIXED_METHOD_LABELS):
            raise RuntimeError("filtered result count is not one complete common cohort per prefix task")

        paths_for_validation = {
            **destination_paths,
            "metadata": temporary / "strict_metadata.json",
            "metadata_sha256": temporary / "strict_metadata.sha256",
        }
        rows = strict._load_and_validate_committed_results(
            destination_paths["results"],
            registered_tasks,
            prefix,
            mode="formal",
            root_seed=strict.DEFAULT_ROOT_SEED,
        )
        strict._validate_committed_predictions(
            destination_paths["predictions"], registered_tasks, prefix, rows
        )
        strict._validate_committed_curves(
            destination_paths["curves"], registered_tasks, prefix, rows
        )

        new_offsets = {
            key: int(destination_paths[key].stat().st_size)
            for key in strict.OUTPUT_STREAM_KEYS
        }
        new_hashes = {
            key: strict.file_sha256(destination_paths[key])
            for key in strict.OUTPUT_STREAM_KEYS
        }
        progress = strict._progress_template(
            contract=contract,
            fingerprint=current_fingerprint,
            expected_cohorts=len(registered_tasks),
            paths=paths_for_validation,
        )
        progress.update(
            committed_cohorts=prefix,
            committed_offsets=new_offsets,
            committed_sha256=new_hashes,
            last_completed={
                "phase": getattr(wanted_tasks[-1][0], "phase"),
                "scenario": last_key[0],
                "replicate": last_key[1],
                "result_rows": len(last_results),
                "prediction_rows": len(last_predictions),
                "curve_rows": len(last_curves),
                "migration": "verified-low-dimensional-prefix/v1",
            },
            status="running",
            updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        strict._atomic_write_json(destination_paths["progress"], progress)
        manifest = {
            "schema_version": "vcam-strict-prefix-migration/1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": (
                "The revised source changes only the documented high-dimensional "
                "Backfitting VCAM tuning branch.  This verified prefix contains "
                "only completed Example 1--2 cohorts and is revalidated against "
                "the current immutable registry before resume."
            ),
            "source": {
                "path": str(source),
                "run_fingerprint": source_progress.get("run_fingerprint"),
                "committed_cohorts": committed,
                "committed_offsets": source_progress.get("committed_offsets"),
                "committed_sha256": source_progress.get("committed_sha256"),
                "source_sha256": source_contract.get("source_sha256"),
            },
            "destination": {
                "path": str(output),
                "run_fingerprint": current_fingerprint,
                "prefix_cohorts": prefix,
                "last_prefix_task": {"scenario": last_key[0], "replicate": last_key[1]},
                "copied_rows": {
                    "results": result_count,
                    "predictions": prediction_count,
                    "curves": curve_count,
                },
                "committed_offsets": new_offsets,
                "committed_sha256": new_hashes,
                "source_sha256": contract.get("source_sha256"),
            },
        }
        strict._atomic_write_json(temporary / "strict_prefix_migration.json", manifest)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-cohorts", type=int, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    output = migrate(parse_args(argv))
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

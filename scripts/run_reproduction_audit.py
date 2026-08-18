"""Run only the locked source-paper reproduction gates.

This preflight is intentionally separate from the common benchmark.  It lets
an implementation or source-protocol mismatch be diagnosed before thousands
of shared-cohort fits are authorized.  The common runner still repeats these
gates in its final immutable result bundle.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_benchmark as strict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=strict.DEFAULT_ROOT_SEED)
    parser.add_argument(
        "--published-targets",
        type=Path,
        default=strict.ROOT / "protocol" / "published_targets.json",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def execute(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "reproduction_results.csv"
    gates_path = output / "reproduction_gates.json"
    metadata_path = output / "reproduction_metadata.json"
    targets = strict.load_published_targets(args.published_targets.resolve())
    scenarios = [
        item
        for item in strict.registered_scenarios(
            quick=False, include_reproduction_audit=True
        )
        if item.phase == "reproduction"
    ]
    replications = {
        item.scenario: int(item.formal_replications) for item in scenarios
    }
    if args.jobs > 1:
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
    tasks = [
        (None, scenario, replicate)
        for scenario in scenarios
        for replicate in range(replications[scenario.scenario])
    ]
    rows: list[dict[str, object]] = []
    started = time.time()
    for scenario, replicate, result in strict._ordered_task_results(
        tasks,
        jobs=int(args.jobs),
        worker=strict._reproduction_task,
        worker_arguments=lambda item, index: (
            item,
            index,
            int(args.seed),
            "formal-reproduction-audit",
            False,
            targets.get(item.scenario),
        ),
    ):
        row, _, _ = result
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == len(tasks):
            strict._write_csv(results_path, strict.RESULT_FIELDS, rows)
            print(
                f"[reproduction] {len(rows)}/{len(tasks)} "
                f"{scenario.scenario} replicate={replicate} "
                f"status={row['attempt_status']}",
                flush=True,
            )
    scenario_index = {
        scenario.scenario: index for index, scenario in enumerate(scenarios)
    }
    rows.sort(
        key=lambda row: (
            scenario_index[str(row["scenario"])],
            int(row["replicate"]),
            str(row["method"]),
        )
    )
    strict._write_csv(results_path, strict.RESULT_FIELDS, rows)
    gates = {
        str(scenario.owner): strict.assess_reproduction_gate(
            rows,
            scenario,
            targets.get(scenario.scenario),
            expected_replications=replications[scenario.scenario],
        )
        for scenario in scenarios
    }
    with gates_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(gates, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    metadata = {
        "schema_version": "vcam-reproduction-audit/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "root_seed": int(args.seed),
        "jobs": int(args.jobs),
        "replications": replications,
        "gates": gates,
        "targets": targets,
        "source_sha256": strict._source_hashes(),
        "results_sha256": strict.file_sha256(results_path),
        "targets_sha256": strict.file_sha256(args.published_targets.resolve()),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "results": results_path,
        "gates": gates_path,
        "metadata": metadata_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    paths = execute(parse_args(argv))
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

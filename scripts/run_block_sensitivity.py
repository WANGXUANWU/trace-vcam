"""Sensitivity of the block-sparse comparison to the configuration it is run at.

Example 3 of the paper is one point of a design family: six candidate blocks,
two of them active, neighbouring blocks correlated at one half, and the active
components at their registered amplitude.  That point is favourable to a
blockwise penalty, so the comparison it supports is only as informative as the
neighbourhood around it.

This runner moves one axis at a time away from that point -- how many blocks are
active, how strongly the blocks are correlated, and how large the active signal
is -- and reruns the proposed estimator against the coefficientwise penalised
competitor, which is the estimator the contrast is actually about.  The other
two published competitors are outside the point of this experiment and are not
attempted here; the full four-method comparison at the registered point stays in
the main paper.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_benchmark import (  # noqa: E402
    DEFAULT_ROOT_SEED,
    RESULT_FIELDS,
    Scenario,
    _registered_split,
    _safe_applicability,
    _stable_seed,
    _subject_dataset,
    adapter_registry,
    run_one_method,
)

#: The two estimators the block-versus-coefficientwise contrast is between.
SWEEP_METHODS = ("TRACE-VCAM", "ZY2025-paper-implementation")

#: The registered Example 3 configuration, which every axis is swept around.
BASE = {"n_covariates": 6, "n_active": 2, "correlation_base": 0.5, "signal_scale": 1.0}

PROTOCOL = {
    "gaussian": "example-4/robust-normal",
    "hhy-mixed-normal": "example-4/robust-mixed-normal",
}


def sweep_points() -> list[tuple[str, str, dict]]:
    """One (axis, label, override) triple per configuration in the sweep."""

    points: list[tuple[str, str, dict]] = []
    for value in (1, 2, 3, 4, 6):
        points.append(("sparsity", f"s{value}", {"n_active": value}))
    for value in (0.0, 0.3, 0.7, 0.9):
        points.append(("correlation", f"rho{value:g}", {"correlation_base": value}))
    for value in (0.25, 0.5, 2.0):
        points.append(("signal", f"a{value:g}", {"signal_scale": value}))
    return points


def scenarios(replications: int, n_subjects: int, laws: tuple[str, ...]) -> list[Scenario]:
    out: list[Scenario] = []
    for law in laws:
        for axis, label, override in sweep_points():
            parameters = dict(BASE)
            parameters.update(override)
            out.append(
                Scenario(
                    f"blocksweep-{axis}-{label}-{law}-n{n_subjects}",
                    "common",
                    "Block sensitivity",
                    PROTOCOL[law],
                    "block_sparse",
                    {
                        "n_subjects": n_subjects,
                        "sigma": 0.4,
                        "error_distribution": law,
                        **parameters,
                    },
                    replications,
                )
            )
    return out


def _task(payload):
    scenario, replicate, root_seed = payload
    seed = _stable_seed(root_seed, scenario.scenario, replicate, "data")
    split_seed = _stable_seed(root_seed, scenario.scenario, replicate, "subject-split")
    raw = scenario.build(seed)
    dataset = _subject_dataset(raw, scenario)
    train, test, split = _registered_split(dataset, raw, scenario, split_seed=split_seed)
    adapters = adapter_registry()
    rows = []
    for method in SWEEP_METHODS:
        applicability, reason = _safe_applicability(method, scenario)
        try:
            row, _predictions, _curves = run_one_method(
                adapters[method],
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
        except Exception as error:  # pragma: no cover - a source method may fail
            row = {field: "" for field in RESULT_FIELDS}
            row.update(
                scenario=scenario.scenario,
                example=scenario.example,
                replicate=replicate,
                seed=seed,
                split_seed=split_seed,
                method=method,
                applicability="applicable",
                attempt_status="failed",
                failure_code="runner_exception",
                failure_message=f"{type(error).__name__}: {error}"[:500],
            )
        row["sweep_axis"] = scenario.scenario.split("-")[1]
        row["sweep_label"] = scenario.scenario.split("-")[2]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--subjects", type=int, default=100)
    parser.add_argument("--laws", nargs="+", default=["hhy-mixed-normal"])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "block_sensitivity"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    chosen = scenarios(args.replications, args.subjects, tuple(args.laws))
    tasks = [
        (scenario, replicate, args.seed)
        for scenario in chosen
        for replicate in range(args.replications)
    ]
    print(f"{len(tasks)} cohorts across {len(chosen)} configurations", flush=True)

    fields = list(RESULT_FIELDS) + ["sweep_axis", "sweep_label"]
    target = args.output / "block_sensitivity_results.csv"
    started = time.perf_counter()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for index, rows in enumerate(pool.map(_task, tasks, chunksize=1), start=1):
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fields})
                handle.flush()
                if index % 10 == 0 or index == len(tasks):
                    print(
                        f"{index}/{len(tasks)} ({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

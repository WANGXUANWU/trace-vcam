"""Where the contamination enters, and whether the number of visits is informative.

The robustness comparison of the paper contaminates the response one visit at a
time.  That is the channel a bounded-influence loss on the residual is built to
control, so reporting only that channel risks reading the result more widely
than it is meant.  This runner keeps the sparse-longitudinal Example 2 design
fixed and moves the contamination instead -- to complete subjects, to whole
trajectories, and to the covariates themselves -- and separately makes the
number of visits depend on the subject's covariate level and latent trajectory,
which is the regime the subject-balanced weighting is written for.

It uses the same per-replication protocol as the strict benchmark: one generated
data set per replication, one subject-level 80/20 split shared by every
applicable method, the same identification map, and the same failure accounting.
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

from benchmarks import FIXED_METHOD_LABELS  # noqa: E402
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

#: (scenario suffix, protocol, contamination channel, cluster-size mechanism).
#: The clean rows are the common reference every contaminated row is read
#: against, and the two informative-cluster-size rows are paired with their own
#: exchangeable counterparts so that the weighting effect is not confounded with
#: the contamination effect.
SCOPE_SETTINGS = (
    ("clean", "scope/clean", "none", "exchangeable"),
    ("response", "scope/response-contamination", "response", "exchangeable"),
    ("subject", "scope/subject-contamination", "subject", "exchangeable"),
    ("trajectory", "scope/trajectory-contamination", "trajectory", "exchangeable"),
    ("leverage", "scope/leverage-contamination", "leverage", "exchangeable"),
    (
        "informative-clean",
        "scope/informative-cluster-size-clean",
        "none",
        "informative",
    ),
    (
        "informative-subject",
        "scope/informative-cluster-size-subject",
        "subject",
        "informative",
    ),
)


def scenarios(replications: int, sample_sizes: tuple[int, ...]) -> list[Scenario]:
    return [
        Scenario(
            f"scope-{suffix}-n{n}",
            "common",
            "Robustness scope",
            protocol,
            "robustness_scope",
            {
                "n_subjects": n,
                "sigma": 0.4,
                "contamination": channel,
                "contamination_rate": 0.05,
                "cluster_size": cluster,
            },
            replications,
        )
        for n in sample_sizes
        for suffix, protocol, channel, cluster in SCOPE_SETTINGS
    ]


def _task(payload):
    scenario, replicate, root_seed = payload
    seed = _stable_seed(root_seed, scenario.scenario, replicate, "data")
    split_seed = _stable_seed(root_seed, scenario.scenario, replicate, "subject-split")
    raw = scenario.build(seed)
    dataset = _subject_dataset(raw, scenario)
    train, test, split = _registered_split(dataset, raw, scenario, split_seed=split_seed)
    adapters = adapter_registry()
    rows = []
    for method in FIXED_METHOD_LABELS:
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
                # Every method here passed the admission gate in the strict
                # benchmark; these protocols move the contamination channel, not
                # the sampling regime, so that decision carries over.
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
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100])
    parser.add_argument(
        "--settings", nargs="+", default=[item[0] for item in SCOPE_SETTINGS]
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "robustness_scope"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    wanted = set(args.settings)
    chosen = [
        scenario
        for scenario in scenarios(args.replications, tuple(args.sizes))
        if scenario.scenario.split("-n")[0].removeprefix("scope-") in wanted
    ]
    tasks = [
        (scenario, replicate, args.seed)
        for scenario in chosen
        for replicate in range(args.replications)
    ]
    print(f"{len(tasks)} cohorts across {len(chosen)} scenarios", flush=True)

    target = args.output / "scope_results.csv"
    started = time.perf_counter()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDS))
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for index, rows in enumerate(pool.map(_task, tasks, chunksize=1), start=1):
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
                handle.flush()
                if index % 10 == 0 or index == len(tasks):
                    print(
                        f"{index}/{len(tasks)} ({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

"""Cache the Example 2 factor curves needed by the revised main-text figures.

The strict curve stream is large and is read once.  This writes one compressed
archive per (scenario, method) so that figure work does not rescan it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

COMPONENTS = ("baseline", "beta_1", "phi_1", "beta_2", "phi_2")


def admitted_replications(
    results: Path, scenarios: set[str]
) -> set[tuple[str, str, int]] | None:
    """Scenario, method, and replicate triples the audit scored as successful."""

    if not results.exists():
        return None
    import pandas as pd

    admitted: set[tuple[str, str, int]] = set()
    columns = ["scenario", "method", "replicate", "attempt_status"]
    for chunk in pd.read_csv(results, usecols=columns, chunksize=500_000):
        chunk = chunk[
            chunk.scenario.isin(scenarios) & (chunk.attempt_status == "success")
        ]
        admitted.update(
            (str(row.scenario), str(row.method), int(row.replicate))
            for row in chunk.itertuples()
        )
    return admitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves",
        type=Path,
        default=ROOT / "results" / "strict_formal_v2_repaired" / "strict_factor_curves.jsonl",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "strict_formal_v2_repaired" / "strict_results.csv",
        help=(
            "Audited result stream used to admit replications.  The curve stream "
            "can contain a replication that the audit scored as a failure, and a "
            "figure must summarise the same replications as the table beside it."
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "example2_curve_cache")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "example2-mixed-normal-n50",
            "example2-mixed-normal-n100",
            "example2-mixed-normal-n200",
            "example2-t2-n50",
            "example2-t2-n200",
            "example2-gaussian-n50-sigma0.4",
        ],
    )
    args = parser.parse_args()

    wanted = set(args.scenarios)
    args.output.mkdir(parents=True, exist_ok=True)
    admitted = admitted_replications(args.results, wanted)

    grids: dict[tuple[str, str, str], np.ndarray] = {}
    stacks: dict[tuple[str, str, str], list[np.ndarray]] = {}
    # Replication labels are kept alongside the curves so that a figure can show
    # several methods on one and the same generated data set, which is how the
    # protocol runs them, rather than on whatever replication happens to occupy
    # the same row of each method's stack.
    labels: dict[tuple[str, str, str], list[int]] = {}
    rejected = 0

    with args.curves.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            # Cheap prefilter: the scenario name appears verbatim in the record.
            if not any(name in line for name in wanted):
                continue
            record = json.loads(line)
            scenario = str(record.get("scenario"))
            if scenario not in wanted:
                continue
            method = str(record.get("method"))
            if admitted is not None:
                key = (scenario, method, int(record["replicate"]))
                if key not in admitted:
                    rejected += 1
                    continue
            for curve in record.get("curves", ()):
                name = str(curve.get("component", ""))
                if name not in COMPONENTS:
                    continue
                grid = np.asarray(curve.get("grid", ()), dtype=float)
                values = np.asarray(curve.get("values", ()), dtype=float)
                if grid.size < 2 or grid.shape != values.shape:
                    continue
                key = (scenario, method, name)
                if key not in grids:
                    grids[key] = grid
                    stacks[key] = []
                    labels[key] = []
                stacks[key].append(np.interp(grids[key], grid, values))
                labels[key].append(int(record["replicate"]))

    payload: dict[str, np.ndarray] = {}
    for (scenario, method, name), stack in stacks.items():
        tag = f"{scenario}|{method}|{name}"
        payload[f"grid::{tag}"] = grids[(scenario, method, name)]
        payload[f"stack::{tag}"] = np.vstack(stack)
        payload[f"replicate::{tag}"] = np.asarray(labels[(scenario, method, name)], dtype=int)
    target = args.output / "example2_curves.npz"
    np.savez_compressed(target, **payload)
    print(f"wrote {target} with {len(stacks)} curve stacks")
    print(f"curve records rejected as unaudited: {rejected}")
    for key in sorted(stacks):
        print(key, np.vstack(stacks[key]).shape)


if __name__ == "__main__":
    main()

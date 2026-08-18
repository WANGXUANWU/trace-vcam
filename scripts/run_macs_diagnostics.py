"""Two diagnostics for the CD4 application, on the fits it already reports.

The first is the realised normalisation margin of each retained block, the
quantity the integral scale divides by, which tells a reader whether the
reported factors are on a well-conditioned scale.  The second is the held-out
rank comparison recommended in the main paper, run on the same subject-level
folds the application already forms, which asks whether the rank-one restriction
costs anything on these data.

Neither refits a number that appears in a reported table: both read the same
data and the same registered tuning rule and report quantities alongside them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID_SIZE = 201


def _subset(dataset, rows: np.ndarray):
    from benchmarks.data import SubjectDataset

    return SubjectDataset(
        time=dataset.time[rows],
        covariates=dataset.covariates[rows],
        response=dataset.response[rows],
        subject_id=dataset.subject_id[rows],
        row_id=None if dataset.row_id is None else dataset.row_id[rows],
        noise_free_target=None,
        covariate_names=dataset.covariate_names,
        metadata=dict(dataset.metadata),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "macs_diagnostics"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from scripts.run_macs_application import (
        _tuning,
        adapter_registry,
        prepare_macs_variant,
        read_macs_csv,
    )
    from scripts.review_common import build_design, fit_delivered, locked_tuning, rank_r_refit
    from src.trace_vcam import OrthonormalSplineBasis, predict_components

    dataset = prepare_macs_variant(
        read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv"), variant="primary"
    )
    adapter = adapter_registry()["TRACE-VCAM"]
    artifact = adapter.fit(dataset, seed=args.seed, tuning=_tuning("TRACE-VCAM", 6, quick=False))
    realised = dict(getattr(artifact, "tuning", {}))
    basis_dimension = int(realised.get("q_time", 6))
    lambda_ratio = float(realised.get("lambda_ratio", locked_tuning()["lambda_ratio"]))
    roughness = float(realised.get("roughness", locked_tuning()["roughness"]))
    huber = float(realised.get("huber_multiplier", 1.345))

    basis = OrthonormalSplineBasis.create(basis_dimension, basis_dimension)
    tuning = locked_tuning()
    full_design = build_design(
        dataset.time, dataset.covariates, dataset.response, dataset.subject_id,
        basis, (0.0, 1.0),
    )
    delivered = fit_delivered(
        full_design, basis, tuning=tuning,
        lambda_ratio=lambda_ratio, roughness=roughness,
    )
    payload: dict[str, object] = {
        "realised_tuning": {
            "q_time": basis_dimension,
            "lambda_ratio": lambda_ratio,
            "roughness": roughness,
            "huber_multiplier": huber,
        },
        "selected_blocks": [int(i) for i in np.flatnonzero(delivered.selected)],
        "normalisation_margin": {
            f"block_{index + 1}": (
                float(value) if np.isfinite(value) else None
            )
            for index, value in enumerate(delivered.time_integral_margin)
        },
    }

    # Held-out rank comparison on complete-subject folds.
    unique = np.unique(dataset.subject_id)
    order = np.random.default_rng(args.seed).permutation(unique)
    assignment = {name: index % args.folds for index, name in enumerate(order)}
    membership = np.asarray([assignment[name] for name in dataset.subject_id])
    per_rank: dict[int, list[float]] = {1: [], 2: [], 3: []}
    ratios: list[float] = []
    for fold in range(args.folds):
        train = _subset(dataset, np.flatnonzero(membership != fold))
        test = _subset(dataset, np.flatnonzero(membership == fold))
        train_design = build_design(
            train.time, train.covariates, train.response, train.subject_id,
            basis, (0.0, 1.0),
        )
        test_design = build_design(
            test.time, test.covariates, test.response, test.subject_id,
            basis, (0.0, 1.0),
        )
        pilot = fit_delivered(
            train_design, basis, tuning=tuning, lambda_ratio=lambda_ratio,
            roughness=roughness, post_rank_one=False,
        )
        for index, values in enumerate(pilot.singular_values):
            if pilot.selected[index] and values[0] > 0.0 and values.size > 1:
                ratios.append(float(values[1] / values[0]))
        for rank in (1, 2, 3):
            gamma, blocks, _ = rank_r_refit(
                train_design, pilot.gamma, pilot.blocks, pilot.selected,
                pilot.delta, rank,
            )
            prediction = predict_components(test_design, gamma, blocks)[0]
            per_rank[rank].append(float(np.mean((prediction - test.response) ** 2)))
    base = float(np.mean(per_rank[1]))
    payload["rank_diagnostic"] = {
        "folds": args.folds,
        "spectral_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "spectral_ratio_max": float(np.max(ratios)) if ratios else None,
        **{f"mspe_rank{r}": float(np.mean(v)) for r, v in per_rank.items()},
        **{
            f"rank{r}_gain": float((base - np.mean(v)) / base)
            for r, v in per_rank.items() if r != 1
        },
    }
    (args.output / "macs_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

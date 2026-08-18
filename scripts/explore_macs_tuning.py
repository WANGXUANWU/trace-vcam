"""Exploratory nested-CV probe for the TRACE tuning rule on the MACS/CD4 data.

This script is a design probe only.  It never writes manuscript output.  It
answers one question: how much held-out prediction error does the simulation
locked penalty pair leave on the table relative to a data-driven subject-level
cross-validated choice on the same folds used by the formal application?
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.adapters.trace import TraceVCAMAdapter  # noqa: E402
from benchmarks.data import SubjectDataset  # noqa: E402
from scripts.run_macs_application import (  # noqa: E402
    prepare_macs_variant,
    read_macs_csv,
)


def subject_folds(subjects: np.ndarray, *, n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    return [shuffled[index::n_folds] for index in range(n_folds)]


def take(dataset: SubjectDataset, mask: np.ndarray) -> SubjectDataset:
    return SubjectDataset(
        time=dataset.time[mask],
        covariates=dataset.covariates[mask],
        response=dataset.response[mask],
        subject_id=dataset.subject_id[mask],
        row_id=dataset.row_id[mask],
        noise_free_target=None,
        covariate_names=dataset.covariate_names,
        metadata=dataset.metadata,
    )


def subject_balanced_mse(subject: np.ndarray, residual: np.ndarray) -> float:
    return float(
        np.mean([float(np.mean(residual[subject == s] ** 2)) for s in np.unique(subject)])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args()

    raw = read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv")
    dataset = prepare_macs_variant(raw, variant="primary")
    adapter = TraceVCAMAdapter()

    lambda_ratios = [0.08, 0.2, 0.35, 0.6]
    roughness = [0.05, 0.5]
    bases = [4, 5, 6]
    multipliers = [1.345, 3.0, 10.0]
    grid = list(itertools.product(bases, lambda_ratios, roughness, multipliers))
    print(f"grid size {len(grid)}; rows {dataset.n_rows}; subjects {dataset.subjects.size}")

    scores: dict[tuple[int, float, float, float], list[tuple[float, float, float]]] = {
        key: [] for key in grid
    }
    started = time.perf_counter()
    for repeat in range(args.repeats):
        folds = subject_folds(dataset.subject_id, n_folds=args.folds, seed=args.seed + repeat)
        for fold_index, held in enumerate(folds):
            test_mask = np.isin(dataset.subject_id, held)
            train = take(dataset, ~test_mask)
            test = take(dataset, test_mask)
            for q, ratio, mu, multiplier in grid:
                tuning = {
                    "time_domain": [0.0, 1.0],
                    "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
                    "q_time": q,
                    "q_covariate": q,
                    "delta_rule": "mad",
                    "huber_multiplier": multiplier,
                    "lambda_ratio": ratio,
                    "roughness": mu,
                    "max_iter": 2000,
                    "tolerance": 1e-7,
                }
                try:
                    artifact = adapter.fit(train, seed=0, tuning=tuning)
                    prediction = adapter.predict(artifact, test)
                    residual = prediction - test.response
                    value = (
                        float(np.mean(residual**2)),
                        subject_balanced_mse(test.subject_id, residual),
                        float(np.mean(np.abs(residual))),
                    )
                except Exception as error:  # pragma: no cover - probe only
                    print(f"  fail q={q} lam={ratio} mu={mu} c={multiplier}: {error}")
                    value = (float("nan"), float("nan"), float("nan"))
                scores[(q, ratio, mu, multiplier)].append(value)
            print(
                f"repeat {repeat} fold {fold_index} done "
                f"({time.perf_counter() - started:.0f}s elapsed)",
                flush=True,
            )

    print("\n=== mean held-out metrics over folds (sorted by MSE) ===")
    summary = {
        key: tuple(float(np.nanmean([item[index] for item in values])) for index in range(3))
        for key, values in scores.items()
    }
    for key, (mse, balanced, mae) in sorted(summary.items(), key=lambda item: item[1][0]):
        print(
            f"q={key[0]:d} lam={key[1]:<5g} mu={key[2]:<5g} c={key[3]:<6g} "
            f"MSE={mse:.4e}  bal={balanced:.4e}  MAE={mae:.2f}"
        )
    print("\n=== best by MAE ===")
    for key, (mse, balanced, mae) in sorted(summary.items(), key=lambda item: item[1][2])[:8]:
        print(
            f"q={key[0]:d} lam={key[1]:<5g} mu={key[2]:<5g} c={key[3]:<6g} "
            f"MSE={mse:.4e}  bal={balanced:.4e}  MAE={mae:.2f}"
        )


if __name__ == "__main__":
    main()

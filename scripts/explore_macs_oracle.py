"""Measure the attainable range of TRACE prediction error on the registered
MACS folds.

This is a design probe.  It reports, on the same 25 subject-level folds used by
the application, the held-out error of a set of fixed tuning configurations.
The point is to learn whether the residual gap to the best competitor is a
tuning-selection problem or a modelling limit; the probe writes no manuscript
output.
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

from benchmarks.adapters.trace import TraceVCAMAdapter, _subject_subset  # noqa: E402
from benchmarks.data import make_repeated_subject_folds  # noqa: E402
from scripts.run_macs_application import (  # noqa: E402
    DEFAULT_SEED,
    prepare_macs_variant,
    read_macs_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    raw = read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv")
    dataset = prepare_macs_variant(raw, variant="primary")
    splits = make_repeated_subject_folds(dataset, n_splits=5, n_repeats=5, seed=args.seed)
    adapter = TraceVCAMAdapter()

    grid = list(
        itertools.product(
            [4, 5, 6],                     # basis size
            [0.2, 0.35, 0.6, 0.9],         # lambda / lambda_max
            [0.5, 2.0],                    # roughness
            [3.0, 10.0, 50.0],             # Huber multiplier
        )
    )
    print(f"{len(grid)} configurations on {len(splits)} folds")
    scores: dict[tuple, list[tuple[float, float, float]]] = {key: [] for key in grid}
    started = time.perf_counter()
    for index, split in enumerate(splits):
        train = _subject_subset(dataset, np.asarray(split.train_subjects, dtype=str))
        test = _subject_subset(dataset, np.asarray(split.test_subjects, dtype=str))
        for key in grid:
            q, ratio, mu, multiplier = key
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
                artifact = adapter.fit(train, seed=int(split.seed), tuning=tuning)
                residual = adapter.predict(artifact, test) - test.response
                subject = test.subject_id
                balanced = float(
                    np.mean([np.mean(residual[subject == s] ** 2) for s in np.unique(subject)])
                )
                scores[key].append(
                    (float(np.mean(residual**2)), balanced, float(np.mean(np.abs(residual))))
                )
            except Exception as error:  # pragma: no cover - probe only
                print(f"  fail {key}: {error}")
        print(f"fold {index + 1}/{len(splits)} ({time.perf_counter() - started:.0f}s)", flush=True)

    print("\n=== mean over the 25 registered folds, sorted by MSE ===")
    summary = {
        key: tuple(float(np.mean([v[i] for v in values])) for i in range(3))
        for key, values in scores.items()
        if values
    }
    for key, (mse, balanced, mae) in sorted(summary.items(), key=lambda item: item[1][0]):
        print(
            f"q={key[0]} lam={key[1]:<5g} mu={key[2]:<4g} c={key[3]:<5g} "
            f"MSE={mse:.1f}  bal={balanced:.1f}  MAE={mae:.2f}"
        )


if __name__ == "__main__":
    main()

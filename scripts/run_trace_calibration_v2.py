"""Widened pilot calibration of the TRACE tuning constants.

The first calibration searched a narrow grid at a single basis size.  This
script repeats the same protocol over a wider grid: pilot seeds that are
disjoint from every formal Monte Carlo seed, one deterministic 80/20
complete-subject split per pilot data set, and the same registered scorer, the
subject-balanced validation Huber loss.  The scorer never reads a noise-free
response or a true component function, so the selection cannot be tuned to the
quantities the paper reports.

Truth-based errors are also recorded, but only as a diagnostic printed
alongside the selection; they take no part in choosing the winner.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.adapters.trace import TraceVCAMAdapter  # noqa: E402
from benchmarks.data import SubjectDataset  # noqa: E402
from experiments.dgp import (  # noqa: E402
    generate_zsy2026,
    generate_zw2015,
    generate_zzw2020,
    subject_split,
)
from scripts.manuscript_common import GRID_SIZE  # noqa: E402

#: Pilot cohorts.  These are calibration settings only; the formal Monte Carlo
#: experiment uses different seeds.
PILOT_SETTINGS = (
    ("zw2015", {"n_subjects": 100}),
    ("zzw2020", {"n_subjects": 100, "sigma": 0.4, "error_distribution": "gaussian"}),
    ("zzw2020", {"n_subjects": 100, "sigma": 0.4, "error_distribution": "hhy-t2"}),
    ("zzw2020", {"n_subjects": 100, "sigma": 0.4, "error_distribution": "hhy-mixed-normal"}),
)

PILOT_SEED_BASE = 991_000_000


def _build(generator: str, parameters: dict, seed: int):
    if generator == "zw2015":
        return generate_zw2015(seed, **parameters)
    if generator == "zzw2020":
        return generate_zzw2020(seed, **parameters)
    if generator == "zsy2026":
        return generate_zsy2026(seed, **parameters)
    raise ValueError(generator)


def _dataset(raw) -> SubjectDataset:
    return SubjectDataset(
        time=raw.time,
        covariates=raw.covariates,
        response=raw.response,
        subject_id=np.asarray([f"s{i}" for i in raw.subject], dtype=str),
        noise_free_target=raw.conditional_mean,
        metadata={
            "time_domain": list(raw.domain_time),
            "covariate_domains": [list(d) for d in raw.domain_covariates],
        },
    )


def _take(dataset: SubjectDataset, rows: np.ndarray) -> SubjectDataset:
    return SubjectDataset(
        time=dataset.time[rows],
        covariates=dataset.covariates[rows],
        response=dataset.response[rows],
        subject_id=dataset.subject_id[rows],
        noise_free_target=None
        if dataset.noise_free_target is None
        else dataset.noise_free_target[rows],
        metadata=dataset.metadata,
    )


def _huber(residual: np.ndarray, subject: np.ndarray, delta: float) -> float:
    absolute = np.abs(residual)
    loss = np.where(absolute <= delta, 0.5 * absolute**2, delta * absolute - 0.5 * delta**2)
    return float(np.mean([np.mean(loss[subject == s]) for s in np.unique(subject)]))


def _truth_errors(adapter, artifact, raw) -> dict[str, float]:
    """Diagnostic only: componentwise error of the identified factors."""

    from scripts.manuscript_common import componentwise_errors

    curves = adapter.factor_curves(artifact)
    record = {
        "scenario": "example2-diagnostic",
        "curves": [
            {
                "component": c["component"],
                "grid": list(c["grid"]),
                "values": list(c["values"]),
            }
            for c in curves
        ],
    }
    try:
        return componentwise_errors(record)
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "protocol" / "trace_tuning_v2.json")
    args = parser.parse_args()

    adapter = TraceVCAMAdapter()
    bases = (6, 8, 10)
    ratios = (0.01, 0.03, 0.08, 0.15, 0.30)
    roughness = (0.0, 0.005, 0.02, 0.05)
    grid = list(itertools.product(bases, ratios, roughness))
    print(f"{len(grid)} candidates x {len(PILOT_SETTINGS)} settings x {args.pilot_seeds} seeds")

    scores: dict[tuple, list[float]] = {key: [] for key in grid}
    diagnostics: dict[tuple, list[dict]] = {key: [] for key in grid}
    started = time.perf_counter()

    for setting_index, (generator, parameters) in enumerate(PILOT_SETTINGS):
        for pilot in range(args.pilot_seeds):
            seed = PILOT_SEED_BASE + 1000 * setting_index + pilot
            raw = _build(generator, dict(parameters), seed)
            dataset = _dataset(raw)
            train_rows, test_rows = subject_split(raw.subject, seed=seed + 7)
            train, validate = _take(dataset, train_rows), _take(dataset, test_rows)
            for key in grid:
                q, ratio, mu = key
                tuning = {
                    "time_domain": list(raw.domain_time),
                    "covariate_domains": [list(d) for d in raw.domain_covariates],
                    "q_time": q,
                    "q_covariate": q,
                    "delta_rule": "mad",
                    "huber_multiplier": 1.345,
                    "lambda_ratio": ratio,
                    "roughness": mu,
                    "max_iter": 2000,
                    "tolerance": 1e-7,
                }
                try:
                    artifact = adapter.fit(train, seed=seed, tuning=tuning)
                    prediction = adapter.predict(artifact, validate)
                    residual = prediction - validate.response
                    delta = float(artifact.tuning["delta_realized"])
                    scores[key].append(_huber(residual, validate.subject_id, delta))
                    diagnostics[key].append(_truth_errors(adapter, artifact, raw))
                except Exception as error:  # pragma: no cover - calibration probe
                    print(f"  fail {key}: {type(error).__name__}: {error}")
                    scores[key].append(float("nan"))
            print(
                f"setting {setting_index} pilot {pilot} done "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    summary = {
        key: float(np.nanmean(values)) for key, values in scores.items() if values
    }
    ranked = sorted(summary.items(), key=lambda item: item[1])
    print("\n=== registered scorer: mean subject-balanced validation Huber loss ===")
    for key, value in ranked[:15]:
        diag = diagnostics[key]
        phi2 = np.nanmean([d.get("phi_2", np.nan) for d in diag]) if diag else float("nan")
        surface = np.nanmean([d.get("surface", np.nan) for d in diag]) if diag else float("nan")
        print(
            f"K={key[0]:2d} lambda={key[1]:<5g} mu={key[2]:<6g} "
            f"score={value:.6f}   [diagnostic phi2={phi2:.4f} surface={surface:.4f}]"
        )

    best = ranked[0][0]
    payload = {
        "schema_version": "trace-tuning/2",
        "protocol": (
            "widened pilot calibration; selection by subject-balanced validation "
            "Huber loss on pilot seeds disjoint from every formal seed"
        ),
        "pilot_seed_base": PILOT_SEED_BASE,
        "pilot_seeds_per_setting": int(args.pilot_seeds),
        "settings": [
            {"generator": g, "parameters": p} for g, p in PILOT_SETTINGS
        ],
        "grid": {"basis": list(bases), "lambda_ratio": list(ratios), "roughness": list(roughness)},
        "selected": {"basis": int(best[0]), "lambda_ratio": float(best[1]), "roughness": float(best[2])},
        "selected_score": float(ranked[0][1]),
        "ranking": [
            {"basis": int(k[0]), "lambda_ratio": float(k[1]), "roughness": float(k[2]), "score": float(v)}
            for k, v in ranked
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nselected {payload['selected']} -> {args.output}")


if __name__ == "__main__":
    main()

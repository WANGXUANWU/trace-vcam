"""Subject-level bootstrap variability bands for the MACS component curves.

Resamples complete subjects with replacement, refits the proposed estimator at
the tuning selected on the full data, and records the identified baseline and
factor curves.  The output is a variability band for the fitted shapes.  It is
not a confidence band: the resampling does not correct for the selection,
thresholding, and singular value decomposition that precede it, and the paper
reports it as a stability diagnostic.
"""

from __future__ import annotations

import argparse
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
from scripts.run_macs_application import (  # noqa: E402
    DEFAULT_SEED,
    prepare_macs_variant,
    read_macs_csv,
)

GRID_SIZE = 401


def resample_subjects(
    dataset: SubjectDataset,
    rng: np.random.Generator | None = None,
    drawn: np.ndarray | None = None,
) -> SubjectDataset:
    """Draw complete subjects with replacement, relabelling duplicated draws.

    Either a generator or an already drawn subject list is supplied, so that the
    draws can be made once in registered order and materialised later.
    """

    subjects = dataset.subjects
    if drawn is None:
        assert rng is not None
        drawn = rng.choice(subjects, size=subjects.size, replace=True)
    time, covariates, response, subject_id, row_id = [], [], [], [], []
    for position, subject in enumerate(drawn):
        mask = dataset.subject_id == subject
        count = int(mask.sum())
        time.append(dataset.time[mask])
        covariates.append(dataset.covariates[mask])
        response.append(dataset.response[mask])
        subject_id.append(np.full(count, f"b{position}", dtype=object))
        row_id.append(np.asarray([f"b{position}-{index}" for index in range(count)], dtype=object))
    return SubjectDataset(
        time=np.concatenate(time),
        covariates=np.vstack(covariates),
        response=np.concatenate(response),
        subject_id=np.concatenate(subject_id).astype(str),
        row_id=np.concatenate(row_id).astype(str),
        noise_free_target=None,
        covariate_names=dataset.covariate_names,
        metadata=dataset.metadata,
    )


def identified_curves(
    adapter: TraceVCAMAdapter, artifact, grid: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the curves on a common grid after Lebesgue identification.

    A block that the estimator sets to zero is returned as the zero function,
    which is what the fitted model says about it, rather than as a missing
    value.  ``retained_k`` records whether block ``k`` survived, and
    ``scale_k`` records the normalising constant \\(\\int\\beta_k\\) before the
    Lebesgue rescaling.  That constant is the whole content of the scale split
    between the two factors: the component surface \\(\\beta_k\\phi_k\\) does not
    depend on it, while the individual factors are divided and multiplied by it,
    so a resample whose constant is near zero produces an arbitrarily large
    factor pair from a perfectly ordinary surface.  It is stored so that the
    reported bands can be conditioned on it rather than being dominated by it.
    """

    raw = {
        str(curve["component"]): (
            np.asarray(curve["grid"], dtype=float),
            np.asarray(curve["values"], dtype=float),
        )
        for curve in adapter.factor_curves(artifact)
    }
    out: dict[str, np.ndarray] = {}
    baseline = (
        np.interp(grid, *raw["baseline"]) if "baseline" in raw else np.zeros_like(grid)
    )
    for index in (1, 2):
        beta_key, phi_key = f"beta_{index}", f"phi_{index}"
        retained = beta_key in raw and phi_key in raw
        out[f"retained_{index}"] = np.array([float(retained)])
        if not retained:
            out[beta_key] = np.zeros_like(grid)
            out[phi_key] = np.zeros_like(grid)
            out[f"scale_{index}"] = np.array([np.nan])
            continue
        beta = np.interp(grid, *raw[beta_key])
        phi = np.interp(grid, *raw[phi_key])
        phi_mean = float(np.trapezoid(phi, grid) / (grid[-1] - grid[0]))
        phi = phi - phi_mean
        baseline = baseline + phi_mean * beta
        beta_mean = float(np.trapezoid(beta, grid) / (grid[-1] - grid[0]))
        out[f"scale_{index}"] = np.array([beta_mean])
        if abs(beta_mean) > 1e-8:
            beta = beta / beta_mean
            phi = phi * beta_mean
        out[beta_key] = beta
        out[phi_key] = phi
    out["baseline"] = baseline
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED + 1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "macs_bootstrap"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv")
    dataset = prepare_macs_variant(raw, variant="primary")
    adapter = TraceVCAMAdapter()

    base_tuning = {
        "time_domain": [0.0, 1.0],
        "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
        "q_time": 6,
        "q_covariate": 6,
        "delta_rule": "mad",
        "huber_multiplier": 1.345,
        "lambda_ratio": 0.03,
        "roughness": 0.5,
        "max_iter": 2000,
        "tolerance": 1e-7,
        "selection": "subject_cv",
        "cv_folds": 3,
        "cv_basis_grid": [5, 6],
        "cv_lambda_ratio_grid": [0.2, 0.6, 0.9],
        "cv_roughness_grid": [0.5],
        "cv_huber_multiplier_grid": [1.345, 3.0, 10.0],
    }
    full = adapter.fit(dataset, seed=DEFAULT_SEED, tuning=base_tuning)
    selected = dict(full.tuning["selection_audit"]["selected"])
    print("full-data selection:", selected, flush=True)

    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    point = identified_curves(adapter, full, grid)

    fixed_tuning = {
        **{k: v for k, v in base_tuning.items() if not k.startswith("cv_")},
        **selected,
        "selection": "fixed",
    }
    # Each resample is drawn from one stream in registered order, so the drawn
    # data sets do not depend on how many workers run them.
    rng = np.random.default_rng(args.seed)
    drawn_subjects = [
        rng.choice(dataset.subjects, size=dataset.subjects.size, replace=True)
        for _ in range(args.replicates)
    ]

    def fit_one(replicate: int):
        sample = resample_subjects(dataset, drawn=drawn_subjects[replicate])
        try:
            artifact = adapter.fit(
                sample, seed=args.seed + replicate, tuning=fixed_tuning
            )
            return replicate, identified_curves(adapter, artifact, grid)
        except Exception as error:  # pragma: no cover - a resample may be infeasible
            print(f"  replicate {replicate} failed: {type(error).__name__}: {error}")
            return replicate, None

    draws: dict[str, list[np.ndarray]] = {name: [] for name in point}
    started = time.perf_counter()
    completed = 0
    if args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(fit_one, range(args.replicates)))
    else:
        outcomes = [fit_one(replicate) for replicate in range(args.replicates)]
    for replicate, curves in sorted(outcomes, key=lambda item: item[0]):
        if curves is None:
            continue
        for name, values in curves.items():
            draws[name].append(values)
        completed += 1
    print(
        f"{completed}/{args.replicates} completed "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )

    # The complete draw matrices are stored so that the reported band can be
    # recomputed without refitting: the scale split between the two factors is
    # not stable across resamples, and how it is handled has to be visible.
    np.savez_compressed(
        args.output / "macs_bootstrap_draws.npz",
        grid=grid,
        **{f"point::{name}": values for name, values in point.items()},
        **{
            f"draws::{name}": np.vstack(stack)
            for name, stack in draws.items()
            if stack
        },
    )

    payload = {
        "schema_version": "vcam-macs-bootstrap/3",
        "replicates_requested": int(args.replicates),
        "replicates_completed": int(completed),
        "seed": int(args.seed),
        "resampling_unit": "subject, with replacement",
        "selected_tuning": selected,
        "grid": grid.tolist(),
        "point_estimate": {name: values.tolist() for name, values in point.items()},
        "retention": {},
        "bands": {},
    }
    payload["scale_quantiles"] = {}
    for name, stack in draws.items():
        if not stack:
            continue
        matrix = np.vstack(stack)
        if name.startswith("retained_"):
            payload["retention"][name] = float(np.mean(matrix))
            continue
        if name.startswith("scale_"):
            finite = matrix[np.isfinite(matrix)]
            payload["scale_quantiles"][name] = (
                np.percentile(finite, [1, 5, 25, 50, 75, 95, 99]).tolist()
                if finite.size
                else []
            )
            continue
        lower, upper = np.nanpercentile(matrix, [2.5, 97.5], axis=0)
        payload["bands"][name] = {"lower": lower.tolist(), "upper": upper.tolist()}
    target = args.output / "macs_bootstrap_bands.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {target} from {completed} completed resamples")


if __name__ == "__main__":
    main()

"""Second application: serum bilirubin in the Mayo PBC sequential follow-up.

The CD4 application of the main paper is a demanding test of the estimator's
computational behaviour but a weak one of its interpretive claim: the predictive
differences between the strongest methods are small and the retained covariate
component is not resolved at that sample size.  This application was added
because the model's own premise -- a covariate effect that is nonlinear and
whose strength changes over follow-up -- is visible in these data.

Data.  ``survival::pbcseq`` records 1,945 laboratory visits on the 312 patients
of the Mayo primary biliary cholangitis trial who have sequential follow-up.
The response is log serum bilirubin, the standard marker of cholestatic
progression; the time variable is years since registration; the time-invariant
covariate is age at registration; and the time-varying covariate is serum
albumin, the synthetic-function marker measured at the same visit.  Both
covariates enter through the model's own additive-nonlinear form.

Cluster size is genuinely informative here, and not by construction: follow-up
ends at death or transplant, so patients with poorer liver function contribute
fewer visits.  The number of visits ranges from one to sixteen and correlates
with the subject's mean albumin.  That is the regime the subject-balanced
weighting of the estimator is written for, and it is a property of the data
rather than of a simulation design.

Protocol.  Five repeats of fivefold cross-validation over complete subjects,
with every method receiving the same training and test subjects in a given fold
and selecting its own tuning by the rule registered for the MACS application.
A subject-level bootstrap over the full-data fit supplies the variability bands.
No inferential claim is attached to either.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.data import SubjectDataset  # noqa: E402

DEFAULT_SEED = 20260816
GRID_SIZE = 201

#: Applicable methods.  The two excluded ones are excluded for the same reason
#: as in the CD4 application: the dense-functional estimator needs covariates
#: that do not vary within a subject, and the author code of the coefficientwise
#: Lasso exposes no out-of-sample prediction interface.
PBC_METHODS = (
    "TRACE-VCAM",
    "ZZW2020",
    "HHY2021-Huber",
    "ZY2025-paper-implementation",
)

COVARIATE_NAMES = ("age_scaled", "albumin_scaled")


@dataclass(frozen=True)
class PreparedPBC:
    dataset: SubjectDataset
    bounds: dict[str, tuple[float, float]]


def _minmax(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        raise ValueError("a coordinate must have positive range")
    return (values - low) / (high - low), (low, high)


def prepare_pbc(path: Path) -> PreparedPBC:
    """Read the exported sequential records and map each domain onto [0,1]."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    required = {"person", "time", "logbili", "age", "albumin"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"the PBC CSV must contain {sorted(required)}")
    person = np.asarray([str(record["person"]) for record in records], dtype=str)
    columns = {
        name: np.asarray([float(record[name]) for record in records], dtype=float)
        for name in ("time", "logbili", "age", "albumin")
    }
    if not all(np.all(np.isfinite(values)) for values in columns.values()):
        raise ValueError("the PBC analysis variables must be finite")

    time_scaled, time_bounds = _minmax(columns["time"])
    age_scaled, age_bounds = _minmax(columns["age"])
    albumin_scaled, albumin_bounds = _minmax(columns["albumin"])
    dataset = SubjectDataset(
        time=time_scaled,
        covariates=np.column_stack([age_scaled, albumin_scaled]),
        response=columns["logbili"],
        subject_id=person,
        row_id=np.asarray([f"pbc-row-{index}" for index in range(len(records))], dtype=str),
        noise_free_target=None,
        covariate_names=COVARIATE_NAMES,
        metadata={
            "response": "log serum bilirubin (mg/dl)",
            "time": "years since registration",
            "age": "age at registration (years)",
            "albumin": "serum albumin at the visit (g/dl)",
            "source": "survival::pbcseq 3.8.3",
            "coordinate_bounds": {
                "time": list(time_bounds),
                "age": list(age_bounds),
                "albumin": list(albumin_bounds),
            },
            "time_domain": [0.0, 1.0],
            "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
            "time_invariant_covariates": False,
        },
    )
    return PreparedPBC(
        dataset=dataset,
        bounds={
            "time": time_bounds,
            "age": age_bounds,
            "albumin": albumin_bounds,
        },
    )


def _subset(dataset: SubjectDataset, rows: np.ndarray) -> SubjectDataset:
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


def subject_folds(
    subject_id: np.ndarray, *, repeats: int, folds: int, seed: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Complete-subject fold assignments, never splitting within a subject."""

    unique = np.unique(subject_id)
    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        order = np.random.default_rng(seed + 101 * repeat).permutation(unique)
        assignment = {name: index % folds for index, name in enumerate(order)}
        membership = np.asarray([assignment[name] for name in subject_id])
        for fold in range(folds):
            test = np.flatnonzero(membership == fold)
            train = np.flatnonzero(membership != fold)
            out.append((repeat, fold, train, test))
    return out


def _tuning(method: str, basis_dimension: int = 6) -> dict[str, object]:
    """The tuning rule each method uses, identical to the CD4 application."""

    from scripts.run_macs_application import _tuning as macs_tuning

    tuning = macs_tuning(method, basis_dimension, quick=False)
    tuning["application"] = "PBC-bilirubin"
    return tuning


def _fold_task(payload):
    repeat, fold, train_rows, test_rows, seed, data_path = payload
    from benchmarks.methods import Applicability, applicability_for
    from scripts.run_macs_application import adapter_registry

    prepared = prepare_pbc(data_path)
    train = _subset(prepared.dataset, train_rows)
    test = _subset(prepared.dataset, test_rows)
    adapters = adapter_registry()

    rows = []
    for method in PBC_METHODS:
        decision = applicability_for(method, "application/MACS-CD4")
        row: dict[str, object] = {
            "repeat": repeat,
            "fold": fold,
            "method": method,
            "n_train_rows": int(train.n_rows),
            "n_test_rows": int(test.n_rows),
            "applicability": decision.status.value,
        }
        if decision.status is not Applicability.APPLICABLE:
            row["attempt_status"] = "N/A by design"
            rows.append(row)
            continue
        started = time.perf_counter()
        try:
            artifact = adapters[method].fit(
                train, seed=seed + 1000 * repeat + fold, tuning=_tuning(method)
            )
            prediction = np.asarray(
                adapters[method].predict(artifact, test), dtype=float
            )
            if prediction.shape != (test.n_rows,) or not np.all(np.isfinite(prediction)):
                raise FloatingPointError("non-finite or wrong-length prediction")
            residual = prediction - test.response
            squared = residual**2
            balanced = float(
                np.mean(
                    [
                        np.mean(squared[test.subject_id == name])
                        for name in np.unique(test.subject_id)
                    ]
                )
            )
            row.update(
                attempt_status="success" if artifact.converged else "failed",
                converged=bool(artifact.converged),
                mspe=float(np.mean(squared)),
                balanced_mspe=balanced,
                mape=float(np.mean(np.abs(residual))),
                runtime_seconds=time.perf_counter() - started,
            )
            if not artifact.converged:
                row["failure_code"] = "nonconvergence"
        except Exception as error:  # pragma: no cover - a source method may fail
            row.update(
                attempt_status="failed",
                converged=False,
                failure_code=str(getattr(error, "code", type(error).__name__)),
                failure_message=f"{type(error).__name__}: {error}"[:400],
                runtime_seconds=time.perf_counter() - started,
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Full-data fit, bootstrap band, and rank diagnostic
# ---------------------------------------------------------------------------


def _trace_full_data(dataset: SubjectDataset, *, seed: int):
    """One cross-validated TRACE fit to every subject, with its curves."""

    from scripts.run_macs_application import adapter_registry

    adapter = adapter_registry()["TRACE-VCAM"]
    artifact = adapter.fit(dataset, seed=seed, tuning=_tuning("TRACE-VCAM"))
    return adapter, artifact


def fixed_tuning_from(artifact) -> dict[str, object]:
    """The tuning the full-data fit selected, frozen so a refit cannot reselect.

    The bootstrap band is meant to describe how much the fitted shapes move when
    the same rule is refitted on a resample, not how much the selection moves as
    well.  Carrying the cross-validation grids into each draw would measure the
    second and cost a full inner search per draw, so the ``cv_*`` entries are
    dropped and the realised choices are pinned in their place.  This is the
    convention the CD4 bootstrap already uses.
    """

    base = _tuning("TRACE-VCAM")
    selected = dict(
        dict(getattr(artifact, "tuning", {}))
        .get("selection_audit", {})
        .get("selected", {})
    )
    return {
        **{key: value for key, value in base.items() if not key.startswith("cv_")},
        **selected,
        "selection": "fixed",
        "tuning_mode": "application_fixed_at_full_data_selection",
    }


def _curves_from_artifact(adapter, artifact) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for curve in adapter.factor_curves(artifact):
        name = str(curve["component"])
        out[name] = np.asarray(curve["values"], dtype=float)
    return out


def _bootstrap_task(payload):
    draw, seed, data_path, tuning = payload
    from scripts.run_macs_application import adapter_registry

    prepared = prepare_pbc(data_path)
    dataset = prepared.dataset
    unique = np.unique(dataset.subject_id)
    rng = np.random.default_rng(seed + draw)
    drawn = rng.choice(unique, size=unique.size, replace=True)
    # A subject drawn twice contributes two clusters, so the duplicate is
    # relabelled rather than merged.
    times, covariates, responses, labels = [], [], [], []
    for position, name in enumerate(drawn):
        rows = np.flatnonzero(dataset.subject_id == name)
        times.append(dataset.time[rows])
        covariates.append(dataset.covariates[rows])
        responses.append(dataset.response[rows])
        labels.append(np.full(rows.size, f"boot-{position}", dtype=object))
    resample = SubjectDataset(
        time=np.concatenate(times),
        covariates=np.vstack(covariates),
        response=np.concatenate(responses),
        subject_id=np.asarray(
            [str(value) for value in np.concatenate(labels)], dtype=str
        ),
        row_id=None,
        noise_free_target=None,
        covariate_names=dataset.covariate_names,
        metadata=dict(dataset.metadata),
    )
    adapter = adapter_registry()["TRACE-VCAM"]
    try:
        artifact = adapter.fit(resample, seed=seed + draw, tuning=dict(tuning))
    except Exception:
        return None
    if not artifact.converged:
        return None
    curves = _curves_from_artifact(adapter, artifact)
    retained = {
        f"retained_{index + 1}": float(
            1.0 if index in set(artifact.selected_blocks) else 0.0
        )
        for index in range(2)
    }
    return {**{name: values.tolist() for name, values in curves.items()}, **retained}


def command_cv(args) -> None:
    prepared = prepare_pbc(args.data)
    folds = subject_folds(
        prepared.dataset.subject_id,
        repeats=args.repeats,
        folds=args.folds,
        seed=args.seed,
    )
    print(
        f"{len(folds)} folds over {np.unique(prepared.dataset.subject_id).size} subjects",
        flush=True,
    )
    tasks = [
        (repeat, fold, train, test, args.seed, args.data)
        for repeat, fold, train, test in folds
    ]
    started = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for index, result in enumerate(pool.map(_fold_task, tasks, chunksize=1), start=1):
            rows.extend(result)
            print(
                f"fold {index}/{len(tasks)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    target = args.output / "pbc_cv_results.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"wrote {target} ({len(rows)} rows)")


def command_full(args) -> None:
    from scripts.review_common import build_design, locked_tuning, rank_r_refit
    from src.trace_vcam import OrthonormalSplineBasis, predict_components

    prepared = prepare_pbc(args.data)
    dataset = prepared.dataset
    adapter, artifact = _trace_full_data(dataset, seed=args.seed)
    curves = _curves_from_artifact(adapter, artifact)
    realized = dict(getattr(artifact, "tuning", {}))
    payload: dict[str, object] = {
        "grid": np.linspace(0.0, 1.0, GRID_SIZE).tolist(),
        "point_estimate": {name: values.tolist() for name, values in curves.items()},
        "selected_blocks": list(artifact.selected_blocks),
        "realized_tuning": {
            key: value
            for key, value in realized.items()
            if isinstance(value, (int, float, str, bool, list))
        },
        "coordinate_bounds": {
            key: list(value) for key, value in prepared.bounds.items()
        },
        "n_subjects": int(np.unique(dataset.subject_id).size),
        "n_rows": int(dataset.n_rows),
    }

    # The rank-one adequacy diagnostic, run on the folds the application already
    # uses, so that the answer is a held-out one.
    basis_dimension = int(realized.get("q_time", 6))
    basis = OrthonormalSplineBasis.create(basis_dimension, basis_dimension)
    folds = subject_folds(
        dataset.subject_id, repeats=1, folds=args.folds, seed=args.seed
    )
    per_rank: dict[int, list[float]] = {1: [], 2: [], 3: []}
    ratios: list[float] = []
    for _repeat, _fold, train_rows, test_rows in folds:
        train = _subset(dataset, train_rows)
        test = _subset(dataset, test_rows)
        train_design = build_design(
            train.time, train.covariates, train.response, train.subject_id,
            basis, (0.0, 1.0),
        )
        test_design = build_design(
            test.time, test.covariates, test.response, test.subject_id,
            basis, (0.0, 1.0),
        )
        from scripts.review_common import fit_delivered

        tuning = locked_tuning()
        pilot = fit_delivered(
            train_design,
            basis,
            tuning=tuning,
            lambda_ratio=float(realized.get("lambda_ratio", tuning["lambda_ratio"])),
            roughness=float(realized.get("roughness", tuning["roughness"])),
            post_rank_one=False,
        )
        for index, values in enumerate(pilot.singular_values):
            if pilot.selected[index] and values[0] > 0.0 and values.size > 1:
                ratios.append(float(values[1] / values[0]))
        for rank in (1, 2, 3):
            gamma, blocks, _converged = rank_r_refit(
                train_design, pilot.gamma, pilot.blocks, pilot.selected,
                pilot.delta, rank,
            )
            prediction = predict_components(test_design, gamma, blocks)[0]
            per_rank[rank].append(float(np.mean((prediction - test.response) ** 2)))
    base = float(np.mean(per_rank[1]))
    payload["rank_diagnostic"] = {
        "folds": len(folds),
        "spectral_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "spectral_ratio_max": float(np.max(ratios)) if ratios else None,
        **{f"mspe_rank{rank}": float(np.mean(values)) for rank, values in per_rank.items()},
        **{
            f"rank{rank}_gain": float((base - np.mean(values)) / base)
            for rank, values in per_rank.items()
            if rank != 1
        },
    }

    if args.bootstrap > 0:
        frozen = fixed_tuning_from(artifact)
        payload["bootstrap_tuning"] = {
            key: value
            for key, value in frozen.items()
            if isinstance(value, (int, float, str, bool))
        }
        draws = [
            (draw, args.seed, args.data, frozen) for draw in range(args.bootstrap)
        ]
        collected: list[dict] = []
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for index, result in enumerate(
                pool.map(_bootstrap_task, draws, chunksize=1), start=1
            ):
                if result is not None:
                    collected.append(result)
                if index % 25 == 0 or index == len(draws):
                    print(
                        f"bootstrap {index}/{len(draws)} "
                        f"({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )
        stacked = {}
        for name in ("baseline", "beta_1", "phi_1", "beta_2", "phi_2"):
            values = [row[name] for row in collected if name in row]
            if values:
                stacked[name] = np.asarray(values, dtype=float)
        for block in ("1", "2"):
            key = f"retained_{block}"
            values = [row[key] for row in collected if key in row]
            if values:
                stacked[key] = np.asarray(values, dtype=float)
        np.savez_compressed(
            args.output / "pbc_bootstrap_draws.npz",
            **{f"draws::{name}": values for name, values in stacked.items()},
        )
        payload["bootstrap_completed"] = len(collected)
        payload["bootstrap_attempted"] = args.bootstrap
        payload["retention"] = {
            block: float(np.mean(stacked[f"retained_{block}"]))
            for block in ("1", "2")
            if f"retained_{block}" in stacked
        }

    (args.output / "pbc_full_data.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload.get("rank_diagnostic", {}), indent=2))
    print(f"wrote {args.output / 'pbc_full_data.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["cv", "full"])
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "raw" / "survival_pbcseq.csv"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pbc")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"cv": command_cv, "full": command_full}[args.command](args)


if __name__ == "__main__":
    main()

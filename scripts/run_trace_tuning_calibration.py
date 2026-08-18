"""Calibrate one global TRACE--VCAM tuning pair on independent pilot data.

The calibration never inspects a noise-free target or a truth curve.  Each
candidate is scored by subject-balanced Huber loss on a held-out 20% of pilot
subjects, using the Huber threshold estimated from that candidate's training
subjects.  Pilot data and split seeds are explicitly disjoint from the formal
Monte Carlo registry.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.adapters import TraceVCAMAdapter  # noqa: E402
from scripts import run_strict_benchmark as strict  # noqa: E402
from src.trace_tuning_protocol import (  # noqa: E402
    DEFAULT_TRACE_TUNING_PATH,
    SCHEMA_VERSION,
    TUNING_MODE,
    load_trace_tuning_lock,
    trace_tuning_content_sha256,
)
from src.trace_vcam import huber_values  # noqa: E402


PILOT_DATA_SEEDS = (73031021, 73031057, 73031111, 73031147, 73031203)
PILOT_SPLIT_ROOT_SEED = 917340521
PILOT_SCENARIOS = (
    "example1-zw2015-n100",
    "example2-gaussian-n50-sigma0.1",
    "example3-gaussian-n50-p10-sigma0.1",
)
LAMBDA_RATIO_GRID = (0.03, 0.05, 0.08, 0.15, 0.2, 0.35)
ROUGHNESS_GRID = (0.0, 0.01, 0.05)
TRAIN_FRACTION = 0.8
Q_TIME = 6
Q_COVARIATE = 6


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pilot_split_seed(scenario: str, data_seed: int) -> int:
    return strict._stable_seed(
        PILOT_SPLIT_ROOT_SEED, TUNING_MODE, scenario, data_seed, "subject-split"
    )


def _subject_balanced_huber_loss(
    response: np.ndarray,
    prediction: np.ndarray,
    subject_id: np.ndarray,
    delta: float,
) -> float:
    """Average within subject first, then give every subject equal weight."""

    residual = np.asarray(response, dtype=float) - np.asarray(prediction, dtype=float)
    subjects = np.asarray(subject_id).reshape(-1)
    if residual.shape != subjects.shape:
        raise ValueError("response/prediction and subject IDs must be row aligned")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("validation Huber threshold must be finite and positive")
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(counts) * counts[inverse])
    value = float(np.dot(weights, huber_values(residual, float(delta))))
    if not np.isfinite(value):
        raise FloatingPointError("validation Huber loss is not finite")
    return value


def _registered_pilot_scenarios() -> dict[str, strict.Scenario]:
    formal = {item.scenario: item for item in strict.registered_scenarios(quick=False)}
    missing = set(PILOT_SCENARIOS) - set(formal)
    if missing:
        raise RuntimeError(f"pilot scenarios missing from formal registry: {sorted(missing)}")
    return {name: formal[name] for name in PILOT_SCENARIOS}


def _seed_independence_audit(
    scenarios: Mapping[str, strict.Scenario],
) -> dict[str, object]:
    formal_scenarios = strict.registered_scenarios(quick=False)
    formal_data_seeds: set[int] = set()
    formal_split_seeds: set[int] = set()
    for scenario in formal_scenarios:
        for replicate in range(scenario.formal_replications):
            formal_data_seeds.add(
                strict._stable_seed(
                    strict.DEFAULT_ROOT_SEED, scenario.scenario, replicate, "data"
                )
            )
            formal_split_seeds.add(
                strict._stable_seed(
                    strict.DEFAULT_ROOT_SEED,
                    scenario.scenario,
                    replicate,
                    "subject-split",
                )
            )
    pilot_data = set(PILOT_DATA_SEEDS)
    pilot_splits = {
        _pilot_split_seed(scenario, seed)
        for scenario in scenarios
        for seed in PILOT_DATA_SEEDS
    }
    data_overlap = sorted(pilot_data & formal_data_seeds)
    split_overlap = sorted(pilot_splits & formal_split_seeds)
    return {
        "passed": not data_overlap and not split_overlap,
        "formal_root_seed": strict.DEFAULT_ROOT_SEED,
        "formal_registered_data_seed_count": len(formal_data_seeds),
        "formal_registered_split_seed_count": len(formal_split_seeds),
        "formal_data_seed_overlap": data_overlap,
        "formal_split_seed_overlap": split_overlap,
        "pilot_split_root_seed": PILOT_SPLIT_ROOT_SEED,
    }


def _candidate_tuning(lambda_ratio: float, roughness: float) -> dict[str, object]:
    return {
        "q_time": Q_TIME,
        "q_covariate": Q_COVARIATE,
        "delta_rule": "mad",
        "huber_multiplier": 1.345,
        "lambda_ratio": float(lambda_ratio),
        "roughness": float(roughness),
        "max_iter": 2000,
        "tolerance": 1e-7,
        "postfit_max_iter": 1000,
        "postfit_tolerance": 2e-7,
        "tuning_mode": "independent_pilot_candidate_v1",
    }


def _run_pilot_cohort(scenario_name: str, data_seed: int) -> dict[str, object]:
    scenario = _registered_pilot_scenarios()[scenario_name]
    split_seed = _pilot_split_seed(scenario_name, data_seed)
    raw = scenario.build(data_seed)
    dataset = strict._subject_dataset(raw, scenario)
    train, validation, split = strict._registered_split(
        dataset, raw, scenario, split_seed=split_seed
    )
    adapter = TraceVCAMAdapter()
    report = adapter.preflight()
    if not report.ready:
        raise RuntimeError(f"TRACE preflight failed: {report.code}: {report.message}")
    evaluations: list[dict[str, object]] = []
    cohort_id = f"{scenario_name}/pilot-{data_seed}"
    for lambda_ratio in LAMBDA_RATIO_GRID:
        for roughness in ROUGHNESS_GRID:
            started = time.perf_counter()
            status = "success"
            failure = ""
            loss: float | None = None
            delta: float | None = None
            converged = False
            selected_blocks: list[int] = []
            try:
                artifact = adapter.fit(
                    train,
                    seed=data_seed,
                    tuning=_candidate_tuning(lambda_ratio, roughness),
                )
                converged = bool(artifact.converged)
                if not converged:
                    raise RuntimeError("TRACE candidate did not meet its stopping rule")
                prediction = adapter.predict(artifact, validation)
                delta = float(artifact.tuning["delta_realized"])
                loss = _subject_balanced_huber_loss(
                    validation.response,
                    prediction,
                    validation.subject_id,
                    delta,
                )
                selected_blocks = [int(item) for item in artifact.selected_blocks]
            except Exception as error:  # retain every failed candidate in the audit
                status = "failed"
                failure = f"{type(error).__name__}: {error}"[:2000]
            evaluations.append(
                {
                    "cohort_id": cohort_id,
                    "scenario": scenario_name,
                    "pilot_data_seed": int(data_seed),
                    "pilot_split_seed": int(split_seed),
                    "lambda_ratio": float(lambda_ratio),
                    "roughness": float(roughness),
                    "validation_huber_loss": loss,
                    "delta_estimated_from_training": delta,
                    "status": status,
                    "converged": converged,
                    "selected_blocks": selected_blocks,
                    "failure": failure,
                    "runtime_seconds": float(time.perf_counter() - started),
                }
            )
    return {
        "cohort": {
            "cohort_id": cohort_id,
            "scenario": scenario_name,
            "generator": scenario.generator,
            "parameters": dict(scenario.parameters),
            "pilot_data_seed": int(data_seed),
            "pilot_split_seed": int(split_seed),
            "n_subjects": dataset.n_subjects,
            "n_train_subjects": train.n_subjects,
            "n_validation_subjects": validation.n_subjects,
            "n_train_rows": train.n_rows,
            "n_validation_rows": validation.n_rows,
            "data_hash": dataset.data_hash,
            "train_subject_hash": split.train_hash,
            "validation_subject_hash": split.test_hash,
        },
        "evaluations": evaluations,
        "estimator_version": str(report.version),
    }


def _summarize_candidates(
    evaluations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    expected = len(PILOT_SCENARIOS) * len(PILOT_DATA_SEEDS)
    for lambda_ratio in LAMBDA_RATIO_GRID:
        for roughness in ROUGHNESS_GRID:
            selected = [
                item
                for item in evaluations
                if float(item["lambda_ratio"]) == lambda_ratio
                and float(item["roughness"]) == roughness
            ]
            losses = [
                float(item["validation_huber_loss"])
                for item in selected
                if item.get("status") == "success"
                and item.get("validation_huber_loss") is not None
            ]
            summaries.append(
                {
                    "lambda_ratio": float(lambda_ratio),
                    "roughness": float(roughness),
                    "evaluation_count": len(selected),
                    "successful_count": len(losses),
                    "failure_count": len(selected) - len(losses),
                    "mean_validation_huber_loss": (
                        float(np.mean(losses))
                        if len(selected) == expected and len(losses) == expected
                        else None
                    ),
                }
            )
    return summaries


def _select_global_pair(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    complete = [
        item for item in summaries if item.get("mean_validation_huber_loss") is not None
    ]
    if len(complete) != len(LAMBDA_RATIO_GRID) * len(ROUGHNESS_GRID):
        raise RuntimeError("at least one TRACE calibration candidate is incomplete")
    selected = min(
        complete,
        key=lambda item: (
            float(item["mean_validation_huber_loss"]),
            float(item["lambda_ratio"]),
            float(item["roughness"]),
        ),
    )
    return {
        "lambda_ratio": float(selected["lambda_ratio"]),
        "roughness": float(selected["roughness"]),
        "mean_validation_huber_loss": float(
            selected["mean_validation_huber_loss"]
        ),
    }


def execute(*, output: Path, jobs: int) -> dict[str, object]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    started = time.perf_counter()
    scenarios = _registered_pilot_scenarios()
    seed_audit = _seed_independence_audit(scenarios)
    if not seed_audit["passed"]:
        raise RuntimeError("pilot seeds overlap the formal Monte Carlo registry")
    tasks = [
        (scenario, seed)
        for scenario in PILOT_SCENARIOS
        for seed in PILOT_DATA_SEEDS
    ]
    results: list[dict[str, object]] = []
    if jobs == 1:
        for scenario, seed in tasks:
            print(f"[calibration] {scenario} seed={seed}", flush=True)
            results.append(_run_pilot_cohort(scenario, seed))
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(_run_pilot_cohort, scenario, seed): (scenario, seed)
                for scenario, seed in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                scenario, seed = future_map[future]
                results.append(future.result())
                print(f"[calibration] complete {scenario} seed={seed}", flush=True)
    results.sort(
        key=lambda item: (
            PILOT_SCENARIOS.index(str(item["cohort"]["scenario"])),
            PILOT_DATA_SEEDS.index(int(item["cohort"]["pilot_data_seed"])),
        )
    )
    cohorts = [dict(item["cohort"]) for item in results]
    evaluations = [
        dict(evaluation)
        for result in results
        for evaluation in result["evaluations"]
    ]
    summaries = _summarize_candidates(evaluations)
    selected = _select_global_pair(summaries)
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "trace_tuning_protocol.py",
        ROOT / "src" / "trace_vcam.py",
        ROOT / "benchmarks" / "adapters" / "trace.py",
        ROOT / "experiments" / "dgp.py",
        ROOT / "scripts" / "run_strict_benchmark.py",
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "tuning_mode": TUNING_MODE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "q_time": Q_TIME,
        "q_covariate": Q_COVARIATE,
        "train_fraction": TRAIN_FRACTION,
        "pilot_data_seeds": list(PILOT_DATA_SEEDS),
        "pilot_scenarios": list(PILOT_SCENARIOS),
        "candidate_grid": {
            "lambda_ratio": list(LAMBDA_RATIO_GRID),
            "roughness": list(ROUGHNESS_GRID),
        },
        "selection_criterion": (
            "unweighted arithmetic mean across 3 scenarios x 5 seeds of "
            "subject-balanced validation Huber loss; threshold estimated from "
            "the corresponding training subjects"
        ),
        "selection_tie_break": "smaller lambda_ratio, then smaller roughness",
        "truth_usage": (
            "noise_free targets and truth curves are never read by the scorer or selector"
        ),
        "split_protocol": "one deterministic 80/20 complete-subject split per pilot cohort",
        "seed_independence_audit": seed_audit,
        "cohorts": cohorts,
        "evaluations": evaluations,
        "candidate_summaries": summaries,
        "selected_pair": selected,
        "calibration_runtime_seconds": float(time.perf_counter() - started),
        "execution": {"jobs": jobs, "task_count": len(tasks)},
        "estimator_versions": sorted(
            {str(result["estimator_version"]) for result in results}
        ),
        "source_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _file_sha256(path)
            for path in source_paths
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }
    payload["content_sha256"] = trace_tuning_content_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    lock = load_trace_tuning_lock(output)
    print(
        "[calibration] selected "
        f"lambda_ratio={lock['lambda_ratio']}, roughness={lock['roughness']}, "
        f"content_sha256={lock['content_sha256']}",
        flush=True,
    )
    return lock


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRACE_TUNING_PATH)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    execute(output=args.output.resolve(), jobs=int(args.jobs))


if __name__ == "__main__":
    main()


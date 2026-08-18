"""Validation and loading for the locked TRACE--VCAM tuning protocol."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_TUNING_PATH = ROOT / "protocol" / "trace_tuning_v1.json"
SCHEMA_VERSION = "trace-tuning-calibration/1"
TUNING_MODE = "independent_pilot_calibration_v1"
EXPECTED_PILOT_COUNT = 5
EXPECTED_EVALUATIONS_PER_CANDIDATE = 15


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def trace_tuning_content_sha256(payload: Mapping[str, object]) -> str:
    """Hash every protocol field except the self-referential hash field."""

    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trace_tuning_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate completeness, the selected minimizer, and the content hash."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected TRACE tuning schema version")
    if payload.get("tuning_mode") != TUNING_MODE:
        raise ValueError("unexpected TRACE tuning mode")
    if int(payload.get("q_time", -1)) != 6 or int(payload.get("q_covariate", -1)) != 6:
        raise ValueError("TRACE calibration must lock q_time=q_covariate=6")
    if float(payload.get("train_fraction", float("nan"))) != 0.8:
        raise ValueError("TRACE calibration must use an 80/20 subject split")

    seeds = payload.get("pilot_data_seeds")
    if not isinstance(seeds, list) or len(seeds) != EXPECTED_PILOT_COUNT:
        raise ValueError("TRACE calibration must register exactly five pilot seeds")
    if len({int(seed) for seed in seeds}) != EXPECTED_PILOT_COUNT:
        raise ValueError("TRACE pilot seeds must be distinct")

    audit = payload.get("seed_independence_audit")
    if not isinstance(audit, Mapping) or audit.get("passed") is not True:
        raise ValueError("TRACE pilot/formal seed independence was not verified")
    if audit.get("formal_data_seed_overlap") or audit.get("formal_split_seed_overlap"):
        raise ValueError("TRACE pilot seeds overlap the formal Monte Carlo registry")

    grid = payload.get("candidate_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("TRACE calibration candidate grid is missing")
    lambda_grid = tuple(float(item) for item in grid.get("lambda_ratio", ()))
    roughness_grid = tuple(float(item) for item in grid.get("roughness", ()))
    if lambda_grid != (0.03, 0.05, 0.08, 0.15, 0.2, 0.35):
        raise ValueError("unexpected TRACE lambda-ratio grid")
    if roughness_grid != (0.0, 0.01, 0.05):
        raise ValueError("unexpected TRACE roughness grid")

    summaries = payload.get("candidate_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(lambda_grid) * len(
        roughness_grid
    ):
        raise ValueError("TRACE candidate summaries are incomplete")
    by_pair: dict[tuple[float, float], Mapping[str, object]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("TRACE candidate summary must be an object")
        pair = (float(summary["lambda_ratio"]), float(summary["roughness"]))
        if pair in by_pair:
            raise ValueError("duplicate TRACE candidate summary")
        if int(summary.get("evaluation_count", -1)) != EXPECTED_EVALUATIONS_PER_CANDIDATE:
            raise ValueError("TRACE candidate does not contain all 15 pilot evaluations")
        if int(summary.get("successful_count", -1)) != EXPECTED_EVALUATIONS_PER_CANDIDATE:
            raise ValueError("TRACE candidate has a failed pilot evaluation")
        loss = float(summary.get("mean_validation_huber_loss", float("nan")))
        if not np.isfinite(loss):
            raise ValueError("TRACE candidate mean loss is not finite")
        by_pair[pair] = summary
    expected_pairs = {
        (lambda_ratio, roughness)
        for lambda_ratio in lambda_grid
        for roughness in roughness_grid
    }
    if set(by_pair) != expected_pairs:
        raise ValueError("TRACE candidate summary pairs do not match the locked grid")

    evaluations = payload.get("evaluations")
    expected_evaluations = len(expected_pairs) * EXPECTED_EVALUATIONS_PER_CANDIDATE
    if not isinstance(evaluations, list) or len(evaluations) != expected_evaluations:
        raise ValueError("TRACE calibration does not contain every candidate loss")

    selected = payload.get("selected_pair")
    if not isinstance(selected, Mapping):
        raise ValueError("TRACE selected pair is missing")
    selected_pair = (float(selected["lambda_ratio"]), float(selected["roughness"]))
    minimizer = min(
        by_pair,
        key=lambda pair: (
            float(by_pair[pair]["mean_validation_huber_loss"]),
            pair[0],
            pair[1],
        ),
    )
    if selected_pair != minimizer:
        raise ValueError("TRACE selected pair is not the registered global loss minimizer")

    declared_hash = str(payload.get("content_sha256", ""))
    computed_hash = trace_tuning_content_sha256(payload)
    if len(declared_hash) != 64 or declared_hash != computed_hash:
        raise ValueError("TRACE tuning content hash mismatch")
    return {
        "lambda_ratio": selected_pair[0],
        "roughness": selected_pair[1],
        "mean_validation_huber_loss": float(
            by_pair[selected_pair]["mean_validation_huber_loss"]
        ),
        "content_sha256": declared_hash,
        "tuning_mode": TUNING_MODE,
    }


@lru_cache(maxsize=8)
def _load_trace_tuning_lock_cached(resolved_text: str) -> dict[str, object]:
    resolved = Path(resolved_text)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("TRACE tuning protocol root must be an object")
    lock = validate_trace_tuning_payload(payload)
    return {
        **lock,
        "path": str(resolved.resolve()),
        "file_sha256": _file_sha256(resolved),
    }


def load_trace_tuning_lock(path: Path | None = None) -> dict[str, object]:
    """Load the immutable calibration artifact used by both production runners."""

    resolved = (DEFAULT_TRACE_TUNING_PATH if path is None else Path(path)).resolve()
    return dict(_load_trace_tuning_lock_cached(str(resolved)))

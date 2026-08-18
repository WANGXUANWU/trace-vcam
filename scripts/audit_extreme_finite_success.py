"""Flag numerically extreme but still finite successful benchmark fits.

This is deliberately a *post-run, read-only* audit.  It never rewrites the
benchmark CSV, never changes a fit's ``attempt_status`` or ``converged``
field, and never decides which rows enter the primary analysis.  Its purpose
is to make visible successful numerical endpoints that merit robust-summary
and presentation review after the locked formal run has finished.

By default the audit covers every method observed in the benchmark; callers
may restrict it with repeatable ``--method`` options.  A row is flagged only
when it is already recorded as a successful, converged fit and has at least
one finite metric that is either (i) above an explicit, documented absolute
threshold or (ii) a robust log-scale outlier within its
scenario/method/metric group.  The raw benchmark remains authoritative; a
flag is not a failure code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "vcam-extreme-finite-success-audit/2"
METRICS = (
    "baseline_ise",
    "component_ise",
    "factor_ise",
    "noise_free_test_mspe",
    "observed_test_mspe",
    "test_mse",
)
IDENTITY_COLUMNS = (
    "example",
    "protocol",
    "scenario",
    "replicate",
    "seed",
    "method",
    "method_display_name",
    "method_version",
    "attempt_status",
    "converged",
    "failure_code",
)
OUTPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "extreme_finite_success_audit_flag",
    "flagged_metrics_json",
    "flag_rules_json",
    "metadata_numerical_mrs_increase",
    "fit_metadata_parse_error",
)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_successful_converged(row: Mapping[str, str]) -> bool:
    return (
        row.get("attempt_status") == "success"
        and str(row.get("converged", "")).strip().lower() == "true"
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_progress(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"progress journal is not an object: {path}")
    return payload


def _completed_progress(progress: Mapping[str, object]) -> bool:
    committed = progress.get("committed_cohorts")
    expected = progress.get("expected_cohorts")
    return isinstance(committed, int) and isinstance(expected, int) and committed == expected


def _robust_threshold(values: Sequence[float], z_limit: float, minimum_group_size: int) -> dict[str, float] | None:
    """Return a deterministic log-scale MAD threshold, or ``None`` if inapplicable."""

    if len(values) < minimum_group_size:
        return None
    transformed = [math.log1p(abs(value)) for value in values]
    center = median(transformed)
    mad = median(abs(value - center) for value in transformed)
    if not math.isfinite(mad) or mad <= 0:
        return None
    scale = 1.4826 * mad
    return {
        "log1p_abs_median": center,
        "log1p_abs_mad": mad,
        "log1p_abs_threshold": center + z_limit * scale,
        "robust_z_limit": z_limit,
        "n": float(len(values)),
    }


def _metadata_indicator(row: Mapping[str, str]) -> tuple[bool | None, bool]:
    """Return ``(numerical_mrs_increase, parse_error)`` without changing the row."""

    raw = row.get("fit_metadata_json", "")
    if not raw:
        return None, False
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, True
    if not isinstance(payload, dict):
        return None, True
    value = payload.get("numerical_mrs_increase")
    return value if isinstance(value, bool) else None, False


def flag_extreme_finite_successes(
    rows: Iterable[Mapping[str, str]],
    *,
    method: str | None = None,
    methods: Sequence[str] | None = None,
    absolute_threshold: float = 1_000.0,
    robust_z_limit: float = 8.0,
    minimum_group_size: int = 20,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return row-level flags and a transparent rule summary.

    Absolute and robust tests are intentionally additive.  The fixed absolute
    threshold makes an overflow-like result auditable even in a small group;
    the log-MAD rule catches material scenario-relative excursions without
    silently reclassifying a converged fit as a failure.
    """

    if not math.isfinite(absolute_threshold) or absolute_threshold <= 0:
        raise ValueError("absolute_threshold must be a positive finite number")
    if not math.isfinite(robust_z_limit) or robust_z_limit <= 0:
        raise ValueError("robust_z_limit must be a positive finite number")
    if minimum_group_size < 3:
        raise ValueError("minimum_group_size must be at least 3")

    materialized_rows = [dict(row) for row in rows]
    if method is not None and methods is not None:
        raise ValueError("pass either method or methods, not both")
    requested = (method,) if method is not None else methods
    if requested is None:
        requested = tuple(
            sorted(
                {
                    str(row.get("method", "")).strip()
                    for row in materialized_rows
                    if str(row.get("method", "")).strip()
                }
            )
        )
    method_scope = tuple(
        dict.fromkeys(str(item).strip() for item in requested if str(item).strip())
    )
    if not method_scope:
        raise ValueError("at least one nonempty method must be selected")
    method_set = set(method_scope)

    candidates = [
        dict(row)
        for row in materialized_rows
        if str(row.get("method", "")) in method_set and _is_successful_converged(row)
    ]
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in candidates:
        scenario = row.get("scenario", "")
        row_method = str(row.get("method", ""))
        for metric in METRICS:
            value = _finite_float(row.get(metric))
            if value is not None:
                groups[(scenario, row_method, metric)].append(value)
    thresholds = {
        key: _robust_threshold(values, robust_z_limit, minimum_group_size)
        for key, values in groups.items()
    }

    flags: list[dict[str, object]] = []
    metric_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    for row in candidates:
        scenario = row.get("scenario", "")
        row_method = str(row.get("method", ""))
        flagged_metrics: dict[str, float] = {}
        rule_payload: dict[str, dict[str, object]] = {}
        for metric in METRICS:
            value = _finite_float(row.get(metric))
            if value is None:
                continue
            rules: list[str] = []
            detail: dict[str, object] = {
                "value": value,
                "absolute_threshold": absolute_threshold,
            }
            if abs(value) >= absolute_threshold:
                rules.append("absolute_magnitude")
            threshold = thresholds.get((scenario, row_method, metric))
            if threshold is not None:
                transformed = math.log1p(abs(value))
                detail["robust_log1p_abs"] = transformed
                detail["robust_threshold"] = threshold
                if transformed > float(threshold["log1p_abs_threshold"]):
                    rules.append("scenario_log_mad_outlier")
            if rules:
                flagged_metrics[metric] = value
                detail["rules"] = rules
                rule_payload[metric] = detail
                metric_counts[metric] += 1
                rule_counts.update(rules)
        if not flagged_metrics:
            continue
        numerical_mrs_increase, metadata_parse_error = _metadata_indicator(row)
        record: dict[str, object] = {
            key: row.get(key, "") for key in IDENTITY_COLUMNS
        }
        record.update(
            {
                "extreme_finite_success_audit_flag": True,
                "flagged_metrics_json": json.dumps(flagged_metrics, sort_keys=True),
                "flag_rules_json": json.dumps(rule_payload, sort_keys=True),
                "metadata_numerical_mrs_increase": numerical_mrs_increase,
                "fit_metadata_parse_error": metadata_parse_error,
            }
        )
        flags.append(record)

    candidates_by_method = {
        selected: sum(str(row.get("method", "")) == selected for row in candidates)
        for selected in method_scope
    }
    flagged_by_method = {
        selected: sum(str(row.get("method", "")) == selected for row in flags)
        for selected in method_scope
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            # Keep the singular field for a simple backwards-readable record
            # when callers intentionally audit just one method.
            "method": method_scope[0] if len(method_scope) == 1 else None,
            "methods": list(method_scope),
            "success_definition": "attempt_status == success and converged == True",
            "metrics": list(METRICS),
        },
        "rules": {
            "absolute_magnitude_threshold": absolute_threshold,
            "robust_rule": "log1p(abs(metric)) > scenario/method/metric median + z * 1.4826 * MAD",
            "robust_z_limit": robust_z_limit,
            "minimum_group_size": minimum_group_size,
            "semantics": "flags do not alter convergence, status, or inclusion; they trigger post-run review",
        },
        "candidate_successful_converged_rows": len(candidates),
        "candidate_successful_converged_rows_by_method": candidates_by_method,
        "flagged_rows": len(flags),
        "flagged_rows_by_method": flagged_by_method,
        "flagged_metrics": dict(sorted(metric_counts.items())),
        "flagged_rule_counts": dict(sorted(rule_counts.items())),
    }
    return flags, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Formal strict_results.csv to inspect read-only.")
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="strict_progress.json; inferred next to --results when omitted.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Directory for sidecar audit files.")
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=None,
        help="Method identifier to audit; repeat to audit several methods. Omit to audit every observed method.",
    )
    parser.add_argument("--absolute-threshold", type=float, default=1_000.0)
    parser.add_argument("--robust-z-limit", type=float, default=8.0)
    parser.add_argument("--minimum-group-size", type=int, default=20)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit an exploratory read before the formal progress journal is complete.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the audit summary and write no files.")
    args = parser.parse_args()
    if args.dry_run and args.output is not None:
        parser.error("--dry-run and --output cannot be used together")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    return args


def main() -> int:
    args = _parse_args()
    results = args.results.resolve()
    if not results.is_file():
        raise FileNotFoundError(results)
    progress = (args.progress or results.parent / "strict_progress.json").resolve()
    if not progress.is_file():
        raise FileNotFoundError(progress)
    progress_payload = _read_progress(progress)
    if not args.allow_incomplete and not _completed_progress(progress_payload):
        raise RuntimeError(
            "formal run is not complete; rerun after committed_cohorts equals expected_cohorts "
            "or use --allow-incomplete for an explicitly exploratory read"
        )

    flags, summary = flag_extreme_finite_successes(
        _read_rows(results),
        methods=args.methods,
        absolute_threshold=args.absolute_threshold,
        robust_z_limit=args.robust_z_limit,
        minimum_group_size=args.minimum_group_size,
    )
    summary["inputs"] = {
        "results": str(results),
        "results_sha256": _sha256(results),
        "progress": str(progress),
        "progress_sha256": _sha256(progress),
        "progress_complete": _completed_progress(progress_payload),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }
    if args.dry_run:
        # The workspace path contains Chinese characters; ASCII console output
        # keeps this audit usable from legacy Windows code pages as well.
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    if output == results.parent.resolve():
        raise ValueError("audit sidecars must be outside the benchmark result directory")
    csv_path = output / "extreme_finite_success_rows.csv"
    json_path = output / "extreme_finite_success_audit.json"
    _atomic_write_csv(csv_path, flags)
    summary["outputs"] = {"rows_csv": str(csv_path), "rows_csv_sha256": _sha256(csv_path)}
    provisional = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    summary["audit_payload_sha256"] = hashlib.sha256(provisional.encode("utf-8")).hexdigest()
    _atomic_write_text(json_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows_csv": str(csv_path), "audit_json": str(json_path), **summary}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

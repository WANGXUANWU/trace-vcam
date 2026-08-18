"""Audit strict VCAM outputs and create publication-scale tables/figures.

The script always produces an audit report.  Exploratory tables and vector PDF
figures may be produced from a quick or incomplete run, but numerical claim
macros are enabled only when the complete formal protocol and all input hashes
pass.  It never promotes a missing/failed external fit into a finite result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.methods import FIXED_METHOD_LABELS, MethodLabel  # noqa: E402


SCHEMA_STRICT = "vcam-strict-benchmark/1"
SCHEMA_MACS = "vcam-macs-application/1"
SCHEMA_EXTREME_AUDIT = "vcam-extreme-finite-success-audit/2"
LEGACY_METHODS = {"LS-ALS", "Huber-ALS", "Coefficient-L1", "Tensor-ridge"}
MANUSCRIPT_TABLES = (
    "example1_main.tex",
    "example2_main.tex",
    "example3_main.tex",
    "scaling_main.tex",
    "macs_cv_main.tex",
    "method_admission.tex",
    "example1_full.tex",
    "example2_full.tex",
    "example3_full.tex",
    "scaling_full.tex",
    "failure_audit.tex",
    "extreme_finite_audit.tex",
    "result_manifest.tex",
    "macs_sensitivity.tex",
)
MANUSCRIPT_FIGURES = (
    "example1_factor_recovery.pdf",
    "example2_robustness.pdf",
    "macs_components.pdf",
    "macs_surfaces.pdf",
    "supp_example1_components.pdf",
    "supp_example2_distributions.pdf",
    "supp_example3_selection.pdf",
)
EXTERNAL_R_METHODS = {"ZW2015", "ZSY2026-author-code"}
ATTEMPT_STATUSES = {"success", "failed"}
PREDICTION_METRICS = {
    "observed_test_mspe",
    "noise_free_test_mspe",
    "test_mse",
    "subject_balanced_test_mse",
}
EXTREME_AUDIT_METRICS = {
    "baseline_ise",
    "component_ise",
    "factor_ise",
    "noise_free_test_mspe",
    "observed_test_mspe",
    "test_mse",
}
EXTREME_METRIC_LABELS = {
    "baseline_ise": "baseline ISE",
    "component_ise": "component ISE",
    "factor_ise": "factor ISE",
    "noise_free_test_mspe": "noise-free MSPE",
    "observed_test_mspe": "observed MSPE",
    "test_mse": "test MSE",
}
FIGURE_WIDTH_IN = 6.15
MIN_SOURCE_FONT_PT = 10.5

# These widths deliberately leave room for inter-column padding at the normal
# manuscript font size.  They are shared by the main and supplemental output,
# so a machine-generated number cannot silently push a table beyond the page.
ESTIMATION_TABLE_ALIGNMENT = (
    r"@{}L{0.13\linewidth}L{0.20\linewidth}C{0.19\linewidth}"
    r"C{0.19\linewidth}C{0.16\linewidth}@{}"
)
PREDICTION_TABLE_ALIGNMENT = ESTIMATION_TABLE_ALIGNMENT
SCALING_TABLE_ALIGNMENT = (
    r"@{}C{0.04\linewidth}L{0.19\linewidth}C{0.18\linewidth}"
    r"C{0.28\linewidth}C{0.14\linewidth}@{}"
)
MACS_TABLE_ALIGNMENT = (
    r"@{}L{0.18\linewidth}C{0.17\linewidth}C{0.20\linewidth}"
    r"C{0.19\linewidth}C{0.12\linewidth}@{}"
)
METHOD_ADMISSION_ALIGNMENT = (
    r"@{}L{0.23\linewidth}L{0.18\linewidth}L{0.34\linewidth}"
    r"C{0.11\linewidth}@{}"
)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": MIN_SOURCE_FONT_PT,
        "ytick.labelsize": MIN_SOURCE_FONT_PT,
        "legend.fontsize": MIN_SOURCE_FONT_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite_success(rows: Iterable[Mapping[str, object]], metric: str) -> np.ndarray:
    values = [
        _float(row.get(metric))
        for row in rows
        if row.get("attempt_status") == "success" and _bool(row.get("converged"))
    ]
    return np.asarray([value for value in values if np.isfinite(value)], dtype=float)


def _attempted_rows(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [row for row in rows if str(row.get("attempt_status")) in ATTEMPT_STATUSES]


def _finite_success_count(rows: Iterable[Mapping[str, object]], metric: str) -> int:
    return len(_finite_success(rows, metric))


def _fit_metadata(row: Mapping[str, object]) -> Mapping[str, object]:
    raw = row.get("fit_metadata_json", "")
    if isinstance(raw, Mapping):
        return raw
    try:
        payload = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _number(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    absolute = abs(value)
    if absolute != 0 and (absolute < 0.001 or absolute >= 10000):
        return f"{value:.2e}"
    if absolute < 0.1:
        return f"{value:.4f}"
    if absolute < 100:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _mean_mcse(values: np.ndarray) -> str:
    if not len(values):
        return "--"
    mcse = np.std(values, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return f"{_number(float(np.mean(values)))} ({_number(float(mcse))})"


def _median_iqr(values: np.ndarray) -> str:
    if not len(values):
        return "--"
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return f"{_number(float(median))} [{_number(float(q1))}, {_number(float(q3))}]"


def _tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _sha256_tex(value: str) -> str:
    """Allow a machine hash to wrap cleanly without changing its value."""

    # Hashes have no natural whitespace.  Break only between fixed eight-digit
    # groups, preserving exact copy/paste reconstruction while keeping the
    # audit manifest inside the page at the normal manuscript font size.
    groups = [value[index : index + 8] for index in range(0, len(value), 8)]
    return r"\texttt{" + r"\VCAMHashBreak{}".join(groups) + "}"


def _method_tex(method: str) -> str:
    return {
        "TRACE-VCAM": r"TRACE-VCAM",
        "ZW2015": r"Two-step spline VCAM (Zhang \& Wang, 2015)",
        "ZZW2020": r"Backfitting VCAM (Zhang et al., 2020)",
        "HHY2021-Huber": r"Three-step M-VCAM (Hu et al., 2021)",
        "ZSY2026-author-code": r"VCAM-Lasso (Zhao et al., 2026)",
        "ZY2025-paper-implementation": r"Penalized robust VCAM (Zhao \& Yang, 2025)",
    }.get(method, _tex_escape(method))


def _method_short_tex(method: str) -> str:
    return {
        "TRACE-VCAM": r"TRACE-VCAM",
        "ZW2015": r"Two-step spline VCAM",
        "ZZW2020": r"Backfitting VCAM",
        "HHY2021-Huber": r"Three-step M-VCAM",
        "ZSY2026-author-code": r"VCAM-Lasso",
        "ZY2025-paper-implementation": r"Penalized robust VCAM",
    }.get(method, _tex_escape(method))


def _method_display(method: str) -> str:
    return {
        "TRACE-VCAM": "TRACE-VCAM",
        "ZW2015": "Two-step spline VCAM",
        "ZZW2020": "Backfitting VCAM",
        "HHY2021-Huber": "Three-step M-VCAM",
        "ZSY2026-author-code": "VCAM-Lasso",
        "ZY2025-paper-implementation": "Penalized robust VCAM",
    }.get(method, method)


def _method_plot_label(method: str) -> str:
    return {
        "TRACE-VCAM": "TRACE-VCAM",
        "ZW2015": "Two-step spline\nVCAM",
        "ZZW2020": "Backfitting\nVCAM",
        "HHY2021-Huber": "Three-step\nM-VCAM",
        "ZSY2026-author-code": "VCAM-Lasso",
        "ZY2025-paper-implementation": "Penalized robust\nVCAM",
    }.get(method, method)


def _humanize_admission_basis(value: object) -> str:
    text = str(value or "same_setting_original_method_comparison")
    return {
        "same_setting_original_method_comparison": (
            "Original method on the same generated data, seed, and subject split"
        ),
        "author_code": "Pinned author implementation",
        "paper_implementation": "Implementation faithful to the published algorithm",
    }.get(text, text.replace("_", " ").strip().capitalize())


def _humanize_admission_status(value: object) -> str:
    text = str(value or "missing")
    return {
        "admitted": "Compared",
        "passed": "Compared",
        "N/A by design": "N/A by design",
        "not_applicable": "N/A by design",
        "missing": "Not recorded",
    }.get(text, text.replace("_", " ").strip().capitalize())


def _variant_tex(variant: str) -> str:
    return {
        "primary": "Primary analysis",
        "delete_outer_fence_subjects": "Delete outer-fence subjects",
        "winsorize_response_1_99": r"Response winsorized at 1\%/99\%",
        "basis_5": "Five basis functions",
        "basis_8": "Eight basis functions",
    }.get(variant, _tex_escape(variant.replace("_", " ").capitalize()))


def _implementation_summary(method: str, value: object) -> str:
    """Keep source identity readable; exact versions remain in audited metadata."""

    text = str(value or "registered source")
    if method == "ZW2015":
        package = next(
            (part.strip() for part in text.split(";") if "fdapace" in part.lower()),
            "CRAN fdapace",
        )
        r_version = text.split(";")[0].replace("R version ", "R ").split(" (")[0]
        return f"{package} ({r_version})"
    if method == "ZZW2020":
        return "Published Algorithm 1"
    if method == "HHY2021-Huber":
        return "Published three-step algorithm"
    if method == "ZSY2026-author-code":
        package = text.split(";")[0]
        if "@" in package:
            name, commit = package.split("@", 1)
            return f"{name}, author commit {commit[:8]}"
        return "Pinned author implementation"
    if method == "ZY2025-paper-implementation":
        return "Published penalized estimating equations"
    return text


def _status_caption(caption: str, claims_eligible: bool) -> str:
    if claims_eligible:
        return caption
    return caption + " (exploratory output; the formal audit has not passed)."


def _write_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: Sequence[str],
    alignment: str,
    body: Sequence[Sequence[str]],
    claims_eligible: bool,
) -> None:
    path.write_text(
        "\n".join(
            _table_lines(
                caption=caption,
                label=label,
                columns=columns,
                alignment=alignment,
                body=body,
                claims_eligible=claims_eligible,
            )
        ),
        encoding="utf-8",
    )


def _table_lines(
    *,
    caption: str,
    label: str,
    columns: Sequence[str],
    alignment: str,
    body: Sequence[Sequence[str]],
    placement: str = "t",
    claims_eligible: bool,
) -> list[str]:
    lines = [
        f"\\begin{{table}}[{placement}]",
        r"\centering",
        f"\\caption{{{_status_caption(caption, claims_eligible)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in body)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def _longtable_lines(
    *,
    caption: str,
    label: str,
    columns: Sequence[str],
    alignment: str,
    body: Sequence[Sequence[str]],
    claims_eligible: bool,
) -> list[str]:
    """Create a readable, page-breaking table at the manuscript font size.

    Large registered examples contain many method-by-setting rows.  Letting
    them float as a single unbreakable ``tabular`` either crops the right edge
    or drives the table into the footer.  ``longtable`` repeats the compact
    header across pages and preserves every audited row without a font-size
    reduction or a resize box.
    """

    n_columns = len(columns)
    header = " & ".join(columns) + r" \\"
    continuation = (
        rf"\multicolumn{{{n_columns}}}{{@{{}}l@{{}}}}"
        r"{\textit{Table~\thetable\ continued}} \\"
    )
    next_page = (
        rf"\multicolumn{{{n_columns}}}{{@{{}}r@{{}}}}"
        r"{\textit{Continued on next page}} \\"
    )
    lines = [
        f"\\begin{{longtable}}{{{alignment}}}",
        f"\\caption{{{_status_caption(caption, claims_eligible)}}}\\label{{{label}}}\\\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        continuation,
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        next_page,
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in body)
    lines.extend([r"\end{longtable}", ""])
    return lines


def _write_longtable(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: Sequence[str],
    alignment: str,
    body: Sequence[Sequence[str]],
    claims_eligible: bool,
) -> None:
    path.write_text(
        "\n".join(
            _longtable_lines(
                caption=caption,
                label=label,
                columns=columns,
                alignment=alignment,
                body=body,
                claims_eligible=claims_eligible,
            )
        ),
        encoding="utf-8",
    )


def _write_split_tables(
    path: Path,
    *,
    blocks: Sequence[
        tuple[str, str, Sequence[str], str, Sequence[Sequence[str]]]
    ],
    claims_eligible: bool,
) -> None:
    """Write page-sized table blocks into one includable supplement file."""

    lines: list[str] = []
    for caption, label, columns, alignment, body in blocks:
        # A page-breaking table is necessary once the readable fixed-width
        # cells occupy more than one ordinary float page.  Short audit tables
        # remain ordinary floats to keep their compact supplement placement.
        if len(body) > 10:
            lines.extend(
                _longtable_lines(
                    caption=caption,
                    label=label,
                    columns=columns,
                    alignment=alignment,
                    body=body,
                    claims_eligible=claims_eligible,
                )
            )
        else:
            lines.extend(
                _table_lines(
                    caption=caption,
                    label=label,
                    columns=columns,
                    alignment=alignment,
                    body=body,
                    placement="p",
                    claims_eligible=claims_eligible,
                )
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_strict(
    rows: Sequence[Mapping[str, object]], metadata: Mapping[str, object], results_path: Path
) -> dict[str, object]:
    issues: list[str] = []
    if metadata.get("schema_version") != SCHEMA_STRICT:
        issues.append("strict metadata schema mismatch")
    if not rows or {row.get("schema_version") for row in rows} != {SCHEMA_STRICT}:
        issues.append("strict result schema mismatch or empty file")
    file_record = dict(metadata.get("files", {})).get("results", {})
    if not isinstance(file_record, Mapping) or file_record.get("sha256") != file_sha256(results_path):
        issues.append("strict_results.csv SHA256 mismatch")
    methods = {str(row.get("method")) for row in rows}
    registered = set(str(item) for item in metadata.get("method_order", FIXED_METHOD_LABELS))
    if methods - registered:
        issues.append(f"unregistered methods present: {sorted(methods - registered)}")
    if methods & LEGACY_METHODS:
        issues.append(f"legacy self-designed baselines present: {sorted(methods & LEGACY_METHODS)}")

    registry = metadata.get("scenario_registry", [])
    replications = metadata.get("replications", {})
    if not isinstance(registry, list) or not isinstance(replications, Mapping):
        issues.append("scenario registry/replication map is absent")
        registry = []
        replications = {}
    expected_keys: set[tuple[str, int, str]] = set()
    for item in registry:
        if not isinstance(item, Mapping):
            continue
        scenario = str(item.get("scenario"))
        count = _int(replications.get(scenario), 0)
        expected_formal = _int(item.get("formal_replications"), -1)
        if metadata.get("mode") == "formal" and count != expected_formal:
            issues.append(f"{scenario}: replication count {count} != locked {expected_formal}")
        methods_here = [str(item.get("owner"))] if item.get("phase") == "reproduction" else sorted(registered)
        expected_keys.update((scenario, replicate, method) for replicate in range(count) for method in methods_here)
    observed_keys = [
        (str(row.get("scenario")), _int(row.get("replicate")), str(row.get("method")))
        for row in rows
    ]
    if len(observed_keys) != len(set(observed_keys)):
        issues.append("duplicate scenario/replicate/method keys")
    missing = expected_keys - set(observed_keys)
    extra = set(observed_keys) - expected_keys
    if missing:
        issues.append(f"missing {len(missing)} registered result rows")
    if extra:
        issues.append(f"found {len(extra)} unexpected result rows")

    cohorts: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        cohorts[(str(row.get("scenario")), _int(row.get("replicate")))].append(row)
    for key, cohort in cohorts.items():
        for field in ("seed", "split_seed", "data_hash", "train_subject_hash", "test_subject_hash"):
            if len({str(row.get(field)) for row in cohort}) != 1:
                issues.append(f"{key}: methods do not share {field}")
        attempted = [
            row
            for row in cohort
            if row.get("attempt_status") in {"success", "failed"}
        ]
        for row in attempted:
            if not np.isfinite(_float(row.get("runtime_seconds"))):
                issues.append(f"{key}/{row.get('method')}: attempted fit lacks finite runtime")
            if row.get("attempt_status") == "success":
                required_metrics = (
                    ("paper_training_function_mse_total", "factor_ise", "component_ise")
                    if row.get("method") == "ZSY2026-author-code"
                    else ("observed_test_mspe", "noise_free_test_mspe")
                )
                for metric in required_metrics:
                    if not np.isfinite(_float(row.get(metric))):
                        issues.append(f"{key}/{row.get('method')}: success lacks finite {metric}")
    cohort_audit = metadata.get("cohort_audit", {})
    if not isinstance(cohort_audit, Mapping) or cohort_audit.get("passed") is not True:
        issues.append("runner cohort audit did not pass")
    gates = metadata.get("admission_gates", {})
    claim_blockers: list[str] = []
    if not isinstance(gates, Mapping):
        issues.append("admission gate registry absent")
    else:
        for method, decision in gates.items():
            if not isinstance(decision, Mapping) or "passed" not in decision:
                issues.append(f"malformed reproduction gate: {method}")
            elif decision.get("passed") is not True:
                claim_blockers.append(
                    f"{method}: {decision.get('status', 'not_admitted')}"
                )
    if metadata.get("mode") != "formal":
        issues.append("strict run is not formal")
    if metadata.get("formal_protocol_complete") is not True:
        issues.append("runner did not mark the formal protocol complete")
    return {
        "schema": SCHEMA_STRICT,
        "passed": not issues,
        "issues": issues,
        "n_rows": len(rows),
        "n_expected_rows": len(expected_keys),
        "results_sha256": file_sha256(results_path),
        "claim_blockers": claim_blockers,
    }


def audit_extreme_sidecar(
    strict_rows: Sequence[Mapping[str, object]],
    strict_results: Path,
    audit_json_path: Path,
    audit_rows_path: Path,
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, object]]:
    """Fail closed unless an all-method extreme-audit sidecar matches strict data."""

    issues: list[str] = []
    try:
        summary = _read_json(audit_json_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (
            {
                "passed": False,
                "issues": [f"unable to read extreme-audit JSON: {error}"],
                "audit_json_sha256": None,
                "audit_rows_sha256": None,
            },
            [],
            {},
        )
    try:
        audit_rows = _read_csv(audit_rows_path)
    except (OSError, ValueError, csv.Error) as error:
        return (
            {
                "passed": False,
                "issues": [f"unable to read extreme-audit CSV: {error}"],
                "audit_json_sha256": file_sha256(audit_json_path),
                "audit_rows_sha256": None,
            },
            [],
            summary,
        )

    if summary.get("schema_version") != SCHEMA_EXTREME_AUDIT:
        issues.append("extreme-audit schema mismatch")
    recorded_payload_hash = str(summary.get("audit_payload_sha256", ""))
    payload_without_hash = dict(summary)
    payload_without_hash.pop("audit_payload_sha256", None)
    expected_payload_hash = hashlib.sha256(
        (json.dumps(payload_without_hash, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    if recorded_payload_hash != expected_payload_hash:
        issues.append("extreme-audit JSON self-hash mismatch")

    inputs = summary.get("inputs", {})
    if not isinstance(inputs, Mapping):
        issues.append("extreme-audit inputs are absent")
        inputs = {}
    strict_hash = file_sha256(strict_results)
    if inputs.get("results_sha256") != strict_hash:
        issues.append("extreme-audit results SHA256 does not match strict_results.csv")
    progress_path = strict_results.parent / "strict_progress.json"
    if not progress_path.is_file():
        issues.append("strict progress journal is absent for extreme-audit validation")
    else:
        if inputs.get("progress_sha256") != file_sha256(progress_path):
            issues.append("extreme-audit progress SHA256 does not match strict_progress.json")
        try:
            progress = _read_json(progress_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"unable to read strict progress journal: {error}")
        else:
            if progress.get("committed_cohorts") != progress.get("expected_cohorts"):
                issues.append("strict progress journal is incomplete")
    if inputs.get("progress_complete") is not True:
        issues.append("extreme audit was not run after progress completion")
    audit_script = ROOT / "scripts" / "audit_extreme_finite_success.py"
    if inputs.get("audit_script_sha256") != file_sha256(audit_script):
        issues.append("extreme-audit script SHA256 does not match the current audit implementation")

    scope = summary.get("scope", {})
    if not isinstance(scope, Mapping):
        issues.append("extreme-audit scope is absent")
        scope = {}
    methods = [str(method) for method in scope.get("methods", [])]
    if len(methods) != len(set(methods)) or set(methods) != set(FIXED_METHOD_LABELS):
        issues.append("extreme audit does not cover every registered method exactly once")
    if set(str(metric) for metric in scope.get("metrics", [])) != EXTREME_AUDIT_METRICS:
        issues.append("extreme-audit metric scope differs from the registered metric set")
    if scope.get("success_definition") != "attempt_status == success and converged == True":
        issues.append("extreme-audit success definition mismatch")

    outputs = summary.get("outputs", {})
    if not isinstance(outputs, Mapping):
        issues.append("extreme-audit output hashes are absent")
        outputs = {}
    audit_rows_hash = file_sha256(audit_rows_path)
    if outputs.get("rows_csv_sha256") != audit_rows_hash:
        issues.append("extreme-audit flagged-row CSV SHA256 mismatch")

    raw_index = {
        (str(row.get("scenario")), str(row.get("replicate")), str(row.get("method"))): row
        for row in strict_rows
    }
    seen_flags: set[tuple[str, str, str]] = set()
    flagged_by_method: Counter[str] = Counter()
    for record in audit_rows:
        key = (
            str(record.get("scenario", "")),
            str(record.get("replicate", "")),
            str(record.get("method", "")),
        )
        if key in seen_flags:
            issues.append(f"duplicate extreme-audit flag identity: {key}")
            continue
        seen_flags.add(key)
        raw = raw_index.get(key)
        if raw is None:
            issues.append(f"extreme-audit flag does not map to strict result: {key}")
            continue
        if raw.get("attempt_status") != "success" or not _bool(raw.get("converged")):
            issues.append(f"extreme-audit flag is not a successful converged strict fit: {key}")
        if not _bool(record.get("extreme_finite_success_audit_flag")):
            issues.append(f"extreme-audit row lacks its flag marker: {key}")
        try:
            flagged_metrics = json.loads(str(record.get("flagged_metrics_json", "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            flagged_metrics = None
        if not isinstance(flagged_metrics, Mapping) or not flagged_metrics:
            issues.append(f"extreme-audit row lacks parseable flagged metrics: {key}")
        else:
            for metric, value in flagged_metrics.items():
                if str(metric) not in EXTREME_AUDIT_METRICS:
                    issues.append(f"extreme-audit row has unregistered metric {metric}: {key}")
                    continue
                raw_value = _float(raw.get(str(metric)))
                flagged_value = _float(value)
                if not np.isfinite(raw_value) or not np.isfinite(flagged_value) or not math.isclose(
                    raw_value, flagged_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    issues.append(f"extreme-audit metric does not match strict result: {key}/{metric}")
        flagged_by_method[key[2]] += 1

    if _int(summary.get("flagged_rows"), -1) != len(audit_rows):
        issues.append("extreme-audit flagged-row count mismatch")
    candidates_raw = summary.get("candidate_successful_converged_rows_by_method", {})
    reported_candidates = candidates_raw if isinstance(candidates_raw, Mapping) else {}
    reported_flags_raw = summary.get("flagged_rows_by_method", {})
    reported_flags = reported_flags_raw if isinstance(reported_flags_raw, Mapping) else {}
    for method in FIXED_METHOD_LABELS:
        expected_candidates = sum(
            row.get("method") == method
            and row.get("attempt_status") == "success"
            and _bool(row.get("converged"))
            for row in strict_rows
        )
        if _int(reported_candidates.get(method), -1) != expected_candidates:
            issues.append(f"extreme-audit candidate count mismatch for {method}")
        if _int(reported_flags.get(method), -1) != flagged_by_method[method]:
            issues.append(f"extreme-audit flagged-row count mismatch for {method}")
    return (
        {
            "schema": SCHEMA_EXTREME_AUDIT,
            "passed": not issues,
            "issues": issues,
            "audit_json_sha256": file_sha256(audit_json_path),
            "audit_rows_sha256": audit_rows_hash,
            "n_flag_rows": len(audit_rows),
            "scope_methods": methods,
        },
        audit_rows,
        summary,
    )


def audit_macs(
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    results_path: Path,
    curves_path: Path | None = None,
) -> dict[str, object]:
    issues: list[str] = []
    if metadata.get("schema_version") != SCHEMA_MACS:
        issues.append("MACS metadata schema mismatch")
    if not rows or {row.get("schema_version") for row in rows} != {SCHEMA_MACS}:
        issues.append("MACS result schema mismatch or empty file")
    file_record = dict(metadata.get("files", {})).get("results", {})
    if not isinstance(file_record, Mapping) or file_record.get("sha256") != file_sha256(results_path):
        issues.append("macs_results.csv SHA256 mismatch")
    curves_sha256: str | None = None
    curve_rows: list[dict[str, object]] = []
    curve_record = dict(metadata.get("files", {})).get("curves", {})
    if curves_path is None:
        issues.append("registered MACS full-data factor curves were not supplied")
    elif not curves_path.is_file():
        issues.append("registered MACS full-data factor-curve file is absent")
    else:
        curves_sha256 = file_sha256(curves_path)
        if not isinstance(curve_record, Mapping) or curve_record.get("sha256") != curves_sha256:
            issues.append("macs_factor_curves.jsonl SHA256 mismatch")
        try:
            curve_rows = _read_jsonl(curves_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"MACS full-data factor curves are unreadable: {error}")
    registered_curve_fit = metadata.get("registered_curve_fit", {})
    curve_protocol = metadata.get("curve_protocol", {})
    if not isinstance(registered_curve_fit, Mapping):
        issues.append("registered MACS full-data curve-fit audit is absent")
        registered_curve_fit = {}
    if not isinstance(curve_protocol, Mapping):
        issues.append("registered MACS curve protocol is absent")
        curve_protocol = {}
    if curve_protocol.get("aggregation") != "none":
        issues.append("MACS factor curves are not registered as a non-aggregated full-data fit")
    if curve_protocol.get("fold_curves_serialized") is not False:
        issues.append("MACS fold-level factor curves were not disabled")
    if registered_curve_fit.get("attempt_status") != "success" or registered_curve_fit.get("converged") is not True:
        issues.append("registered MACS full-data factor fit did not converge successfully")
    registered_hashes = {
        key: str(registered_curve_fit.get(key, ""))
        for key in (
            "tuning_sha256",
            "raw_curves_sha256",
            "identified_curves_sha256",
            "curve_row_sha256",
        )
    }
    for key, value in registered_hashes.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            issues.append(f"registered MACS curve fit lacks a valid {key}")
    if len(curve_rows) != 1:
        issues.append(f"MACS factor-curve file has {len(curve_rows)} rows; expected one full-data fit")
    else:
        curve_row = curve_rows[0]
        if curve_row.get("fit_scope") != "registered_full_data":
            issues.append("MACS factor-curve row is not the registered full-data fit")
        if curve_row.get("fit_id") != registered_curve_fit.get("fit_id"):
            issues.append("MACS factor-curve fit ID does not match metadata")
        if curve_row.get("data_hash") != registered_curve_fit.get("data_hash"):
            issues.append("MACS factor-curve data hash does not match metadata")
        if curve_row.get("tuning_sha256") != registered_hashes["tuning_sha256"]:
            issues.append("MACS factor-curve tuning hash does not match metadata")
        curves = curve_row.get("curves", [])
        identified_hash = _object_sha256(curves)
        if (
            curve_row.get("identified_curves_sha256") != identified_hash
            or registered_hashes["identified_curves_sha256"] != identified_hash
        ):
            issues.append("MACS identified factor-curve payload hash mismatch")
        if _object_sha256(curve_row) != registered_hashes["curve_row_sha256"]:
            issues.append("MACS full-data factor-curve row hash mismatch")
    fold_protocol = metadata.get("fold_protocol", {})
    if not isinstance(fold_protocol, Mapping):
        issues.append("MACS fold protocol absent")
        n_splits = n_repeats = 0
    else:
        n_splits = _int(fold_protocol.get("n_splits"), 0)
        n_repeats = _int(fold_protocol.get("n_repeats"), 0)
        if metadata.get("mode") == "formal" and (n_splits, n_repeats) != (5, 5):
            issues.append("formal MACS protocol is not 5 repeats by 5 folds")
    data_source = metadata.get("data_source", {})
    if not isinstance(data_source, Mapping) or (
        _int(data_source.get("n_rows")) != 2376 or _int(data_source.get("n_subjects")) != 369
    ):
        issues.append("MACS source dimensions are not 2,376 rows/369 subjects")
    if metadata.get("response_transform") != "none (raw CD4)":
        issues.append("MACS response is not registered as raw CD4")
    if metadata.get("inference") != "No confidence intervals are computed or claimed.":
        issues.append("MACS no-CI declaration missing")
    keys = [
        (
            str(row.get("variant")),
            _int(row.get("repeat")),
            _int(row.get("fold")),
            str(row.get("method")),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        issues.append("duplicate MACS variant/repeat/fold/method keys")
    cohorts: dict[tuple[str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        cohorts[(str(row.get("variant")), _int(row.get("repeat")), _int(row.get("fold")))].append(row)
    for key, cohort in cohorts.items():
        if {str(row.get("method")) for row in cohort} != set(FIXED_METHOD_LABELS):
            issues.append(f"{key}: incomplete fixed method cohort")
        for field in ("data_hash", "train_subject_hash", "test_subject_hash"):
            if len({str(row.get(field)) for row in cohort}) != 1:
                issues.append(f"{key}: methods do not share {field}")
        for row in cohort:
            if row.get("attempt_status") == "success":
                for metric in ("test_mse", "subject_balanced_test_mse", "test_mae", "runtime_seconds"):
                    if not np.isfinite(_float(row.get(metric))):
                        issues.append(f"{key}/{row.get('method')}: success lacks finite {metric}")
    cohort_audit = metadata.get("cohort_audit", {})
    if not isinstance(cohort_audit, Mapping) or cohort_audit.get("passed") is not True:
        issues.append("MACS runner cohort audit did not pass")
    if metadata.get("mode") != "formal":
        issues.append("MACS run is not formal")
    if metadata.get("formal_protocol_complete") is not True:
        issues.append("runner did not mark the formal MACS protocol complete")
    return {
        "schema": SCHEMA_MACS,
        "passed": not issues,
        "issues": issues,
        "n_rows": len(rows),
        "n_cohorts": len(cohorts),
        "results_sha256": file_sha256(results_path),
        "curves_sha256": curves_sha256,
        "n_curve_rows": len(curve_rows),
        "registered_curve_hashes": registered_hashes,
    }


def _failure_rate(rows: Sequence[Mapping[str, object]]) -> float:
    attempted = _attempted_rows(rows)
    if not attempted:
        return float("nan")
    failed = sum(
        row.get("attempt_status") != "success" or not _bool(row.get("converged"))
        for row in attempted
    )
    return failed / len(attempted)


def _group(rows: Sequence[Mapping[str, object]], **conditions: object) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]


def _scenario_setting(scenario: str) -> str:
    tokens = scenario.split("-")
    selected = [
        token
        for token in tokens
        if token.startswith("sigma")
        or (token.startswith(("n", "p")) and token[1:].isdigit())
    ]
    noise = next(
        (name for name in ("gaussian", "contamination", "mixed-normal", "t2") if name in scenario),
        "original",
    )
    return ", ".join([noise, *selected])


def _scenario_setting_tex(scenario: str) -> str:
    """Typeset the compact scenario label without allowing an overfull word."""

    value = _tex_escape(_scenario_setting(scenario))
    # The single long descriptor in the registered labels otherwise exceeds a
    # narrow first column.  A discretionary TeX hyphen keeps the printed word
    # unchanged when it fits and yields a conventional line break when it does
    # not.
    return value.replace("contamination", r"contami\-nation")


def _scenario_sample_size(scenario: str) -> int:
    return next(
        (
            int(token[1:])
            for token in scenario.split("-")
            if token.startswith("n") and token[1:].isdigit()
        ),
        -1,
    )


def _is_na_by_design(rows: Sequence[Mapping[str, object]]) -> bool:
    return bool(rows) and all(
        row.get("attempt_status") == "N/A by design" for row in rows
    )


def _has_attempt(rows: Sequence[Mapping[str, object]]) -> bool:
    return bool(_attempted_rows(rows))


def _is_na_by_capability(rows: Sequence[Mapping[str, object]]) -> bool:
    """Recognize an unmodified source's missing held-out-prediction API.

    The runners intentionally retain ``N/A by design`` as the machine-level
    status for rows that were not run.  For presentation, the MACS author-code
    case is more informative when labelled as a capability limitation: the
    source lacks an out-of-sample prediction interface, so constructing a
    surrogate predictor would alter the comparator.
    """

    if not _is_na_by_design(rows):
        return False
    # The same source limitation makes the author-code method unavailable in
    # the simulation designs too, but those entries are primarily outside the
    # method's intended design.  Reserve the capability label for MACS, where
    # the method is otherwise in scope and only lacks a held-out predictor.
    protocols = {str(row.get("protocol", "")) for row in rows}
    if protocols != {"application/MACS-CD4"}:
        return False
    reasons = [str(row.get("applicability_reason", "")).lower() for row in rows]
    return bool(reasons) and all(
        "out-of-sample prediction interface" in reason
        or "held-out predict interface" in reason
        for reason in reasons
    )


def _not_attempted_status(rows: Sequence[Mapping[str, object]]) -> str:
    return "N/A by capability" if _is_na_by_capability(rows) else "N/A by design"


def _metric_is_na_by_capability(
    rows: Sequence[Mapping[str, object]], metric: str
) -> bool:
    """Return true only for an explicit, recorded unavailable metric.

    A failed fit or a nonfinite calculation must never be relabelled as an
    unavailable capability.  The strict runner records the sole current case
    (the unmodified VCAM-Lasso author's missing held-out prediction API) in
    every successful fit's metadata.
    """

    if metric not in PREDICTION_METRICS:
        return False
    if _is_na_by_capability(rows):
        return True
    successful = [
        row
        for row in rows
        if row.get("attempt_status") == "success" and _bool(row.get("converged"))
    ]
    if not successful:
        return False
    return all(
        str(_fit_metadata(row).get("held_out_prediction", "")) == "N/A by capability"
        for row in successful
    )


def _metric_cell(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    summarizer: Callable[[np.ndarray], str],
) -> str:
    """Format a metric without concealing finite-success selection.

    The bracketed denominator always counts all applicable attempted fits,
    while the numerator counts successful, converged fits with a finite value
    for the named metric.  Design and source-capability nonattempts are not
    assigned a fictitious zero denominator.
    """

    if _is_na_by_design(rows):
        return rf"\emph{{{_not_attempted_status(rows)}}}"
    attempted = _attempted_rows(rows)
    values = _finite_success(rows, metric)
    if len(values):
        return f"{summarizer(values)} [{len(values)}/{len(attempted)}]"
    if _metric_is_na_by_capability(rows, metric):
        return rf"\emph{{N/A by capability}} [0/{len(attempted)}]"
    return f"-- [0/{len(attempted)}]" if attempted else "--"


def _failure_cell(rows: Sequence[Mapping[str, object]]) -> str:
    attempted = _attempted_rows(rows)
    failure = _failure_rate(rows)
    if not np.isfinite(failure):
        return "--"
    failed = sum(
        row.get("attempt_status") != "success" or not _bool(row.get("converged"))
        for row in attempted
    )
    return f"{100 * failure:.1f}\\% [{failed}/{len(attempted)}]"


def _na_metric_row(
    setting: str, method: str, n_metrics: int, *, status: str = "N/A by design"
) -> list[str]:
    return [
        setting,
        _method_short_tex(method),
        f"\\multicolumn{{{n_metrics}}}{{l}}{{\\emph{{{status}}}}}",
    ]


def make_method_admission_table(
    metadata: Mapping[str, object], output: Path, claims_eligible: bool
) -> None:
    body: list[list[str]] = []
    gates = metadata.get("admission_gates", {})
    if isinstance(gates, Mapping):
        for method in FIXED_METHOD_LABELS:
            decision = gates.get(method)
            if not isinstance(decision, Mapping):
                continue
            body.append(
                [
                    _method_tex(str(method)),
                    _tex_escape(
                        _implementation_summary(
                            str(method),
                            decision.get("implementation_version", "registered source"),
                        )
                    ),
                    _tex_escape(
                        _humanize_admission_basis(decision.get("admission_basis"))
                    ),
                    _tex_escape(_humanize_admission_status(decision.get("status"))),
                ]
            )
    _write_table(
        output / "method_admission.tex",
        caption="Published-method source identity and same-setting comparison rule.",
        label="tab:method-admission",
        columns=("Method", "Implementation", "Comparison basis", "Status"),
        alignment=METHOD_ADMISSION_ALIGNMENT,
        body=body or [["--", "--", "--", "No admission record"]],
        claims_eligible=claims_eligible,
    )


def make_example_tables(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    for example, stem in (("Example 1", "example1"), ("Example 2", "example2"), ("Example 3", "example3")):
        scenarios = sorted({str(row.get("scenario")) for row in rows if row.get("example") == example})
        sample_sizes = sorted({_scenario_sample_size(scenario) for scenario in scenarios})
        largest_n = max(sample_sizes, default=-1)
        main_scenarios = [
            scenario
            for scenario in scenarios
            if example == "Example 1" or _scenario_sample_size(scenario) == largest_n
        ]
        main_body: list[list[str]] = []
        for scenario in main_scenarios:
            for method in FIXED_METHOD_LABELS:
                subset = _group(rows, scenario=scenario, method=method)
                if not subset or not _has_attempt(subset):
                    continue
                setting = _scenario_setting_tex(scenario)
                main_body.append(
                    [
                        setting,
                        _method_short_tex(str(method)),
                        _metric_cell(subset, "component_ise", _mean_mcse),
                        _metric_cell(subset, "factor_ise", _mean_mcse),
                        _failure_cell(subset),
                    ]
                )
        scope = (
            "all registered settings"
            if example == "Example 1"
            else f"the largest registered sample size ($n={largest_n}$)"
        )
        main_table = dict(
            caption=(
                f"{example}: component-surface and identified-factor recovery for {scope}. "
                "Entries are means with Monte Carlo standard errors in parentheses, followed by "
                "[finite converged/attempted]; failure is [failed/attempted]. The Supplement "
                "reports every registered setting and prediction/computation output."
            ),
            label=f"tab:{stem}",
            columns=(
                "Setting",
                "Method",
                "Component ISE",
                "Factor ISE",
                "Failure",
            ),
            alignment=ESTIMATION_TABLE_ALIGNMENT,
            body=main_body or [["--", "--", "--", "--", "--"]],
            claims_eligible=claims_eligible,
        )
        # The 16--20 readable rows in the two higher-dimensional examples
        # need a page break.  Keeping one table number with a repeated header
        # is clearer than squeezing, rotating, or splitting the results into
        # arbitrary panels.
        writer = _write_longtable if example in {"Example 2", "Example 3"} else _write_table
        writer(output / f"{stem}_main.tex", **main_table)

        full_blocks: list[
            tuple[str, str, Sequence[str], str, Sequence[Sequence[str]]]
        ] = []
        for sample_size in sample_sizes:
            block_scenarios = [
                scenario
                for scenario in scenarios
                if _scenario_sample_size(scenario) == sample_size
            ]
            estimation_body: list[list[str]] = []
            prediction_body: list[list[str]] = []
            for scenario in block_scenarios:
                setting = _scenario_setting_tex(scenario)
                for method in FIXED_METHOD_LABELS:
                    subset = _group(rows, scenario=scenario, method=method)
                    if not subset:
                        continue
                    if _is_na_by_design(subset):
                        status = _not_attempted_status(subset)
                        estimation_body.append(
                            _na_metric_row(setting, str(method), 3, status=status)
                        )
                        prediction_body.append(
                            _na_metric_row(setting, str(method), 3, status=status)
                        )
                        continue
                    estimation_body.append(
                        [
                            setting,
                            _method_short_tex(str(method)),
                            _metric_cell(subset, "component_ise", _mean_mcse),
                            _metric_cell(subset, "factor_ise", _mean_mcse),
                            _failure_cell(subset),
                        ]
                    )
                    prediction_body.append(
                        [
                            setting,
                            _method_short_tex(str(method)),
                            _metric_cell(subset, "noise_free_test_mspe", _mean_mcse),
                            _metric_cell(subset, "runtime_seconds", _median_iqr),
                            _failure_cell(subset),
                        ]
                    )
            suffix = f"n{sample_size}" if sample_size >= 0 else "all"
            sample_text = f" at $n={sample_size}$" if sample_size >= 0 else ""
            full_blocks.extend(
                [
                    (
                        (
                            f"{example}{sample_text}: complete estimation-recovery audit. "
                            "Each metric is [finite converged/attempted]; failure is "
                            "[failed/attempted]."
                        ),
                        f"tab:{stem}-estimation-{suffix}",
                        ("Setting", "Method", "Component ISE", "Factor ISE", "Failure"),
                        ESTIMATION_TABLE_ALIGNMENT,
                        estimation_body or [["--", "--", "--", "--", "--"]],
                    ),
                    (
                        (
                            f"{example}{sample_text}: complete prediction and runtime audit. "
                            "Each metric is [finite converged/attempted]; failure is "
                            "[failed/attempted]."
                        ),
                        f"tab:{stem}-prediction-{suffix}",
                        ("Setting", "Method", "Noise-free MSPE", "Runtime, s", "Failure"),
                        PREDICTION_TABLE_ALIGNMENT,
                        prediction_body or [["--", "--", "--", "--", "--"]],
                    ),
                ]
            )
        _write_split_tables(
            output / f"{stem}_full.tex",
            blocks=full_blocks,
            claims_eligible=claims_eligible,
        )


def make_scaling_table(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    body: list[list[str]] = []
    scenarios = sorted({str(row.get("scenario")) for row in rows if row.get("example") == "Scaling"})
    for scenario in scenarios:
        p = next((token[1:] for token in scenario.split("-") if token.startswith("p")), "--")
        for method in FIXED_METHOD_LABELS:
            subset = _group(rows, scenario=scenario, method=method)
            if not subset or not _has_attempt(subset):
                continue
            memory = (
                r"\emph{N/A by capability: external R}"
                if method in EXTERNAL_R_METHODS
                else _metric_cell(subset, "peak_python_memory_mb", _median_iqr)
            )
            body.append(
                [
                    p,
                    _method_short_tex(str(method)),
                    _metric_cell(subset, "runtime_seconds", _median_iqr),
                    memory,
                    _failure_cell(subset),
                ]
            )
    _write_table(
        output / "scaling_main.tex",
        caption=(
            "Computational scaling in covariate dimension. Peak memory is the "
            "Python-process tracemalloc peak only; it excludes child R-process "
            "allocation and is therefore not a cross-runtime comparison. Entries are "
            "[finite converged/attempted]; failure is [failed/attempted]."
        ),
        label="tab:scaling-runtime",
        columns=("$p$", "Method", "Runtime, s", "Python-process peak MB", "Failure"),
        alignment=SCALING_TABLE_ALIGNMENT,
        body=body or [["--", "--", "--", "--", "--"]],
        claims_eligible=claims_eligible,
    )
    _write_table(
        output / "scaling_full.tex",
        caption=(
            "Full computational scaling audit in covariate dimension. Python "
            "tracemalloc excludes external R-process allocation, so those cells "
            "are N/A by capability and memory is not compared across runtimes. Entries "
            "are [finite converged/attempted]; failure is [failed/attempted]."
        ),
        label="tab:scaling-full",
        columns=("$p$", "Method", "Runtime, s", "Python-process peak MB", "Failure"),
        alignment=SCALING_TABLE_ALIGNMENT,
        body=body or [["--", "--", "--", "--", "--"]],
        claims_eligible=claims_eligible,
    )


def make_macs_tables(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    primary: list[list[str]] = []
    for method in FIXED_METHOD_LABELS:
        subset = _group(rows, variant="primary", method=method)
        if not subset:
            continue
        primary.append(
            [
                _method_short_tex(str(method)),
                _metric_cell(subset, "test_mse", _mean_mcse),
                _metric_cell(subset, "subject_balanced_test_mse", _mean_mcse),
                _metric_cell(subset, "runtime_seconds", _median_iqr),
                _failure_cell(subset),
            ]
        )
    _write_table(
        output / "macs_cv_main.tex",
        caption=(
            "MACS/CD4 repeated subject-level cross-validation. Entries are "
            "[finite converged/attempted]; failure is [failed/attempted]."
        ),
        label="tab:macs-prediction",
        columns=("Method", "Test MSE", "Subject-balanced MSE", "Runtime, s", "Failure"),
        alignment=MACS_TABLE_ALIGNMENT,
        body=primary or [["--", "--", "--", "--", "--"]],
        claims_eligible=claims_eligible,
    )
    sensitivity: list[list[str]] = []
    for variant in sorted({str(row.get("variant")) for row in rows}):
        for method in FIXED_METHOD_LABELS:
            subset = _group(rows, variant=variant, method=method)
            if subset:
                sensitivity.append(
                    [
                        _variant_tex(variant),
                        _method_short_tex(str(method)),
                        _metric_cell(subset, "subject_balanced_test_mse", _mean_mcse),
                    ]
                )
    # This sensitivity grid has 30 method-by-analysis rows.  It must be a
    # page-breaking table: a regular floating tabular exceeds the available
    # supplement page height on Overleaf even though it fits locally.
    _write_longtable(
        output / "macs_sensitivity.tex",
        caption=(
            "MACS/CD4 deletion, winsorization, and basis-size sensitivity. Entries are "
            "[finite converged/attempted]."
        ),
        label="tab:macs-sensitivity",
        columns=("Analysis", "Method", "Subject-balanced MSE"),
        alignment="@{}L{0.31\\linewidth}L{0.31\\linewidth}C{0.25\\linewidth}@{}",
        body=sensitivity or [["--", "--", "--"]],
        claims_eligible=claims_eligible,
    )


def make_failure_audit(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    body: list[list[str]] = []
    for method in FIXED_METHOD_LABELS:
        subset = [row for row in rows if row.get("method") == method]
        attempted = _attempted_rows(subset)
        succeeded = sum(
            row.get("attempt_status") == "success" and _bool(row.get("converged"))
            for row in attempted
        )
        failed = sum(
            row.get("attempt_status") != "success" or not _bool(row.get("converged"))
            for row in attempted
        )
        nonattempted = [
            row for row in subset if row.get("attempt_status") == "N/A by design"
        ]
        n_capability = sum(_is_na_by_capability([row]) for row in nonattempted)
        n_design = len(nonattempted) - n_capability
        body.append(
            [
                _method_short_tex(str(method)),
                str(len(subset)),
                str(len(attempted)),
                str(succeeded),
                str(failed),
                f"{n_capability}/{n_design}",
            ]
        )
    _write_table(
        output / "failure_audit.tex",
        caption=(
            "Attempted successes and failures, plus unattempted N/A by capability and "
            "N/A by design records. The final column is capability/design. "
            "Metric-specific finite/attempted counts are shown in the corresponding "
            "performance tables."
        ),
        label="tab:failure-audit",
        columns=(
            "Method",
            "Registered",
            "Attempted",
            "Successful",
            "Failed",
            "N/A (cap./design)",
        ),
        alignment="@{}p{0.28\\linewidth}ccccc@{}",
        body=body,
        claims_eligible=claims_eligible,
    )


def _flag_summary(records: Sequence[Mapping[str, object]]) -> str:
    """Aggregate a large row-level audit into one readable method-level cell."""

    metric_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    for record in records:
        try:
            metrics_raw = json.loads(str(record.get("flagged_metrics_json", "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            metrics_raw = {}
        try:
            rules_raw = json.loads(str(record.get("flag_rules_json", "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            rules_raw = {}
        if isinstance(metrics_raw, Mapping):
            metric_counts.update(str(metric) for metric in metrics_raw)
        if isinstance(rules_raw, Mapping):
            for detail in rules_raw.values():
                if isinstance(detail, Mapping):
                    raw_rules = detail.get("rules", [])
                    if isinstance(raw_rules, list):
                        rule_counts.update(str(rule) for rule in raw_rules)
    metric_text = ", ".join(
        f"{EXTREME_METRIC_LABELS.get(metric, metric.replace('_', ' '))} ({count})"
        for metric, count in sorted(metric_counts.items())
    )
    rule_text = {
        "absolute_magnitude": "absolute threshold",
        "scenario_log_mad_outlier": "scenario log-MAD",
    }
    rendered_rules = ", ".join(
        f"{rule_text.get(rule, rule.replace('_', ' '))} ({count})"
        for rule, count in sorted(rule_counts.items())
    )
    bits: list[str] = []
    if metric_text:
        bits.append(metric_text)
    if rendered_rules:
        bits.append(rendered_rules)
    return _tex_escape("; ".join(bits))


def make_extreme_finite_audit_table(
    strict_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
    audit_summary: Mapping[str, object],
    output: Path,
    claims_eligible: bool,
) -> None:
    """Write one compact, all-method audit table for finite extreme endpoints."""

    candidates_raw = audit_summary.get("candidate_successful_converged_rows_by_method", {})
    candidates = candidates_raw if isinstance(candidates_raw, Mapping) else {}
    by_method: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in audit_rows:
        by_method[str(record.get("method", ""))].append(record)
    body: list[list[str]] = []
    for method in FIXED_METHOD_LABELS:
        attempted = len(_attempted_rows([row for row in strict_rows if row.get("method") == method]))
        candidate = _int(candidates.get(method), 0)
        flagged = by_method.get(method, [])
        detail = "None" if not flagged else _flag_summary(flagged)
        body.append(
            [
                _method_short_tex(str(method)),
                f"{candidate}/{attempted}" if attempted else "--",
                str(len(flagged)),
                detail,
            ]
        )
    _write_table(
        output / "extreme_finite_audit.tex",
        caption=(
            "All-method post-run audit of numerically extreme, finite successful fits. "
            "The second column is successful-converged/attempted; flags use the locked "
            "absolute and scenario-specific log-MAD rules. Counts in the final column "
            "are criterion hits and can exceed flagged rows because one fit can trigger "
            "multiple criteria. A flag does not reclassify a fit or remove it from any "
            "primary summary. The row-level sidecar is SHA256 matched in the result manifest."
        ),
        label="tab:extreme-finite-audit",
        columns=(
            "Method",
            r"\shortstack{Successful\\/ attempted}",
            r"\shortstack{Flagged\\rows}",
            "Flag mechanisms",
        ),
        alignment=(
            r"@{}L{0.20\linewidth}C{0.15\linewidth}C{0.10\linewidth}"
            r"L{0.36\linewidth}@{}"
        ),
        body=body,
        claims_eligible=claims_eligible,
    )


def write_result_manifest_table(
    path: Path,
    *,
    strict_results: Path,
    strict_metadata: Path,
    macs_results: Path | None,
    macs_metadata: Path | None,
    macs_curves: Path | None,
    extreme_audit_json: Path | None,
    extreme_audit_rows: Path | None,
) -> None:
    body = [
        ["strict results", _sha256_tex(file_sha256(strict_results))],
        ["strict metadata", _sha256_tex(file_sha256(strict_metadata))],
    ]
    if macs_results is not None and macs_metadata is not None and macs_curves is not None:
        body.extend(
            [
                ["MACS results", _sha256_tex(file_sha256(macs_results))],
                ["MACS metadata", _sha256_tex(file_sha256(macs_metadata))],
                [
                    "MACS registered full-data factor curves",
                    _sha256_tex(file_sha256(macs_curves)),
                ],
            ]
        )
    if extreme_audit_json is not None and extreme_audit_rows is not None:
        body.extend(
            [
                ["extreme-finite audit summary", _sha256_tex(file_sha256(extreme_audit_json))],
                ["extreme-finite audit flagged rows", _sha256_tex(file_sha256(extreme_audit_rows))],
            ]
        )
    _write_table(
        path,
        caption="SHA256 manifest for the numerical inputs used by the manuscript.",
        label="tab:result-manifest",
        columns=("Artifact", "SHA256"),
        alignment="@{}L{0.32\\linewidth}L{0.54\\linewidth}@{}",
        body=body,
        claims_eligible=True,
    )


def _method_colors(methods: Sequence[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10")
    return {method: cmap(index % 10) for index, method in enumerate(methods)}


def plot_examples(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    examples = ["Example 1", "Example 2", "Example 3"]
    colors = _method_colors(list(FIXED_METHOD_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.85), constrained_layout=True)
    plotted = False
    for axis, example in zip(axes, examples, strict=True):
        example_rows = [row for row in rows if row.get("example") == example]
        methods, values = [], []
        for method in FIXED_METHOD_LABELS:
            sample = _finite_success(
                [row for row in example_rows if row.get("method") == method],
                "noise_free_test_mspe",
            )
            if len(sample):
                methods.append(str(method))
                values.append(sample)
        if values:
            plotted = True
            positions = np.arange(1, len(values) + 1)
            box = axis.boxplot(values, positions=positions, widths=0.62, patch_artist=True, showfliers=False)
            for patch, method in zip(box["boxes"], methods, strict=True):
                patch.set_facecolor(colors[method])
                patch.set_alpha(0.72)
            axis.set_xticks(
                positions,
                [_method_plot_label(method) for method in methods],
            )
            axis.set_yscale("log")
        else:
            axis.text(0.5, 0.5, "No admitted finite fits", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
        axis.set_title(example)
        axis.set_ylabel("Noise-free test MSPE")
        axis.grid(axis="y", alpha=0.2)
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "strict_prediction.pdf")
    plt.close(fig)
    if not plotted:
        return


def plot_scaling(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    scaling = [row for row in rows if row.get("example") == "Scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 3.05), constrained_layout=True)
    colors = _method_colors(list(FIXED_METHOD_LABELS))
    for method in FIXED_METHOD_LABELS:
        points: list[tuple[int, float, float]] = []
        for p in (10, 25, 50):
            subset = [
                row
                for row in scaling
                if row.get("method") == method and f"-p{p}" in str(row.get("scenario"))
            ]
            runtime = _finite_success(subset, "runtime_seconds")
            memory = _finite_success(subset, "peak_python_memory_mb")
            if len(runtime):
                memory_value = (
                    float("nan")
                    if method in EXTERNAL_R_METHODS or not len(memory)
                    else float(np.median(memory))
                )
                points.append((p, float(np.median(runtime)), memory_value))
        if points:
            p_values, runtime_values, memory_values = map(np.asarray, zip(*points, strict=True))
            axes[0].plot(
                p_values,
                runtime_values,
                marker="o",
                label=_method_display(str(method)),
                color=colors[str(method)],
            )
            finite_memory = np.isfinite(memory_values)
            if np.any(finite_memory):
                axes[1].plot(
                    p_values[finite_memory],
                    memory_values[finite_memory],
                    marker="o",
                    label=_method_display(str(method)),
                    color=colors[str(method)],
                )
    axes[0].set(xlabel="Covariate dimension $p$", ylabel="Median runtime (s)", yscale="log")
    axes[1].set(
        xlabel="Covariate dimension $p$",
        ylabel="Median Python-process peak (MB)",
        title="Not comparable across runtimes",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(handles)))
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "scaling_runtime.pdf")
    plt.close(fig)


def _boxplot_by_method(
    axis: object,
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    ylabel: str,
    colors: Mapping[str, object] | None = None,
    show_method_labels: bool = True,
) -> list[str]:
    """Draw comparable method boxplots and return the displayed methods.

    A method legend is preferable to repeated multi-line x-axis names when a
    figure has several narrow panels.  It preserves descriptive method names
    at a readable final size and avoids an overlap that could obscure data.
    """

    samples: list[np.ndarray] = []
    labels: list[str] = []
    methods: list[str] = []
    for method in FIXED_METHOD_LABELS:
        values = _finite_success(
            [row for row in rows if row.get("method") == method], metric
        )
        if len(values):
            samples.append(values)
            labels.append(_method_plot_label(str(method)))
            methods.append(str(method))
    if samples:
        box = axis.boxplot(samples, showfliers=False, patch_artist=True)
        for patch, method in zip(box["boxes"], methods, strict=True):
            patch.set_facecolor("#4c78a8" if colors is None else colors[method])
            patch.set_alpha(0.80)
        if show_method_labels:
            axis.set_xticks(np.arange(1, len(labels) + 1), labels)
        else:
            axis.set_xticks([])
            axis.tick_params(axis="x", length=0)
        if all(np.all(sample > 0) for sample in samples):
            axis.set_yscale("log")
    else:
        axis.text(0.5, 0.5, "No admitted finite fits", ha="center", va="center", transform=axis.transAxes)
        axis.set_xticks([])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22)
    return methods


def plot_registered_simulation_figures(
    rows: Sequence[Mapping[str, object]], output: Path, claims_eligible: bool
) -> None:
    example1 = [row for row in rows if row.get("example") == "Example 1"]
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 2.9), constrained_layout=True)
    _boxplot_by_method(axes[0], example1, "factor_ise", ylabel="Factor ISE")
    _boxplot_by_method(axes[1], example1, "component_ise", ylabel="Component-surface ISE")
    axes[0].set_title("Identified factors")
    axes[1].set_title("Component surfaces")
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "example1_factor_recovery.pdf")
    fig.savefig(output / "supp_example1_components.pdf")
    plt.close(fig)

    example2 = [row for row in rows if row.get("example") == "Example 2"]
    groups = [
        ("Gaussian", [row for row in example2 if "gaussian" in str(row.get("scenario"))]),
        ("Heavy tail", [row for row in example2 if "t2" in str(row.get("scenario"))]),
        ("Contamination", [row for row in example2 if "mixed-normal" in str(row.get("scenario"))]),
    ]
    colors = _method_colors(list(FIXED_METHOD_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 3.20), constrained_layout=True)
    displayed: set[str] = set()
    for axis, (title, subset) in zip(axes, groups, strict=True):
        displayed.update(
            _boxplot_by_method(
                axis,
                subset,
                "noise_free_test_mspe",
                ylabel="Noise-free MSPE",
                colors=colors,
                show_method_labels=False,
            )
        )
        axis.set_title(title)
    ordered_displayed = [method for method in FIXED_METHOD_LABELS if method in displayed]
    if ordered_displayed:
        fig.legend(
            [Patch(facecolor=colors[method], edgecolor="black") for method in ordered_displayed],
            [_method_display(method) for method in ordered_displayed],
            loc="outside lower center",
            ncol=2,
            frameon=False,
            columnspacing=1.4,
            handlelength=1.1,
        )
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "example2_robustness.pdf")
    fig.savefig(output / "supp_example2_distributions.pdf")
    plt.close(fig)

    example3 = [row for row in rows if row.get("example") in {"Example 3", "Scaling"}]
    colors = _method_colors(list(FIXED_METHOD_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 3.10), constrained_layout=True)
    displayed = set()
    for axis, metric, title in zip(
        axes,
        ("tpr", "fdr", "model_size"),
        ("True-positive rate", "False-discovery rate", "Selected model size"),
        strict=True,
    ):
        displayed.update(
            _boxplot_by_method(
                axis,
                example3,
                metric,
                ylabel=title,
                colors=colors,
                show_method_labels=False,
            )
        )
        axis.set_title(title)
        axis.set_yscale("linear")
    ordered_displayed = [method for method in FIXED_METHOD_LABELS if method in displayed]
    if ordered_displayed:
        fig.legend(
            [Patch(facecolor=colors[method], edgecolor="black") for method in ordered_displayed],
            [_method_display(method) for method in ordered_displayed],
            loc="outside lower center",
            ncol=min(3, len(ordered_displayed)),
            frameon=False,
            columnspacing=1.4,
            handlelength=1.1,
        )
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "supp_example3_selection.pdf")
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _registered_full_fit_curves(
    curve_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the sole registered full-data fit; never average CV factor curves."""

    candidates = [
        row
        for row in curve_rows
        if row.get("fit_scope") == "registered_full_data"
        and row.get("variant") == "primary"
        and row.get("method") == _method_value("TRACE_VCAM", "TRACE-VCAM")
    ]
    if len(candidates) != 1:
        return {}
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    curves = candidates[0].get("curves", [])
    if not isinstance(curves, list):
        return output
    for curve in curves:
        if not isinstance(curve, Mapping):
            continue
        grid = np.asarray(curve.get("grid", []), dtype=float)
        values = np.asarray(curve.get("values", []), dtype=float)
        if (
            len(grid) >= 2
            and grid.shape == values.shape
            and np.all(np.isfinite(grid))
            and np.all(np.isfinite(values))
        ):
            output[str(curve.get("component"))] = (grid, values)
    return output


def plot_macs_components(
    rows: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]],
    output: Path,
    claims_eligible: bool,
) -> None:
    del rows  # Prediction rows are audited separately and never define application curves.
    curves = _registered_full_fit_curves(curve_rows)
    names = ["baseline", "beta_1", "phi_1", "beta_2", "phi_2"]
    titles = [
        "Baseline",
        "Age coefficient",
        "Age additive function",
        "CES-D coefficient",
        "CES-D additive function",
    ]
    xlabels = ["Scaled time", "Scaled time", "Scaled age", "Scaled time", "Scaled CES-D"]
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 5.0), constrained_layout=True)
    for axis, name, title, xlabel in zip(axes.flat, names, titles, xlabels, strict=False):
        if name in curves:
            grid, values = curves[name]
            axis.plot(grid, values, color="#1f4e79", linewidth=2.1)
        else:
            axis.text(0.5, 0.5, "No finite curve", ha="center", va="center", transform=axis.transAxes)
        axis.axhline(0.0, color="0.6", linewidth=0.7)
        axis.set(title=title, xlabel=xlabel, ylabel="Estimate")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "macs_components.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 2.95), constrained_layout=True)
    for axis, index, title, xlabel in (
        (axes[0], 1, "Age component surface", "Scaled age"),
        (axes[1], 2, "CES-D component surface", "Scaled CES-D"),
    ):
        beta, phi = curves.get(f"beta_{index}"), curves.get(f"phi_{index}")
        if beta is None or phi is None:
            axis.text(0.5, 0.5, "No finite surface", ha="center", va="center", transform=axis.transAxes)
            axis.set(title=title, xlabel=xlabel, ylabel="Scaled time")
            continue
        t_grid, beta_values = beta
        z_grid, phi_values = phi
        # A 61-by-61 grid preserves the smooth spline surface while keeping
        # the PDF genuinely vector and small enough for fast LaTeX/Overleaf rendering.
        plot_t = np.linspace(float(t_grid[0]), float(t_grid[-1]), 61)
        plot_z = np.linspace(float(z_grid[0]), float(z_grid[-1]), 61)
        plot_beta = np.interp(plot_t, t_grid, beta_values)
        plot_phi = np.interp(plot_z, z_grid, phi_values)
        surface = plot_beta[:, None] * plot_phi[None, :]
        mesh = axis.pcolormesh(plot_z, plot_t, surface, shading="auto", cmap="coolwarm")
        fig.colorbar(mesh, ax=axis, label="Estimated component")
        axis.set(title=title, xlabel=xlabel, ylabel="Scaled time")
    if not claims_eligible:
        fig.suptitle("Exploratory output: formal audit not passed", color="#8b1a1a")
    fig.savefig(output / "macs_surfaces.pdf")
    plt.close(fig)


def _method_value(name: str, fallback: str) -> str:
    value = getattr(MethodLabel, name, None)
    return fallback if value is None else str(value.value)


def write_claim_macros(
    path: Path,
    claims_eligible: bool,
    audit_hash: str,
    *,
    artifacts_ready: bool | None = None,
) -> None:
    ready = claims_eligible if artifacts_ready is None else bool(artifacts_ready)
    artifact_switch = (
        "\\strictartifactsreadytrue" if ready else "\\strictartifactsreadyfalse"
    )
    if claims_eligible:
        content = (
            "% Generated only after the complete formal hash/count/admission audit.\n"
            "\\newif\\ifstrictclaims\n"
            "\\strictclaimstrue\n"
            "\\newif\\ifstrictartifactsready\n"
            f"{artifact_switch}\n"
            "\\newcommand{\\StrictResultsStatus}{Formal audited results}\n"
            f"\\newcommand{{\\StrictAuditHash}}{{{audit_hash}}}\n"
        )
    else:
        content = (
            "% Global superiority claims are disabled unless every registered gate passes.\n"
            "\\newif\\ifstrictclaims\n"
            "\\strictclaimsfalse\n"
            "\\newif\\ifstrictartifactsready\n"
            f"{artifact_switch}\n"
            "\\newcommand{\\StrictResultsStatus}{Audited descriptive results---global all-method claims disabled}\n"
            f"\\newcommand{{\\StrictAuditHash}}{{{audit_hash}}}\n"
        )
    path.write_text(content, encoding="utf-8")


def execute(args: argparse.Namespace) -> dict[str, Path]:
    strict_results = args.strict_results.resolve()
    strict_metadata_path = args.strict_metadata.resolve()
    strict_rows = _read_csv(strict_results)
    strict_metadata = _read_json(strict_metadata_path)
    strict_audit = audit_strict(strict_rows, strict_metadata, strict_results)
    extreme_audit_json: Path | None = None
    extreme_audit_rows_path: Path | None = None
    extreme_rows: list[dict[str, str]] = []
    extreme_summary: dict[str, object] = {}
    if args.extreme_audit_json is None and args.extreme_audit_rows is None:
        extreme_audit: dict[str, object] = {
            "schema": SCHEMA_EXTREME_AUDIT,
            "passed": False,
            "issues": [
                "all-method extreme-finite audit sidecars were not supplied; formal artifacts are disabled"
            ],
            "audit_json_sha256": None,
            "audit_rows_sha256": None,
            "n_flag_rows": 0,
            "scope_methods": [],
        }
    elif args.extreme_audit_json is None or args.extreme_audit_rows is None:
        raise ValueError("--extreme-audit-json and --extreme-audit-rows must be supplied together")
    else:
        extreme_audit_json = args.extreme_audit_json.resolve()
        extreme_audit_rows_path = args.extreme_audit_rows.resolve()
        extreme_audit, extreme_rows, extreme_summary = audit_extreme_sidecar(
            strict_rows,
            strict_results,
            extreme_audit_json,
            extreme_audit_rows_path,
        )
    macs_rows: list[dict[str, str]] = []
    macs_metadata: dict[str, object] | None = None
    macs_audit: dict[str, object] | None = None
    macs_curves_path: Path | None = None
    if args.macs_results is not None or args.macs_metadata is not None:
        if args.macs_results is None or args.macs_metadata is None:
            raise ValueError("--macs-results and --macs-metadata must be supplied together")
        macs_results = args.macs_results.resolve()
        macs_rows = _read_csv(macs_results)
        macs_metadata = _read_json(args.macs_metadata.resolve())
        macs_curves_path = (
            None if args.macs_curves is None else args.macs_curves.resolve()
        )
        macs_audit = audit_macs(
            macs_rows,
            macs_metadata,
            macs_results,
            macs_curves_path,
        )
    elif args.macs_curves is not None:
        raise ValueError("--macs-curves requires --macs-results and --macs-metadata")
    descriptive_ready = bool(
        strict_audit["passed"]
        and extreme_audit["passed"]
        and (macs_audit is None or macs_audit["passed"])
    )
    claims_eligible = bool(
        descriptive_ready
        and strict_metadata.get("formal_claims_eligible") is True
        and (
            macs_metadata is None
            or macs_metadata.get("formal_claims_eligible") is True
        )
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal_tables = output / "tables"
    formal_figures = output / "figures"
    formal_tables.mkdir(parents=True, exist_ok=True)
    formal_figures.mkdir(parents=True, exist_ok=True)
    # Remove only analyzer-owned manuscript artifacts.  A quick/incomplete run
    # must not leave stale formal numbers visible merely because an older file
    # exists under the expected name.
    for name in MANUSCRIPT_TABLES:
        candidate = formal_tables / name
        if candidate.is_file():
            candidate.unlink()
    for name in MANUSCRIPT_FIGURES:
        candidate = formal_figures / name
        if candidate.is_file():
            candidate.unlink()
    artifact_root = output if descriptive_ready else output / "exploratory"
    tables = artifact_root / "tables"
    figures = artifact_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    make_method_admission_table(strict_metadata, tables, descriptive_ready)
    make_example_tables(strict_rows, tables, descriptive_ready)
    make_scaling_table(strict_rows, tables, descriptive_ready)
    make_failure_audit(strict_rows, tables, descriptive_ready)
    if extreme_audit["passed"]:
        make_extreme_finite_audit_table(
            strict_rows,
            extreme_rows,
            extreme_summary,
            tables,
            descriptive_ready,
        )
    plot_registered_simulation_figures(strict_rows, figures, descriptive_ready)
    if macs_rows:
        make_macs_tables(macs_rows, tables, descriptive_ready)
        if macs_curves_path is not None and macs_curves_path.is_file():
            plot_macs_components(
                macs_rows,
                _read_jsonl(macs_curves_path),
                figures,
                descriptive_ready,
            )

    audit = {
        "schema_version": "vcam-strict-analysis/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "descriptive_artifacts_eligible": descriptive_ready,
        "claims_eligible": claims_eligible,
        "strict": strict_audit,
        "extreme_finite_audit": extreme_audit,
        "macs": macs_audit,
        "inputs": {
            "strict_results": {"path": str(strict_results), "sha256": file_sha256(strict_results)},
            "strict_metadata": {
                "path": str(strict_metadata_path),
                "sha256": file_sha256(strict_metadata_path),
            },
        },
        "format_audit": {
            "vector_figures": True,
            "minimum_configured_source_font_pt": MIN_SOURCE_FONT_PT,
            "maximum_source_figure_width_inches": FIGURE_WIDTH_IN,
            "minimum_estimated_inserted_font_pt": round(
                MIN_SOURCE_FONT_PT * (0.88 * 6.5) / FIGURE_WIDTH_IN, 2
            ),
            "inserted_width_basis": (
                "narrowest manuscript include is 0.88 of the 6.5-inch text width"
            ),
            "max_panels_per_row": 3,
            "forbidden_latex_commands": ["scriptsize", "resizebox"],
        },
    }
    if extreme_audit_json is not None and extreme_audit_rows_path is not None:
        audit["inputs"]["extreme_finite_audit_json"] = {
            "path": str(extreme_audit_json),
            "sha256": (
                file_sha256(extreme_audit_json) if extreme_audit_json.is_file() else None
            ),
        }
        audit["inputs"]["extreme_finite_audit_rows"] = {
            "path": str(extreme_audit_rows_path),
            "sha256": (
                file_sha256(extreme_audit_rows_path)
                if extreme_audit_rows_path.is_file()
                else None
            ),
        }
    if args.macs_results is not None:
        audit["inputs"]["macs_results"] = {
            "path": str(args.macs_results.resolve()),
            "sha256": file_sha256(args.macs_results.resolve()),
        }
        audit["inputs"]["macs_metadata"] = {
            "path": str(args.macs_metadata.resolve()),
            "sha256": file_sha256(args.macs_metadata.resolve()),
        }
        if macs_curves_path is not None and macs_curves_path.is_file():
            audit["inputs"]["macs_registered_full_data_factor_curves"] = {
                "path": str(macs_curves_path),
                "sha256": file_sha256(macs_curves_path),
            }
    provisional = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    audit_hash = hashlib.sha256(provisional.encode("utf-8")).hexdigest()
    audit["audit_payload_sha256"] = audit_hash
    audit_path = output / "strict_analysis_audit.json"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    claims_path = formal_tables / "strict_claims.tex"
    write_claim_macros(
        claims_path,
        claims_eligible,
        file_sha256(audit_path),
        artifacts_ready=descriptive_ready,
    )
    if descriptive_ready:
        write_result_manifest_table(
            formal_tables / "result_manifest.tex",
            strict_results=strict_results,
            strict_metadata=strict_metadata_path,
            macs_results=(None if args.macs_results is None else args.macs_results.resolve()),
            macs_metadata=(None if args.macs_metadata is None else args.macs_metadata.resolve()),
            macs_curves=macs_curves_path,
            extreme_audit_json=extreme_audit_json,
            extreme_audit_rows=extreme_audit_rows_path,
        )
    manifest = {
        str(path.relative_to(output)).replace("\\", "/"): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    manifest_path = output / "analysis_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "audit": audit_path,
        "manifest": manifest_path,
        "tables": tables,
        "figures": figures,
        "formal_claim_switch": claims_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-results", type=Path, required=True)
    parser.add_argument("--strict-metadata", type=Path, required=True)
    parser.add_argument("--macs-results", type=Path, default=None)
    parser.add_argument("--macs-metadata", type=Path, default=None)
    parser.add_argument("--macs-curves", type=Path, default=None)
    parser.add_argument("--extreme-audit-json", type=Path, default=None)
    parser.add_argument("--extreme-audit-rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    paths = execute(parse_args(argv))
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

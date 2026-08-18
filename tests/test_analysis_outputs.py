import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.methods import FIXED_METHOD_LABELS
from scripts.audit_extreme_finite_success import OUTPUT_COLUMNS
from scripts.analyze_strict_results import (
    EXTREME_AUDIT_METRICS,
    SCHEMA_EXTREME_AUDIT,
    SCHEMA_MACS,
    _metric_cell,
    _humanize_admission_basis,
    _method_short_tex,
    _object_sha256,
    _registered_full_fit_curves,
    _sha256_tex,
    audit_extreme_sidecar,
    audit_macs,
    file_sha256,
    make_extreme_finite_audit_table,
    make_example_tables,
    make_failure_audit,
    make_scaling_table,
    write_result_manifest_table,
)


def _simulation_row(example: str, scenario: str, method: str) -> dict[str, object]:
    applicable = method in {
        "TRACE-VCAM",
        "ZZW2020",
        "HHY2021-Huber",
        "ZY2025-paper-implementation",
    }
    return {
        "example": example,
        "scenario": scenario,
        "method": method,
        "attempt_status": "success" if applicable else "N/A by design",
        "converged": applicable,
        "component_ise": 0.12,
        "factor_ise": 0.23,
        "noise_free_test_mspe": 0.34,
        "runtime_seconds": 1.5,
        "peak_python_memory_mb": 8.0,
    }


def _write_extreme_sidecar(
    root: Path, strict_rows: list[dict[str, object]]
) -> tuple[Path, Path, Path]:
    """Create a valid, all-method sidecar tied to a tiny strict-result fixture."""

    strict_results = root / "strict_results.csv"
    strict_results.write_text("fixture strict results\n", encoding="utf-8")
    progress = root / "strict_progress.json"
    progress.write_text(
        json.dumps({"committed_cohorts": 1, "expected_cohorts": 1}), encoding="utf-8"
    )
    rows_path = root / "extreme_finite_success_rows.csv"
    flagged_method = "ZZW2020"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "scenario": "fixture-scenario",
                "replicate": "0",
                "method": flagged_method,
                "attempt_status": "success",
                "converged": "True",
                "extreme_finite_success_audit_flag": True,
                "flagged_metrics_json": json.dumps({"component_ise": 1500.0}),
                "flag_rules_json": json.dumps(
                    {"component_ise": {"rules": ["absolute_magnitude"]}}
                ),
            }
        )
    candidates = {method: 1 for method in FIXED_METHOD_LABELS}
    flagged = {method: int(method == flagged_method) for method in FIXED_METHOD_LABELS}
    audit_script = Path(__file__).resolve().parents[1] / "scripts" / "audit_extreme_finite_success.py"
    summary: dict[str, object] = {
        "schema_version": SCHEMA_EXTREME_AUDIT,
        "scope": {
            "method": None,
            "methods": list(FIXED_METHOD_LABELS),
            "success_definition": "attempt_status == success and converged == True",
            "metrics": sorted(EXTREME_AUDIT_METRICS),
        },
        "rules": {},
        "candidate_successful_converged_rows": len(strict_rows),
        "candidate_successful_converged_rows_by_method": candidates,
        "flagged_rows": 1,
        "flagged_rows_by_method": flagged,
        "flagged_metrics": {"component_ise": 1},
        "flagged_rule_counts": {"absolute_magnitude": 1},
        "inputs": {
            "results_sha256": file_sha256(strict_results),
            "progress_sha256": file_sha256(progress),
            "progress_complete": True,
            "audit_script_sha256": file_sha256(audit_script),
        },
        "outputs": {"rows_csv_sha256": file_sha256(rows_path)},
    }
    provisional = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    summary["audit_payload_sha256"] = hashlib.sha256(provisional.encode("utf-8")).hexdigest()
    json_path = root / "extreme_finite_success_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return strict_results, json_path, rows_path


class AnalysisOutputTests(unittest.TestCase):
    def test_main_estimation_table_is_compact_and_full_output_is_split(self):
        scenarios = [
            f"example2-{noise}-n{sample_size}{suffix}"
            for sample_size in (50, 100, 200)
            for noise, suffix in (
                ("gaussian", "-sigma0.1"),
                ("gaussian", "-sigma0.4"),
                ("t2", ""),
                ("mixed-normal", ""),
            )
        ]
        rows = [
            _simulation_row("Example 2", scenario, method)
            for scenario in scenarios
            for method in FIXED_METHOD_LABELS
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            make_example_tables(rows, output, True)
            main = (output / "example2_main.tex").read_text(encoding="utf-8")
            full = (output / "example2_full.tex").read_text(encoding="utf-8")
        self.assertIn("Component ISE", main)
        self.assertIn("Factor ISE", main)
        self.assertNotIn("Noise-free MSPE", main)
        self.assertIn("n200", main)
        self.assertNotIn("n50", main)
        self.assertNotIn("n100", main)
        self.assertNotIn("ZZW2020", main)
        # Readable fixed-width cells require page-breaking tables for the
        # complete 16-setting audit; the output must never rely on tiny text
        # or a horizontally cropped unbreakable float.
        self.assertIn(r"\begin{longtable}", main)
        self.assertIn(r"L{0.13\linewidth}", main)
        self.assertEqual(full.count(r"\begin{longtable}"), 6)
        self.assertIn("N/A by design", full)
        self.assertNotIn("scriptsize", full)
        self.assertNotIn("resizebox", full)

    def test_scaling_memory_is_not_claimed_for_external_r(self):
        rows = []
        for method in ("TRACE-VCAM", "ZSY2026-author-code"):
            row = _simulation_row("Scaling", "scaling-n200-p10", method)
            row["attempt_status"] = "success"
            row["converged"] = True
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            make_scaling_table(rows, output, True)
            text = (output / "scaling_main.tex").read_text(encoding="utf-8")
        self.assertIn("N/A by capability: external R", text)
        self.assertIn("not a cross-runtime comparison", text)
        self.assertIn("Python-process peak MB", text)

    def test_metric_cells_expose_finite_attempted_and_capability_status(self):
        capable_row = {
            "attempt_status": "success",
            "converged": True,
            "noise_free_test_mspe": "nan",
            "fit_metadata_json": json.dumps({"held_out_prediction": "N/A by capability"}),
        }
        failed_row = {
            "attempt_status": "failed",
            "converged": False,
            "noise_free_test_mspe": "nan",
            "fit_metadata_json": "{}",
        }
        finite_row = {
            "attempt_status": "success",
            "converged": True,
            "component_ise": 0.2,
            "fit_metadata_json": "{}",
        }
        simulation_design_row = {
            "attempt_status": "N/A by design",
            "converged": False,
            "protocol": "example-1/dense-functional",
            "applicability_reason": (
                "The unmodified author function has no subject-ID or "
                "out-of-sample prediction interface."
            ),
        }
        macs_capability_row = {
            "attempt_status": "N/A by design",
            "converged": False,
            "protocol": "application/MACS-CD4",
            "applicability_reason": (
                "The unmodified author function has no subject-ID or "
                "out-of-sample prediction interface."
            ),
        }
        self.assertEqual(
            _metric_cell([capable_row], "noise_free_test_mspe", lambda _: "ignored"),
            r"\emph{N/A by capability} [0/1]",
        )
        self.assertEqual(
            _metric_cell([failed_row], "noise_free_test_mspe", lambda _: "ignored"),
            "-- [0/1]",
        )
        self.assertEqual(
            _metric_cell([finite_row], "component_ise", lambda _: "0.200"),
            "0.200 [1/1]",
        )
        self.assertEqual(
            _metric_cell(
                [simulation_design_row],
                "noise_free_test_mspe",
                lambda _: "ignored",
            ),
            r"\emph{N/A by design}",
        )
        self.assertEqual(
            _metric_cell(
                [macs_capability_row],
                "noise_free_test_mspe",
                lambda _: "ignored",
            ),
            r"\emph{N/A by capability}",
        )

    def test_extreme_sidecar_is_hash_validated_and_rendered_for_all_methods(self):
        strict_rows = []
        for method in FIXED_METHOD_LABELS:
            strict_rows.append(
                {
                    "scenario": "fixture-scenario",
                    "replicate": "0",
                    "method": method,
                    "attempt_status": "success",
                    "converged": True,
                    "component_ise": 1500.0 if method == "ZZW2020" else 0.2,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strict_results, audit_json, audit_rows = _write_extreme_sidecar(root, strict_rows)
            audit, flags, summary = audit_extreme_sidecar(
                strict_rows, strict_results, audit_json, audit_rows
            )
            self.assertTrue(audit["passed"], audit["issues"])
            output = root / "tables"
            output.mkdir()
            make_extreme_finite_audit_table(strict_rows, flags, summary, output, True)
            text = (output / "extreme_finite_audit.tex").read_text(encoding="utf-8")
            self.assertIn("Backfitting VCAM", text)
            self.assertIn("1/1", text)
            self.assertIn("absolute threshold", text)
            payload = json.loads(audit_json.read_text(encoding="utf-8"))
            payload["inputs"]["results_sha256"] = "0" * 64
            without_hash = dict(payload)
            without_hash.pop("audit_payload_sha256", None)
            payload["audit_payload_sha256"] = hashlib.sha256(
                (json.dumps(without_hash, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            ).hexdigest()
            audit_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tampered, _, _ = audit_extreme_sidecar(strict_rows, strict_results, audit_json, audit_rows)
            self.assertFalse(tampered["passed"])
            self.assertTrue(any("results SHA256" in issue for issue in tampered["issues"]))

    def test_human_labels_do_not_expose_internal_codes_or_old_gate_column(self):
        self.assertNotIn("_", _humanize_admission_basis("same_setting_original_method_comparison"))
        self.assertEqual(_method_short_tex("ZZW2020"), "Backfitting VCAM")
        rows = [
            {
                "method": method,
                "attempt_status": "N/A by design",
                "converged": False,
            }
            for method in FIXED_METHOD_LABELS
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            make_failure_audit(rows, output, True)
            text = (output / "failure_audit.tex").read_text(encoding="utf-8")
        self.assertNotIn("Gate-blocked", text)
        self.assertIn("N/A by design", text)
        self.assertIn("N/A (cap./design)", text)

    def test_registered_full_fit_curves_are_not_fold_averaged(self):
        curves = [
            {"component": "baseline", "grid": [0.0, 1.0], "values": [1.0, 2.0]}
        ]
        rows = [
            {
                "fit_scope": "registered_full_data",
                "variant": "primary",
                "method": "TRACE-VCAM",
                "curves": curves,
            },
            {
                "fit_scope": "cross_validation_fold",
                "variant": "primary",
                "method": "TRACE-VCAM",
                "curves": [
                    {"component": "baseline", "grid": [0.0, 1.0], "values": [99.0, 99.0]}
                ],
            },
        ]
        selected = _registered_full_fit_curves(rows)
        self.assertEqual(selected["baseline"][1].tolist(), [1.0, 2.0])

    def test_macs_curve_file_and_registered_payload_are_hash_audited(self):
        curves = [
            {"component": "baseline", "grid": [0.0, 1.0], "values": [1.0, 1.5]},
            {"component": "beta_1", "grid": [0.0, 1.0], "values": [1.0, 1.0]},
            {"component": "phi_1", "grid": [0.0, 1.0], "values": [-0.5, 0.5]},
            {"component": "beta_2", "grid": [0.0, 1.0], "values": [1.0, 1.0]},
            {"component": "phi_2", "grid": [0.0, 1.0], "values": [-0.5, 0.5]},
        ]
        tuning_hash = "1" * 64
        curve_row = {
            "fit_id": "macs-primary-full-data-trace-v1",
            "fit_scope": "registered_full_data",
            "variant": "primary",
            "method": "TRACE-VCAM",
            "data_hash": "data-hash",
            "tuning_sha256": tuning_hash,
            "identified_curves_sha256": _object_sha256(curves),
            "curves": curves,
        }
        registered = {
            "fit_id": curve_row["fit_id"],
            "data_hash": curve_row["data_hash"],
            "tuning_sha256": tuning_hash,
            "raw_curves_sha256": "2" * 64,
            "identified_curves_sha256": curve_row["identified_curves_sha256"],
            "curve_row_sha256": _object_sha256(curve_row),
            "attempt_status": "success",
            "converged": True,
        }
        rows = [
            {
                "schema_version": SCHEMA_MACS,
                "variant": "primary",
                "repeat": 0,
                "fold": 0,
                "method": method,
                "attempt_status": "N/A by design",
                "converged": False,
                "data_hash": "data-hash",
                "train_subject_hash": "train-hash",
                "test_subject_hash": "test-hash",
            }
            for method in FIXED_METHOD_LABELS
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "macs_results.csv"
            results.write_text("registered test fixture\n", encoding="utf-8")
            factor_curves = root / "macs_factor_curves.jsonl"
            factor_curves.write_text(
                json.dumps(curve_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            metadata = {
                "schema_version": SCHEMA_MACS,
                "mode": "formal",
                "formal_protocol_complete": True,
                "data_source": {"n_rows": 2376, "n_subjects": 369},
                "response_transform": "none (raw CD4)",
                "inference": "No confidence intervals are computed or claimed.",
                "fold_protocol": {"n_splits": 5, "n_repeats": 5},
                "cohort_audit": {"passed": True},
                "curve_protocol": {"aggregation": "none", "fold_curves_serialized": False},
                "registered_curve_fit": registered,
                "files": {
                    "results": {"sha256": file_sha256(results)},
                    "curves": {"sha256": file_sha256(factor_curves)},
                },
            }
            audit = audit_macs(rows, metadata, results, factor_curves)
            self.assertTrue(audit["passed"], audit["issues"])
            self.assertEqual(audit["curves_sha256"], file_sha256(factor_curves))
            factor_curves.write_text("{}\n", encoding="utf-8")
            tampered = audit_macs(rows, metadata, results, factor_curves)
            self.assertFalse(tampered["passed"])
            self.assertTrue(any("factor_curves" in issue for issue in tampered["issues"]))

    def test_result_manifest_includes_registered_full_data_curve_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for name in (
                "strict.csv",
                "strict.json",
                "macs.csv",
                "macs.json",
                "curves.jsonl",
                "extreme.json",
                "extreme.csv",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                inputs.append(path)
            target = root / "manifest.tex"
            curves_hash = file_sha256(inputs[4])
            extreme_hash = file_sha256(inputs[5])
            write_result_manifest_table(
                target,
                strict_results=inputs[0],
                strict_metadata=inputs[1],
                macs_results=inputs[2],
                macs_metadata=inputs[3],
                macs_curves=inputs[4],
                extreme_audit_json=inputs[5],
                extreme_audit_rows=inputs[6],
            )
            text = target.read_text(encoding="utf-8")
        self.assertIn("MACS registered full-data factor curves", text)
        self.assertEqual(_sha256_tex(curves_hash), r"\texttt{" + r"\VCAMHashBreak{}".join(
            curves_hash[index : index + 8] for index in range(0, len(curves_hash), 8)
        ) + "}")
        self.assertIn(_sha256_tex(curves_hash), text)
        self.assertIn("extreme-finite audit summary", text)
        self.assertIn(_sha256_tex(extreme_hash), text)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.data import SubjectDataset
from scripts.analyze_strict_results import _write_table, write_claim_macros
from scripts.run_macs_application import (
    HHY as MACS_HHY,
    TRACE,
    ZY as MACS_ZY,
    ZZW as MACS_ZZW,
    _fit_registered_full_data_trace_curves,
    _load_admissions,
    _run_fold_method,
    _tuning as macs_tuning,
    parse_args as parse_macs_args,
    prepare_macs_variant,
    read_macs_csv,
)
from scripts.run_strict_benchmark import (
    ZY,
    ZZW,
    _common_identify_curves,
    _default_tuning,
    _split_dataset,
    _subject_dataset,
    _truth_metrics,
    assess_reproduction_gate,
    parse_args as parse_strict_args,
    registered_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]


class _CurveAuditAdapter:
    label = TRACE

    def __init__(self) -> None:
        self.factor_curve_calls = 0
        self.fit_datasets = []

    def preflight(self):
        return SimpleNamespace(
            ready=True,
            version="test-trace-v1",
            code="ready",
            message="",
            environment={"fixture": True},
        )

    def fit(self, dataset, *, seed, tuning):
        self.fit_datasets.append(dataset)
        return SimpleNamespace(
            converged=True,
            selected_blocks=(0, 1),
            metadata={"seed": seed, "q_time": tuning["q_time"]},
        )

    def predict(self, artifact, test):
        del artifact
        return np.full(test.n_rows, float(np.mean(test.response)))

    def factor_curves(self, artifact):
        del artifact
        self.factor_curve_calls += 1
        grid = np.linspace(0.0, 1.0, 11).tolist()
        return (
            {"component": "baseline", "grid": grid, "values": np.linspace(1.0, 2.0, 11).tolist()},
            {"component": "beta_1", "grid": grid, "values": np.linspace(2.0, 3.0, 11).tolist()},
            {"component": "phi_1", "grid": grid, "values": np.linspace(1.0, 2.0, 11).tolist()},
            {"component": "beta_2", "grid": grid, "values": np.linspace(1.0, 2.0, 11).tolist()},
            {"component": "phi_2", "grid": grid, "values": np.linspace(-2.0, 1.0, 11).tolist()},
        )


def _tiny_macs_dataset() -> SubjectDataset:
    return SubjectDataset(
        time=np.linspace(0.0, 1.0, 6),
        covariates=np.column_stack(
            [np.linspace(0.0, 1.0, 6), np.linspace(1.0, 0.0, 6)]
        ),
        response=np.linspace(100.0, 200.0, 6),
        subject_id=np.asarray(["a", "a", "b", "b", "c", "c"]),
        row_id=np.asarray([f"row-{index}" for index in range(6)]),
        noise_free_target=None,
        covariate_names=("age", "cesd"),
        metadata={
            "time_domain": [0.0, 1.0],
            "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
        },
    )


class StrictPipelineTests(unittest.TestCase):
    def test_macs_published_methods_use_source_specific_knot_vectors(self):
        zzw = macs_tuning(MACS_ZZW, 6, quick=False)
        self.assertEqual(zzw["time_interior_knots"], [2, 1, 2])
        self.assertEqual(zzw["covariate_interior_knots"], [3, 1])
        self.assertIn(
            {"time": [2, 1, 2], "additive": [3, 1]},
            zzw["knot_candidate_vectors"],
        )
        hhy = macs_tuning(MACS_HHY, 6, quick=False)
        self.assertEqual(hhy["pilot_time_interior_knots"], 2)
        self.assertEqual(hhy["pilot_covariate_interior_knots"], [2, 2])
        self.assertEqual(hhy["final_time_interior_knots"], 4)
        self.assertEqual(hhy["final_additive_interior_knots"], [3, 3])
        self.assertIn(4, hhy["bic_knot_candidates"])
        zy = macs_tuning(MACS_ZY, 6, quick=False)
        self.assertEqual(zy["time_interior_knots"], [5, 1, 3])
        self.assertEqual(zy["additive_interior_knots"], [2, 5])

    def test_common_identification_preserves_mean_surface(self):
        grid = np.linspace(0.0, 1.0, 201)
        baseline = 1.0 + grid
        beta = 2.0 + grid
        phi = 3.0 + np.sin(2.0 * np.pi * grid)
        curves = (
            {"component": "baseline", "grid": grid, "values": baseline},
            {"component": "beta_1", "grid": grid, "values": beta},
            {"component": "phi_1", "grid": grid, "values": phi},
        )
        identified, audit = _common_identify_curves(curves, n_covariates=1)
        mapped = {
            str(curve["component"]): np.asarray(curve["values"], dtype=float)
            for curve in identified
        }
        before = baseline[:, None] + beta[:, None] * phi[None, :]
        after = (
            mapped["baseline"][:, None]
            + mapped["beta_1"][:, None] * mapped["phi_1"][None, :]
        )
        np.testing.assert_allclose(after, before, atol=1e-10)
        self.assertAlmostEqual(float(np.trapezoid(mapped["beta_1"], grid)), 1.0)
        self.assertAlmostEqual(float(np.trapezoid(mapped["phi_1"], grid)), 0.0)
        self.assertEqual(audit["invalid_blocks"], [])

    def test_macs_cv_fold_never_extracts_factor_curves(self):
        dataset = _tiny_macs_dataset()
        adapter = _CurveAuditAdapter()
        split = SimpleNamespace(
            repeat=0,
            fold=0,
            seed=123,
            train_hash="train-hash",
            test_hash="test-hash",
        )
        row, predictions, curves = _run_fold_method(
            adapter,
            mode="quick",
            quick=True,
            variant="primary",
            basis_dimension=4,
            split=split,
            dataset=dataset,
            train=dataset,
            test=dataset,
            applicability="applicable",
            reason="fixture",
            admission="not_required",
            preflight_report=adapter.preflight(),
        )
        self.assertEqual(row["attempt_status"], "success")
        self.assertEqual(len(predictions), dataset.n_rows)
        self.assertIsNone(curves)
        self.assertEqual(adapter.factor_curve_calls, 0)

    def test_macs_registered_full_data_fit_is_single_and_jointly_identified(self):
        dataset = _tiny_macs_dataset()
        adapter = _CurveAuditAdapter()
        curve_row, audit = _fit_registered_full_data_trace_curves(
            adapter,
            mode="quick",
            quick=True,
            variant="primary",
            basis_dimension=4,
            dataset=dataset,
            seed=20260810,
            preflight_report=adapter.preflight(),
        )
        self.assertIsNotNone(curve_row)
        assert curve_row is not None
        self.assertEqual(adapter.fit_datasets, [dataset])
        self.assertEqual(adapter.factor_curve_calls, 1)
        self.assertEqual(curve_row["fit_scope"], "registered_full_data")
        self.assertEqual(curve_row["data_hash"], dataset.data_hash)
        self.assertEqual(audit["attempt_status"], "success")
        self.assertEqual(audit["curve_aggregation"], "none")
        self.assertFalse(audit["fold_curves_serialized"])
        self.assertEqual(len(str(audit["tuning_sha256"])), 64)
        self.assertEqual(len(str(audit["identified_curves_sha256"])), 64)
        self.assertEqual(len(str(audit["curve_row_sha256"])), 64)
        mapped = {
            str(item["component"]): (
                np.asarray(item["grid"], dtype=float),
                np.asarray(item["values"], dtype=float),
            )
            for item in curve_row["curves"]
        }
        for index in (1, 2):
            beta_grid, beta_values = mapped[f"beta_{index}"]
            phi_grid, phi_values = mapped[f"phi_{index}"]
            self.assertAlmostEqual(float(np.trapezoid(beta_values, beta_grid)), 1.0)
            self.assertAlmostEqual(float(np.trapezoid(phi_values, phi_grid)), 0.0)
        self.assertEqual(
            audit["common_factor_identification"]["invalid_blocks"], []
        )

    def test_missing_active_factor_receives_finite_zero_estimate_penalty(self):
        truth = SimpleNamespace(
            active=(True,),
            beta0=lambda t: np.ones_like(t),
            beta=(lambda t: 2.0 * np.ones_like(t),),
            phi=(lambda z: z - 0.5,),
        )
        raw = SimpleNamespace(
            domain_time=(0.0, 1.0),
            domain_covariates=((0.0, 1.0),),
            truth=truth,
        )
        curves = (
            {
                "component": "baseline",
                "grid": np.linspace(0.0, 1.0, 11),
                "values": np.ones(11),
            },
        )
        baseline_ise, component_ise, factor_ise = _truth_metrics(raw, curves)
        self.assertAlmostEqual(baseline_ise, 0.0)
        self.assertGreater(component_ise, 0.0)
        self.assertGreater(factor_ise, 0.0)
        self.assertTrue(np.isfinite(component_ise))
        self.assertTrue(np.isfinite(factor_ise))

    def test_published_target_rows_match_their_source_scenarios(self):
        payload = json.loads(
            (ROOT / "protocol" / "published_targets.json").read_text(
                encoding="utf-8"
            )
        )["targets"]
        hhy = payload["repro-hhy2021-n30-t2"]
        self.assertEqual(hhy["reported_value"], 0.0469)
        self.assertIn("0.5 t(2)", hhy["source"])
        zy = payload["ZY2025-table1-n200-sigma0.1"]
        self.assertEqual(zy["metric"], "noise_free_test_mspe")

    def test_parallel_job_count_is_explicit_and_positive(self):
        args = parse_strict_args(
            ["--quick", "--output", "unused", "--jobs", "2"]
        )
        self.assertEqual(args.jobs, 2)
        with self.assertRaises(SystemExit):
            parse_strict_args(
                ["--quick", "--output", "unused", "--jobs", "0"]
            )
        macs_args = parse_macs_args(
            ["--quick", "--output", "unused", "--jobs", "3"]
        )
        self.assertEqual(macs_args.jobs, 3)
        with self.assertRaises(SystemExit):
            parse_macs_args(
                ["--quick", "--output", "unused", "--jobs", "0"]
            )

    def test_macs_can_inherit_audited_admissions_without_global_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict_metadata.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "vcam-strict-benchmark/1",
                        "mode": "formal",
                        "formal_protocol_complete": True,
                        "formal_claims_eligible": False,
                        "admission_gates": {
                            "HHY2021-Huber": {"passed": True, "status": "admitted"},
                            "ZZW2020": {
                                "passed": False,
                                "status": "reproduction_mismatch",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            admissions, source = _load_admissions(path)
            self.assertEqual(admissions["HHY2021-Huber"], "admitted")
            self.assertEqual(admissions["ZZW2020"], "reproduction_mismatch")
            self.assertTrue(source["formal_protocol_complete"])
            self.assertFalse(source["formal_claims_eligible"])

    def test_formal_registry_locks_requested_examples(self):
        scenarios = registered_scenarios(quick=False)
        by_id = {item.scenario: item for item in scenarios}
        self.assertEqual(by_id["example1-zw2015-n100"].formal_replications, 500)
        example2 = [item for item in scenarios if item.example == "Example 2"]
        self.assertEqual({item.formal_replications for item in example2}, {300})
        self.assertEqual(
            {int(item.parameters["n_subjects"]) for item in example2}, {50, 100, 200}
        )
        example3 = [item for item in scenarios if item.example == "Example 3"]
        self.assertEqual({item.formal_replications for item in example3}, {100})
        self.assertEqual({int(item.parameters["n_covariates"]) for item in example3}, {10})
        self.assertTrue(any("contamination" in item.scenario for item in example3))
        scaling = [item for item in scenarios if item.example == "Scaling"]
        self.assertEqual({int(item.parameters["n_covariates"]) for item in scaling}, {10, 25, 50})
        self.assertEqual({item.formal_replications for item in scaling}, {5})

    def test_zzw_uses_cv_only_for_the_source_p2_design(self):
        scenarios = registered_scenarios(quick=False)
        example2 = next(
            item for item in scenarios if item.scenario == "example2-gaussian-n50-sigma0.1"
        )
        example3 = next(
            item for item in scenarios if item.scenario == "example3-gaussian-n50-p10-sigma0.1"
        )
        p2 = _default_tuning(ZZW, example2, quick=False)
        p10 = _default_tuning(ZZW, example3, quick=False)
        self.assertEqual(p2["tuning_mode"], "paper_cv_registered_vectors")
        self.assertIn("knot_candidate_vectors", p2)
        self.assertEqual(p2["cv_folds"], 5)
        self.assertEqual(p10["tuning_mode"], "paper_aligned_fixed_p10_extension")
        self.assertNotIn("knot_candidate_vectors", p10)
        self.assertNotIn("cv_folds", p10)
        self.assertEqual(p10["time_interior_knots"], [4] + [2] * 10)
        self.assertEqual(p10["covariate_interior_knots"], [2] * 10)
        self.assertIn("no high-dimensional Zhang--Zhong--Wang", p10["fixed_knot_provenance"])
        self.assertEqual(p10["max_outer"], 50)
        self.assertEqual(p10["max_inner"], 50)
        self.assertIn("iteration-limit failure", p10["high_dimensional_iteration_budget"])

    def test_quick_registry_is_a_strict_subset(self):
        quick = registered_scenarios(quick=True)
        formal = registered_scenarios(quick=False)
        self.assertLess(len(quick), len(formal))
        self.assertTrue({item.scenario for item in quick}.issubset({item.scenario for item in formal}))
        self.assertEqual(sum(item.phase == "reproduction" for item in quick), 0)
        audited = registered_scenarios(
            quick=True, include_reproduction_audit=True
        )
        self.assertEqual(sum(item.phase == "reproduction" for item in audited), 5)

    def test_gate_requires_sourced_target_and_complete_q(self):
        scenario = next(
            item
            for item in registered_scenarios(
                quick=False, include_reproduction_audit=True
            )
            if item.scenario == "ZY2025-table1-n200-sigma0.1"
        )
        rows = [
            {
                "scenario": scenario.scenario,
                "method": ZY,
                "attempt_status": "success",
                "converged": True,
                "test_mse": 0.0575 + 0.0001 * np.sin(index),
            }
            for index in range(300)
        ]
        pending = assess_reproduction_gate(rows, scenario, None, expected_replications=300)
        self.assertFalse(pending["passed"])
        self.assertEqual(pending["status"], "pending_no_published_target")
        target = {
            "method": ZY,
            "metric": "test_mse",
            "reported_mean": 0.0575,
            "reported_sd": 0.0376,
            "reported_replications": 300,
            "rounding_tolerance": 0.00005,
            "source": "Zhao and Yang (2025), Table 1",
        }
        passed = assess_reproduction_gate(rows, scenario, target, expected_replications=300)
        self.assertTrue(passed["passed"])
        incomplete = assess_reproduction_gate(rows[:-1], scenario, target, expected_replications=300)
        self.assertFalse(incomplete["passed"])
        self.assertEqual(incomplete["status"], "incomplete_reproduction")

    def test_subject_split_and_hash_are_data_level_not_method_level(self):
        scenario = next(
            item
            for item in registered_scenarios(quick=True)
            if item.scenario == "example2-gaussian-n50-sigma0.4"
        )
        raw = scenario.build(123)
        dataset = _subject_dataset(raw, scenario)
        train, test, split = _split_dataset(dataset, raw, split_seed=456)
        self.assertEqual(set(train.subjects).intersection(test.subjects), set())
        self.assertEqual(train.n_subjects + test.n_subjects, dataset.n_subjects)
        self.assertEqual(len(split.train_hash), 64)
        self.assertEqual(len(split.test_hash), 64)

    def test_macs_protocol_uses_raw_cd4_and_scaled_coordinates(self):
        raw = read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv")
        self.assertEqual(len(raw.cd4), 2376)
        self.assertEqual(len(np.unique(raw.person)), 369)
        bounds = {
            "time": (float(raw.time.min()), float(raw.time.max())),
            "age": (float(raw.age.min()), float(raw.age.max())),
            "cesd": (float(raw.cesd.min()), float(raw.cesd.max())),
        }
        primary = prepare_macs_variant(raw, variant="primary", global_bounds=bounds)
        self.assertTrue(np.array_equal(primary.response, raw.cd4))
        self.assertGreaterEqual(float(primary.time.min()), 0.0)
        self.assertLessEqual(float(primary.time.max()), 1.0)
        self.assertTrue(np.all((primary.covariates >= 0.0) & (primary.covariates <= 1.0)))
        deleted = prepare_macs_variant(
            raw, variant="delete_outer_fence_subjects", global_bounds=bounds
        )
        self.assertLess(deleted.n_subjects, primary.n_subjects)
        self.assertEqual(deleted.n_subjects, 359)
        winsor = prepare_macs_variant(
            raw, variant="winsorize_response_1_99", global_bounds=bounds
        )
        self.assertEqual(winsor.n_subjects, 369)
        self.assertLess(float(winsor.response.max()), float(primary.response.max()))

    def test_generated_latex_never_uses_tiny_resize_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.tex"
            _write_table(
                path,
                caption="A focused question.",
                label="tab:test",
                columns=("Method", "MSPE"),
                alignment="lc",
                body=(("TRACE--VCAM", "1.23"),),
                claims_eligible=False,
            )
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("scriptsize", text)
            self.assertNotIn("resizebox", text)
            self.assertIn("exploratory output", text)

    def test_claim_switch_is_false_until_formal_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claims.tex"
            write_claim_macros(path, False, "abc")
            text = path.read_text(encoding="utf-8")
            self.assertIn("strictclaimsfalse", text)
            self.assertNotIn("strictclaimstrue", text)
            write_claim_macros(path, True, "def")
            self.assertIn("strictclaimstrue", path.read_text(encoding="utf-8"))

    def test_audited_artifacts_can_be_visible_while_global_claims_stay_off(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claims.tex"
            write_claim_macros(path, False, "abc", artifacts_ready=True)
            text = path.read_text(encoding="utf-8")
            self.assertIn("strictclaimsfalse", text)
            self.assertIn("strictartifactsreadytrue", text)

    def test_zsy_table9_components_sum_to_registered_target(self):
        registry = json.loads(
            (ROOT / "protocol" / "published_targets.json").read_text(encoding="utf-8")
        )
        target = registry["targets"]["repro-zsy2026-n50-p10-sigma0.1"]
        components = target["reported_components"]
        total = (
            components["beta0"]
            + sum(components["beta1_to_beta10"])
            + sum(components["phi1_to_phi10"])
        )
        self.assertAlmostEqual(total, 30.0234, places=10)
        self.assertAlmostEqual(total, target["reported_value"], places=10)


if __name__ == "__main__":
    unittest.main()

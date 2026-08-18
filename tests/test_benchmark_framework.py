from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from benchmarks.adapters import (
    BenchmarkAdapter,
    FitArtifact,
    HHY2021Adapter,
    PreflightReport,
    ZW2015Adapter,
    ZY2025Adapter,
    ZZW2020Adapter,
)
from benchmarks.admission import PublishedTarget, ZY2025_TABLE1_TARGETS, assess_reproduction
from benchmarks.data import (
    SubjectDataset,
    make_repeated_subject_folds,
    read_exchange_bundle,
    write_exchange_bundle,
)
from benchmarks.methods import (
    FIXED_METHOD_LABELS,
    Applicability,
    MethodLabel,
    Protocol,
    applicability_for,
)
from benchmarks.protocol import FailureInfo, metric_summary, run_replication
from benchmarks.vendor import verify_zsy2026_vendor
from benchmarks.adapters.external import (
    _r_compatible_seed,
    _validate_fdapace_spline_spec,
    _zsy_source_df,
)
from benchmarks.adapters.zzw2020 import _paper_subject_fold_rss
from benchmarks.adapters.splines import (
    SplineBasis,
    SplineFunction,
    solve_subject_balanced_huber,
    solve_paper_lasso,
    ten_fold_minimum_error_lasso,
)
from benchmarks.adapters.hhy2021 import _normalize_final_pair


def small_dataset(*, time_invariant: bool = False, n_subjects: int = 18) -> SubjectDataset:
    rng = np.random.default_rng(123)
    subjects = np.repeat(np.arange(n_subjects), 5)
    time = np.tile(np.linspace(0.02, 0.98, 5), n_subjects)
    if time_invariant:
        subject_x = rng.uniform(0.05, 0.95, size=(n_subjects, 2))
        covariates = subject_x[subjects]
    else:
        covariates = rng.uniform(0.05, 0.95, size=(len(time), 2))
    signal = (
        np.sin(np.pi * time)
        + (1.0 + time) * (covariates[:, 0] - 0.5)
        + (1.0 - 0.5 * time) * (covariates[:, 1] ** 2 - 1.0 / 3.0)
    )
    response = signal + rng.normal(0.0, 0.03, size=len(time))
    return SubjectDataset(
        time=time,
        covariates=covariates,
        response=response,
        noise_free_target=signal,
        subject_id=subjects,
        metadata={
            "time_domain": [0.0, 1.0],
            "covariate_domains": [[0.0, 1.0], [0.0, 1.0]],
            "time_invariant_covariates": time_invariant,
        },
    )


class MeanAdapter(BenchmarkAdapter):
    label = MethodLabel.TRACE_VCAM.value

    def preflight(self) -> PreflightReport:
        return PreflightReport(True, "unit-test")

    def fit(self, train, *, seed, tuning):
        del seed
        return FitArtifact(
            model=float(np.mean(train.response)),
            method=self.label,
            version="unit-test",
            tuning=dict(tuning),
            converged=True,
        )

    def predict(self, artifact, test):
        return np.repeat(artifact.model, test.n_rows)


class BenchmarkContractTests(unittest.TestCase):
    def test_hybrid_lasso_path_matches_fista_cv_rule(self) -> None:
        rng = np.random.default_rng(2026)
        features = rng.normal(size=(90, 10))
        response = features @ np.r_[1.0, -0.7, 0.4, np.zeros(7)]
        response += rng.normal(scale=0.35, size=90)
        penalties = np.logspace(0.0, -3.0, 8)
        reference = ten_fold_minimum_error_lasso(
            features,
            response,
            seed=11,
            penalty_grid=penalties,
            n_folds=5,
            tolerance=1e-7,
            max_iter=3000,
            solver="fista",
        )
        hybrid = ten_fold_minimum_error_lasso(
            features,
            response,
            seed=11,
            penalty_grid=penalties,
            n_folds=5,
            tolerance=1e-7,
            max_iter=3000,
            solver="coordinate_fista_path",
        )
        self.assertEqual(hybrid.penalty, reference.penalty)
        self.assertTrue(
            np.allclose(hybrid.mean_errors, reference.mean_errors, atol=5e-5)
        )
        self.assertEqual(sum(hybrid.nonconverged_folds), 0)

    def test_paper_lasso_honors_registered_deadline(self) -> None:
        with self.assertRaises(TimeoutError):
            solve_paper_lasso(
                np.eye(3),
                np.ones(3),
                0.1,
                deadline=time.perf_counter() - 1.0,
            )

    def test_quantile_spline_places_registered_empirical_knots(self):
        values = np.asarray([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        basis = SplineBasis.quantile(
            values, n_interior=2, order=4, domain=(0.0, 1.0)
        )
        expected = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        self.assertTrue(np.allclose(basis.knots[4:-4], expected))
        self.assertEqual(basis.dimension, 6)

    def test_fixed_registry_and_updated_zy_label(self) -> None:
        self.assertEqual(len(FIXED_METHOD_LABELS), 6)
        self.assertIn("ZY2025-paper-implementation", FIXED_METHOD_LABELS)
        self.assertEqual(len(set(FIXED_METHOD_LABELS)), len(FIXED_METHOD_LABELS))

    def test_subject_folds_and_exchange_round_trip(self) -> None:
        data = small_dataset()
        folds = make_repeated_subject_folds(data, n_splits=3, n_repeats=2, seed=77)
        self.assertEqual(len(folds), 6)
        self.assertFalse(set(folds[0].train_subjects) & set(folds[0].test_subjects))
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_exchange_bundle(directory, data, folds[0])
            restored, split = read_exchange_bundle(directory)
        self.assertEqual(bundle.data_hash, data.data_hash)
        self.assertEqual(restored.data_hash, data.data_hash)
        self.assertEqual(split.train_hash, folds[0].train_hash)

    def test_registered_applicability_is_not_runtime_availability(self) -> None:
        self.assertEqual(
            applicability_for(MethodLabel.ZW2015.value, Protocol.EXAMPLE1_DENSE).status,
            Applicability.APPLICABLE,
        )
        self.assertEqual(
            applicability_for(MethodLabel.ZW2015.value, Protocol.EXAMPLE2_GAUSSIAN).status,
            Applicability.N_A_BY_DESIGN,
        )
        self.assertEqual(
            applicability_for(MethodLabel.ZY2025.value, Protocol.EXAMPLE2_GAUSSIAN).status,
            Applicability.APPLICABLE,
        )
        for method in (MethodLabel.ZZW2020.value, MethodLabel.HHY2021_HUBER.value):
            self.assertEqual(
                applicability_for(method, Protocol.EXAMPLE3_HIGH_DIMENSIONAL).status,
                Applicability.APPLICABLE,
            )
            self.assertEqual(
                applicability_for(
                    method, Protocol.EXAMPLE3_SYMMETRIC_CONTAMINATION
                ).status,
                Applicability.APPLICABLE,
            )
            self.assertEqual(
                applicability_for(method, Protocol.SCALING).status,
                Applicability.N_A_BY_DESIGN,
            )

    def test_success_schema_and_failure_denominator(self) -> None:
        data = small_dataset()
        split = make_repeated_subject_folds(data, n_splits=3, seed=91)[0]
        result = run_replication(
            MeanAdapter(),
            data,
            split,
            protocol=Protocol.EXAMPLE2_GAUSSIAN,
            scenario_id="unit",
            replication_id=0,
            tuning={},
        )
        self.assertTrue(result.successful)
        self.assertEqual(len(result.predictions), data.subset_subjects(split.test_subjects).n_rows)
        failed = replace(
            result,
            replication_id=1,
            converged=False,
            failure=FailureInfo("x", "fit", "failed"),
            predictions=(),
            metrics={},
        )
        summary = metric_summary([result, failed], "test_mse")
        self.assertEqual(summary["n_attempted"], 2)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["failure_rate"], 0.5)

    def test_reproduction_gate_and_published_targets(self) -> None:
        data = small_dataset()
        split = make_repeated_subject_folds(data, n_splits=3, seed=92)[0]
        base = run_replication(
            MeanAdapter(),
            data,
            split,
            protocol=Protocol.EXAMPLE2_GAUSSIAN,
            scenario_id="gate",
            replication_id=0,
            tuning={},
        )
        values = [0.98, 1.00, 1.02, 1.01]
        rows = [replace(base, replication_id=index, metrics={"test_mse": value}) for index, value in enumerate(values)]
        target = PublishedTarget(
            method=base.method,
            protocol=base.protocol,
            scenario_id="gate",
            metric="test_mse",
            published_value=1.0,
            rounding_digits=2,
        )
        decision = assess_reproduction(rows, target)
        self.assertTrue(decision.admitted)
        self.assertEqual(len(ZY2025_TABLE1_TARGETS), 6)

    def test_vendor_snapshot_hashes(self) -> None:
        audit = verify_zsy2026_vendor()
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["commit"], "27d857a71807de807761a022a4e334745737761e")

    def test_large_replication_seed_maps_to_valid_r_integer(self) -> None:
        original = 2**32 + 12345
        mapped = _r_compatible_seed(original)
        self.assertEqual(mapped, original % 2_147_483_647)
        self.assertGreaterEqual(mapped, 0)
        self.assertLess(mapped, 2_147_483_647)

    def test_zsy_source_df_uses_published_spline_dimensions(self) -> None:
        self.assertEqual(_zsy_source_df(2), (8, 6, 6, 6, 6))
        self.assertEqual(_zsy_source_df(2, 7), (7, 7, 7, 7, 7))
        self.assertEqual(_zsy_source_df(2, [8, 5, 6, 7, 8]), (8, 5, 6, 7, 8))
        with self.assertRaises(ValueError):
            _zsy_source_df(2, [8, 6])
        with self.assertRaises(ValueError):
            _zsy_source_df(2, 0)

    def test_zzw_cv_uses_visit_rss_per_held_out_subject(self) -> None:
        residual = np.asarray([1.0, 1.0, 2.0])
        subject_id = np.asarray(["a", "a", "b"])
        self.assertEqual(_paper_subject_fold_rss(residual, subject_id), 3.0)
        with self.assertRaises(ValueError):
            _paper_subject_fold_rss(residual, subject_id[:2])

    def test_zzw_fixed_p10_extension_bypasses_undisclosed_cv(self) -> None:
        data = small_dataset()
        p10 = replace(
            data,
            covariates=np.column_stack([data.covariates] + [data.covariates[:, :1]] * 8),
            covariate_names=tuple(f"x_{index + 1}" for index in range(10)),
            metadata={
                "time_domain": [0.0, 1.0],
                "covariate_domains": [[0.0, 1.0]] * 10,
                "time_invariant_covariates": False,
            },
        )
        split = make_repeated_subject_folds(p10, n_splits=3, seed=93)[0]
        train = p10.subset_subjects(split.train_subjects)
        adapter = ZZW2020Adapter()
        with patch.object(
            adapter,
            "_fit_registered_subject_cv",
            side_effect=AssertionError("fixed high-dimensional extension must not enter CV"),
        ) as subject_cv:
            artifact = adapter.fit(
                train,
                seed=93,
                tuning={
                    "tuning_mode": "paper_aligned_fixed_p10_extension",
                    "time_domain": [0.0, 1.0],
                    "covariate_domains": [[0.0, 1.0]] * 10,
                    "time_interior_knots": [4] + [2] * 10,
                    "covariate_interior_knots": [2] * 10,
                    "max_outer": 1,
                    "max_inner": 1,
                },
            )
        self.assertFalse(subject_cv.called)
        self.assertIn("paper-aligned fixed p=10 extension", artifact.metadata["reproduction_mode"])

    def test_fdapace_public_package_knot_constraint_is_explicit(self) -> None:
        _validate_fdapace_spline_spec((10, 10), (3, 3), stage="additive")
        with self.assertRaisesRegex(ValueError, "paper interior-knot counts"):
            _validate_fdapace_spline_spec((5, 2), (3, 3), stage="additive")

    def test_numpy_paper_adapters_fit_and_predict(self) -> None:
        data = small_dataset()
        split = make_repeated_subject_folds(data, n_splits=3, seed=93)[0]
        train = data.subset_subjects(split.train_subjects)
        test = data.subset_subjects(split.test_subjects)
        zzw = ZZW2020Adapter().fit(
            train,
            seed=93,
            tuning={
                "time_domain": [0, 1],
                "covariate_domains": [[0, 1], [0, 1]],
                "time_interior_knots": [1, 1, 1],
                "covariate_interior_knots": [1, 1],
                "max_outer": 20,
                "max_inner": 20,
            },
        )
        self.assertTrue(np.all(np.isfinite(ZZW2020Adapter().predict(zzw, test))))
        hhy = HHY2021Adapter().fit(
            train,
            seed=93,
            tuning={
                "time_domain": [0, 1],
                "covariate_domains": [[0, 1], [0, 1]],
                "pilot_time_interior_knots": 1,
                "pilot_covariate_interior_knots": [1, 1],
                "final_time_interior_knots": 1,
                "final_additive_interior_knots": [1, 1],
            },
        )
        self.assertTrue(np.all(np.isfinite(HHY2021Adapter().predict(hhy, test))))
        zy = ZY2025Adapter().fit(
            train,
            seed=93,
            tuning={
                "tuning_mode": "paper_locked",
                "lambda_initial_additive": 1e-4,
                "lambda_initial_coefficient": 1e-4,
                "lambda_additive": 1e-4,
                "lambda_coefficient": 1e-4,
                "lambda_baseline": 1e-4,
                "time_domain": [0, 1],
                "covariate_domains": [[0, 1], [0, 1]],
                "time_interior_knots": [1, 1, 1],
                "additive_interior_knots": [1, 1],
                "max_outer": 5,
                "max_inner": 5,
                "inner_mrs_tolerance": 1e-3,
                "outer_mrs_tolerance": 1e-3,
            },
        )
        self.assertTrue(np.all(np.isfinite(ZY2025Adapter().predict(zy, test))))

    def test_paper_lasso_accepts_a_stable_fista_objective_tail(self) -> None:
        rng = np.random.default_rng(20260810)
        base = rng.normal(size=(160, 1))
        features = np.column_stack(
            [base[:, 0], base[:, 0] + 1e-5 * rng.normal(size=160), rng.normal(size=160)]
        )
        response = 1.2 * base[:, 0] + 0.15 * rng.normal(size=160)
        fit = solve_paper_lasso(
            features, response, 1e-3, tolerance=1e-8, max_iter=5000
        )
        self.assertTrue(fit.converged)
        self.assertTrue(np.all(np.isfinite(fit.coefficients)))
        self.assertLess(fit.iterations, 5000)

    def test_hhy_integral_normalization_preserves_component_surface(self) -> None:
        time_basis = SplineBasis.equidistant(
            np.linspace(0.0, 2.0, 31), n_interior=2, order=4, domain=(0.0, 2.0)
        )
        additive_basis = SplineBasis.equidistant(
            np.linspace(0.0, 1.0, 31), n_interior=1, order=4, domain=(0.0, 1.0)
        )
        coefficient = SplineFunction(
            time_basis, np.asarray([0.3, 0.7, 1.2, 0.9, 0.6, 0.4])
        )
        additive = SplineFunction(
            additive_basis, np.asarray([-0.3, 0.1, 0.5, -0.2, 0.4]), offset=-0.15
        )
        time_grid = np.linspace(0.0, 2.0, 51)
        covariate_grid = np.linspace(0.0, 1.0, 47)
        before = coefficient(time_grid)[:, None] * additive(covariate_grid)[None, :]

        normalized_coefficient, rescaled_additive, scale = _normalize_final_pair(
            coefficient, additive
        )
        after = (
            normalized_coefficient(time_grid)[:, None]
            * rescaled_additive(covariate_grid)[None, :]
        )
        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(normalized_coefficient.integral(), 1.0, places=12)
        self.assertAlmostEqual(scale, coefficient.integral(), places=12)

    def test_hhy_objective_tail_acceptance_requires_registered_tail(self) -> None:
        rng = np.random.default_rng(20260811)
        n_rows = 218
        base = rng.normal(size=n_rows)
        features = np.column_stack(
            [base + 1e-5 * rng.normal(size=n_rows) for _ in range(60)]
        )
        response = 1.2 * base + 0.5 * rng.standard_t(df=2, size=n_rows)
        weights = np.full(n_rows, 1.0 / n_rows)

        incomplete = solve_subject_balanced_huber(
            features,
            response,
            weights,
            tolerance=1e-16,
            max_iter=3,
            objective_relative_tolerance=1e-9,
            objective_stable_steps=3,
        )
        self.assertFalse(incomplete.converged)
        self.assertFalse(incomplete.objective_stable)
        self.assertEqual(incomplete.termination, "iteration_limit")

        accepted = solve_subject_balanced_huber(
            features,
            response,
            weights,
            tolerance=1e-16,
            max_iter=50,
            objective_relative_tolerance=1e-9,
            objective_stable_steps=3,
        )
        self.assertTrue(accepted.converged)
        self.assertFalse(accepted.coefficient_converged)
        self.assertTrue(accepted.objective_stable)
        self.assertEqual(accepted.termination, "objective_stable")
        tail = np.asarray(accepted.objective[-4:], dtype=float)
        self.assertTrue(np.all(np.isfinite(tail)))
        self.assertTrue(
            np.all(np.abs(np.diff(tail)) / np.maximum(1.0, np.abs(tail[1:])) <= 1e-9)
        )

    def test_hhy_p10_bic_audits_objective_stable_candidate(self) -> None:
        # This is the formerly failing p=10 Gaussian cohort, with a compact
        # candidate grid to keep the regression fixture fast.  It verifies
        # that an objective-stable finite endpoint is neither discarded nor
        # silently relabelled as strict coefficient convergence.
        from scripts.run_strict_benchmark import (
            HHY,
            _default_tuning,
            _registered_split,
            _stable_seed,
            _subject_dataset,
            registered_scenarios,
        )

        scenario = next(
            item
            for item in registered_scenarios(quick=False)
            if item.scenario == "example3-gaussian-n50-p10-sigma0.1"
        )
        seed = _stable_seed(20260810, scenario.scenario, 0, "data")
        split_seed = _stable_seed(20260810, scenario.scenario, 0, "subject-split")
        raw = scenario.build(seed)
        dataset = _subject_dataset(raw, scenario)
        train, _, _ = _registered_split(
            dataset, raw, scenario, split_seed=split_seed
        )
        tuning = _default_tuning(HHY, scenario, quick=False)
        tuning["bic_knot_candidates"] = [1, 2]
        artifact = HHY2021Adapter().fit(train, seed=seed, tuning=tuning)

        self.assertTrue(artifact.converged)
        accepted = [
            item
            for item in artifact.metadata["bic2_trace"]
            if item["converged"]
        ]
        self.assertTrue(accepted)
        self.assertTrue(
            any(item["stage2"]["objective_stable_accepted"] for item in accepted)
        )
        self.assertTrue(
            any(
                not item["stage2"]["strict_coefficient_converged"]
                for item in accepted
            )
        )
        self.assertIn("objective_stable", artifact.metadata["stage_termination_modes"])

    def test_zw2015_r_smoke_uses_original_fdapace(self) -> None:
        adapter = ZW2015Adapter()
        if not adapter.preflight().ready:
            self.skipTest(adapter.preflight().message)
        data = small_dataset(time_invariant=True, n_subjects=36)
        split = make_repeated_subject_folds(data, n_splits=3, seed=94)[0]
        artifact = adapter.fit(
            data.subset_subjects(split.train_subjects),
            seed=94,
            tuning={
                "add_nknot": [10, 10],
                "add_order": [3, 3],
                "vc_nknot": [10, 10, 10],
                "vc_order": [3, 3, 3],
                "grid_size": 51,
                "timeout_seconds": 120,
            },
        )
        prediction = adapter.predict(artifact, data.subset_subjects(split.test_subjects))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(artifact.metadata["implementation_origin"], "CRAN fdapace::VCAM")


if __name__ == "__main__":
    unittest.main()

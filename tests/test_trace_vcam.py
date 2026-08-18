from __future__ import annotations

import unittest

import numpy as np

import src.trace_vcam as trace_module

from src.simulation_dgp import Scenario, generate_data
from src.trace_vcam import (
    OrthonormalSplineBasis,
    VCAMDesign,
    fit_trace_vcam,
    huber_scores,
    huber_values,
    predict_components,
    robust_scale,
    trace_lambda_max,
)


class TraceVCAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.basis = OrthonormalSplineBasis.create(6, 6)

    def test_basis_constraints(self) -> None:
        grid = self.basis.grid
        time = self.basis.transform_time(grid)
        covariate = self.basis.transform_covariate(grid)
        time_gram = np.trapezoid(
            time[:, :, None] * time[:, None, :], grid, axis=0
        )
        covariate_gram = np.trapezoid(
            covariate[:, :, None] * covariate[:, None, :], grid, axis=0
        )
        self.assertLess(np.max(np.abs(time_gram - np.eye(6))), 1e-9)
        self.assertLess(np.max(np.abs(covariate_gram - np.eye(6))), 1e-9)
        self.assertLess(
            np.max(np.abs(np.trapezoid(covariate, grid, axis=0))), 1e-9
        )

    def test_deprecated_estimators_are_not_in_the_production_api(self) -> None:
        deprecated = (
            "adaptive_block_weights",
            "fit_tensor_ridge",
            "fit_rank_one_als",
            "fit_coefficient_lasso_als",
            "_fit_trace_vcam_legacy_tangent",
            "_tangent_design",
            "_calibration_guard_reasons",
        )
        for name in deprecated:
            with self.subTest(name=name):
                self.assertFalse(hasattr(trace_module, name))

    def _small_design(self) -> VCAMDesign:
        scenario = Scenario(
            "unit-test", "canonical", 30, 2, 2, 0.4, "contaminated"
        )
        data = generate_data(scenario, 12345)
        return VCAMDesign.from_arrays(
            data.time,
            data.covariates,
            data.response,
            data.subject,
            self.basis,
        )

    def test_subject_weights(self) -> None:
        subject = np.array([0, 1, 1, 2, 2, 2])
        design = VCAMDesign.from_arrays(
            time=np.linspace(0.05, 0.95, len(subject)),
            covariates=np.column_stack(
                [
                    np.linspace(0.1, 0.9, len(subject)),
                    np.linspace(0.9, 0.1, len(subject)),
                ]
            ),
            response=np.arange(len(subject), dtype=float),
            subject=subject,
            basis=self.basis,
        )
        subjects = np.unique(design.subject)
        masses = np.array(
            [np.sum(design.weights[design.subject == subject]) for subject in subjects]
        )
        np.testing.assert_allclose(masses, 1.0 / len(subjects), atol=1e-14)
        np.testing.assert_allclose(
            design.weights,
            np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, *([1.0 / 9.0] * 3)]),
            atol=1e-14,
        )
        self.assertAlmostEqual(float(np.sum(design.weights)), 1.0, places=14)

    def test_lambda_max_and_monotone_objective(self) -> None:
        design = self._small_design()
        delta = 1.345 * robust_scale(design.response, design.weights)
        lambda_max, _ = trace_lambda_max(design, delta)
        zero_fit = fit_trace_vcam(
            design,
            1.01 * lambda_max,
            delta,
            max_iter=800,
            tolerance=1e-6,
            post_rank_one=False,
            post_refit=False,
        )
        self.assertFalse(np.any(zero_fit.selected))
        self.assertTrue(zero_fit.convex_converged)
        self.assertLess(zero_fit.convex_kkt_residual, 1e-6)
        self.assertEqual(zero_fit.penalty_weights_source, "unit")
        np.testing.assert_array_equal(zero_fit.block_weights, np.ones(2))
        differences = np.diff(np.asarray(zero_fit.objective))
        self.assertTrue(np.all(differences <= 1e-9))

    def test_fixed_direction_scalar_postfit_is_finite_and_identified(self) -> None:
        design = self._small_design()
        delta = 1.345 * robust_scale(design.response, design.weights)
        lambda_max, _ = trace_lambda_max(design, delta)
        fit = fit_trace_vcam(
            design,
            0.2 * lambda_max,
            delta,
            max_iter=1200,
            tolerance=1e-6,
            mu=0.05,
        )
        self.assertTrue(np.all(np.isfinite(fit.gamma)))
        self.assertTrue(all(np.all(np.isfinite(matrix)) for matrix in fit.matrices))
        self.assertLess(max(np.linalg.norm(matrix) for matrix in fit.matrices), 50.0)
        self.assertTrue(fit.scalar_postfit_attempted)
        self.assertTrue(fit.scalar_postfit_converged)
        self.assertLess(fit.scalar_postfit_kkt_residual, 1e-6)
        self.assertLessEqual(
            fit.postfit_huber_loss, fit.rank_one_anchor_huber_loss + 1e-10
        )
        for active, matrix, beta, phi in zip(
            fit.selected,
            fit.matrices,
            fit.identified_time_factors,
            fit.identified_covariate_factors,
            strict=True,
        ):
            if not active:
                self.assertIsNone(beta)
                self.assertIsNone(phi)
                continue
            assert beta is not None and phi is not None
            self.assertAlmostEqual(float(design.time_integral @ beta), 1.0)
            np.testing.assert_allclose(matrix, np.outer(beta, phi), atol=1e-10)

    def test_huber_values_scores_and_subject_balanced_objective(self) -> None:
        residual = np.array([-2.0, -0.5, 0.5, 2.0])
        expected_values = np.array([1.5, 0.125, 0.125, 1.5])
        expected_scores = np.array([-1.0, -0.5, 0.5, 1.0])
        np.testing.assert_allclose(huber_values(residual, 1.0), expected_values)
        np.testing.assert_allclose(huber_scores(residual, 1.0), expected_scores)
        weights = np.array([0.125, 0.125, 0.375, 0.375])
        self.assertAlmostEqual(
            float(weights @ huber_values(residual, 1.0)),
            float(weights @ expected_values),
        )

    def test_fixed_and_mad_threshold_metadata_are_distinct(self) -> None:
        design = self._small_design()
        fixed = fit_trace_vcam(
            design,
            penalty=10.0,
            delta=1.0,
            max_iter=500,
            tolerance=1e-6,
        )
        practical = fit_trace_vcam(
            design,
            penalty=10.0,
            delta=None,
            threshold_mode="mad",
            max_iter=500,
            tolerance=1e-6,
        )
        self.assertEqual(fixed.huber_threshold_mode, "fixed")
        self.assertEqual(fixed.tuning_regime, "theory-fixed-threshold")
        self.assertTrue(fixed.metadata["fixed_threshold_theory_applies"])
        self.assertIsNone(fixed.huber_scale)
        self.assertEqual(practical.huber_threshold_mode, "mad")
        self.assertEqual(practical.tuning_regime, "practical-data-adaptive-mad")
        self.assertFalse(practical.metadata["fixed_threshold_theory_applies"])
        self.assertGreater(practical.huber_scale, 0.0)
        self.assertAlmostEqual(
            practical.delta, practical.huber_multiplier * practical.huber_scale
        )
        externally_weighted = fit_trace_vcam(
            design,
            penalty=10.0,
            delta=1.0,
            block_weights=np.array([1.0, 2.0]),
            max_iter=500,
            tolerance=1e-6,
        )
        self.assertEqual(
            externally_weighted.penalty_weights_source, "user_fixed"
        )

    def test_fit_predict_matches_component_sum(self) -> None:
        design = self._small_design()
        delta = 1.345 * robust_scale(design.response, design.weights)
        lambda_max, _ = trace_lambda_max(design, delta)
        fit = fit_trace_vcam(
            design,
            penalty=0.2 * lambda_max,
            delta=delta,
            max_iter=1200,
            tolerance=1e-6,
        )
        expected, components = predict_components(
            design, fit.gamma, fit.matrices
        )
        np.testing.assert_allclose(fit.predict(design), expected, atol=1e-12)
        np.testing.assert_allclose(
            expected,
            design.time @ fit.gamma + np.sum(components, axis=0),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

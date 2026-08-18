from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from benchmarks.adapters.base import FitArtifact
from benchmarks.adapters.trace import TraceVCAMAdapter
from src.trace_vcam import (
    OrthonormalSplineBasis,
    VCAMDesign,
    _scalar_postfit,
    huber_values,
    predict_components,
)


class ScalarPostfitTests(unittest.TestCase):
    """Regression tests for the production second stage.

    These four tests replace the former module-level diagnostic functions.
    They are intentionally unittest methods so the
    repository's documented ``unittest discover`` command executes them.
    """

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(90210)
        cls.basis = OrthonormalSplineBasis.create(6, 6)
        subjects = np.repeat(np.arange(40), 4)
        time = rng.uniform(size=len(subjects))
        covariates = rng.uniform(size=(len(subjects), 2))
        cls.design = VCAMDesign.from_arrays(
            time,
            covariates,
            np.zeros(len(subjects)),
            subjects,
            cls.basis,
        )
        cls.gamma = rng.normal(scale=0.2, size=6)
        integral = cls.design.time_integral
        beta = integral / float(np.dot(integral, integral))
        cls.matrices = [
            amplitude * np.outer(beta, rng.normal(size=6))
            for amplitude in (0.8, -0.6)
        ]
        cls.design.response = predict_components(
            cls.design, cls.gamma, cls.matrices
        )[0]
        cls.selected = np.ones(2, dtype=bool)

    def test_exact_fixed_direction_problem_is_recovered(self) -> None:
        result = _scalar_postfit(
            self.design,
            self.gamma,
            self.matrices,
            self.selected,
            delta=1.0,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.kkt_residual, 1e-7)
        np.testing.assert_allclose(
            predict_components(self.design, result.gamma, result.matrices)[0],
            self.design.response,
            atol=1e-8,
        )

    def test_identification_and_factor_reconstruction(self) -> None:
        result = _scalar_postfit(
            self.design,
            self.gamma,
            self.matrices,
            self.selected,
            delta=1.0,
        )
        for matrix, beta, phi in zip(
            result.matrices,
            result.time_factors,
            result.covariate_factors,
            strict=True,
        ):
            self.assertIsNotNone(beta)
            self.assertIsNotNone(phi)
            assert beta is not None and phi is not None
            self.assertAlmostEqual(float(self.design.time_integral @ beta), 1.0)
            np.testing.assert_allclose(matrix, np.outer(beta, phi), atol=1e-10)
            self.assertLessEqual(np.linalg.matrix_rank(matrix, tol=1e-9), 1)

    def test_scalar_huber_postfit_does_not_increase_loss(self) -> None:
        contaminated_response = self.design.response.copy()
        contaminated_response[::17] += 8.0
        contaminated = VCAMDesign(
            time=self.design.time,
            covariates=self.design.covariates,
            response=contaminated_response,
            subject=self.design.subject,
            weights=self.design.weights,
            time_integral=self.design.time_integral,
            time_roughness=self.design.time_roughness,
            covariate_roughness=self.design.covariate_roughness,
        )
        pilot_gamma = self.gamma + 0.15
        pilot_matrices = [0.7 * matrix for matrix in self.matrices]
        result = _scalar_postfit(
            contaminated,
            pilot_gamma,
            pilot_matrices,
            self.selected,
            delta=1.0,
        )
        self.assertLessEqual(
            result.final_huber_loss, result.anchor_huber_loss + 1e-10
        )
        self.assertTrue(np.all(np.diff(result.objective) <= 1e-10))
        manual = np.dot(
            contaminated.weights,
            huber_values(
                contaminated.response
                - predict_components(
                    contaminated, result.gamma, result.matrices
                )[0],
                1.0,
            ),
        )
        self.assertAlmostEqual(result.final_huber_loss, float(manual), places=10)

    def test_trace_adapter_exports_stored_identified_factors(self) -> None:
        result = _scalar_postfit(
            self.design,
            self.gamma,
            self.matrices,
            self.selected,
            delta=1.0,
        )
        fit = SimpleNamespace(
            gamma=result.gamma,
            matrices=result.matrices,
            selected=self.selected,
            identified_time_factors=result.time_factors,
            identified_covariate_factors=result.covariate_factors,
        )
        artifact = FitArtifact(
            model={
                "fit": fit,
                "basis": self.basis,
                "time_domain": (0.0, 2.0),
                "covariate_domains": ((-1.0, 1.0), (0.0, 3.0)),
            },
            method="TRACE-VCAM",
            version="test",
            tuning={},
            converged=True,
        )
        curves = TraceVCAMAdapter().factor_curves(artifact)
        by_name = {str(curve["component"]): curve for curve in curves}
        time_basis = self.basis.transform_time(self.basis.grid)
        covariate_basis = self.basis.transform_covariate(self.basis.grid)
        for index, matrix in enumerate(result.matrices, start=1):
            beta = np.asarray(by_name[f"beta_{index}"]["values"], dtype=float)
            phi = np.asarray(by_name[f"phi_{index}"]["values"], dtype=float)
            beta_grid = np.asarray(by_name[f"beta_{index}"]["grid"], dtype=float)
            self.assertAlmostEqual(
                float(np.trapezoid(beta, beta_grid) / (beta_grid[-1] - beta_grid[0])),
                1.0,
                places=5,
            )
            np.testing.assert_allclose(
                np.outer(beta, phi),
                time_basis @ matrix @ covariate_basis.T,
                atol=1e-9,
            )

    def test_unidentified_time_direction_is_reported(self) -> None:
        integral = self.design.time_integral
        candidate = np.arange(1.0, 7.0)
        left = candidate - integral * float(candidate @ integral) / float(
            integral @ integral
        )
        unidentified = [np.outer(left, np.ones(6)), np.zeros((6, 6))]
        with self.assertRaisesRegex(ValueError, "identification condition"):
            _scalar_postfit(
                self.design,
                self.gamma,
                unidentified,
                np.array([True, False]),
                delta=1.0,
            )


if __name__ == "__main__":
    unittest.main()

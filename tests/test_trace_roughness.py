from __future__ import annotations

import unittest

import numpy as np

from src.simulation_dgp import Scenario, generate_data
from src.trace_vcam import (
    OrthonormalSplineBasis,
    VCAMDesign,
    fit_trace_vcam,
    robust_scale,
    trace_lambda_max,
)
from src.trace_vcam_roughness import (
    fit_trace_vcam_roughness,
    spline_roughness_matrices,
    surface_roughness,
)


class TraceRoughnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.basis = OrthonormalSplineBasis.create()
        scenario = Scenario("unit", "unit", 24, 2, 2, 0.2, "gaussian")
        data = generate_data(scenario, 71)
        cls.design = VCAMDesign.from_arrays(
            data.time, data.covariates, data.response, data.subject, cls.basis
        )
        cls.time_penalty, cls.covariate_penalty = spline_roughness_matrices(
            cls.basis
        )

    def test_roughness_matrices_are_psd_and_normalized(self) -> None:
        for matrix in (self.time_penalty, self.covariate_penalty):
            self.assertLessEqual(float(np.max(np.abs(matrix - matrix.T))), 1e-10)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(matrix)[0]), -1e-10)
            self.assertAlmostEqual(float(np.linalg.eigvalsh(matrix)[-1]), 1.0, places=9)

    def test_zero_roughness_reproduces_trace_convex_solution(self) -> None:
        delta = 1.345 * robust_scale(self.design.response, self.design.weights)
        maximum, _ = trace_lambda_max(self.design, delta)
        arguments = dict(
            design=self.design,
            penalty=0.1 * maximum,
            delta=delta,
            max_iter=900,
            tolerance=1e-6,
            post_rank_one=False,
            post_refit=False,
        )
        baseline = fit_trace_vcam(**arguments)
        extended = fit_trace_vcam_roughness(
            **arguments,
            roughness=0.0,
            time_penalty=self.time_penalty,
            covariate_penalty=self.covariate_penalty,
        )
        self.assertTrue(np.all(np.diff(extended.objective) <= 1e-9))
        self.assertLess(
            np.linalg.norm(baseline.gamma - extended.gamma), 1e-8
        )
        self.assertLess(
            sum(
                np.linalg.norm(left - right)
                for left, right in zip(
                    baseline.matrices, extended.matrices, strict=True
                )
            ),
            1e-8,
        )

    def test_positive_penalty_reduces_raw_surface_roughness(self) -> None:
        delta = 1.345 * robust_scale(self.design.response, self.design.weights)
        maximum, _ = trace_lambda_max(self.design, delta)
        fits = [
            fit_trace_vcam_roughness(
                self.design,
                penalty=0.03 * maximum,
                delta=delta,
                roughness=mu,
                time_penalty=self.time_penalty,
                covariate_penalty=self.covariate_penalty,
                max_iter=1200,
                tolerance=1e-6,
                post_rank_one=False,
                post_refit=False,
            )
            for mu in (0.0, 0.1)
        ]
        values = [
            surface_roughness(
                fit.matrices, self.time_penalty, self.covariate_penalty
            )
            for fit in fits
        ]
        self.assertLessEqual(values[1], values[0] + 1e-8)


if __name__ == "__main__":
    unittest.main()

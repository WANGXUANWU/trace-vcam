"""Auditable implementation of Zhao--Yang (2025) from the paper equations."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np

from ..data import SubjectDataset
from ..methods import MethodLabel
from .base import BenchmarkAdapter, FitArtifact, PreflightReport
from .splines import (
    LassoSolve,
    SplineBasis,
    SplineFunction,
    solve_paper_lasso,
    ten_fold_minimum_error_lasso,
)
from .zzw2020 import _expand_counts, _registered_domains


@dataclass
class ZY2025Model:
    baseline: SplineFunction
    coefficients: tuple[SplineFunction, ...]
    additives: tuple[SplineFunction, ...]
    converged: bool
    outer_iterations: int
    inner_iterations: int
    objective: tuple[float, ...]
    selected_penalties: tuple[dict[str, object], ...]


def _prediction(
    baseline: SplineFunction,
    coefficients: Sequence[SplineFunction],
    additives: Sequence[SplineFunction],
    data: SubjectDataset,
) -> np.ndarray:
    result = baseline(data.time)
    for index, (coefficient, additive) in enumerate(
        zip(coefficients, additives, strict=True)
    ):
        result += coefficient(data.time) * additive(data.covariates[:, index])
    return np.asarray(result, dtype=float)


def _block_penalties(value: object, dimensions: Sequence[int]) -> np.ndarray:
    if np.isscalar(value):
        block_values = [float(value)] * len(dimensions)
    else:
        block_values = [float(item) for item in value]  # type: ignore[union-attr]
        if len(block_values) != len(dimensions):
            raise ValueError("block penalty vector has the wrong length")
    return np.concatenate(
        [np.repeat(block_values[index], dimension) for index, dimension in enumerate(dimensions)]
    )


def _fit_penalized(
    features: np.ndarray,
    response: np.ndarray,
    *,
    tuning: Mapping[str, object],
    key: str,
    block_dimensions: Sequence[int],
    seed: int,
    audit: list[dict[str, object]],
    penalty_cache: dict[str, float],
) -> LassoSolve:
    mode = str(tuning.get("tuning_mode", "paper_cv"))
    tolerance = float(tuning.get("lasso_tolerance", 1e-8))
    max_iter = int(tuning.get("lasso_max_iter", 5000))
    deadline_value = tuning.get("_deadline_monotonic")
    deadline = None if deadline_value is None else float(deadline_value)
    if mode == "paper_cv":
        if key in penalty_cache:
            return solve_paper_lasso(
                features,
                response,
                penalty_cache[key],
                tolerance=tolerance,
                max_iter=max_iter,
                deadline=deadline,
            )
        explicit_grid = tuning.get("penalty_grid")
        if explicit_grid is None and "cv_penalty_count" in tuning:
            n_rows = len(response)
            lambda_max = max(
                2.0 * float(np.max(np.abs(features.T @ response))) / n_rows, 1e-8
            )
            grid = lambda_max * np.logspace(
                0.0, -4.0, int(tuning["cv_penalty_count"])
            )
        else:
            grid = None if explicit_grid is None else np.asarray(explicit_grid, dtype=float)
        cv = ten_fold_minimum_error_lasso(
            features,
            response,
            seed=seed,
            penalty_grid=grid,
            n_folds=int(tuning.get("cv_folds", 10)),
            tolerance=float(tuning.get("cv_tolerance", max(tolerance, 1e-7))),
            max_iter=int(tuning.get("cv_lasso_max_iter", min(max_iter, 3000))),
            solver=str(tuning.get("cv_solver", "fista_warm_path")),
            deadline=deadline,
        )
        audit.append(
            {
                "stage": key,
                "rule": "ten-fold minimum average error",
                "solver": str(tuning.get("cv_solver", "fista_warm_path")),
                "selected_penalty": cv.penalty,
                "candidate_penalties": list(cv.candidate_penalties),
                "mean_cv_errors": list(cv.mean_errors),
                "nonconverged_folds_by_candidate": list(
                    cv.nonconverged_folds
                ),
                "solver_warning_count": int(cv.solver_warnings),
            }
        )
        penalty_cache[key] = cv.penalty
        return solve_paper_lasso(
            features,
            response,
            cv.penalty,
            tolerance=tolerance,
            max_iter=max_iter,
            deadline=deadline,
        )
    if mode != "paper_locked":
        raise ValueError("tuning_mode must be paper_cv or paper_locked")
    if key not in tuning:
        raise ValueError(f"paper_locked mode requires tuning['{key}']")
    penalty = _block_penalties(tuning[key], block_dimensions)
    fit = solve_paper_lasso(
        features,
        response,
        penalty,
        tolerance=tolerance,
        max_iter=max_iter,
        deadline=deadline,
    )
    audit.append(
        {
            "stage": key,
            "rule": "locked penalty supplied by reproduction manifest",
            "selected_penalty": [float(item) for item in np.unique(penalty)],
        }
    )
    return fit


class ZY2025Adapter(BenchmarkAdapter):
    """Squared-loss/L1 estimator in Section 2.1 of Zhao--Yang (2025).

    No author code, supplement, or software link was provided.  Consequently
    this adapter is always labelled ``paper-implementation``.  It is compared
    on the same generated data and subject splits; reproducing a printed table
    entry is retained only as an optional implementation diagnostic.
    """

    label = MethodLabel.ZY2025.value

    def preflight(self) -> PreflightReport:
        try:
            import scipy

            return PreflightReport(True, f"paper-equations/2025; scipy-{scipy.__version__}")
        except Exception as error:  # pragma: no cover
            return PreflightReport(False, "unavailable", "python_dependency_failure", str(error))

    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        p = train.covariates.shape[1]
        solver_tuning = dict(tuning)
        timeout_seconds = float(tuning.get("timeout_seconds", 0.0))
        if timeout_seconds > 0.0:
            solver_tuning["_deadline_monotonic"] = (
                time.perf_counter() + timeout_seconds
            )
        order = int(tuning.get("spline_order", 4))
        time_counts = _expand_counts(tuning.get("time_interior_knots"), p + 1, 2)
        additive_counts = _expand_counts(tuning.get("additive_interior_knots"), p, 2)
        time_domain, covariate_domains = _registered_domains(train, tuning)
        time_bases = tuple(
            SplineBasis.equidistant(
                train.time, n_interior=count, order=order, domain=time_domain
            )
            for count in time_counts
        )
        additive_bases = tuple(
            SplineBasis.equidistant(
                train.covariates[:, index],
                n_interior=count,
                order=order,
                domain=covariate_domains[index],
            )
            for index, count in enumerate(additive_counts)
        )
        penalty_audit: list[dict[str, object]] = []
        penalty_cache: dict[str, float] = {}
        all_lasso_converged = True

        # Step 1a: beta_k=1 turns the VCAM into an additive model.
        init_additive_blocks = [time_bases[0].transform(train.time)]
        init_additive_blocks.extend(
            basis.transform(train.covariates[:, index])
            for index, basis in enumerate(additive_bases)
        )
        init_additive_features = np.column_stack(init_additive_blocks)
        init_additive = _fit_penalized(
            init_additive_features,
            train.response,
            tuning=solver_tuning,
            key="lambda_initial_additive",
            block_dimensions=[basis.dimension for basis in time_bases[:1]]
            + [basis.dimension for basis in additive_bases],
            seed=seed,
            audit=penalty_audit,
            penalty_cache=penalty_cache,
        )
        all_lasso_converged &= init_additive.converged
        cursor = time_bases[0].dimension
        baseline = SplineFunction(time_bases[0], init_additive.coefficients[:cursor])
        additives: list[SplineFunction] = []
        for basis in additive_bases:
            next_cursor = cursor + basis.dimension
            additives.append(
                SplineFunction(
                    basis, init_additive.coefficients[cursor:next_cursor]
                ).centered_lebesgue()
            )
            cursor = next_cursor

        # Step 1b: fixed phi turns the VCAM into a varying-coefficient model.
        init_vc_blocks = [time_bases[0].transform(train.time)]
        for index, basis in enumerate(time_bases[1:]):
            init_vc_blocks.append(
                additives[index](train.covariates[:, index])[:, None]
                * basis.transform(train.time)
            )
        init_vc_features = np.column_stack(init_vc_blocks)
        init_vc = _fit_penalized(
            init_vc_features,
            train.response,
            tuning=solver_tuning,
            key="lambda_initial_coefficient",
            block_dimensions=[basis.dimension for basis in time_bases],
            seed=seed + 1,
            audit=penalty_audit,
            penalty_cache=penalty_cache,
        )
        all_lasso_converged &= init_vc.converged
        cursor = time_bases[0].dimension
        baseline = SplineFunction(time_bases[0], init_vc.coefficients[:cursor])
        coefficients: list[SplineFunction] = []
        for basis in time_bases[1:]:
            next_cursor = cursor + basis.dimension
            candidate = SplineFunction(
                basis, init_vc.coefficients[cursor:next_cursor]
            )
            if np.linalg.norm(candidate.coefficients) > 1e-12:
                candidate = candidate.normalized_mean_one()
            coefficients.append(candidate)
            cursor = next_cursor

        inner_tolerance = float(tuning.get("inner_mrs_tolerance", 1e-4))
        outer_tolerance = float(tuning.get("outer_mrs_tolerance", 1e-4))
        max_inner = int(tuning.get("max_inner", 100))
        max_outer = int(tuning.get("max_outer", 100))
        total_inner = 0
        objective = [
            float(
                np.mean(
                    (train.response - _prediction(baseline, coefficients, additives, train))
                    ** 2
                )
            )
        ]
        outer_converged = False
        for outer_iteration in range(1, max_outer + 1):
            inner_old = objective[-1]
            inner_converged = False
            for inner_iteration in range(1, max_inner + 1):
                # Step 2a: update every phi block simultaneously.
                additive_features = np.column_stack(
                    [
                        coefficients[index](train.time)[:, None]
                        * basis.transform(train.covariates[:, index])
                        for index, basis in enumerate(additive_bases)
                    ]
                )
                additive_fit = _fit_penalized(
                    additive_features,
                    train.response - baseline(train.time),
                    tuning=solver_tuning,
                    key="lambda_additive",
                    block_dimensions=[basis.dimension for basis in additive_bases],
                    seed=seed + 1000 * outer_iteration + 2 * inner_iteration,
                    audit=penalty_audit,
                    penalty_cache=penalty_cache,
                )
                all_lasso_converged &= additive_fit.converged
                cursor = 0
                for index, basis in enumerate(additive_bases):
                    next_cursor = cursor + basis.dimension
                    additives[index] = SplineFunction(
                        basis, additive_fit.coefficients[cursor:next_cursor]
                    ).centered_lebesgue()
                    cursor = next_cursor

                # Step 2b: update every beta_1,...,beta_d block simultaneously.
                coefficient_features = np.column_stack(
                    [
                        additives[index](train.covariates[:, index])[:, None]
                        * basis.transform(train.time)
                        for index, basis in enumerate(time_bases[1:])
                    ]
                )
                coefficient_fit = _fit_penalized(
                    coefficient_features,
                    train.response - baseline(train.time),
                    tuning=solver_tuning,
                    key="lambda_coefficient",
                    block_dimensions=[basis.dimension for basis in time_bases[1:]],
                    seed=seed + 1000 * outer_iteration + 2 * inner_iteration + 1,
                    audit=penalty_audit,
                    penalty_cache=penalty_cache,
                )
                all_lasso_converged &= coefficient_fit.converged
                cursor = 0
                for index, basis in enumerate(time_bases[1:]):
                    next_cursor = cursor + basis.dimension
                    candidate = SplineFunction(
                        basis, coefficient_fit.coefficients[cursor:next_cursor]
                    )
                    if np.linalg.norm(candidate.coefficients) > 1e-12:
                        candidate = candidate.normalized_mean_one()
                    coefficients[index] = candidate
                    cursor = next_cursor
                inner_new = float(
                    np.mean(
                        (
                            train.response
                            - _prediction(baseline, coefficients, additives, train)
                        )
                        ** 2
                    )
                )
                total_inner += 1
                if abs(inner_old - inner_new) <= inner_tolerance:
                    inner_converged = True
                    break
                inner_old = inner_new

            # Step 3: update beta_0 alone, then return to Step 2 if needed.
            component_sum = np.zeros(train.n_rows, dtype=float)
            for index in range(p):
                component_sum += coefficients[index](train.time) * additives[index](
                    train.covariates[:, index]
                )
            baseline_fit = _fit_penalized(
                time_bases[0].transform(train.time),
                train.response - component_sum,
                tuning=solver_tuning,
                key="lambda_baseline",
                block_dimensions=[time_bases[0].dimension],
                seed=seed + 100000 + outer_iteration,
                audit=penalty_audit,
                penalty_cache=penalty_cache,
            )
            all_lasso_converged &= baseline_fit.converged
            baseline = SplineFunction(time_bases[0], baseline_fit.coefficients)
            outer_new = float(
                np.mean(
                    (train.response - _prediction(baseline, coefficients, additives, train))
                    ** 2
                )
            )
            objective.append(outer_new)
            if inner_converged and abs(objective[-2] - outer_new) <= outer_tolerance:
                outer_converged = True
                break

        selected = tuple(
            index
            for index, (coefficient, additive) in enumerate(
                zip(coefficients, additives, strict=True)
            )
            if np.linalg.norm(coefficient.coefficients) > 1e-10
            and np.linalg.norm(additive.coefficients) > 1e-10
        )
        converged = bool(outer_converged and all_lasso_converged)
        model = ZY2025Model(
            baseline=baseline,
            coefficients=tuple(coefficients),
            additives=tuple(additives),
            converged=converged,
            outer_iterations=outer_iteration,
            inner_iterations=total_inner,
            objective=tuple(objective),
            selected_penalties=tuple(penalty_audit),
        )
        recorded_tuning = dict(tuning)
        recorded_tuning.update(
            {
                "tuning_mode": str(tuning.get("tuning_mode", "paper_cv")),
                "spline_order": order,
                "time_interior_knots": list(time_counts),
                "additive_interior_knots": list(additive_counts),
                "time_domain": list(time_domain),
                "covariate_domains": [list(item) for item in covariate_domains],
                "inner_mrs_tolerance": inner_tolerance,
                "outer_mrs_tolerance": outer_tolerance,
            }
        )
        return FitArtifact(
            model=model,
            method=self.label,
            version=self.preflight().version,
            tuning=recorded_tuning,
            converged=converged,
            selected_blocks=selected,
            metadata={
                "implementation_origin": "paper equations, no author code available",
                "algorithm": "Zhao-Yang (2025), Section 2.1, Steps 1--3",
                "loss": "mean squared error plus coefficientwise L1 penalties",
                "internal_tuning": "ten-fold minimum average error" if recorded_tuning["tuning_mode"] == "paper_cv" else "locked reproduction penalties",
                "cv_reuse": "one selected penalty per named penalty family, reused across inner/outer updates",
                "cv_solver": str(tuning.get("cv_solver", "fista_warm_path")),
                "timeout_seconds": timeout_seconds if timeout_seconds > 0 else None,
                "penalty_audit": penalty_audit,
                "outer_iterations": outer_iteration,
                "inner_iterations": total_inner,
                "stopping_note": "paper states MRS convergence but gives no numeric tolerance; registered safety tolerances are recorded",
                "admission_requirement": "same-setting comparison; published-value reproduction is not required",
            },
        )

    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> np.ndarray:
        model: ZY2025Model = artifact.model
        return _prediction(model.baseline, model.coefficients, model.additives, test)

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        model: ZY2025Model = artifact.model
        functions: list[tuple[str, str, SplineFunction]] = [
            ("baseline", "time", model.baseline)
        ]
        functions.extend(
            (f"beta_{index + 1}", "time", function)
            for index, function in enumerate(model.coefficients)
        )
        functions.extend(
            (f"phi_{index + 1}", f"covariate_{index + 1}", function)
            for index, function in enumerate(model.additives)
        )
        curves = []
        for component, domain_name, function in functions:
            grid = np.linspace(function.basis.lower, function.basis.upper, 201)  # type: ignore[union-attr]
            curves.append(
                {
                    "component": component,
                    "domain": domain_name,
                    "grid": grid.tolist(),
                    "values": function(grid).tolist(),
                }
            )
        return tuple(curves)

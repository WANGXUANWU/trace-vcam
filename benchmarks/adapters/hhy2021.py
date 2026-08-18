"""Three-stage subject-balanced Huber adapter for Hu--Huang--You (2021)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from ..data import SubjectDataset
from ..methods import MethodLabel
from .base import BenchmarkAdapter, FitArtifact, PreflightReport
from .splines import (
    CenteredSplineBasis,
    HuberSolve,
    SplineBasis,
    SplineFunction,
    huber_objective,
    solve_subject_balanced_huber,
    subject_balanced_weights,
)
from .zzw2020 import _expand_counts, _registered_domains


def _tensor_rows(time_basis: np.ndarray, covariate_basis: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", time_basis, covariate_basis).reshape(
        len(time_basis), -1
    )


@dataclass
class HHY2021Model:
    baseline: SplineFunction
    coefficients: tuple[SplineFunction, ...]
    additives: tuple[SplineFunction, ...]
    pilot_anchors: tuple[float, ...]
    converged: bool
    stage_iterations: tuple[int, int, int]
    stage_objectives: tuple[tuple[float, ...], ...]
    stage_termination_modes: tuple[str, str, str]
    stage_strict_coefficient_converged: tuple[bool, bool, bool]
    stage_objective_stable: tuple[bool, bool, bool]
    final_coefficient_normalization_scales: tuple[float, ...]


@dataclass
class _PilotFit:
    baseline: SplineFunction
    coefficients: tuple[SplineFunction, ...]
    anchors: tuple[float, ...]
    solve: HuberSolve
    time_knots: int
    additive_knots: int


@dataclass
class _FinalFit:
    baseline: SplineFunction
    coefficients: tuple[SplineFunction, ...]
    additives: tuple[SplineFunction, ...]
    stage2: HuberSolve
    stage3: HuberSolve
    coefficient_knots: int
    additive_knots: int
    coefficient_normalization_scales: tuple[float, ...]


def _huber(
    features: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    delta: float,
    tuning: Mapping[str, object],
) -> HuberSolve:
    objective_relative_tolerance = tuning.get(
        "irls_objective_relative_tolerance", 1e-9
    )
    if objective_relative_tolerance is not None:
        objective_relative_tolerance = float(objective_relative_tolerance)
    return solve_subject_balanced_huber(
        features,
        response,
        weights,
        delta=delta,
        tolerance=float(tuning.get("irls_tolerance", 1e-8)),
        max_iter=int(tuning.get("irls_max_iter", 300)),
        objective_relative_tolerance=objective_relative_tolerance,
        objective_stable_steps=int(tuning.get("irls_objective_stable_steps", 3)),
    )


def _solve_audit(solve: HuberSolve) -> dict[str, object]:
    """Return the explicit IRLS termination record used by BIC selection."""

    return {
        "converged": bool(solve.converged),
        "strict_coefficient_converged": bool(solve.coefficient_converged),
        "objective_stable_accepted": bool(solve.objective_stable),
        "termination": str(solve.termination),
        "iterations": int(solve.iterations),
    }


def _normalize_final_pair(
    coefficient: SplineFunction,
    additive: SplineFunction,
    *,
    tolerance: float = 1e-10,
) -> tuple[SplineFunction, SplineFunction, float]:
    """Impose the paper's beta integral constraint without changing beta*phi.

    The final varying-coefficient M-step estimates each pair on its original
    scale.  Dividing beta by its integral must therefore multiply the paired
    additive function by that same integral; otherwise prediction and BIC are
    evaluated on a different surface from the one just fitted.
    """

    scale = float(coefficient.integral())
    if not np.isfinite(scale) or abs(scale) <= tolerance:
        raise FloatingPointError("final coefficient function has a near-zero Lebesgue integral")
    normalized_coefficient = coefficient.normalized_integral_one(tolerance=tolerance)
    rescaled_additive = SplineFunction(
        additive.basis,
        np.asarray(additive.coefficients, dtype=float) * scale,
        float(additive.offset) * scale,
    )
    return normalized_coefficient, rescaled_additive, scale


def _paper_bic(objective: float, n_parameters: int, n_rows: int) -> float:
    if not np.isfinite(objective) or objective <= 0.0:
        return float("inf")
    return float(math.log(objective) + math.log(n_rows) * n_parameters / (2.0 * n_rows))


def _pilot_fit(
    train: SubjectDataset,
    *,
    order: int,
    delta: float,
    weights: np.ndarray,
    time_domain: tuple[float, float],
    covariate_domains: tuple[tuple[float, float], ...],
    time_knots: int,
    additive_knots: int,
    tuning: Mapping[str, object],
) -> _PilotFit:
    p = train.covariates.shape[1]
    time_basis = SplineBasis.quantile(
        train.time,
        n_interior=time_knots,
        order=order,
        domain=time_domain,
    )
    covariate_raw = tuple(
        SplineBasis.quantile(
            train.covariates[:, index],
            n_interior=additive_knots,
            order=order,
            domain=covariate_domains[index],
        )
        for index in range(p)
    )
    covariate_bases = tuple(
        CenteredSplineBasis.empirical(raw, train.covariates[:, index])
        for index, raw in enumerate(covariate_raw)
    )
    time_matrix = time_basis.transform(train.time)
    stage1_blocks = [time_matrix]
    for index, basis in enumerate(covariate_bases):
        stage1_blocks.append(
            _tensor_rows(time_matrix, basis.transform(train.covariates[:, index]))
        )
    stage1 = _huber(
        np.column_stack(stage1_blocks), train.response, weights, delta, tuning
    )
    cursor = time_basis.dimension
    baseline = SplineFunction(time_basis, stage1.coefficients[:cursor])
    coefficient_functions: list[SplineFunction] = []
    anchors: list[float] = []
    probabilities = np.asarray(
        tuning.get("anchor_quantiles", (0.25, 0.5, 0.75)), dtype=float
    )
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise ValueError("anchor_quantiles must lie strictly between zero and one")
    for index, basis in enumerate(covariate_bases):
        width = time_basis.dimension * basis.dimension
        matrix = stage1.coefficients[cursor : cursor + width].reshape(
            time_basis.dimension, basis.dimension
        )
        cursor += width
        candidates = np.unique(
            np.quantile(train.covariates[:, index], probabilities)
        )
        candidate_values = basis.transform(candidates)
        integrated_slices = time_basis.integral_vector() @ matrix @ candidate_values.T
        anchor_index = int(np.argmax(np.abs(integrated_slices)))
        anchor = float(candidates[anchor_index])
        anchors.append(anchor)
        time_coefficients = matrix @ candidate_values[anchor_index]
        coefficient_functions.append(
            SplineFunction(time_basis, time_coefficients).normalized_integral_one()
        )
    return _PilotFit(
        baseline=baseline,
        coefficients=tuple(coefficient_functions),
        anchors=tuple(anchors),
        solve=stage1,
        time_knots=time_knots,
        additive_knots=additive_knots,
    )


def _final_fit(
    train: SubjectDataset,
    pilot: _PilotFit,
    *,
    order: int,
    delta: float,
    weights: np.ndarray,
    time_domain: tuple[float, float],
    covariate_domains: tuple[tuple[float, float], ...],
    coefficient_knots: int,
    additive_knots: int,
    tuning: Mapping[str, object],
) -> _FinalFit:
    p = train.covariates.shape[1]
    additive_raw = tuple(
        SplineBasis.quantile(
            train.covariates[:, index],
            n_interior=additive_knots,
            order=order,
            domain=covariate_domains[index],
        )
        for index in range(p)
    )
    additive_bases = tuple(
        CenteredSplineBasis.empirical(raw, train.covariates[:, index])
        for index, raw in enumerate(additive_raw)
    )
    stage2_blocks = [
        pilot.coefficients[index](train.time)[:, None]
        * basis.transform(train.covariates[:, index])
        for index, basis in enumerate(additive_bases)
    ]
    stage2 = _huber(
        np.column_stack(stage2_blocks),
        train.response - pilot.baseline(train.time),
        weights,
        delta,
        tuning,
    )
    cursor = 0
    additives: list[SplineFunction] = []
    for index, basis in enumerate(additive_bases):
        next_cursor = cursor + basis.dimension
        additives.append(
            SplineFunction(
                basis, stage2.coefficients[cursor:next_cursor]
            ).centered_empirical(train.covariates[:, index])
        )
        cursor = next_cursor

    time_basis = SplineBasis.quantile(
        train.time,
        n_interior=coefficient_knots,
        order=order,
        domain=time_domain,
    )
    time_matrix = time_basis.transform(train.time)
    stage3_blocks = [time_matrix]
    for index, additive in enumerate(additives):
        stage3_blocks.append(
            additive(train.covariates[:, index])[:, None] * time_matrix
        )
    stage3 = _huber(
        np.column_stack(stage3_blocks), train.response, weights, delta, tuning
    )
    cursor = time_basis.dimension
    baseline = SplineFunction(time_basis, stage3.coefficients[:cursor])
    coefficients: list[SplineFunction] = []
    normalized_additives: list[SplineFunction] = []
    normalization_scales: list[float] = []
    for index in range(p):
        next_cursor = cursor + time_basis.dimension
        raw_coefficient = SplineFunction(
            time_basis, stage3.coefficients[cursor:next_cursor]
        )
        coefficient, additive, scale = _normalize_final_pair(
            raw_coefficient, additives[index]
        )
        coefficients.append(coefficient)
        normalized_additives.append(additive)
        normalization_scales.append(scale)
        cursor = next_cursor
    return _FinalFit(
        baseline=baseline,
        coefficients=tuple(coefficients),
        additives=tuple(normalized_additives),
        stage2=stage2,
        stage3=stage3,
        coefficient_knots=coefficient_knots,
        additive_knots=additive_knots,
        coefficient_normalization_scales=tuple(normalization_scales),
    )


def _predict_model(model: HHY2021Model, data: SubjectDataset) -> np.ndarray:
    prediction = model.baseline(data.time)
    for index, (coefficient, additive) in enumerate(
        zip(model.coefficients, model.additives, strict=True)
    ):
        prediction += coefficient(data.time) * additive(data.covariates[:, index])
    return np.asarray(prediction, dtype=float)


class HHY2021Adapter(BenchmarkAdapter):
    """Equations (2.3)--(2.7) with the paper's Huber threshold 1.345.

    The implementation retains the three distinct fits: a tensor-product
    pilot, an additive M-step with the pilot coefficient functions held fixed,
    and a varying-coefficient M-step with the additive functions held fixed.
    It does not turn the procedure into alternating backfitting.
    """

    label = MethodLabel.HHY2021_HUBER.value

    def preflight(self) -> PreflightReport:
        try:
            import scipy

            return PreflightReport(True, f"paper-three-stage/2021; scipy-{scipy.__version__}")
        except Exception as error:  # pragma: no cover
            return PreflightReport(False, "unavailable", "python_dependency_failure", str(error))

    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        del seed
        p = train.covariates.shape[1]
        order = int(tuning.get("spline_order", 4))
        delta = float(tuning.get("delta", 1.345))
        if delta != 1.345:
            raise ValueError("HHY2021-Huber strict mode fixes delta=1.345")
        time_domain, covariate_domains = _registered_domains(train, tuning)
        weights = subject_balanced_weights(train.subject_id)
        tuning_mode = str(tuning.get("tuning_mode", "paper_bic"))
        bic1_trace: list[dict[str, object]] = []
        bic2_trace: list[dict[str, object]] = []
        if tuning_mode == "paper_bic":
            candidates = tuple(
                int(item) for item in tuning.get("bic_knot_candidates", (1, 2, 3))
            )
            if not candidates or any(item < 0 for item in candidates):
                raise ValueError("bic_knot_candidates must be nonempty and nonnegative")
            candidates = tuple(sorted(set(candidates)))
            pilot_choices: list[tuple[float, _PilotFit]] = []
            for time_knots in candidates:
                for additive_knots in candidates:
                    try:
                        candidate = _pilot_fit(
                            train,
                            order=order,
                            delta=delta,
                            weights=weights,
                            time_domain=time_domain,
                            covariate_domains=covariate_domains,
                            time_knots=time_knots,
                            additive_knots=additive_knots,
                            tuning=tuning,
                        )
                        parameters = (order + time_knots) * (
                            1 + p * (order + additive_knots - 1)
                        )
                        objective = float(candidate.solve.objective[-1])
                        bic = (
                            _paper_bic(objective, parameters, train.n_rows)
                            if candidate.solve.converged
                            else float("inf")
                        )
                        pilot_choices.append((bic, candidate))
                        bic1_trace.append(
                            {
                                "pilot_time_interior_knots": time_knots,
                                "pilot_additive_interior_knots": additive_knots,
                                "objective": objective,
                                "bic": bic,
                                **_solve_audit(candidate.solve),
                            }
                        )
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                        bic1_trace.append(
                            {
                                "pilot_time_interior_knots": time_knots,
                                "pilot_additive_interior_knots": additive_knots,
                                "objective": None,
                                "bic": None,
                                "converged": False,
                                "failure": f"{type(error).__name__}: {error}",
                            }
                        )
            finite_pilots = [item for item in pilot_choices if np.isfinite(item[0])]
            if not finite_pilots:
                raise FloatingPointError("HHY2021 BIC1 found no converged finite candidate")
            _, pilot = min(finite_pilots, key=lambda item: item[0])

            final_choices: list[tuple[float, _FinalFit]] = []
            for coefficient_knots in candidates:
                for additive_knots in candidates:
                    try:
                        candidate = _final_fit(
                            train,
                            pilot,
                            order=order,
                            delta=delta,
                            weights=weights,
                            time_domain=time_domain,
                            covariate_domains=covariate_domains,
                            coefficient_knots=coefficient_knots,
                            additive_knots=additive_knots,
                            tuning=tuning,
                        )
                        fitted = candidate.baseline(train.time)
                        for index in range(p):
                            fitted += candidate.coefficients[index](train.time) * candidate.additives[
                                index
                            ](train.covariates[:, index])
                        objective = huber_objective(
                            train.response - fitted, weights, delta
                        )
                        parameters = p * (order + additive_knots - 1) + (
                            p + 1
                        ) * (order + coefficient_knots)
                        converged_candidate = bool(
                            candidate.stage2.converged and candidate.stage3.converged
                        )
                        bic = (
                            _paper_bic(objective, parameters, train.n_rows)
                            if converged_candidate
                            else float("inf")
                        )
                        final_choices.append((bic, candidate))
                        bic2_trace.append(
                            {
                                "final_coefficient_interior_knots": coefficient_knots,
                                "final_additive_interior_knots": additive_knots,
                                "objective": objective,
                                "bic": bic,
                                "converged": converged_candidate,
                                "stage2": _solve_audit(candidate.stage2),
                                "stage3": _solve_audit(candidate.stage3),
                            }
                        )
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                        bic2_trace.append(
                            {
                                "final_coefficient_interior_knots": coefficient_knots,
                                "final_additive_interior_knots": additive_knots,
                                "objective": None,
                                "bic": None,
                                "converged": False,
                                "failure": f"{type(error).__name__}: {error}",
                            }
                        )
            finite_finals = [item for item in final_choices if np.isfinite(item[0])]
            if not finite_finals:
                raise FloatingPointError("HHY2021 BIC2 found no converged finite candidate")
            _, final = min(finite_finals, key=lambda item: item[0])
        else:
            pilot_time_count = int(tuning.get("pilot_time_interior_knots", 2))
            pilot_covariate_counts = _expand_counts(
                tuning.get("pilot_covariate_interior_knots"), p, 2
            )
            final_time_count = int(tuning.get("final_time_interior_knots", 2))
            final_additive_counts = _expand_counts(
                tuning.get("final_additive_interior_knots"), p, 2
            )
            if len(set(pilot_covariate_counts)) != 1 or len(set(final_additive_counts)) != 1:
                raise ValueError("HHY2021 paper uses one common knot count per additive stage")
            pilot = _pilot_fit(
                train,
                order=order,
                delta=delta,
                weights=weights,
                time_domain=time_domain,
                covariate_domains=covariate_domains,
                time_knots=pilot_time_count,
                additive_knots=pilot_covariate_counts[0],
                tuning=tuning,
            )
            final = _final_fit(
                train,
                pilot,
                order=order,
                delta=delta,
                weights=weights,
                time_domain=time_domain,
                covariate_domains=covariate_domains,
                coefficient_knots=final_time_count,
                additive_knots=final_additive_counts[0],
                tuning=tuning,
            )

        stage1, stage2, stage3 = pilot.solve, final.stage2, final.stage3
        converged = bool(stage1.converged and stage2.converged and stage3.converged)
        model = HHY2021Model(
            baseline=final.baseline,
            coefficients=final.coefficients,
            additives=final.additives,
            pilot_anchors=pilot.anchors,
            converged=converged,
            stage_iterations=(stage1.iterations, stage2.iterations, stage3.iterations),
            stage_objectives=(stage1.objective, stage2.objective, stage3.objective),
            stage_termination_modes=(
                stage1.termination,
                stage2.termination,
                stage3.termination,
            ),
            stage_strict_coefficient_converged=(
                stage1.coefficient_converged,
                stage2.coefficient_converged,
                stage3.coefficient_converged,
            ),
            stage_objective_stable=(
                stage1.objective_stable,
                stage2.objective_stable,
                stage3.objective_stable,
            ),
            final_coefficient_normalization_scales=(
                final.coefficient_normalization_scales
            ),
        )
        recorded_tuning = dict(tuning)
        recorded_tuning.update(
            {
                "tuning_mode": tuning_mode,
                "spline_order": order,
                "delta": delta,
                "irls_tolerance": float(tuning.get("irls_tolerance", 1e-8)),
                "irls_max_iter": int(tuning.get("irls_max_iter", 300)),
                "irls_objective_relative_tolerance": (
                    None
                    if tuning.get("irls_objective_relative_tolerance", 1e-9) is None
                    else float(tuning.get("irls_objective_relative_tolerance", 1e-9))
                ),
                "irls_objective_stable_steps": int(
                    tuning.get("irls_objective_stable_steps", 3)
                ),
                "pilot_time_interior_knots": pilot.time_knots,
                "pilot_covariate_interior_knots": [pilot.additive_knots] * p,
                "final_time_interior_knots": final.coefficient_knots,
                "final_additive_interior_knots": [final.additive_knots] * p,
                "knot_placement": "empirical-quantile",
                "anchor_quantiles": list(
                    np.asarray(tuning.get("anchor_quantiles", (0.25, 0.5, 0.75)), dtype=float)
                ),
                "time_domain": list(time_domain),
                "covariate_domains": [list(item) for item in covariate_domains],
            }
        )
        return FitArtifact(
            model=model,
            method=self.label,
            version=self.preflight().version,
            tuning=recorded_tuning,
            converged=converged,
            metadata={
                "algorithm": "Hu-Huang-You (2021), equations (2.3)--(2.7)",
                "reproduction_mode": (
                    "paper BIC1/BIC2 formulas over the registered candidate grid"
                    if recorded_tuning["tuning_mode"] == "paper_bic"
                    else "explicit fixed-knot sensitivity"
                ),
                "loss": "subject-balanced Huber(delta=1.345)",
                "knot_placement": "equally spaced empirical quantiles",
                "bic_candidate_grid_note": (
                    "The paper gives the BIC formulas but not the searched integer grid; "
                    "the complete registered grid is stored in tuning metadata."
                ),
                "anchor_rule": (
                    "Among empirical x quantiles 0.25, 0.50 and 0.75, select the slice "
                    "with largest absolute fitted time integral; the paper requires only "
                    "a nonzero slice and does not specify an operational choice."
                ),
                "bic1_trace": bic1_trace,
                "bic2_trace": bic2_trace,
                "stage_iterations": list(model.stage_iterations),
                "stage_termination_modes": list(model.stage_termination_modes),
                "stage_strict_coefficient_converged": list(
                    model.stage_strict_coefficient_converged
                ),
                "stage_objective_stable": list(model.stage_objective_stable),
                "objective_stability_acceptance": {
                    "rule": (
                        "finite relative Huber-objective changes at or below the "
                        "registered tolerance for the registered consecutive tail; "
                        "strict coefficient-change convergence remains preferred"
                    ),
                    "relative_tolerance": recorded_tuning[
                        "irls_objective_relative_tolerance"
                    ],
                    "consecutive_changes": recorded_tuning[
                        "irls_objective_stable_steps"
                    ],
                },
                "pilot_anchor_covariates": list(model.pilot_anchors),
                "final_coefficient_normalization_scales": list(
                    model.final_coefficient_normalization_scales
                ),
                "normalization": (
                    "tensor pilot -> integral-one beta; empirical-mean-zero phi; "
                    "final integral-one beta with paired additive rescaling to preserve "
                    "the fitted surface"
                ),
            },
        )

    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> np.ndarray:
        return _predict_model(artifact.model, test)

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        model: HHY2021Model = artifact.model
        curves: list[dict[str, object]] = []
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
        for component, domain_name, function in functions:
            raw = function.basis.raw if isinstance(function.basis, CenteredSplineBasis) else function.basis
            grid = np.linspace(raw.lower, raw.upper, 201)
            curves.append(
                {
                    "component": component,
                    "domain": domain_name,
                    "grid": grid.tolist(),
                    "values": function(grid).tolist(),
                }
            )
        return tuple(curves)

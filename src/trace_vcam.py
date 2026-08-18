"""Core estimator for two-stage TRACE-VCAM.

The production estimator has exactly two stages.  It first solves the convex
subject-balanced Huber problem with block nuclear-norm and spline roughness
penalties.  It then keeps the leading singular-vector directions of the
active blocks fixed and jointly refits the baseline and one scalar amplitude
per block.  No additional shape update or inferential calculation is part of
the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh, null_space
from scipy.optimize import minimize
from sklearn.preprocessing import SplineTransformer


FloatArray = NDArray[np.float64]
__version__ = "2.0.0"

__all__ = [
    "OrthonormalSplineBasis",
    "VCAMDesign",
    "VCAMFit",
    "fit_trace_vcam",
    "huber_scores",
    "huber_values",
    "normalized_surface_roughness",
    "practical_huber_threshold",
    "predict_components",
    "recover_factors",
    "robust_scale",
    "trace_lambda_max",
]


def _symmetric_inv_sqrt(matrix: FloatArray, floor: float = 1e-10) -> FloatArray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, floor)
    return (vectors / np.sqrt(values)) @ vectors.T


def _normalized_psd_operator(matrix: FloatArray) -> FloatArray:
    """Return the PSD part of ``matrix`` with spectral norm at most one."""

    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, 0.0)
    maximum = float(values[-1]) if values.size else 0.0
    if maximum > 0.0:
        values = values / maximum
    return (vectors * values) @ vectors.T


@dataclass
class OrthonormalSplineBasis:
    """Lebesgue-orthonormal marginal spline bases on [0, 1]."""

    time_transformer: SplineTransformer
    covariate_transformer: SplineTransformer
    time_whitener: FloatArray
    covariate_projection: FloatArray
    covariate_whitener: FloatArray
    time_integral: FloatArray
    time_roughness: FloatArray
    covariate_roughness: FloatArray
    grid: FloatArray

    @classmethod
    def create(
        cls,
        time_dimension: int = 6,
        covariate_dimension: int = 6,
        degree: int = 3,
        grid_size: int = 2001,
    ) -> "OrthonormalSplineBasis":
        if time_dimension <= degree:
            raise ValueError("time_dimension must exceed the spline degree")
        if covariate_dimension <= degree:
            raise ValueError("covariate_dimension must exceed the spline degree")

        grid = np.linspace(0.0, 1.0, grid_size)
        x_grid = grid[:, None]
        time_knots = time_dimension - degree + 1
        cov_raw_dimension = covariate_dimension + 1
        cov_knots = cov_raw_dimension - degree + 1

        time_transformer = SplineTransformer(
            n_knots=time_knots,
            degree=degree,
            knots="uniform",
            include_bias=True,
            extrapolation="constant",
        ).fit(x_grid)
        covariate_transformer = SplineTransformer(
            n_knots=cov_knots,
            degree=degree,
            knots="uniform",
            include_bias=True,
            extrapolation="constant",
        ).fit(x_grid)

        time_raw = time_transformer.transform(x_grid)
        cov_raw = covariate_transformer.transform(x_grid)
        time_gram = np.trapezoid(
            time_raw[:, :, None] * time_raw[:, None, :], grid, axis=0
        )
        cov_mean = np.trapezoid(cov_raw, grid, axis=0)
        cov_projection = null_space(cov_mean[None, :])
        cov_centered = cov_raw @ cov_projection
        cov_gram = np.trapezoid(
            cov_centered[:, :, None] * cov_centered[:, None, :], grid, axis=0
        )
        time_whitener = _symmetric_inv_sqrt(time_gram)
        covariate_whitener = _symmetric_inv_sqrt(cov_gram)
        time_basis_grid = time_raw @ time_whitener
        time_integral = np.trapezoid(time_basis_grid, grid, axis=0)
        # Analytic B-spline derivatives avoid a boundary-sensitive finite-
        # difference approximation.  Operator normalization makes one common
        # mu grid meaningful for both marginal bases while retaining the
        # affine null spaces of the integrated second-derivative operators.
        time_raw_second = time_transformer.bsplines_[0].derivative(2)(grid)
        covariate_raw_second = covariate_transformer.bsplines_[0].derivative(2)(
            grid
        )
        time_second = time_raw_second @ time_whitener
        covariate_second = (
            covariate_raw_second @ cov_projection @ covariate_whitener
        )
        time_roughness = _normalized_psd_operator(
            np.trapezoid(
                time_second[:, :, None] * time_second[:, None, :],
                grid,
                axis=0,
            )
        )
        covariate_roughness = _normalized_psd_operator(
            np.trapezoid(
                covariate_second[:, :, None] * covariate_second[:, None, :],
                grid,
                axis=0,
            )
        )
        return cls(
            time_transformer=time_transformer,
            covariate_transformer=covariate_transformer,
            time_whitener=time_whitener,
            covariate_projection=cov_projection,
            covariate_whitener=covariate_whitener,
            time_integral=time_integral,
            time_roughness=time_roughness,
            covariate_roughness=covariate_roughness,
            grid=grid,
        )

    @property
    def time_dimension(self) -> int:
        return int(self.time_whitener.shape[0])

    @property
    def covariate_dimension(self) -> int:
        return int(self.covariate_whitener.shape[0])

    def transform_time(self, values: ArrayLike) -> FloatArray:
        x = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.time_transformer.transform(x) @ self.time_whitener

    def transform_covariate(self, values: ArrayLike) -> FloatArray:
        x = np.asarray(values, dtype=float).reshape(-1, 1)
        raw = self.covariate_transformer.transform(x)
        return raw @ self.covariate_projection @ self.covariate_whitener


@dataclass
class VCAMDesign:
    time: FloatArray
    covariates: list[FloatArray]
    response: FloatArray
    subject: NDArray[np.int64]
    weights: FloatArray
    time_integral: FloatArray
    time_roughness: FloatArray
    covariate_roughness: FloatArray
    lipschitz: float | None = None

    @classmethod
    def from_arrays(
        cls,
        time: ArrayLike,
        covariates: ArrayLike,
        response: ArrayLike,
        subject: ArrayLike,
        basis: OrthonormalSplineBasis,
    ) -> "VCAMDesign":
        time_values = np.asarray(time, dtype=float).reshape(-1)
        z_values = np.asarray(covariates, dtype=float)
        if z_values.ndim == 1:
            z_values = z_values[:, None]
        y_values = np.asarray(response, dtype=float).reshape(-1)
        subjects = np.asarray(subject, dtype=np.int64).reshape(-1)
        sample_size = len(time_values)
        if sample_size == 0:
            raise ValueError("at least one observation is required")
        if z_values.ndim != 2 or z_values.shape[0] != sample_size:
            raise ValueError("covariates must have one row per observation")
        if z_values.shape[1] == 0:
            raise ValueError("at least one covariate is required")
        if len(y_values) != sample_size or len(subjects) != sample_size:
            raise ValueError("time, response, and subject must have equal length")
        if not (
            np.all(np.isfinite(time_values))
            and np.all(np.isfinite(z_values))
            and np.all(np.isfinite(y_values))
        ):
            raise ValueError("time, covariates, and response must be finite")
        unique_subjects, inverse, counts = np.unique(
            subjects, return_inverse=True, return_counts=True
        )
        del unique_subjects
        weights = 1.0 / (len(counts) * counts[inverse])
        return cls(
            time=basis.transform_time(time_values),
            covariates=[basis.transform_covariate(z_values[:, k]) for k in range(z_values.shape[1])],
            response=y_values,
            subject=subjects,
            weights=weights,
            time_integral=basis.time_integral.copy(),
            time_roughness=basis.time_roughness.copy(),
            covariate_roughness=basis.covariate_roughness.copy(),
        )

    def subset(self, subjects: Iterable[int]) -> "VCAMDesign":
        subject_set = np.asarray(list(subjects), dtype=np.int64)
        mask = np.isin(self.subject, subject_set)
        if not np.any(mask):
            raise ValueError("subject subset must contain at least one observation")
        selected_subjects = self.subject[mask]
        unique_subjects, inverse, counts = np.unique(
            selected_subjects, return_inverse=True, return_counts=True
        )
        del unique_subjects
        weights = 1.0 / (len(counts) * counts[inverse])
        return VCAMDesign(
            time=self.time[mask],
            covariates=[item[mask] for item in self.covariates],
            response=self.response[mask],
            subject=selected_subjects,
            weights=weights,
            time_integral=self.time_integral,
            time_roughness=self.time_roughness,
            covariate_roughness=self.covariate_roughness,
        )


def huber_values(residual: FloatArray, delta: float) -> FloatArray:
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta,
        0.5 * residual**2,
        delta * absolute - 0.5 * delta**2,
    )


def huber_scores(residual: FloatArray, delta: float) -> FloatArray:
    return np.clip(residual, -delta, delta)


def _weighted_median(values: FloatArray, weights: FloatArray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def robust_scale(
    values: ArrayLike,
    weights: ArrayLike | None = None,
    floor: float = 1e-3,
) -> float:
    values_array = np.asarray(values, dtype=float)
    if weights is None:
        median = float(np.median(values_array))
        mad_raw = float(np.median(np.abs(values_array - median)))
    else:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != values_array.shape or np.any(weights_array < 0):
            raise ValueError("weights must be nonnegative and match values")
        if not np.sum(weights_array) > 0:
            raise ValueError("weights must have positive total mass")
        median = _weighted_median(values_array, weights_array)
        mad_raw = _weighted_median(np.abs(values_array - median), weights_array)
    return float(max(1.4826 * mad_raw, floor))


def practical_huber_threshold(
    design: "VCAMDesign",
    multiplier: float = 1.345,
) -> tuple[float, float]:
    """Return a practical MAD threshold and the estimated residual scale.

    This data-adaptive rule is deliberately labeled *practical*.  The fixed-
    threshold asymptotic results do not automatically cover it.
    """

    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("multiplier must be finite and positive")
    square_root_weights = np.sqrt(design.weights)
    baseline = np.linalg.lstsq(
        square_root_weights[:, None] * design.time,
        square_root_weights * design.response,
        rcond=None,
    )[0]
    residual = design.response - design.time @ baseline
    scale = robust_scale(residual, design.weights)
    return float(multiplier * scale), float(scale)


def predict_components(
    design: VCAMDesign,
    gamma: FloatArray,
    matrices: Sequence[FloatArray],
) -> tuple[FloatArray, list[FloatArray]]:
    component_predictions = [
        np.einsum("ni,ij,nj->n", design.time, theta, covariate)
        for theta, covariate in zip(matrices, design.covariates, strict=True)
    ]
    prediction = design.time @ gamma
    if component_predictions:
        prediction = prediction + np.sum(component_predictions, axis=0)
    return prediction, component_predictions


@dataclass(frozen=True)
class _HuberLinearResult:
    coefficients: FloatArray
    objective: list[float]
    converged: bool
    iterations: int
    kkt_residual: float


def _solve_huber_linear(
    features: FloatArray,
    response: FloatArray,
    base_weights: FloatArray,
    delta: float,
    max_iter: int = 80,
    tolerance: float = 1e-8,
    initial: FloatArray | None = None,
) -> _HuberLinearResult:
    """Solve a finite-dimensional weighted Huber problem.

    The reported KKT residual is for the exact unpenalized objective.  This
    routine is used by the scalar postfit so its convergence status does not
    depend on an IRLS step-size heuristic.
    """

    features = np.asarray(features, dtype=float)
    response = np.asarray(response, dtype=float).reshape(-1)
    base_weights = np.asarray(base_weights, dtype=float).reshape(-1)
    if features.ndim != 2 or features.shape[0] != len(response):
        raise ValueError("features must have one row per response")
    if base_weights.shape != response.shape or np.any(base_weights < 0.0):
        raise ValueError("base_weights must be nonnegative and match response")
    if not np.sum(base_weights) > 0.0:
        raise ValueError("base_weights must have positive total mass")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    dimension = features.shape[1]

    if initial is None:
        square_root_weights = np.sqrt(base_weights)
        weighted_features = square_root_weights[:, None] * features
        weighted_response = square_root_weights * response
        coefficients = np.linalg.lstsq(
            weighted_features, weighted_response, rcond=None
        )[0]
    else:
        coefficients = np.asarray(initial, dtype=float).reshape(-1).copy()
        if coefficients.shape != (dimension,):
            raise ValueError("initial coefficients have the wrong dimension")

    def objective_and_gradient(value: FloatArray) -> tuple[float, FloatArray]:
        residual = response - features @ value
        objective = float(
            np.dot(base_weights, huber_values(residual, delta))
        )
        gradient = -(
            features.T @ (base_weights * huber_scores(residual, delta))
        )
        return objective, gradient

    objective_history = [objective_and_gradient(coefficients)[0]]

    def callback(value: FloatArray) -> None:
        objective_history.append(objective_and_gradient(value)[0])

    result = minimize(
        objective_and_gradient,
        coefficients,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": int(max_iter),
            "ftol": min(float(tolerance), 1e-12),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    coefficients = np.asarray(result.x, dtype=float)
    final_objective, final_gradient = objective_and_gradient(coefficients)
    if not objective_history or not np.isclose(
        objective_history[-1], final_objective, rtol=0.0, atol=1e-15
    ):
        objective_history.append(final_objective)
    scale = max(1.0, np.linalg.norm(coefficients))
    kkt_residual = float(np.linalg.norm(final_gradient) / scale)
    converged = bool(
        np.all(np.isfinite(coefficients))
        and np.isfinite(final_objective)
        and kkt_residual <= max(10.0 * tolerance, 1e-6)
    )
    return _HuberLinearResult(
        coefficients=coefficients,
        objective=objective_history,
        converged=converged,
        iterations=int(result.nit),
        kkt_residual=kkt_residual,
    )


def _soft_threshold_singular_values(matrix: FloatArray, threshold: float) -> FloatArray:
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    retained = np.maximum(singular - threshold, 0.0)
    if not np.any(retained > 0.0):
        return np.zeros_like(matrix)
    return (left * retained) @ right_t


@dataclass(frozen=True)
class _ScalarPostfitResult:
    gamma: FloatArray
    matrices: list[FloatArray]
    amplitudes: FloatArray
    time_factors: list[FloatArray | None]
    covariate_factors: list[FloatArray | None]
    objective: list[float]
    converged: bool
    iterations: int
    kkt_residual: float
    anchor_huber_loss: float
    final_huber_loss: float


def _scalar_postfit(
    design: VCAMDesign,
    gamma: FloatArray,
    matrices: list[FloatArray],
    selected: NDArray[np.bool_],
    delta: float,
    identification_tolerance: float = 1e-8,
    max_iter: int = 500,
    tolerance: float = 1e-8,
) -> _ScalarPostfitResult:
    """Jointly refit the baseline and fixed rank-one amplitudes.

    For every selected block, the leading left singular vector is normalized
    to have Lebesgue integral one.  The inverse scale is absorbed into the
    covariate factor, so the resulting outer-product direction is unchanged.
    These identified directions are held fixed throughout the finite-
    dimensional Huber solve.  Only the baseline coefficients and one scalar
    amplitude per active block are optimized.
    """

    active = np.flatnonzero(selected)
    if selected.shape != (len(matrices),):
        raise ValueError("selected must have one entry per coefficient block")
    q_time = design.time.shape[1]
    directions: dict[int, FloatArray] = {}
    time_factors: list[FloatArray | None] = [None] * len(matrices)
    covariate_directions: list[FloatArray | None] = [None] * len(matrices)
    amplitude_features: list[FloatArray] = []
    initial_amplitudes: list[float] = []
    for index in active:
        left, singular, right_t = np.linalg.svd(matrices[index], full_matrices=False)
        if singular[0] <= identification_tolerance:
            raise ValueError(f"selected block {index} has zero leading singular value")
        left_vector = left[:, 0].copy()
        right_vector = right_t[0].copy()
        integral = float(design.time_integral @ left_vector)
        if abs(integral) <= identification_tolerance:
            raise ValueError(
                f"selected block {index} violates the integral identification condition"
            )
        if integral < 0.0:
            left_vector *= -1.0
            right_vector *= -1.0
            integral *= -1.0
        beta = left_vector / integral
        phi_direction = integral * right_vector
        direction = np.outer(beta, phi_direction)
        directions[index] = direction
        time_factors[index] = beta
        covariate_directions[index] = phi_direction
        initial_amplitudes.append(float(singular[0]))
        amplitude_features.append(
            np.einsum(
                "ni,ij,nj->n",
                design.time,
                direction,
                design.covariates[index],
            )
        )
    postfit_design = np.column_stack([design.time, *amplitude_features])
    initial = np.concatenate(
        [np.asarray(gamma, dtype=float), np.asarray(initial_amplitudes, dtype=float)]
    )
    result = _solve_huber_linear(
        postfit_design,
        design.response,
        design.weights,
        delta,
        max_iter=max_iter,
        tolerance=tolerance,
        initial=initial,
    )
    solution = result.coefficients
    calibrated = [np.zeros_like(matrix) for matrix in matrices]
    amplitudes = np.zeros(len(matrices), dtype=float)
    covariate_factors: list[FloatArray | None] = [None] * len(matrices)
    for position, index in enumerate(active):
        amplitude = float(solution[q_time + position])
        amplitudes[index] = amplitude
        calibrated[index] = amplitude * directions[index]
        phi_direction = covariate_directions[index]
        assert phi_direction is not None
        covariate_factors[index] = amplitude * phi_direction
    anchor_loss = float(
        np.dot(
            design.weights,
            huber_values(design.response - postfit_design @ initial, delta),
        )
    )
    final_loss = float(
        np.dot(
            design.weights,
            huber_values(design.response - postfit_design @ solution, delta),
        )
    )
    return _ScalarPostfitResult(
        gamma=solution[:q_time],
        matrices=calibrated,
        amplitudes=amplitudes,
        time_factors=time_factors,
        covariate_factors=covariate_factors,
        objective=result.objective,
        converged=result.converged,
        iterations=result.iterations,
        kkt_residual=result.kkt_residual,
        anchor_huber_loss=anchor_loss,
        final_huber_loss=final_loss,
    )


def _nuclear_norm(matrix: FloatArray) -> float:
    return float(np.sum(np.linalg.svd(matrix, compute_uv=False)))


def _dense_tensor_design(design: VCAMDesign) -> FloatArray:
    blocks = [
        np.einsum("ni,nj->nij", design.time, covariate).reshape(
            design.time.shape[0], -1
        )
        for covariate in design.covariates
    ]
    return np.column_stack([design.time, *blocks])


def _lipschitz_bound(design: VCAMDesign) -> float:
    """Return and cache a true spectral Lipschitz bound for the Huber gradient."""

    if design.lipschitz is not None:
        return design.lipschitz
    features = _dense_tensor_design(design)
    weighted = np.sqrt(design.weights)[:, None] * features
    gram = weighted.T @ weighted
    dimension = gram.shape[0]
    eigenvalue = float(
        eigh(
            gram,
            subset_by_index=[dimension - 1, dimension - 1],
            eigvals_only=True,
            check_finite=False,
        )[0]
    )
    # Huber's derivative is at most one, so X'WX dominates every Hessian.
    design.lipschitz = max(1.001 * eigenvalue, 1e-8)
    return design.lipschitz


@dataclass
class VCAMFit:
    method: str
    gamma: FloatArray
    matrices: list[FloatArray]
    selected: NDArray[np.bool_]
    objective: list[float]
    converged: bool
    iterations: int
    runtime_seconds: float
    delta: float | None = None
    penalty: float | None = None
    block_weights: FloatArray | None = None
    mu: float | None = None
    convex_converged: bool | None = None
    convex_kkt_residual: float | None = None
    preprojection_rank1_energy: float | None = None
    scalar_postfit_attempted: bool = False
    scalar_postfit_converged: bool | None = None
    scalar_postfit_iterations: int | None = None
    scalar_postfit_kkt_residual: float | None = None
    scalar_postfit_objective: list[float] | None = None
    scalar_amplitudes: FloatArray | None = None
    identified_time_factors: list[FloatArray | None] | None = None
    identified_covariate_factors: list[FloatArray | None] | None = None
    rank_one_anchor_huber_loss: float | None = None
    postfit_huber_loss: float | None = None
    huber_threshold_mode: str | None = None
    huber_scale: float | None = None
    huber_multiplier: float | None = None
    tuning_regime: str | None = None
    penalty_weights_source: str | None = None
    selection_tolerance: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def predict(self, design: VCAMDesign) -> FloatArray:
        return predict_components(design, self.gamma, self.matrices)[0]


def trace_lambda_max(
    design: VCAMDesign,
    delta: float,
    block_weights: ArrayLike | None = None,
) -> tuple[float, FloatArray]:
    gamma = _solve_huber_linear(
        design.time,
        design.response,
        design.weights,
        delta,
    ).coefficients
    residual = design.response - design.time @ gamma
    score = huber_scores(residual, delta)
    weights = (
        np.ones(len(design.covariates), dtype=float)
        if block_weights is None
        else np.asarray(block_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (len(design.covariates),) or np.any(
        ~np.isfinite(weights)
    ) or np.any(weights <= 0.0):
        raise ValueError("block_weights must be finite, positive, and block-aligned")
    gradient_norms = []
    for covariate, block_weight in zip(design.covariates, weights, strict=True):
        gradient = design.time.T @ (
            (design.weights * score)[:, None] * covariate
        )
        gradient_norms.append(
            np.linalg.svd(gradient, compute_uv=False)[0] / block_weight
        )
    return float(max(gradient_norms, default=0.0)), gamma


def normalized_surface_roughness(
    matrices: Sequence[FloatArray],
    time_roughness: FloatArray,
    covariate_roughness: FloatArray,
) -> float:
    """Evaluate the normalized tensor-product squared roughness."""

    value = 0.0
    for matrix in matrices:
        value += float(np.sum(matrix * (time_roughness @ matrix)))
        value += float(np.sum(matrix * (matrix @ covariate_roughness)))
    return float(max(value, 0.0))


def _huber_data_loss(
    design: VCAMDesign,
    gamma: FloatArray,
    matrices: Sequence[FloatArray],
    delta: float,
) -> float:
    prediction, _ = predict_components(design, gamma, matrices)
    return float(
        np.dot(design.weights, huber_values(design.response - prediction, delta))
    )


@dataclass
class _ConvexTraceResult:
    gamma: FloatArray
    matrices: list[FloatArray]
    objective: list[float]
    converged: bool
    iterations: int
    kkt_residual: float


def _trace_gradient(
    design: VCAMDesign,
    gamma: FloatArray,
    matrices: Sequence[FloatArray],
    delta: float,
    mu: float,
) -> tuple[FloatArray, list[FloatArray]]:
    prediction, _ = predict_components(design, gamma, matrices)
    weighted_score = design.weights * huber_scores(
        design.response - prediction, delta
    )
    gamma_gradient = -(design.time.T @ weighted_score)
    matrix_gradients = [
        -(design.time.T @ (weighted_score[:, None] * covariate))
        + mu
        * (
            design.time_roughness @ matrix
            + matrix @ design.covariate_roughness
        )
        for matrix, covariate in zip(
            matrices, design.covariates, strict=True
        )
    ]
    return gamma_gradient, matrix_gradients


def _trace_convex_objective(
    design: VCAMDesign,
    gamma: FloatArray,
    matrices: Sequence[FloatArray],
    delta: float,
    penalty: float,
    nuclear_weights: FloatArray,
    mu: float,
) -> float:
    return (
        _huber_data_loss(design, gamma, matrices, delta)
        + penalty
        * sum(
            weight * _nuclear_norm(matrix)
            for weight, matrix in zip(
                nuclear_weights, matrices, strict=True
            )
        )
        + 0.5
        * mu
        * normalized_surface_roughness(
            matrices,
            design.time_roughness,
            design.covariate_roughness,
        )
    )


def _trace_proximal_mapping_residual(
    design: VCAMDesign,
    gamma: FloatArray,
    matrices: list[FloatArray],
    delta: float,
    penalty: float,
    nuclear_weights: FloatArray,
    mu: float,
    step: float,
) -> float:
    gamma_gradient, matrix_gradients = _trace_gradient(
        design, gamma, matrices, delta, mu
    )
    checked_gamma = gamma - step * gamma_gradient
    checked_matrices = [
        _soft_threshold_singular_values(
            matrix - step * gradient,
            step * penalty * weight,
        )
        for matrix, gradient, weight in zip(
            matrices, matrix_gradients, nuclear_weights, strict=True
        )
    ]
    mapping_squared = float(np.sum((gamma - checked_gamma) ** 2))
    parameter_squared = float(np.sum(gamma**2))
    for matrix, checked in zip(matrices, checked_matrices, strict=True):
        mapping_squared += float(np.sum((matrix - checked) ** 2))
        parameter_squared += float(np.sum(matrix**2))
    return float(
        np.sqrt(mapping_squared)
        / step
        / max(1.0, np.sqrt(parameter_squared))
    )


def _fit_trace_convex(
    design: VCAMDesign,
    penalty: float,
    delta: float,
    mu: float,
    nuclear_weights: FloatArray,
    max_iter: int,
    tolerance: float,
    initial: VCAMFit | None,
) -> _ConvexTraceResult:
    """Solve only the convex TRACE stage; no rank projection or postfit."""

    q_time = design.time.shape[1]
    q_covariate = design.covariates[0].shape[1]
    p = len(design.covariates)
    if initial is None:
        _, gamma = trace_lambda_max(design, delta, nuclear_weights)
        matrices = [np.zeros((q_time, q_covariate)) for _ in range(p)]
    else:
        gamma = np.asarray(initial.gamma, dtype=float).copy()
        matrices = [np.asarray(matrix, dtype=float).copy() for matrix in initial.matrices]
        if gamma.shape != (q_time,) or len(matrices) != p or any(
            matrix.shape != (q_time, q_covariate) for matrix in matrices
        ):
            raise ValueError("initial fit has incompatible dimensions")

    data_lipschitz = _lipschitz_bound(design)
    roughness_lipschitz = mu * (
        float(np.linalg.eigvalsh(design.time_roughness)[-1])
        + float(np.linalg.eigvalsh(design.covariate_roughness)[-1])
    )
    step = 1.0 / max(data_lipschitz + roughness_lipschitz, 1e-8)
    accelerated_gamma = gamma.copy()
    accelerated_matrices = [matrix.copy() for matrix in matrices]
    momentum = 1.0
    objective_history = [
        _trace_convex_objective(
            design,
            gamma,
            matrices,
            delta,
            penalty,
            nuclear_weights,
            mu,
        )
    ]
    previous_objective = objective_history[-1]
    converged = False
    kkt_residual = np.inf
    completed_iterations = 0

    for iteration in range(1, max_iter + 1):
        completed_iterations = iteration
        gamma_gradient, matrix_gradients = _trace_gradient(
            design,
            accelerated_gamma,
            accelerated_matrices,
            delta,
            mu,
        )
        candidate_gamma = accelerated_gamma - step * gamma_gradient
        candidate_matrices = [
            _soft_threshold_singular_values(
                matrix - step * gradient,
                step * penalty * weight,
            )
            for matrix, gradient, weight in zip(
                accelerated_matrices,
                matrix_gradients,
                nuclear_weights,
                strict=True,
            )
        ]
        candidate_objective = _trace_convex_objective(
            design,
            candidate_gamma,
            candidate_matrices,
            delta,
            penalty,
            nuclear_weights,
            mu,
        )

        if candidate_objective > previous_objective + 1e-10:
            was_accelerated = momentum > 1.0 + 1e-12
            accelerated_gamma = gamma.copy()
            accelerated_matrices = [matrix.copy() for matrix in matrices]
            momentum = 1.0
            if not was_accelerated:
                step *= 0.5
            continue

        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        factor = (momentum - 1.0) / next_momentum
        next_accelerated_gamma = candidate_gamma + factor * (
            candidate_gamma - gamma
        )
        next_accelerated_matrices = [
            candidate + factor * (candidate - old)
            for candidate, old in zip(
                candidate_matrices, matrices, strict=True
            )
        ]
        gamma = candidate_gamma
        matrices = candidate_matrices
        accelerated_gamma = next_accelerated_gamma
        accelerated_matrices = next_accelerated_matrices
        momentum = next_momentum
        objective_history.append(candidate_objective)
        relative_change = abs(previous_objective - candidate_objective) / (
            1.0 + abs(previous_objective)
        )
        previous_objective = candidate_objective

        if relative_change < tolerance or iteration % 10 == 0:
            kkt_residual = _trace_proximal_mapping_residual(
                design,
                gamma,
                matrices,
                delta,
                penalty,
                nuclear_weights,
                mu,
                step,
            )
            if np.isfinite(kkt_residual) and kkt_residual < max(tolerance, 1e-7):
                converged = True
                break

    if not np.isfinite(kkt_residual):
        kkt_residual = _trace_proximal_mapping_residual(
            design,
            gamma,
            matrices,
            delta,
            penalty,
            nuclear_weights,
            mu,
            step,
        )
    return _ConvexTraceResult(
        gamma=gamma,
        matrices=matrices,
        objective=objective_history,
        converged=converged,
        iterations=completed_iterations,
        kkt_residual=kkt_residual,
    )


def _rank_one_projection(
    matrices: Sequence[FloatArray],
    selection_tolerance: float,
) -> tuple[list[FloatArray], NDArray[np.bool_], float]:
    full_singular_values = [
        np.linalg.svd(matrix, compute_uv=False) for matrix in matrices
    ]
    selected = np.asarray(
        [values[0] > selection_tolerance for values in full_singular_values],
        dtype=bool,
    )
    projected: list[FloatArray] = []
    energies: list[float] = []
    for matrix, values, active in zip(
        matrices, full_singular_values, selected, strict=True
    ):
        if not active:
            projected.append(np.zeros_like(matrix))
            continue
        left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
        projected.append(singular[0] * np.outer(left[:, 0], right_t[0]))
        denominator = float(np.sum(values**2))
        if denominator > 0.0:
            energies.append(float(values[0] ** 2 / denominator))
    energy = float(np.mean(energies)) if energies else np.nan
    return projected, selected, energy


def fit_trace_vcam(
    design: VCAMDesign,
    penalty: float,
    delta: float | None = None,
    max_iter: int = 500,
    tolerance: float = 1e-7,
    initial: VCAMFit | None = None,
    post_rank_one: bool = True,
    post_refit: bool = True,
    selection_tolerance: float = 1e-9,
    block_weights: ArrayLike | None = None,
    mu: float = 0.0,
    *,
    threshold_mode: str | None = None,
    huber_multiplier: float = 1.345,
    identification_tolerance: float = 1e-8,
    postfit_max_iter: int = 500,
    postfit_tolerance: float = 1e-8,
) -> VCAMFit:
    """Fit the two-stage TRACE-VCAM estimator.

    Stage 1 is the convex subject-balanced Huber, block nuclear-norm, and
    roughness-penalized estimator.  Stage 2 projects each numerically active
    block onto its leading singular direction, imposes the integral-one
    identification, and jointly refits the baseline plus one scalar amplitude
    per active block.  The convex-stage objective history remains available in
    ``objective``; the low-dimensional Huber history is stored separately in
    ``scalar_postfit_objective``.

    ``threshold_mode='fixed'`` records a user-supplied threshold and is the
    regime addressed by the fixed-threshold theory.  With
    ``threshold_mode='mad'`` (or ``delta=None``), the threshold is estimated
    from a weighted baseline residual MAD and is explicitly labeled as a
    practical tuning rule in the returned metadata.

    The default block penalty is unweighted.  Passing ``block_weights`` is
    allowed only as an externally fixed choice; this routine never estimates
    adaptive weights from the same data.
    """

    start = perf_counter()
    if penalty < 0.0 or not np.isfinite(penalty):
        raise ValueError("penalty must be finite and nonnegative")
    if mu < 0.0 or not np.isfinite(mu):
        raise ValueError("mu must be finite and nonnegative")
    if max_iter <= 0 or postfit_max_iter <= 0:
        raise ValueError("iteration limits must be positive")
    if tolerance <= 0.0 or postfit_tolerance <= 0.0:
        raise ValueError("tolerances must be positive")
    if selection_tolerance < 0.0 or identification_tolerance <= 0.0:
        raise ValueError("selection and identification tolerances are invalid")
    if post_refit and not post_rank_one:
        raise ValueError("post_refit requires post_rank_one=True")

    if threshold_mode is None:
        threshold_mode = "fixed" if delta is not None else "mad"
    if threshold_mode not in {"fixed", "mad"}:
        raise ValueError("threshold_mode must be 'fixed' or 'mad'")
    huber_scale: float | None
    if threshold_mode == "fixed":
        if delta is None or not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("a finite positive delta is required in fixed mode")
        resolved_delta = float(delta)
        huber_scale = None
        tuning_regime = "theory-fixed-threshold"
        recorded_multiplier: float | None = None
    else:
        if delta is not None:
            raise ValueError("delta must be omitted when threshold_mode='mad'")
        resolved_delta, huber_scale = practical_huber_threshold(
            design, multiplier=huber_multiplier
        )
        tuning_regime = "practical-data-adaptive-mad"
        recorded_multiplier = float(huber_multiplier)

    p = len(design.covariates)
    if block_weights is None:
        nuclear_weights = np.ones(p, dtype=float)
        penalty_weights_source = "unit"
    else:
        nuclear_weights = (
            np.asarray(block_weights, dtype=float).reshape(-1).copy()
        )
        penalty_weights_source = "user_fixed"
    if nuclear_weights.shape != (p,) or np.any(~np.isfinite(nuclear_weights)):
        raise ValueError("block_weights must be finite with one entry per block")
    if np.any(nuclear_weights <= 0.0):
        raise ValueError("block_weights must be strictly positive")

    convex = _fit_trace_convex(
        design=design,
        penalty=float(penalty),
        delta=resolved_delta,
        mu=float(mu),
        nuclear_weights=nuclear_weights,
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        initial=initial,
    )
    rank_one_matrices, selected, rank_one_energy = _rank_one_projection(
        convex.matrices, selection_tolerance
    )

    gamma = convex.gamma.copy()
    matrices = [matrix.copy() for matrix in convex.matrices]
    scalar_result: _ScalarPostfitResult | None = None
    if post_rank_one:
        matrices = [matrix.copy() for matrix in rank_one_matrices]
    if post_rank_one and post_refit:
        scalar_result = _scalar_postfit(
            design=design,
            gamma=convex.gamma,
            matrices=rank_one_matrices,
            selected=selected,
            delta=resolved_delta,
            identification_tolerance=identification_tolerance,
            max_iter=postfit_max_iter,
            tolerance=postfit_tolerance,
        )
        gamma = scalar_result.gamma
        matrices = scalar_result.matrices

    scalar_converged = (
        scalar_result.converged if scalar_result is not None else None
    )
    overall_converged = bool(
        convex.converged
        and (scalar_result is None or scalar_result.converged)
        and np.all(np.isfinite(gamma))
        and all(np.all(np.isfinite(matrix)) for matrix in matrices)
    )
    metadata: dict[str, object] = {
        "estimator_version": "trace-vcam-two-stage-v1",
        "stage_one": "convex-huber-nuclear-roughness",
        "stage_two": (
            "fixed-direction-scalar-huber"
            if post_rank_one and post_refit
            else "disabled-diagnostic"
        ),
        "huber_threshold_mode": threshold_mode,
        "tuning_regime": tuning_regime,
        "fixed_threshold_theory_applies": threshold_mode == "fixed",
        "penalty_weights_source": penalty_weights_source,
        "selection_interpretation": "numerical-zero-block-behavior",
    }

    return VCAMFit(
        method="TRACE-VCAM",
        gamma=gamma,
        matrices=matrices,
        selected=selected,
        objective=convex.objective,
        converged=overall_converged,
        iterations=convex.iterations,
        runtime_seconds=perf_counter() - start,
        delta=resolved_delta,
        penalty=float(penalty),
        block_weights=nuclear_weights,
        mu=float(mu),
        convex_converged=convex.converged,
        convex_kkt_residual=convex.kkt_residual,
        preprojection_rank1_energy=rank_one_energy,
        scalar_postfit_attempted=bool(post_rank_one and post_refit),
        scalar_postfit_converged=scalar_converged,
        scalar_postfit_iterations=(
            scalar_result.iterations if scalar_result is not None else None
        ),
        scalar_postfit_kkt_residual=(
            scalar_result.kkt_residual if scalar_result is not None else None
        ),
        scalar_postfit_objective=(
            scalar_result.objective if scalar_result is not None else None
        ),
        scalar_amplitudes=(
            scalar_result.amplitudes if scalar_result is not None else None
        ),
        identified_time_factors=(
            scalar_result.time_factors if scalar_result is not None else None
        ),
        identified_covariate_factors=(
            scalar_result.covariate_factors if scalar_result is not None else None
        ),
        rank_one_anchor_huber_loss=(
            scalar_result.anchor_huber_loss if scalar_result is not None else None
        ),
        postfit_huber_loss=(
            scalar_result.final_huber_loss if scalar_result is not None else None
        ),
        huber_threshold_mode=threshold_mode,
        huber_scale=huber_scale,
        huber_multiplier=recorded_multiplier,
        tuning_regime=tuning_regime,
        penalty_weights_source=penalty_weights_source,
        selection_tolerance=float(selection_tolerance),
        metadata=metadata,
    )


def recover_factors(
    matrix: FloatArray,
    basis: OrthonormalSplineBasis,
    integral_floor: float = 1e-8,
) -> tuple[FloatArray, FloatArray] | None:
    """Return coefficient vectors for beta and phi under integral(beta)=1."""

    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] <= integral_floor:
        return None
    time_coefficients = np.sqrt(singular[0]) * left[:, 0]
    covariate_coefficients = np.sqrt(singular[0]) * right_t[0]
    integral = float(basis.time_integral @ time_coefficients)
    if abs(integral) <= integral_floor:
        return None
    beta = time_coefficients / integral
    phi = integral * covariate_coefficients
    return beta, phi

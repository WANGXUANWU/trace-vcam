"""Small, explicit spline/robust-regression primitives for paper adapters."""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class SplineBasis:
    lower: float
    upper: float
    degree: int
    knots: np.ndarray

    @classmethod
    def equidistant(
        cls,
        values: np.ndarray,
        *,
        n_interior: int,
        order: int = 4,
        domain: tuple[float, float] | None = None,
    ) -> "SplineBasis":
        values = np.asarray(values, dtype=float)
        if order < 1 or n_interior < 0:
            raise ValueError("spline order must be positive and n_interior nonnegative")
        if domain is None:
            lower, upper = float(np.min(values)), float(np.max(values))
        else:
            lower, upper = (float(domain[0]), float(domain[1]))
        if not np.isfinite(lower + upper) or not lower < upper:
            raise ValueError("spline domain must have finite positive width")
        if np.any(values < lower - 1e-12) or np.any(values > upper + 1e-12):
            raise ValueError("observed values fall outside the registered spline domain")
        degree = order - 1
        interior = (
            np.linspace(lower, upper, n_interior + 2, dtype=float)[1:-1]
            if n_interior
            else np.empty(0, dtype=float)
        )
        knots = np.concatenate(
            [
                np.repeat(lower, degree + 1),
                interior,
                np.repeat(upper, degree + 1),
            ]
        )
        return cls(lower=lower, upper=upper, degree=degree, knots=knots)

    @classmethod
    def quantile(
        cls,
        values: np.ndarray,
        *,
        n_interior: int,
        order: int = 4,
        domain: tuple[float, float] | None = None,
    ) -> "SplineBasis":
        """B-spline basis with interior knots at empirical quantiles.

        Hu, Huang and You (2021, Section 5) place knots at equally spaced
        sample quantiles.  The registered domain still supplies the repeated
        boundary knots so held-out prediction uses the training basis without
        silently changing its support.
        """

        values = np.asarray(values, dtype=float)
        if order < 1 or n_interior < 0:
            raise ValueError("spline order must be positive and n_interior nonnegative")
        if domain is None:
            lower, upper = float(np.min(values)), float(np.max(values))
        else:
            lower, upper = (float(domain[0]), float(domain[1]))
        if not np.isfinite(lower + upper) or not lower < upper:
            raise ValueError("spline domain must have finite positive width")
        if np.any(values < lower - 1e-12) or np.any(values > upper + 1e-12):
            raise ValueError("observed values fall outside the registered spline domain")
        degree = order - 1
        if n_interior:
            probabilities = np.arange(1, n_interior + 1, dtype=float) / (
                n_interior + 1.0
            )
            interior = np.asarray(np.quantile(values, probabilities), dtype=float)
            if np.any(np.diff(interior) <= 1e-12):
                raise ValueError("empirical quantile knots are not strictly increasing")
        else:
            interior = np.empty(0, dtype=float)
        knots = np.concatenate(
            [
                np.repeat(lower, degree + 1),
                interior,
                np.repeat(upper, degree + 1),
            ]
        )
        return cls(lower=lower, upper=upper, degree=degree, knots=knots)

    @property
    def dimension(self) -> int:
        return int(len(self.knots) - self.degree - 1)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        coefficients = np.eye(self.dimension)
        matrix = np.asarray(
            BSpline(self.knots, coefficients, self.degree, extrapolate=False)(values),
            dtype=float,
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("prediction values fall outside the fitted spline domain")
        return matrix

    def integral_vector(self) -> np.ndarray:
        coefficients = np.eye(self.dimension)
        spline = BSpline(self.knots, coefficients, self.degree, extrapolate=False)
        return np.asarray(spline.integrate(self.lower, self.upper), dtype=float)


@dataclass(frozen=True)
class CenteredSplineBasis:
    """A full-rank mean-zero contrast of an ordinary B-spline basis."""

    raw: SplineBasis
    contrast: np.ndarray
    centering: str

    @classmethod
    def empirical(
        cls,
        raw: SplineBasis,
        values: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "CenteredSplineBasis":
        matrix = raw.transform(values)
        if weights is None:
            mean = np.mean(matrix, axis=0)
        else:
            weights = np.asarray(weights, dtype=float)
            mean = np.sum(weights[:, None] * matrix, axis=0) / np.sum(weights)
        pivot = int(np.argmax(np.abs(mean)))
        if abs(mean[pivot]) < 1e-14:
            raise ValueError("cannot construct a centered spline basis")
        keep = [index for index in range(raw.dimension) if index != pivot]
        contrast = np.zeros((raw.dimension, raw.dimension - 1), dtype=float)
        for column, index in enumerate(keep):
            contrast[index, column] = 1.0
            contrast[pivot, column] = -mean[index] / mean[pivot]
        return cls(raw=raw, contrast=contrast, centering="empirical-observation-mean")

    @property
    def dimension(self) -> int:
        return int(self.contrast.shape[1])

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.raw.transform(values) @ self.contrast

    def integral_vector(self) -> np.ndarray:
        return self.raw.integral_vector() @ self.contrast


@dataclass
class SplineFunction:
    basis: SplineBasis | CenteredSplineBasis
    coefficients: np.ndarray
    offset: float = 0.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return self.basis.transform(np.asarray(values, dtype=float)) @ self.coefficients + self.offset

    def integral(self) -> float:
        raw = float(self.basis.integral_vector() @ self.coefficients)
        width = self.basis.raw.upper - self.basis.raw.lower if isinstance(self.basis, CenteredSplineBasis) else self.basis.upper - self.basis.lower
        return raw + self.offset * width

    def normalized_integral_one(self, *, tolerance: float = 1e-10) -> "SplineFunction":
        integral = self.integral()
        if not np.isfinite(integral) or abs(integral) <= tolerance:
            raise FloatingPointError("coefficient function has a near-zero Lebesgue integral")
        return SplineFunction(self.basis, self.coefficients / integral, self.offset / integral)

    def normalized_mean_one(self, *, tolerance: float = 1e-10) -> "SplineFunction":
        raw_basis = self.basis.raw if isinstance(self.basis, CenteredSplineBasis) else self.basis
        mean = self.integral() / (raw_basis.upper - raw_basis.lower)
        if not np.isfinite(mean) or abs(mean) <= tolerance:
            raise FloatingPointError("coefficient function has a near-zero domain average")
        return SplineFunction(self.basis, self.coefficients / mean, self.offset / mean)

    def centered_lebesgue(self) -> "SplineFunction":
        raw_basis = self.basis.raw if isinstance(self.basis, CenteredSplineBasis) else self.basis
        mean = self.integral() / (raw_basis.upper - raw_basis.lower)
        return SplineFunction(self.basis, self.coefficients.copy(), self.offset - mean)

    def centered_empirical(self, values: np.ndarray, weights: np.ndarray | None = None) -> "SplineFunction":
        evaluated = self(values)
        center = float(np.mean(evaluated) if weights is None else np.sum(weights * evaluated) / np.sum(weights))
        return SplineFunction(self.basis, self.coefficients.copy(), self.offset - center)


def solve_least_squares(features: np.ndarray, response: np.ndarray) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(
        np.asarray(features, dtype=float), np.asarray(response, dtype=float), rcond=None
    )
    return np.asarray(coefficients, dtype=float)


@dataclass(frozen=True)
class HuberSolve:
    coefficients: np.ndarray
    converged: bool
    iterations: int
    objective: tuple[float, ...]
    # ``converged`` is an accepted stopping decision.  Keep the stricter
    # coefficient-change decision separate so paper adapters can report when
    # a numerically flat objective, rather than coefficient stabilization,
    # ended an otherwise finite IRLS solve.
    coefficient_converged: bool = False
    objective_stable: bool = False
    termination: str = "iteration_limit"


def huber_objective(residual: np.ndarray, weights: np.ndarray, delta: float) -> float:
    absolute = np.abs(residual)
    values = np.where(absolute <= delta, 0.5 * residual**2, delta * absolute - 0.5 * delta**2)
    return float(np.sum(weights * values))


def solve_subject_balanced_huber(
    features: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float = 1.345,
    tolerance: float = 1e-8,
    max_iter: int = 200,
    objective_relative_tolerance: float | None = None,
    objective_stable_steps: int = 3,
) -> HuberSolve:
    """Convex Huber M-estimation by monotone IRLS.

    The ordinary stopping rule is a relative coefficient change.  An adapter
    can additionally register an objective-tail rule for rank-deficient or
    nearly collinear spline designs, where the fitted Huber objective has
    stabilized but nonidentified coefficient coordinates continue to move.
    That rule is opt-in and its termination mode is retained in
    :class:`HuberSolve` for audit purposes.
    """

    features = np.asarray(features, dtype=float)
    response = np.asarray(response, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if delta <= 0 or np.any(weights <= 0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("delta must be positive and subject weights must sum to one")
    if objective_relative_tolerance is not None:
        if not np.isfinite(objective_relative_tolerance) or objective_relative_tolerance <= 0:
            raise ValueError("objective_relative_tolerance must be finite and positive")
        if objective_stable_steps < 1:
            raise ValueError("objective_stable_steps must be positive")
    root_weight = np.sqrt(weights)
    coefficients = solve_least_squares(root_weight[:, None] * features, root_weight * response)
    history: list[float] = []
    converged = False
    coefficient_converged = False
    objective_stable = False
    termination = "iteration_limit"
    for iteration in range(1, max_iter + 1):
        residual = response - features @ coefficients
        absolute = np.abs(residual)
        robust_weight = np.ones_like(absolute)
        mask = absolute > delta
        robust_weight[mask] = delta / absolute[mask]
        combined = np.sqrt(weights * robust_weight)
        candidate = solve_least_squares(combined[:, None] * features, combined * response)
        objective = huber_objective(response - features @ candidate, weights, delta)
        history.append(objective)
        relative_change = np.linalg.norm(candidate - coefficients) / (1.0 + np.linalg.norm(coefficients))
        coefficients = candidate
        if relative_change <= tolerance:
            converged = True
            coefficient_converged = True
            termination = "coefficient_change"
            break
        if (
            objective_relative_tolerance is not None
            and len(history) >= objective_stable_steps + 1
        ):
            tail = np.asarray(history[-(objective_stable_steps + 1) :], dtype=float)
            differences = np.abs(np.diff(tail))
            scales = np.maximum(1.0, np.abs(tail[1:]))
            if np.all(np.isfinite(tail)) and np.all(
                differences / scales <= objective_relative_tolerance
            ):
                converged = True
                objective_stable = True
                termination = "objective_stable"
                break
    return HuberSolve(
        coefficients,
        converged,
        iteration,
        tuple(history),
        coefficient_converged=coefficient_converged,
        objective_stable=objective_stable,
        termination=termination,
    )


def subject_balanced_weights(subject_id: np.ndarray) -> np.ndarray:
    subjects, inverse, counts = np.unique(subject_id.astype(str), return_inverse=True, return_counts=True)
    return 1.0 / (len(subjects) * counts[inverse])


@dataclass(frozen=True)
class LassoSolve:
    coefficients: np.ndarray
    converged: bool
    iterations: int
    objective: tuple[float, ...]


def _soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def solve_paper_lasso(
    features: np.ndarray,
    response: np.ndarray,
    penalty: float | np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iter: int = 5000,
    initial: np.ndarray | None = None,
    lipschitz: float | None = None,
    deadline: float | None = None,
) -> LassoSolve:
    """Minimize ``mean((y-Xb)^2) + sum_j penalty_j |b_j|`` by FISTA."""

    features = np.asarray(features, dtype=float)
    response = np.asarray(response, dtype=float)
    penalty_vector = np.broadcast_to(np.asarray(penalty, dtype=float), (features.shape[1],)).copy()
    if np.any(penalty_vector < 0):
        raise ValueError("L1 penalties must be nonnegative")
    n_rows = len(response)
    if lipschitz is None:
        spectral = (
            float(np.linalg.svd(features, compute_uv=False)[0])
            if features.size
            else 0.0
        )
        lipschitz = max(2.0 * spectral**2 / n_rows, 1e-12)
    else:
        lipschitz = max(float(lipschitz), 1e-12)
    coefficients = (
        np.zeros(features.shape[1], dtype=float)
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    if coefficients.shape != (features.shape[1],):
        raise ValueError("initial Lasso coefficients have the wrong shape")
    extrapolated = coefficients.copy()
    momentum = 1.0
    history: list[float] = []
    converged = False
    objective_stable_iterations = 0
    for iteration in range(1, max_iter + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("paper-Lasso fit exceeded its registered time budget")
        gradient = 2.0 * features.T @ (features @ extrapolated - response) / n_rows
        candidate = _soft_threshold(
            extrapolated - gradient / lipschitz, penalty_vector / lipschitz
        )
        residual = response - features @ candidate
        objective = float(np.mean(residual**2) + np.dot(penalty_vector, np.abs(candidate)))
        if history:
            relative_objective_change = abs(objective - history[-1]) / (
                1.0 + abs(history[-1])
            )
            objective_stable_iterations = (
                objective_stable_iterations + 1
                if relative_objective_change <= tolerance
                else 0
            )
        history.append(objective)
        relative_change = np.linalg.norm(candidate - coefficients) / (
            1.0 + np.linalg.norm(coefficients)
        )
        new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = candidate + ((momentum - 1.0) / new_momentum) * (
            candidate - coefficients
        )
        coefficients = candidate
        momentum = new_momentum
        # Accelerated proximal gradients can keep making tiny oscillatory
        # coefficient moves after the convex objective has reached numerical
        # precision.  Requiring five consecutive stable objective values
        # avoids treating that harmless FISTA tail as nonconvergence while
        # retaining the stricter coefficient-change criterion when it fires.
        if relative_change <= tolerance or objective_stable_iterations >= 5:
            converged = True
            break
    return LassoSolve(coefficients, converged, iteration, tuple(history))


@dataclass(frozen=True)
class LassoCV:
    fit: LassoSolve
    penalty: float
    candidate_penalties: tuple[float, ...]
    mean_errors: tuple[float, ...]
    nonconverged_folds: tuple[int, ...] = ()
    solver_warnings: int = 0


def ten_fold_minimum_error_lasso(
    features: np.ndarray,
    response: np.ndarray,
    *,
    seed: int,
    penalty_grid: np.ndarray | None = None,
    n_folds: int = 10,
    tolerance: float = 1e-7,
    max_iter: int = 3000,
    solver: str = "fista",
    deadline: float | None = None,
) -> LassoCV:
    """Paper-specified ten-fold CV using the minimum average squared error."""

    features = np.asarray(features, dtype=float)
    response = np.asarray(response, dtype=float)
    n_rows = len(response)
    n_folds = min(int(n_folds), n_rows)
    if n_folds < 2:
        raise ValueError("at least two rows are required for cross-validation")
    if penalty_grid is None:
        lambda_max = max(
            2.0 * float(np.max(np.abs(features.T @ response))) / n_rows, 1e-8
        )
        penalty_grid = lambda_max * np.logspace(0.0, -4.0, 30)
    penalties = np.asarray(penalty_grid, dtype=float)
    if penalties.ndim != 1 or len(penalties) == 0 or np.any(penalties < 0):
        raise ValueError("penalty_grid must be a nonempty nonnegative vector")
    permutation = np.random.default_rng(int(seed)).permutation(n_rows)
    folds = np.array_split(permutation, n_folds)
    errors = np.zeros(len(penalties), dtype=float)
    if solver in {"coordinate_path", "coordinate_fista_path"}:
        # The paper specifies the CV criterion, not a particular numerical
        # optimizer.  A warm coordinate-descent path solves exactly the same
        # scalar-L1 objective as ``solve_paper_lasso`` because sklearn's alpha
        # equals one half of the penalty used in mean(RSS)+lambda*|b|_1.
        from sklearn.linear_model import lasso_path

        descending = np.argsort(penalties)[::-1]
        path_penalties = penalties[descending]
        path_alphas = path_penalties / 2.0
        nonconverged = np.zeros(len(penalties), dtype=int)
        solver_warnings = 0
        for fold in folds:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("paper-Lasso CV exceeded its registered time budget")
            validation = np.asarray(fold, dtype=int)
            training = np.setdiff1d(
                np.arange(n_rows), validation, assume_unique=True
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                returned_alphas, coefficients, _, iterations = lasso_path(
                    features[training],
                    response[training],
                    alphas=path_alphas,
                    tol=tolerance,
                    max_iter=max_iter,
                    return_n_iter=True,
                )
            solver_warnings += len(caught)
            iteration_array = np.asarray(iterations, dtype=int)
            if iteration_array.shape != (len(path_penalties),):
                raise RuntimeError("coordinate path returned invalid iteration counts")
            coordinate_nonconverged = iteration_array >= max_iter
            if solver == "coordinate_fista_path":
                training_features = features[training]
                training_response = response[training]
                spectral = (
                    float(np.linalg.svd(training_features, compute_uv=False)[0])
                    if training_features.size
                    else 0.0
                )
                path_lipschitz = max(
                    2.0 * spectral**2 / len(training_response), 1e-12
                )
                refined_nonconverged = np.zeros(len(path_penalties), dtype=bool)
                for path_index, penalty in enumerate(path_penalties):
                    refined = solve_paper_lasso(
                        training_features,
                        training_response,
                        float(penalty),
                        tolerance=tolerance,
                        max_iter=max_iter,
                        initial=coefficients[:, path_index],
                        lipschitz=path_lipschitz,
                        deadline=deadline,
                    )
                    coefficients[:, path_index] = refined.coefficients
                    refined_nonconverged[path_index] = not refined.converged
                nonconverged[descending] += refined_nonconverged
            else:
                nonconverged[descending] += coordinate_nonconverged
            if not np.allclose(returned_alphas, path_alphas, rtol=1e-12, atol=0.0):
                raise RuntimeError("coordinate path changed the registered penalty grid")
            residual = (
                response[validation, None]
                - features[validation] @ coefficients
            )
            errors[descending] += np.mean(residual**2, axis=0) / n_folds
        chosen_index = int(np.argmin(errors))
        chosen = float(penalties[chosen_index])
        final = solve_paper_lasso(
            features,
            response,
            chosen,
            tolerance=tolerance,
            max_iter=max_iter,
            deadline=deadline,
        )
        return LassoCV(
            fit=final,
            penalty=chosen,
            candidate_penalties=tuple(float(item) for item in penalties),
            mean_errors=tuple(float(item) for item in errors),
            nonconverged_folds=tuple(int(item) for item in nonconverged),
            solver_warnings=int(solver_warnings),
        )
    if solver not in {"fista", "fista_warm_path"}:
        raise ValueError(
            "solver must be 'fista', 'fista_warm_path', "
            "'coordinate_path', or 'coordinate_fista_path'"
        )
    for fold in folds:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("paper-Lasso CV exceeded its registered time budget")
        validation = np.asarray(fold, dtype=int)
        training = np.setdiff1d(np.arange(n_rows), validation, assume_unique=True)
        training_features = features[training]
        training_response = response[training]
        path_order = (
            np.argsort(penalties)[::-1]
            if solver == "fista_warm_path"
            else np.arange(len(penalties))
        )
        path_lipschitz: float | None = None
        initial: np.ndarray | None = None
        if solver == "fista_warm_path":
            spectral = (
                float(np.linalg.svd(training_features, compute_uv=False)[0])
                if training_features.size
                else 0.0
            )
            path_lipschitz = max(
                2.0 * spectral**2 / len(training_response), 1e-12
            )
        for index in path_order:
            penalty = penalties[index]
            fit = solve_paper_lasso(
                training_features,
                training_response,
                float(penalty),
                tolerance=tolerance,
                max_iter=max_iter,
                initial=initial,
                lipschitz=path_lipschitz,
                deadline=deadline,
            )
            if solver == "fista_warm_path":
                initial = fit.coefficients
            errors[index] += float(
                np.mean((response[validation] - features[validation] @ fit.coefficients) ** 2)
            ) / n_folds
    chosen_index = int(np.argmin(errors))
    chosen = float(penalties[chosen_index])
    final = solve_paper_lasso(
        features,
        response,
        chosen,
        tolerance=tolerance,
        max_iter=max_iter,
        deadline=deadline,
    )
    return LassoCV(
        fit=final,
        penalty=chosen,
        candidate_penalties=tuple(float(item) for item in penalties),
        mean_errors=tuple(float(item) for item in errors),
    )

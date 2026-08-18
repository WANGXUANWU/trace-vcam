"""Shared fitting helpers for the revision experiments.

The checks added during revision all fit the proposed estimator directly rather
than through the benchmark adapter layer, because each of them varies something
the adapter contract holds fixed -- the penalty level, the roughness weight, the
rank of the projection, or the normalisation used to report the factors.  They
share the helpers below so that every one of them reads the same tuning lock,
builds the same whitened design, and identifies the delivered factors the same
way as the formal benchmark.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID_SIZE = 201


# ---------------------------------------------------------------------------
# Tuning and design
# ---------------------------------------------------------------------------


def locked_tuning() -> dict[str, float | int]:
    """The tuning constants of the formal benchmark, read from the same lock."""

    from src.trace_tuning_protocol import load_trace_tuning_lock

    lock = load_trace_tuning_lock()
    return {
        "q_time": 6,
        "q_covariate": 6,
        "lambda_ratio": float(lock["lambda_ratio"]),
        "roughness": float(lock["roughness"]),
        "max_iter": 2000,
        "tolerance": 1e-7,
        "postfit_max_iter": 1000,
        "postfit_tolerance": 2e-7,
    }


def build_design(time, covariates, response, subject, basis, domain_time):
    """The whitened tensor-spline design on the registered [0,1] coordinate."""

    from src.trace_vcam import VCAMDesign

    low, high = domain_time
    scaled_time = (np.asarray(time, dtype=float) - low) / (high - low)
    return VCAMDesign.from_arrays(
        scaled_time, np.asarray(covariates, dtype=float), response, subject, basis
    )


# ---------------------------------------------------------------------------
# Delivered curves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Delivered:
    """One fit, in the two forms the revision experiments need.

    ``beta``/``phi`` are the factors as the paper reports them, under the
    integral-one convention.  ``beta_l2``/``phi_l2`` are the same fitted
    component written in the alternative convention that normalises the time
    factor in :math:`L^2` and fixes its sign, which is defined whether or not the
    time factor has a nonzero mean.  ``blocks`` keeps the pilot coefficient
    matrices, whose singular spectrum is what the rank-one diagnostic reads.
    """

    baseline: np.ndarray
    beta: tuple[np.ndarray | None, ...]
    phi: tuple[np.ndarray | None, ...]
    beta_l2: tuple[np.ndarray | None, ...]
    phi_l2: tuple[np.ndarray | None, ...]
    gamma: np.ndarray
    blocks: tuple[np.ndarray, ...]
    singular_values: tuple[np.ndarray, ...]
    time_integral_margin: tuple[float, ...]
    selected: np.ndarray
    delta: float
    converged: bool
    runtime_seconds: float


def _l2_convention(
    matrix: np.ndarray, time_basis: np.ndarray, covariate_basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The leading component written with a unit-:math:`L^2` time factor.

    The marginal bases are Lebesgue-orthonormal, so a unit left singular vector
    already gives a time factor of unit :math:`L^2` norm and the whole amplitude
    sits in the covariate factor.  The remaining freedom is one sign, which is
    fixed here by making the time factor positive where it is largest in
    absolute value.  That rule is defined for every nonzero block, including the
    ones whose time factor integrates to zero and for which the integral-one
    convention of the paper has no solution.
    """

    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    beta = time_basis @ left[:, 0]
    phi = singular[0] * (covariate_basis @ right_t[0])
    sign = 1.0 if beta[int(np.argmax(np.abs(beta)))] >= 0.0 else -1.0
    return sign * beta, sign * phi


def fit_delivered(
    design,
    basis,
    *,
    tuning: Mapping[str, object],
    lambda_ratio: float | None = None,
    roughness: float | None = None,
    delta: float | None = None,
    post_rank_one: bool = True,
) -> Delivered:
    """One TRACE fit, returned as curves plus the pilot spectrum.

    ``post_rank_one=False`` returns the convex pilot itself, without the rank-one
    projection and scalar refit.  That variant is not the estimator the paper
    delivers; it is used only as the unrestricted reference against which the
    price of the rank-one restriction is measured.
    """

    from src.trace_vcam import (
        fit_trace_vcam,
        practical_huber_threshold,
        trace_lambda_max,
    )

    ratio = float(tuning["lambda_ratio"] if lambda_ratio is None else lambda_ratio)
    mu = float(tuning["roughness"] if roughness is None else roughness)
    if delta is None:
        resolved, _ = practical_huber_threshold(design, multiplier=1.345)
    else:
        resolved = float(delta)
    lambda_max, _ = trace_lambda_max(design, resolved)
    common = {
        "max_iter": int(tuning["max_iter"]),
        "tolerance": float(tuning["tolerance"]),
        "mu": mu,
        "postfit_max_iter": int(tuning["postfit_max_iter"]),
        "postfit_tolerance": float(tuning["postfit_tolerance"]),
        "post_rank_one": post_rank_one,
        "post_refit": post_rank_one,
    }
    if delta is None:
        fit = fit_trace_vcam(
            design,
            ratio * lambda_max,
            delta=None,
            threshold_mode="mad",
            huber_multiplier=1.345,
            **common,
        )
    else:
        fit = fit_trace_vcam(
            design, ratio * lambda_max, delta=resolved, threshold_mode="fixed", **common
        )

    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    time_basis = basis.transform_time(grid)
    covariate_basis = basis.transform_covariate(grid)
    baseline = time_basis @ fit.gamma

    time_factors = fit.identified_time_factors or []
    covariate_factors = fit.identified_covariate_factors or []
    beta: list[np.ndarray | None] = []
    phi: list[np.ndarray | None] = []
    beta_l2: list[np.ndarray | None] = []
    phi_l2: list[np.ndarray | None] = []
    margins: list[float] = []
    singular: list[np.ndarray] = []
    for index, matrix in enumerate(fit.matrices):
        b = time_factors[index] if index < len(time_factors) else None
        f = covariate_factors[index] if index < len(covariate_factors) else None
        beta.append(None if b is None else time_basis @ b)
        phi.append(None if f is None else covariate_basis @ f)
        values = np.linalg.svd(matrix, compute_uv=False)
        singular.append(values)
        if values[0] <= 0.0:
            beta_l2.append(None)
            phi_l2.append(None)
            margins.append(float("nan"))
            continue
        left, _, _ = np.linalg.svd(matrix, full_matrices=False)
        # The normalisation margin of the identification assumption: the
        # integral-one convention divides by exactly this number.
        margins.append(float(abs(basis.time_integral @ left[:, 0])))
        curve_beta, curve_phi = _l2_convention(matrix, time_basis, covariate_basis)
        beta_l2.append(curve_beta)
        phi_l2.append(curve_phi)

    return Delivered(
        baseline=baseline,
        beta=tuple(beta),
        phi=tuple(phi),
        beta_l2=tuple(beta_l2),
        phi_l2=tuple(phi_l2),
        gamma=np.asarray(fit.gamma, dtype=float),
        blocks=tuple(np.asarray(m, dtype=float) for m in fit.matrices),
        singular_values=tuple(singular),
        time_integral_margin=tuple(margins),
        selected=np.asarray(fit.selected, dtype=bool),
        delta=float(fit.delta if fit.delta is not None else resolved),
        converged=bool(fit.converged),
        runtime_seconds=float(fit.runtime_seconds),
    )


def predict(curves: Delivered, time, covariates, domain_time) -> np.ndarray:
    """Fitted mean at arbitrary points, by interpolating the delivered curves."""

    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    low, high = domain_time
    scaled = (np.asarray(time, dtype=float) - low) / (high - low)
    values = np.interp(scaled, grid, curves.baseline)
    covariates = np.asarray(covariates, dtype=float)
    for index in range(covariates.shape[1]):
        beta, phi = curves.beta[index], curves.phi[index]
        if beta is None or phi is None:
            continue
        values = values + np.interp(scaled, grid, beta) * np.interp(
            covariates[:, index], grid, phi
        )
    return values


def rank_r_refit(
    design,
    gamma: np.ndarray,
    pilot_blocks: Sequence[np.ndarray],
    selected: np.ndarray,
    delta: float,
    rank: int,
    *,
    max_iter: int = 1000,
    tolerance: float = 2e-7,
):
    """Refit amplitudes on the leading ``rank`` directions of each pilot block.

    This is the delivered second stage with the rank of the projection left free
    instead of fixed at one.  At ``rank=1`` it reparametrises the estimator the
    paper delivers -- the identification rescaling of the direction cancels
    against the amplitude, so the fitted surface is the same -- and at higher
    ranks it is the natural comparison a practitioner needs in order to ask
    whether the rank-one restriction costs anything on their data.

    It lives here, and not in the estimator module, because it is a diagnostic:
    nothing in the reported estimator or in any formal benchmark number is
    computed through it.
    """

    from src.trace_vcam import _solve_huber_linear

    active = np.flatnonzero(np.asarray(selected, dtype=bool))
    q_time = design.time.shape[1]
    directions: list[tuple[int, np.ndarray]] = []
    features: list[np.ndarray] = []
    initial_amplitudes: list[float] = []
    for index in active:
        left, singular, right_t = np.linalg.svd(
            pilot_blocks[index], full_matrices=False
        )
        usable = int(min(rank, np.sum(singular > 0.0)))
        for component in range(usable):
            direction = np.outer(left[:, component], right_t[component])
            directions.append((int(index), direction))
            initial_amplitudes.append(float(singular[component]))
            features.append(
                np.einsum(
                    "ni,ij,nj->n", design.time, direction, design.covariates[index]
                )
            )
    if not features:
        return np.asarray(gamma, dtype=float), [
            np.zeros_like(block) for block in pilot_blocks
        ], False

    postfit_design = np.column_stack([design.time, *features])
    initial = np.concatenate(
        [np.asarray(gamma, dtype=float), np.asarray(initial_amplitudes, dtype=float)]
    )
    result = _solve_huber_linear(
        postfit_design,
        design.response,
        design.weights,
        float(delta),
        max_iter=max_iter,
        tolerance=tolerance,
        initial=initial,
    )
    solution = result.coefficients
    blocks = [np.zeros_like(block) for block in pilot_blocks]
    for position, (index, direction) in enumerate(directions):
        blocks[index] = blocks[index] + float(solution[q_time + position]) * direction
    return solution[:q_time], blocks, bool(result.converged)


def predict_on_design(curves: Delivered, design) -> np.ndarray:
    """Fitted mean evaluated through the coefficient blocks themselves.

    A pilot block need not be rank one, so its fitted surface is not the outer
    product of two curves and cannot be evaluated by interpolating them.  This
    route applies the fitted coefficients to a design built on the rows to be
    predicted, and therefore works for the restricted and unrestricted fits
    alike, which is what makes the two comparable.
    """

    from src.trace_vcam import predict_components

    return predict_components(design, curves.gamma, list(curves.blocks))[0]


# ---------------------------------------------------------------------------
# Errors on the identified scale
# ---------------------------------------------------------------------------


def domain_average_squared_error(grid: np.ndarray, error: np.ndarray) -> float:
    length = float(grid[-1] - grid[0])
    if length <= 0.0:
        return float("nan")
    return float(np.trapezoid(error**2, grid) / length)


def surface_error(
    curves: Delivered,
    truth,
    time_grid: np.ndarray,
    covariate_grid: np.ndarray,
    *,
    blocks: Sequence[int] | None = None,
) -> float:
    """Aggregated component-surface error, scoring a missing block at zero."""

    total = 0.0
    wanted = range(len(truth.active)) if blocks is None else blocks
    for index in wanted:
        if not truth.active[index]:
            continue
        beta_true = truth.beta[index](time_grid)
        phi_true = truth.phi[index](covariate_grid)
        beta_hat = curves.beta[index] if index < len(curves.beta) else None
        phi_hat = curves.phi[index] if index < len(curves.phi) else None
        if beta_hat is None or phi_hat is None:
            estimate = np.zeros((time_grid.size, covariate_grid.size))
        else:
            estimate = np.outer(beta_hat, phi_hat)
        total += float(np.mean((estimate - np.outer(beta_true, phi_true)) ** 2))
    return total


def baseline_error(curves: Delivered, truth, time_grid: np.ndarray) -> float:
    return domain_average_squared_error(
        time_grid, curves.baseline - truth.beta0(time_grid)
    )

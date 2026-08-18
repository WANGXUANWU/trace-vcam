"""Compatibility helpers for the integrated TRACE-VCAM roughness penalty.

Roughness is now part of the production estimator in :mod:`src.trace_vcam`.
This module retains the former public helper names for archived diagnostics;
it delegates fitting to the same two-stage implementation and therefore does
not maintain a second optimizer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.trace_vcam import OrthonormalSplineBasis, VCAMDesign, VCAMFit, fit_trace_vcam


FloatArray = NDArray[np.float64]


def spline_roughness_matrices(
    basis: OrthonormalSplineBasis,
) -> tuple[FloatArray, FloatArray]:
    """Return the normalized spline roughness operators stored by ``basis``."""

    return basis.time_roughness.copy(), basis.covariate_roughness.copy()


def surface_roughness(
    matrices: Sequence[FloatArray],
    time_penalty: FloatArray,
    covariate_penalty: FloatArray,
) -> float:
    """Evaluate tensor-product integrated squared second derivatives."""

    value = 0.0
    for matrix in matrices:
        value += float(np.sum(matrix * (time_penalty @ matrix)))
        value += float(np.sum(matrix * (matrix @ covariate_penalty)))
    return float(max(value, 0.0))


def fit_trace_vcam_roughness(
    design: VCAMDesign,
    penalty: float,
    delta: float | None,
    roughness: float,
    time_penalty: FloatArray,
    covariate_penalty: FloatArray,
    max_iter: int = 500,
    tolerance: float = 1e-7,
    initial: VCAMFit | None = None,
    post_rank_one: bool = True,
    post_refit: bool = True,
    selection_tolerance: float = 1e-9,
    block_weights: ArrayLike | None = None,
) -> VCAMFit:
    """Delegate the historical roughness API to production TRACE-VCAM.

    Custom matrices are accepted only when they match the operators carried by
    ``design``.  This prevents the compatibility wrapper from silently fitting
    a criterion different from the one reported by the production object.
    """

    if roughness < 0.0 or not np.isfinite(roughness):
        raise ValueError("roughness must be finite and nonnegative")
    if time_penalty.shape != design.time_roughness.shape or not np.allclose(
        time_penalty, design.time_roughness, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("time_penalty must match the design roughness operator")
    if (
        covariate_penalty.shape != design.covariate_roughness.shape
        or not np.allclose(
            covariate_penalty,
            design.covariate_roughness,
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "covariate_penalty must match the design roughness operator"
        )
    return fit_trace_vcam(
        design=design,
        penalty=penalty,
        delta=delta,
        max_iter=max_iter,
        tolerance=tolerance,
        initial=initial,
        post_rank_one=post_rank_one,
        post_refit=post_refit,
        selection_tolerance=selection_tolerance,
        block_weights=block_weights,
        mu=roughness,
    )

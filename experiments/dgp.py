"""Published simulation designs used by the strict VCAM benchmark.

The generators in this module intentionally separate a paper's original design
from any robustness extension.  Every returned data set records that provenance
so an extension cannot be mislabeled as an original-paper reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.special import ndtr


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Truth:
    beta0: Callable[[FloatArray], FloatArray]
    beta: tuple[Callable[[FloatArray], FloatArray], ...]
    phi: tuple[Callable[[FloatArray], FloatArray], ...]
    active: tuple[bool, ...]


@dataclass(frozen=True)
class PublishedDataset:
    time: FloatArray
    covariates: FloatArray
    response: FloatArray
    subject: IntArray
    conditional_mean: FloatArray
    truth: Truth
    design_id: str
    provenance: str
    time_invariant_covariates: bool
    domain_time: tuple[float, float]
    domain_covariates: tuple[tuple[float, float], ...]

    @property
    def n_subjects(self) -> int:
        return int(np.unique(self.subject).size)


def _open_uniform_basis(
    x: FloatArray,
    domain: tuple[float, float],
    n_interior_knots: int,
    degree: int = 3,
) -> FloatArray:
    lower, upper = domain
    if not lower < upper:
        raise ValueError("spline domain must have positive length")
    if n_interior_knots < 0:
        raise ValueError("n_interior_knots must be nonnegative")
    interior = np.linspace(lower, upper, n_interior_knots + 2)[1:-1]
    knots = np.concatenate(
        [
            np.repeat(lower, degree + 1),
            interior,
            np.repeat(upper, degree + 1),
        ]
    )
    clipped = np.clip(np.asarray(x, dtype=float), lower, upper)
    return BSpline.design_matrix(clipped, knots, degree, extrapolate=False).toarray()


def _spline_function(
    coefficients: FloatArray,
    domain: tuple[float, float],
    n_interior_knots: int,
    *,
    normalize_average: bool = False,
    center_integral: bool = False,
) -> Callable[[FloatArray], FloatArray]:
    coefficients = np.asarray(coefficients, dtype=float)
    expected = n_interior_knots + 4
    if coefficients.shape != (expected,):
        raise ValueError(f"expected {expected} coefficients, got {coefficients.shape}")
    grid = np.linspace(domain[0], domain[1], 20001)
    raw_grid = _open_uniform_basis(grid, domain, n_interior_knots) @ coefficients
    integral = float(np.trapezoid(raw_grid, grid))
    length = domain[1] - domain[0]
    shift = integral / length if center_integral else 0.0
    scale = integral / length if normalize_average else 1.0
    if normalize_average and abs(scale) < 1e-10:
        raise ValueError("cannot normalize a spline with zero average")

    def evaluate(x: FloatArray) -> FloatArray:
        raw = _open_uniform_basis(
            np.asarray(x, dtype=float), domain, n_interior_knots
        ) @ coefficients
        return (raw - shift) / scale

    return evaluate


def _gaussian_copula_uniforms(
    rng: np.random.Generator, n: int, correlation: FloatArray
) -> FloatArray:
    return ndtr(rng.multivariate_normal(np.zeros(correlation.shape[0]), correlation, size=n))


def _fourier_random_effect(
    rng: np.random.Generator, time: FloatArray, subject: IntArray, n_subjects: int
) -> FloatArray:
    score_sd = 1.0 / (np.arange(1, 5, dtype=float) + 1.0)
    scores = rng.normal(size=(n_subjects, 4)) * score_sd
    basis = np.column_stack(
        [
            np.sqrt(2.0) * np.cos(2.0 * np.pi * time),
            np.sqrt(2.0) * np.sin(2.0 * np.pi * time),
            np.sqrt(2.0) * np.cos(4.0 * np.pi * time),
            np.sqrt(2.0) * np.sin(4.0 * np.pi * time),
        ]
    )
    return np.sum(scores[subject] * basis, axis=1)


def _assemble(
    *,
    time: FloatArray,
    covariates: FloatArray,
    subject: IntArray,
    truth: Truth,
    random_effect: FloatArray,
    errors: FloatArray,
    design_id: str,
    provenance: str,
    time_invariant_covariates: bool,
    domain_time: tuple[float, float],
) -> PublishedDataset:
    components = np.zeros_like(time, dtype=float)
    for k, active in enumerate(truth.active):
        if active:
            components += truth.beta[k](time) * truth.phi[k](covariates[:, k])
    conditional_mean = truth.beta0(time) + components
    return PublishedDataset(
        time=np.asarray(time, dtype=float),
        covariates=np.asarray(covariates, dtype=float),
        response=conditional_mean + random_effect + errors,
        subject=np.asarray(subject, dtype=np.int64),
        conditional_mean=conditional_mean,
        truth=truth,
        design_id=design_id,
        provenance=provenance,
        time_invariant_covariates=time_invariant_covariates,
        domain_time=domain_time,
        domain_covariates=tuple((0.0, 1.0) for _ in range(covariates.shape[1])),
    )


def generate_zw2015(seed: int, n_subjects: int = 100) -> PublishedDataset:
    """Zhang--Wang (2015), Section 4, exactly N_i=40 and Q=500 design."""

    rng = np.random.default_rng(seed)
    n_per_subject = 40
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), n_per_subject)
    grid = np.linspace(0.0, 1.0, n_per_subject)
    time = np.tile(grid, n_subjects)
    correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    subject_x = _gaussian_copula_uniforms(rng, n_subjects, correlation)
    covariates = subject_x[subject]

    truth = Truth(
        beta0=lambda t: 1.5 * np.sin(3.0 * np.pi * (t + 0.5)) + 4.0 * t**3,
        beta=(lambda t: 3.0 * (1.0 - t) ** 2, lambda t: 4.0 * t**3),
        phi=(lambda z: np.sin(2.0 * np.pi * z), lambda z: 4.0 * z**3 - 1.0),
        active=(True, True),
    )
    random_effect = _fourier_random_effect(rng, time, subject, n_subjects)
    errors = rng.normal(0.0, 0.1, size=time.size)
    return _assemble(
        time=time,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=random_effect,
        errors=errors,
        design_id="ZW2015-Section4",
        provenance="original",
        time_invariant_covariates=True,
        domain_time=(0.0, 1.0),
    )


def zzw2020_truth() -> Truth:
    """Cubic-spline truth from Zhang--Zhong--Wang (2020), Section 5."""

    beta0 = _spline_function(
        np.array([1.0, 2.0, 4.0, 3.0, -2.0, 0.0, 3.0, 6.0]),
        (0.0, 2.0),
        4,
    )
    beta1 = _spline_function(
        np.array([0.0, 5.0, 3.0, 1.0, 0.0]),
        (0.0, 2.0),
        1,
        normalize_average=True,
    )
    beta2 = _spline_function(
        np.array([0.0, 6.0, 2.0, 0.0, 3.0, 0.0]),
        (0.0, 2.0),
        2,
        normalize_average=True,
    )
    phi1 = _spline_function(
        np.array([0.0, -2.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
        (0.0, 1.0),
        3,
        center_integral=True,
    )
    phi2 = _spline_function(
        np.array([0.0, 0.0, 4.0, 2.0, 0.0, 0.0]),
        (0.0, 1.0),
        2,
        center_integral=True,
    )
    return Truth(beta0=beta0, beta=(beta1, beta2), phi=(phi1, phi2), active=(True, True))


def _draw_errors(
    rng: np.random.Generator, size: int, sigma: float, distribution: str
) -> FloatArray:
    if distribution == "gaussian":
        return rng.normal(0.0, sigma, size=size)
    if distribution == "hhy-mixed-normal":
        contaminated = rng.random(size) < 0.05
        errors = rng.normal(0.0, np.sqrt(0.2), size=size)
        errors[contaminated] = rng.normal(0.0, 12.5, size=int(contaminated.sum()))
        return errors
    if distribution == "hhy-t2":
        return 0.5 * rng.standard_t(2, size=size)
    if distribution == "symmetric-contamination":
        contaminated = rng.random(size) < 0.05
        errors = rng.normal(0.0, sigma, size=size)
        errors[contaminated] = rng.normal(0.0, 5.0 * sigma, size=int(contaminated.sum()))
        return errors
    raise ValueError(f"unknown error distribution: {distribution}")


def generate_zzw2020(
    seed: int,
    *,
    n_subjects: int,
    sigma: float,
    error_distribution: str = "gaussian",
) -> PublishedDataset:
    """Longitudinal-covariate Model 1 of Zhang--Zhong--Wang (2020).

    The Gaussian case is the original design.  The two ``hhy-*`` cases retain
    this covariate design but use the original Hu--Huang--You (2021) error laws;
    they are therefore explicitly marked as robustness extensions.
    """

    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(2, 11, size=n_subjects)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time = rng.uniform(0.0, 2.0, size=subject.size)
    corr_u = np.array([[1.0, 0.6], [0.6, 1.0]])
    corr_v = np.array([[1.0, 0.5], [0.5, 1.0]])
    u = _gaussian_copula_uniforms(rng, n_subjects, corr_u)
    v = _gaussian_copula_uniforms(rng, n_subjects, corr_v)
    scaled_time = 0.5 * time
    covariates = np.column_stack(
        [
            0.5 * u[subject, 0] * scaled_time**0.5 + 0.5 * v[subject, 0],
            0.5 * u[subject, 1] * scaled_time ** (1.0 / 3.0) + 0.5 * v[subject, 1],
        ]
    )
    truth = zzw2020_truth()
    random_effect = _fourier_random_effect(rng, time, subject, n_subjects)
    errors = _draw_errors(rng, time.size, sigma, error_distribution)
    provenance = "original" if error_distribution == "gaussian" else "robustness-extension"
    return _assemble(
        time=time,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=random_effect,
        errors=errors,
        design_id=f"ZZW2020-Model1-longitudinal-{error_distribution}",
        provenance=provenance,
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
    )


def generate_hhy2021(seed: int, *, error_distribution: str) -> PublishedDataset:
    """Hu--Huang--You (2021), Example 1 reproduction design (n=30,m=20)."""

    rng = np.random.default_rng(seed)
    n_subjects, n_per_subject = 30, 20
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), n_per_subject)
    time = rng.uniform(0.0, 1.0, size=subject.size)
    eta_sd = np.sqrt((1.0 + time) / (2.0 + time))
    x = 0.8 * time**2 + rng.normal(0.0, eta_sd)
    # The paper does not bound X; map its realized support only at adapter time.
    covariates = x[:, None]

    beta0 = lambda t: np.cos(2.0 * np.pi * t)
    norm_grid = np.linspace(0.0, 1.0, 20001)
    alpha_raw = 2.0 * norm_grid * np.sin(2.0 * np.pi * norm_grid) + 1.0
    alpha_scale = float(np.trapezoid(alpha_raw, norm_grid))
    beta1 = lambda t: (2.0 * t * np.sin(2.0 * np.pi * t) + 1.0) / alpha_scale

    def phi_raw(z: FloatArray) -> FloatArray:
        return 1.5 * np.sin(np.pi * z / 2.0) - z * (1.0 - z)

    # The source model subtracts the population expectation, not the realized
    # sample mean.  Conditional on T=t, X is Gaussian with mean 0.8 t^2 and
    # variance (1+t)/(2+t), so the sine moment is available analytically.  The
    # remaining one-dimensional integral is deterministic and was evaluated
    # on a 1,000,001-point grid (absolute quadrature change below 1e-12).
    center = 0.7131078461795729
    phi1 = lambda z: phi_raw(z) - center
    truth = Truth(beta0=beta0, beta=(beta1,), phi=(phi1,), active=(True,))

    random_effect = np.zeros_like(time)
    # theta=0 in the primary reproduction target; covariance sensitivity is separate.
    errors = _draw_errors(rng, time.size, np.sqrt(0.2), error_distribution)
    dataset = _assemble(
        time=time,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=random_effect,
        errors=errors,
        design_id=f"HHY2021-Example1-theta0-{error_distribution}",
        provenance="original",
        time_invariant_covariates=False,
        domain_time=(0.0, 1.0),
    )
    object.__setattr__(dataset, "domain_covariates", ((float(x.min()), float(x.max())),))
    return dataset


def generate_zsy2026(
    seed: int,
    *,
    n_subjects: int,
    sigma: float,
    error_distribution: str = "gaussian",
    n_covariates: int = 10,
) -> PublishedDataset:
    """Zhao--Sun--Yang (2026), Example 4 and declared contamination extension.

    The original p=10 design has all ten component blocks active; sparsity is in
    its spline coefficients, not variable support.  This detail is preserved.
    Values p=25 or 50 are reserved for the computational scaling extension.
    """

    if n_covariates not in (10, 25, 50):
        raise ValueError("n_covariates must be one of 10, 25, or 50")
    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(2, 11, size=n_subjects)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time = rng.uniform(0.0, 2.0, size=subject.size)
    indices = np.arange(n_covariates)
    correlation = 0.5 ** np.abs(indices[:, None] - indices[None, :])
    u = _gaussian_copula_uniforms(rng, n_subjects, correlation)
    v = _gaussian_copula_uniforms(rng, n_subjects, correlation)
    covariates = (
        0.5 * u[subject] * (0.5 * time[:, None]) ** 0.5 + 0.5 * v[subject]
    )

    beta0 = _spline_function(
        np.array([1.0, 2.0, 4.0, 3.0, -2.0, 0.0, 3.0, 6.0]),
        (0.0, 2.0),
        4,
    )
    beta_common = _spline_function(
        np.array([0.0, 6.0, 2.0, 0.0, 3.0, 1.0]),
        (0.0, 2.0),
        2,
        normalize_average=True,
    )
    phi_common = _spline_function(
        np.array([3.0, 0.0, 4.0, 2.0, 0.0, 1.0]),
        (0.0, 1.0),
        2,
        center_integral=True,
    )
    # The p=10 published design activates every block.  Scaling-only extra
    # covariates are null so p growth does not silently change signal energy.
    active = tuple(k < 10 for k in range(n_covariates))
    zero = lambda z: np.zeros_like(np.asarray(z, dtype=float))
    truth = Truth(
        beta0=beta0,
        beta=tuple(beta_common for _ in range(n_covariates)),
        phi=tuple(phi_common if is_active else zero for is_active in active),
        active=active,
    )
    random_effect = _fourier_random_effect(rng, time, subject, n_subjects)
    errors = _draw_errors(rng, time.size, sigma, error_distribution)
    provenance = (
        "original"
        if n_covariates == 10 and error_distribution == "gaussian"
        else "computational-or-robustness-extension"
    )
    return _assemble(
        time=time,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=random_effect,
        errors=errors,
        design_id=f"ZSY2026-Example4-p{n_covariates}-{error_distribution}",
        provenance=provenance,
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
    )


def generate_block_sparse(
    seed: int,
    *,
    n_subjects: int,
    sigma: float,
    error_distribution: str = "gaussian",
    n_covariates: int = 6,
    n_active: int = 2,
    correlation_base: float = 0.5,
    signal_scale: float = 1.0,
) -> PublishedDataset:
    """Block-sparse sparse-longitudinal design.

    The covariate and cluster mechanism is the correlated construction of
    Zhang--Zhong--Wang and Zhao--Sun--Yang, but only the first ``n_active``
    blocks carry signal; the remaining blocks are exactly null.  This is our own
    design, not a reproduction of a published one: it isolates the situation the
    block penalty is written for, where a moderate number of candidate
    covariates contains a few genuinely separable effects.

    ``correlation_base`` and ``signal_scale`` exist so that the registered
    configuration can be varied without writing a second design.  Both defaults
    reproduce the registered Example 3 exactly, and neither changes how much of
    the random stream is consumed, so a sweep around that configuration and the
    reported example remain bit-identical at the registered point.
    """

    if not 1 <= n_active <= n_covariates:
        raise ValueError("n_active must lie between one and n_covariates")
    if not 0.0 <= correlation_base < 1.0:
        raise ValueError("correlation_base must lie in [0, 1)")
    if signal_scale <= 0.0 or not np.isfinite(signal_scale):
        raise ValueError("signal_scale must be finite and positive")
    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(2, 11, size=n_subjects)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time = rng.uniform(0.0, 2.0, size=subject.size)
    indices = np.arange(n_covariates)
    correlation = correlation_base ** np.abs(indices[:, None] - indices[None, :])
    u = _gaussian_copula_uniforms(rng, n_subjects, correlation)
    v = _gaussian_copula_uniforms(rng, n_subjects, correlation)
    covariates = 0.5 * u[subject] * (0.5 * time[:, None]) ** 0.5 + 0.5 * v[subject]

    base = zzw2020_truth()
    zero = lambda z: np.zeros_like(np.asarray(z, dtype=float))
    one = lambda t: np.ones_like(np.asarray(t, dtype=float))

    def _scaled(function, factor: float):
        return lambda z, _f=function, _s=factor: _s * _f(z)

    active = tuple(k < n_active for k in range(n_covariates))
    beta = tuple(
        base.beta[k % len(base.beta)] if is_active else one
        for k, is_active in enumerate(active)
    )
    # The signal is scaled on the covariate factor alone, so that the time
    # factors keep their unit average and the identification of the design is
    # untouched by the sweep.
    phi = tuple(
        _scaled(base.phi[k % len(base.phi)], signal_scale) if is_active else zero
        for k, is_active in enumerate(active)
    )
    truth = Truth(beta0=base.beta0, beta=beta, phi=phi, active=active)
    random_effect = _fourier_random_effect(rng, time, subject, n_subjects)
    errors = _draw_errors(rng, time.size, sigma, error_distribution)
    return _assemble(
        time=time,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=random_effect,
        errors=errors,
        # The registered Example 3 keeps the identifier it was run under, so
        # that adding the two sweep parameters does not change the design
        # manifest of results already committed.
        design_id=(
            f"BlockSparse-p{n_covariates}-s{n_active}-{error_distribution}"
            if (correlation_base == 0.5 and signal_scale == 1.0)
            else (
                f"BlockSparse-p{n_covariates}-s{n_active}"
                f"-rho{correlation_base:g}-a{signal_scale:g}-{error_distribution}"
            )
        ),
        provenance="own-design",
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
    )


CONTAMINATION_MODES = ("none", "response", "subject", "trajectory", "leverage")
CLUSTER_SIZE_MODES = ("exchangeable", "informative")


def generate_robustness_scope(
    seed: int,
    *,
    n_subjects: int,
    sigma: float = 0.4,
    contamination: str = "none",
    contamination_rate: float = 0.05,
    cluster_size: str = "exchangeable",
) -> PublishedDataset:
    """The Example 2 design, with the contamination channel left free.

    The published robustness comparisons in this literature contaminate the
    response one visit at a time, which is the only channel a bounded-influence
    loss on the residual is built to control.  This generator keeps the
    covariate, cluster, and component construction of Zhang, Zhong and Wang
    fixed and varies where the contamination enters instead, so that robustness
    to a contaminated response can be separated from robustness to a
    contaminated design:

    ``response``
        The registered visitwise mixture: a fraction of visits draw a
        large-variance error.
    ``subject``
        A fraction of complete subjects have every one of their visits shifted
        by one subject-level draw.  The contamination is a cluster, not a visit,
        so it cannot be diluted by the other visits of the same subject.
    ``trajectory``
        A fraction of complete subjects have a smooth spurious curve added to
        their whole trajectory.  Unlike a shift, this is contamination that
        imitates a genuine subject effect, and averaging within a subject does
        not remove it.
    ``leverage``
        A fraction of visits have their covariates moved to the edge of the
        design while the response continues to be generated from the original
        covariate value.  The residual of such a visit need not be large, so a
        Huber loss on the residual does not down-weight it; this is the channel
        the proposed estimator makes no claim about, and it is included so that
        the claim it does make is not read more widely than it is meant.

    ``cluster_size='informative'`` makes the number of visits depend on the
    subject's own covariate level and latent trajectory, which is the regime the
    subject-balanced weighting is written for.
    """

    if contamination not in CONTAMINATION_MODES:
        raise ValueError(f"unknown contamination mode: {contamination}")
    if cluster_size not in CLUSTER_SIZE_MODES:
        raise ValueError(f"unknown cluster-size mode: {cluster_size}")
    if not 0.0 <= contamination_rate < 0.5:
        raise ValueError("contamination_rate must lie in [0, 0.5)")

    rng = np.random.default_rng(seed)
    corr_u = np.array([[1.0, 0.6], [0.6, 1.0]])
    corr_v = np.array([[1.0, 0.5], [0.5, 1.0]])
    u = _gaussian_copula_uniforms(rng, n_subjects, corr_u)
    v = _gaussian_copula_uniforms(rng, n_subjects, corr_v)
    # The latent trajectory is drawn before the cluster sizes so that the number
    # of visits may depend on it.
    score_sd = 1.0 / (np.arange(1, 5, dtype=float) + 1.0)
    scores = rng.normal(size=(n_subjects, 4)) * score_sd

    if cluster_size == "exchangeable":
        cluster_sizes = rng.integers(2, 11, size=n_subjects)
    else:
        # Subjects with a high covariate level and a large latent amplitude are
        # seen less often, so the visit law and the subject law differ.
        amplitude = np.sqrt(np.sum(scores**2, axis=1))
        severity = 0.5 * u[:, 0] + 0.5 * ndtr(
            (amplitude - amplitude.mean()) / (amplitude.std() + 1e-12)
        )
        cluster_sizes = np.clip(np.round(10.0 - 8.0 * severity), 2, 10).astype(int)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time = rng.uniform(0.0, 2.0, size=subject.size)
    scaled_time = 0.5 * time
    covariates = np.column_stack(
        [
            0.5 * u[subject, 0] * scaled_time**0.5 + 0.5 * v[subject, 0],
            0.5 * u[subject, 1] * scaled_time ** (1.0 / 3.0) + 0.5 * v[subject, 1],
        ]
    )
    truth = zzw2020_truth()

    # The conditional mean is always evaluated at the covariate values that
    # generated the response.  Under leverage contamination the recorded
    # covariate is then replaced, which is what makes that channel a design
    # contamination rather than a response one.
    generating_covariates = covariates.copy()
    components = np.zeros_like(time)
    for index, active in enumerate(truth.active):
        if active:
            components += truth.beta[index](time) * truth.phi[index](
                generating_covariates[:, index]
            )
    conditional_mean = truth.beta0(time) + components

    basis = np.column_stack(
        [
            np.sqrt(2.0) * np.cos(2.0 * np.pi * time),
            np.sqrt(2.0) * np.sin(2.0 * np.pi * time),
            np.sqrt(2.0) * np.cos(4.0 * np.pi * time),
            np.sqrt(2.0) * np.sin(4.0 * np.pi * time),
        ]
    )
    random_effect = np.sum(scores[subject] * basis, axis=1)
    errors = rng.normal(0.0, sigma, size=time.size)
    response = conditional_mean + random_effect + errors

    contaminated_subjects = np.zeros(n_subjects, dtype=bool)
    if contamination == "response":
        hit = rng.random(time.size) < contamination_rate
        response[hit] = (
            conditional_mean[hit] + random_effect[hit] + rng.normal(0.0, 12.5, int(hit.sum()))
        )
    elif contamination == "subject":
        chosen = rng.random(n_subjects) < contamination_rate
        contaminated_subjects = chosen
        shift = rng.normal(0.0, 12.5, size=n_subjects)
        response = response + np.where(chosen[subject], shift[subject], 0.0)
    elif contamination == "trajectory":
        chosen = rng.random(n_subjects) < contamination_rate
        contaminated_subjects = chosen
        amplitude = rng.normal(0.0, 12.5, size=n_subjects)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_subjects)
        spurious = amplitude[subject] * np.sin(np.pi * time + phase[subject])
        response = response + np.where(chosen[subject], spurious, 0.0)
    elif contamination == "leverage":
        hit = rng.random(time.size) < contamination_rate
        # Move a contaminated visit to whichever end of the covariate range it is
        # further from, so the replacement is always a high-leverage point.
        for index in range(covariates.shape[1]):
            target = np.where(covariates[hit, index] < 0.5, 1.0, 0.0)
            covariates[hit, index] = target

    dataset = _assemble(
        time=time,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=np.zeros_like(time),
        errors=np.zeros_like(time),
        design_id=(
            f"RobustnessScope-{contamination}-{cluster_size}-r{contamination_rate:g}"
        ),
        provenance="own-design",
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
    )
    # ``_assemble`` evaluates the true regression function at the *recorded*
    # covariates, which is the right held-out target in every mode: the estimand
    # is the regression function, and it is defined at whatever covariate value a
    # row carries.  Under leverage contamination the response of a contaminated
    # training visit was generated at a different covariate value, so the
    # contamination is a training-set inconsistency between design and response
    # rather than a corrupted evaluation target.
    object.__setattr__(dataset, "response", response)
    del contaminated_subjects
    return dataset


def subject_split(
    subject: IntArray, *, seed: int, train_fraction: float = 0.8
) -> tuple[IntArray, IntArray]:
    """Return row indices for a deterministic subject-level split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    unique = np.unique(subject)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = int(np.floor(train_fraction * unique.size))
    train_subjects = shuffled[:n_train]
    train = np.flatnonzero(np.isin(subject, train_subjects)).astype(np.int64)
    test = np.flatnonzero(~np.isin(subject, train_subjects)).astype(np.int64)
    if np.intersect1d(subject[train], subject[test]).size:
        raise AssertionError("subject leakage in train/test split")
    return train, test

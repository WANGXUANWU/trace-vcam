"""Data-generating mechanisms for the TRACE-VCAM numerical study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, ndtr


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    n_subjects: int
    n_covariates: int
    n_active: int
    sigma: float
    error: str
    informative_size: bool = False
    rank_two_strength: float = 0.0


@dataclass
class SimulatedData:
    time: FloatArray
    covariates: FloatArray
    response: FloatArray
    subject: NDArray[np.int64]
    conditional_mean: FloatArray
    component_means: FloatArray
    cluster_sizes: NDArray[np.int64]


def beta0(time: FloatArray) -> FloatArray:
    return 1.5 * np.sin(3.0 * np.pi * (time + 0.5)) + 4.0 * time**3


def beta(index: int, time: FloatArray) -> FloatArray:
    if index == 0:
        return 3.0 * (1.0 - time) ** 2
    if index == 1:
        return 4.0 * time**3
    if index == 2:
        return 1.0 + 0.75 * np.cos(2.0 * np.pi * time)
    raise ValueError(f"No active beta function for index {index}")


def phi(index: int, covariate: FloatArray) -> FloatArray:
    if index == 0:
        return np.sin(2.0 * np.pi * covariate)
    if index == 1:
        return 4.0 * covariate**3 - 1.0
    if index == 2:
        return 6.0 * covariate * (1.0 - covariate) - 1.0
    raise ValueError(f"No active phi function for index {index}")


def rank_two_component(time: FloatArray, covariate: FloatArray) -> FloatArray:
    second_beta = 1.0 + 0.6 * np.sin(4.0 * np.pi * time)
    second_phi = 2.0 * (covariate - 0.5)
    return second_beta * second_phi


def _fourier_random_effect(
    time: FloatArray,
    subject: NDArray[np.int64],
    scores: FloatArray,
) -> FloatArray:
    basis = np.column_stack(
        [
            np.sqrt(2.0) * np.cos(2.0 * np.pi * time),
            np.sqrt(2.0) * np.sin(2.0 * np.pi * time),
            np.sqrt(2.0) * np.cos(4.0 * np.pi * time),
            np.sqrt(2.0) * np.sin(4.0 * np.pi * time),
        ]
    )
    return np.sum(scores[subject] * basis, axis=1)


def _correlated_uniforms(
    rng: np.random.Generator,
    n_subjects: int,
    n_covariates: int,
    correlation: float,
) -> tuple[FloatArray, FloatArray]:
    indices = np.arange(n_covariates)
    covariance = correlation ** np.abs(indices[:, None] - indices[None, :])
    latent_u = rng.multivariate_normal(
        np.zeros(n_covariates), covariance, size=n_subjects
    )
    latent_v = rng.multivariate_normal(
        np.zeros(n_covariates), covariance, size=n_subjects
    )
    return ndtr(latent_u), ndtr(latent_v)


def _errors(
    rng: np.random.Generator,
    size: int,
    sigma: float,
    error: str,
) -> FloatArray:
    if error == "gaussian":
        return rng.normal(0.0, sigma, size=size)
    if error == "t3":
        return sigma * rng.standard_t(3, size=size) / np.sqrt(3.0)
    if error == "contaminated":
        contaminated = rng.random(size) < 0.05
        values = rng.normal(0.0, sigma, size=size)
        values[contaminated] = rng.normal(0.0, 5.0 * sigma, size=np.sum(contaminated))
        return values
    if error == "none":
        return np.zeros(size)
    raise ValueError(f"Unknown error distribution: {error}")


def generate_data(
    scenario: Scenario,
    seed: int,
    include_noise: bool = True,
) -> SimulatedData:
    rng = np.random.default_rng(seed)
    n = scenario.n_subjects
    p = scenario.n_covariates
    correlation = 0.6 if p <= 2 else 0.5
    subject_u, subject_v = _correlated_uniforms(rng, n, p, correlation)

    score_variances = 1.0 / (np.arange(1, 5) + 1.0) ** 2
    scores = rng.normal(0.0, np.sqrt(score_variances), size=(n, 4))
    if scenario.informative_size:
        probabilities = expit(-0.5 + 1.5 * scores[:, 0] / np.sqrt(score_variances[0]))
        cluster_sizes = 2 + rng.binomial(12, probabilities)
    else:
        cluster_sizes = rng.integers(2, 11, size=n)

    subject = np.repeat(np.arange(n, dtype=np.int64), cluster_sizes)
    time = rng.uniform(0.0, 1.0, size=np.sum(cluster_sizes))
    covariates = np.empty((len(time), p), dtype=float)
    for k in range(p):
        exponent = 1.0 / (2.0 + (k % 3))
        covariates[:, k] = (
            0.5 * subject_u[subject, k] * time**exponent
            + 0.5 * subject_v[subject, k]
        )

    components = np.zeros((len(time), p), dtype=float)
    for k in range(scenario.n_active):
        components[:, k] = beta(k, time) * phi(k, covariates[:, k])
    if scenario.rank_two_strength > 0.0:
        components[:, 0] += scenario.rank_two_strength * rank_two_component(
            time, covariates[:, 0]
        )
    mean = beta0(time) + np.sum(components, axis=1)

    if include_noise:
        trajectory = _fourier_random_effect(time, subject, scores)
        measurement_error = _errors(
            rng, len(time), scenario.sigma, scenario.error
        )
        response = mean + trajectory + measurement_error
    else:
        response = mean.copy()
    return SimulatedData(
        time=time,
        covariates=covariates,
        response=response,
        subject=subject,
        conditional_mean=mean,
        component_means=components,
        cluster_sizes=cluster_sizes,
    )


def primary_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for n in (50, 200):
        scenarios.extend(
            [
                Scenario(f"canonical-n{n}-g01", "canonical", n, 2, 2, 0.1, "gaussian"),
                Scenario(f"canonical-n{n}-g04", "canonical", n, 2, 2, 0.4, "gaussian"),
                Scenario(f"canonical-n{n}-t3", "canonical", n, 2, 2, 0.4, "t3"),
                Scenario(
                    f"canonical-n{n}-cont", "canonical", n, 2, 2, 0.4, "contaminated"
                ),
                Scenario(
                    f"informative-n{n}-g04",
                    "informative",
                    n,
                    2,
                    2,
                    0.4,
                    "gaussian",
                    informative_size=True,
                ),
                Scenario(
                    f"informative-n{n}-cont",
                    "informative",
                    n,
                    2,
                    2,
                    0.4,
                    "contaminated",
                    informative_size=True,
                ),
                Scenario(
                    f"sparse10-n{n}-g04", "sparse", n, 10, 3, 0.4, "gaussian"
                ),
                Scenario(
                    f"sparse10-n{n}-cont",
                    "sparse",
                    n,
                    10,
                    3,
                    0.4,
                    "contaminated",
                ),
            ]
        )
    return scenarios


def supplementary_scenarios() -> list[Scenario]:
    return [
        Scenario(
            "rank2-n100-d025",
            "misspecified",
            100,
            2,
            2,
            0.4,
            "gaussian",
            rank_two_strength=0.25,
        ),
        Scenario(
            "rank2-n100-d050",
            "misspecified",
            100,
            2,
            2,
            0.4,
            "gaussian",
            rank_two_strength=0.50,
        ),
    ]

"""Numerical checks of the two places where the theory and the estimator differ.

Section~3 of the paper analyses an idealised estimator.  Two of its departures
from the implemented one are substantive, and this script measures both instead
of leaving them as caveats.

*Threshold rule.*  The theory fixes the Huber threshold; the implementation sets
it from a median absolute deviation of the same subjects it then fits.  The
split-sample variant covered by Corollary~3 computes the scale on one half of
the training subjects and fits on the other.  Comparing the two rules **on the
same fitted half** isolates the rule from the sample size it is run at, which a
comparison against the full-sample fit would confound.

*Exact separability.*  The model restricts each component to a product.  The
approximate-separability proposition says the estimator then targets the best
separable approximation of the truth, with the separability defect entering as
an explicit bias that no rank-one procedure can remove.  The second check
generates components of rank two with a controllable defect and measures the
distance of the delivered component to the truth and to that best separable
approximation separately.

Both checks reuse the locked tuning of the formal benchmark; neither refits any
number that appears in the main tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID_SIZE = 201
DEFAULT_SEED = 20260213


# ---------------------------------------------------------------------------
# Shared fitting helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedCurves:
    """Delivered baseline and identified factors on the registered [0,1] grids."""

    baseline: np.ndarray
    beta: tuple[np.ndarray | None, ...]
    phi: tuple[np.ndarray | None, ...]
    delta: float
    converged: bool


def _locked_tuning() -> dict[str, float | int]:
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


def _design(time, covariates, response, subject, basis, domain_time):
    from src.trace_vcam import VCAMDesign

    low, high = domain_time
    scaled_time = (np.asarray(time, dtype=float) - low) / (high - low)
    return VCAMDesign.from_arrays(
        scaled_time, np.asarray(covariates, dtype=float), response, subject, basis
    )


def _fit(design, basis, *, delta: float | None, tuning: Mapping[str, object]) -> FittedCurves:
    """One TRACE fit at the locked penalty, with either threshold rule."""

    from src.trace_vcam import fit_trace_vcam, practical_huber_threshold, trace_lambda_max

    if delta is None:
        resolved, _ = practical_huber_threshold(design, multiplier=1.345)
    else:
        resolved = float(delta)
    lambda_max, _ = trace_lambda_max(design, resolved)
    penalty = float(tuning["lambda_ratio"]) * lambda_max
    common = {
        "max_iter": int(tuning["max_iter"]),
        "tolerance": float(tuning["tolerance"]),
        "mu": float(tuning["roughness"]),
        "postfit_max_iter": int(tuning["postfit_max_iter"]),
        "postfit_tolerance": float(tuning["postfit_tolerance"]),
    }
    if delta is None:
        fit = fit_trace_vcam(
            design, penalty, delta=None, threshold_mode="mad", huber_multiplier=1.345, **common
        )
    else:
        fit = fit_trace_vcam(
            design, penalty, delta=resolved, threshold_mode="fixed", **common
        )

    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    time_basis = basis.transform_time(grid)
    covariate_basis = basis.transform_covariate(grid)
    baseline = time_basis @ fit.gamma
    beta: list[np.ndarray | None] = []
    phi: list[np.ndarray | None] = []
    time_factors = fit.identified_time_factors or []
    covariate_factors = fit.identified_covariate_factors or []
    for index in range(len(fit.matrices)):
        b = time_factors[index] if index < len(time_factors) else None
        f = covariate_factors[index] if index < len(covariate_factors) else None
        beta.append(None if b is None else time_basis @ b)
        phi.append(None if f is None else covariate_basis @ f)
    return FittedCurves(
        baseline=baseline,
        beta=tuple(beta),
        phi=tuple(phi),
        delta=float(resolved),
        converged=bool(fit.converged),
    )


def _predict(curves: FittedCurves, time, covariates, domain_time) -> np.ndarray:
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


def _subject_halves(subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split complete subjects in two, never within a subject."""

    unique = np.unique(subjects)
    order = np.random.default_rng(seed).permutation(unique.size)
    cut = unique.size // 2
    return unique[order[:cut]], unique[order[cut:]]


# ---------------------------------------------------------------------------
# Check 1: same-sample against split-sample Huber threshold
# ---------------------------------------------------------------------------


def _surface_error(
    curves: FittedCurves,
    truth,
    time_grid: np.ndarray,
    covariate_grid: np.ndarray,
) -> float:
    total = 0.0
    for index, active in enumerate(truth.active):
        if not active:
            continue
        beta_true = truth.beta[index](time_grid)
        phi_true = truth.phi[index](covariate_grid)
        beta_hat = curves.beta[index]
        phi_hat = curves.phi[index]
        if beta_hat is None or phi_hat is None:
            estimate = np.zeros((time_grid.size, covariate_grid.size))
        else:
            estimate = np.outer(beta_hat, phi_hat)
        total += float(
            np.mean((estimate - np.outer(beta_true, phi_true)) ** 2)
        )
    return total


def _threshold_replication(job: tuple[int, int, str, float]) -> dict[str, object]:
    seed, n_subjects, distribution, sigma = job
    from experiments.dgp import generate_zzw2020, subject_split
    from src.trace_vcam import OrthonormalSplineBasis, practical_huber_threshold

    tuning = _locked_tuning()
    data = generate_zzw2020(
        seed, n_subjects=n_subjects, sigma=sigma, error_distribution=distribution
    )
    train_rows, test_rows = subject_split(data.subject, seed=seed)
    basis = OrthonormalSplineBasis.create(int(tuning["q_time"]), int(tuning["q_covariate"]))
    time_grid = np.linspace(*data.domain_time, GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)

    def build(rows: np.ndarray):
        return _design(
            data.time[rows],
            data.covariates[rows],
            data.response[rows],
            data.subject[rows],
            basis,
            data.domain_time,
        )

    scale_subjects, fit_subjects = _subject_halves(data.subject[train_rows], seed + 7919)
    scale_rows = train_rows[np.isin(data.subject[train_rows], scale_subjects)]
    fit_rows = train_rows[np.isin(data.subject[train_rows], fit_subjects)]

    independent_delta, _ = practical_huber_threshold(build(scale_rows), multiplier=1.345)
    designs = {
        "same-sample, half": (build(fit_rows), None),
        "split-sample, half": (build(fit_rows), independent_delta),
        "same-sample, full": (build(train_rows), None),
    }
    row: dict[str, object] = {
        "seed": seed,
        "n_subjects": n_subjects,
        "error_law": distribution,
    }
    for label, (design, delta) in designs.items():
        curves = _fit(design, basis, delta=delta, tuning=tuning)
        row[f"surface::{label}"] = _surface_error(curves, data.truth, time_grid, covariate_grid)
        row[f"mspe::{label}"] = float(
            np.mean(
                (
                    _predict(
                        curves,
                        data.time[test_rows],
                        data.covariates[test_rows],
                        data.domain_time,
                    )
                    - data.conditional_mean[test_rows]
                )
                ** 2
            )
        )
        row[f"delta::{label}"] = curves.delta
        row[f"converged::{label}"] = curves.converged
    return row


# ---------------------------------------------------------------------------
# Check 2: components that are not exactly separable
# ---------------------------------------------------------------------------


def _rank_two_dataset(seed: int, *, n_subjects: int, defect: float, distribution: str):
    """Example 2 covariates, with a first component of rank two.

    The second factor pair added to block one is taken from a different design
    (the cubic and sine pair of Example 1, mapped to this time domain) so that
    the extra direction is genuinely new and not a copy of another block.
    """

    from experiments.dgp import (
        _assemble,
        _draw_errors,
        _fourier_random_effect,
        _gaussian_copula_uniforms,
        zzw2020_truth,
    )

    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(2, 11, size=n_subjects)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time = rng.uniform(0.0, 2.0, size=subject.size)
    u = _gaussian_copula_uniforms(rng, n_subjects, np.array([[1.0, 0.6], [0.6, 1.0]]))
    v = _gaussian_copula_uniforms(rng, n_subjects, np.array([[1.0, 0.5], [0.5, 1.0]]))
    scaled_time = 0.5 * time
    covariates = np.column_stack(
        [
            0.5 * u[subject, 0] * scaled_time**0.5 + 0.5 * v[subject, 0],
            0.5 * u[subject, 1] * scaled_time ** (1.0 / 3.0) + 0.5 * v[subject, 1],
        ]
    )
    base = zzw2020_truth()
    extra_beta, extra_phi = _extra_pair()
    surface = _rank_two_surface(base, extra_beta, extra_phi, defect)
    components = surface(time, covariates[:, 0]) + base.beta[1](time) * base.phi[1](
        covariates[:, 1]
    )
    conditional_mean = base.beta0(time) + components
    random_effect = _fourier_random_effect(rng, time, subject, n_subjects)
    errors = _draw_errors(rng, time.size, 0.4, distribution)
    from experiments.dgp import PublishedDataset

    return PublishedDataset(
        time=time,
        covariates=covariates,
        response=conditional_mean + random_effect + errors,
        subject=subject,
        conditional_mean=conditional_mean,
        truth=base,
        design_id=f"RankTwo-c{defect:g}-{distribution}",
        provenance="own-design",
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
        domain_covariates=((0.0, 1.0), (0.0, 1.0)),
    )


def _extra_pair() -> tuple[Callable, Callable]:
    """A second factor pair, on the identification scale of the registered ones.

    The time factor averages to one over \\([0,2]\\) and the covariate factor
    integrates to zero over \\([0,1]\\), so the coefficient multiplying this pair
    is directly comparable with the amplitude of the separable term it is added
    to: a coefficient of one makes the two directions of the component equally
    strong.
    """

    def beta(t):
        t = np.asarray(t, dtype=float)
        return 3.0 * (1.0 - t / 2.0) ** 2

    def phi(z):
        z = np.asarray(z, dtype=float)
        return 4.0 * z**3 - 1.0

    return beta, phi


def _rank_two_surface(base, extra_beta, extra_phi, defect: float) -> Callable:
    """Pointwise evaluation at matched one-dimensional time and covariate arrays."""

    def surface(t, z):
        return base.beta[0](t) * base.phi[0](z) + defect * extra_beta(t) * extra_phi(z)

    return surface


def _rank_two_grid(
    base, extra_beta, extra_phi, defect: float, times: np.ndarray, covariates: np.ndarray
) -> np.ndarray:
    """The same surface on a product grid.

    The registered truth functions take one-dimensional arguments, so the two
    rank-one terms are formed as outer products rather than by broadcasting.
    """

    return np.outer(base.beta[0](times), base.phi[0](covariates)) + defect * np.outer(
        extra_beta(times), extra_phi(covariates)
    )


def _best_separable(values: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Best rank-one approximation of a surface sampled on a product grid.

    Both marginal grids are equally spaced, so the Frobenius geometry of the
    sampled matrix is the discretised \\(L^2\\) geometry up to one common
    factor, and the leading singular pair is the best separable approximation.
    """

    left, singular, right = np.linalg.svd(values, full_matrices=False)
    best = singular[0] * np.outer(left[:, 0], right[0])
    defect = float(np.sqrt(np.mean((values - best) ** 2)))
    return best, defect, singular


def _separability_replication(job: tuple[int, int, float, str]) -> dict[str, object]:
    seed, n_subjects, defect, distribution = job
    from experiments.dgp import subject_split, zzw2020_truth
    from src.trace_vcam import OrthonormalSplineBasis

    tuning = _locked_tuning()
    data = _rank_two_dataset(
        seed, n_subjects=n_subjects, defect=defect, distribution=distribution
    )
    train_rows, test_rows = subject_split(data.subject, seed=seed)
    basis = OrthonormalSplineBasis.create(int(tuning["q_time"]), int(tuning["q_covariate"]))
    design = _design(
        data.time[train_rows],
        data.covariates[train_rows],
        data.response[train_rows],
        data.subject[train_rows],
        basis,
        data.domain_time,
    )
    curves = _fit(design, basis, delta=None, tuning=tuning)

    time_grid = np.linspace(*data.domain_time, GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    base = zzw2020_truth()
    extra_beta, extra_phi = _extra_pair()
    truth_surface = _rank_two_grid(
        base, extra_beta, extra_phi, defect, time_grid, covariate_grid
    )
    separable, zeta, singular = _best_separable(truth_surface)
    if curves.beta[0] is None or curves.phi[0] is None:
        estimate = np.zeros_like(truth_surface)
    else:
        estimate = np.outer(curves.beta[0], curves.phi[0])
    return {
        "seed": seed,
        "n_subjects": n_subjects,
        "defect_coefficient": defect,
        "error_law": distribution,
        "zeta": zeta,
        # The leading direction of a component is only well separated when the
        # second singular value of its surface is small relative to the first;
        # this ratio is the quantity the gap condition of the approximate
        # separability result is stated in.
        "second_over_first_singular_value": float(singular[1] / singular[0]),
        "to_truth": float(np.sqrt(np.mean((estimate - truth_surface) ** 2))),
        "to_best_separable": float(np.sqrt(np.mean((estimate - separable) ** 2))),
        "mspe": float(
            np.mean(
                (
                    _predict(
                        curves,
                        data.time[test_rows],
                        data.covariates[test_rows],
                        data.domain_time,
                    )
                    - data.conditional_mean[test_rows]
                )
                ** 2
            )
        ),
        "converged": curves.converged,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


ERROR_LAW_LABEL = {
    "gaussian": r"Normal, $\sigma=0.4$",
    "hhy-mixed-normal": "Contaminated mixture",
    "hhy-t2": r"Scaled $t_2$",
}


def _threshold_table(summary: Sequence[Mapping[str, object]], replications: int) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Same-sample against split-sample Huber threshold. Both columns "
        r"of each pair are fitted on the same half of the training subjects, so "
        r"that the threshold rule is separated from the number of subjects it is "
        r"run at; the last column repeats the full-training-set fit the paper "
        r"reports. Entries are medians over "
        f"{replications}"
        r" replications of the aggregated component-surface error and of the "
        r"held-out noise-free prediction error.}",
        r"\label{tab:supp-threshold-rule}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$\sum_k g_k$} & \multicolumn{3}{c}{MSPE} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Error law & $n$ & same, half & split, half & ratio & same, half & "
        r"split, half & ratio \\",
        r"\midrule",
    ]
    for entry in summary:
        lines.append(
            " & ".join(
                [
                    ERROR_LAW_LABEL.get(str(entry["error_law"]), str(entry["error_law"])),
                    str(entry["n_subjects"]),
                    f"{entry['surface::same-sample, half']:.4f}",
                    f"{entry['surface::split-sample, half']:.4f}",
                    f"{entry['surface_ratio_split_over_same']:.3f}",
                    f"{entry['mspe::same-sample, half']:.4f}",
                    f"{entry['mspe::split-sample, half']:.4f}",
                    f"{entry['mspe_ratio_split_over_same']:.3f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _separability_table(summary: Sequence[Mapping[str, object]], replications: int) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Components that are not exactly separable. The first covariate "
        r"block of the Example 2 design carries a second factor pair with "
        r"coefficient $c$, so that $c=0$ is the registered separable design and "
        r"$c=1$ makes the two directions of the component equally strong. "
        r"$\sigma_2/\sigma_1$ is the spectral ratio of the true component surface "
        r"and $\zeta$ its separability defect, both deterministic given $c$; the "
        r"two distance columns are root mean squared distances of the delivered "
        r"component to the truth and to the best separable approximation of the "
        r"truth, and MSPE is the held-out noise-free prediction error. Entries "
        r"are medians over "
        f"{replications}"
        r" replications at $n=100$ under the contaminated mixture.}",
        r"\label{tab:supp-separability}",
        r"\footnotesize",
        r"\begin{tabular}{@{}cccccc@{}}",
        r"\toprule",
        r"$c$ & $\sigma_2/\sigma_1$ & $\zeta$ & to the truth & "
        r"to the best separable & MSPE \\",
        r"\midrule",
    ]
    for entry in summary:
        lines.append(
            " & ".join(
                [
                    f"{entry['defect_coefficient']:.2f}",
                    f"{entry['second_over_first_singular_value']:.3f}",
                    f"{entry['zeta']:.3f}",
                    f"{entry['to_truth']:.3f}",
                    f"{entry['to_best_separable']:.3f}",
                    f"{entry['mspe']:.4f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _summarise(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = np.array(
            [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        )
        if values.size:
            out[key] = float(np.median(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "theory_gap")
    parser.add_argument("--tables", type=Path, default=ROOT / "manuscript" / "tables")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    threshold_jobs = [
        (args.seed + 1_000_000 * index + replication, n_subjects, law, sigma)
        for index, (n_subjects, law, sigma) in enumerate(
            [
                (50, "hhy-mixed-normal", 0.4),
                (100, "hhy-mixed-normal", 0.4),
                (200, "hhy-mixed-normal", 0.4),
                (100, "gaussian", 0.4),
                (100, "hhy-t2", 0.4),
            ]
        )
        for replication in range(args.replications)
    ]
    separability_jobs = [
        (args.seed + 7_000_000 * index + replication, 100, defect, "hhy-mixed-normal")
        for index, defect in enumerate([0.0, 0.25, 0.5, 1.0])
        for replication in range(args.replications)
    ]

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        threshold_rows = list(pool.map(_threshold_replication, threshold_jobs, chunksize=1))
        separability_rows = list(
            pool.map(_separability_replication, separability_jobs, chunksize=1)
        )

    labels = ["same-sample, half", "split-sample, half", "same-sample, full"]
    threshold_summary = []
    for n_subjects, law in sorted({(row["n_subjects"], row["error_law"]) for row in threshold_rows}):
        block = [
            row
            for row in threshold_rows
            if row["n_subjects"] == n_subjects and row["error_law"] == law
        ]
        entry: dict[str, object] = {
            "n_subjects": n_subjects,
            "error_law": law,
            "replications": len(block),
        }
        entry.update(
            _summarise(
                block,
                [f"{metric}::{label}" for metric in ("surface", "mspe", "delta") for label in labels],
            )
        )
        entry["surface_ratio_split_over_same"] = (
            entry["surface::split-sample, half"] / entry["surface::same-sample, half"]
        )
        entry["mspe_ratio_split_over_same"] = (
            entry["mspe::split-sample, half"] / entry["mspe::same-sample, half"]
        )
        threshold_summary.append(entry)

    separability_summary = []
    for defect in sorted({row["defect_coefficient"] for row in separability_rows}):
        block = [row for row in separability_rows if row["defect_coefficient"] == defect]
        entry = {"defect_coefficient": defect, "replications": len(block)}
        entry.update(
            _summarise(
                block,
                [
                    "zeta",
                    "second_over_first_singular_value",
                    "to_truth",
                    "to_best_separable",
                    "mspe",
                ],
            )
        )
        separability_summary.append(entry)

    payload = {
        "schema_version": "theory-gap-1",
        "seed": args.seed,
        "replications": args.replications,
        "tuning": _locked_tuning(),
        "threshold_rule": {"rows": threshold_rows, "summary": threshold_summary},
        "approximate_separability": {
            "rows": separability_rows,
            "summary": separability_summary,
        },
    }
    (args.output / "theory_gap_checks.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    args.tables.mkdir(parents=True, exist_ok=True)
    (args.tables / "theory_gap_threshold.tex").write_text(
        _threshold_table(threshold_summary, args.replications), encoding="utf-8"
    )
    (args.tables / "theory_gap_separability.tex").write_text(
        _separability_table(separability_summary, args.replications), encoding="utf-8"
    )
    print(json.dumps({"threshold": threshold_summary}, indent=2))
    print(json.dumps({"separability": separability_summary}, indent=2))
    worst = max(
        max(
            abs(entry["surface_ratio_split_over_same"] - 1.0),
            abs(entry["mspe_ratio_split_over_same"] - 1.0),
        )
        for entry in threshold_summary
    )
    print(f"largest split-vs-same departure from parity: {worst:.3f}")


if __name__ == "__main__":
    main()

"""Revision experiments that vary something the benchmark protocol holds fixed.

Four of the points raised in review ask for a numerical answer that the formal
benchmark cannot give, because each of them moves a quantity the benchmark locks
before fitting.  They are collected here, behind one subcommand each, and all of
them fit the proposed estimator directly:

``normalisation``
    What happens to the reported factors when the time factor of an active
    component has a mean approaching zero, so that the integral-one convention
    of the paper approaches a division by zero.  The component surface, the
    prediction, and the factors under an alternative :math:`L^2` convention are
    measured alongside it, which separates a property of the model class from a
    property of the reporting convention.

``selection-path``
    Block-level recovery along the penalty path of the block-sparse design:
    retained fractions, true and false positive rates, and the norms of active
    and inactive blocks, at the locked penalty and around it.

``tuning-sensitivity``
    The estimator over a grid of penalty and roughness levels, together with the
    per-replication oracle choice, so that the accuracy attributable to the pilot
    calibration protocol can be bounded.

``rank-diagnostic``
    A practitioner diagnostic for the rank-one restriction, calibrated on
    components whose second direction is grown from nothing to the strength of
    the first.

None of these refits a number that appears in a formal benchmark table.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.review_common import (  # noqa: E402
    GRID_SIZE,
    build_design,
    domain_average_squared_error,
    fit_delivered,
    locked_tuning,
    predict,
    predict_on_design,
    surface_error,
)

DEFAULT_SEED = 20260816

ERROR_LAWS = ("gaussian", "hhy-mixed-normal", "hhy-t2")
ERROR_LAW_LABEL = {
    "gaussian": r"Normal, $\sigma=0.4$",
    "hhy-mixed-normal": "Contaminated mixture",
    "hhy-t2": r"Scaled $t_2$",
}


def _write(rows: Sequence[dict], target: Path) -> None:
    if not rows:
        raise SystemExit(f"no rows produced for {target}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"wrote {target} ({len(rows)} rows)")


def _run(task: Callable, jobs: Iterable, workers: int, label: str) -> list[dict]:
    jobs = list(jobs)
    started = time.perf_counter()
    rows: list[dict] = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            rows.extend(_as_rows(task(job)))
            if index % 25 == 0 or index == len(jobs):
                print(
                    f"{label} {index}/{len(jobs)} ({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(task, jobs, chunksize=1), start=1):
            rows.extend(_as_rows(result))
            if index % 25 == 0 or index == len(jobs):
                print(
                    f"{label} {index}/{len(jobs)} ({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
    return rows


def _as_rows(result) -> list[dict]:
    if isinstance(result, dict):
        return [result]
    return list(result)


# ---------------------------------------------------------------------------
# 1. The normalisation margin
# ---------------------------------------------------------------------------

#: Means of the first component's time factor.  One is the registered design;
#: the last is a component whose time factor integrates to exactly zero, for
#: which the integral-one convention of the paper has no solution at all.
NORMALISATION_MEANS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.0)


def _shifted_truth(mean: float):
    """Example 2, with the first time factor shifted to a prescribed mean.

    The registered factor averages one over the time domain.  Adding a constant
    moves that average without changing the factor's shape, its roughness, or
    the :math:`L^2` size of the component it multiplies, so the family sweeps the
    normalisation margin while leaving the estimand a perfectly ordinary
    separable effect.  The covariate factor is untouched, so the component
    surface is a fixed shape times a time factor whose mean is the swept
    quantity.
    """

    from experiments.dgp import Truth, zzw2020_truth

    base = zzw2020_truth()
    shift = float(mean) - 1.0

    def beta_shifted(t, _base=base.beta[0], _shift=shift):
        return _base(t) + _shift

    return Truth(
        beta0=base.beta0,
        beta=(beta_shifted, base.beta[1]),
        phi=base.phi,
        active=base.active,
    )


def _normalisation_dataset(seed: int, *, n_subjects: int, mean: float, distribution: str):
    from experiments.dgp import (
        _assemble,
        _draw_errors,
        _fourier_random_effect,
        _gaussian_copula_uniforms,
    )

    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(2, 11, size=n_subjects)
    subject = np.repeat(np.arange(n_subjects, dtype=np.int64), cluster_sizes)
    time_values = rng.uniform(0.0, 2.0, size=subject.size)
    u = _gaussian_copula_uniforms(rng, n_subjects, np.array([[1.0, 0.6], [0.6, 1.0]]))
    v = _gaussian_copula_uniforms(rng, n_subjects, np.array([[1.0, 0.5], [0.5, 1.0]]))
    scaled = 0.5 * time_values
    covariates = np.column_stack(
        [
            0.5 * u[subject, 0] * scaled**0.5 + 0.5 * v[subject, 0],
            0.5 * u[subject, 1] * scaled ** (1.0 / 3.0) + 0.5 * v[subject, 1],
        ]
    )
    truth = _shifted_truth(mean)
    return _assemble(
        time=time_values,
        covariates=covariates,
        subject=subject,
        truth=truth,
        random_effect=_fourier_random_effect(rng, time_values, subject, n_subjects),
        errors=_draw_errors(rng, time_values.size, 0.4, distribution),
        design_id=f"NormalisationMargin-mean{mean:g}-{distribution}",
        provenance="own-design",
        time_invariant_covariates=False,
        domain_time=(0.0, 2.0),
    )


def _l2_pair(beta: np.ndarray, phi: np.ndarray, grid: np.ndarray):
    """Rewrite one factor pair with a unit-:math:`L^2` time factor."""

    length = float(grid[-1] - grid[0])
    norm = float(np.sqrt(np.trapezoid(beta**2, grid) / length))
    if norm <= 0.0:
        return beta, phi
    return beta / norm, phi * norm


def _paired_error(
    beta_hat: np.ndarray,
    phi_hat: np.ndarray,
    beta_ref: np.ndarray,
    phi_ref: np.ndarray,
    time_grid: np.ndarray,
    covariate_grid: np.ndarray,
) -> tuple[float, float]:
    """Factor errors of a pair that is identified only up to a joint sign.

    Normalising the time factor in :math:`L^2` fixes its size but not its
    direction, because a rank-one component is unchanged when both factors change
    sign.  Any sign convention is therefore a choice of representative, and the
    error that describes the estimate rather than the convention is the smaller
    of the two.  This is the same joint sign convention the factor-recovery
    corollary states.
    """

    best: tuple[float, float] | None = None
    for sign in (1.0, -1.0):
        beta_error = domain_average_squared_error(time_grid, sign * beta_hat - beta_ref)
        phi_error = domain_average_squared_error(
            covariate_grid, sign * phi_hat - phi_ref
        )
        if best is None or beta_error + phi_error < best[0] + best[1]:
            best = (beta_error, phi_error)
    assert best is not None
    return best


def _normalisation_replication(job) -> dict:
    seed, n_subjects, mean, distribution = job
    from experiments.dgp import subject_split
    from src.trace_vcam import OrthonormalSplineBasis

    tuning = locked_tuning()
    data = _normalisation_dataset(
        seed, n_subjects=n_subjects, mean=mean, distribution=distribution
    )
    train, test = subject_split(data.subject, seed=seed)
    basis = OrthonormalSplineBasis.create(
        int(tuning["q_time"]), int(tuning["q_covariate"])
    )
    design = build_design(
        data.time[train],
        data.covariates[train],
        data.response[train],
        data.subject[train],
        basis,
        data.domain_time,
    )
    curves = fit_delivered(design, basis, tuning=tuning)

    time_grid = np.linspace(*data.domain_time, GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    truth = data.truth
    beta_true = truth.beta[0](time_grid)
    phi_true = truth.phi[0](covariate_grid)

    row: dict[str, object] = {
        "seed": seed,
        "n_subjects": n_subjects,
        "time_factor_mean": mean,
        "error_law": distribution,
        "converged": curves.converged,
        "normalisation_margin": curves.time_integral_margin[0],
    }
    # The component surface does not depend on how the two factors are
    # normalised, so it measures the estimand rather than the convention.
    row["surface_block1"] = surface_error(
        curves, truth, time_grid, covariate_grid, blocks=[0]
    )
    row["surface_total"] = surface_error(curves, truth, time_grid, covariate_grid)
    row["mspe"] = float(
        np.mean(
            (
                predict(
                    curves, data.time[test], data.covariates[test], data.domain_time
                )
                - data.conditional_mean[test]
            )
            ** 2
        )
    )

    # Factors as the paper reports them.  The truth is put on the same scale, so
    # a divergence here is the convention failing, not the fit.
    integral = float(np.trapezoid(beta_true, time_grid) / (time_grid[-1] - time_grid[0]))
    beta_hat, phi_hat = curves.beta[0], curves.phi[0]
    if abs(integral) > 0.0 and beta_hat is not None and phi_hat is not None:
        beta_ref, phi_ref = beta_true / integral, phi_true * integral
        row["beta_ise_integral"] = domain_average_squared_error(
            time_grid, beta_hat - beta_ref
        )
        row["phi_ise_integral"] = domain_average_squared_error(
            covariate_grid, phi_hat - phi_ref
        )
        # The two factors of one component move in opposite directions as the
        # margin closes: the time factor is divided by a number approaching zero
        # and the covariate factor is multiplied by it.  Reporting both sizes
        # shows that the pair separates while their product does not.
        row["beta_sup_integral"] = float(np.max(np.abs(beta_hat)))
        row["phi_norm_integral"] = float(
            np.sqrt(np.mean(phi_hat**2))
        )
        row["beta_sup_integral_true"] = float(np.max(np.abs(beta_ref)))
    else:
        row["beta_ise_integral"] = float("nan")
        row["phi_ise_integral"] = float("nan")
        row["beta_sup_integral"] = float("nan")
        row["phi_norm_integral"] = float("nan")
        row["beta_sup_integral_true"] = float("nan")

    # The same fitted component under a convention that is always defined.
    beta_ref_l2, phi_ref_l2 = _l2_pair(beta_true, phi_true, time_grid)
    beta_l2, phi_l2 = curves.beta_l2[0], curves.phi_l2[0]
    if beta_l2 is None or phi_l2 is None:
        row["beta_ise_l2"] = domain_average_squared_error(time_grid, beta_ref_l2)
        row["phi_ise_l2"] = domain_average_squared_error(covariate_grid, phi_ref_l2)
    else:
        row["beta_ise_l2"], row["phi_ise_l2"] = _paired_error(
            beta_l2, phi_l2, beta_ref_l2, phi_ref_l2, time_grid, covariate_grid
        )
    return row


def command_normalisation(args) -> None:
    jobs = [
        (DEFAULT_SEED + 1000 * replicate + 7, args.subjects, mean, law)
        for law in args.laws
        for mean in NORMALISATION_MEANS
        for replicate in range(args.replications)
    ]
    rows = _run(_normalisation_replication, jobs, args.jobs, "normalisation")
    _write(rows, args.output / "normalisation_margin.csv")


# ---------------------------------------------------------------------------
# 2. Block selection along the penalty path
# ---------------------------------------------------------------------------

PENALTY_PATH = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50)

#: A block is called retained at a relative level when its coefficient matrix
#: carries at least this share of the largest block's Frobenius norm.  The
#: estimator's own selection flag uses an absolute numerical tolerance and is
#: reported separately; the relative rule is what a practitioner reading a fitted
#: model would apply, and the two answer different questions.
RELATIVE_RETENTION = 0.01


def _selection_replication(job) -> list[dict]:
    seed, n_subjects, distribution, n_covariates, n_active = job
    from experiments.dgp import generate_block_sparse, subject_split
    from src.trace_vcam import OrthonormalSplineBasis

    tuning = locked_tuning()
    data = generate_block_sparse(
        seed,
        n_subjects=n_subjects,
        sigma=0.4,
        error_distribution=distribution,
        n_covariates=n_covariates,
        n_active=n_active,
    )
    train, test = subject_split(data.subject, seed=seed)
    basis = OrthonormalSplineBasis.create(
        int(tuning["q_time"]), int(tuning["q_covariate"])
    )
    design = build_design(
        data.time[train],
        data.covariates[train],
        data.response[train],
        data.subject[train],
        basis,
        data.domain_time,
    )
    time_grid = np.linspace(*data.domain_time, GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    active = np.asarray(data.truth.active, dtype=bool)

    rows: list[dict] = []
    for ratio in PENALTY_PATH:
        curves = fit_delivered(design, basis, tuning=tuning, lambda_ratio=ratio)
        frobenius = np.array(
            [float(np.linalg.norm(block, "fro")) for block in curves.blocks]
        )
        nuclear = np.array([float(np.sum(values)) for values in curves.singular_values])
        widest = frobenius.max() if frobenius.size else 0.0
        relative = frobenius / widest if widest > 0.0 else np.zeros_like(frobenius)
        retained_numeric = curves.selected
        retained_relative = relative >= RELATIVE_RETENTION
        row: dict[str, object] = {
            "seed": seed,
            "n_subjects": n_subjects,
            "error_law": distribution,
            "n_covariates": n_covariates,
            "n_active": n_active,
            "lambda_ratio": ratio,
            "converged": curves.converged,
            "surface_total": surface_error(
                curves, data.truth, time_grid, covariate_grid
            ),
            "mspe": float(
                np.mean(
                    (
                        predict(
                            curves,
                            data.time[test],
                            data.covariates[test],
                            data.domain_time,
                        )
                        - data.conditional_mean[test]
                    )
                    ** 2
                )
            ),
            "model_size_numeric": int(retained_numeric.sum()),
            "model_size_relative": int(retained_relative.sum()),
            "tpr_numeric": float(np.mean(retained_numeric[active])),
            "fpr_numeric": float(np.mean(retained_numeric[~active]))
            if np.any(~active)
            else float("nan"),
            "tpr_relative": float(np.mean(retained_relative[active])),
            "fpr_relative": float(np.mean(retained_relative[~active]))
            if np.any(~active)
            else float("nan"),
            "frobenius_active": float(np.mean(frobenius[active])),
            "frobenius_inactive": float(np.mean(frobenius[~active]))
            if np.any(~active)
            else float("nan"),
            "nuclear_active": float(np.mean(nuclear[active])),
            "nuclear_inactive": float(np.mean(nuclear[~active]))
            if np.any(~active)
            else float("nan"),
            "relative_inactive_max": float(np.max(relative[~active]))
            if np.any(~active)
            else float("nan"),
        }
        rows.append(row)
    return rows


def command_selection_path(args) -> None:
    jobs = [
        (DEFAULT_SEED + 1000 * replicate + 13, args.subjects, law, 6, 2)
        for law in args.laws
        for replicate in range(args.replications)
    ]
    rows = _run(_selection_replication, jobs, args.jobs, "selection-path")
    _write(rows, args.output / "selection_path.csv")


# ---------------------------------------------------------------------------
# 3. Tuning sensitivity
# ---------------------------------------------------------------------------

TUNING_LAMBDAS = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
TUNING_ROUGHNESS = (0.0, 0.01, 0.05, 0.20)


def _tuning_replication(job) -> list[dict]:
    seed, design_name, n_subjects, distribution = job
    from experiments.dgp import generate_block_sparse, generate_zzw2020, subject_split
    from src.trace_vcam import OrthonormalSplineBasis

    tuning = locked_tuning()
    if design_name == "example2":
        data = generate_zzw2020(
            seed, n_subjects=n_subjects, sigma=0.4, error_distribution=distribution
        )
    else:
        data = generate_block_sparse(
            seed,
            n_subjects=n_subjects,
            sigma=0.4,
            error_distribution=distribution,
            n_covariates=6,
            n_active=2,
        )
    train, test = subject_split(data.subject, seed=seed)
    basis = OrthonormalSplineBasis.create(
        int(tuning["q_time"]), int(tuning["q_covariate"])
    )
    design = build_design(
        data.time[train],
        data.covariates[train],
        data.response[train],
        data.subject[train],
        basis,
        data.domain_time,
    )
    time_grid = np.linspace(*data.domain_time, GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)

    rows: list[dict] = []
    for ratio in TUNING_LAMBDAS:
        for mu in TUNING_ROUGHNESS:
            curves = fit_delivered(
                design, basis, tuning=tuning, lambda_ratio=ratio, roughness=mu
            )
            rows.append(
                {
                    "seed": seed,
                    "design": design_name,
                    "n_subjects": n_subjects,
                    "error_law": distribution,
                    "lambda_ratio": ratio,
                    "roughness": mu,
                    "converged": curves.converged,
                    "surface_total": surface_error(
                        curves, data.truth, time_grid, covariate_grid
                    ),
                    "mspe": float(
                        np.mean(
                            (
                                predict(
                                    curves,
                                    data.time[test],
                                    data.covariates[test],
                                    data.domain_time,
                                )
                                - data.conditional_mean[test]
                            )
                            ** 2
                        )
                    ),
                    "runtime_seconds": curves.runtime_seconds,
                }
            )
    return rows


def command_tuning_sensitivity(args) -> None:
    jobs = [
        (DEFAULT_SEED + 1000 * replicate + 29, design, subjects, law)
        for design, subjects in (("example2", 50), ("example3", 100))
        for law in args.laws
        for replicate in range(args.replications)
    ]
    rows = _run(_tuning_replication, jobs, args.jobs, "tuning-sensitivity")
    _write(rows, args.output / "tuning_sensitivity.csv")


# ---------------------------------------------------------------------------
# 4. A practitioner diagnostic for the rank-one restriction
# ---------------------------------------------------------------------------

#: Strength of the second direction added to the first component, on the
#: identification scale of the first.  The registered separability check of the
#: supplement stops at one, where the two directions carry equal amplitude; the
#: diagnostic needs the range where the second direction actually dominates, so
#: that the calibration curve covers the regime it is meant to detect.
DIAGNOSTIC_DEFECTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def spectral_summary(curves, block: int | None = None) -> dict[str, float]:
    """Singular-value read-out of the pilot blocks.

    The perturbation bound behind factor recovery is stated in the ratio of the
    second singular value of a component to its first, so that ratio is the
    natural thing to look at.  On estimated blocks it is biased upwards, because
    the trailing singular values of a noisy matrix are positive even when the
    truth is exactly rank one; the calibration experiment measures that bias
    directly, which is why the ratio is reported rather than tested.
    """

    ratios: list[float] = []
    energies: list[float] = []
    wanted = (
        range(len(curves.singular_values)) if block is None else [block]
    )
    for index in wanted:
        values = curves.singular_values[index]
        if not curves.selected[index] or values[0] <= 0.0:
            continue
        ratios.append(float(values[1] / values[0]) if values.size > 1 else 0.0)
        energies.append(float(values[0] ** 2 / np.sum(values**2)))
    prefix = "" if block is None else f"block{block + 1}_"
    return {
        f"{prefix}spectral_ratio_max": float(max(ratios)) if ratios else float("nan"),
        f"{prefix}rank_one_energy_min": (
            float(min(energies)) if energies else float("nan")
        ),
    }


def held_out_rank_comparison(
    train_design,
    test_design,
    pilot,
    target: np.ndarray,
    *,
    ranks: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    """Held-out prediction of the delivered second stage at several ranks.

    Everything except the rank of the projection is held fixed: the same convex
    pilot, the same retained set, the same Huber threshold, and the same refit.
    A rank-one restriction that costs nothing shows up as a prediction error that
    does not improve when a second direction is allowed, and the comparison is
    out of sample, so the extra parameters do not win by construction.
    """

    from scripts.review_common import rank_r_refit

    out: dict[str, float] = {}
    for rank in ranks:
        gamma, blocks, converged = rank_r_refit(
            train_design,
            pilot.gamma,
            pilot.blocks,
            pilot.selected,
            pilot.delta,
            rank,
        )
        from src.trace_vcam import predict_components

        prediction = predict_components(test_design, gamma, blocks)[0]
        out[f"mspe_rank{rank}"] = float(np.mean((prediction - target) ** 2))
        out[f"converged_rank{rank}"] = bool(converged)
    base = out.get("mspe_rank1", float("nan"))
    for rank in ranks:
        if rank == 1:
            continue
        value = out[f"mspe_rank{rank}"]
        # The share of the rank-one prediction error that a rank-``r`` projection
        # removes.  Positive means the extra direction helps out of sample.
        out[f"rank{rank}_gain"] = (
            float((base - value) / base) if base > 0.0 else float("nan")
        )
    return out


def _diagnostic_replication(job) -> dict:
    seed, n_subjects, defect, distribution = job
    from experiments.dgp import subject_split, zzw2020_truth
    from scripts.run_theory_gap_checks import (
        _best_separable,
        _extra_pair,
        _rank_two_dataset,
        _rank_two_grid,
    )
    from src.trace_vcam import OrthonormalSplineBasis

    tuning = locked_tuning()
    data = _rank_two_dataset(
        seed, n_subjects=n_subjects, defect=defect, distribution=distribution
    )
    train, test = subject_split(data.subject, seed=seed)
    basis = OrthonormalSplineBasis.create(
        int(tuning["q_time"]), int(tuning["q_covariate"])
    )
    train_design = build_design(
        data.time[train],
        data.covariates[train],
        data.response[train],
        data.subject[train],
        basis,
        data.domain_time,
    )
    test_design = build_design(
        data.time[test],
        data.covariates[test],
        data.response[test],
        data.subject[test],
        basis,
        data.domain_time,
    )

    pilot = fit_delivered(train_design, basis, tuning=tuning, post_rank_one=False)

    time_grid = np.linspace(*data.domain_time, GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    base = zzw2020_truth()
    extra_beta, extra_phi = _extra_pair()
    truth_surface = _rank_two_grid(
        base, extra_beta, extra_phi, defect, time_grid, covariate_grid
    )
    _, zeta, singular = _best_separable(truth_surface)

    row: dict[str, object] = {
        "seed": seed,
        "n_subjects": n_subjects,
        "defect_coefficient": defect,
        "error_law": distribution,
        "zeta": zeta,
        # The defect as a share of the size of the component it is a defect of.
        # The bare ratio of the two leading singular values is not a monotone
        # summary of non-separability: once the added direction dominates, the
        # surface is close to rank one again and that ratio falls back, while the
        # amount of the component no single product can represent does not.
        "zeta_relative": float(
            zeta / np.sqrt(np.mean(truth_surface**2))
            if np.any(truth_surface)
            else 0.0
        ),
        "population_second_over_first": float(singular[1] / singular[0]),
        "converged": pilot.converged,
    }
    # The defect is placed in the first block only, so the diagnostic is read on
    # that block; the second block is the unchanged separable control.
    row.update(spectral_summary(pilot, block=0))
    row.update(spectral_summary(pilot, block=1))
    row.update(
        held_out_rank_comparison(
            train_design, test_design, pilot, data.conditional_mean[test]
        )
    )
    return row


def command_rank_diagnostic(args) -> None:
    jobs = [
        (DEFAULT_SEED + 1000 * replicate + 37, args.subjects, defect, law)
        for law in args.laws
        for defect in DIAGNOSTIC_DEFECTS
        for replicate in range(args.replications)
    ]
    rows = _run(_diagnostic_replication, jobs, args.jobs, "rank-diagnostic")
    _write(rows, args.output / "rank_diagnostic.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "normalisation", "selection-path", "tuning-sensitivity", "rank-diagnostic",
    ])
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--laws", nargs="+", default=list(ERROR_LAWS))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "review_experiments"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "normalisation": command_normalisation,
        "selection-path": command_selection_path,
        "tuning-sensitivity": command_tuning_sensitivity,
        "rank-diagnostic": command_rank_diagnostic,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

"""Adapter for the local TRACE-VCAM implementation."""

from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np

from ..data import SubjectDataset
from ..methods import MethodLabel
from .base import BenchmarkAdapter, FitArtifact, PreflightReport


def _numeric_subject_ids(dataset: SubjectDataset) -> np.ndarray:
    mapping = {subject: index for index, subject in enumerate(sorted(dataset.subjects.tolist()))}
    return np.asarray([mapping[subject] for subject in dataset.subject_id], dtype=np.int64)


def _registered_domains(
    dataset: SubjectDataset,
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
    time_value = dataset.metadata.get("time_domain")
    time_domain = (
        (float(np.min(dataset.time)), float(np.max(dataset.time)))
        if time_value is None
        else (float(time_value[0]), float(time_value[1]))
    )
    covariate_value = dataset.metadata.get("covariate_domains")
    if covariate_value is None:
        covariate_domains = tuple(
            (float(np.min(dataset.covariates[:, index])), float(np.max(dataset.covariates[:, index])))
            for index in range(dataset.covariates.shape[1])
        )
    else:
        covariate_domains = tuple(
            (float(item[0]), float(item[1])) for item in covariate_value
        )
    if len(covariate_domains) != dataset.covariates.shape[1]:
        raise ValueError("covariate_domains must contain one interval per block")
    for lower, upper in (time_domain, *covariate_domains):
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("registered domains must be finite nondegenerate intervals")
    return time_domain, covariate_domains


def _to_unit(values: np.ndarray, domain: tuple[float, float]) -> np.ndarray:
    lower, upper = domain
    scaled = (np.asarray(values, dtype=float) - lower) / (upper - lower)
    if np.any(scaled < -1e-10) or np.any(scaled > 1.0 + 1e-10):
        raise ValueError("TRACE input falls outside its registered domain")
    return np.clip(scaled, 0.0, 1.0)


def _subject_subset(dataset: SubjectDataset, keep: np.ndarray) -> SubjectDataset:
    mask = np.isin(dataset.subject_id, keep)
    return SubjectDataset(
        time=dataset.time[mask],
        covariates=dataset.covariates[mask],
        response=dataset.response[mask],
        subject_id=dataset.subject_id[mask],
        row_id=None if dataset.row_id is None else dataset.row_id[mask],
        noise_free_target=None,
        covariate_names=dataset.covariate_names,
        metadata=dataset.metadata,
    )


def _subject_balanced_squared_error(
    subject: np.ndarray, residual: np.ndarray
) -> float:
    return float(
        np.mean(
            [float(np.mean(residual[subject == item] ** 2)) for item in np.unique(subject)]
        )
    )


def _candidate_grid(tuning: Mapping[str, object]) -> tuple[dict[str, float], ...]:
    def _values(key: str, default: Sequence[float]) -> tuple[float, ...]:
        raw = tuning.get(key, default)
        return tuple(float(item) for item in raw)  # type: ignore[union-attr]

    bases = _values("cv_basis_grid", (5.0, 6.0))
    ratios = _values("cv_lambda_ratio_grid", (0.08, 0.2, 0.35))
    roughness = _values("cv_roughness_grid", (0.05, 0.5))
    multipliers = _values("cv_huber_multiplier_grid", (1.345, 3.0, 10.0))
    return tuple(
        {
            "q_time": int(basis),
            "q_covariate": int(basis),
            "lambda_ratio": ratio,
            "roughness": mu,
            "huber_multiplier": multiplier,
        }
        for basis, ratio, mu, multiplier in itertools.product(
            bases, ratios, roughness, multipliers
        )
    )


class TraceVCAMAdapter(BenchmarkAdapter):
    label = MethodLabel.TRACE_VCAM.value

    def _select_by_subject_cv(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> tuple[dict[str, float], dict[str, object]]:
        """Choose the penalty pair, roughness, threshold, and basis size by
        subject-level cross-validation inside the supplied training data.

        The held-out data of the outer split is never touched.  The criterion is
        the subject-balanced held-out squared error, matching the criterion the
        published competitors use when they select their own tuning parameters
        on the same application.
        """

        n_folds = int(tuning.get("cv_folds", 5))
        subjects = train.subjects
        if n_folds < 2 or subjects.size < 2 * n_folds:
            raise ValueError("subject-level cross-validation needs enough subjects")
        rng = np.random.default_rng(int(seed) % (2**32))
        shuffled = rng.permutation(subjects)
        folds = [shuffled[index::n_folds] for index in range(n_folds)]

        candidates = _candidate_grid(tuning)
        base = {
            key: tuning[key]
            for key in ("time_domain", "covariate_domains", "max_iter", "tolerance")
            if key in tuning
        }
        scores = np.full(len(candidates), np.inf)
        for position, candidate in enumerate(candidates):
            total, usable = 0.0, 0
            for held in folds:
                inner_train = _subject_subset(train, np.setdiff1d(subjects, held))
                inner_test = _subject_subset(train, held)
                try:
                    artifact = self.fit(
                        inner_train,
                        seed=seed,
                        tuning={**base, **candidate, "delta_rule": "mad", "selection": "fixed"},
                    )
                    residual = self.predict(artifact, inner_test) - inner_test.response
                    if not np.all(np.isfinite(residual)):
                        raise FloatingPointError("non-finite inner prediction")
                except Exception:  # pragma: no cover - a candidate may be infeasible
                    total, usable = float("nan"), 0
                    break
                total += _subject_balanced_squared_error(inner_test.subject_id, residual)
                usable += 1
            if usable == n_folds:
                scores[position] = total / n_folds
        if not np.any(np.isfinite(scores)):
            raise RuntimeError("no TRACE tuning candidate completed cross-validation")
        best = int(np.argmin(scores))
        audit = {
            "rule": "subject-level cross-validated subject-balanced squared error",
            "cv_folds": n_folds,
            "candidates": len(candidates),
            "selected": dict(candidates[best]),
            "selected_score": float(scores[best]),
            "finite_candidates": int(np.sum(np.isfinite(scores))),
        }
        return dict(candidates[best]), audit

    def preflight(self) -> PreflightReport:
        try:
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
            from src import trace_vcam

            version = getattr(trace_vcam, "__version__", "workspace-two-stage-v1")
            return PreflightReport(True, str(version))
        except Exception as error:  # pragma: no cover - environment-specific
            return PreflightReport(
                False,
                "unavailable",
                code="python_dependency_failure",
                message=f"{type(error).__name__}: {error}",
            )

    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        from src.trace_vcam import (
            OrthonormalSplineBasis,
            VCAMDesign,
            fit_trace_vcam,
            practical_huber_threshold,
            trace_lambda_max,
        )

        selection_audit: dict[str, object] | None = None
        if str(tuning.get("selection", "fixed")) == "subject_cv":
            selected, selection_audit = self._select_by_subject_cv(
                train, seed=seed, tuning=tuning
            )
            tuning = {**tuning, **selected, "selection": "fixed"}

        q_time = int(tuning.get("q_time", 6))
        q_covariate = int(tuning.get("q_covariate", q_time))
        time_domain, covariate_domains = _registered_domains(train)
        scaled_time = _to_unit(train.time, time_domain)
        scaled_covariates = np.column_stack(
            [
                _to_unit(train.covariates[:, index], domain)
                for index, domain in enumerate(covariate_domains)
            ]
        )
        basis = OrthonormalSplineBasis.create(q_time, q_covariate)
        design = VCAMDesign.from_arrays(
            scaled_time,
            scaled_covariates,
            train.response,
            _numeric_subject_ids(train),
            basis,
        )
        delta_rule = str(tuning.get("delta_rule", "mad"))
        if delta_rule == "fixed":
            if "delta" not in tuning:
                raise ValueError("fixed delta_rule requires tuning['delta']")
            delta = float(tuning["delta"])
        elif delta_rule == "mad":
            delta, _ = practical_huber_threshold(
                design, multiplier=float(tuning.get("huber_multiplier", 1.345))
            )
        else:
            raise ValueError("delta_rule must be 'fixed' or 'mad'")

        lambda_max, _ = trace_lambda_max(design, delta)
        penalty = float(tuning.get("penalty", float(tuning.get("lambda_ratio", 0.2)) * lambda_max))
        common = {
            "max_iter": int(tuning.get("max_iter", 2000)),
            "tolerance": float(tuning.get("tolerance", 1e-7)),
            "mu": float(tuning.get("roughness", 0.0)),
            "postfit_max_iter": int(tuning.get("postfit_max_iter", 500)),
            "postfit_tolerance": float(tuning.get("postfit_tolerance", 1e-8)),
        }
        if delta_rule == "mad":
            fit = fit_trace_vcam(
                design,
                penalty,
                delta=None,
                threshold_mode="mad",
                huber_multiplier=float(tuning.get("huber_multiplier", 1.345)),
                **common,
            )
        else:
            fit = fit_trace_vcam(
                design,
                penalty,
                delta=delta,
                threshold_mode="fixed",
                **common,
            )
        recorded_tuning = dict(tuning)
        recorded_tuning.update(
            {
                "q_time": q_time,
                "q_covariate": q_covariate,
                "delta_rule": delta_rule,
                "delta_realized": delta,
                "lambda_max": lambda_max,
                "penalty_realized": penalty,
                "block_weights": fit.block_weights.tolist(),
                "time_domain": list(time_domain),
                "covariate_domains": [list(item) for item in covariate_domains],
                "domain_mapping": "registered linear map to [0,1]",
            }
        )
        if selection_audit is not None:
            recorded_tuning["selection"] = "subject_cv"
            recorded_tuning["selection_audit"] = selection_audit
        return FitArtifact(
            model={
                "fit": fit,
                "basis": basis,
                "time_domain": time_domain,
                "covariate_domains": covariate_domains,
            },
            method=self.label,
            version=self.preflight().version,
            tuning=recorded_tuning,
            converged=bool(fit.converged),
            selected_blocks=tuple(int(index) for index in np.flatnonzero(fit.selected)),
            metadata={
                "iterations": int(fit.iterations),
                "convex_kkt_residual": getattr(fit, "convex_kkt_residual", None),
                "scalar_postfit_converged": getattr(fit, "scalar_postfit_converged", None),
                "scalar_postfit_kkt_residual": getattr(fit, "scalar_postfit_kkt_residual", None),
                "penalty_weights_source": getattr(fit, "penalty_weights_source", None),
                "estimator_metadata": dict(getattr(fit, "metadata", {})),
                "threshold_asymptotics": (
                    "fixed-threshold theory does not cover this data-adaptive MAD value"
                    if delta_rule == "mad"
                    else "fixed threshold"
                ),
                "domain_mapping": {
                    "rule": "registered linear map to [0,1]",
                    "time_domain": list(time_domain),
                    "covariate_domains": [list(item) for item in covariate_domains],
                },
            },
        )

    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> np.ndarray:
        from src.trace_vcam import VCAMDesign

        basis = artifact.model["basis"]
        fit = artifact.model["fit"]
        time_domain = artifact.model["time_domain"]
        covariate_domains = artifact.model["covariate_domains"]
        scaled_time = _to_unit(test.time, time_domain)
        scaled_covariates = np.column_stack(
            [
                _to_unit(test.covariates[:, index], domain)
                for index, domain in enumerate(covariate_domains)
            ]
        )
        design = VCAMDesign.from_arrays(
            scaled_time,
            scaled_covariates,
            np.zeros(test.n_rows),
            _numeric_subject_ids(test),
            basis,
        )
        prediction = np.asarray(fit.predict(design), dtype=float)
        if prediction.shape != (test.n_rows,):
            raise ValueError("TRACE prediction has the wrong row count")
        return prediction

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        basis = artifact.model["basis"]
        fit = artifact.model["fit"]
        unit_grid = np.asarray(basis.grid, dtype=float)
        time_domain = artifact.model["time_domain"]
        covariate_domains = artifact.model["covariate_domains"]
        time_grid = time_domain[0] + unit_grid * (time_domain[1] - time_domain[0])
        time_basis = basis.transform_time(unit_grid)
        covariate_basis = basis.transform_covariate(unit_grid)
        curves: list[dict[str, object]] = [
            {
                "component": "baseline",
                "domain": "time",
                "grid": time_grid.tolist(),
                "values": (time_basis @ fit.gamma).tolist(),
            }
        ]
        time_factors = fit.identified_time_factors
        covariate_factors = fit.identified_covariate_factors
        if time_factors is None or covariate_factors is None:
            raise ValueError("TRACE fit does not contain identified postfit factors")
        if len(time_factors) != len(fit.matrices) or len(covariate_factors) != len(
            fit.matrices
        ):
            raise ValueError("TRACE identified factors are not block-aligned")
        for index, selected in enumerate(np.asarray(fit.selected, dtype=bool)):
            if not selected:
                continue
            beta_coefficients = time_factors[index]
            phi_coefficients = covariate_factors[index]
            if beta_coefficients is None or phi_coefficients is None:
                raise ValueError(
                    f"TRACE selected block {index} lacks identified factor coefficients"
                )
            coefficient = time_basis @ np.asarray(beta_coefficients, dtype=float)
            additive = covariate_basis @ np.asarray(phi_coefficients, dtype=float)
            covariate_domain = covariate_domains[index]
            covariate_grid = covariate_domain[0] + unit_grid * (
                covariate_domain[1] - covariate_domain[0]
            )
            curves.extend(
                [
                    {
                        "component": f"beta_{index + 1}",
                        "domain": "time",
                        "grid": time_grid.tolist(),
                        "values": coefficient.tolist(),
                    },
                    {
                        "component": f"phi_{index + 1}",
                        "domain": f"covariate_{index + 1}",
                        "grid": covariate_grid.tolist(),
                        "values": additive.tolist(),
                    },
                ]
            )
        return tuple(curves)

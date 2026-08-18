"""Paper-faithful modified backfitting adapter for Zhang--Zhong--Wang (2020)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..data import SubjectDataset, make_repeated_subject_folds
from ..methods import MethodLabel
from .base import BenchmarkAdapter, FitArtifact, PreflightReport
from .splines import (
    SplineBasis,
    SplineFunction,
    solve_least_squares,
)


def _expand_counts(value: object, length: int, default: int) -> tuple[int, ...]:
    if value is None:
        return (default,) * length
    if np.isscalar(value):
        return (int(value),) * length
    result = tuple(int(item) for item in value)  # type: ignore[arg-type]
    if len(result) != length:
        raise ValueError(f"expected {length} knot counts, received {len(result)}")
    return result


def _registered_domains(
    dataset: SubjectDataset, tuning: Mapping[str, object]
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
    time_domain_value = tuning.get("time_domain", dataset.metadata.get("time_domain"))
    time_domain = (
        (float(np.min(dataset.time)), float(np.max(dataset.time)))
        if time_domain_value is None
        else (float(time_domain_value[0]), float(time_domain_value[1]))  # type: ignore[index]
    )
    covariate_value = tuning.get("covariate_domains", dataset.metadata.get("covariate_domains"))
    if covariate_value is None:
        covariate_domains = tuple(
            (float(np.min(dataset.covariates[:, index])), float(np.max(dataset.covariates[:, index])))
            for index in range(dataset.covariates.shape[1])
        )
    else:
        covariate_domains = tuple(
            (float(item[0]), float(item[1])) for item in covariate_value  # type: ignore[union-attr]
        )
    if len(covariate_domains) != dataset.covariates.shape[1]:
        raise ValueError("covariate_domains must contain one interval per covariate")
    return time_domain, covariate_domains


def _smooth(
    response: np.ndarray,
    multiplier: np.ndarray,
    argument: np.ndarray,
    basis: SplineBasis,
) -> SplineFunction:
    features = multiplier[:, None] * basis.transform(argument)
    return SplineFunction(basis, solve_least_squares(features, response))


def _component_prediction(
    time: np.ndarray,
    covariates: np.ndarray,
    baseline: SplineFunction,
    coefficients: Sequence[SplineFunction],
    additives: Sequence[SplineFunction],
) -> np.ndarray:
    prediction = baseline(time)
    for index, (coefficient, additive) in enumerate(zip(coefficients, additives, strict=True)):
        prediction = prediction + coefficient(time) * additive(covariates[:, index])
    return np.asarray(prediction, dtype=float)


def _paper_subject_fold_rss(
    residual: np.ndarray, subject_id: np.ndarray
) -> float:
    """Zhang--Zhong--Wang's visit-level RSS divided by fold subjects.

    Their five-fold criterion divides each held-out fold's total visit RSS by
    the number of subjects in that fold.  It does not divide each subject's
    contribution by that subject's number of visits.
    """

    residual_array = np.asarray(residual, dtype=float).reshape(-1)
    subject_array = np.asarray(subject_id).reshape(-1)
    if residual_array.size != subject_array.size:
        raise ValueError("residual and subject_id must have equal lengths")
    n_subjects = int(np.unique(subject_array).size)
    if n_subjects < 1:
        raise ValueError("a CV fold must contain at least one subject")
    return float(np.sum(residual_array**2) / n_subjects)


@dataclass
class ZZW2020Model:
    baseline: SplineFunction
    coefficients: tuple[SplineFunction, ...]
    additives: tuple[SplineFunction, ...]
    converged: bool
    outer_iterations: int
    inner_iterations: tuple[int, ...]
    objective: tuple[float, ...]
    initialization: str


def _initial_longitudinal(
    dataset: SubjectDataset,
    time_bases: Sequence[SplineBasis],
    covariate_bases: Sequence[SplineBasis],
) -> tuple[SplineFunction, list[SplineFunction], list[SplineFunction]]:
    # Initialization 2, Step 1: additive fit with beta_k(t) fixed to one.
    features = [time_bases[0].transform(dataset.time)]
    features.extend(
        basis.transform(dataset.covariates[:, index])
        for index, basis in enumerate(covariate_bases)
    )
    coefficients = solve_least_squares(np.column_stack(features), dataset.response)
    cursor = time_bases[0].dimension
    baseline = SplineFunction(time_bases[0], coefficients[:cursor])
    additives: list[SplineFunction] = []
    for basis in covariate_bases:
        next_cursor = cursor + basis.dimension
        additives.append(
            SplineFunction(basis, coefficients[cursor:next_cursor]).centered_lebesgue()
        )
        cursor = next_cursor

    # Initialization 2, Step 2: varying-coefficient fit with the additive
    # functions from Step 1 treated as known covariates.
    vc_features = [time_bases[0].transform(dataset.time)]
    for index, (basis, additive) in enumerate(zip(time_bases[1:], additives, strict=True)):
        vc_features.append(
            additive(dataset.covariates[:, index])[:, None] * basis.transform(dataset.time)
        )
    vc_solution = solve_least_squares(np.column_stack(vc_features), dataset.response)
    cursor = time_bases[0].dimension
    baseline = SplineFunction(time_bases[0], vc_solution[:cursor])
    coefficient_functions: list[SplineFunction] = []
    for basis in time_bases[1:]:
        next_cursor = cursor + basis.dimension
        coefficient_functions.append(
            SplineFunction(basis, vc_solution[cursor:next_cursor]).normalized_mean_one()
        )
        cursor = next_cursor
    return baseline, coefficient_functions, additives


def _subject_integrated_response(
    dataset: SubjectDataset, time_domain: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = time_domain
    integrated: list[float] = []
    first_rows: list[int] = []
    for subject in sorted(dataset.subjects.tolist()):
        rows = np.flatnonzero(dataset.subject_id == subject)
        order = rows[np.argsort(dataset.time[rows])]
        t = dataset.time[order]
        y = dataset.response[order]
        area = float(np.trapezoid(y, t))
        area += float(y[0] * (t[0] - lower) + y[-1] * (upper - t[-1]))
        integrated.append(area / (upper - lower))
        first_rows.append(int(rows[0]))
    return np.asarray(integrated), np.asarray(first_rows, dtype=int)


def _initial_time_invariant(
    dataset: SubjectDataset,
    time_bases: Sequence[SplineBasis],
    covariate_bases: Sequence[SplineBasis],
    time_domain: tuple[float, float],
) -> tuple[SplineFunction, list[SplineFunction], list[SplineFunction]]:
    # Reject a mislabelled longitudinal design rather than averaging away its
    # time variation.
    for subject in dataset.subjects:
        rows = dataset.subject_id == subject
        if np.max(np.ptp(dataset.covariates[rows], axis=0)) > 1e-10:
            raise ValueError("time-invariant initialization requested for a longitudinal covariate")
    integrated, first_rows = _subject_integrated_response(dataset, time_domain)
    subject_covariates = dataset.covariates[first_rows]
    features = [np.ones((len(first_rows), 1), dtype=float)]
    features.extend(
        basis.transform(subject_covariates[:, index])
        for index, basis in enumerate(covariate_bases)
    )
    additive_solution = solve_least_squares(np.column_stack(features), integrated)
    cursor = 1
    additives: list[SplineFunction] = []
    for basis in covariate_bases:
        next_cursor = cursor + basis.dimension
        additives.append(
            SplineFunction(basis, additive_solution[cursor:next_cursor]).centered_lebesgue()
        )
        cursor = next_cursor

    vc_features = [time_bases[0].transform(dataset.time)]
    for index, (basis, additive) in enumerate(zip(time_bases[1:], additives, strict=True)):
        vc_features.append(
            additive(dataset.covariates[:, index])[:, None] * basis.transform(dataset.time)
        )
    vc_solution = solve_least_squares(np.column_stack(vc_features), dataset.response)
    cursor = time_bases[0].dimension
    baseline = SplineFunction(time_bases[0], vc_solution[:cursor])
    coefficient_functions: list[SplineFunction] = []
    for basis in time_bases[1:]:
        next_cursor = cursor + basis.dimension
        coefficient_functions.append(
            SplineFunction(basis, vc_solution[cursor:next_cursor]).normalized_mean_one()
        )
        cursor = next_cursor
    return baseline, coefficient_functions, additives


class ZZW2020Adapter(BenchmarkAdapter):
    """Algorithm 1 with both published initialization regimes.

    The paper selects knot counts by five-fold subject/curve CV.  Formal
    reproduction runs may lock the paper's knot counts using
    ``tuning_mode='paper_locked'``; this choice is recorded rather than being
    represented as a completed BIC/CV search.
    """

    label = MethodLabel.ZZW2020.value

    def preflight(self) -> PreflightReport:
        try:
            import scipy

            return PreflightReport(True, f"paper-Algorithm-1/2020; scipy-{scipy.__version__}")
        except Exception as error:  # pragma: no cover
            return PreflightReport(False, "unavailable", "python_dependency_failure", str(error))

    def _fit_registered_subject_cv(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        """Apply the paper's five-fold subject CV to an audited vector grid.

        The source defines the CV score but does not report the searched joint
        set of five knot counts.  The registered vectors are therefore stored
        verbatim in the returned artifact rather than being presented as
        author-supplied software defaults.
        """

        raw_vectors = tuning.get(
            "knot_candidate_vectors",
            (
                {"time": (4, 1, 2), "additive": (3, 2)},
                {"time": (4, 2, 2), "additive": (3, 2)},
                {"time": (3, 2, 2), "additive": (3, 2)},
            ),
        )
        vectors: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for item in raw_vectors:  # type: ignore[union-attr]
            if not isinstance(item, Mapping):
                raise ValueError("each ZZW2020 CV knot vector must be a mapping")
            time_counts = tuple(int(value) for value in item["time"])  # type: ignore[index]
            additive_counts = tuple(int(value) for value in item["additive"])  # type: ignore[index]
            if len(time_counts) != train.covariates.shape[1] + 1:
                raise ValueError("ZZW2020 time knot vector has the wrong length")
            if len(additive_counts) != train.covariates.shape[1]:
                raise ValueError("ZZW2020 additive knot vector has the wrong length")
            if any(value < 0 for value in time_counts + additive_counts):
                raise ValueError("ZZW2020 knot counts must be nonnegative")
            vectors.append((time_counts, additive_counts))
        if not vectors:
            raise ValueError("ZZW2020 CV requires at least one registered knot vector")
        folds = make_repeated_subject_folds(
            train,
            n_splits=int(tuning.get("cv_folds", 5)),
            n_repeats=1,
            seed=int(seed),
        )
        trace: list[dict[str, object]] = []
        scores: list[float] = []
        for time_counts, additive_counts in vectors:
            fold_scores: list[float] = []
            converged = True
            for split in folds:
                training = train.subset_subjects(split.train_subjects)
                validation = train.subset_subjects(split.test_subjects)
                fixed = dict(tuning)
                fixed.update(
                    tuning_mode="paper_design_counts",
                    time_interior_knots=list(time_counts),
                    covariate_interior_knots=list(additive_counts),
                    max_inner=min(
                        int(tuning.get("max_inner", 200)),
                        int(tuning.get("cv_max_inner", 50)),
                    ),
                    max_outer=min(
                        int(tuning.get("max_outer", 200)),
                        int(tuning.get("cv_max_outer", 50)),
                    ),
                )
                artifact = self.fit(training, seed=seed, tuning=fixed)
                if not artifact.converged:
                    converged = False
                    fold_scores.append(float("inf"))
                    continue
                prediction = self.predict(artifact, validation)
                fold_scores.append(
                    _paper_subject_fold_rss(
                        validation.response - prediction,
                        validation.subject_id,
                    )
                )
            score = float(np.sum(fold_scores)) if converged else float("inf")
            scores.append(score)
            trace.append(
                {
                    "time_interior_knots": list(time_counts),
                    "covariate_interior_knots": list(additive_counts),
                    "fold_scores": fold_scores,
                    "cv_score": score,
                    "converged": converged,
                }
            )
        finite = [index for index, value in enumerate(scores) if np.isfinite(value)]
        if not finite:
            raise FloatingPointError("ZZW2020 registered CV found no converged candidate")
        selected_index = min(finite, key=lambda index: scores[index])
        selected_time, selected_additive = vectors[selected_index]
        final_tuning = dict(tuning)
        final_tuning.update(
            tuning_mode="paper_design_counts",
            time_interior_knots=list(selected_time),
            covariate_interior_knots=list(selected_additive),
        )
        artifact = self.fit(train, seed=seed, tuning=final_tuning)
        artifact.tuning.update(
            {
                "tuning_mode": "paper_cv_registered_vectors",
                "cv_folds": len(folds),
                "knot_candidate_vectors": [
                    {"time": list(time), "additive": list(additive)}
                    for time, additive in vectors
                ],
                "selected_candidate_index": selected_index,
                "cv_max_inner": int(tuning.get("cv_max_inner", 50)),
                "cv_max_outer": int(tuning.get("cv_max_outer", 50)),
            }
        )
        artifact.metadata.update(
            {
                "reproduction_mode": (
                    "paper five-fold subject CV over a preregistered joint-vector set; "
                    "the source does not disclose its original searched set"
                ),
                "cv_trace": trace,
            }
        )
        return artifact

    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        tuning_mode = str(tuning.get("tuning_mode", "paper_design_counts"))
        if tuning_mode == "paper_cv_registered_vectors":
            return self._fit_registered_subject_cv(
                train, seed=seed, tuning=tuning
            )
        p = train.covariates.shape[1]
        order = int(tuning.get("spline_order", 4))
        time_counts = _expand_counts(tuning.get("time_interior_knots"), p + 1, 2)
        covariate_counts = _expand_counts(tuning.get("covariate_interior_knots"), p, 2)
        time_domain, covariate_domains = _registered_domains(train, tuning)
        time_bases = tuple(
            SplineBasis.equidistant(
                train.time, n_interior=count, order=order, domain=time_domain
            )
            for count in time_counts
        )
        covariate_bases = tuple(
            SplineBasis.equidistant(
                train.covariates[:, index],
                n_interior=count,
                order=order,
                domain=covariate_domains[index],
            )
            for index, count in enumerate(covariate_counts)
        )

        initialization = str(tuning.get("initialization", "auto"))
        if initialization == "auto":
            initialization = (
                "time_invariant"
                if bool(train.metadata.get("time_invariant_covariates", False))
                else "longitudinal"
            )
        if initialization == "time_invariant":
            baseline, coefficients, additives = _initial_time_invariant(
                train, time_bases, covariate_bases, time_domain
            )
        elif initialization == "longitudinal":
            baseline, coefficients, additives = _initial_longitudinal(
                train, time_bases, covariate_bases
            )
        else:
            raise ValueError("initialization must be auto, time_invariant, or longitudinal")

        epsilon_outer = float(tuning.get("epsilon_outer", 1e-3))
        epsilon_inner = float(tuning.get("epsilon_inner", 1e-2))
        max_outer = int(tuning.get("max_outer", 200))
        max_inner = int(tuning.get("max_inner", 200))
        if epsilon_outer != 1e-3 or epsilon_inner != 1e-2:
            raise ValueError("ZZW2020 strict mode fixes epsilon_outer=1e-3 and epsilon_inner=1e-2")
        objective: list[float] = []
        inner_totals = np.zeros(p, dtype=int)
        outer_converged = False
        numerical_increase = False
        old_mrs = float(
            np.mean(
                (
                    train.response
                    - _component_prediction(
                        train.time, train.covariates, baseline, coefficients, additives
                    )
                )
                ** 2
            )
        )
        objective.append(old_mrs)
        for outer_iteration in range(1, max_outer + 1):
            for index in range(p):
                other = baseline(train.time)
                for other_index in range(p):
                    if other_index != index:
                        other = other + coefficients[other_index](train.time) * additives[
                            other_index
                        ](train.covariates[:, other_index])
                residual = train.response - other
                inner_old = float(
                    np.mean(
                        (
                            residual
                            - coefficients[index](train.time)
                            * additives[index](train.covariates[:, index])
                        )
                        ** 2
                    )
                )
                for inner_iteration in range(1, max_inner + 1):
                    coefficients[index] = _smooth(
                        residual,
                        additives[index](train.covariates[:, index]),
                        train.time,
                        time_bases[index + 1],
                    ).normalized_mean_one()
                    additives[index] = _smooth(
                        residual,
                        coefficients[index](train.time),
                        train.covariates[:, index],
                        covariate_bases[index],
                    ).centered_lebesgue()
                    inner_new = float(
                        np.mean(
                            (
                                residual
                                - coefficients[index](train.time)
                                * additives[index](train.covariates[:, index])
                            )
                            ** 2
                        )
                    )
                    inner_totals[index] += 1
                    if not np.isfinite(inner_new):
                        raise FloatingPointError("ZZW2020 inner MRS became nonfinite")
                    # Algorithm 1 normalizes beta and recenters phi after the
                    # two conditional least-squares updates.  Those
                    # identification transforms need not preserve the raw
                    # finite-sample MRS, so a temporary increase is an audit
                    # diagnostic, not a stopping failure.  The paper stops on
                    # the magnitude of the MRS change.
                    if inner_new > inner_old + 1e-8:
                        numerical_increase = True
                    if abs(inner_old - inner_new) < epsilon_inner:
                        break
                    inner_old = inner_new

            component_sum = np.zeros(train.n_rows, dtype=float)
            for index in range(p):
                component_sum += coefficients[index](train.time) * additives[index](
                    train.covariates[:, index]
                )
            baseline = _smooth(
                train.response - component_sum,
                np.ones(train.n_rows),
                train.time,
                time_bases[0],
            )
            new_mrs = float(
                np.mean(
                    (
                        train.response
                        - _component_prediction(
                            train.time, train.covariates, baseline, coefficients, additives
                        )
                    )
                    ** 2
                )
            )
            objective.append(new_mrs)
            if not np.isfinite(new_mrs):
                raise FloatingPointError("ZZW2020 outer MRS became nonfinite")
            if new_mrs > old_mrs + 1e-8:
                numerical_increase = True
            if abs(old_mrs - new_mrs) < epsilon_outer:
                outer_converged = True
                break
            old_mrs = new_mrs

        model = ZZW2020Model(
            baseline=baseline,
            coefficients=tuple(coefficients),
            additives=tuple(additives),
            converged=bool(outer_converged),
            outer_iterations=outer_iteration,
            inner_iterations=tuple(int(item) for item in inner_totals),
            objective=tuple(objective),
            initialization=initialization,
        )
        recorded_tuning = dict(tuning)
        recorded_tuning.update(
            {
                "tuning_mode": str(tuning.get("tuning_mode", "paper_locked")),
                "spline_order": order,
                "time_interior_knots": list(time_counts),
                "covariate_interior_knots": list(covariate_counts),
                "time_domain": list(time_domain),
                "covariate_domains": [list(item) for item in covariate_domains],
                "epsilon_outer": epsilon_outer,
                "epsilon_inner": epsilon_inner,
                "initialization": initialization,
            }
        )
        return FitArtifact(
            model=model,
            method=self.label,
            version=self.preflight().version,
            tuning=recorded_tuning,
            converged=bool(outer_converged),
            metadata={
                "algorithm": "Zhang-Zhong-Wang (2020), Algorithm 1",
                "reproduction_mode": (
                    "paper data-generating knot counts locked; the source gives five-fold "
                    "subject CV but omits its searched candidate set"
                    if recorded_tuning["tuning_mode"] == "paper_design_counts"
                    else (
                        "paper-aligned fixed p=10 extension using the reported "
                        "Zhao--Sun--Yang design-count vector; this is not a "
                        "source-original Zhang--Zhong--Wang CV search"
                        if recorded_tuning["tuning_mode"]
                        == "paper_aligned_fixed_p10_extension"
                        else "caller-declared tuning mode"
                    )
                ),
                "outer_iterations": outer_iteration,
                "inner_iterations": list(model.inner_iterations),
                "numerical_mrs_increase": numerical_increase,
                "normalization": "domain-average(beta)=1 and Lebesgue-average(phi)=0 after each inner update",
            },
        )

    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> np.ndarray:
        model: ZZW2020Model = artifact.model
        return _component_prediction(
            test.time, test.covariates, model.baseline, model.coefficients, model.additives
        )

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        model: ZZW2020Model = artifact.model
        curves: list[dict[str, object]] = []
        functions = [("baseline", "time", model.baseline)]
        for index, function in enumerate(model.coefficients):
            functions.append((f"beta_{index + 1}", "time", function))
        for index, function in enumerate(model.additives):
            functions.append((f"phi_{index + 1}", f"covariate_{index + 1}", function))
        for component, domain_name, function in functions:
            raw = function.basis.raw if hasattr(function.basis, "raw") else function.basis
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

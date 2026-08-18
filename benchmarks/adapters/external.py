"""Adapters that invoke immutable R implementations through narrow wrappers."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from ..data import SubjectDataset
from ..methods import MethodLabel
from ..vendor import VENDOR_ROOT, sha256_file, verify_zsy2026_vendor
from .base import (
    AdapterUnavailable,
    BenchmarkAdapter,
    FitArtifact,
    PreflightReport,
    UnsupportedPrediction,
)
from .hhy2021 import HHY2021Adapter
from .zy2025 import ZY2025Adapter
from .zzw2020 import ZZW2020Adapter


RUNNER_ROOT = Path(__file__).resolve().parents[1] / "runners"
ZSY_ASCII_STAGE_ROOT_ENV = "VCAM_ZSY2026_ASCII_STAGE_ROOT"


def _path_is_ascii(path: Path) -> bool:
    """Return whether an absolute operating-system path is ASCII encodable."""

    return str(path).isascii()


def _ascii_stage_parent() -> tuple[Path, str]:
    """Find a writable ASCII parent for the immutable author-code copy.

    R on Windows can fail before ``source()`` when an argument contains a
    non-ASCII workspace path.  The workspace is intentionally allowed to be
    non-ASCII, so only this narrow external-code boundary is relocated.  A
    caller can set ``VCAM_ZSY2026_ASCII_STAGE_ROOT`` to make the location
    explicit; otherwise the system temporary directory is used when it is
    ASCII.  ``C:\\Temp`` is a final conventional Windows fallback.
    """

    configured = os.environ.get(ZSY_ASCII_STAGE_ROOT_ENV)
    candidates = [
        (configured, "environment-configured-ascii-stage-root"),
        (tempfile.gettempdir(), "system-temporary-directory"),
        (r"C:\Temp", "windows-c-temp-fallback"),
    ]
    failures: list[str] = []
    for raw, strategy in candidates:
        if not raw:
            continue
        parent = Path(raw).expanduser()
        if not _path_is_ascii(parent):
            failures.append(f"{parent}: non-ASCII")
            continue
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            failures.append(f"{parent}: {type(error).__name__}")
            continue
        if not parent.is_dir():
            failures.append(f"{parent}: not a directory")
            continue
        return parent, strategy
    details = "; ".join(failures) or "no candidate directory"
    raise AdapterUnavailable(
        "VCAM-Lasso requires an ASCII staging directory for R on Windows; " + details
    )


@dataclass(frozen=True)
class ZSY2026AsciiStage:
    """Verified, ephemeral ASCII copy of the runner and author snapshot."""

    root: Path
    runner_path: Path
    vendor_source_path: Path
    parent_strategy: str
    source_sha256: Mapping[str, str]
    staged_sha256: Mapping[str, str]
    vendor_commit: str

    def audit_metadata(self) -> dict[str, object]:
        """Return serializable provenance without leaking an expired temp path."""

        return {
            "author_code_path_strategy": "verified-ascii-temporary-staging",
            "author_code_stage_parent_strategy": self.parent_strategy,
            "author_code_staged_paths_ascii": True,
            "author_code_staged_layout": {
                "runner": "runner/zsy2026_author_code.R",
                "vendor_source": "vendor/zsy2026_vcampackage/R/VCAMLasso.R",
            },
            "author_code_source_sha256": dict(self.source_sha256),
            "author_code_staged_sha256": dict(self.staged_sha256),
            "author_code_hashes_match_after_staging": (
                dict(self.source_sha256) == dict(self.staged_sha256)
            ),
            "author_code_vendor_commit": self.vendor_commit,
            "author_code_vendor_copy_policy": "read-only exact copy; no source patch applied",
        }


def _safe_vendor_relative_path(value: object) -> Path:
    """Restrict an audit-manifest path to the vendored snapshot tree."""

    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise AdapterUnavailable(f"unsafe vendored author-code path: {relative}")
    return relative


@contextmanager
def _stage_zsy2026_author_code() -> Iterator[ZSY2026AsciiStage]:
    """Copy the pinned runner/snapshot into a verified ASCII temp directory.

    The author source is never edited in place.  Every file named in the
    pinned ``ORIGIN.json`` manifest, plus the I/O-only runner, is copied with
    ``copy2`` and re-hashed before R is allowed to execute it.  The stage is
    process-local and removed after a fit, which avoids sharing mutable state
    across simultaneous benchmark replications.
    """

    vendor = verify_zsy2026_vendor()
    if not vendor["valid"]:
        raise AdapterUnavailable("pinned author-code hash verification failed before staging")
    origin = vendor["origin"]
    source_files = origin["files"]
    if not isinstance(source_files, Mapping):
        raise AdapterUnavailable("invalid pinned author-code manifest")

    runner_source = RUNNER_ROOT / "zsy2026_author_code.R"
    if not runner_source.is_file():
        raise AdapterUnavailable(f"missing VCAM-Lasso R runner: {runner_source}")
    parent, parent_strategy = _ascii_stage_parent()
    with tempfile.TemporaryDirectory(prefix="vcam-zsy2026-", dir=parent) as temporary:
        root = Path(temporary)
        if not _path_is_ascii(root):
            raise AdapterUnavailable(f"generated non-ASCII staging path: {root}")
        staged_runner = root / "runner" / runner_source.name
        staged_runner.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runner_source, staged_runner)

        source_hashes: dict[str, str] = {
            "runner/zsy2026_author_code.R": sha256_file(runner_source)
        }
        staged_hashes: dict[str, str] = {
            "runner/zsy2026_author_code.R": sha256_file(staged_runner)
        }
        if source_hashes["runner/zsy2026_author_code.R"] != staged_hashes[
            "runner/zsy2026_author_code.R"
        ]:
            raise AdapterUnavailable("VCAM-Lasso runner hash changed while staging")

        staged_vendor_root = root / "vendor" / "zsy2026_vcampackage"
        for raw_relative, expected_hash in sorted(source_files.items()):
            relative = _safe_vendor_relative_path(raw_relative)
            source = VENDOR_ROOT / relative
            staged = staged_vendor_root / relative
            observed_source = sha256_file(source)
            if observed_source != str(expected_hash):
                raise AdapterUnavailable(
                    f"pinned author-code hash changed while staging: {relative}"
                )
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            observed_staged = sha256_file(staged)
            key = "vendor/" + relative.as_posix()
            source_hashes[key] = observed_source
            staged_hashes[key] = observed_staged
            if observed_staged != observed_source:
                raise AdapterUnavailable(
                    f"VCAM-Lasso staged copy hash mismatch: {relative}"
                )

        staged_vendor_source = staged_vendor_root / "R" / "VCAMLasso.R"
        if not staged_vendor_source.is_file():
            raise AdapterUnavailable("staged VCAM-Lasso source is missing")
        yield ZSY2026AsciiStage(
            root=root,
            runner_path=staged_runner,
            vendor_source_path=staged_vendor_source,
            parent_strategy=parent_strategy,
            source_sha256=source_hashes,
            staged_sha256=staged_hashes,
            vendor_commit=str(vendor["commit"]),
        )


def _r_runtime() -> tuple[str | None, dict[str, str]]:
    candidates = [
        os.environ.get("VCAM_RSCRIPT"),
        r"C:\Users\24481084\.cache\vcam-r\R-4.6.1\bin\Rscript.exe",
        shutil.which("Rscript"),
    ]
    executable = next((item for item in candidates if item and Path(item).is_file()), None)
    environment = os.environ.copy()
    # Codex's Windows shell can inherit the POSIX locale ``C.UTF-8``.  The
    # bundled UCRT R reports that locale as unsupported, falls back to the
    # byte-oriented ``C`` locale, and then rejects the UTF-8 comments in the
    # *unmodified* author source.  Let Windows R select its own UTF-8 locale
    # instead.  This changes only the child-process environment, never the
    # source file or the user's shell locale.
    if os.name == "nt":
        for name in ("LANG", "LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_MONETARY", "LC_TIME"):
            if environment.get(name, "").strip().lower() in {"c.utf-8", "c.utf8"}:
                environment.pop(name, None)
    cached_library = Path(r"C:\Users\24481084\.cache\vcam-r\library-4.6.1")
    if cached_library.is_dir():
        environment.setdefault("R_LIBS_USER", str(cached_library))
    return executable, environment


def _r_compatible_seed(seed: int) -> int:
    """Map an unsigned replication seed into R's accepted integer range."""

    return int(seed) % 2_147_483_647


def _probe_r(packages: tuple[str, ...]) -> PreflightReport:
    executable, environment = _r_runtime()
    if executable is None:
        return PreflightReport(False, "unavailable", "rscript_missing", "Rscript was not found")
    package_expression = ",".join(f'"{item}"' for item in packages)
    expression = (
        f"p<-c({package_expression}); ok<-vapply(p,requireNamespace,logical(1),quietly=TRUE); "
        "cat(R.version.string,'\\n'); "
        "for(i in seq_along(p)) cat(p[[i]], if(ok[[i]]) as.character(packageVersion(p[[i]])) else 'MISSING', '\\n'); "
        "if(!all(ok)) quit(status=17)"
    )
    completed = subprocess.run(
        [executable, "--vanilla", "-e", expression],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=60,
        check=False,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    runtime = lines[0] if lines else "R version unknown"
    versions = {line.split()[0]: " ".join(line.split()[1:]) for line in lines[1:] if len(line.split()) >= 2}
    if completed.returncode != 0:
        return PreflightReport(
            False,
            runtime,
            "r_dependency_failure",
            (completed.stderr or completed.stdout).strip(),
            {"rscript": executable, "packages": versions},
        )
    version = runtime + "; " + "; ".join(f"{name}-{versions.get(name, 'unknown')}" for name in packages)
    return PreflightReport(
        True, version, environment={"rscript": executable, "packages": versions}
    )


def _write_observations(path: Path, data: SubjectDataset) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_id", "subject_id", "time", "response", *data.covariate_names])
        for index in range(data.n_rows):
            writer.writerow(
                [
                    data.row_id[index],
                    data.subject_id[index],
                    format(float(data.time[index]), ".17g"),
                    format(float(data.response[index]), ".17g"),
                    *(format(float(value), ".17g") for value in data.covariates[index]),
                ]
            )


def _run_r(
    arguments: list[str], *, timeout: int, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    executable, environment = _r_runtime()
    if executable is None:
        raise AdapterUnavailable("Rscript was not found")
    completed = subprocess.run(
        [executable, "--vanilla", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        cwd=None if cwd is None else str(cwd),
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise AdapterUnavailable(f"R process exited {completed.returncode}: {message}")
    return completed


@dataclass
class ZW2015Model:
    curves: dict[str, tuple[np.ndarray, np.ndarray]]
    fitted_by_row: dict[str, float]


def _validate_fdapace_spline_spec(
    n_knot: tuple[int, ...], order: tuple[int, ...], *, stage: str
) -> None:
    """Fail early on the public package's stricter knot convention.

    ``fdapace::VCAM`` computes ``nIntKnot = nKnot - order - 1`` and its
    ``GenBSpline`` then requires ``nIntKnot >= order``.  This is not the same
    parameterization as the paper's printed number of interior knots.
    """

    if len(n_knot) != len(order):
        raise ValueError(f"{stage} nKnot and order vectors must have equal lengths")
    infeasible = [
        index
        for index, (count, degree) in enumerate(zip(n_knot, order, strict=True))
        if degree < 0 or count - degree - 1 < degree
    ]
    if infeasible:
        raise ValueError(
            f"fdapace::VCAM {stage} spline specification is infeasible at "
            f"blocks {infeasible}: the public package requires "
            "nKnot - order - 1 >= order; paper interior-knot counts cannot "
            "be passed as package nKnot values"
        )


class ZW2015Adapter(BenchmarkAdapter):
    label = MethodLabel.ZW2015.value

    def preflight(self) -> PreflightReport:
        return _probe_r(("fdapace",))

    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        del seed
        p = train.covariates.shape[1]
        for subject in train.subjects:
            rows = train.subject_id == subject
            if np.max(np.ptp(train.covariates[rows], axis=0)) > 1e-10:
                raise ValueError("ZW2015 requires time-invariant covariates within subject")
        add_nknot = _integer_vector(tuning.get("add_nknot", 10), p)
        add_order = _integer_vector(tuning.get("add_order", 3), p)
        vc_nknot = _integer_vector(tuning.get("vc_nknot", 10), p + 1)
        vc_order = _integer_vector(tuning.get("vc_order", 3), p + 1)
        _validate_fdapace_spline_spec(add_nknot, add_order, stage="additive")
        _validate_fdapace_spline_spec(vc_nknot, vc_order, stage="varying-coefficient")
        grid_size = int(tuning.get("grid_size", 201))
        timeout = int(tuning.get("timeout_seconds", 900))
        time_domain_value = tuning.get("time_domain", train.metadata.get("time_domain"))
        time_domain = (
            (float(np.min(train.time)), float(np.max(train.time)))
            if time_domain_value is None
            else (float(time_domain_value[0]), float(time_domain_value[1]))  # type: ignore[index]
        )
        covariate_value = tuning.get("covariate_domains", train.metadata.get("covariate_domains"))
        covariate_domains = (
            tuple(
                (float(np.min(train.covariates[:, index])), float(np.max(train.covariates[:, index])))
                for index in range(p)
            )
            if covariate_value is None
            else tuple((float(item[0]), float(item[1])) for item in covariate_value)  # type: ignore[union-attr]
        )
        with tempfile.TemporaryDirectory(prefix="vcam-zw2015-") as temporary:
            root = Path(temporary)
            input_path = root / "observations.csv"
            curves_path = root / "curves.csv"
            fitted_path = root / "fitted.csv"
            _write_observations(input_path, train)
            _run_r(
                [
                    str(RUNNER_ROOT / "zw2015_fdapace.R"),
                    str(input_path),
                    str(curves_path),
                    str(fitted_path),
                    ",".join(map(str, add_nknot)),
                    ",".join(map(str, add_order)),
                    ",".join(map(str, vc_nknot)),
                    ",".join(map(str, vc_order)),
                    str(grid_size),
                    ",".join(format(item, ".17g") for item in time_domain),
                    ";".join(
                        ",".join(format(item, ".17g") for item in domain)
                        for domain in covariate_domains
                    ),
                ],
                timeout=timeout,
            )
            curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            with curves_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for component in sorted({row["component"] for row in rows}):
                selected = [row for row in rows if row["component"] == component]
                grid = np.asarray([float(row["grid"]) for row in selected])
                values = np.asarray([float(row["value"]) for row in selected])
                order = np.argsort(grid)
                curves[component] = (grid[order], values[order])
            with fitted_path.open("r", encoding="utf-8", newline="") as handle:
                fitted = {
                    row["row_id"]: float(row["prediction"]) for row in csv.DictReader(handle)
                }
        model = ZW2015Model(curves, fitted)
        version = self.preflight().version
        return FitArtifact(
            model=model,
            method=self.label,
            version=version,
            tuning={
                **dict(tuning),
                "add_nknot": list(add_nknot),
                "add_order": list(add_order),
                "vc_nknot": list(vc_nknot),
                "vc_order": list(vc_order),
                "grid_size": grid_size,
                "time_domain": list(time_domain),
                "covariate_domains": [list(item) for item in covariate_domains],
            },
            converged=True,
            metadata={
                "implementation_origin": "CRAN fdapace::VCAM",
                "plot_device": "isolated temporary PDF device removed with adapter temp directory",
                "normalization": "unmodified package behavior",
                "source_paper_knot_parameterization": "K denotes the number of interior knots",
                "public_package_knot_parameterization": (
                    "nIntKnot=nKnot-order-1, with GenBSpline requiring nIntKnot>=order"
                ),
                "source_paper_bic_available_in_public_package": False,
                "tuning_departure": (
                    "The unmodified public package is run with the recorded feasible nKnot "
                    "vectors; its API cannot reproduce the paper's smaller BIC-selected "
                    "interior-knot counts and exposes no AIC/BIC selector."
                ),
            },
        )

    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> np.ndarray:
        model: ZW2015Model = artifact.model
        baseline_grid, baseline_values = model.curves["baseline"]
        prediction = _strict_interp(test.time, baseline_grid, baseline_values)
        for index in range(test.covariates.shape[1]):
            beta_grid, beta_values = model.curves[f"beta_{index + 1}"]
            phi_grid, phi_values = model.curves[f"phi_{index + 1}"]
            prediction += _strict_interp(test.time, beta_grid, beta_values) * _strict_interp(
                test.covariates[:, index], phi_grid, phi_values
            )
        return prediction

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        model: ZW2015Model = artifact.model
        result = []
        for component, (grid, values) in model.curves.items():
            domain = "time" if component == "baseline" or component.startswith("beta_") else "covariate_" + component.split("_")[-1]
            result.append(
                {"component": component, "domain": domain, "grid": grid.tolist(), "values": values.tolist()}
            )
        return tuple(result)


def _integer_vector(value: object, length: int) -> tuple[int, ...]:
    if np.isscalar(value):
        return (int(value),) * length
    result = tuple(int(item) for item in value)  # type: ignore[union-attr]
    if len(result) != length:
        raise ValueError(f"expected {length} integers")
    return result


def _zsy_source_df(n_covariates: int, value: object | None = None) -> tuple[int, ...]:
    """Return the spline dimensions used by Zhao--Sun--Yang's source design.

    The author implementation expects one dimension for the baseline followed
    by one coefficient and one additive dimension per covariate.  In the
    published high-dimensional experiment the cubic baseline has four
    interior knots (dimension eight), while every other cubic spline has two
    interior knots (dimension six).
    """

    if n_covariates < 1:
        raise ValueError("ZSY2026 requires at least one covariate")
    width = 1 + 2 * n_covariates
    dimensions = (
        (8,) + (6,) * (width - 1)
        if value is None
        else _integer_vector(value, width)
    )
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("ZSY2026 spline dimensions must be positive")
    return dimensions


def _strict_interp(values: np.ndarray, grid: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    if np.any(values < grid[0] - 1e-10) or np.any(values > grid[-1] + 1e-10):
        raise ValueError("held-out values fall outside the original method's fitted grid")
    return np.interp(values, grid, estimates)


@dataclass
class ZSY2026Model:
    row_id: tuple[str, ...]
    time: np.ndarray
    covariates: np.ndarray
    result_columns: dict[str, np.ndarray]
    author_reported_mse: float


class ZSY2026AuthorCodeAdapter(BenchmarkAdapter):
    label = MethodLabel.ZSY2026_AUTHOR_CODE.value

    def preflight(self) -> PreflightReport:
        vendor = verify_zsy2026_vendor()
        if not vendor["valid"]:
            return PreflightReport(
                False,
                "VCAMLasso-0.1.0",
                "vendor_hash_mismatch",
                f"missing={vendor['missing']}; mismatched={vendor['mismatched']}",
                {"commit": vendor["commit"]},
            )
        report = _probe_r(("glmnet", "splines2"))
        return PreflightReport(
            report.ready,
            f"VCAMLasso-0.1.0@{vendor['commit']}; {report.version}",
            report.code,
            report.message,
            {**dict(report.environment), "vendor_commit": vendor["commit"]},
        )

    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        vendor = verify_zsy2026_vendor()
        if not vendor["valid"]:
            raise AdapterUnavailable("pinned author-code hash verification failed")
        p = train.covariates.shape[1]
        degrees = _zsy_source_df(p, tuning.get("df"))
        timeout = int(tuning.get("timeout_seconds", 1800))
        r_seed = _r_compatible_seed(seed)
        with _stage_zsy2026_author_code() as stage:
            root = stage.root
            input_path = root / "observations.csv"
            result_path = root / "result.csv"
            mse_path = root / "mse.txt"
            _write_observations(input_path, train)
            _run_r(
                [
                    str(stage.runner_path),
                    str(stage.vendor_source_path),
                    str(input_path),
                    str(result_path),
                    str(mse_path),
                    str(p),
                    ",".join(map(str, degrees)),
                    str(r_seed),
                ],
                timeout=timeout,
                cwd=root,
            )
            with result_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            columns = {
                name: np.asarray([float(row[name]) for row in rows], dtype=float)
                for name in rows[0]
                if name != "row_id"
            }
            row_id = tuple(row["row_id"] for row in rows)
            author_mse = float(mse_path.read_text(encoding="utf-8").strip())
        selected = tuple(
            index
            for index in range(p)
            if np.linalg.norm(columns[f"beta{index + 1}"] * columns[f"phi{index + 1}"]) > 1e-10
        )
        return FitArtifact(
            model=ZSY2026Model(row_id, train.time.copy(), train.covariates.copy(), columns, author_mse),
            method=self.label,
            version=self.preflight().version,
            tuning={
                **dict(tuning),
                "df": list(degrees),
                "replication_seed": int(seed),
                "seed_forwarded_to_R": r_seed,
            },
            converged=True,
            selected_blocks=selected,
            metadata={
                "implementation_origin": "unmodified pinned author code",
                "vendor_commit": vendor["commit"],
                "vendor_source_sha256": vendor["origin"]["files"]["R/VCAMLasso.R"],
                **stage.audit_metadata(),
                "author_code_r_locale_strategy": (
                    "Windows R child removes unsupported inherited C.UTF-8 locale "
                    "variables before sourcing the unchanged UTF-8 author bytes"
                ),
                "replication_seed": int(seed),
                "r_seed": r_seed,
                "r_seed_mapping": "replication_seed modulo 2147483647",
                "author_returned_mse": author_mse,
                "author_mse_warning": "author return is square(mean residual), retained verbatim and not used as benchmark MSPE",
                "prediction_capability": "none for held-out rows",
                "known_differences_from_paper": vendor["origin"]["known_interface_limits"],
            },
        )

    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> np.ndarray:
        raise UnsupportedPrediction(
            "The pinned author function returns fitted rows but no coefficients/predict method; no silent adapter is permitted."
        )

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        model: ZSY2026Model = artifact.model
        curves = []
        baseline_order = np.argsort(model.time)
        curves.append(
            {
                "component": "baseline",
                "domain": "time-observed-design",
                "grid": model.time[baseline_order].tolist(),
                "values": model.result_columns["beta0"][baseline_order].tolist(),
            }
        )
        p = model.covariates.shape[1]
        for index in range(p):
            time_order = np.argsort(model.time)
            x_order = np.argsort(model.covariates[:, index])
            curves.extend(
                [
                    {
                        "component": f"beta_{index + 1}",
                        "domain": "time-observed-design",
                        "grid": model.time[time_order].tolist(),
                        "values": model.result_columns[f"beta{index + 1}"][time_order].tolist(),
                    },
                    {
                        "component": f"phi_{index + 1}",
                        "domain": f"covariate_{index + 1}-observed-design",
                        "grid": model.covariates[x_order, index].tolist(),
                        "values": model.result_columns[f"phi{index + 1}"][x_order].tolist(),
                    },
                ]
            )
        return tuple(curves)

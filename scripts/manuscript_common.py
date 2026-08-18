"""Shared helpers for building the manuscript tables and figures.

The strict benchmark already stores, for every successful fit, the estimated
baseline and the identified factor curves on a common grid.  This module reads
those streams and recomputes componentwise accuracy so that the paper can be
laid out the way the source literature lays out its own tables, without
re-running any simulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dgp import Truth, generate_zw2015, zzw2020_truth  # noqa: E402


# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

METHOD_ORDER: tuple[str, ...] = (
    "TRACE-VCAM",
    "ZW2015",
    "ZZW2020",
    "HHY2021-Huber",
    "ZSY2026-author-code",
    "ZY2025-paper-implementation",
)

#: Short labels used inside tables and figure legends.  The long descriptive
#: names stay in the section text, where the source paper is cited once.
SHORT_NAME: dict[str, str] = {
    "TRACE-VCAM": "TRACE",
    "ZW2015": "ZW",
    "ZZW2020": "ZZW",
    "HHY2021-Huber": "HHY",
    "ZSY2026-author-code": "ZSY",
    "ZY2025-paper-implementation": "ZY",
}

LONG_NAME: dict[str, str] = {
    "TRACE-VCAM": "TRACE-VCAM",
    "ZW2015": "Zhang and Wang (2015)",
    "ZZW2020": "Zhang, Zhong and Wang (2020)",
    "HHY2021-Huber": "Hu, Huang and You (2021)",
    "ZSY2026-author-code": "Zhao, Sun and Yang (2026)",
    "ZY2025-paper-implementation": "Zhao and Yang (2025)",
}


# ---------------------------------------------------------------------------
# Registered truths
# ---------------------------------------------------------------------------


def _zsy2026_truth(n_covariates: int = 10) -> Truth:
    from experiments.dgp import _spline_function

    beta0 = _spline_function(
        np.array([1.0, 2.0, 4.0, 3.0, -2.0, 0.0, 3.0, 6.0]), (0.0, 2.0), 4
    )
    beta_common = _spline_function(
        np.array([0.0, 6.0, 2.0, 0.0, 3.0, 1.0]), (0.0, 2.0), 2, normalize_average=True
    )
    phi_common = _spline_function(
        np.array([3.0, 0.0, 4.0, 2.0, 0.0, 1.0]), (0.0, 1.0), 2, center_integral=True
    )
    return Truth(
        beta0=beta0,
        beta=tuple(beta_common for _ in range(n_covariates)),
        phi=tuple(phi_common for _ in range(n_covariates)),
        active=tuple(True for _ in range(n_covariates)),
    )


def registered_truth(scenario: str) -> tuple[Truth, tuple[float, float], int]:
    """Return the truth, the time domain, and the covariate count of a scenario."""

    if scenario.startswith("example1"):
        return generate_zw2015(seed=0, n_subjects=2).truth, (0.0, 1.0), 2
    if scenario.startswith("example2"):
        return zzw2020_truth(), (0.0, 2.0), 2
    if scenario.startswith("example3") or scenario.startswith("scaling"):
        return _zsy2026_truth(10), (0.0, 2.0), 10
    raise ValueError(f"unregistered scenario: {scenario}")


# ---------------------------------------------------------------------------
# Componentwise accuracy
# ---------------------------------------------------------------------------

GRID_SIZE = 201


def _domain_average_squared_error(grid: np.ndarray, error: np.ndarray) -> float:
    length = float(grid[-1] - grid[0])
    if length <= 0.0:
        return float("nan")
    return float(np.trapezoid(error**2, grid) / length)


def curve_map(record: Mapping[str, object]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for curve in record.get("curves", ()):  # type: ignore[union-attr]
        name = str(curve.get("component", ""))
        grid = np.asarray(curve.get("grid", ()), dtype=float)
        values = np.asarray(curve.get("values", ()), dtype=float)
        if grid.size >= 2 and grid.shape == values.shape:
            mapped[name] = (grid, values)
    return mapped


def componentwise_errors(
    record: Mapping[str, object], *, max_blocks: int = 2
) -> dict[str, float]:
    """Return the domain-averaged squared error of each identified component.

    A block the method did not deliver is scored at the zero estimate, matching
    the Monte Carlo convention of the strict benchmark: a missed component is a
    modelling error, not a reason to drop the replication.
    """

    scenario = str(record["scenario"])
    truth, time_domain, n_covariates = registered_truth(scenario)
    mapped = curve_map(record)
    t_grid = np.linspace(time_domain[0], time_domain[1], GRID_SIZE)
    z_grid = np.linspace(0.0, 1.0, GRID_SIZE)

    errors: dict[str, float] = {}
    if "baseline" in mapped:
        grid, values = mapped["baseline"]
        baseline = np.interp(t_grid, grid, values)
    else:
        baseline = np.zeros_like(t_grid)
    errors["beta_0"] = _domain_average_squared_error(t_grid, baseline - truth.beta0(t_grid))

    surface_total = 0.0
    for index in range(n_covariates):
        if not truth.active[index]:
            continue
        beta_true = truth.beta[index](t_grid)
        phi_true = truth.phi[index](z_grid)
        beta_key, phi_key = f"beta_{index + 1}", f"phi_{index + 1}"
        if beta_key in mapped and phi_key in mapped:
            grid, values = mapped[beta_key]
            beta_hat = np.interp(t_grid, grid, values)
            grid, values = mapped[phi_key]
            phi_hat = np.interp(z_grid, grid, values)
        else:
            beta_hat = np.zeros_like(t_grid)
            phi_hat = np.zeros_like(z_grid)
        if index < max_blocks:
            errors[f"beta_{index + 1}"] = _domain_average_squared_error(
                t_grid, beta_hat - beta_true
            )
            errors[f"phi_{index + 1}"] = _domain_average_squared_error(
                z_grid, phi_hat - phi_true
            )
        surface_total += float(
            np.mean((beta_hat[:, None] * phi_hat[None, :] - beta_true[:, None] * phi_true[None, :]) ** 2)
        )
    errors["surface"] = surface_total
    return errors


def iter_curve_records(
    path: Path, *, scenarios: Sequence[str] | None = None
) -> Iterator[dict[str, object]]:
    wanted = None if scenarios is None else set(scenarios)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if wanted is not None and str(record.get("scenario")) not in wanted:
                continue
            yield record


def stacked_curves(
    path: Path, scenario: str, method: str, components: Iterable[str]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per component, the common grid, a replication-by-grid matrix, and the labels.

    The replication labels travel with the curves so that a display can put
    several methods on one and the same generated data set, which is how the
    protocol runs them, instead of on whatever replication happens to occupy the
    same row of each method's matrix.
    """

    wanted = list(components)
    grids: dict[str, np.ndarray] = {}
    stacks: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    labels: dict[str, list[int]] = {name: [] for name in wanted}
    for record in iter_curve_records(path, scenarios=[scenario]):
        if str(record.get("method")) != method:
            continue
        mapped = curve_map(record)
        for name in wanted:
            if name not in mapped:
                continue
            grid, values = mapped[name]
            if name not in grids:
                grids[name] = grid
            stacks[name].append(np.interp(grids[name], grid, values))
            labels[name].append(int(record["replicate"]))
    return {
        name: (
            grids[name],
            np.vstack(stacks[name]),
            np.asarray(labels[name], dtype=int),
        )
        for name in wanted
        if name in grids and stacks[name]
    }


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def fmt(value: float, digits: int = 4) -> str:
    """Format one table entry the way the source literature does."""

    if value is None or not np.isfinite(value):
        return "--"
    if value != 0.0 and (abs(value) >= 10.0**4 or abs(value) < 10.0**-(digits - 1)):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{value:.{digits}f}"


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def latex_escape(text: str) -> str:
    for source, target in (("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        text = text.replace(source, target)
    return text

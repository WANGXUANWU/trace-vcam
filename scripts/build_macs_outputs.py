"""Build the application tables and figures from the audited MACS output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_manuscript_outputs import (  # noqa: E402
    FIGURE_WIDTH_IN,
    METHOD_COLOR,
    _tidy_log_axis,
    panel,
)
from scripts.manuscript_common import METHOD_ORDER, SHORT_NAME, bold, fmt  # noqa: E402

VARIANT_LABEL = {
    "primary": "Primary analysis",
    "delete_outer_fence_subjects": "Delete outlying subjects",
    "winsorize_response_1_99": "Winsorise response",
    "basis_5": "Basis size $K=5$",
    "basis_8": "Basis size $K=8$",
}


# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------


def fold_metrics(
    predictions: pd.DataFrame, results: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-fold prediction summaries computed from the stored row predictions.

    A method can leave predictions for a fold that it then reports as a failed
    fit, for instance when a finite fit does not meet its stopping rule.  When
    the result stream is supplied, those folds are dropped, so that the summary
    and the completion record describe the same set of fits.
    """

    if results is not None:
        successful = set(
            zip(
                results.loc[results.attempt_status == "success", "variant"],
                results.loc[results.attempt_status == "success", "repeat"].astype(int),
                results.loc[results.attempt_status == "success", "fold"].astype(int),
                results.loc[results.attempt_status == "success", "method"],
            )
        )
        keep = [
            (v, int(r), int(f), m) in successful
            for v, r, f, m in zip(
                predictions["variant"],
                predictions["repeat"],
                predictions["fold"],
                predictions["method"],
            )
        ]
        predictions = predictions[keep]

    records = []
    grouped = predictions.groupby(["variant", "repeat", "fold", "method"], sort=False)
    for (variant, repeat, fold, method), block in grouped:
        residual = block["prediction"].to_numpy() - block["observed_cd4"].to_numpy()
        squared = residual**2
        subject = block["subject_id"].to_numpy()
        balanced = float(
            np.mean([np.mean(squared[subject == s]) for s in np.unique(subject)])
        )
        keep = squared <= np.quantile(squared, 0.9)
        records.append(
            {
                "variant": variant,
                "repeat": int(repeat),
                "fold": int(fold),
                "method": method,
                "mspe": float(np.mean(squared)),
                "balanced_mspe": balanced,
                "mape": float(np.mean(np.abs(residual))),
                "trimmed_mspe": float(np.mean(squared[keep])),
            }
        )
    return pd.DataFrame(records)


def common_folds(frame: pd.DataFrame, methods: Sequence[str]) -> set[tuple[int, int]]:
    common: set[tuple[int, int]] | None = None
    for method in methods:
        keys = set(
            zip(
                frame[frame.method == method]["repeat"],
                frame[frame.method == method]["fold"],
            )
        )
        common = keys if common is None else (common & keys)
    return common or set()


def _number(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) >= 10.0**5:
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{value:,.{digits}f}".replace(",", r"\,")


def _cell(values: np.ndarray, *, best: bool, digits: int) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "--"
    centre = float(np.mean(finite))
    spread = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
    body = f"{_number(centre, digits)} ({_number(spread, digits)})"
    return bold(body) if best else body


def macs_main_table(
    metrics: pd.DataFrame,
    runtimes: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> tuple[str, dict[str, object]]:
    frame = metrics[metrics.variant == "primary"]
    attempted = frame.groupby("method").size().max()
    # A method that returns a fit on only a minority of folds cannot be compared
    # on a common set of held-out subjects without discarding most of the
    # experiment; it is reported in the Supplementary Material instead.
    methods = [
        m
        for m in METHOD_ORDER
        if m in set(frame.method)
        and (frame.method == m).sum() >= 0.8 * attempted
    ]
    shared = common_folds(frame, methods)
    restricted = frame[
        [(int(r), int(f)) in shared for r, f in zip(frame["repeat"], frame["fold"])]
    ]
    run = runtimes[runtimes.variant == "primary"]
    run = run[
        [(int(r), int(f)) in shared for r, f in zip(run["repeat"], run["fold"])]
    ]

    columns = (
        ("mspe", r"MSPE ($\times10^{3}$)", 1, 1e-3),
        ("balanced_mspe", r"Balanced MSPE ($\times10^{3}$)", 1, 1e-3),
        ("mape", "MAPE", 2, 1.0),
        ("runtime_seconds", "Time (s)", 2, 1.0),
    )
    values = {}
    for method in methods:
        row = {}
        for key, _, _, scale in columns:
            source = run if key == "runtime_seconds" else restricted
            row[key] = source[source.method == method][key].to_numpy(dtype=float) * scale
        values[method] = row
    winners = {}
    for key, _, _, _ in columns:
        finite = {
            m: float(np.mean(values[m][key]))
            for m in methods
            if values[m][key].size and np.all(np.isfinite(values[m][key]))
        }
        winners[key] = min(finite, key=finite.get) if finite else None

    caption = caption.replace("NFOLDS", str(len(shared)))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}l" + "c" * len(columns) + r"@{}}",
        r"\toprule",
        "Method & " + " & ".join(title for _, title, _, _ in columns) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        row = [SHORT_NAME[method]]
        for key, _, digits, _ in columns:
            row.append(
                _cell(values[method][key], best=winners[key] == method, digits=digits)
            )
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    audit = {
        "common_folds": len(shared),
        "methods": methods,
        "winner_by_metric": {key: winners[key] for key, _, _, _ in columns},
        "means": {
            SHORT_NAME[m]: {k: float(np.mean(values[m][k])) for k, _, _, _ in columns}
            for m in methods
        },
    }
    return "\n".join(lines), audit


def macs_completion_table(
    results: pd.DataFrame, *, caption: str, label: str
) -> str:
    """Attempted and successful fits per method and analysis."""

    applicable = results[results.applicability == "applicable"]
    variants = [v for v in VARIANT_LABEL if v in set(applicable.variant)]
    methods = [m for m in METHOD_ORDER if m in set(applicable.method)]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\begin{tabular}{@{}l" + "c" * len(variants) + r"@{}}",
        r"\toprule",
        "Method & " + " & ".join(VARIANT_LABEL[v] for v in variants) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        row = [SHORT_NAME[method]]
        for variant in variants:
            sub = applicable[
                (applicable.method == method) & (applicable.variant == variant)
            ]
            folds = sub[sub["mode"] == "formal"] if "mode" in sub.columns else sub
            ok = int((folds.attempt_status == "success").sum())
            row.append(f"{ok}/{len(folds)}" if len(folds) else r"\emph{N/A}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def macs_sensitivity_table(
    metrics: pd.DataFrame, *, caption: str, label: str
) -> str:
    primary = metrics[metrics.variant == "primary"]
    attempted = primary.groupby("method").size().max() if len(primary) else 0
    methods = [
        m
        for m in METHOD_ORDER
        if m in set(metrics.method)
        and (primary.method == m).sum() >= 0.8 * attempted
    ]
    variants = [v for v in VARIANT_LABEL if v in set(metrics.variant)]
    if not variants or not methods:
        return ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l" + "c" * len(methods) + r"@{}}",
        r"\toprule",
        "Analysis & " + " & ".join(SHORT_NAME[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    for variant in variants:
        frame = metrics[metrics.variant == variant]
        shared = common_folds(frame, methods)
        restricted = frame[
            [(int(r), int(f)) in shared for r, f in zip(frame["repeat"], frame["fold"])]
        ]
        centres = {
            m: restricted[restricted.method == m]["mspe"].to_numpy(dtype=float) * 1e-3
            for m in methods
        }
        finite = {m: float(np.mean(v)) for m, v in centres.items() if v.size}
        best = min(finite, key=finite.get) if finite else None
        row = [VARIANT_LABEL[variant]]
        for method in methods:
            row.append(_cell(centres[method], best=best == method, digits=1))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

CURVE_TITLE = {
    "baseline": panel("%s", r"Baseline $\beta_0$"),
    "beta_1": panel("%s", r"$\beta_1$: Time modulation of age"),
    "phi_1": panel("%s", r"$\phi_1$: Age effect"),
    "beta_2": panel("%s", r"$\beta_2$: Time modulation of CES-D"),
    "phi_2": panel("%s", r"$\phi_2$: CES-D effect"),
}

#: The registered analysis maps every observed domain linearly onto [0,1].  The
#: figures undo that mapping, so that the horizontal axes carry the units the
#: data were recorded in and the reader can place a curve without translating a
#: scaled coordinate first.  The bounds are the ones stored with the prepared
#: data set; ``age`` is reported as age at seroconversion in years, undoing the
#: centring at 30 years that ``catdata::aids`` supplies.
COORDINATE_AXIS = {
    "time": ("time", 0.0, "Years since seroconversion"),
    "age": ("age", 30.0, "Age at seroconversion (years)"),
    "cesd": ("cesd", 0.0, "CES-D score"),
}
CURVE_COORDINATE = {
    "baseline": "time",
    "beta_1": "time",
    "phi_1": "age",
    "beta_2": "time",
    "phi_2": "cesd",
}
#: Vertical axes are not comparable across panels, and the identification in
#: (2) is what makes them incomparable: a time factor integrates to one and is
#: therefore a unitless modulation, while the covariate factor carries the whole
#: amplitude of the component and is read in cells.  Each panel says which it
#: is, so that the difference in vertical scale is not read as a difference in
#: effect size.
CURVE_YLABEL = {
    "baseline": "CD4 cells",
    "beta_1": "Relative effect",
    "phi_1": "CD4 cells",
    "beta_2": "Relative effect",
    "phi_2": "CD4 cells",
}

#: Bounds of the observed domains, as recorded with the prepared primary data
#: set.  They are duplicated here so that a figure can be rebuilt from stored
#: output alone, and they are checked against the data when it is available.
COORDINATE_BOUNDS = {
    "time": (-2.989733, 5.459274),
    "age": (-11.29, 29.08),
    "cesd": (-7.0, 49.0),
}

#: Competitors shown alongside the proposed estimator in the application
#: figure, in the order they are drawn.  They are reference curves rather than a
#: second result of equal standing -- only one competitor returns a converged
#: full-data fit at all -- so they are drawn thin and grey, one visual level
#: below the estimate the figure is about.
MACS_COMPARISON_STYLE = {
    "HHY2021-Huber": ((0, (4.0, 1.6)), 0.9),
    "ZY2025-paper-implementation": ((0, (5, 1, 1, 1)), 0.9),
}


#: Central design range of each observed coordinate, as quantiles of its own
#: empirical distribution.  Every curve is estimated on the whole registered
#: [0,1] coordinate, but the ends of that interval carry almost no observations,
#: so a plot drawn over the full interval spends its edges on extrapolation.
#: Only the display is restricted; nothing in the fit or in any reported number
#: changes.
MACS_DISPLAY_QUANTILES = (0.025, 0.975)


def observed_display_ranges(
    quantiles: tuple[float, float] = MACS_DISPLAY_QUANTILES,
) -> dict[str, tuple[float, float]]:
    """Central design range of time and of each covariate, in scaled units."""

    from scripts.run_macs_application import prepare_macs_variant, read_macs_csv

    dataset = prepare_macs_variant(
        read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv"), variant="primary"
    )
    ranges = {
        "time": tuple(np.quantile(dataset.time, quantiles)),
        "age": tuple(np.quantile(dataset.covariates[:, 0], quantiles)),
        "cesd": tuple(np.quantile(dataset.covariates[:, 1], quantiles)),
    }
    return {key: (float(low), float(high)) for key, (low, high) in ranges.items()}


def _axis_values(grid: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    key = CURVE_COORDINATE[name]
    bound_key, offset, label = COORDINATE_AXIS[key]
    low, high = COORDINATE_BOUNDS[bound_key]
    return low + offset + grid * (high - low), label


def _display_window(
    name: str, ranges: Mapping[str, tuple[float, float]] | None
) -> tuple[float, float] | None:
    """Visible window of a panel, in the same units as ``_axis_values``."""

    if ranges is None:
        return None
    key = CURVE_COORDINATE[name]
    window = ranges.get(key)
    if window is None:
        return None
    bound_key, offset, _ = COORDINATE_AXIS[key]
    low, high = COORDINATE_BOUNDS[bound_key]
    return tuple(low + offset + np.asarray(window) * (high - low))


def observed_design() -> dict[str, np.ndarray]:
    """Observed time and covariate values, in the registered scaled coordinates."""

    from scripts.run_macs_application import prepare_macs_variant, read_macs_csv

    dataset = prepare_macs_variant(
        read_macs_csv(ROOT / "data" / "raw" / "catdata_aids.csv"), variant="primary"
    )
    return {
        "time": np.asarray(dataset.time, dtype=float),
        "age": np.asarray(dataset.covariates[:, 0], dtype=float),
        "cesd": np.asarray(dataset.covariates[:, 1], dtype=float),
    }


def _design_values(name: str, design: Mapping[str, np.ndarray]) -> np.ndarray:
    """Observed values of the coordinate a panel is drawn against, in data units."""

    key = CURVE_COORDINATE[name]
    bound_key, offset, _ = COORDINATE_AXIS[key]
    low, high = COORDINATE_BOUNDS[bound_key]
    return low + offset + np.asarray(design[key], dtype=float) * (high - low)


def _design_strip(axis, values: np.ndarray, *, share: float = 0.05) -> None:
    """Draw a light marginal histogram of the design along the bottom of a panel.

    A curve drawn over a window the design barely visits is an extrapolation,
    and the reader cannot tell that from the curve itself.  The strip puts the
    observed coverage in the same frame.  It is deliberately faint and short: it
    is context for the estimate above it, and a strip drawn heavily enough to be
    read as a second data display competes with the curve for the panel.
    """

    left, right = axis.get_xlim()
    bottom, top = axis.get_ylim()
    edges = np.linspace(left, right, 41)
    counts, _ = np.histogram(values, bins=edges)
    if counts.max() == 0:
        return
    axis.fill_between(
        0.5 * (edges[:-1] + edges[1:]),
        bottom,
        bottom + counts / counts.max() * share * (top - bottom),
        step="mid",
        color="0.55",
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    axis.set_ylim(bottom, top)


#: Legend entries of the application component figure.  They are kept to a few
#: words each so that the legend reads as a key rather than as a second caption;
#: what the two bands mean is stated once, in the caption itself.
#: Display forms of the retention keys.  The keys themselves stay lowercase
#: because they are also the machine-readable names in the returned record.
RETENTION_LABEL = {
    "age only": "Age only",
    "CES-D only": "CES-D only",
    "both": "Both",
    "neither": "Neither",
}

INNER_BAND_LABEL = "IQR, retained draws"
OUTER_BAND_LABEL = "$10\\%$–$90\\%$, all draws"


def _selection_panel(axis, draws: Mapping[str, np.ndarray], *, letter: str) -> dict[str, int]:
    """How often each block survives the fit, over the bootstrap resamples.

    The curve panels condition on retention, because a resample that drops a
    block contributes the zero function and a band mixing the two describes
    neither.  That conditioning is exactly what a reader cannot see in the
    curves, so it is drawn here instead of being left to a sentence.

    The four patterns are exhaustive, so the quantity to read off is an ordering
    and a set of proportions rather than a position in a coordinate system.  The
    panel therefore carries no tick scaffolding, but it does anchor the bars on a
    visible zero rule and reserve a fixed label column, so that it reads as a
    ranked display rather than as four rectangles floating beside the curves.
    The horizontal extent is derived from the largest share, so the bars and
    their value labels fill the panel instead of leaving a third of it blank.
    """

    keep = {
        block: np.asarray(draws[f"draws::retained_{block}"], dtype=float).ravel() > 0.5
        for block in ("1", "2")
        if f"draws::retained_{block}" in draws
    }
    if len(keep) < 2:
        axis.axis("off")
        return {}
    age, cesd = keep["1"], keep["2"]
    counts = {
        "age only": int(np.sum(age & ~cesd)),
        "CES-D only": int(np.sum(~age & cesd)),
        "both": int(np.sum(age & cesd)),
        "neither": int(np.sum(~age & ~cesd)),
    }
    total = int(age.size)
    shares = {key: 100.0 * count / total for key, count in counts.items()}
    widest = max(shares.values()) or 1.0

    # Geometry in data units, all of it a multiple of the largest share, so that
    # the label column, the bars, and the value labels keep their proportions
    # whatever the resampling returns.
    label_column = 0.74 * widest
    value_gap = 0.045 * widest
    label_gap = 0.075 * widest
    right_margin = 0.42 * widest

    positions = np.arange(len(counts))[::-1]
    highlight = METHOD_COLOR["TRACE-VCAM"]
    # The bars grow from a rule rather than from the edge of the frame: without
    # it the eye has no common origin to compare four lengths against.
    axis.axvline(
        0.0,
        color="0.55",
        linewidth=0.7,
        ymin=0.06,
        ymax=0.94,
        zorder=1,
    )
    for position, (key, share) in zip(positions, shares.items()):
        chosen = key == "age only"
        if chosen:
            # One faint stripe across the whole row ties its three parts -- the
            # name, the bar, and the value -- into a single object, so that the
            # highlight is a property of the pattern rather than of the bar.
            axis.axhspan(
                position - 0.44,
                position + 0.44,
                color=highlight,
                alpha=0.07,
                linewidth=0,
                zorder=0,
            )
        if share > 0.0:
            axis.barh(
                position,
                share,
                height=0.52,
                color=highlight if chosen else "0.82",
                linewidth=0,
                zorder=2,
            )
        # A pattern that never occurs is reported as a zero, not as a sliver of
        # bar: a one-pixel rectangle reads as a small positive count.
        axis.text(
            share + value_gap,
            position,
            f"{share:.1f}\\%" if plt.rcParams["text.usetex"] else f"{share:.1f}%",
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold" if chosen else "normal",
            color=highlight if chosen else "0.40",
            zorder=3,
        )
        axis.text(
            -label_gap,
            position + (0.13 if chosen else 0.0),
            RETENTION_LABEL[key],
            va="center",
            ha="right",
            fontsize=8,
            color="0.10" if chosen else "0.42",
            zorder=3,
        )
        if chosen:
            # The one thing a reader has to take from this panel is which
            # pattern the reported fit returns.  Naming it under its own label
            # keeps that inside the left column, where there is room for it,
            # instead of pushing it under a bar where it collides with the
            # value it is meant to explain.
            axis.text(
                -label_gap,
                position - 0.28,
                "the full-data fit",
                va="center",
                ha="right",
                fontsize=6.5,
                style="italic",
                color=highlight,
                zorder=3,
            )
    axis.set_xlim(-label_column, widest + right_margin)
    axis.set_ylim(-0.72, len(counts) - 0.28)
    axis.set_title(panel(letter, "Blocks the fit retains"), pad=3)
    axis.set_xlabel(f"Share of {total:,} resamples", labelpad=1.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    counts["total"] = total
    return counts


def figure_components(
    bootstrap: Mapping[str, object],
    output: Path,
    *,
    components: Sequence[str],
    filename: str,
    height: float,
    layout: tuple[int, int],
    draws: Mapping[str, np.ndarray] | None = None,
    comparison: Mapping[str, Mapping[str, object]] | None = None,
    band: tuple[float, float] = (25.0, 75.0),
    outer_band: tuple[float, float] = (10.0, 90.0),
    display_ranges: Mapping[str, tuple[float, float]] | None = None,
    design: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Fitted curves, two resampling bands, the design, and the block selection.

    Three things a reader needs in order not to over-read the display are put in
    the display rather than in the text.  The inner band conditions on the
    resamples that retain the block, and is the quantity a caption can describe
    as the spread of the estimated shape; the outer band is unconditional, so a
    resample that drops the block enters it as the zero function and the band
    shows what the selection step costs.  The last panel gives the selection
    frequencies themselves, and a marginal strip under each curve gives the
    observed design.  None of this is inference: the resampling does not correct
    for the thresholding, tuning, and decomposition that precede it.
    """

    grid = np.asarray(bootstrap["grid"], dtype=float)
    point = bootstrap["point_estimate"]
    fallback = bootstrap["bands"]
    rows, columns = layout
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(FIGURE_WIDTH_IN, height),
        constrained_layout=True,
        gridspec_kw=(
            {"width_ratios": [1.0] * (columns - 1) + [0.85]} if rows == 1 else None
        ),
    )
    flat = np.atleast_1d(axes).ravel()
    audit: dict[str, object] = {
        "inner_band_percentiles": list(band),
        "outer_band_percentiles": list(outer_band),
        "width": {},
    }
    letters = "abcdefgh"
    for index, (axis, name) in enumerate(zip(flat, components)):
        horizontal, xlabel = _axis_values(grid, name)
        values = np.asarray(point[name], dtype=float)
        drawn: list[np.ndarray] = [values]
        inner_low = inner_high = outer_low = outer_high = None
        if draws is not None and f"draws::{name}" in draws:
            stack = np.asarray(draws[f"draws::{name}"], dtype=float)
            outer_low, outer_high = np.percentile(stack, list(outer_band), axis=0)
            widest_low, widest_high = np.percentile(stack, [2.5, 97.5], axis=0)
            audit.setdefault("outer_contains_zero", {})[name] = bool(
                np.all((widest_low <= 0.0) & (widest_high >= 0.0))
            )
            block = name.split("_")[-1]
            retained_key = f"draws::retained_{block}"
            conditional = stack
            if retained_key in draws:
                keep = np.asarray(draws[retained_key], dtype=float).ravel() > 0.5
                if keep.sum() >= 20:
                    conditional = stack[keep]
                    audit.setdefault("retention", {})[name] = float(np.mean(keep))
            inner_low, inner_high = np.percentile(conditional, list(band), axis=0)
            audit["resamples_used"] = int(conditional.shape[0])
        elif name in fallback:
            inner_low = np.asarray(fallback[name]["lower"], dtype=float)
            inner_high = np.asarray(fallback[name]["upper"], dtype=float)
        if outer_low is not None:
            axis.fill_between(
                horizontal, outer_low, outer_high,
                color=METHOD_COLOR["TRACE-VCAM"], alpha=0.13, linewidth=0,
                label=OUTER_BAND_LABEL, zorder=2,
            )
        if inner_low is not None:
            axis.fill_between(
                horizontal, inner_low, inner_high,
                color=METHOD_COLOR["TRACE-VCAM"], alpha=0.32, linewidth=0,
                label=INNER_BAND_LABEL, zorder=3,
            )
            audit["width"][name] = float(np.mean(inner_high - inner_low))
        # The competitor is a reference curve, not a second result: it is drawn
        # thin and grey so that it cannot be mistaken for a second estimate of
        # equal standing in the reading order of the panel.
        for method, (style, width) in MACS_COMPARISON_STYLE.items():
            entry = (comparison or {}).get(method)
            if not entry or entry.get("attempt_status") != "success":
                continue
            other = np.asarray(entry["curves"][name], dtype=float)
            resampled = np.interp(
                horizontal, np.linspace(horizontal[0], horizontal[-1], other.size), other
            )
            axis.plot(
                horizontal, resampled, color="0.40", linestyle=style, linewidth=width,
                label=f"{SHORT_NAME[method]} full-data fit", zorder=4,
            )
            drawn.append(resampled)
        axis.plot(
            horizontal, values, color=METHOD_COLOR["TRACE-VCAM"], linewidth=1.7,
            label=f"{SHORT_NAME['TRACE-VCAM']} full-data fit", zorder=5,
        )
        axis.axhline(0.0, color="0.6", linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
        axis.set_title(CURVE_TITLE[name] % letters[index], pad=3)
        axis.set_xlabel(xlabel, labelpad=1.5)
        axis.set_ylabel(CURVE_YLABEL[name], labelpad=2)
        window = _display_window(name, display_ranges)
        if window is not None:
            axis.set_xlim(*window)
            inside = (horizontal >= window[0]) & (horizontal <= window[1])
            visible = [item[inside] for item in drawn]
            if outer_low is not None:
                visible += [outer_low[inside], outer_high[inside]]
            elif inner_low is not None:
                visible += [inner_low[inside], inner_high[inside]]
            low = min(float(np.min(item)) for item in visible)
            high = max(float(np.max(item)) for item in visible)
            pad = 0.10 * (high - low)
            axis.set_ylim(low - pad, high + pad)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.22)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        axis.tick_params(length=2.5, width=0.6)
        if design is not None:
            _design_strip(axis, _design_values(name, design))
    if draws is not None and len(flat) > len(components):
        audit["selection"] = _selection_panel(
            flat[len(components)], draws, letter=letters[len(components)]
        )
    for axis in flat[len(components) + 1:]:
        axis.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    order = [
        f"{SHORT_NAME['TRACE-VCAM']} full-data fit",
        INNER_BAND_LABEL,
        OUTER_BAND_LABEL,
        *(f"{SHORT_NAME[method]} full-data fit" for method in MACS_COMPARISON_STYLE),
    ]
    ordered = [label for label in order if label in seen]
    fig.legend(
        [seen[label] for label in ordered],
        ordered,
        loc="outside lower center",
        # One row: the key is four short entries, and stacking them into two
        # rows spends figure height that the curve panels can use instead.
        ncol=min(len(ordered), 4),
        frameon=False,
        fontsize=7.5,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    fig.savefig(output / filename)
    plt.close(fig)
    return audit


def _variant_curves(path: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Full-data curves under each registered perturbation, on a common grid."""

    if not path.exists():
        return np.array([]), {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = np.asarray(payload["grid"], dtype=float)
    out = {
        variant: {
            name: np.asarray(values, dtype=float)
            for name, values in block["curves"].items()
            if not name.startswith("retained_")
        }
        for variant, block in payload["variants"].items()
    }
    return grid, out


def figure_stability(
    grid: np.ndarray,
    curves: Mapping[str, dict],
    output: Path,
    *,
    components: Sequence[str],
    filename: str,
    height: float,
    display_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    styles = {
        "primary": ("-", 1.7, "0.10"),
        "delete_outer_fence_subjects": ("--", 1.2, "#d62728"),
        "winsorize_response_1_99": ("-.", 1.2, "#1f77b4"),
        "basis_5": (":", 1.4, "#2ca02c"),
        "basis_8": ((0, (3, 1, 1, 1)), 1.2, "#9467bd"),
    }
    columns = min(3, len(components))
    single_row = len(components) <= columns
    rows = 1 if single_row else int(np.ceil((len(components) + 1) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(FIGURE_WIDTH_IN, height), constrained_layout=True
    )
    flat = np.atleast_1d(axes).ravel()
    letters = "abcdefgh"
    for index, (axis, name) in enumerate(zip(flat, components)):
        horizontal, xlabel = _axis_values(grid, name)
        for variant, (style, width, color) in styles.items():
            if variant not in curves or name not in curves[variant]:
                continue
            axis.plot(
                horizontal,
                curves[variant][name],
                linestyle=style,
                linewidth=width,
                color=color,
                label=VARIANT_LABEL[variant],
            )
        axis.set_title(CURVE_TITLE[name] % letters[index])
        axis.set_xlabel(xlabel)
        if CURVE_YLABEL[name]:
            axis.set_ylabel(CURVE_YLABEL[name])
        axis.axhline(0.0, color="0.55", linewidth=0.6, linestyle=":")
        window = _display_window(name, display_ranges)
        if window is not None:
            axis.set_xlim(*window)
            inside = (horizontal >= window[0]) & (horizontal <= window[1])
            stacked = [
                np.asarray(block[name])[inside]
                for block in curves.values()
                if name in block
            ]
            low = min(float(np.min(item)) for item in stacked)
            high = max(float(np.max(item)) for item in stacked)
            pad = 0.10 * (high - low)
            axis.set_ylim(low - pad, high + pad)
        axis.grid(linewidth=0.3, alpha=0.35)
        axis.set_axisbelow(True)
    handles, labels = flat[0].get_legend_handles_labels()
    for axis in flat[len(components):]:
        axis.axis("off")
    if single_row:
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=len(labels),
            frameon=False,
            fontsize=7.5,
        )
    else:
        flat[len(components)].legend(
            handles, labels, loc="center", frameon=False, fontsize=7.5
        )
    fig.savefig(output / filename)
    plt.close(fig)


def figure_surfaces(
    grid: np.ndarray,
    curves: Mapping[str, dict],
    output: Path,
    display_ranges: Mapping[str, tuple[float, float]] | None = None,
    design: Mapping[str, np.ndarray] | None = None,
    filename: str = "supp_macs_surfaces.pdf",
) -> dict[str, object]:
    """What the separable form does and does not say about the fitted component.

    The surface here is a deterministic transform of two curves already drawn:
    \\(\\widehat g_1=\\widehat\\beta_1\\widehat\\phi_1\\) adds no statistical
    evidence to them, and drawing it as a smooth filled result invites a reader
    to take a two-dimensional pattern as a two-dimensional finding.  It is drawn
    as an explanation instead.  Panel (a) keeps the zero-centred diverging scale
    the quantity calls for, marks the zero set, which separability forces to be
    a union of horizontal and vertical lines rather than an estimated curve, and
    puts every observed visit on the surface it is supposed to support.  Panel
    (b) deliberately does not repeat that encoding: it reads the same fit as
    fitted mean trajectories at three observed ages, which is the form in which
    the size of the age contribution can be compared with the baseline it
    modulates.
    """

    primary = curves.get("primary")
    if not primary:
        return {}
    if primary.get("phi_1") is None or not np.any(np.abs(primary["phi_1"]) > 1e-10):
        return {}

    times, time_label = _axis_values(grid, "baseline")
    ages, age_label = _axis_values(grid, "phi_1")
    time_window = _display_window("baseline", display_ranges)
    age_window = _display_window("phi_1", display_ranges)
    rows_kept = (
        np.ones_like(times, dtype=bool)
        if time_window is None
        else (times >= time_window[0]) & (times <= time_window[1])
    )
    columns_kept = (
        np.ones_like(ages, dtype=bool)
        if age_window is None
        else (ages >= age_window[0]) & (ages <= age_window[1])
    )
    t_axis, z_axis = times[rows_kept], ages[columns_kept]
    beta = primary["beta_1"][rows_kept]
    phi = primary["phi_1"][columns_kept]
    baseline = primary["baseline"][rows_kept]
    component = np.outer(beta, phi)

    observed_t = observed_z = None
    if design is not None:
        low, high = COORDINATE_BOUNDS["time"]
        observed_t = low + np.asarray(design["time"], dtype=float) * (high - low)
        low, high = COORDINATE_BOUNDS["age"]
        observed_z = low + 30.0 + np.asarray(design["age"], dtype=float) * (high - low)
        inside = (
            (observed_t >= t_axis[0]) & (observed_t <= t_axis[-1])
            & (observed_z >= z_axis[0]) & (observed_z <= z_axis[-1])
        )
        observed_t, observed_z = observed_t[inside], observed_z[inside]

    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_IN, 2.9), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 0.95]},
    )

    axis = axes[0]
    limit = float(np.max(np.abs(component)))
    # Rasterised: the filled mesh is a photograph of a smooth surface, and as
    # vector paths it costs several megabytes for no visible gain.  The
    # contours, points, labels, and axes stay vector.
    mesh = axis.pcolormesh(
        z_axis, t_axis, component, shading="auto", cmap="RdBu_r",
        vmin=-limit, vmax=limit, rasterized=True,
    )
    if observed_t is not None:
        axis.scatter(
            observed_z, observed_t, s=1.1, color="0.12", alpha=0.20, linewidths=0,
            rasterized=True, zorder=3,
        )
    axis.contour(
        z_axis, t_axis, component, levels=[0.0], colors="0.1", linewidths=0.9,
        linestyles=[(0, (4.0, 1.8))], zorder=4,
    )
    crossings = np.flatnonzero(np.diff(np.sign(phi)) != 0)
    if crossings.size:
        axis.annotate(
            r"$\widehat\phi_1=0$", xy=(float(z_axis[crossings[0]]), t_axis[-1]),
            xytext=(3, -11), textcoords="offset points", fontsize=7.5, color="0.1", zorder=5,
        )
    crossings = np.flatnonzero(np.diff(np.sign(beta)) != 0)
    if crossings.size:
        axis.annotate(
            r"$\widehat\beta_1=0$", xy=(z_axis[0], float(t_axis[crossings[0]])),
            xytext=(3, 3), textcoords="offset points", fontsize=7.5, color="0.1", zorder=5,
        )
    # The colour bar carries no separate label: it sits immediately left of the
    # vertical axis label of panel (b), and two units stacked side by side read
    # as one confused axis.  The unit goes in the title instead.
    bar = fig.colorbar(mesh, ax=axis, fraction=0.05, pad=0.02)
    bar.ax.tick_params(labelsize=7.5)
    axis.set_title(
        panel("a", r"Age component $\widehat\beta_1(t)\widehat\phi_1(z)$, CD4 cells"),
        pad=3,
    )
    axis.set_xlabel(age_label, labelpad=1.5)
    axis.set_ylabel(time_label, labelpad=2)
    axis.tick_params(length=2.5, width=0.6)

    axis = axes[1]
    reference = observed_z if observed_z is not None else z_axis
    picked = np.quantile(reference, [0.1, 0.5, 0.9])
    shades = list(plt.get_cmap("RdBu_r")(np.array([0.10, 0.90])))
    shades.insert(1, matplotlib.colors.to_rgba("0.45"))
    offsets = [(4, -8), (4, -8), (4, 9)]
    fitted = []
    for value, shade, offset in zip(picked, shades, offsets):
        curve = baseline + beta * float(np.interp(value, z_axis, phi))
        fitted.append(curve)
        axis.plot(t_axis, curve, color=shade, linewidth=1.2, zorder=3)
        axis.annotate(
            f"Age {value:.0f}", xy=(t_axis[0], curve[0]), xytext=offset,
            textcoords="offset points", fontsize=7.5, color=shade, va="center", ha="left",
        )
    stack = np.vstack(fitted)
    axis.fill_between(
        t_axis, stack.min(axis=0), stack.max(axis=0),
        color=METHOD_COLOR["TRACE-VCAM"], alpha=0.10, linewidth=0, zorder=1,
        label="Spread attributable to age",
    )
    axis.plot(
        t_axis, baseline, color="black", linewidth=1.5, zorder=4,
        label=r"Baseline $\widehat\beta_0$",
    )
    axis.set_title(panel("b", "Fitted mean CD4 at three observed ages"), pad=3)
    axis.set_xlabel(time_label, labelpad=1.5)
    axis.set_ylabel("CD4 cells", labelpad=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", linewidth=0.3, alpha=0.22)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.5, width=0.6)
    axis.legend(loc="upper right", frameon=False, fontsize=7.5)
    if observed_t is not None:
        _design_strip(axis, observed_t, share=0.07)
    fig.savefig(output / filename)
    plt.close(fig)
    return {
        "ages_shown": [round(float(value), 1) for value in picked],
        "component_range": [round(float(component.min()), 1), round(float(component.max()), 1)],
        "age_spread_at_peak": round(
            float(np.max(stack.max(axis=0) - stack.min(axis=0))), 1
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--macs", type=Path, default=ROOT / "results" / "macs_formal_cv")
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=ROOT / "results" / "macs_bootstrap" / "macs_bootstrap_bands.json",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=Path,
        default=ROOT / "results" / "macs_bootstrap" / "macs_bootstrap_draws.npz",
    )
    parser.add_argument(
        "--method-curves",
        type=Path,
        default=ROOT / "results" / "macs_method_curves" / "macs_method_curves.json",
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=ROOT / "results" / "macs_stability" / "macs_variant_curves.json",
    )
    parser.add_argument("--tables", type=Path, default=ROOT / "manuscript" / "tables")
    parser.add_argument("--figures", type=Path, default=ROOT / "manuscript" / "figures")
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.macs / "macs_predictions.csv")
    results = pd.read_csv(args.macs / "macs_results.csv")
    results = results[results["mode"] == "formal"] if "mode" in results.columns else results
    metrics = fold_metrics(predictions, results)
    runtimes = results[results.attempt_status == "success"][
        ["variant", "repeat", "fold", "method", "runtime_seconds"]
    ]

    table, audit = macs_main_table(
        metrics,
        runtimes,
        caption=(
            "MACS CD4 data. Five repeats of fivefold subject-level "
            "cross-validation; every method is fitted on the same training "
            "subjects and evaluated on the same held-out subjects, and the "
            "summary uses the NFOLDS folds on which all displayed methods "
            "returned a fit. Entries are means over folds with standard "
            "deviations in parentheses. MSPE is the mean squared prediction "
            "error, its subject-balanced version averages within subjects "
            "before averaging across them, and MAPE is the mean absolute "
            "prediction error; elapsed time includes each method's own tuning "
            "search. Boldface marks the best value in a column."
        ),
        label="tab:macs",
    )
    (args.tables / "macs_cv_main.tex").write_text(table, encoding="utf-8")
    print(json.dumps(audit, indent=2))

    (args.tables / "macs_sensitivity.tex").write_text(
        macs_sensitivity_table(
            metrics,
            caption=(
                "MACS CD4 data: prespecified sensitivity analyses. Entries are the "
                "mean held-out MSPE in thousands, over the folds of that analysis on "
                "which all displayed methods returned a fit, with standard "
                "deviations in parentheses. Boldface marks the best value in a row."
            ),
            label="tab:macs-sensitivity",
        ),
        encoding="utf-8",
    )

    (args.tables / "macs_completion.tex").write_text(
        macs_completion_table(
            results,
            caption=(
                "MACS CD4 data: fits returned over folds attempted, by method and "
                "analysis. A method outside its registered data regime is not "
                "attempted."
            ),
            label="tab:macs-completion",
        ),
        encoding="utf-8",
    )

    main_components = ["baseline", "beta_1", "phi_1"]
    all_components = ["baseline", "beta_1", "phi_1", "beta_2", "phi_2"]

    display_ranges = observed_display_ranges()
    print("display ranges (scaled):", display_ranges)
    design = observed_design()

    grid, curves = _variant_curves(args.stability)
    if curves:
        figure_stability(
            grid, curves, args.figures,
            components=main_components, filename="macs_stability.pdf", height=2.9,
            display_ranges=display_ranges,
        )
        figure_stability(
            grid, curves, args.figures,
            components=all_components, filename="supp_macs_stability.pdf", height=4.3,
            display_ranges=display_ranges,
        )
        # The reconstructed surface is a deterministic transform of two curves
        # the component figure already shows, so it belongs with the supporting
        # material rather than in the main text.
        print(
            "surface figure:",
            json.dumps(
                figure_surfaces(
                    grid, curves, args.figures,
                    display_ranges=display_ranges, design=design,
                ),
                indent=2,
            ),
        )
    if args.bootstrap.exists():
        bootstrap = json.loads(args.bootstrap.read_text(encoding="utf-8"))
        draws = np.load(args.bootstrap_draws) if args.bootstrap_draws.exists() else None
        comparison = (
            json.loads(args.method_curves.read_text(encoding="utf-8"))["methods"]
            if args.method_curves.exists()
            else None
        )
        band_audit = figure_components(
            bootstrap, args.figures,
            components=main_components, filename="macs_components.pdf", height=2.75,
            layout=(1, 4), draws=draws, comparison=comparison,
            display_ranges=display_ranges, design=design,
        )
        figure_components(
            bootstrap, args.figures,
            components=all_components, filename="supp_macs_components.pdf", height=4.5,
            layout=(2, 3), draws=draws, comparison=comparison,
            display_ranges=display_ranges, design=design,
        )
        print("bootstrap replicates:", bootstrap.get("replicates_completed"))
        print("bootstrap retention:", bootstrap.get("retention"))
        print("band audit:", json.dumps(band_audit, indent=2))
        if comparison:
            print(
                "full-data competitor status:",
                {
                    method: entry.get("attempt_status", entry.get("applicability"))
                    for method, entry in comparison.items()
                },
            )
    print("application tables and figures rebuilt")


if __name__ == "__main__":
    main()

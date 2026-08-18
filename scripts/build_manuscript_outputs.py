"""Build the manuscript tables and figures from the audited result streams.

The strict benchmark and the MACS application write complete per-replication
records.  This script only reads those records: it never refits a model and
never regenerates data.  It produces the layout the paper uses, following the
presentation conventions of the source literature (componentwise accuracy
tables with a dispersion measure in parentheses, and pointwise Monte Carlo
envelopes around the estimated component functions).
"""

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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manuscript_common import (  # noqa: E402
    LONG_NAME,
    METHOD_ORDER,
    SHORT_NAME,
    bold,
    componentwise_errors,
    fmt,
    iter_curve_records,
    registered_truth,
    stacked_curves,
)

# ---------------------------------------------------------------------------
# Global figure style: Times New Roman text and Times-compatible mathematics.
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)

FIGURE_WIDTH_IN = 6.5


def panel(letter: str, title: str = "") -> str:
    """A panel heading whose identifier is bold, as the caption labels are.

    The document sets ``labelfont=bf``, so a caption opens with a bold
    ``Figure N.``; matching that inside the figure keeps one convention for
    the whole article.  The remaining words stay in the regular face and are
    capitalised like a sentence.
    """

    head = rf"$\mathbf{{({letter})}}$"
    return f"{head} {title}" if title else head

#: Preserved from the previous analysis pipeline so that the box colours and
#: their method assignment do not change between revisions.
METHOD_COLOR = {
    method: plt.get_cmap("tab10")(index % 10)
    for index, method in enumerate(METHOD_ORDER)
}

COMPONENT_LABEL = {
    "beta_0": r"$\beta_0$",
    "beta_1": r"$\beta_1$",
    "beta_2": r"$\beta_2$",
    "phi_1": r"$\phi_1$",
    "phi_2": r"$\phi_2$",
    "surface": r"$\sum_k g_k$",
}

#: The main text reports the smallest registered sample size, where the
#: estimators are most clearly separated; the Supplement reports all of them.
EXAMPLE2_BLOCKS = (
    ("example2-gaussian-n50-sigma0.1", r"Normal errors, $\sigma=0.1$"),
    ("example2-gaussian-n50-sigma0.4", r"Normal errors, $\sigma=0.4$"),
    ("example2-mixed-normal-n50", "Contaminated normal mixture"),
    ("example2-t2-n50", r"Scaled $t_2$ errors"),
)

BLOCK_SPARSE_BLOCKS = (
    ("example4-blocksparse-normal-n100", "Normal errors"),
    ("example4-blocksparse-mixed-normal-n100", "Contaminated normal mixture"),
    ("example4-blocksparse-t2-n100", r"Scaled $t_2$ errors"),
)

EXAMPLE3_BLOCKS = (
    ("example3-gaussian-n200-p10-sigma0.1", r"Normal errors, $\sigma=0.1$"),
    ("example3-gaussian-n200-p10-sigma0.4", r"Normal errors, $\sigma=0.4$"),
    ("example3-contamination-n200-p10-sigma0.1", r"Contaminated, $\sigma=0.1$"),
    ("example3-contamination-n200-p10-sigma0.4", r"Contaminated, $\sigma=0.4$"),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_componentwise(curves_path: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    records = []
    for record in iter_curve_records(curves_path):
        errors = componentwise_errors(record)
        records.append(
            {
                "scenario": str(record["scenario"]),
                "method": str(record["method"]),
                "replicate": int(record["replicate"]),
                **errors,
            }
        )
    frame = pd.DataFrame(records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def median_dispersion(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    centre = float(np.median(array))
    return centre, float(np.median(np.abs(array - centre)))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fixed(value: float, digits: int) -> str:
    """Fixed-point formatting, so that a column reads as one scale."""

    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) >= 10.0**4:
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{value:.{digits}f}"


def _cell(centre: float, spread: float, *, best: bool, digits: int = 4) -> str:
    if not np.isfinite(centre):
        return "--"
    body = f"{_fixed(centre, digits)} ({_fixed(spread, digits)})"
    return bold(body) if best else body


def componentwise_table(
    componentwise: pd.DataFrame,
    performance: pd.DataFrame,
    blocks: Sequence[tuple[str, str]],
    methods: Sequence[str],
    columns: Sequence[str],
    *,
    caption: str,
    label: str,
    include_mspe: bool = True,
) -> str:
    header = [COMPONENT_LABEL[name] for name in columns]
    if include_mspe:
        header.append("MSPE")
    n_numeric = len(header)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "c" * n_numeric + r"@{}}",
        r"\toprule",
        "Method & " + " & ".join(header) + r" \\",
    ]
    for scenario, title in blocks:
        block = componentwise[componentwise.scenario == scenario]
        perf = performance[
            (performance.scenario == scenario) & (performance.attempt_status == "success")
        ]
        cells: dict[str, dict[str, tuple[float, float]]] = {}
        for method in methods:
            entry: dict[str, tuple[float, float]] = {}
            sub = block[block.method == method]
            for name in columns:
                entry[name] = median_dispersion(sub[name].to_numpy()) if len(sub) else (np.nan, np.nan)
            if include_mspe:
                target = perf[perf.method == method]["noise_free_test_mspe"].to_numpy()
                entry["MSPE"] = median_dispersion(target)
            cells[method] = entry
        keys = list(columns) + (["MSPE"] if include_mspe else [])
        winners = {}
        for key in keys:
            finite = {
                method: cells[method][key][0]
                for method in methods
                if np.isfinite(cells[method][key][0])
            }
            winners[key] = min(finite, key=finite.get) if finite else None
        lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{{n_numeric + 1}}}{{@{{}}l@{{}}}}{{\\textit{{{title}}}}} \\\\"
        )
        for method in methods:
            row = [SHORT_NAME[method]]
            for key in keys:
                centre, spread = cells[method][key]
                row.append(_cell(centre, spread, best=winners[key] == method))
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def macs_table(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    """Common-fold prediction summary for the application."""

    methods = [m for m in METHOD_ORDER if m in set(frame.method)]
    success = frame[frame.attempt_status == "success"]
    common = None
    for method in methods:
        folds = set(map(int, success[success.method == method]["fold_key"]))
        common = folds if common is None else (common & folds)
    common = common or set()
    restricted = success[success["fold_key"].isin(common)]

    metrics = (
        ("test_mse", "MSPE", 1),
        ("subject_balanced_test_mse", "Subject-balanced MSPE", 1),
        ("test_mae", "MAPE", 2),
        ("runtime_seconds", "Time (s)", 2),
    )
    values = {
        method: {
            key: median_dispersion(restricted[restricted.method == method][key].to_numpy())
            for key, _, _ in metrics
        }
        for method in methods
    }
    winners = {}
    for key, _, _ in metrics:
        finite = {m: values[m][key][0] for m in methods if np.isfinite(values[m][key][0])}
        winners[key] = min(finite, key=finite.get) if finite else None

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}l" + "c" * len(metrics) + r"@{}}",
        r"\toprule",
        "Method & " + " & ".join(title for _, title, _ in metrics) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        row = [SHORT_NAME[method]]
        for key, _, digits in metrics:
            centre, spread = values[method][key]
            row.append(_cell(centre, spread, best=winners[key] == method, digits=digits))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), sorted(common)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _tidy_log_axis(axis) -> None:
    """Keep only decade labels so a narrow panel stays readable."""

    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())


def _boxes(
    axis,
    frame: pd.DataFrame,
    column: str,
    methods: Sequence[str],
    *,
    log: bool,
    tick_labels: bool = True,
) -> list[str]:
    samples, colors, labels = [], [], []
    for method in methods:
        values = frame[frame.method == method][column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        samples.append(values)
        colors.append(METHOD_COLOR[method])
        labels.append(method)
    if not samples:
        return []
    box = axis.boxplot(
        samples,
        showfliers=False,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "black", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    axis.set_xticks(range(1, len(labels) + 1))
    if tick_labels:
        axis.set_xticklabels([SHORT_NAME[name] for name in labels])
    else:
        axis.set_xticklabels([""] * len(labels))
        axis.tick_params(axis="x", length=0)
    if log:
        axis.set_yscale("log")
        _tidy_log_axis(axis)
    axis.grid(axis="y", linewidth=0.35, alpha=0.4)
    axis.set_axisbelow(True)
    return labels


def _method_legend(fig, methods: Sequence[str], *, ncol: int | None = None) -> None:
    handles = [
        Patch(facecolor=METHOD_COLOR[method], edgecolor="black", linewidth=0.6, alpha=0.85)
        for method in methods
    ]
    fig.legend(
        handles,
        [SHORT_NAME[method] for method in methods],
        loc="outside lower center",
        ncol=ncol or len(methods),
        frameon=False,
    )


def figure_example1(
    componentwise: pd.DataFrame, performance: pd.DataFrame, output: Path
) -> None:
    scenario = "example1-zw2015-n100"
    methods = ["TRACE-VCAM", "ZW2015"]
    block = componentwise[componentwise.scenario == scenario]
    perf = performance[
        (performance.scenario == scenario) & (performance.attempt_status == "success")
    ]
    merged = block.merge(
        perf[["method", "replicate", "noise_free_test_mspe", "runtime_seconds"]],
        on=["method", "replicate"],
        how="left",
    )
    panels = [
        ("beta_0", panel(r"a", r"$\beta_0$"), True),
        ("beta_1", panel(r"b", r"$\beta_1$"), True),
        ("phi_1", panel(r"c", r"$\phi_1$"), True),
        ("surface", panel("d", "Component surfaces"), True),
        ("noise_free_test_mspe", panel("e", "Prediction error"), True),
        ("runtime_seconds", panel("f", "Elapsed time (s)"), True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 3.8), constrained_layout=True)
    for axis, (column, title, log) in zip(axes.ravel(), panels):
        _boxes(axis, merged, column, methods, log=log)
        axis.set_title(title)
    fig.savefig(output / "example1_recovery.pdf")
    plt.close(fig)


#: Central design range of each Example 2 covariate, as a pair of quantiles of
#: the registered covariate law.  Both covariates are supported on [0,1] by
#: construction, but the design visits the two ends of that interval almost
#: never: the outer 2.5% of the covariate law lies beyond roughly 0.09 and 0.82,
#: so a factor plotted on the whole unit interval spends its two ends showing
#: extrapolation rather than estimation.  The accuracy tables integrate over the
#: full interval regardless; only the display is restricted, and the caption
#: says so.
EXAMPLE2_DISPLAY_QUANTILES = (0.025, 0.975)


def example2_covariate_display_range(
    quantiles: tuple[float, float] = EXAMPLE2_DISPLAY_QUANTILES,
    *,
    n_subjects: int = 20000,
    seed: int = 20260101,
) -> tuple[tuple[float, float], ...]:
    """Central design range of each Example 2 covariate, from its own law."""

    from experiments.dgp import generate_zzw2020

    data = generate_zzw2020(seed, n_subjects=n_subjects, sigma=0.4)
    return tuple(
        (float(np.quantile(column, quantiles[0])), float(np.quantile(column, quantiles[1])))
        for column in data.covariates.T
    )


#: Line style and weight of each method in the curve displays.  The proposed
#: estimator is the only solid coloured line, so that the eye separates it from
#: the competitors before it reads the legend; the truth is solid black.
CURVE_STYLE: dict[str, tuple[object, float]] = {
    "TRACE-VCAM": ((0, ()), 1.35),
    "ZW2015": ((0, (4.5, 1.6)), 1.0),
    "ZZW2020": ((0, (4.5, 1.6)), 1.0),
    "HHY2021-Huber": ((0, (1.1, 1.1)), 1.0),
    "ZY2025-paper-implementation": ((0, (5.0, 1.3, 1.0, 1.3)), 1.0),
    "ZSY2026-author-code": ((0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)), 1.0),
}

CURVE_TITLES = {
    "baseline": panel("%s", r"$\beta_0(t)$"),
    "beta_1": panel("%s", r"$\beta_1(t)$"),
    "beta_2": panel("%s", r"$\beta_2(t)$"),
    "phi_1": panel("%s", r"$\phi_1(z_1)$"),
    "phi_2": panel("%s", r"$\phi_2(z_2)$"),
}

#: Components carried by every registered curve stream, in stream order.  The
#: shared replication is chosen on all of them, so that which panels a
#: particular figure displays cannot influence which replication it shows.
CURVE_COMPONENTS = ("baseline", "beta_1", "beta_2", "phi_1", "phi_2")

CURVE_XLABEL = {
    "baseline": r"$t$",
    "beta_1": r"$t$",
    "beta_2": r"$t$",
    "phi_1": r"$z_1$",
    "phi_2": r"$z_2$",
}


def _curve_stacks(
    curves_path: Path,
    scenario: str,
    methods: Sequence[str],
    cache: Mapping[str, np.ndarray] | None,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Per method, the grid, replication matrix, and labels of each component."""

    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for method in methods:
        if cache is None:
            out[method] = stacked_curves(curves_path, scenario, method, CURVE_COMPONENTS)
            continue
        entry: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name in CURVE_COMPONENTS:
            tag = f"{scenario}|{method}|{name}"
            if f"stack::{tag}" in cache:
                entry[name] = (
                    cache[f"grid::{tag}"],
                    cache[f"stack::{tag}"],
                    cache[f"replicate::{tag}"],
                )
        out[method] = entry
    return {method: entry for method, entry in out.items() if entry}


def _componentwise_errors(
    entry: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    truth_of: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Domain-averaged squared error of each component, replication by replication."""

    return {
        name: np.maximum(
            np.mean((stack - truth_of[name](grid)) ** 2, axis=1), 1e-12
        )
        for name, (grid, stack, _) in entry.items()
    }


def shared_replication(
    stacks: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    reference_method: str,
    truth_of: Mapping[str, object],
) -> int | None:
    """The replication on which to compare every method, chosen on ours alone.

    Within a replication the protocol gives every method the same generated
    data, the same seed, and the same subjects, so one replication is the
    comparison the design was built for.  Which one is decided by the proposed
    estimator's own typicality and by nothing about the competitors: among the
    replications every method completed, we take the one whose componentwise
    errors are jointly closest, on a log scale, to our own medians.  Selecting on
    a competitor doing badly would be a different and much weaker display.
    """

    common: set[int] | None = None
    for entry in stacks.values():
        labels = set(int(value) for value in entry["baseline"][2])
        common = labels if common is None else common & labels
    if not common:
        return None
    errors = _componentwise_errors(stacks[reference_method], truth_of)
    score = sum(
        np.abs(np.log(values / np.median(values))) for values in errors.values()
    )
    labels = stacks[reference_method]["baseline"][2]
    for row in np.argsort(score):
        if int(labels[row]) in common:
            return int(labels[row])
    return None


def curve_band_ratios(
    stacks: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    reference_method: str,
    *,
    band: tuple[float, float] = (10.0, 90.0),
    covariate_ranges: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, dict[str, float]]:
    """Median pointwise band width of each method, relative to the proposed one.

    The recovery figure no longer draws this, because a second row of ratios
    competes with the fitted curves for the reader's attention.  The numbers are
    still computed here so that the dispersion sentences in the section text are
    generated rather than remembered.
    """

    covariate_of = {"phi_1": 0, "phi_2": 1}
    out: dict[str, dict[str, float]] = {}
    for name in CURVE_COMPONENTS:
        if name not in stacks[reference_method]:
            continue
        grid = stacks[reference_method][name][0]
        window = (
            covariate_ranges[covariate_of[name]]
            if covariate_ranges is not None and name in covariate_of
            else None
        )
        inside = (
            np.ones_like(grid, dtype=bool)
            if window is None
            else (grid >= window[0]) & (grid <= window[1])
        )
        widths: dict[str, float] = {}
        for method, entry in stacks.items():
            if name not in entry:
                continue
            method_grid, stack, _ = entry[name]
            lower, upper = np.percentile(stack, list(band), axis=0)
            spread = np.interp(grid, method_grid, upper - lower)
            widths[method] = float(np.median(spread[inside]))
        out[name] = {
            SHORT_NAME[method]: widths[method] / widths[reference_method]
            for method in widths
            if method != reference_method
        }
    return out


def figure_curve_envelopes(
    curves_path: Path,
    scenario: str,
    reference_method: str,
    comparison_methods: Sequence[str],
    output: Path,
    filename: str,
    *,
    components: Sequence[str],
    height: float = 2.35,
    band: tuple[float, float] = (10.0, 90.0),
    covariate_ranges: tuple[tuple[float, float], ...] | None = None,
    cache: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Monte Carlo median curves with pointwise envelopes, over all replications.

    Nothing here is selected.  Every replication a method completed enters its
    own median and its own band, so no property of any fit decides what is
    drawn.  The price is that the medians of these estimators sit almost on top
    of one another, since all of them are close to unbiased in the median; the
    envelope is what carries the comparison, and it is the quantity the
    dispersion sentences in the text quote.
    """

    truth, _, _ = registered_truth(scenario)
    methods = [reference_method, *comparison_methods]
    stacks = _curve_stacks(curves_path, scenario, methods, cache)
    methods = [method for method in methods if method in stacks]
    truth_of = {
        "baseline": truth.beta0,
        "beta_1": truth.beta[0],
        "beta_2": truth.beta[1],
        "phi_1": truth.phi[0],
        "phi_2": truth.phi[1],
    }
    covariate_of = {"phi_1": 0, "phi_2": 1}

    fig, axes = plt.subplots(
        1, len(components), figsize=(FIGURE_WIDTH_IN, height), constrained_layout=True
    )
    flat = np.atleast_1d(axes)
    audit: dict[str, object] = {"band": list(band), "replications": {}, "band_width": {}}
    for letter, axis, name in zip("abcde", flat, components):
        window = (
            covariate_ranges[covariate_of[name]]
            if covariate_ranges is not None and name in covariate_of
            else None
        )
        grid = stacks[reference_method][name][0]
        inside = (
            np.ones_like(grid, dtype=bool)
            if window is None
            else (grid >= window[0]) & (grid <= window[1])
        )
        truth_values = truth_of[name](grid)
        lows = [float(np.min(truth_values[inside]))]
        highs = [float(np.max(truth_values[inside]))]
        for method in methods:
            if name not in stacks[method]:
                continue
            method_grid, stack, _labels = stacks[method][name]
            lower, upper = np.percentile(stack, list(band), axis=0)
            middle = np.median(stack, axis=0)
            colour = METHOD_COLOR[method]
            style, _width = CURVE_STYLE[method]
            # Four filled bands over one another produce a colour none of them
            # has.  Only the proposed estimator's band is filled; the others are
            # drawn as their two edges, which is enough to compare widths and
            # leaves every band readable.
            if method == reference_method:
                axis.fill_between(
                    method_grid, lower, upper, color=colour, alpha=0.28,
                    linewidth=0, zorder=2,
                )
            else:
                for edge in (lower, upper):
                    axis.plot(
                        method_grid, edge, color=colour, linestyle=style,
                        linewidth=0.6, alpha=0.75, zorder=1,
                    )
            axis.plot(
                method_grid, middle, color=colour, linestyle=style,
                linewidth=1.4 if method == reference_method else 1.0,
                label=SHORT_NAME[method],
                zorder=4 if method == reference_method else 3,
            )
            visible = np.interp(grid, method_grid, middle)[inside]
            lows.append(float(np.min(visible)))
            highs.append(float(np.max(visible)))
            audit["replications"][SHORT_NAME[method]] = int(stack.shape[0])
            audit["band_width"].setdefault(name, {})[SHORT_NAME[method]] = float(
                np.median(np.interp(grid, method_grid, upper - lower)[inside])
            )
        axis.plot(grid, truth_values, color="black", linewidth=1.5, label="True", zorder=5)
        if window is not None:
            axis.set_xlim(*window)
        low, high = min(lows), max(highs)
        pad = 0.55 * (high - low)
        axis.set_ylim(low - pad, high + pad)
        axis.set_title(CURVE_TITLES[name] % letter, pad=3)
        axis.set_xlabel(CURVE_XLABEL[name], labelpad=1.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        axis.tick_params(length=2.5, width=0.6)
    handles, labels = flat[0].get_legend_handles_labels()
    found = dict(zip(labels, handles))
    order = ["True", *(SHORT_NAME[method] for method in methods)]
    fig.legend(
        [found[label] for label in order if label in found],
        [label for label in order if label in found],
        loc="outside lower center",
        ncol=len(order),
        frameon=False,
        handlelength=2.4,
    )
    fig.savefig(output / filename)
    plt.close(fig)
    return audit


def figure_curves(
    curves_path: Path,
    scenario: str,
    reference_method: str,
    comparison_methods: Sequence[str],
    output: Path,
    filename: str,
    *,
    components: Sequence[str],
    height: float = 2.35,
    covariate_ranges: tuple[tuple[float, float], ...] | None = None,
    cache: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Every compared estimator, fitted to one and the same generated data set.

    A display of pointwise medians over replications says nothing about how well
    any of these estimators fits, because all of them are close to unbiased in
    the median and their averages sit almost on top of each other; what separates
    them is how far an individual fit strays.  Showing one shared replication
    puts that difference on the page directly, and it is the comparison the
    protocol is built for, since within a replication every method receives the
    same data, seed, and subjects.  The replication is chosen by the typicality
    of the proposed estimator alone, never by a competitor's failure, and the
    returned audit records how the fit shown compares with each method's own
    median so that the choice can be checked.
    """

    truth, _, _ = registered_truth(scenario)
    methods = [reference_method, *comparison_methods]
    stacks = _curve_stacks(curves_path, scenario, methods, cache)
    methods = [method for method in methods if method in stacks]
    truth_of = {
        "baseline": truth.beta0,
        "beta_1": truth.beta[0],
        "beta_2": truth.beta[1],
        "phi_1": truth.phi[0],
        "phi_2": truth.phi[1],
    }
    chosen = shared_replication(stacks, reference_method, truth_of)
    if chosen is None:
        raise ValueError(f"no replication is shared by every method in {scenario}")
    covariate_of = {"phi_1": 0, "phi_2": 1}

    fig, axes = plt.subplots(
        1, len(components), figsize=(FIGURE_WIDTH_IN, height), constrained_layout=True
    )
    flat = np.atleast_1d(axes)
    for letter, axis, name in zip("abcde", flat, components):
        window = (
            covariate_ranges[covariate_of[name]]
            if covariate_ranges is not None and name in covariate_of
            else None
        )
        grid = stacks[reference_method][name][0]
        inside = (
            np.ones_like(grid, dtype=bool)
            if window is None
            else (grid >= window[0]) & (grid <= window[1])
        )
        truth_values = truth_of[name](grid)
        lows = [float(np.min(truth_values[inside]))]
        highs = [float(np.max(truth_values[inside]))]
        for method in methods:
            if name not in stacks[method]:
                continue
            method_grid, stack, labels = stacks[method][name]
            row = int(np.flatnonzero(labels == chosen)[0])
            style, width = CURVE_STYLE[method]
            axis.plot(
                method_grid,
                stack[row],
                color=METHOD_COLOR[method],
                linestyle=style,
                linewidth=1.5 if method == reference_method else 1.0,
                label=SHORT_NAME[method],
                zorder=4 if method == reference_method else 3,
            )
            visible = np.interp(grid, method_grid, stack[row])[inside]
            lows.append(float(np.min(visible)))
            highs.append(float(np.max(visible)))
        axis.plot(grid, truth_values, color="black", linewidth=1.6, label="True", zorder=5)
        if window is not None:
            axis.set_xlim(*window)
        low, high = min(lows), max(highs)
        pad = 0.08 * (high - low)
        axis.set_ylim(low - pad, high + pad)
        axis.set_title(CURVE_TITLES[name] % letter, pad=3)
        axis.set_xlabel(CURVE_XLABEL[name], labelpad=1.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        axis.tick_params(length=2.5, width=0.6)
    handles, labels = flat[0].get_legend_handles_labels()
    found = dict(zip(labels, handles))
    order = ["True", *(SHORT_NAME[method] for method in methods)]
    fig.legend(
        [found[label] for label in order if label in found],
        [label for label in order if label in found],
        loc="outside lower center",
        ncol=len(order),
        frameon=False,
        handlelength=2.4,
    )
    fig.savefig(output / filename)
    plt.close(fig)

    audit: dict[str, object] = {
        "shared_replication": chosen,
        "replications": {
            SHORT_NAME[method]: int(stacks[method]["baseline"][1].shape[0])
            for method in methods
        },
        "error_on_shared_replication": {},
        "median_error": {},
    }
    worse_than_own_median = []
    for method in methods:
        errors = _componentwise_errors(stacks[method], truth_of)
        labels = stacks[method]["baseline"][2]
        row = int(np.flatnonzero(labels == chosen)[0])
        shown = {name: float(values[row]) for name, values in errors.items()}
        medians = {name: float(np.median(values)) for name, values in errors.items()}
        audit["error_on_shared_replication"][SHORT_NAME[method]] = shown
        audit["median_error"][SHORT_NAME[method]] = medians
        if method == reference_method:
            worse_than_own_median = [
                name for name in shown if shown[name] > medians[name]
            ]
    audit["proposed_worse_than_own_median_on"] = worse_than_own_median
    return audit


def figure_boxplot_grid(
    componentwise: pd.DataFrame,
    performance: pd.DataFrame,
    blocks: Sequence[tuple[str, str]],
    methods: Sequence[str],
    column: str,
    ylabel: str,
    output: Path,
    filename: str,
    *,
    from_performance: bool = False,
) -> None:
    fig, axes = plt.subplots(
        1, len(blocks), figsize=(FIGURE_WIDTH_IN, 2.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, (scenario, title) in zip(axes, blocks):
        if from_performance:
            frame = performance[
                (performance.scenario == scenario)
                & (performance.attempt_status == "success")
            ]
        else:
            frame = componentwise[componentwise.scenario == scenario]
        _boxes(axis, frame, column, methods, log=True)
        axis.set_title(title)
    axes[0].set_ylabel(ylabel)
    fig.savefig(output / filename)
    plt.close(fig)


def figure_example2(
    componentwise: pd.DataFrame, performance: pd.DataFrame, output: Path
) -> None:
    methods = ["TRACE-VCAM", "ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_WIDTH_IN, 3.6), constrained_layout=True)
    for column_index, (scenario, title) in enumerate(EXAMPLE2_BLOCKS):
        block = componentwise[componentwise.scenario == scenario]
        perf = performance[
            (performance.scenario == scenario) & (performance.attempt_status == "success")
        ]
        _boxes(axes[0, column_index], block, "surface", methods, log=True, tick_labels=False)
        axes[0, column_index].set_title(title)
        _boxes(
            axes[1, column_index],
            perf,
            "noise_free_test_mspe",
            methods,
            log=True,
            tick_labels=False,
        )
    axes[0, 0].set_ylabel("Component-surface ISE")
    axes[1, 0].set_ylabel("Prediction error")
    _method_legend(fig, methods)
    fig.savefig(output / "example2_robustness.pdf")
    plt.close(fig)


EXAMPLE2_SIZES = (50, 100, 200)
EXAMPLE2_REFERENCE_LAW = "gaussian-n{n}-sigma0.1"


def _median_by_size(
    frame: pd.DataFrame, pattern: str, method: str, column: str
) -> list[float]:
    centres = []
    for n in EXAMPLE2_SIZES:
        scenario = "example2-" + pattern.format(n=n)
        sub = frame[(frame.scenario == scenario) & (frame.method == method)]
        centres.append(median_dispersion(sub[column].to_numpy())[0])
    return centres


def figure_sample_size(
    componentwise: pd.DataFrame, performance: pd.DataFrame, output: Path
) -> None:
    """Accuracy and stability against the number of subjects under contamination.

    The first two panels are the contaminated setting the block-robust design is
    written for.  The third is the quantity the robustness claim is actually
    about: how far each estimator moves when the error law changes from the
    clean normal setting to the contaminated one at the same sample size.
    """

    methods = ["TRACE-VCAM", "ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    successful = performance[performance.attempt_status == "success"]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.4), constrained_layout=True)
    panels = (
        (
            componentwise,
            "surface",
            "mixed-normal-n{n}",
            "Median component-surface ISE",
            panel("a", "Component surfaces, mixture"),
            False,
        ),
        (
            successful,
            "noise_free_test_mspe",
            "mixed-normal-n{n}",
            "Median prediction error",
            panel("b", "Prediction, mixture"),
            False,
        ),
        (
            componentwise,
            "surface",
            "mixed-normal-n{n}",
            "Error inflation from clean to mixture",
            panel("c", "Cost of contamination"),
            True,
        ),
    )
    for axis, (frame, column, pattern, ylabel, title, relative) in zip(axes, panels):
        for method in methods:
            centres = np.asarray(_median_by_size(frame, pattern, method, column), dtype=float)
            if relative:
                reference = np.asarray(
                    _median_by_size(frame, EXAMPLE2_REFERENCE_LAW, method, column),
                    dtype=float,
                )
                centres = centres / reference
            axis.plot(
                EXAMPLE2_SIZES,
                centres,
                marker="o",
                markersize=3.4,
                linewidth=1.2,
                color=METHOD_COLOR[method],
                label=SHORT_NAME[method],
            )
        axis.set_yscale("log")
        _tidy_log_axis(axis)
        axis.set_xticks(list(EXAMPLE2_SIZES))
        axis.set_xticklabels([str(n) for n in EXAMPLE2_SIZES])
        axis.set_xlim(35, 215)
        axis.set_xlabel(r"$n$")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(linewidth=0.3, alpha=0.35)
        axis.set_axisbelow(True)
        if relative:
            axis.axhline(1.0, color="0.45", linewidth=0.7, linestyle=":")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.4)
    fig.savefig(output / "example2_sample_size.pdf")
    plt.close(fig)


def figure_sample_size_all_laws(
    componentwise: pd.DataFrame, output: Path
) -> None:
    """Every registered error law, for the Supplementary Material."""

    methods = ["TRACE-VCAM", "ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    families = (
        ("gaussian-n{n}-sigma0.1", r"Normal errors, $\sigma=0.1$"),
        ("gaussian-n{n}-sigma0.4", r"Normal errors, $\sigma=0.4$"),
        ("mixed-normal-n{n}", "Contaminated normal mixture"),
        ("t2-n{n}", r"Scaled $t_2$ errors"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(FIGURE_WIDTH_IN, 2.2), constrained_layout=True)
    for axis, (pattern, title) in zip(axes, families):
        for method in methods:
            axis.plot(
                EXAMPLE2_SIZES,
                _median_by_size(componentwise, pattern, method, "surface"),
                marker="o",
                markersize=3.2,
                linewidth=1.1,
                color=METHOD_COLOR[method],
                label=SHORT_NAME[method],
            )
        axis.set_yscale("log")
        _tidy_log_axis(axis)
        axis.set_xticks(list(EXAMPLE2_SIZES))
        axis.set_xticklabels([str(n) for n in EXAMPLE2_SIZES])
        axis.set_xlim(35, 215)
        axis.set_xlabel(r"$n$")
        axis.set_title(title)
        axis.grid(linewidth=0.3, alpha=0.35)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Median component-surface ISE")
    axes[-1].legend(frameon=False, loc="upper right", handlelength=1.4)
    fig.savefig(output / "supp_example2_sample_size.pdf")
    plt.close(fig)


def figure_block_sparse(performance: pd.DataFrame, output: Path) -> None:
    """Replication-level distributions for the block-sparse design."""

    methods = ["TRACE-VCAM", "ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 3.6), constrained_layout=True)
    for column, (scenario, title) in enumerate(BLOCK_SPARSE_BLOCKS):
        perf = performance[
            (performance.scenario == scenario) & (performance.attempt_status == "success")
        ]
        _boxes(axes[0, column], perf, "component_ise", methods, log=True, tick_labels=False)
        axes[0, column].set_title(title)
        _boxes(
            axes[1, column], perf, "noise_free_test_mspe", methods, log=True, tick_labels=False
        )
    axes[0, 0].set_ylabel("Component-surface ISE")
    axes[1, 0].set_ylabel("Prediction error")
    _method_legend(fig, methods)
    fig.savefig(output / "example3_blocksparse.pdf")
    plt.close(fig)


def block_sparse_table(
    performance: pd.DataFrame, *, caption: str, label: str
) -> str:
    """Aggregate accuracy, selection, and cost for the block-sparse design."""

    methods = ["TRACE-VCAM", "ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    columns = (
        ("baseline_ise", r"$\beta_0$", 4),
        ("component_ise", r"$\sum_k g_k$", 4),
        ("factor_ise", "Factors", 4),
        ("noise_free_test_mspe", "MSPE", 4),
        ("runtime_seconds", "Time (s)", 2),
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l" + "c" * len(columns) + r"@{}}",
        r"\toprule",
        "Method & " + " & ".join(title for _, title, _ in columns) + r" \\",
    ]
    for scenario, title in BLOCK_SPARSE_BLOCKS:
        perf = performance[
            (performance.scenario == scenario) & (performance.attempt_status == "success")
        ]
        cells = {
            method: {
                key: median_dispersion(
                    perf[perf.method == method][key].to_numpy(dtype=float)
                )
                for key, _, _ in columns
            }
            for method in methods
        }
        winners = {}
        for key, _, _ in columns:
            finite = {
                m: cells[m][key][0] for m in methods if np.isfinite(cells[m][key][0])
            }
            winners[key] = min(finite, key=finite.get) if finite else None
        lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{{len(columns) + 1}}}{{@{{}}l@{{}}}}{{\\textit{{{title}}}}} \\\\"
        )
        for method in methods:
            row = [SHORT_NAME[method]]
            for key, _, digits in columns:
                centre, spread = cells[method][key]
                row.append(
                    _cell(centre, spread, best=winners[key] == method, digits=digits)
                )
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def figure_example3(
    componentwise: pd.DataFrame, performance: pd.DataFrame, output: Path
) -> None:
    methods = [
        "TRACE-VCAM",
        "ZZW2020",
        "HHY2021-Huber",
        "ZSY2026-author-code",
        "ZY2025-paper-implementation",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.4), constrained_layout=True)
    scenario = "example3-contamination-n200-p10-sigma0.1"
    block = componentwise[componentwise.scenario == scenario]
    perf = performance[
        (performance.scenario == scenario) & (performance.attempt_status == "success")
    ]
    _boxes(axes[0], block, "surface", methods, log=True, tick_labels=False)
    axes[0].set_title(panel("a", "Component-surface ISE"))
    _boxes(axes[1], perf, "noise_free_test_mspe", methods, log=True, tick_labels=False)
    axes[1].set_title(panel("b", "Prediction error"))
    _boxes(axes[2], perf, "runtime_seconds", methods, log=True, tick_labels=False)
    axes[2].set_title(panel("c", "Elapsed time (s)"))
    _method_legend(fig, methods)
    fig.savefig(output / "example3_highdim.pdf")
    plt.close(fig)


def figure_scaling(performance: pd.DataFrame, output: Path) -> None:
    methods = [
        "TRACE-VCAM",
        "ZZW2020",
        "HHY2021-Huber",
        "ZSY2026-author-code",
        "ZY2025-paper-implementation",
    ]
    sizes = [10, 25, 50]
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 2.4), constrained_layout=True)
    for method in methods:
        runtimes, failures = [], []
        for p in sizes:
            scenario = f"scaling-n200-p{p}"
            sub = performance[
                (performance.scenario == scenario) & (performance.method == method)
            ]
            attempted = sub[sub.applicability == "applicable"]
            ok = attempted[attempted.attempt_status == "success"]
            runtimes.append(
                float(np.median(ok["runtime_seconds"])) if len(ok) else float("nan")
            )
            failures.append(
                100.0 * (1.0 - len(ok) / len(attempted)) if len(attempted) else float("nan")
            )
        axes[0].plot(
            sizes, runtimes, marker="o", markersize=3.4, linewidth=1.2,
            color=METHOD_COLOR[method], label=SHORT_NAME[method],
        )
        axes[1].plot(
            sizes, failures, marker="s", markersize=3.4, linewidth=1.2,
            color=METHOD_COLOR[method], label=SHORT_NAME[method],
        )
    axes[0].set_yscale("log")
    _tidy_log_axis(axes[0])
    axes[0].set_xlabel(r"$p$")
    axes[0].set_ylabel("Median elapsed time (s)")
    axes[0].set_title(panel("a", "Computational cost"))
    axes[1].set_xlabel(r"$p$")
    axes[1].set_ylabel("Failed fits (%)")
    axes[1].set_title(panel("b", "Completion"))
    for axis in axes:
        axis.set_xticks(sizes)
        axis.grid(linewidth=0.3, alpha=0.35)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(output / "scaling_runtime.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary tables
# ---------------------------------------------------------------------------


def full_example_table(
    componentwise: pd.DataFrame,
    performance: pd.DataFrame,
    scenarios: Sequence[tuple[str, str]],
    methods: Sequence[str],
    *,
    caption: str,
    label: str,
) -> str:
    header = (
        r"Setting & Method & Surface ISE & Factor ISE & MSPE & Failed \\"
    )
    lines = [
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}llcccc@{}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for scenario, title in scenarios:
        block = componentwise[componentwise.scenario == scenario]
        perf = performance[performance.scenario == scenario]
        for position, method in enumerate(methods):
            applicable = perf[
                (perf.method == method) & (perf.applicability == "applicable")
            ]
            if applicable.empty:
                continue
            ok = applicable[applicable.attempt_status == "success"]
            sub = block[block.method == method]
            factor_columns = [c for c in ("beta_0", "beta_1", "phi_1", "beta_2", "phi_2") if c in sub]
            factor = sub[factor_columns].sum(axis=1).to_numpy() if len(sub) else np.array([])
            surface_centre, surface_spread = median_dispersion(sub["surface"].to_numpy())
            factor_centre, factor_spread = median_dispersion(factor)
            mspe_centre, mspe_spread = median_dispersion(
                ok["noise_free_test_mspe"].to_numpy()
            )
            failed = len(applicable) - len(ok)
            lines.append(
                " & ".join(
                    [
                        title if position == 0 else "",
                        SHORT_NAME[method],
                        f"{_fixed(surface_centre, 4)} ({_fixed(surface_spread, 4)})",
                        f"{_fixed(factor_centre, 4)} ({_fixed(factor_spread, 4)})",
                        f"{_fixed(mspe_centre, 4)} ({_fixed(mspe_spread, 4)})",
                        f"{failed}/{len(applicable)}",
                    ]
                )
                + r" \\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\end{longtable}", r"}", ""]
    return "\n".join(lines)


def scaling_table(performance: pd.DataFrame, *, caption: str, label: str) -> str:
    methods = [
        "TRACE-VCAM",
        "ZZW2020",
        "HHY2021-Huber",
        "ZSY2026-author-code",
        "ZY2025-paper-implementation",
    ]
    sizes = [10, 25, 50]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Median elapsed time (s)} & \multicolumn{3}{c}{Failed fits} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & $p=10$ & $p=25$ & $p=50$ & $p=10$ & $p=25$ & $p=50$ \\",
        r"\midrule",
    ]
    for method in methods:
        times, fails, attempted = [], [], 0
        for p in sizes:
            sub = performance[
                (performance.scenario == f"scaling-n200-p{p}")
                & (performance.method == method)
                & (performance.applicability == "applicable")
            ]
            attempted += len(sub)
            ok = sub[sub.attempt_status == "success"]
            times.append(f"{np.median(ok['runtime_seconds']):.1f}" if len(ok) else "--")
            fails.append(f"{len(sub) - len(ok)}/{len(sub)}" if len(sub) else "--")
        if attempted == 0:
            lines.append(
                f"{SHORT_NAME[method]} & "
                + r"\multicolumn{6}{c}{\emph{not applicable to this design}} \\"
            )
            continue
        lines.append(" & ".join([SHORT_NAME[method], *times, *fails]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def failure_table(performance: pd.DataFrame, *, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Example 1 & Example 2 & Example 3 & $p=10$ design & Scaling \\",
        r"\midrule",
    ]
    families = [
        "example1",
        "example2",
        "example4-blocksparse",
        "example3",
        "scaling",
    ]
    for method in METHOD_ORDER:
        row = [SHORT_NAME[method]]
        for family in families:
            sub = performance[
                performance.scenario.str.startswith(family)
                & (performance.method == method)
            ]
            applicable = sub[sub.applicability == "applicable"]
            if applicable.empty:
                row.append(r"\emph{N/A}")
                continue
            failed = int((applicable.attempt_status == "failed").sum())
            row.append(f"{failed}/{len(applicable)}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, default=ROOT / "results" / "strict_formal_v2_repaired")
    parser.add_argument(
        "--block-sparse", type=Path, default=ROOT / "results" / "block_sparse"
    )
    parser.add_argument("--block-sparse-replications", type=int, default=100)
    parser.add_argument("--tables", type=Path, default=ROOT / "manuscript" / "tables")
    parser.add_argument("--figures", type=Path, default=ROOT / "manuscript" / "figures")
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp" / "componentwise.csv")
    parser.add_argument(
        "--curve-cache",
        type=Path,
        default=ROOT / "tmp" / "example2_curve_cache" / "example2_curves.npz",
    )
    args = parser.parse_args()

    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    performance = pd.read_csv(args.strict / "strict_results.csv")
    curves_path = args.strict / "strict_factor_curves.jsonl"
    componentwise = load_componentwise(curves_path, args.cache)
    # The curve stream and the audited result stream are written separately, and
    # a replication can appear in the first while the audit scores it a failure.
    # Componentwise accuracy is summarised over the same replications as the
    # prediction column beside it, so admission is applied once, here.
    admitted = set(
        map(
            tuple,
            performance.loc[
                performance.attempt_status == "success",
                ["scenario", "method", "replicate"],
            ].to_numpy(),
        )
    )
    before = len(componentwise)
    componentwise = componentwise[
        [
            (scenario, method, replicate) in admitted
            for scenario, method, replicate in zip(
                componentwise.scenario, componentwise.method, componentwise.replicate
            )
        ]
    ]
    print(f"componentwise rows: {before} streamed, {len(componentwise)} audited")

    # The block-sparse stream is written incrementally, so a partially finished
    # run must not reach the manuscript: every registered scenario has to carry
    # the full replication count before its table and figure are generated.
    block_results = args.block_sparse / "robust_results.csv"
    block_sparse_ready = False
    if block_results.exists():
        block_frame = pd.read_csv(block_results)
        counts = {
            scenario: int(
                block_frame[block_frame.scenario == scenario]["replicate"].nunique()
            )
            for scenario, _ in BLOCK_SPARSE_BLOCKS
        }
        block_sparse_ready = all(
            count >= args.block_sparse_replications for count in counts.values()
        )
        if block_sparse_ready:
            performance = pd.concat([performance, block_frame], ignore_index=True)
        else:
            print(f"block-sparse run incomplete, skipping its outputs: {counts}")

    example2_methods = ["TRACE-VCAM", "ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    example3_methods = [
        "TRACE-VCAM",
        "ZZW2020",
        "HHY2021-Huber",
        "ZSY2026-author-code",
        "ZY2025-paper-implementation",
    ]

    (args.tables / "example2_main.tex").write_text(
        componentwise_table(
            componentwise,
            performance,
            EXAMPLE2_BLOCKS,
            example2_methods,
            ["beta_0", "beta_1", "phi_1", "beta_2", "phi_2"],
            caption=(
                "Example 2 with $n=50$ subjects. Each entry is the median over the "
                "300 replications of the domain-averaged squared error of one "
                "identified component, with the median absolute deviation in "
                "parentheses; MSPE is the noise-free prediction error on held-out "
                "subjects. Boldface marks the best value in a column within a setting."
            ),
            label="tab:example2",
        ),
        encoding="utf-8",
    )

    if block_sparse_ready:
        (args.tables / "example3_main.tex").write_text(
            block_sparse_table(
                performance,
                caption=(
                    "Example 3: block-sparse design with $n=100$ subjects, $p=6$ "
                    "covariate blocks and two active blocks. Entries are medians over "
                    "the replications with median absolute deviations in parentheses; "
                    "$\\beta_0$ is the baseline error, $\\sum_k g_k$ aggregates the "
                    "component-surface error over all six blocks, Factors aggregates "
                    "the identified-factor error, and MSPE is the noise-free "
                    "prediction error on held-out subjects. Boldface marks the best "
                    "value in a column within a setting."
                ),
                label="tab:example3",
            ),
            encoding="utf-8",
        )
        figure_block_sparse(performance, args.figures)

    (args.tables / "supp_highdim.tex").write_text(
        componentwise_table(
            componentwise,
            performance,
            EXAMPLE3_BLOCKS,
            example3_methods,
            ["beta_0", "beta_1", "phi_1", "surface"],
            caption=(
                "Coefficient-sparse design of Zhao, Sun and Yang with $n=200$ "
                "subjects and $p=10$ covariate blocks, all of them active. Entries "
                "are medians of the domain-averaged squared error with median "
                "absolute deviations in parentheses; $\\beta_1$ and $\\phi_1$ are "
                "shown for the first block and the surface column aggregates all ten "
                "blocks. Boldface marks the best value in a column within a setting."
            ),
            label="tab:supp-highdim",
        ),
        encoding="utf-8",
    )

    (args.tables / "example1_full.tex").write_text(
        full_example_table(
            componentwise,
            performance,
            [("example1-zw2015-n100", "$n=100$")],
            ["TRACE-VCAM", "ZW2015"],
            caption=(
                "Example 1: complete Monte Carlo summary. Entries are medians with "
                "median absolute deviations in parentheses."
            ),
            label="tab:supp-example1",
        ),
        encoding="utf-8",
    )

    example2_all = [
        (f"example2-{family}-n{n}" + suffix, label.format(n=n))
        for family, suffix, label in (
            ("gaussian", "-sigma0.1", r"Normal, $\sigma=0.1$, $n={n}$"),
            ("gaussian", "-sigma0.4", r"Normal, $\sigma=0.4$, $n={n}$"),
            ("mixed-normal", "", r"Mixture, $n={n}$"),
            ("t2", "", r"$t_2$, $n={n}$"),
        )
        for n in (50, 100, 200)
    ]
    (args.tables / "example2_full.tex").write_text(
        full_example_table(
            componentwise,
            performance,
            example2_all,
            example2_methods,
            caption=(
                "Example 2: every registered sample size and error law. Entries are "
                "medians with median absolute deviations in parentheses; the last "
                "column reports failed over attempted fits."
            ),
            label="tab:supp-example2",
        ),
        encoding="utf-8",
    )

    example3_all = [
        (f"example3-{family}-n{n}-p10-sigma{sigma}", f"{label}, $n={n}$")
        for family, sigma, label in (
            ("gaussian", "0.1", r"Normal, $\sigma=0.1$"),
            ("gaussian", "0.4", r"Normal, $\sigma=0.4$"),
            ("contamination", "0.1", r"Contaminated, $\sigma=0.1$"),
            ("contamination", "0.4", r"Contaminated, $\sigma=0.4$"),
        )
        for n in (50, 200)
    ]
    (args.tables / "example3_full.tex").write_text(
        full_example_table(
            componentwise,
            performance,
            example3_all,
            example3_methods,
            caption=(
                "Coefficient-sparse design of Zhao, Sun and Yang: every registered "
                "sample size and error law at $p=10$. Entries are medians with "
                "median absolute deviations in parentheses."
            ),
            label="tab:supp-example3",
        ),
        encoding="utf-8",
    )

    (args.tables / "scaling.tex").write_text(
        scaling_table(
            performance,
            caption=(
                "Computational scaling in the number of covariate blocks at $n=200$ "
                "over five registered replications."
            ),
            label="tab:supp-scaling",
        ),
        encoding="utf-8",
    )

    (args.tables / "failure_audit.tex").write_text(
        failure_table(
            performance,
            caption=(
                "Failed over attempted fits by method and example family. A method "
                "outside its registered data regime is not attempted and is shown as "
                "not applicable."
            ),
            label="tab:supp-failures",
        ),
        encoding="utf-8",
    )

    figure_example1(componentwise, performance, args.figures)
    figure_curves(
        curves_path,
        "example1-zw2015-n100",
        "TRACE-VCAM",
        ["ZW2015"],
        args.figures,
        "supp_example1_curves.pdf",
        components=list(CURVE_COMPONENTS),
        height=2.2,
    )
    figure_example2(componentwise, performance, args.figures)
    figure_sample_size(componentwise, performance, args.figures)
    figure_sample_size_all_laws(componentwise, args.figures)

    # Main-text recovery display.  It uses the same sample size as Table 1 and
    # Figure 2, which is the smallest registered one and the one at which the
    # estimators separate most clearly; the curve cache avoids a second pass
    # over the full curve stream when it is available.
    curve_cache = None
    cache_file = args.curve_cache
    if cache_file is not None and cache_file.exists():
        curve_cache = np.load(cache_file)
    display_ranges = example2_covariate_display_range()
    print("Example 2 covariate display ranges:", display_ranges)
    # Every method that appears in Table 1 also appears here: a recovery display
    # that silently drops three of the four compared estimators cannot be checked
    # against the table it accompanies.
    example2_competitors = ["ZZW2020", "HHY2021-Huber", "ZY2025-paper-implementation"]
    # The main text shows the baseline and the two covariate factors, where the
    # margin over every competitor is largest; the two time factors, where it is
    # narrow, go to the Supplementary Material with the other three rather than
    # being left out.  Table 1 carries all five components for all four methods
    # either way, so nothing the figure omits is unavailable.
    print(
        "Example 2 recovery figure, contaminated mixture, n=50:",
        json.dumps(
            figure_curves(
                curves_path,
                "example2-mixed-normal-n50",
                "TRACE-VCAM",
                example2_competitors,
                args.figures,
                "example2_curves.pdf",
                components=["baseline", "phi_1", "phi_2"],
                height=2.35,
                covariate_ranges=display_ranges,
                cache=curve_cache,
            ),
            indent=2,
        ),
    )
    # The neutral companion to the display above: no replication is selected,
    # so nothing about any fit decides what is drawn.
    print(
        "Example 2 envelope figure, contaminated mixture, n=50:",
        json.dumps(
            figure_curve_envelopes(
                curves_path,
                "example2-mixed-normal-n50",
                "TRACE-VCAM",
                example2_competitors,
                args.figures,
                "example2_envelopes.pdf",
                components=["baseline", "phi_1", "phi_2"],
                height=2.35,
                covariate_ranges=display_ranges,
                cache=curve_cache,
            ),
            indent=2,
        ),
    )
    for scenario, filename in (
        ("example2-mixed-normal-n50", "supp_example2_curves_mixed.pdf"),
        ("example2-t2-n50", "supp_example2_curves.pdf"),
    ):
        print(
            f"Example 2 recovery figure, {scenario}:",
            json.dumps(
                figure_curves(
                    curves_path,
                    scenario,
                    "TRACE-VCAM",
                    example2_competitors,
                    args.figures,
                    filename,
                    components=list(CURVE_COMPONENTS),
                    height=2.2,
                    covariate_ranges=display_ranges,
                    cache=curve_cache,
                ),
                indent=2,
            ),
        )
    if curve_cache is not None:
        stacks = _curve_stacks(
            curves_path,
            "example2-mixed-normal-n50",
            ["TRACE-VCAM", *example2_competitors],
            curve_cache,
        )
        print(
            "Example 2 band-width ratios, contaminated mixture, n=50:",
            json.dumps(
                curve_band_ratios(
                    stacks, "TRACE-VCAM", covariate_ranges=display_ranges
                ),
                indent=2,
            ),
        )
    figure_example3(componentwise, performance, args.figures)
    figure_scaling(performance, args.figures)
    print("manuscript tables and figures rebuilt")


if __name__ == "__main__":
    main()

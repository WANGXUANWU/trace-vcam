"""Table and figure for the PBC bilirubin application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_manuscript_outputs import FIGURE_WIDTH_IN, METHOD_COLOR, panel  # noqa: E402
from scripts.manuscript_common import SHORT_NAME, bold  # noqa: E402

METHODS = (
    "TRACE-VCAM",
    "ZZW2020",
    "HHY2021-Huber",
    "ZY2025-paper-implementation",
)

CURVE_TITLE = {
    "baseline": panel("%s", r"Baseline $\beta_0$"),
    "beta_2": panel("%s", r"$\beta_2$: Time modulation of albumin"),
    "phi_2": panel("%s", r"$\phi_2$: Albumin effect"),
}
CURVE_YLABEL = {
    "baseline": "log bilirubin",
    "beta_2": "Relative effect",
    "phi_2": "log bilirubin",
}
CURVE_COORDINATE = {"baseline": "time", "beta_2": "time", "phi_2": "albumin"}
AXIS_LABEL = {
    "time": "Years since registration",
    "albumin": "Serum albumin (g/dl)",
}


def cv_table(frame: pd.DataFrame, *, caption: str, label: str) -> tuple[str, dict]:
    """Fold-level prediction summaries on the folds all methods completed."""

    ok = frame[frame.attempt_status == "success"]
    keys = {
        method: set(zip(ok[ok.method == method]["repeat"], ok[ok.method == method]["fold"]))
        for method in METHODS
    }
    shared = set.intersection(*keys.values()) if keys else set()
    audit: dict[str, object] = {"shared_folds": len(shared)}

    rows: list[str] = []
    best: dict[str, float] = {}
    cells: dict[str, dict[str, tuple[float, float]]] = {}
    for method in METHODS:
        block = ok[
            (ok.method == method)
            & [(r, f) in shared for r, f in zip(ok["repeat"], ok["fold"])]
        ]
        cells[method] = {}
        for column in ("mspe", "balanced_mspe", "mape", "runtime_seconds"):
            values = block[column].to_numpy(dtype=float)
            cells[method][column] = (float(np.mean(values)), float(np.std(values)))
        audit[SHORT_NAME[method]] = {
            "fits": f"{len(keys[method])}/{int((frame.method == method).sum())}",
            **{k: v[0] for k, v in cells[method].items()},
        }
    for column in ("mspe", "balanced_mspe", "mape", "runtime_seconds"):
        best[column] = min(cells[m][column][0] for m in METHODS)

    for method in METHODS:
        entries = [SHORT_NAME[method]]
        completed = len(keys[method])
        attempted = int((frame.method == method).sum())
        entries.append(f"{completed}/{attempted}")
        for column, digits in (
            ("mspe", 3), ("balanced_mspe", 3), ("mape", 3), ("runtime_seconds", 1)
        ):
            mean, sd = cells[method][column]
            text = f"{mean:.{digits}f} ({sd:.{digits}f})"
            entries.append(bold(text) if abs(mean - best[column]) < 1e-12 else text)
        rows.append(" & ".join(entries) + r" \\")

    lines = [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Method & Fits & MSPE & Subject-balanced & MAPE & Time (s) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    return "\n".join(lines), audit


def components_figure(payload: dict, draws, output: Path, filename: str) -> dict:
    bounds = payload["coordinate_bounds"]
    point = payload["point_estimate"]
    components = ["baseline", "beta_2", "phi_2"]
    # The adapter evaluates its factor curves on its own grid, which need not be
    # the one recorded alongside them; take the length from the curves so the
    # two cannot disagree.
    grid = np.linspace(0.0, 1.0, len(point["baseline"]))

    fig, axes = plt.subplots(
        1, 4, figsize=(FIGURE_WIDTH_IN, 2.75), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.85]},
    )
    colour = METHOD_COLOR["TRACE-VCAM"]
    audit: dict[str, object] = {}
    for letter, axis, name in zip("abc", axes, components):
        key = CURVE_COORDINATE[name]
        low, high = bounds[key]
        horizontal = low + grid * (high - low)
        values = np.asarray(point[name], dtype=float)
        drawn = [values]
        if draws is not None and f"draws::{name}" in draws:
            stack = np.asarray(draws[f"draws::{name}"], dtype=float)
            block = name.split("_")[-1]
            retained = draws.get(f"draws::retained_{block}")
            conditional = stack
            if retained is not None:
                keep = np.asarray(retained, dtype=float).ravel() > 0.5
                if keep.sum() >= 20:
                    conditional = stack[keep]
            outer_low, outer_high = np.percentile(stack, [10.0, 90.0], axis=0)
            inner_low, inner_high = np.percentile(conditional, [25.0, 75.0], axis=0)
            axis.fill_between(horizontal, outer_low, outer_high, color=colour,
                              alpha=0.13, linewidth=0, zorder=2)
            axis.fill_between(horizontal, inner_low, inner_high, color=colour,
                              alpha=0.32, linewidth=0, zorder=3)
            drawn += [outer_low, outer_high]
            audit.setdefault("band_width", {})[name] = float(
                np.mean(inner_high - inner_low)
            )
            audit.setdefault("outer_contains_zero", {})[name] = bool(
                np.all((outer_low <= 0.0) & (outer_high >= 0.0))
            )
        axis.plot(horizontal, values, color=colour, linewidth=1.7, zorder=5,
                  label="TRACE full-data fit")
        axis.axhline(0.0, color="0.6", linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
        axis.set_title(CURVE_TITLE[name] % letter, pad=3)
        axis.set_xlabel(AXIS_LABEL[key], labelpad=1.5)
        axis.set_ylabel(CURVE_YLABEL[name], labelpad=2)
        lows = [float(np.min(item)) for item in drawn]
        highs = [float(np.max(item)) for item in drawn]
        pad = 0.10 * (max(highs) - min(lows))
        axis.set_ylim(min(lows) - pad, max(highs) + pad)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.22)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        axis.tick_params(length=2.5, width=0.6)

    # Retention panel, in the idiom of the CD4 figure.
    axis = axes[3]
    age = np.asarray(draws["draws::retained_1"], dtype=float).ravel() > 0.5
    alb = np.asarray(draws["draws::retained_2"], dtype=float).ravel() > 0.5
    counts = {
        "Albumin only": int(np.sum(~age & alb)),
        "Both": int(np.sum(age & alb)),
        "Age only": int(np.sum(age & ~alb)),
        "Neither": int(np.sum(~age & ~alb)),
    }
    # Ranked, like the CD4 panel: the reader is comparing four proportions, and
    # the pattern the full-data fit returns need not be the most frequent one.
    counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
    total = int(age.size)
    shares = {k: 100.0 * v / total for k, v in counts.items()}
    widest = max(shares.values()) or 1.0
    label_column, value_gap, label_gap = 0.74 * widest, 0.045 * widest, 0.075 * widest
    positions = np.arange(len(counts))[::-1]
    axis.axvline(0.0, color="0.55", linewidth=0.7, ymin=0.06, ymax=0.94, zorder=1)
    for position, (key, share) in zip(positions, shares.items()):
        chosen = key == "Albumin only"
        if chosen:
            axis.axhspan(position - 0.44, position + 0.44, color=colour,
                         alpha=0.07, linewidth=0, zorder=0)
        if share > 0.0:
            axis.barh(position, share, height=0.52,
                      color=colour if chosen else "0.82", linewidth=0, zorder=2)
        axis.text(share + value_gap, position, f"{share:.1f}%", va="center",
                  ha="left", fontsize=7.5, fontweight="bold" if chosen else "normal",
                  color=colour if chosen else "0.40", zorder=3)
        axis.text(-label_gap, position + (0.13 if chosen else 0.0), key, va="center",
                  ha="right", fontsize=8, color="0.10" if chosen else "0.42", zorder=3)
        if chosen:
            axis.text(-label_gap, position - 0.28, "the full-data fit", va="center",
                      ha="right", fontsize=6.5, style="italic", color=colour, zorder=3)
    axis.set_xlim(-label_column, widest + 0.42 * widest)
    axis.set_ylim(-0.72, len(counts) - 0.28)
    axis.set_title(panel("d", "Blocks the fit retains"), pad=3)
    axis.set_xlabel(f"Share of {total:,} resamples", labelpad=1.5)
    axis.set_xticks([]); axis.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    audit["retention_patterns"] = {**counts, "total": total}

    fig.savefig(output / filename)
    plt.close(fig)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbc", type=Path, default=ROOT / "results" / "pbc")
    parser.add_argument("--tables", type=Path, default=ROOT / "manuscript" / "tables")
    parser.add_argument("--figures", type=Path, default=ROOT / "manuscript" / "figures")
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    payload = json.loads((args.pbc / "pbc_full_data.json").read_text(encoding="utf-8"))
    draws_path = args.pbc / "pbc_bootstrap_draws.npz"
    draws = np.load(draws_path) if draws_path.exists() else None
    audit: dict[str, object] = {"figure": components_figure(
        payload, draws, args.figures, "pbc_components.pdf"
    )}

    cv_path = args.pbc / "pbc_cv_results.csv"
    if cv_path.exists():
        frame = pd.read_csv(cv_path)
        table, cv_audit = cv_table(
            frame,
            caption=(
                "Mayo PBC sequential data. Five repeats of fivefold subject-level "
                "cross-validation on 312 patients; every method is fitted on the "
                "same training subjects and evaluated on the same held-out "
                "subjects, and the summary uses the folds on which all methods "
                "returned a fit. The response is log serum bilirubin. Entries are "
                "means over folds with standard deviations in parentheses; the "
                "subject-balanced error averages within subjects before averaging "
                "across them. Boldface marks the best value in a column."
            ),
            label="tab:pbc",
        )
        (args.tables / "pbc_cv_main.tex").write_text(table, encoding="utf-8")
        audit["cv"] = cv_audit

    audit["rank_diagnostic"] = payload.get("rank_diagnostic")
    audit["retention"] = payload.get("retention")
    audit["bootstrap"] = [payload.get("bootstrap_completed"), payload.get("bootstrap_attempted")]
    print(json.dumps(audit, indent=2, default=float))


if __name__ == "__main__":
    main()

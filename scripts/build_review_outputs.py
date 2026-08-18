"""Tables and figures for the revision experiments.

Reads the result streams written by ``run_review_experiments.py``,
``run_robustness_scope.py``, and ``run_block_sensitivity.py``, together with the
existing strict benchmark stream, and writes the manuscript artefacts each of
them supports.  It never refits a model.
"""

from __future__ import annotations

import argparse
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
    panel,
)
from scripts.manuscript_common import SHORT_NAME, bold, fmt  # noqa: E402

ERROR_LAW_LABEL = {
    "gaussian": r"Normal, $\sigma=0.4$",
    "hhy-mixed-normal": "Contaminated mixture",
    "hhy-t2": r"Scaled $t_2$",
}
LAW_ORDER = ("gaussian", "hhy-mixed-normal", "hhy-t2")

LOCKED_LAMBDA = 0.03
LOCKED_ROUGHNESS = 0.05


def _table(
    *,
    caption: str,
    label: str,
    column_spec: str,
    header: Sequence[str],
    body: Sequence[str],
    small: bool = True,
    column_sep: str | None = "4pt",
) -> str:
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if small:
        lines.append(r"\footnotesize")
    if column_sep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{column_sep}}}")
    lines.append(rf"\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}")
    lines.append(r"\toprule")
    lines.extend(header)
    lines.append(r"\midrule")
    lines.extend(body)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _sci(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    if 1e-3 <= abs(value) < 1e4:
        return f"{value:.{digits + 2}f}"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


# ---------------------------------------------------------------------------
# 1. The normalisation margin
# ---------------------------------------------------------------------------


def normalisation_outputs(path: Path, tables: Path, figures: Path) -> dict:
    frame = pd.read_csv(path)
    grouped = frame.groupby("time_factor_mean")
    summary = grouped.median(numeric_only=True).sort_index(ascending=False)
    replications = int(len(frame) / frame["time_factor_mean"].nunique())

    body = []
    for mean, row in summary.iterrows():
        body.append(
            " & ".join(
                [
                    f"{mean:g}",
                    f"{row['normalisation_margin']:.3f}",
                    f"{row['surface_block1']:.3f}",
                    f"{row['mspe']:.3f}",
                    _sci(row["beta_ise_integral"]),
                    f"{row['beta_sup_integral']:.1f}",
                    f"{row['phi_norm_integral']:.3f}",
                    f"{row['beta_ise_l2']:.3f}",
                    f"{row['phi_ise_l2']:.3f}",
                ]
            )
            + r" \\"
        )
    table = _table(
        caption=(
            "Components whose time factor has a mean approaching zero. The first "
            "component of the Example 2 design is shifted so that its time factor "
            r"averages $\bar\beta_1$ over follow-up; nothing else in the design "
            "changes, and the component remains an ordinary nonzero separable "
            "effect throughout. Entries are medians over "
            f"{replications}"
            " replications, pooled over the three error laws, which agree. "
            r"$\widehat c_q$ is the realised normalisation margin "
            r"$\lvert q^\trans\widehat u_1\rvert$. The component surface and the "
            "prediction do not depend on how the two factors are normalised; the "
            "reported factors do."
        ),
        label="tab:supp-normalisation",
        column_spec="cccccccccc"[:9],
        header=[
            r" & & \multicolumn{2}{c}{Convention-free} & "
            r"\multicolumn{3}{c}{$\int\beta_1=1$} & \multicolumn{2}{c}{$L^2$ and sign} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
            r"$\bar\beta_1$ & $\widehat c_q$ & $g_1$ & MSPE & "
            r"$\beta_1$ & $\lVert\widehat\beta_1\rVert_\infty$ & "
            r"$\lVert\widehat\phi_1\rVert_2$ & $\beta_1$ & $\phi_1$ \\",
        ],
        body=body,
    )
    (tables / "normalisation_margin.tex").write_text(table, encoding="utf-8")

    # ---- figure -----------------------------------------------------------
    means = summary.index.to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.35), constrained_layout=True)
    colour = METHOD_COLOR["TRACE-VCAM"]

    axis = axes[0]
    axis.plot(means, summary["surface_block1"], "o-", color=colour, markersize=3.2,
              linewidth=1.4, label=r"Component $g_1$")
    axis.plot(means, summary["mspe"], "s--", color="0.35", markersize=3.0,
              linewidth=1.1, label="Prediction")
    axis.set_ylim(0.0, max(summary["surface_block1"].max(), summary["mspe"].max()) * 1.5)
    axis.set_title(panel("a", "What the estimator delivers"), pad=3)
    axis.set_ylabel("Domain-averaged squared error", labelpad=2)
    axis.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=1)

    axis = axes[1]
    axis.plot(means, summary["beta_ise_integral"], "o-", color="#b2182b", markersize=3.2,
              linewidth=1.4, label=r"$\int\beta_1=1$")
    axis.plot(means, summary["beta_ise_l2"], "s-", color=colour, markersize=3.2,
              linewidth=1.4, label=r"$L^2$ and sign")
    axis.set_yscale("log")
    axis.set_title(panel("b", "Reported time factor"), pad=3)
    axis.set_ylabel(r"Error in $\beta_1$", labelpad=2)
    axis.legend(frameon=False, fontsize=7.5, loc="lower left")

    axis = axes[2]
    axis.plot(means, summary["beta_sup_integral"], "o-", color="#b2182b", markersize=3.2,
              linewidth=1.4, label=r"$\|\widehat\beta_1\|_\infty$")
    axis.plot(means, summary["phi_norm_integral"], "s-", color="#2166ac", markersize=3.2,
              linewidth=1.4, label=r"$\|\widehat\phi_1\|_2$")
    axis.set_yscale("log")
    axis.set_title(panel("c", "The pair separates"), pad=3)
    axis.set_ylabel("Size of the reported factor", labelpad=2)
    axis.legend(frameon=False, fontsize=7.5, loc="center left")

    for axis in axes:
        axis.set_xlabel(r"Mean of the time factor, $\bar\beta_1$", labelpad=1.5)
        axis.invert_xaxis()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.6)
    fig.savefig(figures / "normalisation_margin.pdf")
    plt.close(fig)

    return {
        "replications": replications,
        "surface_range": [float(summary["surface_block1"].min()), float(summary["surface_block1"].max())],
        "mspe_range": [float(summary["mspe"].min()), float(summary["mspe"].max())],
        "beta_integral_range": [float(summary["beta_ise_integral"].min()), float(summary["beta_ise_integral"].max())],
        "beta_l2_range": [float(summary["beta_ise_l2"].min()), float(summary["beta_ise_l2"].max())],
    }


# ---------------------------------------------------------------------------
# 2. Block selection along the penalty path
# ---------------------------------------------------------------------------


def selection_outputs(path: Path, tables: Path, figures: Path) -> dict:
    frame = pd.read_csv(path)
    ratios = sorted(frame["lambda_ratio"].unique())
    replications = int(frame.groupby(["error_law", "lambda_ratio"]).size().iloc[0])

    def cell(block: pd.DataFrame) -> dict[str, float]:
        return {
            "size": float(np.median(block["model_size_numeric"])),
            "tpr": float(np.mean(block["tpr_numeric"])),
            "fpr": float(np.mean(block["fpr_numeric"])),
            "exact": float(np.mean(block["model_size_numeric"] == 2)),
            "surface": float(np.median(block["surface_total"])),
        }

    summary = {
        (law, ratio): cell(
            frame[(frame.error_law == law) & (frame.lambda_ratio == ratio)]
        )
        for law in LAW_ORDER
        for ratio in ratios
    }

    body = []
    for ratio in ratios:
        marker = r"$^\dagger$" if abs(ratio - LOCKED_LAMBDA) < 1e-12 else ""
        cells = [f"{ratio:g}{marker}"]
        for law in LAW_ORDER:
            item = summary[(law, ratio)]
            cells += [
                f"{item['size']:.0f}",
                f"{item['fpr']:.2f}",
                f"{item['exact']:.2f}",
                f"{item['surface']:.3f}",
            ]
        body.append(" & ".join(cells) + r" \\")

    table = _table(
        caption=(
            "Block recovery along the penalty path of the Example 3 design, over "
            f"{replications}"
            r" replications at $n=100$, $p=6$, and two active blocks. "
            r"$\lvert\widehat{\mathcal A}\rvert$ is the median number of "
            "numerically nonzero blocks, FPR the mean share of the four inactive "
            "blocks that are retained, ``exact'' the share of replications whose "
            "retained set is precisely the two active blocks, and "
            r"$\sum_k g_k$ the median aggregated component-surface error. The "
            "true positive rate is one at every penalty level in every "
            "replication and every error law, and is therefore not tabulated. "
            r"$^\dagger$ marks the locked penalty the paper reports."
        ),
        label="tab:supp-selection-path",
        column_spec="l" + "cccc" * 3,
        header=[
            " & "
            + " & ".join(
                rf"\multicolumn{{4}}{{c}}{{{ERROR_LAW_LABEL[law]}}}" for law in LAW_ORDER
            )
            + r" \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
            r"$\lambda/\lambda_{\max}$ & "
            + " & ".join(
                [r"$\lvert\widehat{\mathcal A}\rvert$ & FPR & exact & $\sum_k g_k$"]
                * 3
            )
            + r" \\",
        ],
        body=body,
    )
    (tables / "selection_path.tex").write_text(table, encoding="utf-8")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.35), constrained_layout=True)
    law_style = {
        "gaussian": ("-", "#4d4d4d"),
        "hhy-mixed-normal": ("-", METHOD_COLOR["TRACE-VCAM"]),
        "hhy-t2": ("--", "#b2182b"),
    }

    axis = axes[0]
    for law, (style, colour) in law_style.items():
        block = frame[frame.error_law == law].groupby("lambda_ratio").median(numeric_only=True)
        axis.plot(block.index, block["frobenius_active"], style, color=colour, linewidth=1.4)
        axis.plot(block.index, block["frobenius_inactive"], style, color=colour,
                  linewidth=1.0, alpha=0.55)
    axis.set_title(panel("a", "Fitted block size"), pad=3)
    axis.set_ylabel(r"$\|\widehat\Theta_k\|_F$", labelpad=2)
    axis.annotate("active", xy=(0.06, 1.16), fontsize=7.5, color="0.2")
    axis.annotate("inactive", xy=(0.06, 0.30), fontsize=7.5, color="0.45")

    axis = axes[1]
    for law, (style, colour) in law_style.items():
        block = frame[frame.error_law == law].groupby("lambda_ratio").mean(numeric_only=True)
        axis.plot(block.index, block["fpr_numeric"], style, color=colour, linewidth=1.4,
                  label=ERROR_LAW_LABEL[law])
    axis.set_ylim(-0.04, 1.06)
    axis.set_title(panel("b", "Inactive blocks retained"), pad=3)
    axis.set_ylabel("False positive rate", labelpad=2)
    axis.legend(frameon=False, fontsize=7, loc="lower left")

    axis = axes[2]
    for law, (style, colour) in law_style.items():
        block = frame[frame.error_law == law].groupby("lambda_ratio").median(numeric_only=True)
        axis.plot(block.index, block["surface_total"], style, color=colour, linewidth=1.4)
    axis.set_title(panel("c", "Accuracy cost"), pad=3)
    axis.set_ylabel(r"$\sum_k\mathrm{ISE}(\widehat g_k)$", labelpad=2)

    for axis in axes:
        axis.axvline(LOCKED_LAMBDA, color="0.35", linewidth=0.7, linestyle=(0, (1, 2)))
        axis.set_xscale("log")
        axis.set_xlabel(r"$\lambda/\lambda_{\max}$", labelpad=1.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.6)
    # The locked penalty is named once, in the panel with room for it, rather
    # than in every panel where the rule is drawn.
    axes[2].annotate(
        "locked penalty",
        xy=(LOCKED_LAMBDA, 1.0), xycoords=("data", "axes fraction"),
        xytext=(3, -7), textcoords="offset points",
        fontsize=6.5, color="0.35", va="top", ha="left",
    )
    fig.savefig(figures / "selection_path.pdf")
    plt.close(fig)

    exact_at = {
        law: {
            ratio: summary[(law, ratio)]["exact"] for ratio in ratios
        }
        for law in LAW_ORDER
    }
    return {
        "replications": replications,
        "exact_recovery": exact_at,
        "summary": {f"{law}@{ratio:g}": value for (law, ratio), value in summary.items()},
    }


# ---------------------------------------------------------------------------
# 3. Tuning sensitivity
# ---------------------------------------------------------------------------


#: Which registered benchmark scenario each tuning-sweep setting corresponds to,
#: so that the sweep can be read against the competitor medians it is meant to
#: put in perspective rather than against itself.
TUNING_REFERENCE = {
    ("example2", "gaussian"): "example2-gaussian-n50-sigma0.4",
    ("example2", "hhy-mixed-normal"): "example2-mixed-normal-n50",
    ("example2", "hhy-t2"): "example2-t2-n50",
    ("example3", "gaussian"): "example4-blocksparse-normal-n100",
    ("example3", "hhy-mixed-normal"): "example4-blocksparse-mixed-normal-n100",
    ("example3", "hhy-t2"): "example4-blocksparse-t2-n100",
}


def _competitor_medians(strict: Path, block_sparse: Path) -> dict[str, float]:
    """Best competitor median component error in each registered scenario."""

    frames = [pd.read_csv(strict, low_memory=False)]
    if block_sparse.exists():
        frames.append(pd.read_csv(block_sparse, low_memory=False))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[
        (frame.applicability == "applicable")
        & (frame.attempt_status == "success")
        & (frame.method != "TRACE-VCAM")
    ]
    out: dict[str, float] = {}
    for scenario, block in frame.groupby("scenario"):
        medians = block.groupby("method")["component_ise"].median().dropna()
        if len(medians):
            out[str(scenario)] = float(medians.min())
    return out


def tuning_outputs(path: Path, strict: Path, block_sparse: Path, tables: Path) -> dict:
    frame = pd.read_csv(path)
    reference = _competitor_medians(strict, block_sparse)
    body = []
    audit: dict[str, object] = {}
    for design, design_label in (("example2", "Example 2, $n=50$"), ("example3", "Example 3, $n=100$")):
        for law in LAW_ORDER:
            block = frame[(frame.design == design) & (frame.error_law == law)]
            if block.empty:
                continue
            per_cell = block.groupby(["lambda_ratio", "roughness"])["surface_total"].median()
            locked = float(per_cell.loc[(LOCKED_LAMBDA, LOCKED_ROUGHNESS)])
            best = float(per_cell.min())
            worst = float(per_cell.max())
            # The per-replication oracle: the grid point that minimises this
            # replication's own error, which no rule could beat.
            oracle = float(
                np.median(
                    block.groupby("seed")["surface_total"].min().to_numpy(dtype=float)
                )
            )
            rival = reference.get(TUNING_REFERENCE[(design, law)], float("nan"))
            beaten = (
                int(np.sum(per_cell.to_numpy() < rival)) if np.isfinite(rival) else 0
            )
            body.append(
                " & ".join(
                    [
                        design_label if law == LAW_ORDER[0] else "",
                        ERROR_LAW_LABEL[law],
                        f"{locked:.3f}",
                        f"{best:.3f}",
                        f"{worst:.3f}",
                        f"{oracle:.3f}",
                        fmt(rival, 3),
                        f"{beaten}/{per_cell.size}",
                    ]
                )
                + r" \\"
            )
            audit[f"{design}/{law}"] = {
                "locked": locked, "best_grid": best, "worst_grid": worst,
                "per_replication_oracle": oracle, "locked_over_best": locked / best,
                "best_competitor": rival, "grid_points_ahead": beaten,
                "grid_points": int(per_cell.size),
            }
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    replications = int(frame.groupby(["design", "error_law", "lambda_ratio", "roughness"]).size().iloc[0])
    table = _table(
        caption=(
            "Sensitivity of the proposed estimator to its tuning protocol. Every "
            "entry is a median aggregated component-surface error over "
            f"{replications}"
            r" replications, refitting over the grid $\lambda/\lambda_{\max}\in"
            r"\{0.01,0.02,0.03,0.05,0.10,0.20\}$ and $\mu\in\{0,0.01,0.05,0.20\}$. "
            r"``Locked'' is the pilot-calibrated pair the paper reports; "
            "``best'' and ``worst'' are the best and worst single grid points for "
            "that setting; ``oracle'' gives each replication its own best grid "
            "point, which no tuning rule can attain. ``Rival'' is the smallest "
            "median attained by any competitor in the corresponding registered "
            "scenario, each competitor using the tuning rule of its own source "
            "paper, and the last column counts the grid points at which the "
            "proposed estimator is still ahead of it."
        ),
        label="tab:supp-tuning-sensitivity",
        column_spec="llcccccc",
        header=[
            r"Design & Error law & Locked & Best & Worst & Oracle & Rival & Ahead \\",
        ],
        body=body,
    )
    (tables / "tuning_sensitivity.tex").write_text(table, encoding="utf-8")
    return audit


# ---------------------------------------------------------------------------
# 4. Rank-one adequacy diagnostic
# ---------------------------------------------------------------------------


def rank_outputs(path: Path, tables: Path, figures: Path) -> dict:
    frame = pd.read_csv(path)
    defects = sorted(frame["defect_coefficient"].unique())
    replications = int(frame.groupby(["error_law", "defect_coefficient"]).size().iloc[0])

    body = []
    for defect in defects:
        block = frame[frame.defect_coefficient == defect]
        cells = [f"{defect:g}", f"{np.median(block['zeta']):.2f}"]
        for law in LAW_ORDER:
            law_block = block[block.error_law == law]
            cells += [
                f"{np.median(law_block['block1_spectral_ratio_max']):.3f}",
                f"{np.median(law_block['rank2_gain']):+.2f}",
                f"{np.mean(law_block['rank2_gain'] > 0.0):.2f}",
            ]
        body.append(" & ".join(cells) + r" \\")

    table = _table(
        caption=(
            "A practical diagnostic for the rank-one restriction, calibrated on "
            "components whose second direction is grown from nothing to eight "
            "times the strength of the first. "
            r"$\zeta$ is the separability defect of the first component; "
            r"$\widehat\sigma_2/\widehat\sigma_1$ is the singular-value ratio of "
            "the fitted pilot block for that component; ``gain'' is the share of "
            "held-out prediction error that a rank-two projection removes, "
            "negative when the extra direction costs more than it buys; and "
            "``detect'' is the share of replications in which that gain is "
            "positive. Entries are medians and proportions over "
            f"{replications}"
            r" replications at $n=200$. The first row is an exactly separable "
            "component, so the detection column there is a false-positive rate."
        ),
        label="tab:supp-rank-diagnostic",
        column_spec="ll" + "ccc" * 3,
        header=[
            r" & & "
            + " & ".join(
                rf"\multicolumn{{3}}{{c}}{{{ERROR_LAW_LABEL[law]}}}" for law in LAW_ORDER
            )
            + r" \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
            r"$c$ & $\zeta$ & "
            + " & ".join(
                [r"$\widehat\sigma_2/\widehat\sigma_1$ & gain & detect"] * 3
            )
            + r" \\",
        ],
        body=body,
    )
    (tables / "rank_diagnostic.tex").write_text(table, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN * 0.72, 2.35), constrained_layout=True)
    law_style = {
        "gaussian": ("-", "#4d4d4d"),
        "hhy-mixed-normal": ("-", METHOD_COLOR["TRACE-VCAM"]),
        "hhy-t2": ("--", "#b2182b"),
    }
    zeta = frame.groupby("defect_coefficient")["zeta"].median()
    for law, (style, colour) in law_style.items():
        block = frame[frame.error_law == law]
        gain = block.groupby("defect_coefficient")["rank2_gain"].median()
        detect = block.groupby("defect_coefficient")["rank2_gain"].apply(lambda v: float(np.mean(v > 0)))
        axes[0].plot(zeta.to_numpy(), gain.to_numpy(), style, color=colour, linewidth=1.4,
                     marker="o", markersize=3.0, label=ERROR_LAW_LABEL[law])
        axes[1].plot(zeta.to_numpy(), detect.to_numpy(), style, color=colour, linewidth=1.4,
                     marker="o", markersize=3.0, label=ERROR_LAW_LABEL[law])
    axes[0].axhline(0.0, color="0.4", linewidth=0.7, linestyle=(0, (1, 2)))
    axes[0].set_title(panel("a", "Gain from a second direction"), pad=3)
    axes[0].set_ylabel("Share of held-out error removed", labelpad=2)
    axes[1].set_ylim(-0.04, 1.06)
    axes[1].set_title(panel("b", "How often the gain is positive"), pad=3)
    axes[1].set_ylabel("Detection rate", labelpad=2)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    for axis in axes:
        axis.set_xlabel(r"Separability defect $\zeta$", labelpad=1.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.6)
    fig.savefig(figures / "rank_diagnostic.pdf")
    plt.close(fig)

    return {
        "replications": replications,
        "false_positive_rate": {
            law: float(
                np.mean(
                    frame[(frame.error_law == law) & (frame.defect_coefficient == 0.0)][
                        "rank2_gain"
                    ]
                    > 0.0
                )
            )
            for law in LAW_ORDER
        },
        "power_at_max": {
            law: float(
                np.mean(
                    frame[
                        (frame.error_law == law)
                        & (frame.defect_coefficient == max(defects))
                    ]["rank2_gain"]
                    > 0.0
                )
            )
            for law in LAW_ORDER
        },
    }


# ---------------------------------------------------------------------------
# 5. Failure-aware comparison
# ---------------------------------------------------------------------------

FAILURE_SCENARIOS = (
    ("example2-gaussian-n50-sigma0.1", r"Example 2, normal $\sigma=0.1$, $n=50$"),
    ("example2-mixed-normal-n50", r"Example 2, mixture, $n=50$"),
    ("example2-t2-n50", r"Example 2, $t_2$, $n=50$"),
)


def zero_estimate_component_ise(scenario: str) -> float:
    """Aggregated component-surface error of the estimate that returns nothing.

    This is the penalty the benchmark already applies to a component a method
    does not deliver.  Applying it to a fit that does not return at all is the
    same convention one level up, and it is a fixed property of the design, so a
    failure-aware summary cannot be moved by one extreme completed replication.
    """

    from scripts.manuscript_common import GRID_SIZE, registered_truth

    key = scenario
    if scenario.startswith("example4-blocksparse"):
        # The block-sparse design carries the Example 2 factor pairs on its two
        # active blocks, and the remaining blocks are the zero surface.
        key = "example2"
    truth, time_domain, _ = registered_truth(key)
    time_grid = np.linspace(time_domain[0], time_domain[1], GRID_SIZE)
    covariate_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    total = 0.0
    for index, active in enumerate(truth.active):
        if not active:
            continue
        surface = np.outer(truth.beta[index](time_grid), truth.phi[index](covariate_grid))
        total += float(np.mean(surface**2))
    return total


def failure_outputs(
    strict: Path, block_sparse: Path, tables: Path
) -> dict:
    frames = [pd.read_csv(strict, low_memory=False)]
    if block_sparse.exists():
        frames.append(pd.read_csv(block_sparse, low_memory=False))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.get("mode", "formal") == "formal"] if "mode" in frame else frame

    scenarios = list(FAILURE_SCENARIOS) + [
        ("example4-blocksparse-normal-n100", r"Example 3, normal, $n=100$"),
        ("example4-blocksparse-mixed-normal-n100", r"Example 3, mixture, $n=100$"),
        ("example4-blocksparse-t2-n100", r"Example 3, $t_2$, $n=100$"),
    ]

    body = []
    audit: dict[str, object] = {}
    for scenario, label in scenarios:
        block = frame[frame.scenario == scenario]
        if block.empty:
            continue
        applicable = block[block.applicability == "applicable"]
        methods = [m for m in SHORT_NAME if m in set(applicable.method)]
        succeeded = {
            method: set(
                applicable[
                    (applicable.method == method) & (applicable.attempt_status == "success")
                ]["replicate"]
            )
            for method in methods
        }
        shared = set.intersection(*succeeded.values()) if succeeded else set()
        # A failed fit delivers nothing, so the failure-aware summary scores it
        # exactly as a component a method does not deliver is already scored: at
        # the zero estimate.  That penalty is a property of the design, not of
        # the observed fits, so it cannot be moved by an extreme replication.
        zero_penalty = zero_estimate_component_ise(scenario)
        worst = {}
        for method in methods:
            rows = applicable[applicable.method == method]
            values = [
                float(row["component_ise"])
                if row["attempt_status"] == "success"
                and np.isfinite(row["component_ise"])
                else zero_penalty
                for _, row in rows.iterrows()
            ]
            worst[method] = float(np.median(values))
        first = True
        for method in methods:
            rows = applicable[applicable.method == method]
            attempted = int(len(rows))
            wins = int((rows.attempt_status == "success").sum())
            conditional = float(
                np.median(rows[rows.attempt_status == "success"]["component_ise"])
            ) if wins else float("nan")
            matched = rows[rows.replicate.isin(shared) & (rows.attempt_status == "success")]
            matched_median = (
                float(np.median(matched["component_ise"])) if len(matched) else float("nan")
            )
            body.append(
                " & ".join(
                    [
                        label if first else "",
                        SHORT_NAME[method],
                        f"{wins}/{attempted}",
                        fmt(conditional, 3),
                        fmt(matched_median, 3),
                        fmt(worst[method], 3),
                    ]
                )
                + r" \\"
            )
            first = False
            audit.setdefault(scenario, {})[SHORT_NAME[method]] = {
                "attempted": attempted, "succeeded": wins,
                "conditional": conditional, "matched": matched_median,
                "failure_scored": worst[method],
            }
        audit.setdefault(scenario, {})["shared_replications"] = len(shared)
        body.append(
            r"\multicolumn{6}{@{}l}{\footnotesize Replications completed by every "
            rf"method: {len(shared)}.}} \\"
        )
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    table = _table(
        caption=(
            "Three summaries of the same fits, under three policies for "
            "computational failure. ``Conditional'' is the median aggregated "
            "component-surface error over the replications a method completed, "
            "which is the summary the main tables report and which each method "
            "computes on its own subset. ``Matched'' restricts every method to "
            "the replications that all of them completed, so that the comparison "
            "is on common data. ``Failure scored'' returns a failed fit to the "
            "denominator at the zero-estimate penalty already used for a component "
            "a method does not deliver. A method that completes every attempt has "
            "the same entry three times."
        ),
        label="tab:supp-failure-policy",
        column_spec="llcccc",
        header=[
            r"Setting & Method & Fits & Conditional & Matched & Failure scored \\",
        ],
        body=body,
    )
    (tables / "failure_policy.tex").write_text(table, encoding="utf-8")
    return audit


# ---------------------------------------------------------------------------
# 6. Where the contamination enters
# ---------------------------------------------------------------------------

SCOPE_ROWS = (
    ("scope-clean-n100", "Clean", "exchangeable"),
    ("scope-response-n100", "Response, per visit", "exchangeable"),
    ("scope-subject-n100", "Response, per subject", "exchangeable"),
    ("scope-trajectory-n100", "Response, per trajectory", "exchangeable"),
    ("scope-leverage-n100", "Covariates (leverage)", "exchangeable"),
    ("scope-informative-clean-n100", "Clean", "informative"),
    ("scope-informative-subject-n100", "Response, per subject", "informative"),
)
SCOPE_METHODS = (
    "TRACE-VCAM",
    "ZZW2020",
    "HHY2021-Huber",
    "ZY2025-paper-implementation",
)


def scope_outputs(path: Path, tables: Path, figures: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame.applicability == "applicable"]
    ok = frame[frame.attempt_status == "success"]

    def median(scenario: str, method: str, column: str) -> float:
        block = ok[(ok.scenario == scenario) & (ok.method == method)][column]
        return float(np.nanmedian(block)) if len(block) else float("nan")

    def completion(scenario: str, method: str) -> tuple[int, int]:
        block = frame[(frame.scenario == scenario) & (frame.method == method)]
        return (
            int((block.attempt_status == "success").sum()),
            int(len(block)),
        )

    body: list[str] = []
    audit: dict[str, object] = {}
    previous_cluster = None
    for scenario, label, cluster in SCOPE_ROWS:
        if previous_cluster is not None and cluster != previous_cluster:
            body.append(r"\addlinespace")
            body.append(
                r"\multicolumn{9}{@{}l}{\itshape Cluster size depends on the "
                r"covariate level and the latent trajectory} \\"
            )
        previous_cluster = cluster
        cells = [label]
        for method in SCOPE_METHODS:
            cells.append(fmt(median(scenario, method, "component_ise"), 3))
            cells.append(fmt(median(scenario, method, "noise_free_test_mspe"), 3))
            wins, attempted = completion(scenario, method)
            audit.setdefault(scenario, {})[SHORT_NAME[method]] = {
                "component_ise": median(scenario, method, "component_ise"),
                "mspe": median(scenario, method, "noise_free_test_mspe"),
                "fits": f"{wins}/{attempted}",
            }
        body.append(" & ".join(cells) + r" \\")

    replications = int(
        frame.groupby(["scenario", "method"]).size().max()
    )
    table = _table(
        caption=(
            "Where the contamination enters.  The sparse-longitudinal design of "
            "Example 2 is held fixed and only the contamination channel and the "
            "cluster-size mechanism move; every contaminated setting perturbs "
            r"\(5\%\) of its unit.  Entries are medians over "
            f"{replications}"
            " replications at $n=100$ of the aggregated component-surface error "
            "and of the held-out noise-free prediction error, over the "
            "replications each method completed.  The first four rows contaminate "
            "the response at three different granularities and then the "
            "covariates; the last two make the number of visits depend on the "
            "subject's covariate level and latent trajectory."
        ),
        label="tab:supp-scope",
        column_spec="l" + "cc" * 4,
        column_sep="3pt",
        header=[
            " & "
            + " & ".join(
                rf"\multicolumn{{2}}{{c}}{{{SHORT_NAME[m]}}}" for m in SCOPE_METHODS
            )
            + r" \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            "Contamination & "
            + " & ".join([r"$\sum_k g_k$ & MSPE"] * 4)
            + r" \\",
        ],
        body=body,
    )
    (tables / "scope_robustness.tex").write_text(table, encoding="utf-8")

    # ---- figure: the price of each channel, per method ---------------------
    channels = [
        ("scope-response-n100", "Response,\nper visit"),
        ("scope-subject-n100", "Response,\nper subject"),
        ("scope-trajectory-n100", "Response,\nper trajectory"),
        ("scope-leverage-n100", "Covariates\n(leverage)"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_IN, 2.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    axis = axes[0]
    width = 0.2
    positions = np.arange(len(channels))
    for index, method in enumerate(SCOPE_METHODS):
        clean = median("scope-clean-n100", method, "component_ise")
        ratios = [
            median(scenario, method, "component_ise") / clean
            if np.isfinite(clean) and clean > 0
            else np.nan
            for scenario, _ in channels
        ]
        axis.bar(
            positions + (index - 1.5) * width, ratios, width * 0.92,
            color=METHOD_COLOR[method], label=SHORT_NAME[method],
            linewidth=0, zorder=2,
        )
        audit.setdefault("price", {})[SHORT_NAME[method]] = {
            label.replace("\n", " "): float(value)
            for (_scenario, label), value in zip(channels, ratios)
        }
    axis.axhline(1.0, color="0.4", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in channels], fontsize=7.5)
    axis.set_title(panel("a", "Price of contamination"), pad=3)
    axis.set_ylabel("Error relative to own clean fit", labelpad=2)

    # The second panel reports levels rather than ratios.  A ratio against a
    # method's own exchangeable fit rewards a method whose exchangeable fit was
    # already bad, which is the opposite of what the panel is meant to show.
    axis = axes[1]
    settings = (
        ("scope-informative-clean-n100", "Clean"),
        ("scope-informative-subject-n100", "Contaminated"),
    )
    positions = np.arange(len(settings))
    for index, method in enumerate(SCOPE_METHODS):
        levels = [median(scenario, method, "component_ise") for scenario, _ in settings]
        axis.bar(
            positions + (index - 1.5) * width, levels, width * 0.92,
            color=METHOD_COLOR[method], linewidth=0, zorder=2,
        )
        audit.setdefault("informative_level", {})[SHORT_NAME[method]] = {
            label: float(value) for (_s, label), value in zip(settings, levels)
        }
        audit.setdefault("informative_price", {})[SHORT_NAME[method]] = {
            label: float(
                median(informative, method, "component_ise")
                / median(exchangeable, method, "component_ise")
            )
            for exchangeable, informative, label in (
                ("scope-clean-n100", "scope-informative-clean-n100", "Clean"),
                ("scope-subject-n100", "scope-informative-subject-n100", "Contaminated"),
            )
        }
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in settings], fontsize=7.5)
    axis.set_title(panel("b", "Informative cluster size"), pad=3)
    axis.set_ylabel(r"$\sum_k\mathrm{ISE}(\widehat g_k)$", labelpad=2)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, frameon=False,
        fontsize=7.5, handlelength=1.4, columnspacing=1.8,
    )
    fig.savefig(figures / "scope_robustness.pdf")
    plt.close(fig)
    return audit


# ---------------------------------------------------------------------------
# 7. Around the block-sparse configuration
# ---------------------------------------------------------------------------

SWEEP_AXIS_LABEL = {
    "sparsity": "Active blocks",
    "correlation": "Cross-block correlation",
    "signal": "Signal scale",
}
SWEEP_VALUE_LABEL = {
    "s1": "1", "s2": "2 (registered)", "s3": "3", "s4": "4", "s6": "6",
    "rho0": "0", "rho0.3": "0.3", "rho0.5": "0.5 (registered)",
    "rho0.7": "0.7", "rho0.9": "0.9",
    "a0.25": "0.25", "a0.5": "0.5", "a1": "1 (registered)", "a2": "2",
}


def block_sensitivity_outputs(path: Path, tables: Path, figures: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame.applicability == "applicable"]
    ok = frame[frame.attempt_status == "success"]
    methods = ("TRACE-VCAM", "ZY2025-paper-implementation")

    body: list[str] = []
    audit: dict[str, object] = {}
    for axis_name in ("sparsity", "correlation", "signal"):
        block = frame[frame.sweep_axis == axis_name]
        if block.empty:
            continue
        labels = sorted(
            block.sweep_label.unique(),
            key=lambda item: float(item.lstrip("srhoa")) if item.lstrip("srhoa") else 0.0,
        )
        first = True
        for label in labels:
            cells = [SWEEP_AXIS_LABEL[axis_name] if first else "",
                     SWEEP_VALUE_LABEL.get(label, label)]
            first = False
            for method in methods:
                rows = ok[(ok.sweep_label == label) & (ok.sweep_axis == axis_name)
                          & (ok.method == method)]
                attempted = frame[(frame.sweep_label == label)
                                  & (frame.sweep_axis == axis_name)
                                  & (frame.method == method)]
                cells.append(fmt(float(np.nanmedian(rows["component_ise"])) if len(rows) else np.nan, 3))
                cells.append(fmt(float(np.nanmedian(rows["noise_free_test_mspe"])) if len(rows) else np.nan, 3))
                cells.append(f"{len(rows)}/{len(attempted)}")
                audit.setdefault(f"{axis_name}/{label}", {})[SHORT_NAME[method]] = {
                    "component_ise": float(np.nanmedian(rows["component_ise"])) if len(rows) else None,
                    "mspe": float(np.nanmedian(rows["noise_free_test_mspe"])) if len(rows) else None,
                    "fits": f"{len(rows)}/{len(attempted)}",
                }
            body.append(" & ".join(cells) + r" \\")
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    table = _table(
        caption=(
            "Around the block-sparse configuration.  Each axis moves one feature "
            "of the Example 3 design away from its registered value, holding the "
            "rest fixed, under the contaminated mixture at $n=100$ and $p=6$. "
            "Entries are medians over the replications each method completed, of "
            "the aggregated component-surface error and the held-out prediction "
            "error, with fits returned over fits attempted. The comparison is "
            "against the coefficientwise penalised estimator, which is the "
            "estimator the blockwise-versus-coefficientwise contrast is about; "
            "the full four-method comparison at the registered point is in the "
            "main paper."
        ),
        label="tab:supp-block-sensitivity",
        column_spec="ll" + "ccc" * 2,
        column_sep="4pt",
        header=[
            r" & & \multicolumn{3}{c}{TRACE} & \multicolumn{3}{c}{ZY} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"Axis & Value & $\sum_k g_k$ & MSPE & fits & "
            r"$\sum_k g_k$ & MSPE & fits \\",
        ],
        body=body,
    )
    (tables / "block_sensitivity.tex").write_text(table, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=ROOT / "results" / "review_experiments")
    parser.add_argument("--strict", type=Path,
                        default=ROOT / "results" / "strict_formal_v2_repaired" / "strict_results.csv")
    parser.add_argument("--block-sparse", type=Path,
                        default=ROOT / "results" / "block_sparse" / "robust_results.csv")
    parser.add_argument("--scope", type=Path,
                        default=ROOT / "results" / "robustness_scope" / "scope_results.csv")
    parser.add_argument("--sweep", type=Path,
                        default=ROOT / "results" / "block_sensitivity"
                        / "block_sensitivity_results.csv")
    parser.add_argument("--tables", type=Path, default=ROOT / "manuscript" / "tables")
    parser.add_argument("--figures", type=Path, default=ROOT / "manuscript" / "figures")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    import json

    audit: dict[str, object] = {}
    wanted = None if not args.only else set(args.only)

    def run(name: str, function, *positional):
        if wanted is not None and name not in wanted:
            return
        source = positional[0]
        if isinstance(source, Path) and not source.exists():
            print(f"skipping {name}: {source} is missing")
            return
        if function is tuning_outputs:
            audit[name] = function(
                *positional, args.strict, args.block_sparse, args.tables
            )
        else:
            audit[name] = function(*positional, args.tables, args.figures)

    run("normalisation", normalisation_outputs, args.review / "normalisation_margin.csv")
    run("selection", selection_outputs, args.review / "selection_path.csv")
    run("tuning", tuning_outputs, args.review / "tuning_sensitivity.csv")
    run("rank", rank_outputs, args.review / "rank_diagnostic.csv")
    run("scope", scope_outputs, args.scope)
    run("sweep", block_sensitivity_outputs, args.sweep)
    if wanted is None or "failure" in wanted:
        audit["failure"] = failure_outputs(args.strict, args.block_sparse, args.tables)

    print(json.dumps(audit, indent=2, default=float))


if __name__ == "__main__":
    main()

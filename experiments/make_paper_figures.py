"""Generate paper figures for the gastric robustness study from the final sweep CSVs.

Reads results/gastric/chemo_robust_robustness_summary_final_*_sweep.csv and writes
PDF + PNG to results/figures/:
  fig_headline        worst-case (bar) + mean (marker) joint feasibility, 4 methods x 2 modes
  fig_tradeoff        OS vs worst-case feasibility (Pareto), 4 methods x 2 modes
  fig_rhs_frontier    worst-case feasibility & OS vs toxicity bound (frac=0.5)
  fig_frac_frontier   worst-case feasibility & OS vs data fraction (rhs=0.6)

Feasibility = joint toxicity-constraint satisfaction (the ``all_constraints``
conjunction row); OS = Overall_Survival mean (months). Palette = Okabe-Ito
(colorblind-safe); each method also has a distinct marker (identity never
color-alone). CP is the contribution.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results/gastric"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

METHODS = ["nominal", "robust_reg", "wrapper", "cp"]
LABEL = {"nominal": "Nominal", "robust_reg": "Robust Reg.", "wrapper": "Wrapper", "cp": "CP (ours)"}
COLOR = {"nominal": "#595959", "robust_reg": "#E69F00", "wrapper": "#009E73", "cp": "#0072B2"}
MARKER = {"nominal": "o", "robust_reg": "s", "wrapper": "^", "cp": "D"}
MODES = ["all_constraints", "dlt_only"]
MODE_TITLE = {"all_constraints": "All toxicity constraints", "dlt_only": "DLT only"}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _load(tag):
    return pd.read_csv(f"{RES}/chemo_robust_robustness_summary_{tag}_sweep.csv")


def _feas(df, mode):
    return df[(df.outcome == "all_constraints") & (df.constraint_mode == mode)]


def _os(df, mode):
    return df[(df.outcome == "Overall_Survival") & (df.constraint_mode == mode)]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf (+.png)")


# ---------------------------------------------------------------- headline ---
def fig_headline():
    conf = _load("final_confirm")
    wrap = _load("final_wrapper")
    allm = pd.concat([conf, wrap], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, mode in zip(axes, MODES):
        f = _feas(allm, mode).set_index("method")
        xs = np.arange(len(METHODS))
        for i, m in enumerate(METHODS):
            worst = f.loc[m, "worst_case"]; mean = f.loc[m, "prescribed_mean"]
            ax.bar(i, worst, width=0.62, color=COLOR[m], zorder=3)
            ax.plot(i, mean, marker=MARKER[m], color=COLOR[m], mec="white", mew=1.2,
                    ms=10, zorder=4, linestyle="none")
            ax.vlines(i, worst, mean, color=COLOR[m], lw=1.2, alpha=0.6, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=20, ha="right")
        ax.set_title(MODE_TITLE[mode]); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Joint feasibility across draws\n(bar = worst-case, marker = mean)")
    fig.suptitle("Joint toxicity feasibility at the reference threshold (rhs=0.6, frac=0.5)", y=1.02, fontsize=12)
    _save(fig, "fig_headline")


# --------------------------------------------------------------- tradeoff ---
def fig_tradeoff():
    conf = _load("final_confirm"); wrap = _load("final_wrapper")
    allm = pd.concat([conf, wrap], ignore_index=True)
    # per-method label offsets (points) to avoid collisions when markers are close
    OFF = {"nominal": (-8, -15), "robust_reg": (8, 7), "wrapper": (9, 4), "cp": (9, 4)}
    HA = {"nominal": "right", "robust_reg": "left", "wrapper": "left", "cp": "left"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, mode in zip(axes, MODES):
        f = _feas(allm, mode).set_index("method"); o = _os(allm, mode).set_index("method")
        for m in METHODS:
            ax.scatter(f.loc[m, "worst_case"], o.loc[m, "prescribed_mean"],
                       s=130, color=COLOR[m], marker=MARKER[m], edgecolor="white",
                       linewidth=1.3, zorder=3)
            ax.annotate(LABEL[m], (f.loc[m, "worst_case"], o.loc[m, "prescribed_mean"]),
                        textcoords="offset points", xytext=OFF[m], ha=HA[m],
                        fontsize=9.5, color=COLOR[m], zorder=4)
        ax.set_title(MODE_TITLE[mode]); ax.set_xlabel("Worst-case joint feasibility")
        ax.margins(0.24)
    axes[0].set_ylabel("Overall survival (months)")
    fig.suptitle("Robustness–survival trade-off (rhs=0.6, frac=0.5): right = more robust, up = "
                 "higher survival\nCP buys large tail-robustness gains for a small survival cost",
                 y=1.05, fontsize=11)
    _save(fig, "fig_tradeoff")


# --------------------------------------------------------------- frontier ---
def _frontier(tag, xcol, xlabel, fname, title, fixed=None):
    df = _load(tag)
    if fixed:
        col, val = fixed
        df = df[np.isclose(pd.to_numeric(df[col], errors="coerce"), val)]
    methods = [m for m in ["nominal", "robust_reg", "cp"] if m in df.method.unique()]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7), sharex="col")
    for col, mode in enumerate(MODES):
        for row, (getter, ylab, ylim) in enumerate([
            (_feas, "Worst-case joint feasibility", (0, 1.02)),
            (_os, "Overall survival (months)", None)]):
            ax = axes[row][col]
            sub = getter(df, mode)
            for m in methods:
                s = sub[sub.method == m].copy()
                s[xcol] = pd.to_numeric(s[xcol], errors="coerce")
                s = s.sort_values(xcol)
                ax.plot(s[xcol], s["prescribed_mean" if getter is _os else "worst_case"],
                        marker=MARKER[m], color=COLOR[m], lw=2, ms=7, mec="white", mew=1,
                        label=LABEL[m], zorder=3)
            if ylim:
                ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(MODE_TITLE[mode])
            if row == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylab)
    axes[0][1].legend(loc="best", fontsize=9)
    fig.suptitle(title, y=1.01, fontsize=12)
    _save(fig, fname)


def main():
    print("Generating paper figures ->", OUT)
    fig_headline()
    fig_tradeoff()
    _frontier("final_rhs", "rhs", "Toxicity upper bound (percentile)", "fig_rhs_frontier",
              "Frontier vs constraint tightness (frac=0.5): CP protects the tail where nominal is fragile")
    _frontier("final_frac", "frac", "Fraction of training data", "fig_frac_frontier",
              "Frontier vs data scarcity (rhs=0.6): CP's edge grows as data gets scarcer, fades when plentiful",
              fixed=("rhs", 0.6))
    print("done.")


if __name__ == "__main__":
    main()

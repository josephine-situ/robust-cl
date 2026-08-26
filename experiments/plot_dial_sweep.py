"""THE primary figure: objective vs held-out feasibility, one panel per problem.

Reads ``results/rho_sweep/{problem}_dial_curve{cell}.csv`` from
``run_dial_sweep.py`` and writes to results/figures/:

  fig_dial_frontier_{problem}   every method, every dial value, every rho column
  fig_dial_cpalpha_{problem}    the coverage-cap ablation (``--cp-alpha-ablate``)

Why this figure and not another rho curve: the question the contribution has to
answer is "at equal held-out feasibility, whose decisions are better", and that is
a statement about two axes at once. A rho curve answers "how much assumed
uncertainty does each method absorb", which is about D -- interesting, but it is
the supporting reading now.

Three encoding decisions, each load-bearing:

- **Colour is the METHOD; the rho column is fill and linestyle.** A rho variation
  of a method is its own series (solid + filled for the larger rho, dashed + open
  for the smaller), but it keeps the method's colour, so the eye groups by method
  first. Colour never carries two things at once, and identity is never
  colour-alone -- every method also has its own marker, matching
  ``plot_rho_sweep.py`` and ``make_paper_figures.py``.
- **Marker size is the SOLVED FRACTION.** Without it the figure lies. Gastric
  margin at m=0.75 reads objective 8.78 at feasibility 1.000, which looks
  dominating until you see it is a 20% cohort -- the other 80% of contexts had no
  solution at all. A conditional-on-solved objective rewards a cell for solving
  few, so the thing that makes it conditional has to be on the same mark.
- **The Pareto direction follows the problem's own objective sense** (carried in
  the curve CSV, off the oracle), because gastric maximises survival and the
  reactor minimises cost. The arrow and the non-dominated set both flip with it;
  hard-coding "up and to the right" would silently invert one of the two problems.

Marked: the 0.9 feasibility rule (the vertical line every protocol point is read
against) and C-MICL's alpha=0.1, which is asserted rather than chosen.

Cells below ``--min-solved`` are drawn hollow and excluded from the Pareto set --
kept visible, because "this dial value renders most contexts unsolvable" is a
result, but never allowed to win a frontier.

Usage::

    python experiments/plot_dial_sweep.py --problem gastric --suffix _incoh
    python experiments/plot_dial_sweep.py --all --suffix _incoh
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = "results/rho_sweep"
OUT = "results/figures"

# Fixed hue order, never cycled -- the same assignment plot_rho_sweep.py and
# make_paper_figures.py use, so a method is one colour across every figure in the
# deck. Okabe-Ito; nominal is deliberately achromatic because it is the reference
# level rather than a categorical slot.
METHODS = ["nominal", "wrapper", "margin", "cmicl", "cp"]
LABEL = {"nominal": "Nominal", "wrapper": "Wrapper",
         "cmicl": r"C-MICL (no $\mathcal{D}$)",
         "margin": r"Tuned nominal, RHS margin (no $\mathcal{D}$)",
         "cp": "CP (ours)"}
# Shorter forms for the legend, which carries up to 7 series plus 4 keys and is
# laid out in columns UNDER the panel -- the full names above stay for prose.
SHORT = {"margin": r"RHS margin (no $\mathcal{D}$)"}
COLOR = {"nominal": "#595959", "wrapper": "#009E73", "cmicl": "#CC79A7",
         "margin": "#56B4E9", "cp": "#0072B2"}
MARKER = {"nominal": "*", "wrapper": "^", "cmicl": "v", "margin": "P", "cp": "D"}
DIAL_TEX = {"tau": r"$\tau$", "alpha": r"$\alpha$", "margin": r"$m$",
            "none": ""}

PTITLE = {"synthetic": "Synthetic", "gastric": "Gastric (OptiCL)",
          "reactor": "DMA-MR reactor (C-MICL)"}
OBJ_LABEL = {"synthetic": r"Objective $c^\top x^*$ (lower better)",
             "gastric": "Overall survival, months (higher better)",
             "reactor": "Operating cost (lower better)"}

# Marker area in pt^2 at solved_frac 0 and 1. The floor keeps a 20%-cohort cell
# visible rather than vanishing -- it has to be readable to be disbelieved.
SIZE_MIN, SIZE_MAX = 28.0, 190.0

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 13,
    "axes.titlesize": 13, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _load(problem, kind, suffix):
    path = f"{RES}/{problem}_{kind}{suffix}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def _sizes(solved):
    s = np.asarray(solved, dtype=float)
    s = np.where(np.isfinite(s), np.clip(s, 0.0, 1.0), 0.0)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * s


def _pareto(df, sense):
    """Indices of the non-dominated cells: no other cell is at least as good on
    BOTH axes and strictly better on one.

    "Better" on feasibility is always larger; on the objective it follows the
    problem's own sense, which is why ``sense`` is a parameter rather than a
    constant. O(n^2), and n here is a few dozen.
    """
    f = df["feasibility"].to_numpy(float)
    o = df["objective"].to_numpy(float)
    if sense == "min":
        o = -o                       # now "larger is better" on both axes
    keep = []
    for i in range(len(df)):
        if not (np.isfinite(f[i]) and np.isfinite(o[i])):
            continue
        dominated = ((f >= f[i]) & (o >= o[i]) &
                     ((f > f[i]) | (o > o[i])) &
                     np.isfinite(f) & np.isfinite(o))
        if not dominated.any():
            keep.append(i)
    return df.index[keep]


def _series_key(row):
    """(method, rho) -- one series. rho is NaN for the methods that face no D."""
    return (row["method"], row["rho"])


def frontier(problem, suffix, min_solved, target, out_name=None):
    df = _load(problem, "dial_curve", suffix)
    if df is None:
        print(f"  no {RES}/{problem}_dial_curve{suffix}.csv -- skipping")
        return
    main = df[df.get("phase", "dial") == "dial"]
    ref = df[df.get("phase", "dial") == "reference"]
    sense = str(df["objective_sense"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    # The rule every protocol point is read against, drawn before the data so it
    # sits behind it.
    ax.axvline(target, color="#C0392B", ls="--", lw=1.2, zorder=1)

    handles = []
    # nominal: a horizontal reference at its objective plus its own marker. It is
    # the level the whole panel is read against, so it gets a line, not just a dot.
    for _, r in ref.iterrows():
        ax.axhline(float(r["objective"]), color=COLOR["nominal"], ls=":", lw=1.1,
                   zorder=1)
        ax.scatter([r["feasibility"]], [r["objective"]],
                   s=_sizes([r["solved_frac"]]), marker=MARKER["nominal"],
                   c=COLOR["nominal"], edgecolors="white", linewidths=1.2,
                   zorder=5)
        handles.append(Line2D([], [], color=COLOR["nominal"], ls=":", lw=1.4,
                              marker=MARKER["nominal"], markersize=11,
                              label=LABEL["nominal"]))

    # Rho columns get fill + linestyle; the method keeps the colour.
    rhos = sorted({float(r) for r in main["rho"].dropna().unique()})
    fill_for = {r: (i == len(rhos) - 1) for i, r in enumerate(rhos)}
    ls_for = {r: ("-" if i == len(rhos) - 1 else "--") for i, r in enumerate(rhos)}

    for method in METHODS:
        g_m = main[main["method"] == method]
        if g_m.empty:
            continue
        for rho, g in g_m.groupby("rho", dropna=False):
            g = g.sort_values("dial")
            filled = True if not np.isfinite(rho) else fill_for[float(rho)]
            ls = "-" if not np.isfinite(rho) else ls_for[float(rho)]
            col = COLOR[method]
            # The path through dial order: it says which way the dial moves the
            # decision, which a bare cloud of points does not.
            ax.plot(g["feasibility"], g["objective"], ls=ls, lw=1.4, color=col,
                    alpha=0.55, zorder=2)
            ok = g["solved_frac"] >= min_solved
            for mask, face in ((ok, col if filled else "none"),
                               (~ok, "none")):
                sub = g[mask]
                if sub.empty:
                    continue
                ax.scatter(sub["feasibility"], sub["objective"],
                           s=_sizes(sub["solved_frac"]), marker=MARKER[method],
                           facecolors=face, edgecolors=col,
                           linewidths=1.6 if face == "none" else 1.0,
                           zorder=4)
            lab = SHORT.get(method, LABEL[method]) + (
                "" if not np.isfinite(rho) else rf",  $\rho={rho:g}$")
            handles.append(Line2D([], [], color=col, ls=ls, lw=1.6,
                                  marker=MARKER[method], markersize=8,
                                  markerfacecolor=col if filled else "none",
                                  markeredgecolor=col, label=lab))

    # C-MICL's protocol point: asserted, not chosen, so it is called out rather
    # than left as one dot among six. On gastric it is expected NOT to solve, and a
    # cell with no finite coordinates cannot be a dot at all -- which is exactly
    # when it most needs saying, so it goes into the legend instead of vanishing.
    cm = main[(main["method"] == "cmicl") &
              (main["note"].astype(str) == "protocol point")]
    for _, r in cm.iterrows():
        if np.isfinite(r["feasibility"]) and np.isfinite(r["objective"]):
            ax.annotate(rf"C-MICL protocol $\alpha={r['dial']:g}$",
                        (r["feasibility"], r["objective"]),
                        textcoords="offset points", xytext=(10, 10), fontsize=10,
                        color=COLOR["cmicl"],
                        arrowprops=dict(arrowstyle="-", color=COLOR["cmicl"],
                                        lw=1.0))
        else:
            handles.append(Line2D(
                [], [], ls="", marker="x", color=COLOR["cmicl"], markersize=8,
                label=rf"C-MICL protocol $\alpha={r['dial']:g}$: no solution"))

    # Every OTHER cell that produced nothing. A dial value at which the master is
    # infeasible on every fold is a result -- "C-MICL cannot be run at this level
    # on this instance" -- and an empty region of the plot does not say it.
    dead = main[~np.isfinite(main["feasibility"])]
    if not dead.empty:
        parts = []
        for method, g in dead.groupby("method"):
            vals = ", ".join(f"{v:g}" for v in sorted(g["dial"].unique()))
            parts.append(f"{LABEL[method].split(' (')[0]} "
                         f"{DIAL_TEX.get(str(g['dial_name'].iloc[0]), '')}"
                         f" = {vals}")
        ax.text(0.0, -0.155, "no solution on any fold: " + "; ".join(parts),
                transform=ax.transAxes, fontsize=9.5, color="#7A2E2E")

    # Cells the adaptive search never scored. A gap in a curve is otherwise
    # indistinguishable from a cell that produced nothing, and those are opposite
    # claims: one says "not measured", the other says "no solution exists here".
    # The dead cells above already have their own line, in red; this one is grey
    # and says how many were pruned on the rules (a feasibility of 0 below, an
    # unsolvable cell above) versus how many the eval budget simply did not
    # reach. Absent when the whole grid was walked.
    skip = _load(problem, "dial_skipped", suffix)
    if skip is not None and not skip.empty:
        why = skip["reason"].astype(str)
        n_pruned = int(why.str.startswith("pruned").sum())
        bits = []
        for method, g in skip.groupby("method"):
            vals = ", ".join(f"{v:g}" for v in sorted(g["dial"].unique()))
            bits.append(f"{LABEL.get(method, method).split(' (')[0]} = {vals}")
        ax.text(0.0, -0.195,
                f"not scored ({n_pruned} pruned on the search rules, "
                f"{len(skip) - n_pruned} outside the eval budget): "
                + "; ".join(bits),
                transform=ax.transAxes, fontsize=9.0, color="#6B6B6B")

    # The Pareto set, over the cells that clear the solved floor. Ringed rather
    # than recoloured, so a point keeps its method identity.
    elig = main[main["solved_frac"] >= min_solved]
    if not elig.empty:
        front = elig.loc[_pareto(elig, sense)].sort_values("feasibility")
        ax.scatter(front["feasibility"], front["objective"],
                   s=_sizes(front["solved_frac"]) + 90, marker="o",
                   facecolors="none", edgecolors="#4D4D4D", linewidths=0.9,
                   zorder=3)

    ax.set_xlabel("Held-out feasibility (fraction of contexts)")
    ax.set_ylabel(OBJ_LABEL.get(problem, "Objective"))
    better = "up" if sense == "max" else "down"
    ax.set_title(f"{PTITLE.get(problem, problem)}: objective vs feasibility "
                 f"(better = right and {better})")
    ax.set_xlim(-0.03, 1.05)

    # One legend, UNDER the panel. Anchoring it outside the axes on the right got
    # clipped once the series count grew, and there is no corner of this panel free
    # of data -- the whole figure is a frontier running across it.
    extra = [
        Line2D([], [], color="#C0392B", ls="--", lw=1.2,
               label=f"feasibility target {target:g}"),
        Line2D([], [], ls="", marker="o", markerfacecolor="none",
               markeredgecolor="#4D4D4D", markersize=9, label="Pareto-optimal"),
    ] + [
        # Marker size IS the solved fraction, so its key belongs in the same
        # legend as the series rather than in a box of its own.
        Line2D([], [], ls="", marker="o", markerfacecolor="#9A9A9A",
               markeredgecolor="#4D4D4D",
               markersize=np.sqrt(_sizes([v])[0]), label=f"solved {v:.0%}")
        for v in (0.25, 0.5, 1.0)
    ]
    fig.legend(handles=handles + extra, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=10)

    name = out_name or f"fig_dial_frontier_{problem}{suffix}"
    _save(fig, name)


def cp_alpha_panel(problem, suffix, min_solved, target):
    """The coverage-cap ablation: feasibility AND solved fraction against alpha.

    Two stacked panels, one measure each -- never two y-scales on one axis. The
    whole question is whether the second curve pays for the first, which a reader
    can only judge if both are on their own scale.
    """
    df = _load(problem, "dial_curve", suffix)
    if df is None:
        return
    ab = df[df.get("phase", "") == "cp_alpha_ablation"].sort_values("cp_alpha")
    if ab.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
    for ax, col, lab, rule in (
            (axes[0], "feasibility", "Held-out feasibility", target),
            (axes[1], "solved_frac", "Solved fraction", min_solved)):
        ax.plot(ab["cp_alpha"], ab[col], marker=MARKER["cp"], color=COLOR["cp"],
                lw=1.8, markersize=8)
        ax.axhline(rule, color="#C0392B", ls="--", lw=1.2)
        ax.set_ylabel(lab)
        ax.set_ylim(-0.03, 1.05)
    rho_a = float(ab["rho"].iloc[0])
    # note is "tau*=<value>" from run_dial_sweep; render the symbol properly.
    tau_star = str(ab["note"].iloc[0]).replace("tau*=", "")
    axes[1].set_xlabel(r"CP coverage cap $\alpha$ (fraction of anchors a cut may break)")
    axes[0].set_title(f"{PTITLE.get(problem, problem)}: coverage cap at "
                      rf"$\rho={rho_a:g}$, $\tau^\ast={tau_star}$")
    _save(fig, f"fig_dial_cpalpha_{problem}{suffix}")


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf/.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "reactor", "gastric"),
                   default="gastric")
    p.add_argument("--all", action="store_true", help="every problem with a curve")
    p.add_argument("--suffix", default="_incoh",
                   help="sweep CELL suffix, e.g. _incoh, _coh, _incoh_s7")
    p.add_argument("--min-solved", type=float, default=0.5)
    p.add_argument("--feas-target", type=float, default=0.9)
    args = p.parse_args()

    problems = (["synthetic", "reactor", "gastric"] if args.all
                else [args.problem])
    for prob in problems:
        frontier(prob, args.suffix, float(args.min_solved),
                 float(args.feas_target))
        cp_alpha_panel(prob, args.suffix, float(args.min_solved),
                       float(args.feas_target))


if __name__ == "__main__":
    main()

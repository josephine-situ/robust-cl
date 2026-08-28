"""THE primary figure: objective vs held-out feasibility, one panel per problem.

Reads ``results/rho_sweep/{problem}_dial_curve{cell}.csv`` from
``run_dial_sweep.py`` and writes to results/figures/:

  fig_dial_frontier_{problem}   every method, every dial value, every rho column
  fig_dial_solved_{problem}     the SOLVED FRACTION of those same cells
  fig_dial_cpalpha_{problem}    the coverage-cap ablation (``--cp-alpha-ablate``)

Why this figure and not another rho curve: the question the contribution has to
answer is "at equal held-out feasibility, whose decisions are better", and that is
a statement about two axes at once. A rho curve answers "how much assumed
uncertainty does each method absorb", which is about D -- interesting, but it is
the supporting reading now.

**These are TUNING scores, and the panel now says so.** Every point is a mean over
the sweep's own CV folds under the *proxy* judge that instance tunes against
(``cv_calibrate.make_cv_oracle``, fit on training rows only), and ``dial*`` is
picked from exactly these cells. The feasibility axis is held out *within* a fold
-- the fold-val contexts on gastric, the single fold decision on synthetic and the
reactor -- and is not the test stage. ``run_dial_test.py`` is what re-scores each
method at its own ``dial*`` under a judge the dial never faced (and on gastric
under a different cohort as well); a number off this figure is not that number.

Encoding decisions, each load-bearing:

- **Colour is the METHOD; the rho column is a SHADE of it plus a linestyle.** A
  rho variation of a method is its own series (dark + solid for the larger rho,
  light + dashed for the smaller) inside the method's own hue, so the eye groups
  by method first. The shade is relative to that METHOD's own columns, because
  methods need not share them -- on the reactor the wrapper runs at rho 5/6 where
  CP runs at 2/3, and reading the split off the panel's global max would draw
  both of CP's columns as the same light dashed series.
  The RHS margin used to be a second blue one shade off CP's, which put the
  contribution and the baseline it has to beat in one colour family; it is
  orange, and the only two series sharing a hue are now the two rho columns of a
  single method.
- **Marker SHAPE says whether the method faces D at all** -- a circle for the
  shared-uncertainty-set methods (CP, wrapper), a square for the two that face no
  D (RHS margin, C-MICL), a star for the nominal reference. Three shapes carrying
  one real distinction, instead of a shape per method carrying none.
- **A HOLLOW marker means the cell is below ``--min-solved``, and nothing else.**
  Fill has exactly one meaning. Those cells stay on the panel -- "this dial value
  renders most contexts unsolvable" is a result -- but are excluded from the
  Pareto set and labelled with their solved fraction, because a
  conditional-on-solved objective rewards a cell for solving few.
- **Solved fraction is its OWN panel** (``fig_dial_solved_*``), on the same x
  axis and the same styling, so a frontier point is read by dropping straight
  down. It used to be the marker AREA on the frontier itself, which gave every
  point a different size and cost the primary panel its legibility.
- **The Pareto direction follows the problem's own objective sense** (carried in
  the curve CSV, off the oracle), because gastric maximises survival and the
  reactor minimises cost. The arrow and the non-dominated set both flip with it;
  hard-coding "up and to the right" would silently invert one of the two problems.

Marked: the 0.9 feasibility rule (the vertical line every protocol point is read
against) and C-MICL's alpha=0.1, which is asserted rather than chosen.

Usage::

    python experiments/plot_dial_sweep.py --problem gastric --suffix _incoh
    python experiments/plot_dial_sweep.py --all --suffix _incoh
"""
import argparse
import os
import textwrap

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
# Shorter forms for the legend, which carries up to 7 series plus its keys and is
# laid out in columns UNDER the panel -- the full names above stay for prose.
SHORT = {"margin": r"RHS margin (no $\mathcal{D}$)"}
COLOR = {"nominal": "#595959", "wrapper": "#009E73", "cmicl": "#CC79A7",
         "margin": "#E69F00", "cp": "#0072B2"}
# Shape = does this method face the shared uncertainty set D.
MARKER = {"nominal": "*", "wrapper": "o", "cmicl": "s", "margin": "s", "cp": "o"}
MSIZE = 8.0            # one marker size for every cell; see the module docstring
DIAL_TEX = {"tau": r"$\tau$", "alpha": r"$\alpha$", "margin": r"$m$",
            "none": ""}

PTITLE = {"synthetic": "Synthetic", "gastric": "Gastric (OptiCL)",
          "reactor": "DMA-MR reactor (C-MICL)"}
OBJ_LABEL = {"synthetic": r"Objective $c^\top x^*$ (lower better)",
             "gastric": "Overall survival, months (higher better)",
             "reactor": "Operating cost (lower better)"}

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


def _lighten(hex_color, frac=0.55):
    """Blend towards white. The SMALLER rho column of a method is drawn in this,
    so its two columns stay one hue family and the method is read first."""
    c = hex_color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(v + (1.0 - v) * frac for v in rgb)


def _n_folds(problem, suffix):
    """Fold count, off the per-context records. Used only to caption what the
    numbers are; a missing file just drops the count from the caption."""
    ctx = _load(problem, "dial_contexts", suffix)
    if ctx is None or ctx.empty or "fold" not in ctx.columns:
        return None
    return int(ctx["fold"].nunique())


def _subtitle(problem, suffix):
    """One line naming which folds and which judge produced every point here.

    The failure mode this guards against is quoting a tuned cell's own tuning
    score as the result, which is what run_dial_test.py exists to avoid.
    """
    nf = _n_folds(problem, suffix)
    folds = f"{nf} CV folds" if nf else "the sweep's CV folds"
    unit = ("held-out contexts within each fold" if problem == "gastric"
            else "one held-out decision per fold")
    return (f"TUNING scores: {folds}, {unit}, train-only proxy judge. "
            r"$\mathrm{dial}^\ast$ is chosen on these same cells; "
            "the test stage is run_dial_test.py.")


def _series(main):
    """The (method, rho) groups in fixed method order, with the style each gets.

    Shared by the frontier and the solved-fraction panel, so the two are read
    against each other without a second legend to reconcile.
    """
    out = []
    for method in METHODS:
        g_m = main[main["method"] == method]
        if g_m.empty:
            continue
        # Dark+solid is the method's OWN larger column, not the panel's largest
        # rho. Methods may sit on different columns (`METHOD_RHO_COLUMNS`: the
        # wrapper runs at rho 5/6 on the reactor where CP runs at 2/3), and a
        # global max would then render BOTH of CP's columns as the light dashed
        # one -- two distinct series drawn identically.
        own = sorted({float(r) for r in g_m["rho"].dropna().unique()})
        for rho, g in g_m.groupby("rho", dropna=False):
            big = (not np.isfinite(rho)) or float(rho) == own[-1]
            col = COLOR[method] if big else _lighten(COLOR[method])
            ls = "-" if big else "--"
            lab = SHORT.get(method, LABEL[method]) + (
                "" if not np.isfinite(rho) else rf",  $\rho={rho:g}$")
            out.append((method, g.sort_values("dial"), col, ls, lab))
    return out


def _draw_series(ax, method, g, col, ls, ycol, min_solved):
    """One series: the dial path, then filled/hollow markers on it."""
    ax.plot(g["feasibility"], g[ycol], ls=ls, lw=1.3, color=col, alpha=0.55,
            zorder=2)
    ok = g["solved_frac"] >= min_solved
    for mask, face in ((ok, col), (~ok, "none")):
        sub = g[mask]
        if sub.empty:
            continue
        ax.plot(sub["feasibility"], sub[ycol], ls="", marker=MARKER[method],
                markersize=MSIZE, markerfacecolor=face, markeredgecolor=col,
                markeredgewidth=1.5 if face == "none" else 1.0, zorder=4)


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


def _dead_footer(main, compact):
    """Cells that produced no decision on any fold.

    A dial value at which the master is infeasible everywhere is a result --
    "C-MICL cannot be run at this level on this instance" -- and an empty region
    of the panel does not say it.
    """
    dead = main[~np.isfinite(main["feasibility"])]
    if dead.empty:
        return None
    parts = []
    for method, g in dead.groupby("method"):
        vals = ", ".join(f"{v:g}" for v in sorted(g["dial"].unique()))
        parts.append(f"{LABEL[method].split(' (')[0]} "
                     f"{DIAL_TEX.get(str(g['dial_name'].iloc[0]), '')} = {vals}")
    body = ("; ".join(sorted(LABEL[k].split(" (")[0]
                             for k in dead["method"].unique()))
            if compact else "; ".join(parts))
    return ("no solution on any fold (not plotted): " + body, "#7A2E2E", 9.5)


def _skipped_footer(problem, suffix, compact):
    """Cells the adaptive search never scored.

    A gap in a curve is otherwise indistinguishable from a cell that produced
    nothing, and those are opposite claims: one says "not measured", the other
    "no solution exists here". The dead cells above get their own red line; this
    one is grey and separates cells pruned on the search rules from cells the
    eval budget simply did not reach. Absent when the whole grid was walked.
    """
    skip = _load(problem, "dial_skipped", suffix)
    if skip is None or skip.empty:
        return None
    n_pruned = int(skip["reason"].astype(str).str.startswith("pruned").sum())
    bits = []
    for method, g in skip.groupby("method"):
        vals = ", ".join(f"{v:g}" for v in sorted(g["dial"].unique()))
        bits.append(f"{LABEL.get(method, method).split(' (')[0]} = {vals}")
    # COMPACT drops the per-method dial lists, which run to three wrapped lines
    # and shrink the panel to under half the figure. The counts stay -- "some
    # cells were not scored" is the part a reader must not miss -- and the dial
    # values live in {problem}_dial_skipped{cell}.csv, which the caption names.
    detail = ("; see *_dial_skipped*.csv" if compact else ": " + "; ".join(bits))
    return (f"not scored ({n_pruned} pruned on the search rules, "
            f"{len(skip) - n_pruned} outside the eval budget){detail}",
            "#6B6B6B", 9.0)


def _layout(fig, ax, footers, handles, ncol):
    """Footnote lines under the axes, then the legend under those.

    Both are laid out relative to the axes rather than at fixed figure
    coordinates: the legend's anchor is a function of how many footnote lines
    there turned out to be, because a constant that happened to fit once put the
    two on top of each other.
    """
    y = -0.155                  # clears the tick labels and the x-axis label
    for text, color, size in footers:
        for line in textwrap.wrap(text, width=100) or [""]:
            y -= 0.040
            fig.text(0.02, y, line, transform=ax.transAxes, fontsize=size,
                     color=color, va="top")
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, y - 0.03), bbox_transform=ax.transAxes,
               ncol=ncol, fontsize=10)


def _set_xlim(ax, main, xlim):
    """x-span. FIXED (the default) keeps the full 0-1 fraction, so panels from
    different problems are read against the same axis -- and on the reactor the
    wrapper genuinely sits at 0.0, so the span is the data. AUTO tightens to the
    data with padding, for a problem whose whole frontier lives in a corner of
    it: every gastric cell scores between 0.85 and 1.00, and on the fixed span
    that is the right sixth of the panel with the differences invisible. It
    changes no number, only the magnification, so the 0.9 rule stays drawn.
    """
    f = main["feasibility"].to_numpy(float)
    f = f[np.isfinite(f)]
    if xlim == "auto" and f.size:
        lo, hi = float(f.min()), float(f.max())
        pad = max(0.02, 0.08 * (hi - lo))
        ax.set_xlim(max(-0.03, lo - pad), min(1.05, hi + pad))
    else:
        ax.set_xlim(-0.03, 1.05)


def frontier(problem, suffix, min_solved, target, out_name=None, xlim="fixed",
             compact=False):
    df = _load(problem, "dial_curve", suffix)
    if df is None:
        print(f"  no {RES}/{problem}_dial_curve{suffix}.csv -- skipping")
        return
    main = df[df.get("phase", "dial") == "dial"]
    ref = df[df.get("phase", "dial") == "reference"]
    sense = str(df["objective_sense"].iloc[0])

    # COMPACT is the deck aspect: wider and shorter, so that once the condensed
    # footnotes and the 4-column legend are stacked underneath, the PANEL is
    # still most of the height. At the report aspect a slide renders the plot at
    # about 45% of the figure.
    fig, ax = plt.subplots(figsize=(10.0, 5.2) if compact else (8.2, 5.8))

    # The rule every protocol point is read against, drawn before the data so it
    # sits behind it.
    ax.axvline(target, color="#C0392B", ls="--", lw=1.2, zorder=1)

    handles = []
    # nominal: just a marker.
    for _, r in ref.iterrows():
        ax.plot([r["feasibility"]], [r["objective"]], ls="",
                marker=MARKER["nominal"], markersize=15,
                markerfacecolor=COLOR["nominal"], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        handles.append(Line2D([], [], color=COLOR["nominal"], ls=":", lw=1.4,
                              marker=MARKER["nominal"], markersize=12,
                              label=LABEL["nominal"]))

    below = []                  # cells under the solved floor, labelled in place
    for method, g, col, ls, lab in _series(main):
        _draw_series(ax, method, g, col, ls, "objective", min_solved)
        for _, r in g[g["solved_frac"] < min_solved].iterrows():
            if np.isfinite(r["feasibility"]) and np.isfinite(r["objective"]):
                below.append((r["feasibility"], r["objective"],
                              float(r["solved_frac"]), col))
        handles.append(Line2D([], [], color=col, ls=ls, lw=1.6,
                              marker=MARKER[method], markersize=8,
                              markerfacecolor=col, markeredgecolor=col,
                              label=lab))

    # A hollow marker says "most contexts had no decision here"; the number says
    # how few. Cheap while there are a handful -- past that the reader is better
    # served by the companion panel than by a thicket of labels.
    if len(below) <= 6:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.07 * (hi - lo), hi)
        for x, y, frac, col in below:
            ax.annotate(f"{frac:.0%} solved", (x, y),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=8.5, color=col)

    # C-MICL's protocol point: asserted, not chosen, so it is called out rather
    # than left as one dot among six. On gastric it is expected NOT to solve, and
    # a cell with no finite coordinates cannot be a dot at all -- which is exactly
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

    # The Pareto set, over the cells that clear the solved floor. Ringed rather
    # than recoloured, so a point keeps its method identity.
    elig = main[main["solved_frac"] >= min_solved]
    if not elig.empty:
        front = elig.loc[_pareto(elig, sense)].sort_values("feasibility")
        ax.plot(front["feasibility"], front["objective"], ls="", marker="o",
                markersize=MSIZE + 7, markerfacecolor="none",
                markeredgecolor="#4D4D4D", markeredgewidth=0.9, zorder=3)

    ax.set_xlabel("Held-out feasibility (fraction of contexts)")
    ax.set_ylabel(OBJ_LABEL.get(problem, "Objective"))
    better = "up" if sense == "max" else "down"
    ax.set_title(f"{PTITLE.get(problem, problem)}: objective vs feasibility "
                 f"(better = right and {better})", pad=24)
    ax.text(0.0, 1.015, _subtitle(problem, suffix), transform=ax.transAxes,
            fontsize=9, color="#555555", va="bottom")
    _set_xlim(ax, main, xlim)

    footers = [f for f in (_dead_footer(main, compact),
                           _skipped_footer(problem, suffix, compact)) if f]
    extra = [
        Line2D([], [], color="#C0392B", ls="--", lw=1.2,
               label=f"feasibility target {target:g}"),
        Line2D([], [], ls="", marker="o", markerfacecolor="none",
               markeredgecolor="#4D4D4D", markersize=11, label="Pareto-optimal"),
        # Fill carries exactly one thing, and this is it.
        Line2D([], [], ls="", marker="o", markerfacecolor="none",
               markeredgecolor="#9A9A9A", markeredgewidth=1.5, markersize=8,
               label=f"hollow: solved < {min_solved:g}"),
    ]
    _layout(fig, ax, footers, handles + extra, 4 if compact else 3)
    _save(fig, out_name or f"fig_dial_frontier_{problem}{suffix}")


def solved_panel(problem, suffix, min_solved, target, out_name=None,
                 xlim="fixed", compact=False):
    """Solved fraction of the same cells, on the same x axis as the frontier.

    The conditional-on-solved objective in the frontier is only readable beside
    this: a cell that scores feasibility 1.00 on 14% of contexts and one that
    scores it on 100% are the same dot up there and are not the same result.
    Same colours, shapes and linestyles, so no second legend has to be
    reconciled with the first.
    """
    df = _load(problem, "dial_curve", suffix)
    if df is None:
        return
    main = df[df.get("phase", "dial") == "dial"]
    ref = df[df.get("phase", "dial") == "reference"]

    fig, ax = plt.subplots(figsize=(10.0, 4.4) if compact else (8.2, 4.8))
    ax.axvline(target, color="#C0392B", ls="--", lw=1.2, zorder=1)
    ax.axhline(min_solved, color="#7A2E2E", ls=":", lw=1.2, zorder=1)

    handles = []
    for _, r in ref.iterrows():
        ax.plot([r["feasibility"]], [r["solved_frac"]], ls="",
                marker=MARKER["nominal"], markersize=15,
                markerfacecolor=COLOR["nominal"], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        handles.append(Line2D([], [], color=COLOR["nominal"], ls="",
                              marker=MARKER["nominal"], markersize=12,
                              label=LABEL["nominal"]))

    for method, g, col, ls, lab in _series(main):
        _draw_series(ax, method, g, col, ls, "solved_frac", min_solved)
        handles.append(Line2D([], [], color=col, ls=ls, lw=1.6,
                              marker=MARKER[method], markersize=8,
                              markerfacecolor=col, markeredgecolor=col,
                              label=lab))

    ax.set_xlabel("Held-out feasibility (fraction of contexts)")
    ax.set_ylabel("Solved fraction (contexts with a decision)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"{PTITLE.get(problem, problem)}: how much of the cohort each "
                 f"cell could prescribe for", pad=24)
    ax.text(0.0, 1.015,
            "Same cells as the frontier panel. Feasibility and objective there "
            "are CONDITIONAL on this fraction.",
            transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")
    _set_xlim(ax, main, xlim)

    footers = [f for f in (_dead_footer(main, compact),) if f]
    extra = [
        Line2D([], [], color="#C0392B", ls="--", lw=1.2,
               label=f"feasibility target {target:g}"),
        Line2D([], [], color="#7A2E2E", ls=":", lw=1.2,
               label=f"solved floor {min_solved:g}"),
    ]
    _layout(fig, ax, footers, handles + extra, 4 if compact else 3)
    _save(fig, out_name or f"fig_dial_solved_{problem}{suffix}")


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
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png")


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
    p.add_argument("--compact", action="store_true",
                   help="deck aspect: wider panel, condensed footnotes, "
                        "4-column legend. Writes <name>_slide.")
    p.add_argument("--xlim", choices=("fixed", "auto"), default="fixed",
                   help="fixed: the whole 0-1 feasibility span, comparable "
                        "across problems. auto: tighten to the data, for a "
                        "frontier that lives in one corner of it (gastric).")
    args = p.parse_args()

    problems = (["synthetic", "reactor", "gastric"] if args.all
                else [args.problem])
    for prob in problems:
        tag = "_slide" if args.compact else ""
        frontier(prob, args.suffix, float(args.min_solved),
                 float(args.feas_target), xlim=args.xlim, compact=args.compact,
                 out_name=f"fig_dial_frontier_{prob}{args.suffix}{tag}")
        solved_panel(prob, args.suffix, float(args.min_solved),
                     float(args.feas_target), xlim=args.xlim,
                     compact=args.compact,
                     out_name=f"fig_dial_solved_{prob}{args.suffix}{tag}")
        cp_alpha_panel(prob, args.suffix, float(args.min_solved),
                       float(args.feas_target))


if __name__ == "__main__":
    main()

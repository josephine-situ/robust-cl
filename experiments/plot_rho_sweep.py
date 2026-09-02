"""Figures for the shared-D rho sweep, from results/rho_sweep/*.csv.

Reads the curve / rho* / ablation CSVs written by ``run_rho_sweep.py`` for one
sweep CELL (``--suffix``, default ``_coh``) and writes PDF + PNG to
results/figures/:

  fig_rho_feasibility   held-out feasibility vs rho, 4 methods x 2 problems
  fig_rho_objective     objective vs rho (the price of robustness)
  fig_rho_solved        solved fraction vs rho -- the artefact guard
  fig_rho_time          gastric wall clock, master phase vs per-test-point phase
  fig_rho_ablation      tau (CP) and alpha (wrapper) frontiers at the chosen rho

The rho CURVE is the primary reading and rho* the derived one, so every figure
shows the whole axis; rho* is marked, never plotted alone. Palette = Okabe-Ito
(colorblind-safe) and each method also carries a distinct marker, matching
``plot_dial_sweep.py`` -- identity is never colour-alone.
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results/rho_sweep"
OUT = "results/figures"

# cmicl and margin are plotted only when the curve carries them (both are opt-in
# on the sweep). NEITHER faces D, so their lines are flat in rho by construction
# -- reference levels, not further readings of the axis; dashed, so they do not
# read as one.
METHODS = ["nominal", "robust_reg", "wrapper", "margin", "cmicl", "cp"]
LABEL = {"nominal": "Nominal", "robust_reg": "Robust Reg.",
         "wrapper": "Wrapper", "cmicl": r"C-MICL (no $\mathcal{D}$)",
         "margin": r"Tuned nominal, RHS margin (no $\mathcal{D}$)",
         "cp": "CP (ours)"}
COLOR = {"nominal": "#595959", "robust_reg": "#E69F00",
         "wrapper": "#009E73", "cmicl": "#CC79A7", "margin": "#56B4E9",
         "cp": "#0072B2"}
MARKER = {"nominal": "o", "robust_reg": "s", "wrapper": "^", "cmicl": "v",
          "margin": "P", "cp": "D"}
LINESTYLE = {"cmicl": "--", "margin": "--"}
# Objective sense per problem: synthetic minimises c'x, gastric maximises OS.
PROBLEMS = ["synthetic", "gastric"]
PTITLE = {"synthetic": "Synthetic", "gastric": "Gastric"}
OBJ_LABEL = {"synthetic": "Objective $c^\\top x^*$ (lower better)",
             "gastric": "Overall survival, months (higher better)"}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 13,
    "axes.titlesize": 13, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name, legend_ax=None, extra=None):
    """Save PDF+PNG. `legend_ax` supplies the handles for one shared legend
    below the panels, so no series is ever hidden behind a box. `extra` names
    the red reference line, which goes in that legend rather than in an in-panel
    label -- there is no one corner free of data in both panels."""
    if legend_ax is not None:
        h, l = legend_ax.get_legend_handles_labels()
        if extra is not None:
            h.append(plt.Line2D([], [], color="#C0392B", ls="--", lw=1.2))
            l.append(extra)
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=12,
                   bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf/.png")


def _load(problem, kind, suffix):
    path = f"{RES}/{problem}_{kind}{suffix}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def _series(df, method):
    """One method's rho curve, sorted along the axis."""
    return df[df["method"] == method].sort_values("rho")


def _rho_axis(ax, rhos):
    ax.set_xscale("log")
    # Each method is swept on its OWN parameter (run_rho_sweep.SWEEP_PARAM): a
    # radius for the shared-D methods, the RHS shift m for the margin baseline.
    # Both are in unexplained sds, which is what puts them on one axis -- but the
    # label must not call the margin's value a radius.
    ax.set_xlabel(r"Conservatism parameter, unexplained sd"
                  "\n"
                  r"($\rho$ = shared-$D$ radius; margin: $m$)",
                  fontsize=12)
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.minorticks_off()


def _curve_panel(ax, df, col, star=None):
    """Plot every method's `col` against rho; ring any capped cell."""
    for m in METHODS:
        s = _series(df, m)
        if s.empty:
            continue
        ax.plot(s["rho"], s[col], marker=MARKER[m], color=COLOR[m],
                label=LABEL[m], lw=1.8, ms=6.5, mec="white", mew=0.8,
                ls=LINESTYLE.get(m, "-"))
        # A capped cell is an incumbent at max_iterations, not a converged answer.
        cap = s[s["n_capped"] > 0]
        if not cap.empty:
            ax.plot(cap["rho"], cap[col], marker=MARKER[m], color=COLOR[m],
                    ls="none", ms=9, mec="black", mew=1.4, mfc="none")
    if star is not None:
        for m in METHODS:
            r = star[star["method"] == m]
            if r.empty or pd.isna(r["rho_star"].iloc[0]):
                continue
            ax.axvline(float(r["rho_star"].iloc[0]), color=COLOR[m],
                       ls=":", lw=1.0, alpha=0.55)


def fig_feasibility(curves, stars, target):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, p in zip(axes, PROBLEMS):
        df = curves[p]
        _curve_panel(ax, df, "feasibility", stars.get(p))
        ax.axhline(target, color="#C0392B", ls="--", lw=1.2)
        _rho_axis(ax, sorted(df["rho"].unique()))
        ax.set_ylabel("Held-out feasibility")
        ax.set_title(PTITLE[p])
        # Synthetic spans the whole unit range; gastric lives in a narrow band
        # around the target, so forcing 0-1 there would hide the entire signal.
        if p == "synthetic":
            ax.set_ylim(-0.05, 1.10)
    _save(fig, "fig_rho_feasibility", legend_ax=axes[0],
          extra=f"target {target:g}")


def fig_objective(curves):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, p in zip(axes, PROBLEMS):
        _curve_panel(ax, curves[p], "objective")
        _rho_axis(ax, sorted(curves[p]["rho"].unique()))
        ax.set_ylabel(OBJ_LABEL[p])
        ax.set_title(PTITLE[p])
    _save(fig, "fig_rho_objective", legend_ax=axes[0])


def fig_solved(curves, min_solved):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, p in zip(axes, PROBLEMS):
        df = curves[p]
        _curve_panel(ax, df, "solved_frac")
        ax.axhline(min_solved, color="#C0392B", ls="--", lw=1.2)
        _rho_axis(ax, sorted(df["rho"].unique()))
        ax.set_ylabel("Fraction of held-out contexts solved")
        ax.set_title(PTITLE[p])
        ax.set_ylim(-0.05, 1.10)
    _save(fig, "fig_rho_solved", legend_ax=axes[0],
          extra=f"min solved {min_solved:g}")


def fig_time(curves):
    """Where the wall clock goes. Gastric only -- synthetic is not contextual, so
    it has no prescribe phase and its test-point time is identically zero."""
    df = curves["gastric"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    panels = [(axes[0], "master_time_s", "Master phase (train + cut loop + solve)"),
              (axes[1], "test_time_per_point_s", "Per held-out context (prescribe)")]
    for ax, col, ttl in panels:
        _curve_panel(ax, df, col)
        ax.set_yscale("log")
        _rho_axis(ax, sorted(df["rho"].unique()))
        ax.set_ylabel("seconds")
        ax.set_title(ttl, fontsize=12)
    _save(fig, "fig_rho_time", legend_ax=axes[0])


def _knob_clusters(s, xspan, yspan, tol=0.02):
    """Group consecutive knob settings that land on the same point to plotting
    precision, so a saturated dial gets ONE label ("0.1/0.01/0.001") instead of
    three overlapping ones. Returns (x, y, label) per cluster."""
    out = []
    for _, r in s.iterrows():
        x, y, k = r["objective"], r["feasibility"], f"{r['knob']:g}"
        if out and (abs(x - out[-1][0]) <= tol * xspan
                    and abs(y - out[-1][1]) <= tol * yspan):
            out[-1][2].append(k)
        else:
            out.append([x, y, [k]])
    return [(x, y, "/".join(ks)) for x, y, ks in out]


def fig_ablation(abls):
    """Feasibility-objective frontier traced by each method's OWN dial, at the
    one rho the ablation was run at. Points are annotated with the knob value."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for ax, p in zip(axes, PROBLEMS):
        df = abls.get(p)
        if df is None or df.empty:
            ax.set_visible(False)
            continue
        rho = df["rho"].iloc[0]
        xspan = max(df["objective"].max() - df["objective"].min(), 1e-9)
        yspan = max(df["feasibility"].max() - df["feasibility"].min(), 1e-9)
        # Only tau and alpha are ablated. The margin is not: it is swept on the
        # MAIN curve now (it is that method's own parameter), so an ablation of it
        # would just re-plot the sweep. Its frontier is in fig_rho_feasibility /
        # fig_rho_objective alongside the shared-D methods.
        for m, knob in [("cp", r"$\tau$"), ("wrapper", r"$\alpha$")]:
            s = df[df["method"] == m].sort_values("knob")
            if s.empty:
                continue
            ax.plot(s["objective"], s["feasibility"], marker=MARKER[m],
                    color=COLOR[m], lw=1.6, ms=7, mec="white", mew=0.8,
                    label=f"{LABEL[m]}, {knob}")
            # A cell that stopped on the coverage cap never reached its tau.
            cap = s[s["status"].astype(str) != "optimal"]
            if not cap.empty:
                ax.plot(cap["objective"], cap["feasibility"], marker=MARKER[m],
                        color=COLOR[m], ls="none", ms=11, mec="black", mew=1.3,
                        mfc="none", label="_nolegend_")
            for i, (x, y, lab) in enumerate(
                    _knob_clusters(s, xspan, yspan)):
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(7, 6) if i % 2 == 0 else (7, -12),
                            fontsize=10, color=COLOR[m])
        ax.set_xlabel(OBJ_LABEL[p])
        ax.set_ylabel("Held-out feasibility")
        ax.set_title(f"{PTITLE[p]}  ($\\rho={rho:g}$)")
        ax.margins(x=0.14, y=0.14)
        h, lab = ax.get_legend_handles_labels()
        if (df["status"].astype(str) != "optimal").any():
            h.append(plt.Line2D([], [], ls="none", marker="o", ms=9,
                                mfc="none", mec="black", mew=1.3))
            lab.append("stopped on coverage cap")
        ax.legend(h, lab, loc="best", fontsize=11)
    _save(fig, "fig_rho_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_coh",
                    help="sweep cell suffix on the CSV names "
                         "(_coh/_incoh[_matchbank])")
    ap.add_argument("--feas-target", type=float, default=0.9)
    ap.add_argument("--min-solved", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    curves = {p: _load(p, "rho_curve", args.suffix) for p in PROBLEMS}
    stars = {p: _load(p, "rho_star", args.suffix) for p in PROBLEMS}
    abls = {p: _load(p, "ablations", args.suffix) for p in PROBLEMS}
    missing = [p for p, d in curves.items() if d is None]
    if missing:
        raise SystemExit(f"no rho curve for {missing} at suffix {args.suffix!r}")

    fig_feasibility(curves, stars, args.feas_target)
    fig_objective(curves)
    fig_solved(curves, args.min_solved)
    fig_time(curves)
    fig_ablation(abls)


if __name__ == "__main__":
    main()

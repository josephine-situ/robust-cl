"""PRIMARY axis: every method walked along its OWN dial at a FIXED rho.

This is the headline experiment. ``run_rho_sweep.py`` is now supporting evidence:
it answers "how much assumed uncertainty does each method absorb", which is a
question about D, while the question the contribution actually has to answer is
"at equal held-out feasibility, whose decisions are better". That one is read off
a curve in **objective x feasibility** space, and each method reaches its points
by moving the dial it really has -- not by being dragged along a rho axis two of
them do not face at all.

The grid::

    method      dial      rho columns                       notes
    ---------------------------------------------------------------------------
    cp          tau       gastric {0.5, 1.0}, reactor {1,2}  grid re-read per rho
    wrapper     alpha     same                               P is a bank prefix
    margin      m         --                                 faces no D
    cmicl       alpha     --                                 alpha=0.1 = protocol
    nominal     none      --                                 one reference point

``robust_reg`` is deliberately **absent**. Its dial ``label_eps`` IS D's radius,
so at a fixed rho it has no dial to move -- there is no curve for it on this axis.
It keeps its place on the rho sweep, where the axis is the quantity it actually
walks. Asking for it here is an error rather than a silent flat line.

**Both rho columns share one output CSV.** The ``rho`` column already tells them
apart, and one file per (problem, coherence, seed) is what the plot reads. The
checkpoint cell key is ``(method@rho, dial)``, so resume is unchanged.

Three things this file exists to get right, in the order they had to be fixed:

1. **Per-context records.** ``{problem}_dial_contexts{cell}.csv`` carries
   ``(fold, context_idx, solved, feasible, objective)`` for every cell. Primary
   scoring is unchanged -- each cell is still scored conditional on its OWN solved
   contexts, and that independence is load-bearing. But the objective is now the
   deliverable rather than a side column, and a conditional mean of it rewards
   whichever cell solved least. The same-cohort comparison is derivable from these
   rows afterwards; making it primary would couple every cell to every other.

2. **One bank per (rho column, fold).** A bank is a pure function of
   ``(instance, D, seed, B)``: neither tau nor alpha reaches it. So ONE bank of
   B=200 serves CP's whole tau grid and the wrapper's whole alpha grid -- the
   wrapper's P models are a prefix of CP's B. Gastric drops from ~14 bank builds
   per rho column to 1 per fold. See :class:`src.methods.cv_calibrate.FoldCache`.

3. **The tau grid is re-read per rho column.** tau's scale tracks D. The same
   absolute grid separates far harder at rho=0.5 than at rho=1.0, and more cells
   collapse to nominal. Each column is therefore PROBED -- one CP run at a tau so
   large it stops at iteration 0, reading ``CPHistory.iter0_tau`` -- and its grid
   is set as fractions of that. ``tau_frac = 1`` is exactly the value that stops
   before any cut, so that endpoint IS nominal and anchors the curve.
   **Fixing rho is what makes this legitimate rather than circular**: the grid is
   placed from a statistic of the ASSUMED set, not from the feasibility the grid
   is about to be scored on.

Ablation (``--cp-alpha-ablate``): sweep tau, read tau* at the target, hold tau*
and walk ``cp_alpha``. It asks whether relaxing the coverage cap lifts CP's
feasibility ceiling and what that costs in solved fraction. One rho column only
(``--cp-alpha-rho``; gastric 1.0, reactor 2.0). **Structurally inert on
synthetic and the reactor** -- both take CP's basic separation path, which has no
protected-anchor test to relax -- and the runner says so and skips rather than
producing a flat curve that looks like a measurement.

Outputs, all scoped by the same cell suffix ``run_rho_sweep`` uses
(``_coh``/``_incoh``[``_matchbank``][``_f<n>``][``_m<model>``][``_s<seed>``]):

  ``{problem}_dial_scores{cell}.csv``   resume checkpoint, keyed (method@rho, dial)
  ``{problem}_dial_contexts{cell}.csv`` per-context records
  ``{problem}_dial_curve{cell}.csv``    THE primary output; what the plot reads
  ``{problem}_dial_star{cell}.csv``     derived: each series' protocol point
  ``{problem}_tau_probe{cell}.json``    the placement statistic per rho column
  ``{problem}_cp_alpha{cell}.csv``      the coverage-cap ablation

Usage::

    python experiments/run_dial_sweep.py --problem gastric
    python experiments/run_dial_sweep.py --problem reactor --rho-columns 1 2 3
    python experiments/run_dial_sweep.py --problem gastric --coherent
    python experiments/run_dial_sweep.py --problem gastric --cp-alpha-ablate
    python experiments/plot_dial_sweep.py --problem gastric --suffix _incoh
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.cv_calibrate import (
    FoldCache, cv_score_knob, load_detail_checkpoint, append_score,
    append_contexts,
)
from experiments.run_rho_sweep import (
    _setup_synthetic, _setup_reactor, _setup_gastric, _variant_suffix,
    _bank_seed, _synth_n_folds,
)

OUT_DIR = "results/rho_sweep"

# Which dial each method is walked along at a fixed rho, and whether it faces D.
#
#   cp / wrapper  face D. One rho column is one shared set; within it the two are
#                 separated only by what they DO with it, which is the comparison.
#   margin        the RHS shift m. Faces no D, so it is scored once per problem
#                 and its curve is the same one against every rho column.
#   cmicl         the conformal miscoverage alpha. Faces no D either; alpha=0.1 is
#                 the PROTOCOL point (the conformal level and the feasibility
#                 target are the same quantity) and is flagged as such -- the rest
#                 of its grid is there to find where it first solves, not to be
#                 chosen from.
#   nominal       nothing to move: one reference point.
DIAL = {"cp": "tau", "wrapper": "alpha", "margin": "margin",
        "cmicl": "alpha", "nominal": None}
FACES_D = ("cp", "wrapper")
# The keyword each D-facing solver takes a prebuilt ScenarioBank under.
BANK_KWARG = {"cp": "cp_bank", "wrapper": "bank"}

# rho columns per problem. Two per problem: enough to show whether the ordering of
# the methods is a property of the methods or of the assumed radius, without
# doubling into a rho sweep by the back door.
#   gastric  0.5 / 1.0 -- where CP's committed curve is strongest, and the point
#            below it. Both are headline columns now, not sensitivity checks.
#   reactor  1 / 2 -- nominal misses the benzene target by ~4 units of F and
#            rho=1 buys ~2.2, so 2 is right at the edge. Add 3 if it is short:
#            --rho-columns 1 2 3.
DEFAULT_RHO_COLUMNS = {"gastric": [0.5, 1.0], "reactor": [1.0, 2.0],
                       "synthetic": [0.5, 1.0]}

# CP's tau grid is RELATIVE: these are fractions of the column's own iteration-0
# distance (CPHistory.iter0_tau). 1.0 stops before any cut and is therefore
# nominal, which anchors the curve at a known point; 0.02 cuts essentially
# everything the bank offers.
DEFAULT_TAU_FRACS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
# A tau far above any iteration-0 distance: the probe run stops at iteration 0 and
# reports the statistic instead of separating.
PROBE_TAU = 1e9

DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
# m=0 is bit-identical to nominal (same fit, same MIP, same x*), so the baseline's
# curve starts AT the nominal point rather than near it.
DEFAULT_MARGIN_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
# Extended UPWARD on purpose. C-MICL is measured infeasible on gastric at
# alpha=0.1 under both multiplicity settings, and n_cal=80 means alpha >= 0.02 is
# needed for a finite conformal quantile at all. Where it FIRST solves is the
# result; 0.1 is the protocol point whether or not it solves there.
DEFAULT_CMICL_ALPHA_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
# The coverage cap, as a fraction of the anchors a cut may newly break.
DEFAULT_CP_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def _setup(config, args):
    return {"synthetic": _setup_synthetic, "reactor": _setup_reactor,
            "gastric": _setup_gastric}[args.problem](config, args)


def _bank_size(config, args):
    """B for the SHARED bank: enough for CP's separation pool and the wrapper's P.

    CP separates over B draws and the wrapper embeds the first P of them, so one
    bank of ``max(B, P)`` serves both. ``--match-bank`` sets CP's B to P, which is
    the whole point of that flag; the max then collapses to P and the two methods
    sample D at the same density.
    """
    unc = config.get("uncertainty", {}) or {}
    methods = config.get("methods", {}) or {}
    p = int((methods.get("wrapper", {}) or {}).get("n_estimators",
                                                   unc.get("n_bootstrap", 20)))
    p = max(p, int(unc.get("n_bootstrap", 20)))
    b = p if args.match_bank else int((methods.get("cp", {}) or {})
                                      .get("n_scenarios", 200))
    return max(b, p)


def _make_bank_factory(config, args, uset, model_spec, seed):
    """``fold_instance -> ScenarioBank`` for one rho column."""
    from src.methods.uncertainty import build_bank_for_instance

    model_type, model_params = model_spec
    n = _bank_size(config, args)

    def factory(fold_instance):
        return build_bank_for_instance(fold_instance, model_type, model_params,
                                       uset, n_scenarios=n, seed=seed,
                                       verbose=True)
    return factory


# ---------------------------------------------------------------------------
# The tau probe: place CP's grid from THIS rho column's own distances
# ---------------------------------------------------------------------------
def _probe_iter0_tau(su, uset, cache, args, n_probe_folds):
    """The tau at which CP would stop before cutting, on this rho column.

    One CP run per probed fold at ``PROBE_TAU`` -- far above any distance, so the
    loop separates once, reports, and stops at iteration 0. What comes back is
    ``CPHistory.iter0_tau``: the iteration-0 separation statistic expressed in
    tau's own units, whatever units the path being taken logs in. Taking it from
    the history rather than parsing the log is what makes this work on the basic
    and the contextual paths alike, which measure different statistics.

    The MAX over probed folds is used, not the mean: the grid has to bracket the
    largest iteration-0 distance any fold shows, or ``tau_frac = 1`` stops before
    any cut on some folds and cuts on others, and the endpoint stops being
    nominal. Probing one fold (the default) is a placement heuristic and is
    labelled as one; it never enters a scored cell.
    """
    build = su.make_build("cp", uset)
    solver = build(PROBE_TAU)
    vals = []
    for k, (train_idx, val_idx) in enumerate(su.folds):
        if k >= n_probe_folds:
            break
        val_rows = (su.instance.X_train[val_idx]
                    if (su.contextual and su.instance.X_train is not None) else None)
        fi = cache.instance(k, su.instance, train_idx, val_rows,
                            su.constraint_names)
        extra = {}
        bank = cache.bank(k, fi)
        if bank is not None:
            extra["cp_bank"] = bank
        out = solver(fi, **extra)
        hist = out[1] if isinstance(out, tuple) else None
        v = getattr(hist, "iter0_tau", None)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return max(vals) if vals else None


def _tau_grid(iter0_tau, fracs):
    """Absolute tau values from the probe, deduplicated and sorted descending."""
    if iter0_tau is None:
        return None
    out = sorted({float(f) * float(iter0_tau) for f in fracs}, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run(config, args):
    from src.methods.uncertainty import uncertainty_set_from_config

    os.makedirs(OUT_DIR, exist_ok=True)
    problem = args.problem
    su = _setup(config, args)
    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )
    bank_seed = _bank_seed(config, args)
    var = _variant_suffix(args)

    scores_path = os.path.join(OUT_DIR, f"{problem}_dial_scores{var}.csv")
    ctx_path = os.path.join(OUT_DIR, f"{problem}_dial_contexts{var}.csv")
    probe_path = os.path.join(OUT_DIR, f"{problem}_tau_probe{var}.json")
    if args.refresh:
        for p in (scores_path, ctx_path, probe_path):
            if os.path.exists(p):
                os.remove(p)
    ckpt = load_detail_checkpoint(scores_path) if not args.refresh else {}
    probes = json.load(open(probe_path)) if os.path.exists(probe_path) else {}

    methods = list(args.methods or ["nominal", "cp", "wrapper", "margin", "cmicl"])
    bad = [m for m in methods if m not in DIAL]
    if bad:
        extra = (" robust_reg has no dial at a fixed rho -- its dial IS the "
                 "radius. It belongs on run_rho_sweep.py."
                 if "robust_reg" in bad else "")
        raise SystemExit(f"[dial-sweep] not on this axis: {bad}.{extra}")
    rho_cols = [float(r) for r in (args.rho_columns
                                   or DEFAULT_RHO_COLUMNS[problem])]
    d_methods = [m for m in methods if m in FACES_D]
    flat_methods = [m for m in methods if m not in FACES_D]

    print(f"[dial-sweep] problem={problem} geometry=ellipsoid "
          f"coherent={args.coherent} folds={len(su.folds)} seed={bank_seed} "
          f"sense={su.oracle.objective_sense}", flush=True)
    print(f"[dial-sweep] rho columns={rho_cols} (D-facing: {d_methods or 'none'})",
          flush=True)
    print(f"[dial-sweep] dials: "
          + ", ".join(f"{m}={DIAL[m] or 'none'}" for m in methods), flush=True)
    print(f"[dial-sweep] shared bank B={_bank_size(config, args)} per "
          f"(rho column, fold); CP's tau grid and the wrapper's alpha grid both "
          f"read it", flush=True)
    if not su.contextual:
        print(f"[dial-sweep] WARNING: single-decision problem -- feasibility is "
              f"quantized to 1/{len(su.folds)} = {1/len(su.folds):.2f}. Read the "
              f"scatter, not the protocol point.", flush=True)

    rows = []

    def score(tag, method, uset, dial, cache=None, cp_alpha=None, phase="dial",
              rho=np.nan, note=""):
        """One scored cell, resumable, with per-context records."""
        ckey = (tag, float(dial))
        if ckey not in ckpt:
            build = su.make_build(method, uset, cp_alpha=cp_alpha)
            d = cv_score_knob(
                build, dial, su.folds, su.oracle, su.instance,
                constraint_names=su.constraint_names, contextual=su.contextual,
                return_details=True, return_contexts=True,
                fold_cache=cache, bank_kwarg=BANK_KWARG.get(method),
            )
            append_score(scores_path, tag, dial, d["feas"], d["obj"], d["solved"], d)
            append_contexts(ctx_path, tag, dial, d.pop("contexts", []))
            ckpt[ckey] = d
        d = ckpt[ckey]
        cap = f" CAPPED({d['n_capped']}/{len(su.folds)})" if d.get("n_capped") else ""
        print(f"  {tag:<28s} {DIAL[method] or 'dial'}={dial:<10.5g} "
              f"feas={d['feas']:.3f} obj={d['obj']:+.4f} solved={d['solved']:.3f} "
              f"master={d['master_time_s']:.1f}s "
              f"test/pt={d['test_time_per_point_s']:.2f}s "
              f"[{d['status']}]{cap}", flush=True)
        rows.append(dict(
            problem=problem, method=method, rho=rho, dial=float(dial),
            dial_name=DIAL[method] or "none", faces_D=method in FACES_D,
            feasibility=d["feas"], objective=d["obj"], solved_frac=d["solved"],
            status=d["status"], n_capped=d["n_capped"],
            master_time_s=d["master_time_s"], test_time_s=d["test_time_s"],
            test_time_per_point_s=d["test_time_per_point_s"],
            objective_sense=su.oracle.objective_sense,
            coherent=bool(args.coherent), matched_bank=bool(args.match_bank),
            seed=bank_seed, cp_alpha=(np.nan if cp_alpha is None else float(cp_alpha)),
            phase=phase, note=note,
        ))
        return d

    # ---- the D-facing methods, one rho column at a time --------------------
    cache = None
    for rho in rho_cols:
        if not d_methods:
            break
        uset = dataclasses.replace(base_uset, rho=rho)
        # A fresh cache per column: the bank is keyed on the fold index alone, so
        # carrying one across columns would hand rho=1.0 the rho=0.5 bank.
        if cache is not None:
            cache.clear()
        cache = FoldCache(
            bank_factory=_make_bank_factory(config, args, uset, su.model_spec,
                                            bank_seed),
            key=(rho, bool(args.coherent), bank_seed, _bank_size(config, args)),
        )
        print(f"\n[dial-sweep] === rho = {rho:g} ===", flush=True)

        tau_vals = None
        if "cp" in d_methods:
            if args.tau_grid:
                tau_vals = [float(t) for t in args.tau_grid]
                print(f"[dial-sweep] tau grid FORCED (absolute): {tau_vals} -- "
                      f"not placed from this column's distances", flush=True)
            else:
                key = f"{rho:g}"
                if key not in probes:
                    probes[key] = _probe_iter0_tau(
                        su, uset, cache, args, int(args.tau_probe_folds))
                    json.dump(probes, open(probe_path, "w"), indent=1)
                iter0 = probes[key]
                tau_vals = _tau_grid(iter0, args.tau_frac_grid or DEFAULT_TAU_FRACS)
                if tau_vals is None:
                    print(f"[dial-sweep] WARNING: probe returned no iteration-0 "
                          f"statistic at rho={rho:g}; skipping cp on this column",
                          flush=True)
                else:
                    print(f"[dial-sweep] tau probe at rho={rho:g}: "
                          f"iter0_tau={iter0:.5g} over "
                          f"{int(args.tau_probe_folds)} fold(s); grid="
                          f"{[round(t, 6) for t in tau_vals]} "
                          f"(fracs {args.tau_frac_grid or DEFAULT_TAU_FRACS}; "
                          f"the largest stops before any cut = nominal)",
                          flush=True)

        for method in d_methods:
            grid = (tau_vals if method == "cp"
                    else [float(a) for a in (args.alpha_grid or DEFAULT_ALPHA_GRID)])
            if grid is None:
                continue
            for dial in grid:
                score(f"{method}@rho={rho:g}", method, uset, dial, cache=cache,
                      rho=rho)

    # ---- the methods that face no D: scored ONCE -------------------------
    # No rho column: neither reads any uncertainty.* key that a column moves, so a
    # per-column repeat would re-measure one number. The plot draws them once and
    # they sit against every column.
    for method in flat_methods:
        if DIAL[method] is None:
            print(f"\n[dial-sweep] === {method} (no dial: one reference point) ===",
                  flush=True)
            score(method, method, base_uset, 0.0, rho=np.nan, phase="reference",
                  note="no dial")
            continue
        grid = {
            "margin": args.margin_grid or DEFAULT_MARGIN_GRID,
            "cmicl": args.cmicl_alpha_grid or DEFAULT_CMICL_ALPHA_GRID,
        }[method]
        print(f"\n[dial-sweep] === {method} (dial={DIAL[method]}, faces no D) ===",
              flush=True)
        if method == "cmicl":
            print(f"[dial-sweep] NOTE: cmicl's PROTOCOL point is "
                  f"alpha={1 - float(args.feas_target):g} -- the conformal level "
                  f"and the feasibility target are the same quantity, so it is "
                  f"asserted, not chosen. The rest of the grid is here to find "
                  f"where it first SOLVES; on gastric it is measured infeasible "
                  f"at 0.1 under either multiplicity setting.", flush=True)
        for dial in [float(v) for v in grid]:
            protocol = (method == "cmicl"
                        and np.isclose(dial, 1 - float(args.feas_target)))
            score(method, method, base_uset, dial, rho=np.nan,
                  note="protocol point" if protocol else "")

    curve = os.path.join(OUT_DIR, f"{problem}_dial_curve{var}.csv")
    star = _dial_star(pd.DataFrame(rows), problem, float(args.feas_target),
                      su.oracle.objective_sense, float(args.min_solved),
                      out_suffix=var)

    # ---- the coverage-cap ablation ---------------------------------------
    # Runs AFTER the protocol table, because it holds tau at the tau* that table
    # reports. Its rows land in the same curve CSV under phase="cp_alpha_ablation"
    # -- the plot draws them as their own series and _dial_star excludes them.
    if args.cp_alpha_ablate:
        _cp_alpha_ablation(su, config, args, base_uset, star, score, problem,
                           bank_seed, var)

    df = pd.DataFrame(rows)
    df.to_csv(curve, index=False)
    print(f"\n[dial-sweep] wrote {curve}", flush=True)
    return df


def _cp_alpha_ablation(su, config, args, base_uset, star, score, problem,
                       bank_seed, var):
    """Hold tau at tau*, walk ``cp_alpha``.

    The question: CP's held-out feasibility tops out near 0.984 on gastric, and the
    cap is the obvious suspect -- a cut that would break one protected anchor is
    rolled back whole, so the adversary CP is allowed to admit is bounded by the
    single most fragile patient. Relaxing the cap admits stronger cuts at the price
    of dropping anchors, which shows up as solved fraction. Both columns are
    reported; neither is a free win.

    tau* comes from the sweep just run, at ``--cp-alpha-rho``, so the ablation
    varies ONE thing against a point the main curve already reports.
    """
    if not su.contextual:
        print(f"\n[dial-sweep] SKIPPING the cp_alpha ablation on {problem}: it is "
              f"structurally inert here. Single-decision problems take CP's BASIC "
              f"separation path, which has no protected-anchor test -- there is no "
              f"coverage cap to relax, so every alpha would return the same run. "
              f"See cp._BasicSeparation.", flush=True)
        return
    rho_a = float(args.cp_alpha_rho if args.cp_alpha_rho is not None
                  else max(args.rho_columns or DEFAULT_RHO_COLUMNS[problem]))
    sel = star[(star["method"] == "cp") & np.isclose(star["rho"], rho_a)]
    if sel.empty or not np.isfinite(sel["dial_star"].iloc[0]):
        print(f"\n[dial-sweep] SKIPPING the cp_alpha ablation: cp has no tau* at "
              f"rho={rho_a:g} (it never reaches feasibility "
              f">= {args.feas_target:g} on enough solved contexts there), so "
              f"there is no point to hold tau at.", flush=True)
        return
    tau_star = float(sel["dial_star"].iloc[0])
    uset = dataclasses.replace(base_uset, rho=rho_a)
    cache = FoldCache(
        bank_factory=_make_bank_factory(config, args, uset, su.model_spec,
                                        bank_seed),
        key=("cp_alpha", rho_a, bool(args.coherent), bank_seed),
    )
    print(f"\n[dial-sweep] === coverage-cap ablation: rho={rho_a:g}, "
          f"tau*={tau_star:.5g} held fixed ===", flush=True)
    print(f"[dial-sweep] cp_alpha > 0 lets a cut break up to that fraction of the "
          f"anchors. It buys a stronger adversary by dropping patients, so read "
          f"feasibility and solved_frac together -- neither alone is the result.",
          flush=True)
    for a in [float(v) for v in (args.cp_alpha_grid or DEFAULT_CP_ALPHA_GRID)]:
        score(f"cp@rho={rho_a:g}@cpalpha={a:g}", "cp", uset, tau_star,
              cache=cache, cp_alpha=a, phase="cp_alpha_ablation", rho=rho_a,
              note=f"tau*={tau_star:.5g}")
    cache.clear()


def _dial_star(df, problem, target, sense, min_solved, out_suffix=""):
    """The PROTOCOL POINT of each series: the best objective that still delivers.

    A series here is one (method, rho column). Unlike ``rho*`` -- "the largest
    assumed radius the method absorbs" -- a dial has no direction that is
    automatically "more"; alpha and tau run opposite ways round from m. So the
    point picked is the operationally meaningful one: among the cells meeting the
    feasibility target on enough solved contexts, the one with the **best
    objective** in the problem's own sense. That is the decision a user would take
    from this series, and it needs no monotonicity assumption.

    ``solved_frac >= min_solved`` is applied FIRST and is the artefact guard: a
    dial that renders most contexts unsolvable and gets the survivors right scores
    1.0 feasibility. It is why the scatter encodes solved fraction too -- the
    table's floor is a cliff, the plot's marker size is the gradient.

    ``bound`` says what stopped the search, because the three cases mean different
    things: ``grid_end`` (the winning cell sits at an end of the dial grid, so the
    grid may be the limit rather than the method), ``interior`` (a genuine
    optimum), ``none`` (the series never reaches the target at all).
    """
    rows = []
    main = df[df["phase"] == "dial"] if "phase" in df else df
    ref = df[df["phase"] == "reference"] if "phase" in df else df.iloc[0:0]
    better = (lambda a, b: a < b) if sense == "min" else (lambda a, b: a > b)

    for (method, rho), g_all in main.groupby(["method", "rho"], dropna=False):
        g_all = g_all.sort_values("dial")
        ends = {float(g_all["dial"].min()), float(g_all["dial"].max())}
        g = g_all[g_all["solved_frac"] >= min_solved]
        ok = g[g["feasibility"] >= target]
        if ok.empty:
            rows.append(dict(method=method, rho=rho, dial_name=g_all["dial_name"].iloc[0],
                             dial_star=np.nan, feasibility=np.nan, objective=np.nan,
                             solved_frac=np.nan, n_capped=0, master_time_s=np.nan,
                             test_time_per_point_s=np.nan, bound="none",
                             note=f"never reaches feas>={target:g} at "
                                  f"solved_frac>={min_solved:g}"))
            continue
        best = ok.iloc[0]
        for _, r in ok.iterrows():
            if better(float(r["objective"]), float(best["objective"])):
                best = r
        notes = []
        if float(best["dial"]) in ends:
            notes.append("at a grid end -- the dial grid may be the limit, "
                         "not the method")
        if int(best.get("n_capped", 0) or 0):
            notes.append(f"n_capped={int(best['n_capped'])} (incumbent, not converged)")
        rows.append(dict(method=method, rho=rho, dial_name=str(best["dial_name"]),
                         dial_star=float(best["dial"]),
                         feasibility=float(best["feasibility"]),
                         objective=float(best["objective"]),
                         solved_frac=float(best["solved_frac"]),
                         n_capped=int(best.get("n_capped", 0) or 0),
                         master_time_s=float(best["master_time_s"]),
                         test_time_per_point_s=float(best["test_time_per_point_s"]),
                         bound=("grid_end" if float(best["dial"]) in ends
                                else "interior"),
                         note="; ".join(notes)))
    for _, r in ref.iterrows():
        # nominal: no dial, so no protocol point -- carried as the reference level
        # the whole plot is read against.
        rows.append(dict(method=r["method"], rho=np.nan, dial_name="none",
                         dial_star=np.nan, feasibility=float(r["feasibility"]),
                         objective=float(r["objective"]),
                         solved_frac=float(r["solved_frac"]),
                         n_capped=int(r.get("n_capped", 0) or 0),
                         master_time_s=float(r["master_time_s"]),
                         test_time_per_point_s=float(r["test_time_per_point_s"]),
                         bound="no_dial", note="reference level, nothing to move"))
    out = pd.DataFrame(rows)
    out.insert(0, "feas_target", target)
    out.insert(1, "min_solved", min_solved)
    out.insert(2, "objective_sense", sense)
    path = os.path.join(OUT_DIR, f"{problem}_dial_star{out_suffix}.csv")
    out.to_csv(path, index=False)
    print(f"\n[dial-sweep] protocol point per series "
          f"(feasibility >= {target:g}, solved_frac >= {min_solved:g}, "
          f"best objective, sense={sense})")
    print(out.to_string(index=False))
    print(f"[dial-sweep] wrote {path}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "reactor", "gastric"),
                   default="gastric")
    p.add_argument("--methods", nargs="+", default=None,
                   help="default: nominal cp wrapper margin cmicl. robust_reg is "
                        "NOT on this axis -- its dial is the radius, so at a fixed "
                        "rho it has none; it stays on run_rho_sweep.py")
    p.add_argument("--rho-columns", type=float, nargs="+", default=None,
                   help=f"rho values to run the D-facing methods at "
                        f"(default {DEFAULT_RHO_COLUMNS}). Both columns share one "
                        f"output CSV; the rho column tells them apart")
    p.add_argument("--tau-frac-grid", type=float, nargs="+", default=None,
                   help=f"CP's tau grid as FRACTIONS of each rho column's own "
                        f"iteration-0 distance (default {DEFAULT_TAU_FRACS}). 1.0 "
                        f"stops before any cut and is therefore nominal")
    p.add_argument("--tau-grid", type=float, nargs="+", default=None,
                   help="absolute tau values, overriding the probe. Legal, but "
                        "one absolute grid does not transfer across rho columns "
                        "-- tau's scale tracks D")
    p.add_argument("--tau-probe-folds", type=int, default=1,
                   help="folds to probe for the iteration-0 statistic (default 1). "
                        "The max over them places the grid; it never enters a "
                        "scored cell")
    p.add_argument("--alpha-grid", type=float, nargs="+", default=None,
                   help=f"wrapper alpha grid (default {DEFAULT_ALPHA_GRID}); "
                        "0/0.1/0.2/0.5 are OptiCL's published WFP values")
    p.add_argument("--margin-grid", type=float, nargs="+", default=None,
                   help=f"margin m grid, unexplained-sd units "
                        f"(default {DEFAULT_MARGIN_GRID}); m=0 IS nominal")
    p.add_argument("--cmicl-alpha-grid", type=float, nargs="+", default=None,
                   help=f"C-MICL miscoverage grid (default "
                        f"{DEFAULT_CMICL_ALPHA_GRID}). Extended upward on purpose: "
                        f"where it FIRST solves on gastric is the result")
    p.add_argument("--cp-alpha-ablate", action="store_true",
                   help="after the sweep, hold tau at tau* and walk CP's coverage "
                        "cap. Contextual problems only (gastric)")
    p.add_argument("--cp-alpha-rho", type=float, default=None,
                   help="rho column for the coverage-cap ablation (default: the "
                        "largest column). gastric 1.0, reactor 2.0")
    p.add_argument("--cp-alpha-grid", type=float, nargs="+", default=None,
                   help=f"coverage-cap values (default {DEFAULT_CP_ALPHA_GRID}); "
                        "0 is the production pin")
    p.add_argument("--feas-target", type=float, default=0.9,
                   help="held-out feasibility target (default 0.9). Also fixes "
                        "C-MICL's protocol point at 1 - target")
    p.add_argument("--min-solved", type=float, default=0.5,
                   help="drop cells below this solved fraction before reading the "
                        "protocol point (default 0.5)")
    p.add_argument("--incoherent", dest="coherent", action="store_false",
                   default=False)
    p.add_argument("--coherent", dest="coherent", action="store_true")
    p.add_argument("--separation", dest="separation", default=None,
                   choices=("auto", "coherent", "incoherent"))
    p.add_argument("--match-bank", action="store_true",
                   help="set CP's bank B to the wrapper's P, removing the B!=P "
                        "confound")
    p.add_argument("--n-folds", type=int, default=None,
                   help="single-decision problems only; default "
                        "cv_calibration.n_kfold")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the DRAWS FROM D only; the data and the folds "
                        "keep the config seed. Scopes every output with _s<seed>")
    p.add_argument("--refresh", action="store_true",
                   help="discard the score cache, the context records and the tau "
                        "probe")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    args = p.parse_args()

    import yaml
    config = yaml.safe_load(open(args.config))

    if args.separation is None:
        args.separation = (
            config.get("methods", {}).get("cp", {}).get("separation", "auto"))
    if args.problem in ("synthetic", "reactor"):
        args.n_folds = _synth_n_folds(config, args)
        from experiments.run_sweep import synth_model_spec, reactor_model_spec
        spec = (synth_model_spec if args.problem == "synthetic"
                else reactor_model_spec)
        args.synth_model = spec(config)[0]

    run(config, args)


if __name__ == "__main__":
    main()

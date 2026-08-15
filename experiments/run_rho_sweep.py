"""Global rho sweep: every method read off one shared-D axis.

``rho`` is the single size parameter of the ellipsoidal D
(``R_c = rho * scale(y_c) * sqrt(n)``, see :mod:`src.methods.uncertainty`). It
defines the problem all three methods solve, so it is **swept and reported, never
selected**: choosing it per method would dissolve the shared-D premise, and
choosing it globally has no honest criterion -- against the GT ensemble it tunes
to the judge, against synthetic's known ``noise_std`` it calibrates D to the
data-generating process.

Every output is scoped by the sweep CELL -- coherence and whether CP's bank B was
matched to the wrapper's P -- via a ``_coh``/``_incoh``[``_matchbank``] suffix, so
the pair of runs the workflow asks for coexist instead of one silently resuming
from and overwriting the other. Names below omit that suffix.

Two outputs, from one pass:

  1. ``{problem}_rho_sweep.csv`` -- the PRIMARY reading. Held-out feasibility and
     objective for every (method, rho) cell, all methods on the same axis with
     their own conservatism dials FIXED. D stays literally shared at each rho, so
     a gap between methods is a difference in method.

  2. ``{problem}_rho_star.csv`` -- the DERIVED reading, and what now plays the role
     stage-1 knob CV used to: rho*(method) = the largest rho whose held-out
     feasibility still meets ``--feas-target``, i.e. how much assumed uncertainty
     each method absorbs before it stops delivering, and what that costs in
     objective and in time. A *result read off the sweep*, not a fitted parameter.

  3. ``{problem}_ablations.csv`` (``--ablate``) -- tau and alpha swept at ONE
     chosen rho. Those are each method's own dial, held fixed for the shared-D
     comparison above; the ablation exists to show the fixed value was not
     cherry-picked, which needs one rho, not the whole grid.

Every cell carries ``status``, ``n_capped``, and the wall clock split into the
MASTER phase (train + build + solve to the final master; for CP the entire cut
loop) and the TEST-POINT phase (one prescribe solve per held-out context). The
split is the point: CP pays up front in the cut loop and then prescribes from a
small master, while the wrapper embeds all P models and pays again on every test
point. ``n_capped`` counts folds that hit ``max_iterations``; those cells are
KEPT (the incumbent is still usable) and flagged, not dropped.

**rho* is a reporting choice, so it is re-derivable without re-solving.** The
curve CSV carries every column the criteria could need, and ``--rho-star-only``
recomputes the table from it under a new ``--feas-target`` / ``--min-solved`` /
``--exclude-capped``, writing to ``--out-suffix`` so several criteria coexist
instead of overwriting each other. The chosen criteria are recorded as columns in
the output, so a table is never ambiguous about which rule produced it.

Scoring is ``cv_calibrate.cv_score_knob`` -- the same held-out folds, oracle, and
conditional-on-solved convention stage-1 knob CV uses, so the two are comparable.
``solved_frac`` is carried through: high feasibility at a low solved fraction is
an artefact (a cell that renders most contexts unsolvable and gets the survivors
right scores 1.0), not a win.

Two things to know before reading a sweep:

- **robust_reg's knob IS rho.** ``label_eps`` is already a D radius in units of
  ``scale(y)``, so under the ellipsoid it and rho are the same number; holding it
  fixed while rho moves would train the adversary against a different set than
  CP and the wrapper draw from. It therefore tracks rho, and ``--knobs`` cannot
  override it. CP's tau and the wrapper's alpha are genuinely separate dials and
  stay fixed.
- **B != P is a confound on the rho* table.** CP samples D with B=200 draws, the
  wrapper with P=20. A wrapper that needs a smaller rho may simply be sampling
  the same D more sparsely. ``--match-bank`` sets B=P for an apples-to-apples
  comparison; without it, report the gap as confounded.

Usage:
    python experiments/run_rho_sweep.py --problem synthetic
    python experiments/run_rho_sweep.py --problem synthetic --match-bank
    python experiments/run_rho_sweep.py --problem gastric --coherent --ablate
    # re-derive rho* under different criteria, no solving. The cell flags pick the
    # curve; --out-suffix names the criteria:
    python experiments/run_rho_sweep.py --problem gastric --coherent \
        --rho-star-only --feas-target 0.8 --out-suffix _t080
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
    make_folds, make_cv_oracle, cv_score_knob, load_detail_checkpoint,
    append_score, lookup_knob,
)

DEFAULT_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
# Ablation grids, run at the CHOSEN rho rather than swept jointly with it.
DEFAULT_TAU_GRID = [1.0, 0.1, 0.01, 0.001]
DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
OUT_DIR = "results/rho_sweep"


# ---------------------------------------------------------------------------
# Per-problem setup: instance, folds, oracle, and build(knob) -> solver
# ---------------------------------------------------------------------------
def _setup_synthetic(config, args):
    from experiments.run_sweep import _synth_instance, _synth_build

    seed = config["uncertainty"].get("bootstrap_seed", 42)
    inst = _synth_instance(config)
    n_kfold = int(args.n_folds or
                  config.get("cv_calibration", {}).get("n_kfold", 4))
    folds = make_folds(inst, "kfold", n_kfold=n_kfold, seed=seed)
    oracle = make_cv_oracle(inst)

    def make_build(method, uset):
        # _synth_build reads the uncertainty set off the config dict, so the swept
        # rho is injected there rather than passed.
        cfg = json.loads(json.dumps(config))          # deep copy, config is plain data
        cfg["uncertainty"].update(
            geometry=uset.geometry, rho=uset.rho,
            coherent=uset.coherent,
            coherent_exclude=list(uset.coherent_exclude),
        )
        if args.match_bank:
            cfg.setdefault("methods", {}).setdefault("cp", {})["n_scenarios"] = \
                int(cfg["uncertainty"].get("n_bootstrap", 20))
        return _synth_build(method, cfg, config["model"]["type"],
                            config["model"]["params"], seed)

    return inst, folds, oracle, make_build, None, False


def _setup_gastric(config, args):
    from experiments.run_chemo_robust import (
        _resolve_run_settings, _method_build_map, _cs_ranges, ALL_CONSTRAINTS,
    )
    from src.data.generate import gastric_cancer

    cv_configs, gt_configs = _load_cv_configs(args)
    settings = _resolve_run_settings(config, _chemo_args())
    inst = gastric_cancer(fixed_constraint_configs=cv_configs,
                          fixed_gt_ensemble_configs=gt_configs)
    cvc = config.get("cv_calibration", {})
    folds = make_folds(inst, cvc.get("fold_scheme", "auto"),
                       tuple(cvc.get("fold_cutoffs", (2004, 2005, 2006, 2007))),
                       int(cvc.get("n_kfold", 4)), settings["bootstrap_seed"])
    oracle = make_cv_oracle(inst, gt_specs=gt_configs)
    ranges = _cs_ranges(settings)

    def make_build(method, uset):
        cell = dict(settings)
        cell["uncertainty_set"] = uset
        if args.match_bank:
            cell["cp_n_scenarios"] = int(config["uncertainty"].get("n_bootstrap", 20))
        build, _ = _method_build_map(method, cell, ranges,
                                     config["model"]["type"],
                                     config["model"]["params"], None, None)
        return build

    return inst, folds, oracle, make_build, ALL_CONSTRAINTS, bool(inst.context_var_indices)


def _chemo_args():
    """The flags ``run_chemo_robust._resolve_run_settings`` reads, at full-run values.

    The sweep's parser is its own; it shares no flags with the gastric runner, and
    its ``--methods`` means "methods to sweep", not ``methods_to_run`` (the sweep
    builds one solver per method by name, so that key is unused here). Passing the
    sweep namespace straight through therefore both crashed on ``args.quick`` and
    would have silently repurposed ``--methods``. Full-run settings are what the
    sweep wants anyway: --quick shrinks B, the anchor count and the iteration cap,
    which would change what each rho cell measures.
    """
    return argparse.Namespace(
        quick=False,
        max_test_rows=None,
        methods=None,
        output=None,
        cp_robustify_objective=None,
        cp_eval_mode=None,
    )


def _variant_suffix(args):
    """Filename suffix naming the sweep CELL: coherence, and whether B was matched to P.

    Every output is scoped by it because the cells are *different experiments* that
    the workflow explicitly asks you to run as a pair (coherent vs incoherent; B=200
    vs B=P). Sharing one filename across them fails in two ways, both silent:

      - the resume checkpoint is keyed ``(method@rho, knob)`` only, so a second
        cell RESUMES the first one's rows and reports them as its own;
      - ``{problem}_rho_curve.csv`` is written, not appended, so the second cell
        OVERWRITES the first -- including the curve ``--rho-star-only`` re-derives
        from, which is the whole point of saving it.

    The rows already carried ``coherent`` and ``matched_bank`` columns, so they were
    always meant to coexist; only the paths had not caught up.
    """
    return ("_coh" if args.coherent else "_incoh") + \
           ("_matchbank" if getattr(args, "match_bank", False) else "")


def _load_cv_configs(args):
    """Frozen constraint-model / GT-ensemble configs, as run_chemo_robust reads them."""
    cv_configs = gt_configs = None
    base = getattr(args, "cv_configs", None)
    if base and os.path.exists(base):
        cv_configs = json.load(open(base))
        gt_path = base.replace("_selected_configs", "_gt_ensemble_configs")
        if os.path.exists(gt_path):
            gt_configs = json.load(open(gt_path))
    return cv_configs, gt_configs


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(config, args):
    from src.methods.uncertainty import uncertainty_set_from_config

    os.makedirs(OUT_DIR, exist_ok=True)
    problem = args.problem
    setup = _setup_synthetic if problem == "synthetic" else _setup_gastric
    inst, folds, oracle, make_build, cnames, contextual = setup(config, args)

    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )
    knobs = _fixed_knobs(problem, args, config)
    grid = [float(r) for r in (args.rho_grid or DEFAULT_GRID)]
    methods = args.methods or ["nominal", "cp", "wrapper", "robust_reg"]

    var = _variant_suffix(args)
    scores_path = os.path.join(OUT_DIR, f"{problem}_rho_sweep{var}.csv")
    if args.refresh and os.path.exists(scores_path):
        os.remove(scores_path)
    ckpt = load_detail_checkpoint(scores_path) if not args.refresh else {}

    print(f"[rho-sweep] problem={problem} geometry=ellipsoid "
          f"coherent={args.coherent} folds={len(folds)} "
          f"sense={oracle.objective_sense}", flush=True)
    print(f"[rho-sweep] grid={grid}", flush=True)
    print(f"[rho-sweep] fixed knobs: {knobs}", flush=True)
    if not contextual:
        # Single-decision problems solve ONCE per fold, so held-out feasibility is
        # a mean of len(folds) binary outcomes -- quantized to 1/len(folds). The
        # curve's shape is still informative; a rho* read at a fine target is not.
        print(f"[rho-sweep] WARNING: single-decision problem -- feasibility is "
              f"quantized to 1/{len(folds)} = {1/len(folds):.2f}. Read the curve, "
              f"not rho*; raise --n-folds or use --problem gastric for a target "
              f"of {args.feas_target:g}.", flush=True)
    if args.match_bank:
        print("[rho-sweep] --match-bank: CP bank B set to the wrapper's P", flush=True)

    def score(tag, method, uset, knob):
        """One scored cell, resumable, with status + timings."""
        ckey = (tag, float(knob))
        if ckey not in ckpt:
            d = cv_score_knob(make_build(method, uset), knob, folds, oracle, inst,
                              constraint_names=cnames, contextual=contextual,
                              return_details=True)
            append_score(scores_path, tag, knob, d["feas"], d["obj"], d["solved"], d)
            ckpt[ckey] = d
        d = ckpt[ckey]
        cap = f" CAPPED({d['n_capped']}/{len(folds)})" if d.get("n_capped") else ""
        print(f"  {tag:<26s} knob={knob:<7g} feas={d['feas']:.3f} "
              f"obj={d['obj']:+.4f} solved={d['solved']:.3f} "
              f"master={d['master_time_s']:.1f}s "
              f"test/pt={d['test_time_per_point_s']:.2f}s "
              f"[{d['status']}]{cap}", flush=True)
        return d

    def row(d, method, **extra):
        return dict(problem=problem, method=method,
                    feasibility=d["feas"], objective=d["obj"],
                    solved_frac=d["solved"], status=d["status"],
                    n_capped=d["n_capped"],
                    master_time_s=d["master_time_s"],
                    test_time_s=d["test_time_s"],
                    test_time_per_point_s=d["test_time_per_point_s"],
                    coherent=bool(args.coherent),
                    matched_bank=bool(args.match_bank), **extra)

    rows = []
    for rho in grid:
        uset = dataclasses.replace(base_uset, rho=rho)
        for method in methods:
            # robust_reg's label_eps IS the D radius -- it must track rho, or its
            # adversary trains against a different set than the bank draws from.
            knob = rho if method == "robust_reg" else knobs.get(method, 0.0)
            d = score(f"{method}@rho={rho:g}", method, uset, knob)
            rows.append(row(d, method, rho=rho, knob=knob, phase="rho_sweep"))

    df = pd.DataFrame(rows)
    tidy = os.path.join(OUT_DIR, f"{problem}_rho_curve{var}.csv")
    df.to_csv(tidy, index=False)
    print(f"\n[rho-sweep] wrote {tidy}", flush=True)
    star = _rho_star(df, problem, float(args.feas_target), oracle.objective_sense,
                     float(args.min_solved), out_suffix=var)

    # ---- ablations, at ONE chosen rho -------------------------------------
    # Deliberately not swept jointly with rho: tau and alpha are each method's own
    # dial, held fixed for the shared-D comparison above. The ablation exists to
    # show the fixed value was not cherry-picked, which only needs one rho.
    if args.ablate:
        rho_a = args.ablate_rho
        if rho_a is None:
            rho_a = _pick_ablation_rho(star, grid)
        uset_a = dataclasses.replace(base_uset, rho=float(rho_a))
        print(f"\n[rho-sweep] ablations at rho={rho_a:g}", flush=True)
        ab = []
        for tau in (args.tau_grid or DEFAULT_TAU_GRID):
            if "cp" not in methods:
                break
            d = score(f"cp@rho={rho_a:g}@tau={tau:g}", "cp", uset_a, tau)
            ab.append(row(d, "cp", rho=rho_a, knob=tau, phase="tau_ablation"))
        for alpha in (args.alpha_grid or DEFAULT_ALPHA_GRID):
            if "wrapper" not in methods:
                break
            d = score(f"wrapper@rho={rho_a:g}@alpha={alpha:g}", "wrapper",
                      uset_a, alpha)
            ab.append(row(d, "wrapper", rho=rho_a, knob=alpha,
                          phase="alpha_ablation"))
        if ab:
            adf = pd.DataFrame(ab)
            apath = os.path.join(OUT_DIR, f"{problem}_ablations{var}.csv")
            adf.to_csv(apath, index=False)
            print(f"[rho-sweep] wrote {apath}", flush=True)
            df = pd.concat([df, adf], ignore_index=True)
    return df


def _rho_star_from_csv(problem, target, min_solved, sense="min",
                       exclude_capped=False, out_suffix="", variant=""):
    """Recompute rho* from a SAVED curve, with no solving.

    The point of writing the full per-cell curve is that the rho* criteria are a
    reporting choice, not a modelling one -- the target, the solved-fraction
    floor, and whether to trust capped cells can all be revisited without paying
    for the sweep again. Every column those choices need travels in
    ``{problem}_rho_curve.csv``: feasibility, objective, solved_frac, status,
    n_capped, both timing columns, rho, knob, coherent, matched_bank, phase.

    ``out_suffix`` names the output so several criteria can coexist rather than
    overwriting one another. ``variant`` selects WHICH curve to read: the coherence
    and match-bank flags must match the sweep that produced it, since each cell now
    writes its own curve (see :func:`_variant_suffix`).
    """
    path = os.path.join(OUT_DIR, f"{problem}_rho_curve{variant}.csv")
    if not os.path.exists(path):
        avail = sorted(f for f in os.listdir(OUT_DIR)
                       if f.startswith(f"{problem}_rho_curve")) \
            if os.path.isdir(OUT_DIR) else []
        raise SystemExit(
            f"no {path}; run the sweep first. Curves present: {avail or 'none'} "
            f"(pass the same --coherent/--incoherent and --match-bank flags the "
            f"sweep used)")
    df = pd.read_csv(path)
    df = df[df.get("phase", "rho_sweep") == "rho_sweep"]
    if exclude_capped:
        df = df[df["n_capped"] == 0]
    return _rho_star(df, problem, target, sense, min_solved,
                     out_suffix=variant + out_suffix)


def _pick_ablation_rho(star, grid):
    """Default ablation rho: the median rho* across methods, else the grid median.

    A rho where the methods actually differ is more informative than an endpoint,
    and taking it from the sweep's own answer avoids inventing a third number.
    """
    if star is not None and "rho_star" in star:
        vals = [v for v in star["rho_star"].tolist() if v == v]   # drop NaN
        if vals:
            return float(np.median(vals))
    return float(np.median(grid))


def _fixed_knobs(problem, args, config):
    """Each method's own dial, held fixed across the sweep.

    Read from **config.yaml**, not from ``*_robustness_knobs.json``. Under the rho
    parameterization tau and alpha are fixed CONSTANTS with their own ablations --
    the thing being calibrated is rho -- so the stage-1 CV selections are no longer
    the right source. Worse, they would be silently mis-scaled: those tau values
    were selected under ``tolerance_basis: "d0"``, where tau is a fraction of a
    bank quantile, while tau is now a multiple of the label scale. tau=0.1 means
    two different things in the two regimes.

    ``--knobs-from-cv`` restores the old behaviour for reproducing prior runs;
    ``--knobs`` overrides everything explicitly.

    Whatever the source, the pair must not sit at the degenerate corner: tau -> 0
    with alpha = 0 makes CP identical to the wrapper (the alpha=0 / tau->0
    equivalence), collapsing two of the three methods into one solver and leaving
    nothing to compare on decisions.
    """
    if args.knobs:
        out = {k: float(v) for k, v in
               (kv.split("=") for kv in args.knobs.split(","))}
        out.setdefault("nominal", 0.0)
        _warn_if_degenerate(out)
        return out

    if args.knobs_from_cv:
        path = f"results/cv/{problem}_robustness_knobs.json"
        if not os.path.exists(path):
            raise SystemExit(f"no {path}; drop --knobs-from-cv or pass --knobs")
        raw = json.load(open(path))
        out = {}
        for m in ("cp", "wrapper", "robust_reg"):
            v = lookup_knob(raw, m, bool(args.coherent))
            if v is not None:
                out[m] = float(v)
        out["nominal"] = 0.0
        print("[rho-sweep] WARNING: --knobs-from-cv reads tau selected under the "
              "d0 basis; it is NOT the same quantity as tau under "
              "tolerance_basis='scale'.", flush=True)
        _warn_if_degenerate(out)
        return out

    methods_cfg = (config or {}).get("methods", {}) or {}
    out = {
        "nominal": 0.0,
        "cp": float((methods_cfg.get("cp", {}) or {}).get("dist_tol_rel", 0.01)),
        "wrapper": float((methods_cfg.get("wrapper", {}) or {}).get("alpha", 0.2)),
        # robust_reg's knob tracks rho and is overwritten per cell; the value here
        # is never used, but is kept so the printed dict is not misleadingly empty.
        "robust_reg": float("nan"),
    }
    _warn_if_degenerate(out)
    return out


def _warn_if_degenerate(knobs, tau_tol=1e-6, alpha_tol=1e-9):
    """CP at tau->0 and the wrapper at alpha=0 are the SAME solver on a shared
    bank prefix -- the alpha=0 / tau->0 equivalence. Fixing both there collapses
    two of the three methods into one and leaves nothing to compare on decisions
    (only MIP size). Warn rather than refuse: the collapse is a legitimate thing
    to measure deliberately, just never by accident.
    """
    tau, alpha = knobs.get("cp"), knobs.get("wrapper")
    if alpha is not None and alpha <= alpha_tol:
        msg = (f"wrapper alpha is {alpha:g} -- its strongest setting, where the "
               f"wrapper requires ALL P models to satisfy the constraint.")
        if tau is not None and tau <= tau_tol:
            msg += (f" CP tau is also {tau:g}: at B=P these are the SAME solver "
                    f"and the decision comparison is vacuous.")
        else:
            msg += (f" CP tau={tau:g} keeps them distinct, but the two sit close "
                    f"to the equivalence corner -- read any CP-vs-wrapper gap "
                    f"with that in mind.")
        print(f"[rho-sweep] WARNING: {msg}", flush=True)


def _rho_star(df, problem, target, sense, min_solved, out_suffix=""):
    """rho*(method): the LARGEST rho still meeting the feasibility target.

    Largest, not smallest -- rho is the assumed uncertainty, so a method that
    holds the target further out along the axis is absorbing more of it. Cells
    below ``min_solved`` are dropped first: conditional feasibility at a low
    solved fraction measures the survivors, not the method.
    """
    rows = []
    for method, g in df.groupby("method"):
        # Capped cells are KEPT. CP at max_iterations still returns a usable
        # incumbent, and dropping those cells silently discards data the reader may
        # want. n_capped travels with every row instead, so the caller can filter
        # after the fact -- see _rho_star_from_csv, which recomputes this table from
        # a saved curve under whatever criteria are chosen later.
        g = g[g["solved_frac"] >= min_solved].sort_values("rho")
        ok = g[g["feasibility"] >= target]
        if ok.empty:
            rows.append(dict(method=method, rho_star=np.nan, feasibility=np.nan,
                             objective=np.nan, solved_frac=np.nan, n_capped=0,
                             master_time_s=np.nan, test_time_per_point_s=np.nan,
                             note=f"never reaches feas>={target:g}"))
            continue
        best = ok.iloc[-1]
        # Censored if the grid ran out before the method did: rho* is a lower bound.
        censored = bool(np.isclose(best["rho"], g["rho"].max()))
        notes = []
        if censored:
            notes.append("censored at grid max")
        if int(best.get("n_capped", 0) or 0):
            notes.append(f"n_capped={int(best['n_capped'])} (incumbent, not converged)")
        rows.append(dict(method=method, rho_star=float(best["rho"]),
                         feasibility=float(best["feasibility"]),
                         objective=float(best["objective"]),
                         solved_frac=float(best["solved_frac"]),
                         n_capped=int(best.get("n_capped", 0) or 0),
                         master_time_s=float(best["master_time_s"]),
                         test_time_per_point_s=float(best["test_time_per_point_s"]),
                         note="; ".join(notes)))
    out = pd.DataFrame(rows).sort_values("rho_star", ascending=False)
    out.insert(0, "feas_target", target)
    out.insert(1, "min_solved", min_solved)
    path = os.path.join(OUT_DIR, f"{problem}_rho_star{out_suffix}.csv")
    out.to_csv(path, index=False)
    print(f"\n[rho-sweep] rho*(method) at feasibility >= {target:g} "
          f"(objective sense: {sense}, min solved_frac {min_solved:g})")
    print(out.to_string(index=False))
    print(f"[rho-sweep] wrote {path}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "gastric"), default="synthetic")
    p.add_argument("--rho-grid", type=float, nargs="+", default=None,
                   help=f"rho values to sweep (default {DEFAULT_GRID})")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--knobs", default=None,
                   help="fixed dials, e.g. 'cp=0.01,wrapper=0.2'; default reads "
                        "methods.cp.dist_tol_rel and methods.wrapper.alpha from "
                        "config.yaml")
    p.add_argument("--knobs-from-cv", action="store_true",
                   help="read the fixed dials from "
                        "results/cv/{problem}_robustness_knobs.json instead. Those "
                        "tau values were selected under tolerance_basis='d0' and "
                        "are NOT the same quantity as tau under 'scale'")
    p.add_argument("--feas-target", type=float, default=0.9,
                   help="held-out feasibility target defining rho* (default 0.9)")
    p.add_argument("--min-solved", type=float, default=0.5,
                   help="drop cells whose solved_frac is below this before "
                        "reading rho* (default 0.5)")
    p.add_argument("--coherent", action="store_true", default=True)
    p.add_argument("--incoherent", dest="coherent", action="store_false")
    p.add_argument("--match-bank", action="store_true",
                   help="set CP's bank B to the wrapper's P, removing the B!=P "
                        "confound from the rho* comparison")
    p.add_argument("--ablate", action="store_true",
                   help="after the sweep, run the tau and alpha ablations at one "
                        "chosen rho (default: median rho* across methods)")
    p.add_argument("--ablate-rho", type=float, default=None,
                   help="rho to run the ablations at; overrides the default pick")
    p.add_argument("--tau-grid", type=float, nargs="+", default=None,
                   help=f"CP tau ablation grid (default {DEFAULT_TAU_GRID}); "
                        "tau is in unexplained-sd units")
    p.add_argument("--alpha-grid", type=float, nargs="+", default=None,
                   help=f"wrapper alpha ablation grid (default {DEFAULT_ALPHA_GRID}); "
                        "0/0.1/0.2/0.5 are OptiCL's published WFP values")
    p.add_argument("--n-folds", type=int, default=None,
                   help="synthetic only: KFold splits. Held-out feasibility is "
                        "quantized to 1/n_folds there (one solve per fold), so "
                        "raise this before reading rho* off synthetic")
    p.add_argument("--rho-star-only", action="store_true",
                   help="re-derive rho* from the SAVED {problem}_rho_curve.csv "
                        "under new criteria; no solving. Combine with "
                        "--feas-target / --min-solved / --exclude-capped / "
                        "--out-suffix, and pass the same --coherent/--incoherent "
                        "and --match-bank flags the sweep used (they select which "
                        "curve is read)")
    p.add_argument("--exclude-capped", action="store_true",
                   help="drop cells that hit max_iterations when deriving rho* "
                        "(kept by default; n_capped travels with every row)")
    p.add_argument("--sense", choices=("min", "max"), default="min",
                   help="objective sense, --rho-star-only only (the sweep reads "
                        "it off the oracle)")
    p.add_argument("--out-suffix", default="",
                   help="EXTRA suffix for the rho_star output, appended after the "
                        "cell suffix, so several CRITERIA can coexist within one "
                        "cell, e.g. --out-suffix _t080 (--rho-star-only)")
    p.add_argument("--refresh", action="store_true", help="discard the score cache")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    args = p.parse_args()

    if args.rho_star_only:
        # Pure post-processing: re-derive rho* from the saved curve under new
        # criteria. No instance, no oracle, no solving.
        _rho_star_from_csv(args.problem, float(args.feas_target),
                           float(args.min_solved), sense=args.sense,
                           exclude_capped=args.exclude_capped,
                           out_suffix=args.out_suffix,
                           variant=_variant_suffix(args))
        return

    import yaml
    config = yaml.safe_load(open(args.config))
    run_sweep(config, args)


if __name__ == "__main__":
    main()

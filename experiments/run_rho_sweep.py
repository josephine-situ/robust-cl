"""Global PARAMETER sweep: every method walked along its own conservatism dial.

Each method is swept over the parameter that actually makes it conservative
(``SWEEP_PARAM``), not over one quantity imposed on all of them:

  ``rho``     cp, wrapper, robust_reg -- the radius of the shared ellipsoidal D
              (``R_c = rho * scale(y_c) * sqrt(n)``, see
              :mod:`src.methods.uncertainty`). **Within one grid value these
              three still face the SAME D**, which is what the shared-D
              comparison rests on. Only robust_reg's own dial tracks the axis;
              CP's tau and the wrapper's alpha stay fixed and are ablated.
  ``margin``  margin -- the RHS shift ``m`` in ``f(x) <= b - m*scale(y_c)``. It
              builds no D, so sweeping rho for it would re-measure one number.
  ``None``    nominal, cmicl -- no conservatism parameter. nominal has none by
              definition; cmicl's alpha is pinned to ``1 - feas_target`` at
              evaluation rather than chosen here. Each is scored ONCE and
              repeated across the axis as a reference level.

**One grid, because the units are shared.** rho, tau and the margin are all in
unexplained standard deviations, so the same number means the same size of
assumption for each method. What differs is what that assumption IS: for
cp/wrapper/robust_reg a point on the axis is an assumed radius, for margin it is
a tightening the optimizer simply pays for. Reading a margin curve as a "rho" is
the one mistake this file is arranged to prevent -- hence the ``param_swept``
column on every row. ``--margin-grid`` gives the margin its own values when the
useful range of a direct RHS shift differs from that of an assumed radius.

The parameter is **swept and reported, not fitted**, and the derived
param*(method) is what the EVALUATION run uses -- each method at its own -- so
evaluation matches held-out feasibility rather than D. The criterion is what
keeps that honest: held-out feasibility on training folds. Never fit rho against
the GT ensemble (it tunes to the judge) or against synthetic's known
``noise_std`` (it calibrates D to the data-generating process, which CP would
then win by construction).

**What param* means differs by method, and the table says which.** For a
rho-swept method it is capacity: the largest assumed uncertainty it still
delivers under. For margin it is price: the largest RHS shift that still solves
on enough contexts. Both are read by the same rule from the same curve, and the
comparison that matters is not param* against param* but **objective at equal
feasibility** -- which is why the curve, not the star table, is the primary
output.

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
     objective and in time. Read off the sweep at a fixed feasibility target, not
     fitted -- and it is what the EVALUATION run uses: each method is evaluated at
     its own rho*, so D is shared across methods on the curve above but NOT at
     evaluation, where the match is on held-out feasibility instead.

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
- **C-MICL is NOT on this axis and is not ablated here.** It faces no D, so every
  cell of a rho curve would re-measure the same number, and its alpha is not a
  dial to be chosen either: it is FIXED at ``1 - feas_target`` by definition,
  because the conformal level and the feasibility target are the same quantity
  (see ``methods.cmicl.alpha`` in config.yaml). It therefore enters at
  EVALUATION only, at the target this sweep's rho* is read at -- the other
  methods search for the rho that delivers 0.9, C-MICL asserts alpha=0.1 and is
  scored on whether it does. Passing ``--methods ... cmicl`` still works and is
  occasionally useful as a flat reference line, but it is not part of the
  protocol.
- **The margin baseline is a reference line too, and the one to beat.**
  ``margin`` is nominal against a tightened RHS (``rhs - m * scale(y_c)``, one
  dimensionless dial for the whole problem), so like C-MICL it faces no D and is
  flat in rho. It is here because it is the cheapest thing that buys feasibility:
  a shared-D curve that does not sit above it at equal feasibility is paying for
  machinery that a one-line RHS shift delivers. Its own axis is the margin,
  swept by ``--ablate``, and it is monotone -- an ``m*`` at any target exists,
  which is exactly what robust_reg's falling gastric curve does not give.
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
# C-MICL has no grid here on purpose: its alpha is pinned to 1 - feas_target
# rather than chosen, so there is nothing to ablate. See the module docstring.
# The margin baseline's dial, in unexplained-sd units -- the SAME units as rho,
# so this deliberately mirrors DEFAULT_GRID. A margin of m and a rho of m both
# mean "one m-th of an unexplained sd", and running them on one grid is what
# makes "CP at rho=0.75 vs a 0.75-sd RHS shift" a sentence worth writing.
DEFAULT_MARGIN_GRID = list(DEFAULT_GRID)
OUT_DIR = "results/rho_sweep"

# Which parameter the sweep moves for each method. This is what makes the run a
# PARAMETER sweep rather than a rho sweep: every method is walked along its own
# conservatism dial, and the shared grid is read in the units they have in common
# (unexplained sds), NOT as one physical set D imposed on all of them.
#
#   rho     cp / wrapper / robust_reg -- the radius of the shared D. Within one
#           grid value these three still face the SAME set, which is what the
#           shared-D comparison rests on; only robust_reg's own dial tracks it.
#   margin  margin -- the RHS shift m. It builds no D at all, so sweeping rho for
#           it would re-measure one number; its dial IS the axis.
#   None    nominal / cmicl -- no conservatism parameter to move. nominal has
#           none by definition; cmicl's alpha is pinned to 1 - feas_target at
#           evaluation rather than chosen here. Scored ONCE, then repeated across
#           the axis as a reference line.
SWEEP_PARAM = {
    "cp": "rho", "wrapper": "rho", "robust_reg": "rho",
    "margin": "margin",
    "nominal": None, "cmicl": None,
}


def sweep_param(method):
    """The parameter ``method`` is swept over, or ``None`` if it has none."""
    return SWEEP_PARAM.get(method, "rho")


# ---------------------------------------------------------------------------
# Per-problem setup: instance, folds, oracle, and build(knob) -> solver
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Setup:
    """What a ``_setup_*`` hands back: everything a sweep needs that is
    problem-specific, in one object rather than a positional tuple.

    ``run_dial_sweep.py`` reads the SAME setups -- there is one place per problem
    where an instance, its folds and its oracle are built, and neither sweep may
    drift from the other on any of the three, or a dial curve and a rho curve
    would not be readable against each other.

    ``model_spec`` is the ``(model_type, model_params)`` DEFAULT pair, the one
    ``resolve_constraint_config`` falls back to per constraint. It is here so a
    caller can build the scenario bank OUTSIDE the solvers and share it across a
    dial grid (:func:`src.methods.uncertainty.build_bank_for_instance`); the rho
    sweep does not use it, because each of its cells has its own D anyway.
    """
    instance: object
    folds: object
    oracle: object
    make_build: object
    constraint_names: object
    contextual: bool
    model_spec: tuple = (None, None)
    config: object = None
def _bank_seed(config, args):
    """Seed for the METHODS' own randomness, holding the data and the folds fixed.

    D is sampled, not enumerated: CP cuts against B=200 vertices and the wrapper
    embeds P=20, so a curve read off one bank confounds "this method absorbs more
    rho" with "this bank happened to miss the direction that breaks it". Repeating
    the sweep at several seeds is what turns a single curve into a spread.

    It reaches everything downstream of the training rows: the ``ScenarioBank``
    draws AND the ``random_state`` of every model fit. The second one is not a
    side effect to be engineered away -- it moves the out-of-fold residual sd, so
    ``R_c = rho * scale(y_c) * sqrt(n)`` wobbles a few percent between seeds
    (measured on synthetic, fold 1: ``oof_sd`` 0.1314 at seed 7 vs 0.1238 at seed
    42, 5.8%). That is the estimation noise a single-seed curve hides, and D stays
    shared ACROSS METHODS within a seed, which is what the comparison needs.

    It is deliberately NOT ``config["uncertainty"]["bootstrap_seed"]``, which also
    seeds ``synthetic_nonlinear`` (the DATA) and the synthetic KFold split. Moving
    that would resample the problem and the evaluation folds too, and the sources
    of variation could not be told apart afterwards. Here the instance and the
    folds stay bit-identical across seeds.
    """
    s = getattr(args, "seed", None)
    return int(s) if s is not None else int(
        config["uncertainty"].get("bootstrap_seed", 42))


def _synth_n_folds(config, args):
    """KFold splits for the synthetic sweep: --n-folds, else cv_calibration.n_kfold.

    Both single-decision problems (synthetic, reactor) solve ONCE per fold and yield
    ONE binary feasibility outcome, so held-out feasibility is QUANTIZED TO
    1/n_folds and the fold count is the resolution of the whole curve. The default
    is ``cv_calibration.n_kfold`` = 10, which is deliberately NOT run_cv.py's 5:
    model selection scores R^2 over rows, where 5 folds are ample, while this scores
    one bit per fold. At 10, feasibility takes 0, 0.1, ..., 1.0 and the 0.9 rho*
    target is exactly representable (2026-08-19 deck, next step 3).
    """
    return int(args.n_folds or
               config.get("cv_calibration", {}).get("n_kfold", 10))


def _setup_synthetic(config, args):
    from experiments.run_sweep import _synth_instance, _synth_build, synth_model_spec

    seed = config["uncertainty"].get("bootstrap_seed", 42)   # data + folds
    bank_seed = _bank_seed(config, args)                     # draws from D
    # The embedded model is a CV result whenever
    # results/cv/synthetic_selected_configs.json exists (run_cv.py --problem
    # synthetic), and config.yaml's hard-coded rf otherwise. Which one is in force
    # scopes the cell name -- see _variant_suffix.
    model_type, model_params, _from_cv = synth_model_spec(config, verbose=True)
    inst = _synth_instance(config)
    n_kfold = _synth_n_folds(config, args)
    folds = make_folds(inst, "kfold", n_kfold=n_kfold, seed=seed)
    # Mixed-type ensemble on the noisy training labels -- deliberately NOT the
    # class the candidate embeds, which is what made the synthetic judge share its
    # approximation error with the thing it judged.
    oracle = make_cv_oracle(inst)

    def make_build(method, uset, cp_alpha=None):
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
        return _synth_build(method, cfg, model_type, model_params, bank_seed,
                            cp_alpha=cp_alpha)

    return Setup(inst, folds, oracle, make_build, None, False,
                 (model_type, model_params), config)


def _setup_reactor(config, args):
    """The C-MICL DMA-MR instance on the rho axis.

    Mirrors ``_setup_synthetic`` -- single-decision, KFold folds, one learned
    constraint -- and reuses ``_synth_build`` verbatim, because the method builders
    read the instance for everything problem-specific (the constraint's sign, the
    domain constraints, the variable box) and take only the model type and the
    uncertainty set as arguments.

    The oracle here is the PROXY ensemble, not the ODEs. Tuning rho against the
    exact truth would calibrate D to the thing it is scored by; the ODE oracle
    (``cv_calibrate.make_gt_oracle``) is for final evaluation and for auditing this
    proxy, which is the one thing this instance can do that gastric cannot.
    """
    from experiments.run_sweep import (
        _reactor_instance, _synth_build, reactor_model_spec,
    )

    seed = config["uncertainty"].get("bootstrap_seed", 42)   # data + folds
    bank_seed = _bank_seed(config, args)                     # draws from D
    model_type, model_params, _from_cv = reactor_model_spec(config, verbose=True)
    inst = _reactor_instance(config)
    folds = make_folds(inst, "kfold", n_kfold=_synth_n_folds(config, args), seed=seed)
    oracle = make_cv_oracle(inst)

    def make_build(method, uset, cp_alpha=None):
        cfg = json.loads(json.dumps(config))          # deep copy, config is plain data
        cfg["uncertainty"].update(
            geometry=uset.geometry, rho=uset.rho,
            coherent=uset.coherent,
            coherent_exclude=list(uset.coherent_exclude),
        )
        if args.match_bank:
            cfg.setdefault("methods", {}).setdefault("cp", {})["n_scenarios"] = \
                int(cfg["uncertainty"].get("n_bootstrap", 20))
        return _synth_build(method, cfg, model_type, model_params, bank_seed,
                            cp_alpha=cp_alpha)

    return Setup(inst, folds, oracle, make_build, None, False,
                 (model_type, model_params), config)


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

    bank_seed = _bank_seed(config, args)

    def make_build(method, uset, cp_alpha=None):
        cell = dict(settings)
        cell["uncertainty_set"] = uset
        # Bank seed only: `folds` above were already built from the config seed
        # (and gastric's scheme is temporal, so they do not move at all).
        cell["bootstrap_seed"] = bank_seed
        if getattr(args, "separation", None):
            cell["cp_separation"] = args.separation
        if args.match_bank:
            cell["cp_n_scenarios"] = int(config["uncertainty"].get("n_bootstrap", 20))
        # None -> the pinned 0. Only the coverage-cap ablation ever sets it.
        cell["cp_alpha"] = cp_alpha
        build, _ = _method_build_map(method, cell, ranges,
                                     config["default_model"]["type"],
                                     config["default_model"]["params"], None, None)
        return build

    return Setup(inst, folds, oracle, make_build, ALL_CONSTRAINTS,
                 bool(inst.context_var_indices),
                 (config["default_model"]["type"],
                  config["default_model"]["params"]), config)


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

    ``--n-folds`` scopes the cell for the same reason. The resume key is
    ``(method@rho, knob)`` and carries no fold count, so a 10-fold synthetic run
    would resume -- and then report as its own -- rows scored on 4 folds. The fold
    count changes what feasibility MEANS on the single-decision problem (it is
    quantized to 1/n_folds), so those are different experiments. Any explicit
    ``--n-folds`` therefore gets its own suffix, including one that happens to
    equal the config default: a redundant file is recoverable, a silently merged
    checkpoint is not.

    ``--seed`` scopes it for the third time, and there a shared file would fail
    worse than in either case above: repeated seeds exist precisely to be compared,
    so one checkpoint would hand every seed the FIRST seed's rows and the spread
    would read as exactly zero -- a bank-variance study able only to report "no
    variance". Seeds never share a file.

The single-decision problems (``--problem synthetic``, ``--problem reactor``)
    scope it a fourth time, with ``_m<model_type>``. The embedded model there is no
    longer fixed: it is the CV winner when the problem's
    ``results/cv/*_selected_configs.json`` exists and ``config.yaml``'s block
    otherwise, and those two train different constraint models on identical data. The resume key carries no model, so without the token a post-CV
    run would resume, and then re-report, the pre-CV rf's rows. The token is
    written in BOTH cases, the fallback included, so no synthetic cell shares a
    name with a curve produced before the model was a choice at all -- and the
    synthetic CV oracle changed in the same commit (one rf -> a six-class
    ensemble), so those older curves are not comparable either.
    """
    n = getattr(args, "n_folds", None)
    seed = getattr(args, "seed", None)
    model = getattr(args, "synth_model", None)
    # CP's separation path normally follows the coherence the cell is already named
    # for (coherent bank -> coherent separation, incoherent -> per-constraint), so
    # `_coh`/`_incoh` scopes it with no extra token -- which is why every committed
    # `_coh` curve still reproduces. A FORCED mismatch does need one: the two paths
    # cut different adversaries and measure tau on different statistics (mean over
    # units vs per unit), and the resume key carries neither, so a shared
    # checkpoint would hand the forced run the auto run's rows.
    sep = getattr(args, "separation", None)
    auto_sep = "coherent" if getattr(args, "coherent", False) else "incoherent"
    sep_token = f"_sep{sep[:5]}" if (sep and sep != "auto" and sep != auto_sep) else ""
    return (("_coh" if args.coherent else "_incoh")
            + sep_token
            + ("_matchbank" if getattr(args, "match_bank", False) else "")
            + (f"_f{int(n)}" if n else "")
            + (f"_m{model}" if model else "")
            + (f"_s{int(seed)}" if seed is not None else ""))


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
    setup = {"synthetic": _setup_synthetic, "reactor": _setup_reactor,
             "gastric": _setup_gastric}[problem]
    su = setup(config, args)
    inst, folds, oracle, make_build = su.instance, su.folds, su.oracle, su.make_build
    cnames, contextual = su.constraint_names, su.contextual

    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )
    knobs = _fixed_knobs(problem, args, config)
    bank_seed = _bank_seed(config, args)
    grid = [float(r) for r in (args.rho_grid or DEFAULT_GRID)]
    margin_grid = [float(m) for m in (args.margin_grid or grid)]
    methods = args.methods or ["nominal", "cp", "wrapper", "robust_reg"]

    def method_grid(method):
        """The values THIS method is swept over.

        One shared grid by default: rho, tau and the margin are all in unexplained-sd
        units, so the same numbers mean the same size of assumption for each. The
        margin can be given its own (``--margin-grid``) when the useful range of a
        direct RHS shift differs from the useful range of an assumed radius.
        """
        return margin_grid if sweep_param(method) == "margin" else grid

    var = _variant_suffix(args)
    scores_path = os.path.join(OUT_DIR, f"{problem}_rho_sweep{var}.csv")
    if args.refresh and os.path.exists(scores_path):
        os.remove(scores_path)
    ckpt = load_detail_checkpoint(scores_path) if not args.refresh else {}

    print(f"[rho-sweep] problem={problem} geometry=ellipsoid "
          f"coherent={args.coherent} folds={len(folds)} bank_seed={bank_seed} "
          f"sense={oracle.objective_sense}", flush=True)
    print(f"[rho-sweep] grid={grid}", flush=True)
    if margin_grid != grid and "margin" in methods:
        print(f"[rho-sweep] margin grid={margin_grid}", flush=True)
    print("[rho-sweep] swept parameter: "
          + ", ".join(f"{m}={sweep_param(m) or 'none (reference level)'}"
                      for m in methods), flush=True)
    print(f"[rho-sweep] fixed knobs: {knobs}", flush=True)
    if not contextual:
        # Single-decision problems solve ONCE per fold, so held-out feasibility is
        # a mean of len(folds) binary outcomes -- quantized to 1/len(folds). The
        # curve's shape is still informative; a rho* read at a fine target is not.
        print(f"[rho-sweep] WARNING: single-decision problem -- feasibility is "
              f"quantized to 1/{len(folds)} = {1/len(folds):.2f}. Read the curve, "
              f"not rho*; raise --n-folds or use --problem gastric for a target "
              f"of {args.feas_target:g}.", flush=True)
    if "margin" in methods:
        # Swept on its OWN parameter, so its curve is a real curve -- but the axis
        # under it is not the same physical quantity as the other methods'.
        print(f"[rho-sweep] NOTE: margin is swept on its RHS margin m, not on rho "
              f"({len(margin_grid)} values) -- it builds no D. Its curve is a "
              f"conservatism curve like the others and in the same unexplained-sd "
              f"units, but a point on it is a TIGHTENING, not an assumed radius. "
              f"m* is the baseline a shared-D rho* must beat at equal feasibility.",
              flush=True)
    if "cmicl" in methods:
        # cmicl is not part of the sweep protocol -- it belongs to the evaluation,
        # at alpha = 1 - feas_target. Passed explicitly it still runs, as a flat
        # reference line; say so rather than let a flat curve look like a result.
        print(f"[rho-sweep] NOTE: cmicl is not a rho method -- it faces no D, so "
              f"this curve is FLAT by construction and every cell re-measures the "
              f"same number. Its alpha is pinned to 1 - feas_target "
              f"({1 - float(args.feas_target):g}) at evaluation, not chosen here.",
              flush=True)
    if args.match_bank:
        print("[rho-sweep] --match-bank: CP bank B set to the wrapper's P", flush=True)

    def score(tag, method, uset, knob):
        """One scored cell, resumable, with status + timings."""
        ckey = (tag, float(knob))
        cell = f"{tag} knob={knob:.6g}"
        if ckey not in ckpt:
            # Same reason as run_dial_sweep.score: the summary below is printed
            # after the cell finishes, so the solver output in between has no
            # marker saying which cell (or which fold) produced it.
            print(f"\n[cell] BEGIN {cell}  [{len(folds)} folds]", flush=True)
            d = cv_score_knob(make_build(method, uset), knob, folds, oracle, inst,
                              constraint_names=cnames, contextual=contextual,
                              return_details=True, label=cell)
            append_score(scores_path, tag, knob, d["feas"], d["obj"], d["solved"], d)
            ckpt[ckey] = d
        else:
            print(f"\n[cell] RESUMED {cell} (from the score checkpoint)", flush=True)
        d = ckpt[ckey]
        cap = f" CAPPED({d['n_capped']}/{len(folds)})" if d.get("n_capped") else ""
        print(f"[cell] END   {tag:<26s} knob={knob:<7g} feas={d['feas']:.3f} "
              f"obj={d['obj']:+.4f} solved={d['solved']:.3f} "
              f"master={d['master_time_s']:.1f}s "
              f"test/pt={d['test_time_per_point_s']:.2f}s "
              f"[{d['status']}]{cap}", flush=True)
        return d

    def row(d, method, **extra):
        # `param`/`param_swept` say WHAT the axis value means for this method;
        # `rho` is kept as the axis column so saved curves, pool_rho_seeds.py and
        # the figures keep reading the same name. For a margin row `rho` is the
        # margin, which is why param_swept must be consulted before calling any
        # axis value a radius.
        extra.setdefault("param", extra.get("rho"))
        extra.setdefault("param_swept", sweep_param(method) or "none")
        return dict(problem=problem, method=method,
                    feasibility=d["feas"], objective=d["obj"],
                    solved_frac=d["solved"], status=d["status"],
                    n_capped=d["n_capped"],
                    master_time_s=d["master_time_s"],
                    test_time_s=d["test_time_s"],
                    test_time_per_point_s=d["test_time_per_point_s"],
                    coherent=bool(args.coherent),
                    matched_bank=bool(args.match_bank),
                    seed=bank_seed, **extra)

    rows = []
    for method in methods:
        pname = sweep_param(method)
        if pname is None:
            # No swept parameter: score ONCE, then emit the same row at every grid
            # point so the figure keeps its horizontal reference line. Before this
            # the cell was re-SOLVED at each rho for an identical answer -- 7
            # nominal solves per sweep. The repeat is explicit in param_swept.
            d = score(method, method, base_uset, knobs.get(method, 0.0))
            for p_val in grid:
                rows.append(row(d, method, rho=p_val, param=np.nan,
                                param_swept="none",
                                knob=knobs.get(method, 0.0), phase="param_sweep"))
            continue
        for p_val in method_grid(method):
            if pname == "rho":
                # cp / wrapper / robust_reg move D itself. robust_reg's label_eps
                # IS the radius, so its dial tracks the axis; cp's tau and the
                # wrapper's alpha stay fixed at their own values.
                uset = dataclasses.replace(base_uset, rho=p_val)
                knob = p_val if method == "robust_reg" else knobs.get(method, 0.0)
            else:
                # margin: the axis IS its dial, and it never builds a D.
                uset, knob = base_uset, p_val
            d = score(f"{method}@{pname}={p_val:g}", method, uset, knob)
            rows.append(row(d, method, rho=p_val, param=p_val, param_swept=pname,
                            knob=knob, phase="param_sweep"))

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
        # No margin ablation: the margin IS the main axis now, so an ablation of
        # it would re-run the sweep. m* is read off the sweep curve, like rho*.
        # No C-MICL ablation: its alpha is pinned to 1 - feas_target, not chosen,
        # so there is no dial to show was uncherry-picked.
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
    # "param_sweep" is the current name; "rho_sweep" is what curves written before
    # the sweep became per-method parameter carry. Both are the main-curve rows, so
    # a saved curve from either era re-derives.
    df = df[df.get("phase", "param_sweep").isin(("param_sweep", "rho_sweep"))]
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
        st = star
        # Only the rho-swept methods. A margin m* is in the same units but is not
        # a radius, and the ablations being placed are CP's tau and the wrapper's
        # alpha -- both of which live at a point on the D axis.
        if "param_swept" in st:
            st = st[st["param_swept"] == "rho"]
        vals = [v for v in st["rho_star"].tolist() if v == v]   # drop NaN
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
        # C-MICL's dial. Unlike the other three it is NOT a setting on D -- see
        # the flat-in-rho warning in run_sweep.
        "cmicl": float((methods_cfg.get("cmicl", {}) or {}).get("alpha", 0.1)),
        # The margin baseline's dial. Like cmicl's it is NOT a setting on D, so
        # the sweep holds it fixed and the ablation is where it actually moves.
        "margin": float((methods_cfg.get("margin", {}) or {}).get("margin", 0.5)),
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

    **What BOUNDED rho* is reported separately from rho* itself** (``bound``
    column). Three different things stop the search and they mean different things
    to a reader: the grid ran out (``grid_max`` -- rho* is a lower bound, the
    method was never pushed to its limit), feasibility fell below the target
    (``feasibility`` -- a real limit of the method), or the next rho up was dropped
    by the solved-fraction floor (``solved_floor`` -- the method still met the
    feasibility target there but on too few survivors to count). Censoring is
    tested against the FULL grid max, not the post-filter maximum: testing after
    the filter labels a solved-floor bound as "censored at grid max", which is the
    opposite claim (that the axis ran out rather than that the method ran out).
    """
    rows = []
    for method, g_all in df.groupby("method"):
        g_all = g_all.sort_values("rho")
        # A method with no swept parameter has no param*: its rows are one scored
        # cell repeated along the axis. Reporting "grid_max, censored" for it would
        # assert it absorbed the whole grid, the opposite of the truth -- it was
        # never moved. Its feasibility is still carried, as the reference level.
        swept = str(g_all["param_swept"].iloc[0]) if "param_swept" in g_all else "rho"
        if swept == "none":
            b = g_all.iloc[-1]
            rows.append(dict(method=method, param_swept=swept, rho_star=np.nan,
                             feasibility=float(b["feasibility"]),
                             objective=float(b["objective"]),
                             solved_frac=float(b["solved_frac"]),
                             n_capped=int(b.get("n_capped", 0) or 0),
                             master_time_s=float(b["master_time_s"]),
                             test_time_per_point_s=float(b["test_time_per_point_s"]),
                             bound="no_param",
                             note="no swept parameter; constant reference level"))
            continue
        # Censoring is a property of the GRID, so the max comes from the unfiltered
        # group -- before solved_frac drops anything.
        grid_max = float(g_all["rho"].max())
        # Capped cells are KEPT. CP at max_iterations still returns a usable
        # incumbent, and dropping those cells silently discards data the reader may
        # want. n_capped travels with every row instead, so the caller can filter
        # after the fact -- see _rho_star_from_csv, which recomputes this table from
        # a saved curve under whatever criteria are chosen later.
        g = g_all[g_all["solved_frac"] >= min_solved]
        ok = g[g["feasibility"] >= target]
        if ok.empty:
            rows.append(dict(method=method, param_swept=swept,
                             rho_star=np.nan, feasibility=np.nan,
                             objective=np.nan, solved_frac=np.nan, n_capped=0,
                             master_time_s=np.nan, test_time_per_point_s=np.nan,
                             bound="none", note=f"never reaches feas>={target:g}"))
            continue
        best = ok.iloc[-1]
        rho_b = float(best["rho"])
        notes = []
        if np.isclose(rho_b, grid_max):
            # The grid ran out before the method did: rho* is a lower bound.
            bound = "grid_max"
            notes.append("censored at grid max")
        else:
            # Something above rho_b disqualified. Name WHICH, from the unfiltered
            # rows -- a cell dropped by the solved floor is a different statement
            # about the method than one whose feasibility fell short.
            above = g_all[g_all["rho"] > rho_b]
            by_floor = bool((above["solved_frac"] < min_solved).any())
            by_feas = bool(((above["solved_frac"] >= min_solved) &
                            (above["feasibility"] < target)).any())
            if by_floor and not by_feas:
                bound = "solved_floor"
                notes.append(f"NOT grid-censored: solved_frac<{min_solved:g} "
                             f"above rho={rho_b:g}")
            elif by_feas and not by_floor:
                bound = "feasibility"
                notes.append(f"NOT grid-censored: feasibility<{target:g} "
                             f"above rho={rho_b:g}")
            else:
                bound = "mixed"
                notes.append(f"NOT grid-censored: solved floor and feasibility "
                             f"both bind above rho={rho_b:g}")
        if int(best.get("n_capped", 0) or 0):
            notes.append(f"n_capped={int(best['n_capped'])} (incumbent, not converged)")
        rows.append(dict(method=method, param_swept=swept,
                         rho_star=float(best["rho"]),
                         feasibility=float(best["feasibility"]),
                         objective=float(best["objective"]),
                         solved_frac=float(best["solved_frac"]),
                         n_capped=int(best.get("n_capped", 0) or 0),
                         master_time_s=float(best["master_time_s"]),
                         test_time_per_point_s=float(best["test_time_per_point_s"]),
                         bound=bound, note="; ".join(notes)))
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
    p.add_argument("--problem", choices=("synthetic", "reactor", "gastric"),
                   default="synthetic",
                   help="reactor is the C-MICL DMA-MR case study: the only "
                        "instance with a MECHANISTIC (ODE) ground truth")
    p.add_argument("--rho-grid", type=float, nargs="+", default=None,
                   help=f"values to sweep (default {DEFAULT_GRID}). Read as rho "
                        f"by cp/wrapper/robust_reg and as the margin m by margin "
                        f"unless --margin-grid overrides; all are in "
                        f"unexplained-sd units")
    p.add_argument("--methods", nargs="+", default=None,
                   help="default: nominal cp wrapper robust_reg. Add margin for "
                        "the feasibility-tuned nominal baseline -- it is swept on "
                        "its own dial m (it faces no D), so its curve is the one "
                        "a shared-D rho* has to beat. cmicl also runs if named, "
                        "but has no swept parameter: its alpha is pinned to "
                        "1 - feas_target, so it enters as a reference level")
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
    # Incoherent is the production cell since 2026-08-21 (config.yaml says why);
    # --coherent is the ablation. The cell suffix follows the flag either way, so
    # no curve changes meaning -- but the DEFAULT cell is now _incoh, and an
    # existing _coh checkpoint will not be resumed by a bare invocation.
    p.add_argument("--incoherent", dest="coherent", action="store_false",
                   default=False)
    p.add_argument("--coherent", dest="coherent", action="store_true")
    p.add_argument("--separation", dest="separation", default=None,
                   choices=("auto", "coherent", "incoherent"),
                   help="CP separation path. Default 'auto' follows the bank: a "
                        "coherent bank cuts one shared draw per iteration (where "
                        "the alpha=0 == tau->0 wrapper equivalence lives), an "
                        "incoherent one ranks the draws per constraint and admits "
                        "a model for each. Forcing a mismatch is legal, reported, "
                        "and gets its own cell. Gastric-only in effect.")
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
    p.add_argument("--margin-grid", type=float, nargs="+", default=None,
                   help="values for the margin method only; default is the same "
                        "grid as --rho-grid, since a margin m and a radius rho "
                        "are both in unexplained-sd units. Give it its own when "
                        "the useful range of a direct RHS shift differs from that "
                        "of an assumed radius. m=0 is exactly nominal")
    p.add_argument("--n-folds", type=int, default=None,
                   help="single-decision problems only (synthetic, reactor): "
                        "KFold splits, default cv_calibration.n_kfold = 10. "
                        "Held-out feasibility is quantized to 1/n_folds there (one "
                        "solve per fold), which is why this is 10 and run_cv.py's "
                        "model CV is 5. Always in the cell name")
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
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the DRAWS FROM D (CP's bank of B, the wrapper's "
                        "P) only -- the data and the folds keep the config seed, so "
                        "repeating the sweep across seeds isolates bank variance. "
                        "Scopes every output with _s<seed>")
    p.add_argument("--refresh", action="store_true", help="discard the score cache")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    args = p.parse_args()

    import yaml
    config = yaml.safe_load(open(args.config))

    # Resolve the two things that scope a SYNTHETIC cell but are not flags:
    # the fold count (an implicit --n-folds, since the config default now decides
    # what feasibility can resolve) and the embedded model type (CV winner or
    # config fallback). Both are settled here, before --rho-star-only branches, so
    # a re-derivation reads back the curve the sweep wrote. Gastric is untouched:
    # its folds are temporal (n_kfold is inert there) and its models come from
    # --cv-configs, so adding either token would only make its filenames lie.
    # Settle CP's separation path before anything reads a cell name: --rho-star-only
    # re-derives rho* from the curve the sweep wrote, so it has to resolve the
    # same suffix a solving run would.
    if args.separation is None:
        args.separation = (
            config.get("methods", {}).get("cp", {}).get("separation", "auto")
        )

    if args.problem in ("synthetic", "reactor"):
        args.n_folds = _synth_n_folds(config, args)
        from experiments.run_sweep import synth_model_spec, reactor_model_spec
        spec = (synth_model_spec if args.problem == "synthetic"
                else reactor_model_spec)
        args.synth_model = spec(config)[0]

    if args.rho_star_only:
        # Pure post-processing: re-derive rho* from the saved curve under new
        # criteria. No instance, no oracle, no solving.
        _rho_star_from_csv(args.problem, float(args.feas_target),
                           float(args.min_solved), sense=args.sense,
                           exclude_capped=args.exclude_capped,
                           out_suffix=args.out_suffix,
                           variant=_variant_suffix(args))
        return

    run_sweep(config, args)


if __name__ == "__main__":
    main()

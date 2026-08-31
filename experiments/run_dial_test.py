"""The TEST stage: each method run at its own ``dial*``, judged by the truth.

``run_dial_sweep.py`` TUNES. Every number it reports is a held-out fold score
under the judge that instance tunes against -- the proxy ensemble on synthetic
and the reactor, the GT ensemble on gastric -- and the dial it picks
(``dial_star``) is fitted to exactly that column. Reporting a tuned dial's own
tuning score as the result is the error this file exists to avoid.

So: read ``{problem}_dial_star{cell}.csv``, hold each method at its own
``dial*``, and score it again under a judge the dial was never fitted against.

**Two phases, and they answer different questions.**

``folds``  the SAME folds the sweep used, each fitting on its (n-1)/n of the
           rows, re-solved at ``dial*`` and judged by the TRUTH rather than the
           proxy. This is a rate over folds, so it has spread; what it is not is
           independent of the tuning, since dial* was chosen on these folds. Read
           it as "the tuned dial, scored by a judge that was not part of the
           tuning", not as a held-out estimate.
``full``   one refit on ALL the rows, one solve, one decision per method, judged
           by the same truth. This is the procedure as it would actually be
           deployed and the objectives are directly comparable across methods --
           one x* each, same data, same judge. Feasibility here is a single bit
           per method, which is why it is reported beside the fold rate and not
           instead of it.
``subsample``  **gastric only**: the repo's standing robustness protocol --
           m-out-of-n subsampling (``--subsample-frac 0.5``,
           ``--n-realizations 10``) of the CONSTRAINT FIT ROWS against the fixed
           full-cohort GT oracle, each realization prescribing for the held-out
           ``X_test`` arms. It is ``full`` repeated over training draws, and on
           gastric it is what gives the test stage a spread that is not the fold
           spread ``dial*`` was tuned on. Common random numbers -- the subsample
           seed is ``bootstrap_seed + 1000*(r+1)``, a function of the realization
           ALONE, exactly as ``run_chemo_robust.py``'s Table 6 probe computes it
           -- so every method sees the same 10 draws and the comparison is
           paired. Two feasibility columns are reported: the usual one,
           conditional on each series' OWN solved arms, and
           ``feasibility_samestore``, over the arms EVERY series solved in that
           realization (recomputed per draw, the Table 6 convention).

           **Gastric needs it because gastric has no ground truth.** Its judge
           is a FITTED ensemble and its "test" is a cohort split, so the
           training draw is the only axis left that is neither fixed by
           construction nor part of the tuning. Synthetic and the reactor are
           judged by the analytic ``f_true`` and by the ODE -- a judge the dial
           never faced and that reads no training row -- so ``folds`` is already
           out of sample there in the way that matters, and the phase is refused
           rather than silently skipped.

**The judge, per problem.**

  synthetic  ``instance.gt_constraints[0]`` -- the ANALYTIC ``f_true``. The proxy
             ensemble that tuned dial* has a fitted error of its own precisely on
             the boundary a constrained optimum sits on (Known gap #8: 26% of
             verdicts flipped inside the decision band), so it is the wrong thing
             to conclude with.
  reactor    ``cv_calibrate.make_gt_oracle`` -- the ODE system, integrated. The
             only exact judge in the repo, and the one C-MICL report against.
  gastric    the **full-cohort GT ensemble** -- fit on all **416** arms
             (train + test), the fixed evaluation oracle every Table 6 number is
             against -- on the ``X_test`` arms. NOT ``make_cv_oracle``'s
             train-only 320-arm ensemble, which is the tuning proxy the sweep
             scores dial* against. There is no ground truth here, so what makes
             ``full`` a test is the cohort AND the switch from proxy to the real
             oracle.

Usage::

    python experiments/run_dial_test.py --problem reactor
    python experiments/run_dial_test.py --problem synthetic
    python experiments/run_dial_test.py --problem gastric --phases full
    python experiments/run_dial_test.py --problem gastric --phases subsample
    python experiments/run_dial_test.py --problem reactor --phases folds full

The cell flags mirror ``run_dial_sweep.py`` exactly (``--coherent``,
``--match-bank``, ``--n-folds``, ``--seed``) because the star file this reads is
scoped by them. Outputs:

  ``{problem}_dial_test_points{cell}.csv``  one row per (method, phase, fold/realization/context)
  ``{problem}_dial_test{cell}.csv``         the summary, one row per (method, phase)

**The objective carries a spread too** (``objective_sd``, and
``objective_samestore_sd`` in the ``subsample`` phase), over the same axis
``spread_over`` names for feasibility -- folds in ``folds``, realizations in
``subsample``, NaN in ``full``, computed the same way as the feasibility spread
beside it in that phase. It is what says whether a gap between two methods'
objectives is a method effect or the draw: on the 2026-08-27 gastric run every
method's ``objective_samestore_sd`` is 0.52-0.60, wider than any gap between
their means, and the comparison survives only because the draws are PAIRED by
CRN. The objective's tail is reported too (``objective_worst`` /
``objective_q10``, and the ``_samestore`` pair in ``subsample``), but it cannot
be a plain ``min`` the way ``feas_worst_case`` is: which tail is the bad one
depends on ``judge.objective_sense``, so both go through :func:`_obj_tail`,
which reads that sense. It is the column that says whether robustification buys
objective STABILITY rather than a better mean -- the same question the tail
answers for feasibility.
"""

import argparse
import dataclasses
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.cv_calibrate import _fold_instance, make_gt_oracle, GastricOracle
from src.data.generate import filter_constraints
from experiments.run_rho_sweep import (
    _setup_synthetic, _setup_reactor, _setup_gastric, _variant_suffix, _bank_seed,
    _load_cv_configs,
)

OUT_DIR = "results/rho_sweep"
# The lower tail reported beside the mean in the `subsample` phase. 0.1 is what
# `chemo_metrics.aggregate_realizations` uses for Table 6, and the two tables are
# meant to be read on the same terms.
TAIL_QUANTILE = 0.1


def _obj_tail(vals, sense):
    """``(worst, tail_decile)`` of a realized objective, on the BAD side.

    Feasibility has one bad direction, so ``feas_worst_case`` is a plain ``min``.
    The objective does not: gastric MAXIMISES survival and the reactor MINIMISES
    cost, so the bad draw is the ``min`` on one and the ``max`` on the other.
    Both are keyed off ``judge.objective_sense`` -- already carried on every
    summary row -- so ``objective_worst`` is the worst draw in the problem's own
    sense and ``objective_q10`` is the decile on that same side (the 10th
    percentile under ``max``, the 90th under ``min``).

    Reported because the mean cannot answer whether robustification buys
    objective STABILITY across training draws: on the 2026-08-27 gastric run the
    between-method objective gaps sit INSIDE every method's own draw sd, so the
    tail is the only column that can separate them -- exactly as for
    feasibility, where the means are 0.959/0.945 and the worst cases 0.889/0.682.
    """
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    if sense == "min":                  # smaller is better -> the bad tail is high
        return float(np.max(v)), float(np.quantile(v, 1.0 - TAIL_QUANTILE))
    return float(np.min(v)), float(np.quantile(v, TAIL_QUANTILE))


class TruthOracle:
    """``gt_constraints`` / ``gt_objective``: the instance's own exact judge.

    Feasible iff EVERY constraint's ground-truth function is within its ``rhs``.
    That is the same test ``evaluation/metrics.py`` applies, kept in one place so
    the test stage and the synthetic metrics cannot drift apart on the tolerance.

    On synthetic this is the analytic ``f_true``; on the reactor
    ``make_gt_oracle`` is preferred instead (same function, but it is the
    documented ODE entry point). It is deliberately unavailable on gastric, which
    has no ground truth -- see ``_judge``.
    """
    def __init__(self, instance, sense: str, tol: float = 1e-4):
        self.instance = instance
        self.objective_sense = sense
        self.tol = float(tol)

    def _val(self, fn, x):
        v = fn(np.asarray(x, dtype=float).ravel())
        return float(np.asarray(v).flat[0])

    def feasible(self, x) -> bool:
        for c_idx, c in enumerate(self.instance.constraints):
            v = self._val(self.instance.gt_constraints[c_idx], x)
            if not np.isfinite(v) or v - float(c.rhs) > self.tol:
                return False
        return True

    def objective(self, x) -> float:
        return self._val(self.instance.gt_objective, x)


def _gastric_full_cohort_oracle(instance):
    """The FIXED evaluation oracle: the GT ensemble fit on all **416** arms.

    This is NOT ``su.oracle``. ``cv_calibrate.make_cv_oracle`` builds a SECOND,
    train-only ensemble on the 320 fit arms -- a tuning proxy, deliberately blind
    to the test years -- and that is the one the sweep scores dial* against. The
    evaluation oracle is the one ``generate.gastric_cancer`` trains on
    ``X_valid = arrays['full_X']`` (train + test) and hangs off the instance as
    ``eval_outcomes[*].gt_fn``; it is the same object ``chemo_metrics`` reports
    Table 6 from, and every gastric number in the repo is against it.

    Using the train-only proxy here would score the held-out arms with a judge
    that never saw them -- which sounds stricter and is simply a different, and
    worse, measurement: the protocol is a FIXED oracle fit on the full clean
    cohort, with only the constraint fit rows resampled.

    Rebuilt as a :class:`GastricOracle` rather than reused directly so the
    objective column stays predicted OS on the problem's own ``max`` sense;
    ``instance.gt_objective`` returns ``-OS``.
    """
    tox, os_model, tox_ub = {}, None, None
    for out in (instance.eval_outcomes or []):
        if out.is_survival:
            os_model = out.gt_fn
        else:
            tox[out.name] = out.gt_fn
            tox_ub = float(out.rhs)
    if os_model is None or not tox:
        return None
    return GastricOracle(tox, os_model, tox_ub)


def _judge(problem, su):
    """``(oracle, name)``: the judge the TEST stage scores under, per problem.

    Never the object the sweep tuned against. On the two single-decision problems
    that means an exact judge instead of the proxy ensemble; on gastric, where no
    ground truth exists, it means the **full-cohort (416-arm) GT ensemble** --
    the fixed evaluation oracle -- rather than the train-only proxy
    ``make_cv_oracle`` builds for tuning. The held-out thing on gastric is the
    COHORT (``X_test``); the judge changes too, from proxy to the real one.
    """
    if problem == "reactor":
        o = make_gt_oracle(su.instance)
        if o is None:                       # a filtered instance: fall back
            return TruthOracle(su.instance, su.oracle.objective_sense), "ode(fallback)"
        return o, "ode"
    if problem == "synthetic":
        return TruthOracle(su.instance, su.oracle.objective_sense), "analytic_f_true"
    o = _gastric_full_cohort_oracle(su.instance)
    if o is None:
        raise SystemExit("[dial-test] gastric instance carries no eval_outcomes, "
                         "so the full-cohort GT ensemble is unavailable. Refusing "
                         "to fall back to the train-only tuning proxy.")
    return o, "gt_ensemble_full416"


def _dial_star_rows(problem, var, methods, fallback_best_feas=False):
    """(method, rho, dial_name, dial_star) per series, from the sweep's star file.

    Series that never reached the target carry ``dial_star = NaN`` and are
    SKIPPED with the reason printed: there is no tuned dial to test. ``nominal``
    has no dial and is always carried at 0 -- it is the reference every phase is
    read against.

    ``fallback_best_feas`` tests those series anyway, at ``best_feas_dial`` --
    the most-feasible cell the series managed (already filtered on
    ``--min-solved`` by the sweep, so it is a cell that really solved). They
    carry ``kind="best_feas"``, NOT ``"tested"``, and the distinction is the
    whole point: ``dial*`` is the best OBJECTIVE among cells that had already
    cleared the feasibility target, so its tested feasibility is not selected
    on; ``best_feas_dial`` is the argmax of the very quantity the test stage
    then re-reports. That is a CEILING for the method -- "how feasible does it
    get at its best, under a fresh judge" -- and never a protocol point. The
    selection is at least made under the *tuning* judge and re-scored under the
    test judge, so the two are not the same column, but they are correlated and
    a fallback row must never be read against a ``tested`` one as if both were
    tuned the same way.

    A series with no ``best_feasibility`` at all is still skipped: that is the
    ``solved_frac >= --min-solved`` guard biting at every dial, and there is no
    cell to fall back TO. Gastric C-MICL is exactly this case -- the fallback
    does not rescue it, and only a re-run on the extended alpha grid can.
    """
    path = os.path.join(OUT_DIR, f"{problem}_dial_star{var}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"[dial-test] no star file at {path}. Run run_dial_sweep.py with the "
            f"same cell flags first -- the test stage tests a TUNED dial, so "
            f"there is nothing to do without one.")
    df = pd.read_csv(path)
    out, skipped = [], []
    for _, r in df.iterrows():
        m = str(r["method"])
        if methods and m not in methods:
            continue
        if str(r.get("bound", "")) == "no_dial" or m == "nominal":
            out.append((m, np.nan, "none", 0.0, "reference"))
            continue
        if not np.isfinite(r.get("dial_star", np.nan)):
            # `best_feasibility` was added to the star table on 2026-08-26; a
            # star file written before that has no such column, which is not the
            # same thing as a series that scored nothing.
            # Three distinct states, and they call for different fixes: the
            # column is absent (an old star file -- re-run the sweep), it is
            # present but EMPTY (every cell fell under --min-solved, so the
            # series has no scored cell at all and no fallback exists -- widen
            # the dial grid), or it holds a real number (a fallback point).
            has_col = "best_feasibility" in r.index
            best = r.get("best_feasibility", np.nan)
            has_best = has_col and best is not None and np.isfinite(best)
            bdial = r.get("best_feas_dial", np.nan)
            if fallback_best_feas and has_best and np.isfinite(bdial):
                out.append((m, float(r["rho"]) if np.isfinite(r["rho"]) else np.nan,
                            str(r["dial_name"]), float(bdial), "best_feas"))
                continue
            if has_best:
                why = "best feasibility {:.3f}".format(float(best))
            elif not has_col:
                why = ("how close it got is not recorded -- this star file "
                       "predates the best_feasibility column; re-run the sweep")
            else:
                why = ("it never scored a feasibility at all -- every dial fell "
                       "under --min-solved, so there is no best_feas cell to "
                       "fall back to; widening the dial grid is the fix, not "
                       "--fallback-best-feas")
            skipped.append(f"{m}@rho={r['rho']} (no dial*: {why})")
            continue
        out.append((m, float(r["rho"]) if np.isfinite(r["rho"]) else np.nan,
                    str(r["dial_name"]), float(r["dial_star"]), "tested"))
    return out, skipped


def _solver_for(su, base_uset, method, rho, dial):
    """The method's solver at its tuned dial, on this problem's shared D."""
    uset = (base_uset if not np.isfinite(rho)
            else dataclasses.replace(base_uset, rho=float(rho)))
    return su.make_build(method, uset)(dial)


def _prescriptions(su, result, contextual, instance=None, rows=None):
    """``[(context_idx, x_opt)]`` from one solved master.

    Single-decision: the master IS the decision, so one record. Contextual: one
    prescribe solve per held-out context, with ``x_opt=None`` where the context
    is unsolvable -- counted in ``solved``, excluded from both means, exactly as
    ``cv_calibrate.cv_score_knob`` does it.

    ``instance``/``rows`` override ``su``: the ``subsample`` phase rebuilds the
    gastric instance once per training draw, so the instance the prescribe solve
    embeds is NOT ``su.instance`` there. Everything else passes neither and gets
    the old behaviour.
    """
    if not contextual:
        # An infeasible master is NOT a decision. Every solver returns
        # `x_opt=np.zeros(n_features), obj_value=inf, status="infeasible"` when
        # the MIP has no solution, and zeros are FINITE -- so a finiteness check
        # alone scores a phantom decision at the origin, which on synthetic and
        # the reactor sits comfortably inside every constraint. This is the same
        # gate `cv_calibrate._score_single_decision` carries (see the long note
        # there); the test stage was missing it, and on the reactor the origin is
        # not merely wrong but has T=0, which the ODE judge cannot integrate.
        # `max_iterations` / `coverage_cap` / `cycle_detected` are CP returning a
        # real INCUMBENT and must still be scored -- only "infeasible" is out.
        x = getattr(result, "x_opt", None)
        obj_v = getattr(result, "obj_value", None)
        ok = (x is not None and np.all(np.isfinite(x))
              and str(getattr(result, "status", "")) != "infeasible"
              and obj_v is not None and np.isfinite(float(obj_v)))
        return [(-1, np.asarray(x, dtype=float) if ok else None)]
    from src.evaluation.chemo_metrics import solve_for_test_cohort
    inst = su.instance if instance is None else instance
    rows = getattr(su, "_test_rows", None) if rows is None else rows
    out = []
    for ci, row in enumerate(rows):
        _, x = solve_for_test_cohort(result, inst, row)
        out.append((ci, x))
    return out


def _score(su, judge, result, contextual, instance=None, rows=None):
    """``(feasibility, objective, solved_frac, per-point records)`` under ``judge``."""
    feas, obj, recs = [], [], []
    for ci, x in _prescriptions(su, result, contextual, instance, rows):
        if x is None:
            recs.append((ci, 0.0, np.nan, np.nan))
            continue
        f = 1.0 if judge.feasible(x) else 0.0
        o = float(judge.objective(x))
        feas.append(f)
        obj.append(o)
        recs.append((ci, 1.0, f, o))
    n = len(recs) or 1
    return (float(np.mean(feas)) if feas else np.nan,
            float(np.mean(obj)) if obj else np.nan,
            len(feas) / n, recs)


def _subsample_phase(config, args, su, judge, judge_name, series, points, summary,
                     write):
    """m-out-of-n subsampling at ``dial*``, on the held-out cohort. Gastric only.

    **Why gastric and not the others**, since the draw itself would generalise:
    gastric has no ground truth. Its judge is a FITTED ensemble and its "test" is
    a cohort split, so the training draw is the only axis left that is neither
    fixed by construction nor part of the tuning. Synthetic and the reactor are
    judged by the analytic ``f_true`` and by the ODE -- a judge the dial never
    faced, which reads no training row at all -- so ``folds`` is already out of
    sample there in the way that matters, and repeating it over draws would buy
    a spread, not a validity the fold rate lacks.

    This is the repo's standing robustness protocol (see CLAUDE.md, Evaluation)
    applied to the TEST stage instead of to Table 6: resample the CONSTRAINT FIT
    rows without replacement at ``--subsample-frac`` (0.5), refit and re-solve
    every method at its own ``dial*``, prescribe for the ``X_test`` arms, and
    judge with the FIXED full-cohort GT ensemble. Repeat ``--n-realizations``
    times.

    **Why it is here and ``full`` is not enough.** ``full`` is one refit on all
    the rows: one decision per method, one bit of feasibility, no spread. The
    ``folds`` phase has a spread, but it is over the very folds ``dial*`` was
    chosen on. The training DRAW is the one axis that is neither fixed by
    construction nor part of the tuning, so it is the only spread the test stage
    can honestly report -- and it is the same uncertainty (over training draws,
    outer to the set D the methods assume) every Table 6 number in this repo is
    already reported over.

    **Common random numbers.** ``subsample_seed = bootstrap_seed + 1000*(r+1)``,
    a function of the realization ALONE -- byte-identical to
    ``run_chemo_robust.run_robustness_probe``, so realization r here is the same
    draw as realization r there. Every method sees the same draws, so the method
    comparison is paired and the draw-to-draw variance is a common effect that
    cancels in the ordering.

    **The oracle does not move.** It is built once, from the FULL-data instance,
    and reused for every realization. ``gastric_cancer`` already keeps the GT
    ensemble and its percentile reference on the full training targets, so the
    judge is identical across draws; building it once makes that a guarantee
    rather than a property to be re-derived, and it is the protocol -- a fixed
    oracle, with only the fit rows resampled.

    **Two feasibility columns.** ``feasibility`` is conditional on each series'
    OWN solved arms, matching every other phase and cell in this repo.
    ``feasibility_samestore`` is over the arms EVERY tested series solved in that
    realization, recomputed per draw -- the Table 6 convention, and the one that
    does not flatter whoever solved least. Both are means over realizations of a
    per-realization rate; ``feas_sd_across_folds`` carries the sd ACROSS
    REALIZATIONS here, and ``spread_over`` says which.

    **The mean is not the point; the TAIL is.** ``feas_worst_case`` (min over
    realizations) and ``feas_q10`` (10th percentile) are reported beside it, for
    both cohorts, because that is where this protocol has historically separated
    the methods when the means did not -- Known gap #13. Same three statistics
    ``chemo_metrics.aggregate_realizations`` reports, computed the same way
    (``ddof=1`` on the sd, ``np.quantile``), so a dial-stage table and a Table 6
    one are read on the same terms. With 10 draws the min IS one draw and the
    10th percentile is interpolated between the two lowest: report them as a
    range over draws, never as a bound.
    """
    import collections
    from src.data.generate import gastric_cancer
    from src.methods.uncertainty import uncertainty_set_from_config

    frac = float(args.subsample_frac)
    n_real = int(args.n_realizations)
    base_seed = int(config.get("uncertainty", {}).get("bootstrap_seed", 42))
    cv_configs, gt_configs = _load_cv_configs(args)
    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )

    print(f"\n[dial-test] phase=subsample: {n_real} realizations, frac={frac} of "
          f"the constraint fit rows (without replacement), CRN seeds "
          f"{base_seed}+1000*(r+1). The judge is the fixed full-cohort GT "
          f"ensemble; only the FIT rows are resampled.", flush=True)

    per_real = collections.OrderedDict()   # series key -> [(feas, obj, solved)]
    for r in range(n_real):
        sub_seed = base_seed + 1000 * (r + 1)
        print(f"\n[realization {r + 1}/{n_real}] subsample_seed={sub_seed} "
              f"frac={frac}", flush=True)
        inst = gastric_cancer(fixed_constraint_configs=cv_configs,
                              fixed_gt_ensemble_configs=gt_configs,
                              train_subsample_frac=frac, subsample_seed=sub_seed)
        if su.constraint_names is not None:
            inst = filter_constraints(inst, su.constraint_names)
        rows = inst.X_test
        for method, rho, dial_name, dial, kind in series:
            key = (method, rho, dial_name, dial, kind)
            t0 = time.time()
            label = (f"{method}" + (f"@rho={rho:g}" if np.isfinite(rho) else "")
                     + f" {dial_name}={dial:g}")
            print(f"[cell] BEGIN {label}  phase=subsample r={r}", flush=True)
            res = _solver_for(su, base_uset, method, rho, dial)(inst)
            if isinstance(res, tuple):
                res = res[0]
            feas, obj, solved, recs = _score(su, judge, res, su.contextual,
                                             instance=inst, rows=rows)
            for ci, sv, fv, ov in recs:
                points.append(dict(problem="gastric", method=method, rho=rho,
                                   dial_name=dial_name, dial_star=dial,
                                   phase="subsample", judge=judge_name, fold=-1,
                                   realization=r, subsample_seed=sub_seed,
                                   context_idx=ci, solved=sv, feasible=fv,
                                   objective=ov))
            per_real.setdefault(key, []).append((feas, obj, solved))
            dt = time.time() - t0
            print(f"[cell] END   {label:<34s} phase=subsample r={r} "
                  f"feas={feas:.3f} obj={obj:+.4f} solved={solved:.3f} "
                  f"({dt:.1f}s)", flush=True)
            write()

    # ---- samestore: the arms EVERY series solved, recomputed per draw ------
    # `_score` is conditional on a series' own solved arms, which is right for
    # comparability with every other cell in the repo and wrong for comparing
    # objectives across methods -- a method that renders most arms unsolvable
    # and gets the survivors right scores well on both of its own columns. The
    # intersection is over every TESTED series (each rho column counting
    # separately) and is recomputed per realization, because which arms are
    # solvable moves with the draw.
    sub_pts = [q for q in points if q["phase"] == "subsample"]
    solved_by = collections.defaultdict(set)          # (r, key4) -> {context_idx}
    scored = {}                                       # (r, key4, ci) -> (feas, obj)
    for q in sub_pts:
        k4 = (q["method"], q["rho"], q["dial_name"], q["dial_star"])
        if q["solved"] == 1.0:
            solved_by[(q["realization"], k4)].add(q["context_idx"])
            scored[(q["realization"], k4, q["context_idx"])] = (q["feasible"],
                                                                q["objective"])
    keys4 = [k[:4] for k in per_real]
    cohort, n_empty = {}, 0
    for r in range(n_real):
        sets = [solved_by.get((r, k4), set()) for k4 in keys4]
        cohort[r] = set.intersection(*sets) if sets else set()
        if not cohort[r]:
            n_empty += 1
    if n_empty:
        print(f"[dial-test] WARNING: the samestore cohort is EMPTY on "
              f"{n_empty}/{n_real} realizations (some series solved no arm the "
              f"others also solved); those draws drop out of the samestore "
              f"columns only, never out of `feasibility`.", flush=True)
    sizes = [len(cohort[r]) for r in range(n_real)]
    print(f"[dial-test] samestore cohort per realization: "
          f"min={min(sizes) if sizes else 0} max={max(sizes) if sizes else 0} "
          f"mean={float(np.mean(sizes)) if sizes else 0.0:.1f} of "
          f"{len(su.instance.X_test)} X_test arms", flush=True)

    def _tail(vals):
        """``(mean, sd, worst, q10)`` over realizations, or NaNs.

        ``ddof=1`` and ``np.quantile`` exactly as
        ``chemo_metrics.aggregate_realizations`` computes them, so the two
        tables are commensurable.
        """
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        return (float(np.mean(v)),
                float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                float(np.min(v)), float(np.quantile(v, TAIL_QUANTILE)))

    for key, vals in per_real.items():
        method, rho, dial_name, dial, kind = key
        k4 = key[:4]
        feas_r = [v[0] for v in vals if np.isfinite(v[0])]
        obj_r = [v[1] for v in vals if np.isfinite(v[1])]
        solved_r = [v[2] for v in vals if np.isfinite(v[2])]
        ss_f, ss_o = [], []
        for r in range(n_real):
            ids = cohort.get(r, set())
            fv = [scored[(r, k4, ci)][0] for ci in ids if (r, k4, ci) in scored]
            ov = [scored[(r, k4, ci)][1] for ci in ids if (r, k4, ci) in scored]
            if fv:
                ss_f.append(float(np.mean(fv)))
                ss_o.append(float(np.mean(ov)))
        f_mean, f_sd, f_worst, f_q10 = _tail(feas_r)
        s_mean, s_sd, s_worst, s_q10 = _tail(ss_f)
        # The mean and sd are sense-free and come from `_tail` unchanged. The
        # WORST and the decile are directional -- gastric maximises survival,
        # the reactor minimises cost -- so they go through `_obj_tail`, which
        # reads `judge.objective_sense` instead of assuming `min` is the bad
        # side. `_tail`'s own worst/q10 slots stay discarded for that reason.
        o_mean, o_sd, _, _ = _tail(obj_r)
        so_mean, so_sd, _, _ = _tail(ss_o)
        o_worst, o_q10 = _obj_tail(obj_r, judge.objective_sense)
        so_worst, so_q10 = _obj_tail(ss_o, judge.objective_sense)
        summary.append(dict(
            problem="gastric", method=method, rho=rho, dial_name=dial_name,
            dial_star=dial, kind=kind, phase="subsample", judge=judge_name,
            feasibility=f_mean, feas_sd_across_folds=f_sd,
            feas_worst_case=f_worst, feas_q10=f_q10,
            spread_over="realizations",
            objective=o_mean, objective_sd=o_sd,
            objective_worst=o_worst, objective_q10=o_q10,
            solved_frac=float(np.mean(solved_r)) if solved_r else np.nan,
            feasibility_samestore=s_mean, feas_samestore_sd=s_sd,
            feas_samestore_worst_case=s_worst, feas_samestore_q10=s_q10,
            objective_samestore=so_mean, objective_samestore_sd=so_sd,
            objective_samestore_worst=so_worst,
            objective_samestore_q10=so_q10,
            n_samestore=float(np.mean(sizes)) if sizes else 0.0,
            n_realizations=n_real, subsample_frac=frac,
            n_points=int(sum(1 for q in sub_pts
                             if q["method"] == method
                             and (q["rho"] == rho or
                                  (not np.isfinite(rho)
                                   and not np.isfinite(q["rho"]))))),
            objective_sense=judge.objective_sense,
            coherent=bool(args.coherent), matched_bank=bool(args.match_bank),
            seed=_bank_seed(config, args), wall_s=np.nan,
        ))
    write()


def run(config, args):
    os.makedirs(OUT_DIR, exist_ok=True)
    problem = args.problem
    # Checked BEFORE the instance is built, so an illegal phase costs nothing.
    # Not a defaulted-away option, and NOT merely a plumbing limit: gastric needs
    # this phase because its judge is a fitted ensemble and its "test" is a cohort
    # split, leaving the training draw as the only axis that is neither fixed by
    # construction nor part of the tuning. Synthetic and the reactor are judged by
    # the analytic f_true and by the ODE -- a judge the dial never faced, reading
    # no training row -- so `folds` is already out of sample there in the way that
    # matters. Silently dropping the phase would report a test stage that did not
    # run, so it is refused instead.
    if "subsample" in args.phases and problem != "gastric":
        raise SystemExit(f"[dial-test] --phases subsample is gastric only (got "
                         f"--problem {problem}). Gastric needs it because its "
                         f"judge is FITTED and its test is a cohort split; the "
                         f"reactor is judged by the ODE, so `--phases folds full` "
                         f"is already out of sample.")
    su = {"synthetic": _setup_synthetic, "reactor": _setup_reactor,
          "gastric": _setup_gastric}[problem](config, args)
    var = _variant_suffix(args)
    judge, judge_name = _judge(problem, su)

    from src.methods.uncertainty import uncertainty_set_from_config
    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )

    series, skipped = _dial_star_rows(problem, var, args.methods,
                                      fallback_best_feas=args.fallback_best_feas)
    phases = list(args.phases)
    solve_phases = [ph for ph in phases if ph != "subsample"]

    print(f"[dial-test] problem={problem} cell={var or '(none)'} "
          f"judge={judge_name} sense={judge.objective_sense} "
          f"folds={len(su.folds)} phases={phases}", flush=True)
    print(f"[dial-test] each method held at its OWN dial*, read from "
          f"{problem}_dial_star{var}.csv", flush=True)
    if problem == "gastric":
        print(f"[dial-test] NOTE: gastric has no ground truth. The judge here is "
              f"the FULL-COHORT GT ensemble (all 416 arms, train + test) -- the "
              f"fixed evaluation oracle -- NOT the train-only 320-arm proxy the "
              f"sweep tuned dial* against. `full` prescribes for the "
              f"{len(su.instance.X_test)} X_test arms, which no method's "
              f"constraint fit has seen.", flush=True)
    ceilings = [s for s in series if s[4] == "best_feas"]
    if ceilings:
        print(f"[dial-test] --fallback-best-feas: {len(ceilings)} series with no "
              f"dial* are tested at their MOST-FEASIBLE cell instead. These rows "
              f"carry kind='best_feas' and are a CEILING for the method, not a "
              f"protocol point -- the dial was picked by maximising the quantity "
              f"being re-reported. Do not read them against kind='tested' rows "
              f"as if both were tuned the same way.", flush=True)
        for m, rho, dn, d, _ in ceilings:
            print(f"[dial-test]   CEILING {m}"
                  + (f"@rho={rho:g}" if np.isfinite(rho) else "")
                  + f" at {dn}={d:g}", flush=True)
    for s in skipped:
        print(f"[dial-test] SKIPPING {s}", flush=True)
    print(flush=True)

    points, summary = [], []

    def _write():
        """Checkpoint both outputs. Long solves; a killed job keeps what ran."""
        pd.DataFrame(points).to_csv(
            os.path.join(OUT_DIR, f"{problem}_dial_test_points{var}.csv"),
            index=False)
        pd.DataFrame(summary).to_csv(
            os.path.join(OUT_DIR, f"{problem}_dial_test{var}.csv"), index=False)

    for method, rho, dial_name, dial, kind in series:
        for phase in solve_phases:
            t0 = time.time()
            label = (f"{method}" + (f"@rho={rho:g}" if np.isfinite(rho) else "")
                     + f" {dial_name}={dial:g}")
            print(f"[cell] BEGIN {label}  phase={phase}", flush=True)
            if phase == "folds":
                feas_f, obj_f, solved_f = [], [], []
                for k, (train_idx, val_idx) in enumerate(su.folds):
                    val_rows = (su.instance.X_train[val_idx]
                                if (su.contextual and su.instance.X_train is not None)
                                else None)
                    print(f"  [fold {k + 1}/{len(su.folds)}] {label} phase=folds "
                          f"(n_train={len(train_idx)})", flush=True)
                    fi = _fold_instance(su.instance, train_idx, val_rows)
                    if su.constraint_names is not None:
                        fi = filter_constraints(fi, su.constraint_names)
                    su._test_rows = val_rows if su.contextual else None
                    res = _solver_for(su, base_uset, method, rho, dial)(fi)
                    if isinstance(res, tuple):
                        res = res[0]
                    fe, ob, so, recs = _score(su, judge, res, su.contextual)
                    for ci, sv, fv, ov in recs:
                        points.append(dict(problem=problem, method=method, rho=rho,
                                           dial_name=dial_name, dial_star=dial,
                                           phase=phase, judge=judge_name, fold=k,
                                           context_idx=ci, solved=sv, feasible=fv,
                                           objective=ov))
                    if np.isfinite(fe):
                        feas_f.append(fe)
                    if np.isfinite(ob):
                        obj_f.append(ob)
                    solved_f.append(so)
                feas = float(np.mean(feas_f)) if feas_f else np.nan
                obj = float(np.mean(obj_f)) if obj_f else np.nan
                solved = float(np.mean(solved_f)) if solved_f else np.nan
                # Spread ACROSS FOLDS, the only variation this phase has. It is
                # not a CI: the folds are fixed by construction and dial* was
                # chosen on them.
                # `ddof=0` here against the `subsample` phase's `ddof=1` -- these
                # are the population sd of the folds in hand rather than an
                # estimate for a population of draws. Kept as it was so no
                # committed `feas_sd_across_folds` moves; `obj_spread` is
                # computed the SAME way as the feasibility spread beside it, so
                # the two columns of one row are always commensurable.
                spread = float(np.std(feas_f)) if len(feas_f) > 1 else np.nan
                obj_spread = float(np.std(obj_f)) if len(obj_f) > 1 else np.nan
                obj_worst, obj_q10 = _obj_tail(obj_f, judge.objective_sense)
            else:                                     # full: one refit, all rows
                inst = su.instance
                if su.constraint_names is not None:
                    inst = filter_constraints(inst, su.constraint_names)
                su._test_rows = inst.X_test if su.contextual else None
                res = _solver_for(su, base_uset, method, rho, dial)(inst)
                if isinstance(res, tuple):
                    res = res[0]
                feas, obj, solved, recs = _score(su, judge, res, su.contextual)
                for ci, sv, fv, ov in recs:
                    points.append(dict(problem=problem, method=method, rho=rho,
                                       dial_name=dial_name, dial_star=dial,
                                       phase=phase, judge=judge_name, fold=-1,
                                       context_idx=ci, solved=sv, feasible=fv,
                                       objective=ov))
                # One refit, one decision per method: there is no spread to
                # report on either column, which is the point of the phase.
                spread = obj_spread = np.nan
                obj_worst = obj_q10 = np.nan
            dt = time.time() - t0
            summary.append(dict(
                problem=problem, method=method, rho=rho, dial_name=dial_name,
                dial_star=dial, kind=kind, phase=phase, judge=judge_name,
                feasibility=feas, feas_sd_across_folds=spread,
                spread_over=("folds" if phase == "folds" else ""),
                objective=obj, objective_sd=obj_spread,
                objective_worst=obj_worst, objective_q10=obj_q10,
                solved_frac=solved,
                n_points=int(sum(1 for p in points
                                 if p["method"] == method and p["phase"] == phase
                                 and (p["rho"] == rho or
                                      (not np.isfinite(rho) and not np.isfinite(p["rho"]))))),
                objective_sense=judge.objective_sense,
                coherent=bool(args.coherent), matched_bank=bool(args.match_bank),
                seed=_bank_seed(config, args), wall_s=dt,
            ))
            print(f"[cell] END   {label:<34s} phase={phase:<5s} "
                  f"feas={feas:.3f} obj={obj:+.4f} solved={solved:.3f} "
                  f"({dt:.1f}s)", flush=True)
            # Written after every cell: these are long solves and a killed job
            # should leave what it finished on disk.
            _write()

    if "subsample" in phases:
        _subsample_phase(config, args, su, judge, judge_name, series, points,
                         summary, _write)

    out = pd.DataFrame(summary)
    print(f"\n[dial-test] each method at its own dial*, judged by {judge_name} "
          f"(sense={judge.objective_sense})")
    cols = ["method", "rho", "dial_name", "dial_star", "phase", "feasibility",
            "feas_sd_across_folds", "spread_over", "objective", "solved_frac",
            "n_points"]
    cols += [c for c in ("feas_worst_case", "feas_q10", "feasibility_samestore",
                         "feas_samestore_worst_case", "objective_samestore",
                         "n_samestore") if c in out.columns]
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    print(f"\n[dial-test] wrote {OUT_DIR}/{problem}_dial_test{var}.csv and "
          f"{problem}_dial_test_points{var}.csv", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "reactor", "gastric"),
                   default="reactor")
    p.add_argument("--methods", nargs="+", default=None,
                   help="default: every series in the star file that has a dial*")
    p.add_argument("--fallback-best-feas", dest="fallback_best_feas",
                   action="store_true", default=True,
                   help="DEFAULT ON. Also test series with NO dial*, at their "
                        "most-feasible cell (`best_feas_dial`: argmax of TUNED "
                        "feasibility under the tuning judge, ties broken on "
                        "objective). Those rows carry kind='best_feas' and are a "
                        "CEILING, not a protocol point -- the dial was chosen by "
                        "maximising the very quantity being re-reported, so they "
                        "must not be read against a kind='tested' row as if "
                        "tuned alike. Series whose every cell fell under "
                        "--min-solved have no cell to fall back to and are "
                        "skipped regardless.")
    p.add_argument("--no-fallback-best-feas", dest="fallback_best_feas",
                   action="store_false",
                   help="skip series with no dial* instead, reporting only "
                        "tuned protocol points")
    p.add_argument("--phases", nargs="+", default=None,
                   choices=("folds", "full", "subsample"),
                   help="`folds` re-solves the sweep's folds under the truth "
                        "judge (a rate, with spread, but over the folds dial* "
                        "was tuned on); `full` refits on all rows once (the "
                        "deployed procedure, one bit of feasibility); "
                        "`subsample` is m-out-of-n subsampling of the fit rows "
                        "against the fixed full-cohort oracle on the held-out "
                        "X_test arms -- GASTRIC ONLY, because gastric's judge is "
                        "fitted and its test is a cohort split; the reactor is "
                        "judged by the ODE, so its `folds` are already out of "
                        "sample. Default: folds full [subsample on gastric]")
    p.add_argument("--n-realizations", type=int, default=10,
                   help="training draws in the `subsample` phase (default 10, "
                        "the Table 6 protocol)")
    p.add_argument("--subsample-frac", type=float, default=0.5,
                   help="fraction of the constraint fit rows kept per draw "
                        "(default 0.5, the Table 6 protocol)")
    # Cell flags -- these must match the sweep run whose star file is being read.
    # --incoherent is the production cell and the default; both spellings exist
    # so submit_dial_sweep.sh's one COHERENCE variable reaches both stages.
    p.add_argument("--incoherent", dest="coherent", action="store_false",
                   default=False)
    p.add_argument("--coherent", dest="coherent", action="store_true")
    p.add_argument("--match-bank", action="store_true")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    args = p.parse_args()
    if args.phases is None:
        # m-out-of-n is the standing gastric protocol, so it is in the default
        # phases there and is not a legal phase anywhere else -- the reactor and
        # synthetic get their out-of-sample from the JUDGE, not from the draw.
        args.phases = (["folds", "full", "subsample"] if args.problem == "gastric"
                       else ["folds", "full"])

    import yaml
    config = yaml.safe_load(open(args.config))

    # Resolved EXACTLY as run_dial_sweep.main resolves them. `_variant_suffix`
    # reads `n_folds`, `synth_model` and `separation`, and the suffix has to come
    # out identical or this reads the wrong star file -- or, worse, none, and
    # says so about a cell that does exist.
    args.separation = config.get("methods", {}).get("cp", {}).get("separation", "auto")
    if args.problem in ("synthetic", "reactor"):
        from experiments.run_rho_sweep import _synth_n_folds
        from experiments.run_sweep import synth_model_spec, reactor_model_spec
        args.n_folds = _synth_n_folds(config, args)
        spec = (synth_model_spec if args.problem == "synthetic"
                else reactor_model_spec)
        args.synth_model = spec(config)[0]
    run(config, args)


if __name__ == "__main__":
    main()

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
    python experiments/run_dial_test.py --problem reactor --phases folds full

The cell flags mirror ``run_dial_sweep.py`` exactly (``--coherent``,
``--match-bank``, ``--n-folds``, ``--seed``) because the star file this reads is
scoped by them. Outputs:

  ``{problem}_dial_test_points{cell}.csv``  one row per (method, phase, fold/context)
  ``{problem}_dial_test{cell}.csv``         the summary, one row per (method, phase)
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
)

OUT_DIR = "results/rho_sweep"


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


def _dial_star_rows(problem, var, methods):
    """(method, rho, dial_name, dial_star) per series, from the sweep's star file.

    Series that never reached the target carry ``dial_star = NaN`` and are
    SKIPPED with the reason printed: there is no tuned dial to test. ``nominal``
    has no dial and is always carried at 0 -- it is the reference every phase is
    read against.
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
            best = r.get("best_feasibility", None)
            why = ("best feasibility {:.3f}".format(float(best))
                   if best is not None and np.isfinite(best)
                   else "how close it got is not recorded -- this star file "
                        "predates the best_feasibility column; re-run the sweep")
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


def _prescriptions(su, result, contextual):
    """``[(context_idx, x_opt)]`` from one solved master.

    Single-decision: the master IS the decision, so one record. Contextual: one
    prescribe solve per held-out context, with ``x_opt=None`` where the context
    is unsolvable -- counted in ``solved``, excluded from both means, exactly as
    ``cv_calibrate.cv_score_knob`` does it.
    """
    if not contextual:
        x = getattr(result, "x_opt", None)
        ok = x is not None and np.all(np.isfinite(x))
        return [(-1, np.asarray(x, dtype=float) if ok else None)]
    from src.evaluation.chemo_metrics import solve_for_test_cohort
    rows = getattr(su, "_test_rows", None)
    out = []
    for ci, row in enumerate(rows):
        _, x = solve_for_test_cohort(result, su.instance, row)
        out.append((ci, x))
    return out


def _score(su, judge, result, contextual):
    """``(feasibility, objective, solved_frac, per-point records)`` under ``judge``."""
    feas, obj, recs = [], [], []
    for ci, x in _prescriptions(su, result, contextual):
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


def run(config, args):
    os.makedirs(OUT_DIR, exist_ok=True)
    problem = args.problem
    su = {"synthetic": _setup_synthetic, "reactor": _setup_reactor,
          "gastric": _setup_gastric}[problem](config, args)
    var = _variant_suffix(args)
    judge, judge_name = _judge(problem, su)

    from src.methods.uncertainty import uncertainty_set_from_config
    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )

    series, skipped = _dial_star_rows(problem, var, args.methods)
    phases = list(args.phases)

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
    for s in skipped:
        print(f"[dial-test] SKIPPING {s}", flush=True)
    print(flush=True)

    points, summary = [], []
    for method, rho, dial_name, dial, kind in series:
        for phase in phases:
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
                spread = float(np.std(feas_f)) if len(feas_f) > 1 else np.nan
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
                spread = np.nan
            dt = time.time() - t0
            summary.append(dict(
                problem=problem, method=method, rho=rho, dial_name=dial_name,
                dial_star=dial, kind=kind, phase=phase, judge=judge_name,
                feasibility=feas, feas_sd_across_folds=spread, objective=obj,
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
            pd.DataFrame(points).to_csv(
                os.path.join(OUT_DIR, f"{problem}_dial_test_points{var}.csv"),
                index=False)
            pd.DataFrame(summary).to_csv(
                os.path.join(OUT_DIR, f"{problem}_dial_test{var}.csv"), index=False)

    out = pd.DataFrame(summary)
    print(f"\n[dial-test] each method at its own dial*, judged by {judge_name} "
          f"(sense={judge.objective_sense})")
    cols = ["method", "rho", "dial_name", "dial_star", "phase", "feasibility",
            "feas_sd_across_folds", "objective", "solved_frac", "n_points"]
    print(out[cols].to_string(index=False))
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
    p.add_argument("--phases", nargs="+", default=["folds", "full"],
                   choices=("folds", "full"),
                   help="`folds` re-solves the sweep's folds under the truth "
                        "judge (a rate, with spread); `full` refits on all rows "
                        "once (the deployed procedure, one bit of feasibility). "
                        "Default: both")
    # Cell flags -- these must match the sweep run whose star file is being read.
    p.add_argument("--coherent", action="store_true")
    p.add_argument("--match-bank", action="store_true")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    args = p.parse_args()

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

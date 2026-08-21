"""Robustness-parameter cross-validation: select each method's single robustness
knob on held-out folds, trading feasibility against the objective.

This is the **robustness-parameter CV** -- run AFTER the model-selection CVs
(``run_cv.py``, which fix model *types*); it never re-selects a model type. For a
knob grid it fits each method on every fold-train, prescribes for the fold-val
contexts, scores ``(feasibility, objective)`` under a **train-only proxy oracle**
(the test cohort is never seen), averages over folds, and picks

    theta* = argmax feasibility  s.t.  objective within ``os_tolerance_frac`` of
             the nominal objective (in the problem's worse direction).

Problem-adaptive: gastric uses temporal forward-chaining folds + a train-only GT
ensemble oracle; synthetic uses random KFold + a proxy ensemble fit on the
training labels (its analytic ground truth is reserved for FINAL evaluation only).
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold

from src.data.generate import ProblemInstance, filter_constraints
from src.evaluation.chemo_metrics import solve_for_test_cohort
from src.models.train import train_fixed_ensemble

# CV-tuned members for the synthetic oracle, written by
# `run_cv.py --problem synthetic --ensemble`. Absent -> the fallback specs in
# src/data/synthetic_model_specs.py.
SYNTHETIC_GT_CONFIGS = os.path.join(
    "results", "cv", "synthetic_gt_ensemble_configs.json")
REACTOR_GT_CONFIGS = os.path.join(
    "results", "cv", "reactor_gt_ensemble_configs.json")
# Constraint name that marks the reactor instance; the oracle dispatch keys on it
# rather than on a problem string, so a filtered or re-derived instance still
# routes correctly.
REACTOR_OUTCOME = "benzene_constraint"


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def make_folds(instance: ProblemInstance, scheme: str = "auto",
               cutoffs: Sequence[int] = (2004, 2005, 2006, 2007),
               n_kfold: int = 4, seed: int = 42):
    """Return ``[(train_idx, val_idx), ...]`` index pairs into ``instance.X_train``.

    ``temporal`` (gastric): forward-chaining -- train = rows with ``Pub_Year <=
    cutoff``, val = rows in ``cutoff+1``. ``kfold`` (synthetic, no time): random
    KFold. ``auto`` picks temporal iff ``train_pub_years`` is present.
    """
    years = instance.train_pub_years
    use_temporal = scheme == "temporal" or (scheme == "auto" and years is not None)
    # n_train from X_train when present, else from the first constraint's fit data
    # (synthetic instances may not carry a contextual X_train).
    n = (instance.X_train.shape[0] if instance.X_train is not None
         else instance.constraints[0].models_data[0].X_train.shape[0])
    if use_temporal:
        if years is None:
            raise ValueError("temporal folds requested but train_pub_years is None")
        years = np.asarray(years)
        folds = []
        for c in cutoffs:
            tr = np.where(years <= c)[0]
            va = np.where((years > c) & (years <= c + 1))[0]
            if len(tr) > 0 and len(va) > 0:
                folds.append((tr, va))
        if not folds:
            raise ValueError(f"no non-empty temporal folds for cutoffs {cutoffs}")
        return folds
    kf = KFold(n_splits=n_kfold, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(np.arange(n))]


# ---------------------------------------------------------------------------
# Oracles (data-fit proxies -- never the analytic truth)
# ---------------------------------------------------------------------------
class GastricOracle:
    """Train-only GT-ensemble oracle: feasible iff every toxicity outcome's mean
    ensemble prediction is <= ``tox_ub``; objective = predicted OS (maximize)."""
    objective_sense = "max"

    def __init__(self, tox_models: dict, os_model, tox_ub: float):
        self.tox_models = tox_models
        self.os_model = os_model
        self.tox_ub = float(tox_ub)

    def feasible(self, x: np.ndarray) -> bool:
        xr = np.atleast_2d(x)
        return all(float(m.predict(xr)[0]) <= self.tox_ub + 1e-9
                   for m in self.tox_models.values())

    def objective(self, x: np.ndarray) -> float:
        return float(self.os_model.predict(np.atleast_2d(x))[0])


class SyntheticOracle:
    """Proxy-ensemble oracle for a single-constraint problem: feasible iff
    ``weight * prediction <= rhs``; objective = c'x (minimize).

    ``model`` is a MIXED-TYPE ensemble (seven model classes averaged). Until
    2026-08-21 it was a single model of the same class the candidate embeds, so
    oracle and candidate shared their approximation error and an rf artifact the
    optimizer exploited was judged by an rf that had it too. The `mlp` member is
    deliberately kept even though synthetic and reactor now embed an MLP: a judge
    missing the candidate's class cannot follow it into the region where it is
    wrong, and one shared class out of seven is diluted to 1/7 of the average.
    Gastric is the exception and averages six -- its ensemble replicates Table
    EC.12, which has no MLP.

    ``weight`` carries the constraint's sign, matching ``LearnedConstraint``'s
    ``sum(w_i f_i(x)) <= rhs``. Synthetic is an upper bound (``w = +1``); the
    reactor's requirement is a LOWER bound on benzene flow, stored as ``w = -1``
    against ``rhs = -50``. Hard-coding ``+1`` here would silently inverted the
    reactor's feasibility column.

    KNOWN LIMITATION, and it is not small. This judge is a fitted model, so it has
    an error of its own precisely where a constrained optimum lives -- on the
    boundary. Measured on synthetic (2026-08-21, seven members, |f_true - b| <
    0.05): error sd 0.033 in the decision band against margins of 0.015-0.020,
    26% of verdicts flipped versus the analytic truth inside that band, and 5 of 5
    nominal decisions misjudged. Six members (no mlp) gave 0.038 and 28%. The
    bias is not neutral between methods: robust methods leave slack, where the
    judge is reliable, while nominal sits on the boundary, where it is not. Prefer
    the reactor's ODE oracle for any FINAL feasibility claim.
    """
    objective_sense = "min"

    def __init__(self, model, rhs: float, cost_vector: np.ndarray,
                 weight: float = 1.0):
        self.model = model
        self.rhs = float(rhs)
        self.weight = float(weight)
        self.cost_vector = np.asarray(cost_vector, dtype=float)

    def feasible(self, x: np.ndarray) -> bool:
        pred = float(self.model.predict(np.atleast_2d(x))[0])
        return self.weight * pred <= self.rhs + 1e-9

    def objective(self, x: np.ndarray) -> float:
        return float(np.dot(self.cost_vector, np.asarray(x, dtype=float)))


class ReactorODEOracle:
    """GROUND TRUTH for the reactor: integrate the ODEs, do not predict them.

    This is the judge C-MICL reports against ("empirical ground-truth feasibility
    rate"), and the only exact one in this repo. It is for FINAL EVALUATION and for
    auditing the proxy above -- never for choosing rho, which would calibrate the
    uncertainty set against the very truth it is scored by.
    """
    objective_sense = "min"

    def __init__(self, rhs: float, cost_vector: np.ndarray, weight: float = -1.0):
        self.rhs = float(rhs)
        self.weight = float(weight)
        self.cost_vector = np.asarray(cost_vector, dtype=float)

    def feasible(self, x: np.ndarray) -> bool:
        from src.data.dma_mr import benzene_flow
        flow = benzene_flow(np.asarray(x, dtype=float).ravel())
        if not np.isfinite(flow):
            return False          # an unusable design, not a feasible one
        return self.weight * flow <= self.rhs + 1e-9

    def objective(self, x: np.ndarray) -> float:
        return float(np.dot(self.cost_vector, np.asarray(x, dtype=float)))


def _single_outcome_gt_specs(gt_specs, json_path, outcome, fallback) -> list:
    """Ensemble member specs for a single-constraint proxy oracle, by precedence.

    Explicit argument > the CV-tuned JSON (``run_cv.py --problem <p> --ensemble``)
    > the hand-set fallback list. Accepts a bare list of specs or the
    ``{outcome: [specs]}`` dict shape the JSON and the gastric branch use.
    """
    if gt_specs is None and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            gt_specs = json.load(f)
    if gt_specs is None:
        return fallback
    if isinstance(gt_specs, dict):
        # One outcome, but do not guess its key: prefer the canonical name, then
        # the sole entry, and only then give up to the fallback.
        if outcome in gt_specs:
            return gt_specs[outcome]
        if len(gt_specs) == 1:
            return next(iter(gt_specs.values()))
        return fallback
    return list(gt_specs)


def make_cv_oracle(instance: ProblemInstance, gt_specs=None, verbose: bool = True):
    """Build the train-only proxy oracle for this problem.

    Gastric: per-outcome GT ensemble fit on the (train-only) constraint fit data
    with train percentile targets, thresholded at each constraint's ``rhs``.

    Synthetic: a **mixed-type ensemble** -- the same six model classes gastric's GT
    ensemble averages -- fit on the full noisy ``y_train``. It was a single model of
    the class the candidate embeds until 2026-08-21, which made the judge share its
    approximation error with the thing it judged (2026-08-19 deck, next step 3);
    averaging classes is what breaks that. Two properties are deliberately kept:

      - it is fit on the **noisy** labels, and the analytic ``gt_constraints`` stay
        reserved for final evaluation. An oracle that knew the data-generating
        process would score every method against the very truth D is calibrated in
        units of, which CP would then win by construction;
      - it is fit on the **full** training rows, a superset of any fold's train
        rows -- exactly as gastric's oracle is. The judge therefore shares rows
        with the candidate on both problems; that is the protocol, not a synthetic
        defect, and it is why synthetic held-out feasibility is an m-out-of-n
        statement rather than a genuinely held-out one.
    """
    if instance.train_pub_years is not None:  # gastric
        if gt_specs is None:
            from src.data.gastric_model_specs import GT_ENSEMBLE_SPECS as gt_specs  # noqa: N811
        tox_models, os_model, tox_ub = {}, None, None
        for c in instance.constraints:
            md = c.models_data[0]
            name = c.name.replace("_constraint", "")
            if name == "os":
                os_model = train_fixed_ensemble(md.X_train, md.y_train, gt_specs["os"])
            else:
                tox_models[name] = train_fixed_ensemble(md.X_train, md.y_train, gt_specs[name])
                tox_ub = c.rhs
        return GastricOracle(tox_models, os_model, tox_ub)
    # single-constraint problems (synthetic, reactor): mixed-type ensemble on the
    # full noisy training labels.
    c = instance.constraints[0]
    md = c.models_data[0]
    if c.name == REACTOR_OUTCOME:
        from src.data.reactor_model_specs import REACTOR_GT_ENSEMBLE_SPECS
        specs = _single_outcome_gt_specs(gt_specs, REACTOR_GT_CONFIGS,
                                         REACTOR_OUTCOME, REACTOR_GT_ENSEMBLE_SPECS)
        label = "reactor"
    else:
        from src.data.synthetic_model_specs import (
            SYNTHETIC_GT_ENSEMBLE_SPECS, SYNTHETIC_OUTCOME,
        )
        specs = _single_outcome_gt_specs(gt_specs, SYNTHETIC_GT_CONFIGS,
                                         SYNTHETIC_OUTCOME,
                                         SYNTHETIC_GT_ENSEMBLE_SPECS)
        label = "synthetic"
    model = train_fixed_ensemble(md.X_train, md.y_train, specs)
    if verbose:
        print(f"    [oracle] {label}: {len(specs)}-model ensemble "
              f"({', '.join(s['model_type'] for s in specs)}) on n="
              f"{len(md.y_train)} noisy labels", flush=True)
    return SyntheticOracle(model, c.rhs, instance.cost_vector, weight=md.weight)


def make_gt_oracle(instance: ProblemInstance):
    """The EXACT judge, where one exists. Final evaluation and audits only.

    Reactor -> the ODE system. Anything else -> ``None``: gastric has no ground
    truth at all, and synthetic's analytic ``f_true`` is reserved for
    ``evaluation/metrics.py``. Callers must handle ``None`` rather than silently
    falling back to the proxy, since the whole point of this function is to be a
    different judge from :func:`make_cv_oracle`.
    """
    c = instance.constraints[0]
    if c.name != REACTOR_OUTCOME:
        return None
    return ReactorODEOracle(c.rhs, instance.cost_vector,
                            weight=c.models_data[0].weight)


# ---------------------------------------------------------------------------
# Fold instances + scoring
# ---------------------------------------------------------------------------
def _fold_instance(base: ProblemInstance, train_idx: np.ndarray,
                   val_rows: Optional[np.ndarray]) -> ProblemInstance:
    """Base instance with each constraint's fit data subset to ``train_idx`` (the
    fold varies which rows are fit; the percentile LABEL transform stays full-train
    by design -- it is what makes ``rhs`` mean one fixed raw toxicity for every fold
    and for the oracle). For contextual problems ``X_test`` is set to the fold-val
    rows; for single-decision (synthetic, ``X_train`` is None) it is left unchanged.
    Mirrors ``train_subsample_frac``.

    ``train_pub_years`` MUST be subset alongside the rows. It is the fold scheme
    ``uncertainty.instance_folds`` reads to estimate D's radius ``scale(y_c)``, so a
    full-length copy would (a) index past the fold's own rows -- an ``IndexError``
    inside ``label_scale`` on every gastric fold -- and (b) were it in range, put
    held-out rows into the scale estimate. D's radius is now estimated from the
    fold's rows alone."""
    new_constraints = []
    for c in base.constraints:
        new_mds = [
            dataclasses.replace(
                md,
                X_train=md.X_train[train_idx],
                y_train=md.y_train[train_idx],
                y_true=(md.y_true[train_idx] if md.y_true is not None else None),
            )
            for md in c.models_data
        ]
        new_constraints.append(dataclasses.replace(c, models_data=new_mds))
    tr_pts = base.trust_region_points
    years = base.train_pub_years
    return dataclasses.replace(
        base,
        constraints=new_constraints,
        X_test=val_rows if val_rows is not None else base.X_test,
        X_train=(base.X_train[train_idx] if base.X_train is not None else None),
        trust_region_points=(tr_pts[train_idx] if tr_pts is not None else None),
        train_pub_years=(years[train_idx] if years is not None else None),
    )


def cv_score_knob(build_solver: Callable[[float], Callable], knob: float,
                  folds, oracle, base: ProblemInstance,
                  constraint_names: Optional[List[str]] = None,
                  contextual: bool = True,
                  return_details: bool = False):
    """Mean held-out ``(feasibility, objective, solved_frac)`` for one knob.

    With ``return_details=True`` returns a dict instead, adding the solver
    ``status`` across folds, how many folds hit ``max_iterations``, and the wall
    clock split into the **master** phase (train + build + solve to the final
    master; for CP the whole cut loop) and the **test-point** phase (one prescribe
    solve per held-out context). The split matters because the two scale
    differently: CP pays up front in the cut loop and prescribes from a small
    master, while the wrapper embeds all P models and pays on every test point.

    Both feasibility and objective are scored **conditional on the optimizer
    returning a prescription**, over that knob's own solved contexts. Nothing is
    shared across knobs or methods, so a cell's score never depends on what any
    other cell could solve.

    This also puts the two columns on the same cohort. Previously ``obj`` was
    already conditional (only solved rows contribute) while ``feas`` scored an
    unsolvable context as 0 -- so they were means over different sets, and the
    feasibility column silently penalised whichever knob added the most
    constraints.

    ``solved_frac`` is returned because conditional feasibility on its own has a
    perverse incentive: a knob that renders most contexts unsolvable and gets the
    survivors right scores 1.0. Callers must report it, and treat a high
    feasibility at a low solved fraction as the artefact it is.
    """
    import time as _time
    solver_fn = build_solver(knob)
    fold_feas, fold_obj, fold_solved = [], [], []
    # Detail channels, reported only when return_details is set. Kept out of the
    # 3-tuple so existing callers are untouched.
    statuses, master_times, test_times, test_points = [], [], [], []
    for train_idx, val_idx in folds:
        val_rows = base.X_train[val_idx] if (contextual and base.X_train is not None) else None
        fi = _fold_instance(base, train_idx, val_rows)
        if constraint_names is not None:
            fi = filter_constraints(fi, constraint_names)
        # MASTER phase: train + build + solve to the final master. For CP this is
        # the whole cut loop, which is why it is timed separately from prescribing.
        _t0 = _time.time()
        result = solver_fn(fi)
        master_times.append(_time.time() - _t0)
        # solve_cp returns (SolutionResult, history); the baselines return a bare
        # SolutionResult. Unwrap as calibrate.infeasible_fraction does -- otherwise
        # the contextual path raises (tuple has no .x) and the single-decision path
        # silently scores NaN via getattr(result, "x_opt", None).
        if isinstance(result, tuple):
            result = result[0]
        statuses.append(str(getattr(result, "status", "unknown")))
        if contextual:
            feas_vals, obj_vals, n_total = [], [], 0
            # TEST-POINT phase: one prescribe solve per held-out context.
            _t1 = _time.time()
            for row in val_rows:
                n_total += 1
                _, x_opt = solve_for_test_cohort(result, fi, row)
                if x_opt is None:
                    continue           # unsolvable: excluded, counted in solved_frac
                feas_vals.append(1.0 if oracle.feasible(x_opt) else 0.0)
                obj_vals.append(oracle.objective(x_opt))
            test_times.append(_time.time() - _t1)
            test_points.append(n_total)
            if feas_vals:
                fold_feas.append(float(np.mean(feas_vals)))
            if obj_vals:
                fold_obj.append(float(np.mean(obj_vals)))
            if n_total:
                fold_solved.append(len(feas_vals) / n_total)
        else:
            # Single-decision: the master IS the prescription, so there is no
            # separate test-point solve to time. Recorded as 0 rather than NaN so
            # the column means "no prescribe phase", not "not measured".
            test_times.append(0.0)
            test_points.append(1)
            x_opt = getattr(result, "x_opt", None)
            solved = x_opt is not None and np.all(np.isfinite(x_opt))
            if solved:
                fold_feas.append(1.0 if oracle.feasible(x_opt) else 0.0)
                fold_obj.append(oracle.objective(x_opt))
            fold_solved.append(1.0 if solved else 0.0)
    feas = float(np.mean(fold_feas)) if fold_feas else float("nan")
    obj = float(np.mean(fold_obj)) if fold_obj else float("nan")
    solved = float(np.mean(fold_solved)) if fold_solved else float("nan")
    if not return_details:
        return feas, obj, solved
    n_pts = float(np.sum(test_points)) or 1.0
    return {
        "feas": feas, "obj": obj, "solved": solved,
        # Distinct statuses across folds, joined. A cell showing
        # "max_iterations|optimal" converged on some folds and hit the cap on
        # others -- which is exactly the thing a single mean would hide.
        "status": "|".join(sorted(set(statuses))) if statuses else "unknown",
        "n_capped": int(sum(s == "max_iterations" for s in statuses)),
        "master_time_s": float(np.mean(master_times)) if master_times else float("nan"),
        "test_time_s": float(np.mean(test_times)) if test_times else float("nan"),
        # Per-point, so gastric's 96 contexts and synthetic's single decision are
        # comparable numbers rather than a fold total dominated by cohort size.
        "test_time_per_point_s": float(np.sum(test_times)) / n_pts,
    }


def select_knob_cv(build_solver: Callable[[float], Callable], knob_grid: Sequence[float],
                   folds, oracle, base: ProblemInstance, os_tol_frac: float,
                   nominal_obj: float, constraint_names: Optional[List[str]] = None,
                   contextual: bool = True, method: str = "",
                   score_fn: Optional[Callable[[float], tuple]] = None,
                   log: Callable = print) -> tuple[float, list]:
    """Score every knob (or reuse ``score_fn`` for checkpointing) and return
    ``(theta*, rows)`` where ``rows = [(knob, feas, obj, solved), ...]``.

    ``theta*`` = argmax feasibility among knobs whose objective stays within
    ``os_tol_frac`` of ``nominal_obj`` in the problem's worse direction. **Ties
    break to the best objective**, so when two knobs are equally feasible we take
    the one that gave up less -- there is no reason to buy extra conservatism that
    bought nothing. (Previously ties broke to the later grid index, i.e. always to
    the stronger knob.) Falls back to the best-objective knob if none pass.

    ``feas`` is conditional on the optimizer returning a prescription, so a knob
    that renders contexts unsolvable is not charged for them -- which is why
    ``solved`` is logged beside it. High feasibility at a low solved fraction is
    an artefact, not a result.
    """
    rows = []
    for knob in knob_grid:
        scored = (score_fn(knob) if score_fn is not None
                  else cv_score_knob(build_solver, knob, folds, oracle, base,
                                     constraint_names, contextual))
        feas, obj, solved = scored
        rows.append((knob, feas, obj, solved))
        log(f"    [cv] {method}: knob={knob:.4g} feas={feas:.3f} obj={obj:.3f} "
            f"solved={solved:.3f}", flush=True)

    # Budget is an ADDITIVE tolerance of os_tol_frac * |nominal| in the worse
    # direction (additive handles negative objectives, e.g. synthetic cost c'x < 0;
    # for a positive objective it equals the multiplicative nominal*(1-+tol)).
    sense = oracle.objective_sense
    margin = os_tol_frac * abs(nominal_obj)
    if sense == "max":
        thresh = nominal_obj - margin
        ok = lambda o: np.isfinite(o) and o >= thresh - 1e-9
    else:
        thresh = nominal_obj + margin
        ok = lambda o: np.isfinite(o) and o <= thresh + 1e-9

    # Signed objective: larger is better under "max", smaller under "min", so one
    # comparison serves both senses.
    signed = (lambda o: o) if sense == "max" else (lambda o: -o)

    passing = [(k, f, o, sv) for (k, f, o, sv) in rows if np.isfinite(f) and ok(o)]
    if passing:
        best = max(passing, key=lambda r: (r[1], signed(r[2])))
        theta = best[0]
        log(f"    [cv] {method}: theta*={theta:.4g} (feas={best[1]:.3f} "
            f"obj={best[2]:.3f} solved={best[3]:.3f}; ties -> best objective; "
            f"budget {sense} obj {'>=' if sense=='max' else '<='} {thresh:.3f} "
            f"[nominal {nominal_obj:.3f}, tol {os_tol_frac:.2f}])", flush=True)
        if np.isfinite(best[3]) and best[3] < 0.9:
            log(f"    [cv] {method}: WARNING theta* solved only "
                f"{best[3]*100:.0f}% of held-out contexts -- its feasibility is "
                f"conditional on that subset and is not comparable at face value.",
                flush=True)
    else:
        finite = [(k, f, o, sv) for (k, f, o, sv) in rows if np.isfinite(o)]
        if not finite:
            raise ValueError(f"{method}: no finite CV scores; cannot select a knob")
        theta = max(finite, key=lambda r: signed(r[2]))[0]
        log(f"    [cv] {method}: WARNING no knob meets the objective budget "
            f"(thresh {thresh:.3f}); falling back to best-objective knob theta*={theta:.4g}",
            flush=True)
    return theta, rows


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def load_score_checkpoint(path: str) -> dict:
    """Return ``{(method, knob): (feas, obj, solved)}`` from a scores CSV, or ``{}``.

    Rows written before ``solved`` existed load with ``solved = nan``; they are
    still usable for resume, they just cannot report the solved fraction. Note
    their ``feas`` also used the old definition (unsolvable scored 0), so a mixed
    checkpoint is not internally comparable -- refresh when that matters.
    """
    import pandas as pd
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        solved = float(r["solved"]) if "solved" in df.columns else float("nan")
        out[(str(r["method"]), float(r["knob"]))] = (
            float(r["feas"]), float(r["obj"]), solved,
        )
    return out


DETAIL_COLS = ("status", "n_capped", "master_time_s", "test_time_s",
               "test_time_per_point_s")


def load_detail_checkpoint(path: str) -> dict:
    """``{(method, knob): detail_dict}`` for checkpoints written with details.

    Separate from :func:`load_score_checkpoint` so the plain 3-tuple resume path
    keeps working against old files; cells lacking the detail columns are simply
    absent here and get re-scored rather than resumed with fabricated timings.
    """
    import pandas as pd
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if not all(c in df.columns for c in DETAIL_COLS):
        return {}
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r.get("status")):
            continue
        out[(str(r["method"]), float(r["knob"]))] = {
            "feas": float(r["feas"]), "obj": float(r["obj"]),
            "solved": float(r["solved"]), "status": str(r["status"]),
            "n_capped": int(r["n_capped"]),
            "master_time_s": float(r["master_time_s"]),
            "test_time_s": float(r["test_time_s"]),
            "test_time_per_point_s": float(r["test_time_per_point_s"]),
        }
    return out


def append_score(path: str, method: str, knob: float, feas: float, obj: float,
                 solved: float = float("nan"), detail: Optional[dict] = None) -> None:
    """Append one scored cell to the checkpoint CSV (header written once).

    ``detail`` adds the status/timing columns from ``cv_score_knob(...,
    return_details=True)``. Files are written with the detail columns either way,
    so a checkpoint never changes shape mid-run.
    """
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = ["method", "knob", "feas", "obj", "solved", *DETAIL_COLS]
    _migrate_score_header(path, header)
    new = not os.path.exists(path)
    d = detail or {}
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow([method, knob, feas, obj, solved,
                    *(d.get(c, "") for c in DETAIL_COLS)])


def _migrate_score_header(path: str, header: list) -> None:
    """Widen an older checkpoint in place so appends stay column-aligned.

    The header is written only for a NEW file, so appending today's wider rows to
    a checkpoint written before ``solved`` / the detail columns existed would
    silently shift every field. Rewrite once with the new header, back-filling the
    missing columns as blank, rather than corrupting a resumable file.
    """
    import csv as _csv
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(_csv.reader(f))
    if not rows or rows[0] == header:
        return
    old = rows[0]
    idx = {c: i for i, c in enumerate(old)}
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in rows[1:]:
            if not r:
                continue
            w.writerow([r[idx[c]] if c in idx and idx[c] < len(r) else ""
                        for c in header])


def knob_key(method: str, coherent: Optional[bool]) -> str:
    """Checkpoint / knobs key for one (method, coherence) cell.

    theta* is calibrated PER CELL. A coherent draw moves every outcome together,
    so at a fixed radius it is the stronger adversary: the objective budget binds
    sooner and coherent theta* <= incoherent theta* in general. Reusing one theta*
    across both would confound "coherence" with "more conservatism" -- the exact
    confound the shared uncertainty set exists to remove.

    ``coherent=None`` gives the bare method name, for problems where coherence is
    vacuous (a single outcome, e.g. synthetic).
    """
    if coherent is None:
        return method
    return f"{method}@{'coherent' if coherent else 'incoherent'}"


def lookup_knob(knobs: dict, method: str, coherent: Optional[bool] = None,
                default=None):
    """theta* for a cell, falling back to the bare ``method`` key so knobs JSONs
    written before per-cell keying still load."""
    if not knobs:
        return default
    for k in (knob_key(method, coherent), method):
        if k in knobs:
            return knobs[k]
    return default


def write_knobs(path: str, knobs: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(knobs, f, indent=2)
    print(f"Wrote robustness knobs -> {path}: {knobs}", flush=True)

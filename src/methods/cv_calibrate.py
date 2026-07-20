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
from src.models.train import train_fixed_ensemble, train_model


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
    """Proxy-ensemble oracle for the synthetic problem: feasible iff the proxy
    constraint prediction is <= rhs; objective = c'x (minimize)."""
    objective_sense = "min"

    def __init__(self, model, rhs: float, cost_vector: np.ndarray):
        self.model = model
        self.rhs = float(rhs)
        self.cost_vector = np.asarray(cost_vector, dtype=float)

    def feasible(self, x: np.ndarray) -> bool:
        return float(self.model.predict(np.atleast_2d(x))[0]) <= self.rhs + 1e-9

    def objective(self, x: np.ndarray) -> float:
        return float(np.dot(self.cost_vector, np.asarray(x, dtype=float)))


def make_cv_oracle(instance: ProblemInstance, gt_specs: Optional[dict] = None):
    """Build the train-only proxy oracle for this problem.

    Gastric: per-outcome GT ensemble fit on the (train-only) constraint fit data
    with train percentile targets, thresholded at each constraint's ``rhs``.
    Synthetic: a proxy model fit on the full synthetic training labels; the
    analytic ``gt_constraints`` are NOT used here (final-eval only).
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
    # synthetic: proxy model on full training labels (more data than any fold)
    c = instance.constraints[0]
    md = c.models_data[0]
    cfg = (instance.constraint_model_configs or [{}])[0]
    mt = cfg.get("model_type", "rf")
    mp = cfg.get("model_params", cfg.get("params", {}))
    model = train_model(md.X_train, md.y_train, mt, mp)
    return SyntheticOracle(model, c.rhs, instance.cost_vector)


# ---------------------------------------------------------------------------
# Fold instances + scoring
# ---------------------------------------------------------------------------
def _fold_instance(base: ProblemInstance, train_idx: np.ndarray,
                   val_rows: np.ndarray) -> ProblemInstance:
    """Base instance with each constraint's fit data subset to ``train_idx`` (the
    fold varies which rows are fit; percentile scale stays full-train) and
    ``X_test`` set to the fold-val rows. Mirrors ``train_subsample_frac``."""
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
    return dataclasses.replace(
        base,
        constraints=new_constraints,
        X_test=val_rows,
        X_train=base.X_train[train_idx],
        trust_region_points=(tr_pts[train_idx] if tr_pts is not None else None),
    )


def cv_score_knob(build_solver: Callable[[float], Callable], knob: float,
                  folds, oracle, base: ProblemInstance,
                  constraint_names: Optional[List[str]] = None,
                  contextual: bool = True) -> tuple[float, float]:
    """Mean held-out ``(feasibility, objective)`` for one knob over the folds.

    Contextual (gastric): prescribe for every fold-val context (optimizer-infeasible
    contexts score feasibility 0 and are excluded from the objective mean).
    Single-decision (synthetic): one solve per fold; score the single ``x*``.
    """
    solver_fn = build_solver(knob)
    fold_feas, fold_obj = [], []
    for train_idx, val_idx in folds:
        val_rows = base.X_train[val_idx]
        fi = _fold_instance(base, train_idx, val_rows)
        if constraint_names is not None:
            fi = filter_constraints(fi, constraint_names)
        result = solver_fn(fi)
        if contextual:
            feas_vals, obj_vals = [], []
            for row in val_rows:
                _, x_opt = solve_for_test_cohort(result, fi, row)
                if x_opt is None:
                    feas_vals.append(0.0)  # no prescription -> not robustly feasible
                    continue
                feas_vals.append(1.0 if oracle.feasible(x_opt) else 0.0)
                obj_vals.append(oracle.objective(x_opt))
            if feas_vals:
                fold_feas.append(float(np.mean(feas_vals)))
            if obj_vals:
                fold_obj.append(float(np.mean(obj_vals)))
        else:
            x_opt = getattr(result, "x_opt", None)
            if x_opt is not None and np.all(np.isfinite(x_opt)):
                fold_feas.append(1.0 if oracle.feasible(x_opt) else 0.0)
                fold_obj.append(oracle.objective(x_opt))
    feas = float(np.mean(fold_feas)) if fold_feas else float("nan")
    obj = float(np.mean(fold_obj)) if fold_obj else float("nan")
    return feas, obj


def select_knob_cv(build_solver: Callable[[float], Callable], knob_grid: Sequence[float],
                   folds, oracle, base: ProblemInstance, os_tol_frac: float,
                   nominal_obj: float, constraint_names: Optional[List[str]] = None,
                   contextual: bool = True, method: str = "",
                   score_fn: Optional[Callable[[float], tuple]] = None,
                   log: Callable = print) -> tuple[float, list]:
    """Score every knob (or reuse ``score_fn`` for checkpointing) and return
    ``(theta*, rows)`` where ``rows = [(knob, feas, obj), ...]``.

    ``theta*`` = argmax feasibility among knobs whose objective stays within
    ``os_tol_frac`` of ``nominal_obj`` in the problem's worse direction; ties break
    to the stronger knob (later grid index). Falls back to the best-objective knob
    if none pass the budget.
    """
    rows = []
    for knob in knob_grid:
        feas, obj = (score_fn(knob) if score_fn is not None
                     else cv_score_knob(build_solver, knob, folds, oracle, base,
                                        constraint_names, contextual))
        rows.append((knob, feas, obj))
        log(f"    [cv] {method}: knob={knob:.4g} feas={feas:.3f} obj={obj:.3f}", flush=True)

    sense = oracle.objective_sense
    if sense == "max":
        thresh = nominal_obj * (1.0 - os_tol_frac)
        ok = lambda o: np.isfinite(o) and o >= thresh - 1e-9
    else:
        thresh = nominal_obj * (1.0 + os_tol_frac)
        ok = lambda o: np.isfinite(o) and o <= thresh + 1e-9

    passing = [(i, k, f, o) for i, (k, f, o) in enumerate(rows)
               if np.isfinite(f) and ok(o)]
    if passing:
        # max feasibility; ties -> later grid index (stronger knob)
        best = max(passing, key=lambda r: (r[2], r[0]))
        theta = best[1]
        log(f"    [cv] {method}: theta*={theta:.4g} (feas={best[2]:.3f} obj={best[3]:.3f}; "
            f"budget {sense} obj {'>=' if sense=='max' else '<='} {thresh:.3f} "
            f"[nominal {nominal_obj:.3f}, tol {os_tol_frac:.2f}])", flush=True)
    else:
        finite = [(k, f, o) for k, f, o in rows if np.isfinite(o)]
        if not finite:
            raise ValueError(f"{method}: no finite CV scores; cannot select a knob")
        best = (min if sense == "max" else max)  # best objective as fallback
        theta = (max(finite, key=lambda r: r[2]) if sense == "max"
                 else min(finite, key=lambda r: r[2]))[0]
        log(f"    [cv] {method}: WARNING no knob meets the objective budget "
            f"(thresh {thresh:.3f}); falling back to best-objective knob theta*={theta:.4g}",
            flush=True)
    return theta, rows


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def load_score_checkpoint(path: str) -> dict:
    """Return ``{(method, knob): (feas, obj)}`` from a scores CSV, or ``{}``."""
    import pandas as pd
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[(str(r["method"]), float(r["knob"]))] = (float(r["feas"]), float(r["obj"]))
    return out


def append_score(path: str, method: str, knob: float, feas: float, obj: float) -> None:
    """Append one scored cell to the checkpoint CSV (header written once)."""
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["method", "knob", "feas", "obj"])
        w.writerow([method, knob, feas, obj])


def write_knobs(path: str, knobs: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(knobs, f, indent=2)
    print(f"Wrote robustness knobs -> {path}: {knobs}", flush=True)

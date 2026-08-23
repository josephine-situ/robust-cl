"""Feasibility-tuned nominal: the same nominal fit, against a tightened RHS.

The **baseline the robust methods have to beat**, and the cheapest possible way
to buy feasibility: keep the nominal model, keep the nominal MIP, and move the
constraint in by a margin::

    sum_m w_m f_m(x)  <=  rhs - m_c,        m_c = margin * sum_m |w_m| * scale(y_m)

``margin`` is the ONE dial, and it is the whole method. It answers the question
a robust method has to answer better: *how much of the reported feasibility is
the machinery, and how much is simply asking for less?* Nothing here models
label noise, propagates a set, or separates a scenario -- a robust method that
does not beat this curve is buying its feasibility the expensive way.

WHY THE MARGIN IS IN LABEL-SCALE UNITS
--------------------------------------
One dial per problem, not one per constraint. Gastric has five learned
constraints on five different toxicities; a single *absolute* margin would mean
five different things at once, and five separate margins would be five dials
fitted against the same feasibility number -- which is no longer a baseline, it
is a five-parameter fit. So the dial is dimensionless and each constraint's
tightening is scaled by its own outcome's ``scale(y_c)``, the out-of-fold
residual sd, via :func:`uncertainty.instance_label_scales` -- literally the same
estimator that sets D's radius ``R_c = rho * scale(y_c) * sqrt(n)`` and the same
one CP's tau is measured against. ``margin = 1`` therefore means "give up one
unexplained standard deviation of headroom on every constraint", directly
readable against ``rho = 1`` and ``tau = 1``.

The ``|w_m|`` weighting is the worst-case direction over the models inside one
constraint, matching what C-MICL does with its interval half-width
(``abs(w) * q * u``). It is also what keeps the SIGN right: the reactor states
its requirement as a lower bound and carries ``weight = -1`` against
``rhs = -50``, so ``|w|`` tightens it to ``F_C6H6 >= 50 + margin * scale`` rather
than loosening it.

WHAT THIS IS AND IS NOT
-----------------------
- **It faces no D.** Like :mod:`src.methods.cmicl`, and unlike cp / wrapper /
  robust_reg, nothing here reads ``rho``, so on ``run_rho_sweep.py`` its curve is
  FLAT in rho by construction. That is what it is for: a horizontal reference the
  shared-D curves are read against. Its own axis is the margin, swept by
  ``--ablate``.
- **Monotone in ``margin``, which is the point.** The feasible set shrinks weakly
  as the margin grows, so held-out feasibility is (up to fold noise) increasing
  and the objective weakly worsening -- ``m*`` at any target always exists, which
  is exactly what makes this a fair "tuned" baseline. The robust methods have no
  such guarantee: robust_reg's gastric feasibility *falls* with rho (2026-08-19
  deck).
- **It carries NO guarantee of any kind.** Not a coverage statement, not a
  chance constraint, not a worst case over a set. ``m`` is fitted to held-out
  feasibility and means only what that fit means -- which is also true of a
  fitted rho, and is the honest comparison.
- **The objective term is deliberately untouched.** A margin on a learned
  objective model adds ``margin * |a| * scale``, a CONSTANT: it shifts the
  reported objective without moving the argmin, so it would corrupt the column
  the methods are compared on while changing no decision. There is therefore no
  ``robustify_objective`` flag here, unlike cp / wrapper / cmicl -- the option
  would be vacuous rather than merely off.
- **Large margins go infeasible, not conservative.** On a percentile-scored
  outcome ``rhs - m_c`` eventually drops below anything the model can predict and
  the MIP has no solution. That is reported (see ``_unreachable_note``) and shows
  up as a falling solved fraction, which is what ``--min-solved`` guards. It is a
  real property of the method, not a failure to handle.
"""

import copy
import dataclasses
import time
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    DEFAULT_MIP_GAP,
    SolutionResult,
    resolve_constraint_config,
    train_constraint_models,
    build_decision_vars,
    add_problem_constraints,
    embed_constraints,
    build_and_set_objective,
)
from src.methods.uncertainty import instance_label_scales


def constraint_margins(instance: ProblemInstance,
                       margin: float,
                       model_type: str = "rf",
                       model_params: Optional[dict] = None,
                       scale_stat: str = "oof_sd",
                       seed: int = 42,
                       folds=None):
    """``({c_idx -> m_c}, {c_idx -> [(name, w, scale)]})`` -- the per-constraint
    tightening for one dimensionless ``margin``.

    ``m_c = margin * sum_m |w_m| * scale(y_m)`` over the constraint's non-objective
    models. Constraints that are purely an objective term get no entry: they never
    become a ``<=`` row, so there is no RHS to move.
    """
    model_config_map = {}
    config_idx = 0
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            model_config_map[id(model_data)] = resolve_constraint_config(
                instance, config_idx, model_type, model_params)
            config_idx += 1

    scales = instance_label_scales(instance, model_config_map,
                                   stat=scale_stat, seed=seed, folds=folds)

    margins, parts = {}, {}
    for c_idx, constraint in enumerate(instance.constraints):
        rows = [(constraint.name, float(md.weight), float(scales[id(md)]))
                for md in constraint.models_data if md.obj_weight == 0.0]
        if not rows:
            continue
        margins[c_idx] = float(margin) * sum(abs(w) * s for _n, w, s in rows)
        parts[c_idx] = rows
    return margins, parts


def _unreachable_note(constraint, m_c: float) -> Optional[str]:
    """Warn when ``rhs - m_c`` sits below anything the label range can produce.

    Only meaningful where ``label_bounds`` is set (gastric's five toxicities). It
    is a DIAGNOSTIC, not a check: tree and forest predictions are averages of
    training labels and so do stay inside the range, but a linear or MLP fit can
    predict outside it, so the note says "no achievable prediction" rather than
    "infeasible". The MIP's own status is the authority either way -- this only
    saves reading a bare `infeasible` and guessing why.
    """
    mds = [md for md in constraint.models_data if md.obj_weight == 0.0]
    if not mds or any(md.label_bounds is None for md in mds):
        return None
    best = sum(md.weight * (md.label_bounds[0] if md.weight > 0
                            else md.label_bounds[1])
               for md in mds)
    if constraint.rhs - m_c < best:
        return (f"{constraint.name}: tightened rhs {constraint.rhs - m_c:.4f} is "
                f"below the best value the label range allows ({best:.4f}) -- no "
                f"achievable prediction satisfies it")
    return None


def solve_margin(instance: ProblemInstance,
                 model_type: str = "rf",
                 model_params: dict = None,
                 margin: float = 0.0,
                 scale_stat: str = "oof_sd",
                 seed: int = 42,
                 rho: float = 0.0,
                 embedding_mode: str = "hard",
                 rf_alpha: float = 0.25,
                 mip_gap: float = DEFAULT_MIP_GAP) -> SolutionResult:
    """Nominal, solved against ``rhs - margin * scale(y_c)`` on every constraint.

    ``margin`` is the single dial, in unexplained-sd units (see the module
    docstring); ``margin = 0`` is exactly :func:`nominal.solve_nominal` -- same
    fit, same MIP, same solution -- which is what makes the baseline's own curve
    start at the nominal point rather than near it.

    ``seed`` reaches only the fold scheme behind ``scale(y_c)``, the same role it
    plays for D's radius. There are no draws to seed here.
    """
    start = time.time()
    print(f"    [margin] Training constraint models (margin={margin:g} "
          f"x scale(y), stat={scale_stat})...", flush=True)
    trained_constraints = train_constraint_models(instance, model_type, model_params)

    margins, parts = constraint_margins(
        instance, margin, model_type=model_type, model_params=model_params,
        scale_stat=scale_stat, seed=seed,
    )
    for c_idx, m_c in margins.items():
        constraint = instance.constraints[c_idx]
        detail = " + ".join(f"|{w:g}|x{s:.4f}" for _n, w, s in parts[c_idx])
        print(f"    [margin] {constraint.name}: rhs {constraint.rhs:.4f} -> "
              f"{constraint.rhs - m_c:.4f}  (m={m_c:.4f} = {margin:g} x [{detail}])",
              flush=True)
        note = _unreachable_note(constraint, m_c)
        if note:
            print(f"    [margin] NOTE: {note}", flush=True)

    # Tighten by rebuilding the constraint records on a SHALLOW copy: the
    # MLModelData objects are shared, so `id(md)`-keyed caches (the trained models
    # above, embed's X_ref) still line up, and the caller's instance is untouched.
    tightened = copy.copy(instance)
    tightened.constraints = [
        dataclasses.replace(c, rhs=c.rhs - margins[i]) if i in margins else c
        for i, c in enumerate(instance.constraints)
    ]

    opt = gp.Model("margin")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = mip_gap
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, tightened)
    models_embedded, _, obj_terms = embed_constraints(
        opt, x, tightened, trained_constraints,
        rho=rho, embedding_mode=embedding_mode, rf_alpha=rf_alpha,
        name_prefix="margin",
    )
    add_problem_constraints(opt, x, tightened)
    # Nominal objective: a margin on a learned objective term is a constant and
    # would move the reported objective without moving x*. See the module docstring.
    build_and_set_objective(opt, x, tightened, obj_terms)
    opt.update()

    print(f"    [margin] MIP built ({models_embedded} models embedded; "
          f"{opt.NumVars} vars / {opt.NumConstrs} constrs); solving...", flush=True)
    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        return SolutionResult(
            x_opt=np.array([v.X for v in x]),
            obj_value=opt.ObjVal,
            status="optimal",
            models_embedded=models_embedded,
            solve_time=elapsed,
            opt=opt,
            x=x,
        )
    return SolutionResult(
        x_opt=np.zeros(instance.n_features),
        obj_value=np.inf,
        status="infeasible",
        models_embedded=models_embedded,
        solve_time=elapsed,
        opt=opt,
        x=x,
    )

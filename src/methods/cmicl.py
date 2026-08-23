"""C-MICL: conformal mixed-integer constraint learning (Ovalle et al. 2025).

The fourth uncertainty-aware method, and the only one that does **not** face the
shared set D. Where ``cp`` / ``wrapper`` / ``robust_reg`` are handed one D and
differ only in what they do with it, C-MICL builds its own tightening out of
held-out residuals: no relabelings, no bank, no radius.

Per learned constraint model, on that constraint's own training rows:

1. split the rows into PROPER-TRAIN and CALIBRATION (one shared split, see
   :func:`_split_rows`);
2. fit ``h`` on proper-train -- this is the model that gets embedded;
3. fit the WIDTH model ``u`` on proper-train, target ``|y - h(x)|``;
4. score the held-out calibration rows with
   ``s_i = |y_i - h(x_i)| / max(u(x_i), floor)`` and take the split-conformal
   quantile ``q = s_(k)``, ``k = ceil((n_cal + 1) * (1 - alpha_eff))``;
5. embed ``h`` AND ``u``, and require the conservative end of the predictive
   interval to satisfy the constraint::

       sum_m [ w_m * h_m(x) + |w_m| * q_m * u_m(x) ]  <=  rhs

   which is ``sup { sum_m w_m f_m : f_m in [h_m - q_m u_m, h_m + q_m u_m] }``.

``alpha`` is the method's ONE dial, in the same role as CP's tau and the
wrapper's alpha: the miscoverage level of the predictive interval.

WHAT THIS IS AND IS NOT
-----------------------
- **The guarantee is marginal** over ``P_XY`` (and over the calibration draw): the
  interval covers ``y`` at a fresh exchangeable ``(x, y)`` with probability
  ``1 - alpha_eff``. It does NOT say ``x*`` is feasible. ``x*`` is an argmin, not
  a random draw, and it sits ON the constraint by construction -- exactly where a
  marginal statement says least (CLAUDE.md, Known gaps #8). Feasibility read off
  this method is the empirical consequence of that statement, not a delivered
  guarantee.
- **Exchangeability is assumed and is false on gastric**, whose folds are
  temporal by design (train = rows up to a cutoff year, val = the next). The
  split below is random *within* the fold's training rows, so coverage is
  marginal over the training years, not over the validation year. That is
  Known gap #6 and implementing the method does not make the assumption true.
- **h is fit on fewer rows than every other method's model.** Split conformal
  spends ``cal_frac`` of the training rows on calibration and does not give them
  back. Intrinsic to the method, not a handicap imposed here -- and the reason
  ``cal_frac`` is a structural setting rather than a second dial.
- **D plays no part.** On ``run_rho_sweep.py`` C-MICL's curve is FLAT IN RHO up to
  fold noise, by construction. That is why it is worth running there: it is the
  axis-free competitor the shared-D methods are read against, not another point
  on the same axis. Its own dial (alpha) is what moves it.

FIDELITY TO THE PAPER (arXiv:2506.03531, Sec. 4.1 and 5.1)
----------------------------------------------------------
Matched deliberately, so a disagreement with their Figure 1 is about the setting
and not about the recipe:

- the score is `s = |h(x_i) - y_i| / u(x_i)` with `u` trained on the ABSOLUTE
  RESIDUALS of `h` over the training rows (their two-step procedure) -- in-sample,
  as they specify;
- their quantile is `Quantile(s_1..s_N; (1-alpha)(1+1/N))`, which is the
  `ceil((N+1)(1-alpha))`-th order statistic used here -- the same number;
- their constraint is `[h(x) +- q u(x)] subset Y`, i.e. the whole interval must
  lie in the feasible region. For the reactor's lower bound `F >= 50` that is
  `h - q u >= 50`, which is exactly `w*h + |w|*q*u <= rhs` at `w = -1`,
  `rhs = -50`;
- `cal_frac = 0.2` is their 80/20 split;
- `alpha = 0.1` is their main-text target (they report `alpha = 0.05` in an
  appendix), and it is pinned to `1 - feas_target` here for the same reason.

**One deliberate deviation, forced by the stack.** Their `u` is "a ReLU NN with
two hidden layers, each with 32 units", shared across base models. A ReLU OUTPUT
layer is non-negative by construction; sklearn's `MLPRegressor` has a LINEAR
output, so the same architecture fitted to `|y - h(x)|` predicts negative widths
-- measured on the reactor (n=1000, 80/20, seed 42): min `u = -5.37`, 5th pct
`-2.14`, **26% of calibration rows on the floor**, score q90 **36.1** vs 3.0 for
the fallback, `q = 27.6`, mean half-width **3.9 sd(y)** on labels spanning 11-70.
That is the floor amplifying a negative prediction, not conformal being
conservative. The default is therefore the embedded model's own type/params
(`width_model_type: null`), which on the same split predicts `min u = +0.38` and
never touches the floor. `config.yaml` says how to reproduce their architecture.

**Not matched, and it is the protocol, not the method**: they average empirical
ground-truth feasibility over **100 randomly sampled cost vectors** with the
calibration set held fixed. This repo fixes `cost_vector` at ones on purpose (a
new `c` is a different problem, see `src/data/generate.py`), so our rate is over
**training/calibration draws at one `c`**. Both are "empirical ground-truth
feasibility"; they are not the same average, and a gap between the two numbers
should be read with that in front.

WIDTH MODEL
-----------
``u`` is fit in-sample on ``h``'s own proper-train residuals, which biases it
small. That costs nothing in validity -- split conformal keeps its coverage for
ANY deterministic score function, and ``q`` absorbs the bias globally -- but it
means ``u`` shapes the half-width better than it sizes it. ``width_floor_frac``
guards the other end: a linear or MLP width model can predict a NEGATIVE width,
which would loosen the constraint below nominal, so ``u`` is floored at
``width_floor_frac`` of the mean absolute proper-train residual. The floor is
applied identically when scoring the calibration rows and inside the MIP, so it
is part of the score function rather than a post-hoc fudge.

MULTIPLICITY
------------
``multiplicity="bonferroni"`` divides alpha by the number of learned constraints
so the JOINT statement holds at ``1 - alpha``; ``"none"`` (default, and what
Ovalle et al. report) leaves each constraint marginal at ``1 - alpha``. On
gastric's five toxicity constraints the Bonferroni level is often finer than the
calibration set can resolve (``ceil((n_cal + 1) * (1 - alpha/C)) > n_cal``), and
there is then no finite quantile to embed. That is reported as an infeasible
solve with the reason printed, never clipped to the largest score.
"""

import math
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
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    model_X_ref,
)
from src.models.train import train_model
from src.models.embed import embed_model


def _split_rows(n: int, cal_frac: float, seed: int):
    """One (proper-train, calibration) split of ``n`` rows, SHARED across outcomes.

    Gastric's outcomes are five labels on the same patients, so splitting each
    one independently would put a patient in the fit for one toxicity and in the
    calibration set for another: no single set of trials would be held out from
    the problem as a whole, and the joint statement would be about nothing in
    particular. Keying the permutation on ``(seed, n)`` alone gives every outcome
    of the same length the same split.
    """
    n_cal = int(round(cal_frac * n))
    n_cal = max(1, min(n - 1, n_cal))
    perm = np.random.default_rng(seed + 1000 * n).permutation(n)
    return perm[n_cal:], perm[:n_cal]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile ``s_(k)``, ``k = ceil((n + 1) * (1 - alpha))``.

    Returns ``inf`` when ``k > n``: the calibration set is too small to certify
    that level, so no finite tightening carries the guarantee. Clipping to
    ``max(scores)`` instead would name a level the data cannot support, which is
    why the caller fails the solve rather than pretending.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[max(k, 1) - 1])


def calibrate_conformal_model(X, y, model_type, model_params, *,
                              alpha, cal_frac, seed,
                              width_model_type=None, width_model_params=None,
                              width_floor_frac=0.05, label=""):
    """Fit ``h``, fit the width model ``u``, calibrate ``q``.

    Returns ``(h, u, q, floor, info)``. ``h`` sees the proper-train rows ONLY --
    holding the calibration rows out of it is what makes their scores
    exchangeable with a fresh point's.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    tr_idx, cal_idx = _split_rows(len(y), cal_frac, seed)

    h = train_model(X[tr_idx], y[tr_idx], model_type, model_params)

    resid_tr = np.abs(y[tr_idx] - h.predict(X[tr_idx]))
    # Positive unless the fit is exact; the 1e-12 keeps the score's denominator
    # finite in that degenerate case.
    floor = max(float(width_floor_frac) * float(np.mean(resid_tr)), 1e-12)
    u = train_model(X[tr_idx], resid_tr,
                    width_model_type or model_type,
                    width_model_params if width_model_type else model_params)

    u_cal = np.maximum(u.predict(X[cal_idx]), floor)
    scores = np.abs(y[cal_idx] - h.predict(X[cal_idx])) / u_cal
    q = conformal_quantile(scores, alpha)

    half = q * float(np.mean(u_cal)) if np.isfinite(q) else float("inf")
    sd_y = float(np.std(y))
    info = {
        "n_train": int(len(tr_idx)), "n_cal": int(len(cal_idx)),
        "alpha_eff": float(alpha), "q": q, "floor": floor,
        "mean_width": float(np.mean(u_cal)), "mean_half_width": half,
        "sd_y": sd_y,
    }
    print(
        f"    [cmicl] {label}: n_fit={info['n_train']} n_cal={info['n_cal']} "
        f"alpha_eff={alpha:.4g} q={q:.4g} mean_u={info['mean_width']:.4g} "
        f"-> mean half-width {half:.4g} ({half / max(sd_y, 1e-12):.2f} sd(y))",
        flush=True,
    )
    return h, u, q, floor, info


def solve_cmicl(instance: ProblemInstance,
                model_type: str = "rf",
                model_params: dict = None,
                alpha: float = 0.1,
                cal_frac: float = 0.25,
                width_model_type: Optional[str] = None,
                width_model_params: Optional[dict] = None,
                width_floor_frac: float = 0.05,
                multiplicity: str = "none",
                seed: int = 42,
                rho: float = 0.0,
                robustify_objective: bool = False,
                mip_gap: float = DEFAULT_MIP_GAP) -> SolutionResult:
    """Ovalle et al.'s conformal MICL: embed the conservative end of a
    split-conformal predictive interval instead of the point prediction.

    ``alpha`` is the single dial. ``cal_frac``, ``width_*`` and ``multiplicity``
    are structural settings with one production value each, in the same sense as
    CP's ``separation``: they are not swept.

    ``robustify_objective`` (default ``False``, matching CP and the wrapper)
    leaves the learned objective a plain nominal fit on ALL training rows, so the
    objective column stays comparable across methods. Under ``True`` the
    objective term takes its own conformal bound, in whichever direction makes it
    worse, and is then fit on the proper-train rows like every constraint model.
    """
    start = time.time()
    if multiplicity not in ("none", "bonferroni"):
        raise ValueError(f"unknown multiplicity: {multiplicity!r}")

    constraint_idxs = [
        c_idx for c_idx, c in enumerate(instance.constraints)
        if not any(md.obj_weight != 0 for md in c.models_data)
    ]
    n_units = max(len(constraint_idxs), 1)
    alpha_eff = (alpha / n_units) if multiplicity == "bonferroni" else alpha

    bonf = (f" -> alpha_eff={alpha_eff:.4g} over {n_units} constraints"
            if multiplicity == "bonferroni" else "")
    print(
        f"    [cmicl] Calibrating (alpha={alpha:g}, "
        f"multiplicity={multiplicity}{bonf}, cal_frac={cal_frac:g})...",
        flush=True,
    )

    # ---- calibrate one (h, u, q) per constraint model ---------------------
    fitted = {}          # id(model_data) -> (h, u, q, floor); u None = nominal
    warned_multi_clip = False
    config_idx = 0
    for constraint in instance.constraints:
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)
        for model_data in constraint.models_data:
            md_id = id(model_data)
            m_type, m_params = resolve_constraint_config(
                instance, config_idx, model_type, model_params)
            config_idx += 1
            if md_id in fitted:
                continue
            if is_obj and not robustify_objective:
                # Nominal objective, fit on ALL rows: the split belongs to the
                # conformal CONSTRAINT machinery, and shrinking the objective fit
                # too would move the objective column for a reason that has
                # nothing to do with the method being compared.
                fitted[md_id] = (
                    train_model(model_data.X_train, model_data.y_train,
                                m_type, m_params),
                    None, 0.0, 0.0,
                )
                continue
            h, u, q, floor, _info = calibrate_conformal_model(
                model_data.X_train, model_data.y_train, m_type, m_params,
                alpha=alpha_eff, cal_frac=cal_frac, seed=seed,
                width_model_type=width_model_type,
                width_model_params=width_model_params,
                width_floor_frac=width_floor_frac,
                label=constraint.name,
            )
            if not np.isfinite(q):
                print(
                    f"    [cmicl] INFEASIBLE: {constraint.name} needs "
                    f"ceil((n_cal+1)*(1-{alpha_eff:.4g})) <= n_cal calibration "
                    f"rows and has {int(round(cal_frac * len(model_data.y_train)))}. "
                    f"Raise alpha, raise cal_frac, or drop "
                    f"multiplicity='bonferroni'.",
                    flush=True,
                )
                return SolutionResult(
                    x_opt=np.zeros(instance.n_features), obj_value=np.inf,
                    status="infeasible", models_embedded=0,
                    solve_time=time.time() - start,
                )
            fitted[md_id] = (h, u, q, floor)

    # ---- build the MIP ----------------------------------------------------
    opt = gp.Model("cmicl")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = mip_gap
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    models_embedded = 0
    obj_terms = []
    n_vacuous = 0

    def _embed(ml_model, prefix, c_idx, m_idx):
        nonlocal models_embedded
        models_embedded += 1
        return embed_model(
            opt, ml_model, x, instance.variable_lb, instance.variable_ub,
            name_prefix=prefix, rho=rho,
            X_ref=model_X_ref(instance, c_idx, m_idx),
        )

    def _width_var(u_model, floor, prefix, c_idx, m_idx):
        """``max(u(x), floor)``, with NO binaries.

        ``u_eff`` enters with a positive coefficient on the ``<=`` side of every
        constraint it appears in, so a smaller value is always weakly better for
        the optimizer; bounding it below by both ``u(x)`` and the floor therefore
        pins it at their max at any optimum. Same direction, same trick, for the
        objective under ``robustify_objective``.
        """
        u_raw = _embed(u_model, prefix, c_idx, m_idx)
        u_eff = opt.addVar(lb=floor, name=f"{prefix}_ueff")
        opt.addConstr(u_eff >= u_raw, name=f"{prefix}_ufloor")
        return u_eff

    for c_idx, constraint in enumerate(instance.constraints):
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)
        single = len(constraint.models_data) == 1

        if is_obj:
            for m_idx, model_data in enumerate(constraint.models_data):
                h, u, q, floor = fitted[id(model_data)]
                a = model_data.obj_weight * model_data.weight
                h_var = _embed(h, f"cmicl_obj_c{c_idx}_m{m_idx}", c_idx, m_idx)
                if u is None:
                    obj_terms.append(a * h_var)
                else:
                    u_eff = _width_var(u, floor,
                                       f"cmicl_objw_c{c_idx}_m{m_idx}",
                                       c_idx, m_idx)
                    obj_terms.append(a * h_var + abs(a) * q * u_eff)
            continue

        terms = []
        vacuous = False
        for m_idx, model_data in enumerate(constraint.models_data):
            h, u, q, floor = fitted[id(model_data)]
            w = model_data.weight
            h_var = _embed(h, f"cmicl_c{c_idx}_m{m_idx}", c_idx, m_idx)
            u_eff = _width_var(u, floor, f"cmicl_w_c{c_idx}_m{m_idx}",
                               c_idx, m_idx)
            terms.append(w * h_var + abs(w) * q * u_eff)

            # The interval is intersected with the label range -- the conformal
            # analogue of uncertainty.clip_labels. For a SINGLE-model constraint
            # that clip is one of exactly two things, and neither needs a binary:
            # the worst clipped value is w*hi (w > 0) or w*lo (w < 0), so either
            # it already satisfies the rhs -- the constraint is vacuous and is
            # dropped -- or the clip cannot bind and the raw bound is exactly
            # equivalent. Every instance in this repo has one model per
            # constraint. With more than one the exact clip needs a per-term
            # min/max; the raw bound stays VALID there (it dominates the clipped
            # one), only more conservative, so it is used and said so.
            if model_data.label_bounds is not None:
                lo, hi = model_data.label_bounds
                worst_clipped = w * hi if w > 0 else w * lo
                if single:
                    if worst_clipped <= constraint.rhs:
                        vacuous = True
                elif not warned_multi_clip:
                    warned_multi_clip = True
                    print(
                        f"    [cmicl] NOTE: {constraint.name} combines several "
                        f"models with label_bounds; the interval clip is dropped "
                        f"(conservative) rather than encoded exactly.",
                        flush=True,
                    )
        if vacuous:
            # min(h + q u, hi) <= rhs holds at every x: the label range alone
            # satisfies it. Adding the raw bound instead would impose a
            # constraint the method does not make.
            n_vacuous += 1
            continue
        opt.addConstr(gp.quicksum(terms) <= constraint.rhs,
                      name=f"cmicl_constr_{c_idx}")

    add_problem_constraints(opt, x, instance)
    build_and_set_objective(opt, x, instance, obj_terms)
    opt.update()
    vac = (f", {n_vacuous} constraint(s) vacuous under the label range"
           if n_vacuous else "")
    print(
        f"    [cmicl] MIP built (alpha_eff={alpha_eff:.4g}, "
        f"{models_embedded} models embedded{vac}; "
        f"{opt.NumVars} vars / {opt.NumConstrs} constrs); solving...",
        flush=True,
    )
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

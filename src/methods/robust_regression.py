"""
Robust regression via a label-robust training counterpart (Bertsimas, Dunn,
Pawlowski, Zhuo 2019, "Robust Classification", Section 5), adapted from
classification (Gamma label flips) to our regression outcomes with a bounded
*additive* label uncertainty set

    D = { dy : ||dy||_1 <= gamma, |dy_i| <= eps }.

For each constraint outcome we train a single model that solves the label-robust
counterpart ``min_theta max_{dy in D} loss(theta; X, y + dy)`` and embed it
nominally (one model per constraint) -- a *robust model*, in contrast to CP which
robustifies the *decision*. Dispatch on model class:

- linear (squared loss): exact convex counterpart solved as a Gurobi QP (no
  iteration). The inner max is attained at a vertex of D, giving the objective
  ``(1/2n)[||r||^2 + 2 eps * S_m(|r|)] + elastic-net penalty`` where ``S_m`` is the
  sum of the ``m = gamma/eps`` largest absolute residuals (a convex, LP-representable
  term).
- tree / xgb / other: adversarial-training min-max loop (no tractable closed form),
  reusing ``worst_case_label_shift`` + ``retrain_on_perturbed``.

The radius ``eps`` is scaled per outcome by the label standard deviation so a single
unitless ``label_eps`` knob applies across percentile-toxicity and OS-months targets.
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    SolutionResult,
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    embed_constraints,
    resolve_constraint_config,
)
from src.models.train import train_model, retrain_on_perturbed
from src.methods.uncertainty import label_scale
from src.utils.perturbations import worst_case_label_shift


def _label_robust_linear(X, y, params, eps_abs, gamma):
    """Exact convex label-robust ElasticNet counterpart, solved as a QP.

    Returns a fitted ``Pipeline(StandardScaler, ElasticNet)`` whose coefficients
    minimize ``(1/2n)[sum r_i^2 + 2 eps * (sum of m largest |r_i|)] + enet_penalty``
    with ``r = y - beta^T z - b0`` on standardized features ``z`` and
    ``m = gamma / eps_abs``. Matches sklearn's ElasticNet objective scaling so it is
    comparable to the nominal fit.
    """
    n, p = X.shape
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    alpha = float(params.get("alpha", 1.0))
    l1_ratio = float(params.get("l1_ratio", 0.5))
    m = gamma / eps_abs if eps_abs > 0 else 0.0        # number of fully-shifted points

    qp = gp.Model("robust_reg_linear")
    qp.Params.OutputFlag = 0
    beta = qp.addVars(p, lb=-GRB.INFINITY, name="beta")
    b0 = qp.addVar(lb=-GRB.INFINITY, name="b0")
    r = qp.addVars(n, lb=-GRB.INFINITY, name="r")
    a = qp.addVars(n, lb=0.0, name="a")                # a_i = |r_i|
    q = qp.addVars(n, lb=0.0, name="q")                # top-m sum epigraph
    t = qp.addVar(lb=-GRB.INFINITY, name="t")
    babs = qp.addVars(p, lb=0.0, name="babs")          # |beta_j| for the l1 term

    for i in range(n):
        qp.addConstr(r[i] == y[i] - gp.quicksum(beta[j] * Z[i, j] for j in range(p)) - b0)
        qp.addConstr(a[i] >= r[i])
        qp.addConstr(a[i] >= -r[i])
        qp.addConstr(q[i] >= a[i] - t)                 # S_m = min m*t + sum q_i
    for j in range(p):
        qp.addConstr(babs[j] >= beta[j])
        qp.addConstr(babs[j] >= -beta[j])

    sq = gp.quicksum(r[i] * r[i] for i in range(n))
    top_m = m * t + gp.quicksum(q[i] for i in range(n))
    data_term = (1.0 / (2.0 * n)) * (sq + 2.0 * eps_abs * top_m)
    l1_term = alpha * l1_ratio * gp.quicksum(babs[j] for j in range(p))
    l2_term = 0.5 * alpha * (1.0 - l1_ratio) * gp.quicksum(beta[j] * beta[j] for j in range(p))
    qp.setObjective(data_term + l1_term + l2_term, GRB.MINIMIZE)
    qp.optimize()

    beta_val = np.array([beta[j].X for j in range(p)], dtype=float)
    b0_val = float(b0.X)

    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    enet.coef_ = beta_val
    enet.intercept_ = b0_val
    enet.n_features_in_ = p
    return Pipeline([("scaler", scaler), ("model", enet)])


def _label_robust_loop(X, y, m_type, m_params, eps_abs, gamma, K):
    """Adversarial-training approximation of the label-robust counterpart for model
    classes with no closed form (trees, xgb, mlp). Alternates: find the worst-case
    label shift for the current model, retrain on the shifted labels, repeat."""
    model = train_model(X, y, m_type, m_params)
    if eps_abs <= 0 or gamma <= 0:
        return model
    prev = None
    for _ in range(K):
        residuals = y - model.predict(X)
        delta = worst_case_label_shift(residuals, eps_abs, gamma)
        if prev is not None and np.allclose(delta, prev, atol=1e-12):
            break
        model = retrain_on_perturbed(X, y, delta, m_type, m_params)
        prev = delta
    return model


def _train_label_robust_model(X, y, m_type, m_params, label_eps, budget_frac, K,
                              scale_stat="sd"):
    """One label-robust model for a single outcome; dispatch on model class."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    scale = label_scale(y, stat=scale_stat)
    eps_abs = label_eps * scale                        # per-outcome radius (unitless knob)
    gamma = budget_frac * n * eps_abs
    if label_eps <= 0 or scale == 0.0:
        return train_model(X, y, m_type, m_params)
    if m_type == "linear":
        return _label_robust_linear(X, y, m_params or {}, eps_abs, gamma)
    return _label_robust_loop(X, y, m_type, m_params, eps_abs, gamma, K)


def _train_coherent_label_robust(specs, label_eps, budget_frac, K, scale_stat="sd"):
    """Label-robust models for ALL outcomes against one **shared** row set.

    The incoherent default (``solve_robust_regression``'s per-outcome loop) lets
    each outcome pick its own worst rows, so the realized adversary is a different
    relabeling per constraint. Coherent instead picks a single row set S once per
    iteration -- by mean normalized |residual| across outcomes -- and shifts every
    outcome's labels on S in its own worst direction. That is the same coherence
    the scenario bank and the wrapper's shared ``z[p]`` express: one plausible
    relabeling of the trial, moving all outcomes together.

    Trade-off, stated rather than hidden: with delta chosen jointly the exact QP
    counterpart (:func:`_label_robust_linear`) no longer applies -- its inner max
    assumes the adversary optimizes that outcome alone -- so every model class
    retrains on shifted labels here. Coherence is a crossed factor, so the exact
    linear result stays available from the incoherent arm.

    ``specs`` is ``[(X, y, m_type, m_params), ...]``; returns models in that order.
    """
    ys = [np.asarray(y, dtype=float) for _, y, _, _ in specs]
    n = len(ys[0])
    if any(len(y) != n for y in ys):
        raise ValueError("coherent robust regression needs one shared row set: "
                         "all outcomes must have the same number of training rows")
    scales = [label_scale(y, stat=scale_stat) for y in ys]
    models = [train_model(X, y, mt, mp) for (X, _, mt, mp), y in zip(specs, ys)]
    if label_eps <= 0 or not any(s > 0 for s in scales):
        return models

    m = max(1, min(n, int(round(budget_frac * n))))
    prev = None
    for _ in range(K):
        # Rank rows by mean NORMALIZED |residual| so no single large-scale outcome
        # (OS in months vs percentile toxicities) decides the shared row set.
        norm = np.zeros(n)
        resid = []
        for (X, _, _, _), y, model, s in zip(specs, ys, models, scales):
            r = y - model.predict(X)
            resid.append(r)
            norm += np.abs(r) / (s if s > 0 else 1.0)
        S = np.argsort(-norm / len(specs))[:m]
        if prev is not None and np.array_equal(np.sort(S), prev):
            break
        prev = np.sort(S)
        models = []
        for (X, _, mt, mp), y, r, s in zip(specs, ys, resid, scales):
            delta = np.zeros(n)
            # Each outcome moves in ITS OWN worst direction on the shared rows.
            delta[S] = (label_eps * s) * np.where(r[S] >= 0, 1.0, -1.0)
            models.append(retrain_on_perturbed(X, y, delta, mt, mp))
    return models


def solve_robust_regression(
        instance: ProblemInstance,
        model_type: str = "rf",
        model_params: dict = None,
        label_eps: float = 0.1,
        budget_frac: float = 0.5,
        K: int = 5,
        rho: float = 0.0,
        embedding_mode: str = "hard",
        rf_alpha: float = 0.25,
        seed: int = 42,
        coherent: bool = None,
        uncertainty_set=None,
        scale_stat: str = None,
        **_ignored) -> SolutionResult:
    """Train a label-robust model per outcome (Bertsimas et al. counterpart), embed
    them nominally, and solve the constraint-learning MIP.

    ``coherent`` (default: D's setting) makes the adversary pick ONE shared row
    set across outcomes instead of a separate worst set per outcome -- the same
    coherence flag the scenario bank and the wrapper take. ``label_eps`` remains a
    multiplier on ``scale(y)`` either way, so the CV knob grid is unchanged.
    """
    start = time.time()
    if coherent is None:
        coherent = bool(getattr(uncertainty_set, "coherent", False))
    if scale_stat is None:
        scale_stat = str(getattr(uncertainty_set, "scale_stat", "sd"))
    print(
        f"    [robust_reg] Training label-robust models "
        f"(label_eps={label_eps:.3f}, budget_frac={budget_frac}, K={K}, "
        f"coherent={coherent})...",
        flush=True,
    )

    trained_constraints = []
    config_idx = 0
    if coherent:
        # One shared adversary over every outcome; see _train_coherent_label_robust.
        specs, layout = [], []
        for constraint in instance.constraints:
            row = []
            for model_data in constraint.models_data:
                m_type, m_params = resolve_constraint_config(
                    instance, config_idx, model_type, model_params
                )
                specs.append((model_data.X_train, model_data.y_train, m_type, m_params))
                row.append(model_data)
                config_idx += 1
            layout.append(row)
        models = _train_coherent_label_robust(
            specs, label_eps, budget_frac, K, scale_stat=scale_stat
        )
        it = iter(models)
        trained_constraints = [
            [(md.weight, next(it), md.obj_weight) for md in row] for row in layout
        ]
    else:
        for constraint in instance.constraints:
            row = []
            for model_data in constraint.models_data:
                m_type, m_params = resolve_constraint_config(
                    instance, config_idx, model_type, model_params
                )
                model = _train_label_robust_model(
                    model_data.X_train, model_data.y_train,
                    m_type, m_params, label_eps, budget_frac, K,
                    scale_stat=scale_stat,
                )
                row.append((model_data.weight, model, model_data.obj_weight))
                config_idx += 1
            trained_constraints.append(row)

    opt = gp.Model("robust_regression")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 0.01
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    models_embedded, _, obj_terms = embed_constraints(
        opt, x, instance, trained_constraints,
        rho=rho, embedding_mode=embedding_mode, rf_alpha=rf_alpha,
        name_prefix="robust_reg",
    )
    add_problem_constraints(opt, x, instance)
    build_and_set_objective(opt, x, instance, obj_terms)

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

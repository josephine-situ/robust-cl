"""
Nominal constraint learning: single model, no robustness.

min c'x  s.t.  f(x; theta*) <= b,  x in X
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Optional, List, Tuple

from sklearn.ensemble import RandomForestRegressor

from src.data.generate import ProblemInstance
from src.models.train import train_model, ModelType
from src.models.embed import embed_model, embed_single_tree
from src.utils.trust_region import add_trust_region


# Relative MIP gap for EVERY method's solve, the CP cut loop, and the
# prescribe-time re-solve. It is shared on purpose: the methods are compared on
# their objective, so a method solved to 1% on a gastric objective of ~10 carries
# ~0.1 months of solver slack -- the same order as the differences being reported.
# 1e-4 is the value the cut loop needs (scenario distances are ~0.007), so it sets
# the floor for everyone. Override once, via `optimization.mip_gap` in config.yaml.
DEFAULT_MIP_GAP = 1e-4


def resolve_mip_gap(config: dict) -> float:
    """The one MIP gap for a run, from ``optimization.mip_gap``.

    ``methods.cp.mip_gap`` is read as a legacy fallback so old configs still load;
    it used to be CP-only, which is exactly the asymmetry this removes.
    """
    opt_cfg = (config or {}).get("optimization", {}) or {}
    if "mip_gap" in opt_cfg:
        return float(opt_cfg["mip_gap"])
    cp_cfg = ((config or {}).get("methods", {}) or {}).get("cp", {}) or {}
    return float(cp_cfg.get("mip_gap", DEFAULT_MIP_GAP))


@dataclass
class SolutionResult:
    """Result from solving a constraint learning problem."""
    x_opt: np.ndarray
    obj_value: float
    status: str
    models_embedded: int
    solve_time: float
    opt: gp.Model = None
    x: list = None
    iterations: Optional[int] = None


def resolve_constraint_config(instance: ProblemInstance,
                              config_idx: int,
                              default_type: str,
                              default_params: dict) -> Tuple[str, dict]:
    if instance.constraint_model_configs and config_idx < len(instance.constraint_model_configs):
        cfg = instance.constraint_model_configs[config_idx]
        return cfg.get("model_type", default_type), cfg.get("model_params", default_params)
    return default_type, default_params


def train_constraint_models(instance: ProblemInstance,
                            model_type: str = "rf",
                            model_params: dict = None) -> List[List[tuple]]:
    """Train one model per MLModelData entry; returns (weight, model, obj_weight) rows."""
    trained_models_cache = {}
    trained_constraints = []
    config_idx = 0
    for constraint in instance.constraints:
        row = []
        for model_data in constraint.models_data:
            md_id = id(model_data)
            if md_id not in trained_models_cache:
                m_type, m_params = resolve_constraint_config(
                    instance, config_idx, model_type, model_params
                )
                trained_models_cache[md_id] = train_model(
                    model_data.X_train, model_data.y_train, m_type, m_params
                )
            row.append((
                model_data.weight,
                trained_models_cache[md_id],
                model_data.obj_weight,
            ))
            config_idx += 1
        trained_constraints.append(row)
    return trained_constraints


def build_decision_vars(opt: gp.Model, instance: ProblemInstance) -> list:
    d = instance.n_features
    binary = set(instance.binary_var_indices or [])
    x = []
    for j in range(d):
        vtype = GRB.BINARY if j in binary else GRB.CONTINUOUS
        x.append(opt.addVar(
            lb=instance.variable_lb[j],
            ub=instance.variable_ub[j],
            vtype=vtype,
            name=f"x_{j}",
        ))
    return x


def add_domain_constraints(opt: gp.Model, x, instance: ProblemInstance) -> None:
    d = instance.n_features
    for k, dc in enumerate(instance.domain_constraints):
        opt.addConstr(
            gp.quicksum(dc.coeffs[j] * x[j] for j in range(d)) <= dc.rhs,
            name=f"domain_{k}",
        )


def add_problem_constraints(opt: gp.Model, x, instance: ProblemInstance) -> None:
    add_domain_constraints(opt, x, instance)
    add_trust_region(opt, x, instance)


def _add_rf_tree_violation(opt, rf_model, x, instance, rhs, alpha, prefix, rho, M_val=1e5):
    """Embed each RF tree; at least (1-alpha) fraction must satisfy f_t <= rhs."""
    T = len(rf_model.estimators_)
    z = opt.addVars(T, vtype=GRB.BINARY, name=f"{prefix}_z")
    for t, tree in enumerate(rf_model.estimators_):
        f_t = embed_single_tree(
            opt, tree, x, instance.variable_lb, instance.variable_ub,
            name_prefix=f"{prefix}_t{t}", rho=rho,
        )
        opt.addConstr(f_t <= rhs + M_val * (1 - z[t]), name=f"{prefix}_ind_{t}")
    opt.addConstr(
        (1.0 / T) * gp.quicksum(z[t] for t in range(T)) >= 1 - alpha,
        name=f"{prefix}_chance",
    )
    return T


def model_X_ref(instance: ProblemInstance, c_idx: int, m_idx: int):
    """Training matrix behind one constraint model, for embed_model's X_ref.

    Only used to clamp tree split bands away from real rows, so a stale or
    resampled matrix is harmless -- it can cost a row its leaf but never
    misroutes one (see the SPLIT_EPS note in src/models/embed.py).
    """
    try:
        return instance.constraints[c_idx].models_data[m_idx].X_train
    except (AttributeError, IndexError, TypeError):
        return None


def embed_constraints(opt: gp.Model,
                      x,
                      instance: ProblemInstance,
                      trained_constraints: List[List[tuple]],
                      *,
                      rho: float = 0.0,
                      embedding_mode: str = "hard",
                      rf_alpha: float = 0.25,
                      name_prefix: str = "nominal") -> Tuple[int, dict, list]:
    """
    Embed learned models as constraints / objective terms.

    Returns (models_embedded, embedded_cache, obj_terms).
    """
    models_embedded = 0
    embedded_cache = {}
    obj_terms = []

    for c_idx, constraint_models in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        f_pred_vars = []

        for m_idx, (weight, ml_model, obj_weight) in enumerate(constraint_models):
            prefix = f"{name_prefix}_c{c_idx}_m{m_idx}"
            rhs = constraint.rhs

            if obj_weight != 0.0:
                m_id = id(ml_model)
                if m_id not in embedded_cache:
                    embedded_cache[m_id] = embed_model(
                        opt, ml_model, x, instance.variable_lb, instance.variable_ub,
                        name_prefix=prefix, rho=rho,
                        X_ref=model_X_ref(instance, c_idx, m_idx),
                    )
                    models_embedded += 1
                obj_terms.append(obj_weight * embedded_cache[m_id])
            elif (
                embedding_mode == "tree_violation"
                and isinstance(ml_model, RandomForestRegressor)
            ):
                models_embedded += _add_rf_tree_violation(
                    opt, ml_model, x, instance, rhs, rf_alpha, prefix, rho,
                )
            else:
                m_id = id(ml_model)
                if m_id not in embedded_cache:
                    embedded_cache[m_id] = embed_model(
                        opt, ml_model, x, instance.variable_lb, instance.variable_ub,
                        name_prefix=prefix, rho=rho,
                        X_ref=model_X_ref(instance, c_idx, m_idx),
                    )
                    models_embedded += 1
                f_pred_vars.append(weight * embedded_cache[m_id])

        if f_pred_vars:
            opt.addConstr(
                gp.quicksum(f_pred_vars) <= rhs,
                name=f"{name_prefix}_constr_{c_idx}",
            )

    return models_embedded, embedded_cache, obj_terms


def build_and_set_objective(opt: gp.Model, x, instance: ProblemInstance, obj_terms: list):
    """Set the objective in epigraph form: ``min c'x + t`` s.t. ``t >= sum(obj_terms)``.

    The epigraph variable ``t`` upper-bounds the learned objective contribution,
    so it can be robustified by adding worst-case cuts ``sum(obj_terms^s) <= t``
    (each raises ``t`` and makes the objective more conservative). When there is
    no learned objective term the objective stays the plain linear ``c'x``.

    Returns the epigraph variable ``t`` (or ``None`` for a purely linear objective).
    """
    d = instance.n_features
    base_cost = gp.quicksum(instance.cost_vector[j] * x[j] for j in range(d))
    if not obj_terms:
        opt.setObjective(base_cost, GRB.MINIMIZE)
        return None
    t = opt.addVar(lb=-GRB.INFINITY, name="t_obj")
    opt.addConstr(t >= gp.quicksum(obj_terms), name="epigraph_obj")
    opt.setObjective(base_cost + t, GRB.MINIMIZE)
    return t


def build_and_set_robust_objective(opt: gp.Model, x, instance: ProblemInstance,
                                   obj_term_scenarios: list):
    """Worst-case epigraph objective ``min c'x + t`` with ``t >= sum(terms)`` for
    **each** scenario in ``obj_term_scenarios``.

    Each entry of ``obj_term_scenarios`` is a list of objective terms for one
    plausible relabeling (e.g. one bootstrap replicate). Minimizing ``t`` then
    drives it to the most pessimistic scenario, robustifying the learned
    objective the same way CP raises its epigraph floor. With no scenarios the
    objective stays the plain linear ``c'x``.

    Returns the epigraph variable ``t`` (or ``None``).
    """
    d = instance.n_features
    base_cost = gp.quicksum(instance.cost_vector[j] * x[j] for j in range(d))
    scenarios = [terms for terms in obj_term_scenarios if terms]
    if not scenarios:
        opt.setObjective(base_cost, GRB.MINIMIZE)
        return None
    t = opt.addVar(lb=-GRB.INFINITY, name="t_obj")
    for k, terms in enumerate(scenarios):
        opt.addConstr(t >= gp.quicksum(terms), name=f"epigraph_obj_s{k}")
    opt.setObjective(base_cost + t, GRB.MINIMIZE)
    return t


def solve_nominal(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  rho: float = 0.0,
                  embedding_mode: str = "hard",
                  rf_alpha: float = 0.25,
                  mip_gap: float = DEFAULT_MIP_GAP) -> SolutionResult:
    """Solve the nominal constraint learning problem."""
    import time

    start = time.time()
    print(
        f"    [nominal] Training constraint models "
        f"(embedding_mode={embedding_mode})...",
        flush=True,
    )
    trained_constraints = train_constraint_models(instance, model_type, model_params)

    opt = gp.Model("nominal")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = mip_gap
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    models_embedded, _, obj_terms = embed_constraints(
        opt, x, instance, trained_constraints,
        rho=rho, embedding_mode=embedding_mode, rf_alpha=rf_alpha,
    )
    add_problem_constraints(opt, x, instance)
    build_and_set_objective(opt, x, instance, obj_terms)

    print(
        f"    [nominal] MIP built ({models_embedded} models embedded); solving...",
        flush=True,
    )
    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        x_opt = np.array([v.X for v in x])
        return SolutionResult(
            x_opt=x_opt,
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

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
    return [
        opt.addVar(lb=instance.variable_lb[j], ub=instance.variable_ub[j], name=f"x_{j}")
        for j in range(d)
    ]


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


def _add_rf_tree_violation(opt, rf_model, x, instance, rhs, alpha, prefix, rho, M_val=1e4):
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
                    )
                    models_embedded += 1
                f_pred_vars.append(weight * embedded_cache[m_id])

        if f_pred_vars:
            opt.addConstr(
                gp.quicksum(f_pred_vars) <= rhs,
                name=f"{name_prefix}_constr_{c_idx}",
            )

    return models_embedded, embedded_cache, obj_terms


def build_and_set_objective(opt: gp.Model, x, instance: ProblemInstance, obj_terms: list) -> None:
    d = instance.n_features
    base_cost = gp.quicksum(instance.cost_vector[j] * x[j] for j in range(d))
    opt.setObjective(base_cost + gp.quicksum(obj_terms), GRB.MINIMIZE)


def solve_nominal(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  rho: float = 0.0,
                  embedding_mode: str = "hard",
                  rf_alpha: float = 0.25) -> SolutionResult:
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
    opt.Params.MIPGap = 0.01
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

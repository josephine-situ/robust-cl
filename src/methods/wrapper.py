"""
Maragno et al. (2025) model wrapper approach.

Train P estimators (bootstrap or different methods) on the same
data. Require at least (1 - alpha) * P satisfy the constraint.

h_i(x) <= tau + M(1 - z_i)    for i = 1,...,P
(1/P) sum z_i >= 1 - alpha
z_i in {0, 1}
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from sklearn.ensemble import RandomForestRegressor

from src.data.generate import ProblemInstance
from src.methods.nominal import SolutionResult
from src.models.train import train_model
from src.models.embed import embed_model, embed_single_tree
from src.utils.trust_region import add_trust_region


def _train_models_for_instance(instance, model_type, model_params):
    """Train one model per MLModelData entry (shared by wrapper solvers)."""
    trained_models_cache = {}
    trained_constraints = []
    config_idx = 0
    for c_idx, constraint in enumerate(instance.constraints):
        row = []
        for m_idx, model_data in enumerate(constraint.models_data):
            md_id = id(model_data)
            if md_id not in trained_models_cache:
                if instance.constraint_model_configs and config_idx < len(instance.constraint_model_configs):
                    cfg = instance.constraint_model_configs[config_idx]
                    m_type = cfg.get("model_type", model_type)
                    m_params = cfg.get("model_params", model_params)
                else:
                    m_type = model_type
                    m_params = model_params
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


def _add_domain_constraints(opt, x, instance):
    d = instance.n_features
    for k, dc in enumerate(instance.domain_constraints):
        opt.addConstr(
            gp.quicksum(dc.coeffs[j] * x[j] for j in range(d)) <= dc.rhs,
            name=f"domain_{k}",
        )


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
    return 1


def _train_bootstrap_ensemble(X_train: np.ndarray,
                              y_train: np.ndarray,
                              model_type: str,
                              model_params: dict,
                              n_estimators: int,
                              seed: int = 42):
    """Train P models via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    models = []
    n = len(y_train)

    for p in range(n_estimators):
        idx = rng.choice(n, size=n, replace=True)
        params = (model_params or {}).copy()
        params["random_state"] = seed + p
        models.append(train_model(X_train[idx], y_train[idx], model_type, params))

    return models


def solve_wrapper(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  n_estimators: int = 20,
                  alpha: float = 0.1,
                  seed: int = 42,
                  rho: float = 0.0) -> SolutionResult:
    """Solve using the Maragno et al. wrapper approach."""
    start = time.time()

    models_embedded = 0
    trained_ensembles_cache = {}
    trained_constraints = []
    config_idx = 0
    for c_idx, constraint in enumerate(instance.constraints):
        constraint_trained_ensembles = []
        for m_idx, model_data in enumerate(constraint.models_data):
            md_id = id(model_data)
            if md_id not in trained_ensembles_cache:
                if instance.constraint_model_configs and config_idx < len(instance.constraint_model_configs):
                    cfg = instance.constraint_model_configs[config_idx]
                    m_type = cfg.get("model_type", model_type)
                    m_params = cfg.get("model_params", model_params)
                else:
                    m_type = model_type
                    m_params = model_params
                ensemble = _train_bootstrap_ensemble(
                    model_data.X_train, model_data.y_train,
                    m_type, m_params, n_estimators,
                    seed + c_idx * 100 + m_idx,
                )
                trained_ensembles_cache[md_id] = ensemble
            constraint_trained_ensembles.append((
                model_data.weight,
                model_data.obj_weight,
                trained_ensembles_cache[md_id],
            ))
            config_idx += 1
        trained_constraints.append(constraint_trained_ensembles)

    opt = gp.Model("wrapper")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 0.01
    opt.Params.MIPFocus = 1

    d = instance.n_features
    P = n_estimators

    x = [
        opt.addVar(lb=instance.variable_lb[j],
                   ub=instance.variable_ub[j],
                   name=f"x_{j}")
        for j in range(d)
    ]

    M_val = 1e4
    embedded_models_cache = {}
    obj_terms = []

    for c_idx, constraint_ensembles in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)

        if is_obj:
            for m_idx, (weight, obj_weight, ensemble) in enumerate(constraint_ensembles):
                preds = []
                for p in range(P):
                    ml_model = ensemble[p]
                    m_id = id(ml_model)
                    if m_id not in embedded_models_cache:
                        f_p = embed_model(
                            opt, ml_model, x,
                            instance.variable_lb, instance.variable_ub,
                            name_prefix=f"wrapper_c{c_idx}_m{m_idx}_p{p}", rho=rho,
                        )
                        embedded_models_cache[m_id] = f_p
                        models_embedded += 1
                    preds.append(weight * embedded_models_cache[m_id])
                avg_pred = (1.0 / P) * gp.quicksum(preds)
                obj_terms.append(obj_weight * avg_pred)
        else:
            z = opt.addVars(P, vtype=GRB.BINARY, name=f"z_wrapper_c{c_idx}")
            for p in range(P):
                f_pred_vars = []
                for m_idx, (weight, _, ensemble) in enumerate(constraint_ensembles):
                    ml_model = ensemble[p]
                    m_id = id(ml_model)
                    if m_id not in embedded_models_cache:
                        f_p = embed_model(
                            opt, ml_model, x,
                            instance.variable_lb, instance.variable_ub,
                            name_prefix=f"wrapper_c{c_idx}_m{m_idx}_p{p}", rho=rho,
                        )
                        embedded_models_cache[m_id] = f_p
                        models_embedded += 1
                    f_pred_vars.append(weight * embedded_models_cache[m_id])
                opt.addConstr(
                    gp.quicksum(f_pred_vars) <= constraint.rhs + M_val * (1 - z[p]),
                    name=f"wrapper_indicator_c{c_idx}_p{p}",
                )
            opt.addConstr(
                (1.0 / P) * gp.quicksum(z[p] for p in range(P)) >= 1 - alpha,
                name=f"wrapper_chance_c{c_idx}",
            )

    _add_domain_constraints(opt, x, instance)
    add_trust_region(opt, x, instance)

    base_cost = gp.quicksum(instance.cost_vector[j] * x[j] for j in range(d))
    opt.setObjective(base_cost + gp.quicksum(obj_terms), GRB.MINIMIZE)

    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        x_opt = np.array([x[j].X for j in range(d)])
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
        x_opt=np.zeros(d),
        obj_value=np.inf,
        status="infeasible",
        models_embedded=models_embedded,
        solve_time=elapsed,
        opt=opt,
        x=x,
    )


def solve_tree_violation_wrapper(instance: ProblemInstance,
                                 model_type: str = "rf",
                                 model_params: dict = None,
                                 alpha: float = 0.25,
                                 rho: float = 0.0) -> SolutionResult:
    """
    OptiCL chemo-style wrapper: embed each RF tree separately with a
    chance constraint over trees. GBM/CART/other models use a hard bound
    on the (aggregated) prediction, like nominal.
    """
    start = time.time()
    trained_constraints = _train_models_for_instance(instance, model_type, model_params)

    opt = gp.Model("tree_violation_wrapper")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 0.01
    opt.Params.MIPFocus = 1

    d = instance.n_features
    x = [
        opt.addVar(lb=instance.variable_lb[j], ub=instance.variable_ub[j], name=f"x_{j}")
        for j in range(d)
    ]

    models_embedded = 0
    embedded_cache = {}
    obj_terms = []

    for c_idx, constraint_models in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        f_pred_vars = []

        for m_idx, (weight, ml_model, obj_weight) in enumerate(constraint_models):
            prefix = f"tvw_c{c_idx}_m{m_idx}"
            rhs = constraint.rhs

            if obj_weight != 0.0:
                m_id = id(ml_model)
                if m_id not in embedded_cache:
                    embedded_cache[m_id] = embed_model(
                        opt, ml_model, x, instance.variable_lb, instance.variable_ub,
                        name_prefix=prefix, rho=rho,
                    )
                    models_embedded += 1
                f_pred = embedded_cache[m_id]
                obj_terms.append(obj_weight * f_pred)
            elif isinstance(ml_model, RandomForestRegressor):
                models_embedded += _add_rf_tree_violation(
                    opt, ml_model, x, instance, rhs, alpha, prefix, rho,
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
            opt.addConstr(gp.quicksum(f_pred_vars) <= rhs, name=f"tvw_constr_{c_idx}")

    _add_domain_constraints(opt, x, instance)
    add_trust_region(opt, x, instance)

    base_cost = gp.quicksum(instance.cost_vector[j] * x[j] for j in range(d))
    opt.setObjective(base_cost + gp.quicksum(obj_terms), GRB.MINIMIZE)
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
        x_opt=np.zeros(d),
        obj_value=np.inf,
        status="infeasible",
        models_embedded=models_embedded,
        solve_time=elapsed,
        opt=opt,
        x=x,
    )

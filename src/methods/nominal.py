"""
Nominal constraint learning: single model, no robustness.

min c'x  s.t.  f(x; theta*) <= b,  x in X
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Optional

from src.data.generate import ProblemInstance
from src.models.train import train_model, ModelType
from src.models.embed import embed_model


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


def solve_nominal(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  rho: float = 0.0) -> SolutionResult:
    """Solve the nominal constraint learning problem."""
    import time

    start = time.time()
    
    # Pre-train all nominal models and deduplicate by MLModelData object identity
    trained_models_cache = {}
    trained_constraints = []
    config_idx = 0
    for c_idx, constraint in enumerate(instance.constraints):
        constraint_trained_models = []
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
                ml_model = train_model(
                    model_data.X_train, model_data.y_train, m_type, m_params
                )
                trained_models_cache[md_id] = ml_model
            constraint_trained_models.append((model_data.weight, trained_models_cache[md_id], model_data.obj_weight))
            config_idx += 1
        trained_constraints.append(constraint_trained_models)

    # Build optimization model
    opt = gp.Model("nominal")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 0.01
    opt.Params.MIPFocus = 1

    d = instance.n_features
    x = [
        opt.addVar(lb=instance.variable_lb[j],
                   ub=instance.variable_ub[j],
                   name=f"x_{j}")
        for j in range(d)
    ]

    models_embedded = 0
    embedded_models_cache = {} # id(ml_model) -> f_pred Gurobi variable
    obj_terms = []

    # Embed ML models as constraints
    for c_idx, constraint_models in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        
        f_pred_vars = []
        for m_idx, (model_data_weight, ml_model, obj_weight) in enumerate(constraint_models):
            m_id = id(ml_model)
            if m_id not in embedded_models_cache:
                f_pred = embed_model(
                    opt, ml_model, x,
                    instance.variable_lb, instance.variable_ub,
                    name_prefix=f"nominal_c{c_idx}_m{m_idx}", rho=rho
                )
                embedded_models_cache[m_id] = f_pred
                models_embedded += 1
                
            f_pred_vars.append(model_data_weight * embedded_models_cache[m_id])
            if obj_weight != 0.0:
                obj_terms.append(obj_weight * embedded_models_cache[m_id])
            
        opt.addConstr(gp.quicksum(f_pred_vars) <= constraint.rhs, name=f"ml_constr_{c_idx}")

    for k, dc in enumerate(instance.domain_constraints):
        opt.addConstr(
            gp.quicksum(dc.coeffs[j] * x[j] for j in range(d)) <= dc.rhs,
            name=f"domain_{k}",
        )

    # Objective
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
            x=x
        )
    else:
        return SolutionResult(
            x_opt=np.zeros(d),
            obj_value=np.inf,
            status="infeasible",
            models_embedded=models_embedded,
            solve_time=elapsed,
            opt=opt,
            x=x
        )
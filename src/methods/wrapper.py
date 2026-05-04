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

from src.data.generate import ProblemInstance
from src.methods.nominal import SolutionResult
from src.models.train import train_model
from src.models.embed import embed_model


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
        # Bootstrap sample
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]

        # Vary random state for each model
        params = (model_params or {}).copy()
        params["random_state"] = seed + p

        model = train_model(X_boot, y_boot, model_type, params)
        models.append(model)

    return models


def solve_wrapper(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  n_estimators: int = 20,
                  alpha: float = 0.1,
                  seed: int = 42,
                  rho: float = 0.0) -> SolutionResult:
    """
    Solve using the Maragno et al. wrapper approach.
    """
    
    start = time.time()
    
    models_embedded = 0
    # Pre-train ensemble for each model in each constraint
    trained_ensembles_cache = {}
    trained_constraints = []
    for c_idx, constraint in enumerate(instance.constraints):
        constraint_trained_ensembles = []
        for m_idx, model_data in enumerate(constraint.models_data):
            md_id = id(model_data)
            if md_id not in trained_ensembles_cache:
                ensemble = _train_bootstrap_ensemble(
                    model_data.X_train, model_data.y_train, model_type, model_params, n_estimators, seed + c_idx*100 + m_idx
                )
                trained_ensembles_cache[md_id] = ensemble
            constraint_trained_ensembles.append((model_data.weight, trained_ensembles_cache[md_id]))
        trained_constraints.append(constraint_trained_ensembles)

    opt = gp.Model("wrapper")
    opt.Params.OutputFlag = 0

    d = instance.n_features
    P = n_estimators

    x = [
        opt.addVar(lb=instance.variable_lb[j],
                   ub=instance.variable_ub[j],
                   name=f"x_{j}")
        for j in range(d)
    ]

    opt.setObjective(
        gp.quicksum(
            instance.cost_vector[j] * x[j] for j in range(d)
        ),
        GRB.MINIMIZE,
    )

    M_val = 1e4  # Big-M
    embedded_models_cache = {}

    for c_idx, constraint_ensembles in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        
        # Binary variables for violation indicators for this constraint
        z = opt.addVars(P, vtype=GRB.BINARY, name=f"z_wrapper_c{c_idx}")
        
        for p in range(P):
            f_pred_vars = []
            
            for m_idx, (weight, ensemble) in enumerate(constraint_ensembles):
                ml_model = ensemble[p]
                m_id = id(ml_model)
                if m_id not in embedded_models_cache:
                    f_p = embed_model(
                        opt, ml_model, x,
                        instance.variable_lb, instance.variable_ub,
                        name_prefix=f"wrapper_c{c_idx}_m{m_idx}_p{p}", rho=rho
                    )
                    embedded_models_cache[m_id] = f_p
                    models_embedded += 1
                f_pred_vars.append(weight * embedded_models_cache[m_id])
            
            # Big-M constraint
            opt.addConstr(
                gp.quicksum(f_pred_vars) <= constraint.rhs + M_val * (1 - z[p]),
                name=f"wrapper_indicator_c{c_idx}_p{p}",
            )
            
        # At least (1 - alpha) fraction must be satisfied for THIS constraint
        opt.addConstr(
            (1.0 / P) * gp.quicksum(z[p] for p in range(P)) >= 1 - alpha,
            name=f"wrapper_chance_c{c_idx}",
        )


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
        )
    else:
        return SolutionResult(
            x_opt=np.zeros(d),
            obj_value=np.inf,
            status="infeasible",
            models_embedded=models_embedded,
            solve_time=elapsed,
        )
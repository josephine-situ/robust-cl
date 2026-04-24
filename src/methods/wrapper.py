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


def _train_bootstrap_ensemble(instance: ProblemInstance,
                              model_type: str,
                              model_params: dict,
                              n_estimators: int,
                              seed: int = 42):
    """Train P models via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    models = []
    n = len(instance.y_train)

    for p in range(n_estimators):
        # Bootstrap sample
        idx = rng.choice(n, size=n, replace=True)
        X_boot = instance.X_train[idx]
        y_boot = instance.y_train[idx]

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
    
    # Train ensemble
    ensemble = _train_bootstrap_ensemble(
        instance, model_type, model_params, n_estimators, seed
    )

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

    # Embed each estimator
    f_preds = []
    for p, ml_model in enumerate(ensemble):
        f_p = embed_model(
            opt, ml_model, x,
            instance.variable_lb, instance.variable_ub,
            name_prefix=f"wrapper_{p}", rho=rho
        )
        f_preds.append(f_p)

    # Binary variables for violation indicators
    z = opt.addVars(P, vtype=GRB.BINARY, name="z_wrapper")

    # Big-M constraints: if z_p = 1, constraint must hold
    M_val = 1e4  # Should be calibrated to problem
    b = instance.constraint_rhs
    for p in range(P):
        opt.addConstr(
            f_preds[p] <= b + M_val * (1 - z[p]),
            name=f"wrapper_indicator_{p}",
        )

    # At least (1 - alpha) fraction must be satisfied
    opt.addConstr(
        (1.0 / P) * gp.quicksum(z[p] for p in range(P)) >= 1 - alpha,
        name="wrapper_chance",
    )

    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        x_opt = np.array([x[j].X for j in range(d)])
        return SolutionResult(
            x_opt=x_opt,
            obj_value=opt.ObjVal,
            status="optimal",
            models_embedded=P,
            solve_time=elapsed,
        )
    else:
        return SolutionResult(
            x_opt=np.zeros(d),
            obj_value=np.inf,
            status="infeasible",
            models_embedded=P,
            solve_time=elapsed,
        )
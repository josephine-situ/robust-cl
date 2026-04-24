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
    iterations: Optional[int] = None


def solve_nominal(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  rho: float = 0.0) -> SolutionResult:
    """Solve the nominal constraint learning problem."""
    import time

    start = time.time()
    # Train model on noisy data
    ml_model = train_model(
        instance.X_train, instance.y_train, model_type, model_params
    )

    # Build optimization model
    opt = gp.Model("nominal")
    opt.Params.OutputFlag = 0

    d = instance.n_features
    x = [
        opt.addVar(lb=instance.variable_lb[j],
                   ub=instance.variable_ub[j],
                   name=f"x_{j}")
        for j in range(d)
    ]

    # Objective
    opt.setObjective(
        gp.quicksum(
            instance.cost_vector[j] * x[j] for j in range(d)
        ),
        GRB.MINIMIZE,
    )

    # Embed ML model as constraint
    f_pred = embed_model(
        opt, ml_model, x,
        instance.variable_lb, instance.variable_ub,
        name_prefix="nominal", rho=rho
    )
    opt.addConstr(f_pred <= instance.constraint_rhs, name="ml_constr")

    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        x_opt = np.array([x[j].X for j in range(d)])
        return SolutionResult(
            x_opt=x_opt,
            obj_value=opt.ObjVal,
            status="optimal",
            models_embedded=1,
            solve_time=elapsed,
        )
    else:
        return SolutionResult(
            x_opt=np.zeros(d),
            obj_value=np.inf,
            status="infeasible",
            models_embedded=1,
            solve_time=elapsed,
        )
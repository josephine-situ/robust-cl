"""
Random scenario approach.

Sample S perturbations delta_1,...,delta_S from D.
Train a model on each. Require all to satisfy the constraint.

min c'x  s.t.  f(x; theta*(delta_s)) <= b  for all s = 1,...,S
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from src.data.generate import ProblemInstance
from src.methods.nominal import SolutionResult
from src.models.train import retrain_on_perturbed
from src.models.embed import embed_model
from src.utils.perturbations import sample_multiple_perturbations


def solve_random_scenarios(instance: ProblemInstance,
                           model_type: str = "rf",
                           model_params: dict = None,
                           n_scenarios: int = 20,
                           delta_bar: float = 0.2,
                           gamma: float = 5.0,
                           seed: int = 42) -> SolutionResult:
    """
    Solve using random scenario sampling from D.
    """
    n = len(instance.y_train)

    # Sample perturbations
    perturbations = sample_multiple_perturbations(
        n, delta_bar, gamma, n_scenarios, seed
    )
    # Always include the nominal (zero perturbation)
    perturbations = [np.zeros(n)] + perturbations

    # Train a model for each perturbation
    scenario_models = []
    for delta in perturbations:
        model = retrain_on_perturbed(
            instance.X_train, instance.y_train, delta,
            model_type, model_params,
        )
        scenario_models.append(model)

    # Build and solve optimization
    start = time.time()
    opt = gp.Model("random_scenarios")
    opt.Params.OutputFlag = 0

    d = instance.n_features
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

    # Embed each scenario model and add constraint
    b = instance.constraint_rhs
    for s, ml_model in enumerate(scenario_models):
        f_s = embed_model(
            opt, ml_model, x,
            instance.variable_lb, instance.variable_ub,
            name_prefix=f"scenario_{s}",
        )
        opt.addConstr(f_s <= b, name=f"scenario_constr_{s}")

    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        x_opt = np.array([x[j].X for j in range(d)])
        return SolutionResult(
            x_opt=x_opt,
            obj_value=opt.ObjVal,
            status="optimal",
            models_embedded=len(scenario_models),
            solve_time=elapsed,
        )
    else:
        return SolutionResult(
            x_opt=np.zeros(d),
            obj_value=np.inf,
            status="infeasible",
            models_embedded=len(scenario_models),
            solve_time=elapsed,
        )
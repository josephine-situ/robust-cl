"""Convex-hull trust region for prescriptive optimization (OptiCL Section 2.3 / 5.2)."""

import gurobipy as gp

from src.data.generate import ProblemInstance


def add_trust_region(opt: gp.Model, x, instance: ProblemInstance, name_prefix: str = "tr") -> None:
    """
    Restrict decision variables to the convex hull of observed training treatments.

    x_dec = sum_i lambda_i * X_train[i],  lambda_i >= 0,  sum_i lambda_i = 1
    """
    if instance.trust_region_points is None:
        return

    X_tr = instance.trust_region_points
    n_train, _ = X_tr.shape
    if n_train == 0:
        return

    lambdas = opt.addVars(n_train, lb=0.0, ub=1.0, name=f"{name_prefix}_lambda")
    opt.addConstr(gp.quicksum(lambdas[i] for i in range(n_train)) == 1.0, name=f"{name_prefix}_sum")

    for local_j, global_j in enumerate(instance.decision_var_indices):
        opt.addConstr(
            x[global_j] == gp.quicksum(lambdas[i] * float(X_tr[i, local_j]) for i in range(n_train)),
            name=f"{name_prefix}_x_{global_j}",
        )

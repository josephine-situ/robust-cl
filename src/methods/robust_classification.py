"""
Robust classification / regression approach:
Train a single model robustly (protecting against label noise),
then embed it as a standard constraint.

This is the "modular" approach: robustify training, then embed.
Not decision-aware.

Approximation: train on multiple perturbed datasets, select the
model with best worst-case training loss.
"""

import numpy as np
from src.data.generate import ProblemInstance
from src.methods.nominal import SolutionResult
from src.models.train import train_model, retrain_on_perturbed
from src.models.embed import embed_model
from src.utils.perturbations import sample_multiple_perturbations
import gurobipy as gp
from gurobipy import GRB
import time


def solve_robust_classification(
        instance: ProblemInstance,
        model_type: str = "rf",
        model_params: dict = None,
        delta_bar: float = 0.2,
        gamma: float = 5.0,
        n_perturbations: int = 50,
        seed: int = 42) -> SolutionResult:
    """
    Robust classification approach.

    1. Sample many perturbations delta_1, ..., delta_M
    2. For each, train model on (X, y + delta_m)
    3. Select model with best worst-case loss across all
       perturbations (minimax over training loss)
    4. Embed that single model

    This is an approximation to true robust training.
    """
    n = len(instance.y_train)
    perturbations = sample_multiple_perturbations(
        n, delta_bar, gamma, n_perturbations, seed
    )
    # Add the zero perturbation
    perturbations = [np.zeros(n)] + perturbations

    # Train a model on each perturbation
    models = []
    for delta in perturbations:
        m = retrain_on_perturbed(
            instance.X_train, instance.y_train, delta,
            model_type, model_params,
        )
        models.append(m)

    # For each model, compute worst-case loss across all
    # perturbed datasets
    from sklearn.metrics import mean_squared_error

    best_model = None
    best_worst_loss = np.inf

    for model in models:
        worst_loss = 0.0
        for delta in perturbations:
            y_pert = instance.y_train + delta
            pred = model.predict(instance.X_train)
            loss = mean_squared_error(y_pert, pred)
            worst_loss = max(worst_loss, loss)

        if worst_loss < best_worst_loss:
            best_worst_loss = worst_loss
            best_model = model

    # Embed the selected robust model
    start = time.time()
    opt = gp.Model("robust_classification")
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

    f_pred = embed_model(
        opt, best_model, x,
        instance.variable_lb, instance.variable_ub,
        name_prefix="robust_cls",
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
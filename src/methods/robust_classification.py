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
        seed: int = 42,
        rho: float = 0.0) -> SolutionResult:
    """
    Robust classification approach.

    1. Sample many perturbations delta_1, ..., delta_M
    2. For each, train model on (X, y + delta_m)
    3. Select model with best worst-case loss across all
       perturbations (minimax over training loss)
    4. Embed that single model

    This is an approximation to true robust training.
    """
    from sklearn.metrics import mean_squared_error
    
    start = time.time()
    models_embedded = 0
    
    # Train robust models for constraints
    trained_models_cache = {}
    trained_constraints = []
    
    for c_idx, constraint in enumerate(instance.constraints):
        constraint_trained_models = []
        for m_idx, model_data in enumerate(constraint.models_data):
            md_id = id(model_data)
            
            if md_id not in trained_models_cache:
                n = len(model_data.y_train)
                perturbations = sample_multiple_perturbations(
                    n, delta_bar, gamma, n_perturbations, seed + c_idx*100 + m_idx
                )
                # Add the zero perturbation
                perturbations = [np.zeros(n)] + perturbations

                # Train a model on each perturbation
                models = []
                for delta in perturbations:
                    m = retrain_on_perturbed(
                        model_data.X_train, model_data.y_train, delta,
                        model_type, model_params,
                    )
                    models.append(m)

                best_model = None
                best_worst_loss = np.inf

                for model in models:
                    worst_loss = 0.0
                    for delta in perturbations:
                        y_pert = model_data.y_train + delta
                        pred = model.predict(model_data.X_train)
                        loss = mean_squared_error(y_pert, pred)
                        worst_loss = max(worst_loss, loss)

                    if worst_loss < best_worst_loss:
                        best_worst_loss = worst_loss
                        best_model = model
                
                trained_models_cache[md_id] = best_model
                
            constraint_trained_models.append((model_data.weight, trained_models_cache[md_id]))
            
        trained_constraints.append(constraint_trained_models)


    # Embed the selected robust models
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

    embedded_models_cache = {}

    for c_idx, constraint_models in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        
        f_pred_vars = []
        for m_idx, (weight, best_model) in enumerate(constraint_models):
            m_id = id(best_model)
            if m_id not in embedded_models_cache:
                f_pred = embed_model(
                    opt, best_model, x,
                    instance.variable_lb, instance.variable_ub,
                    name_prefix=f"robust_cls_c{c_idx}_m{m_idx}", rho=rho
                )
                embedded_models_cache[m_id] = f_pred
                models_embedded += 1
            f_pred_vars.append(weight * embedded_models_cache[m_id])
            
        opt.addConstr(gp.quicksum(f_pred_vars) <= constraint.rhs, name=f"ml_constr_{c_idx}")


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
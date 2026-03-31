"""
Column-and-Constraint Generation for robust constraint learning.

Iteratively:
1. Master: min c'x s.t. f(x; theta_s) <= b for s = 1,...,k
2. Separate: find worst-case delta, retrain, check violation
3. If violated, add new scenario to master
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.data.generate import ProblemInstance
from src.methods.nominal import SolutionResult
from src.models.train import train_model, retrain_on_perturbed, ModelType
from src.models.embed import embed_model
from src.utils.perturbations import greedy_adversarial_perturbation


@dataclass
class CCGHistory:
    """Track C&CG iteration history."""
    iterations: int = 0
    violations: List[float] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)
    x_solutions: List[np.ndarray] = field(default_factory=list)


def solve_ccg(instance: ProblemInstance,
              model_type: str = "rf",
              model_params: dict = None,
              delta_bar: float = 0.2,
              gamma: float = 5.0,
              max_iterations: int = 50,
              separation_strategy: str = "greedy",
              n_greedy_candidates: int = 20,
              seed: int = 42) -> tuple[SolutionResult, CCGHistory]:
    """
    Solve using Column-and-Constraint Generation.

    Returns
    -------
    result : SolutionResult
    history : CCGHistory tracking convergence
    """
    history = CCGHistory()
    n = len(instance.y_train)
    d = instance.n_features
    b = instance.constraint_rhs

    # Initialize: train nominal model
    nominal_model = train_model(
        instance.X_train, instance.y_train,
        model_type, model_params,
    )
    scenario_models: List[ModelType] = [nominal_model]
    scenario_deltas: List[np.ndarray] = [np.zeros(n)]

    total_start = time.time()

    for iteration in range(max_iterations):
        # === MASTER PROBLEM ===
        opt = gp.Model(f"ccg_master_iter{iteration}")
        opt.Params.OutputFlag = 0

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

        # Embed all scenario models
        for s, ml_model in enumerate(scenario_models):
            f_s = embed_model(
                opt, ml_model, x,
                instance.variable_lb, instance.variable_ub,
                name_prefix=f"ccg_s{s}",
            )
            opt.addConstr(f_s <= b, name=f"ccg_constr_{s}")

        opt.optimize()

        if opt.Status != GRB.OPTIMAL:
            # Master infeasible — too constrained
            return SolutionResult(
                x_opt=np.zeros(d),
                obj_value=np.inf,
                status="infeasible",
                models_embedded=len(scenario_models),
                solve_time=time.time() - total_start,
                iterations=iteration,
            ), history

        x_current = np.array([x[j].X for j in range(d)])
        obj_current = opt.ObjVal

        history.objectives.append(obj_current)
        history.x_solutions.append(x_current.copy())

        # === SEPARATION SUBPROBLEM ===
        if separation_strategy == "greedy":
            best_delta, best_value, best_model = \
                greedy_adversarial_perturbation(
                    instance.X_train,
                    instance.y_train,
                    x_current,
                    delta_bar, gamma,
                    model_type, model_params,
                    n_greedy_candidates,
                    seed=seed + iteration,
                )
        elif separation_strategy == "random":
            # Random search as baseline
            best_delta, best_value, best_model = \
                _random_separation(
                    instance, x_current,
                    delta_bar, gamma,
                    model_type, model_params,
                    n_candidates=n_greedy_candidates,
                    seed=seed + iteration,
                )
        else:
            raise ValueError(
                f"Unknown separation strategy: {separation_strategy}"
            )

        violation = best_value - b
        history.violations.append(violation)
        history.iterations = iteration + 1

        # Check termination
        if violation <= 1e-6:
            # No violation found — robust feasible
            elapsed = time.time() - total_start
            return SolutionResult(
                x_opt=x_current,
                obj_value=obj_current,
                status="optimal",
                models_embedded=len(scenario_models),
                solve_time=elapsed,
                iterations=iteration + 1,
            ), history

        # Add violating scenario
        scenario_models.append(best_model)
        scenario_deltas.append(best_delta)

    # Max iterations reached
    elapsed = time.time() - total_start
    return SolutionResult(
        x_opt=x_current,
        obj_value=obj_current,
        status="max_iterations",
        models_embedded=len(scenario_models),
        solve_time=elapsed,
        iterations=max_iterations,
    ), history


def _random_separation(instance, x_current, delta_bar, gamma,
                       model_type, model_params,
                       n_candidates=20, seed=42):
    """Random search separation oracle (baseline)."""
    from src.utils.perturbations import sample_multiple_perturbations

    n = len(instance.y_train)
    x_2d = np.atleast_2d(x_current)

    perturbations = sample_multiple_perturbations(
        n, delta_bar, gamma, n_candidates, seed
    )

    best_delta = np.zeros(n)
    base_model = retrain_on_perturbed(
        instance.X_train, instance.y_train, best_delta,
        model_type, model_params,
    )
    best_value = base_model.predict(x_2d)[0]
    best_model = base_model

    for delta in perturbations:
        model = retrain_on_perturbed(
            instance.X_train, instance.y_train, delta,
            model_type, model_params,
        )
        value = model.predict(x_2d)[0]
        if value > best_value:
            best_value = value
            best_delta = delta
            best_model = model

    return best_delta, best_value, best_model
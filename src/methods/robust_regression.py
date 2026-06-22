"""
Robust regression approach:
Train a single model robustly via bootstrap minimax (OOB error),
then embed it as a standard constraint.
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    SolutionResult,
    resolve_constraint_config,
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    embed_constraints,
)
from src.methods.wrapper import _get_shared_bootstrap_indices
from src.models.train import train_bootstrap_models, oob_worst_case_error


def solve_robust_regression(
        instance: ProblemInstance,
        model_type: str = "rf",
        model_params: dict = None,
        n_bootstrap: int = 25,
        seed: int = 42,
        rho: float = 0.0,
        embedding_mode: str = "hard",
        rf_alpha: float = 0.25,
        bootstrap_cache=None) -> SolutionResult:
    """
    Bootstrap minimax robust training:
    1. Train P models on shared bootstrap resamples
    2. Select model with lowest worst-case OOB error
    3. Embed that single model
    """
    start = time.time()
    models_embedded = 0

    if bootstrap_cache is None:
        bootstrap_cache = _get_shared_bootstrap_indices(
            instance, model_type, model_params, n_bootstrap, seed
        )

    trained_models_cache = {}
    trained_constraints = []
    config_idx = 0

    for constraint in instance.constraints:
        constraint_trained_models = []
        for model_data in constraint.models_data:
            md_id = id(model_data)
            if md_id not in trained_models_cache:
                m_type, m_params = resolve_constraint_config(
                    instance, config_idx, model_type, model_params
                )
                print(
                    f"    [robust_reg] Bootstrap minimax for {constraint.name} "
                    f"({n_bootstrap} models, type={m_type})...",
                    flush=True,
                )
                t0 = time.time()
                bootstrap_indices = bootstrap_cache[md_id]
                ensemble = train_bootstrap_models(
                    model_data.X_train, model_data.y_train,
                    m_type, m_params, bootstrap_indices,
                    seed + config_idx * 100,
                )
                best_idx, best_oob, _ = oob_worst_case_error(
                    ensemble, bootstrap_indices,
                    model_data.X_train, model_data.y_train,
                )
                trained_models_cache[md_id] = ensemble[best_idx]
                print(
                    f"    [robust_reg] {constraint.name} selected model {best_idx} "
                    f"(worst OOB err={best_oob:.4f}) in {time.time() - t0:.1f}s",
                    flush=True,
                )
            constraint_trained_models.append((
                model_data.weight,
                trained_models_cache[md_id],
                model_data.obj_weight,
            ))
            config_idx += 1
        trained_constraints.append(constraint_trained_models)

    opt = gp.Model("robust_regression")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 0.01
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    models_embedded, _, obj_terms = embed_constraints(
        opt, x, instance, trained_constraints,
        rho=rho, embedding_mode=embedding_mode, rf_alpha=rf_alpha,
        name_prefix="robust_reg",
    )
    add_problem_constraints(opt, x, instance)
    build_and_set_objective(opt, x, instance, obj_terms)

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
        x_opt=np.zeros(instance.n_features),
        obj_value=np.inf,
        status="infeasible",
        models_embedded=models_embedded,
        solve_time=elapsed,
        opt=opt,
        x=x,
    )

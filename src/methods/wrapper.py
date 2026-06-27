"""
Maragno et al. (2025) model wrapper approach.

Train P estimators (bootstrap) on the same data. Require at least
(1 - alpha) * P satisfy the constraint.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    SolutionResult,
    resolve_constraint_config,
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    build_and_set_robust_objective,
    embed_constraints,
)
from src.models.train import generate_bootstrap_samples, train_bootstrap_models
from src.models.embed import embed_model


def _get_shared_bootstrap_indices(instance, model_type, model_params, n_bootstrap, seed):
    """Fixed P bootstrap index vectors per MLModelData (shared by wrapper and robust_reg)."""
    cache = {}
    config_idx = 0
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            md_id = id(model_data)
            if md_id not in cache:
                n = len(model_data.y_train)
                cache[md_id] = generate_bootstrap_samples(
                    n, n_bootstrap, seed + config_idx * 100
                )
            config_idx += 1
    return cache


def _coherent_bootstrap_indices(instance, n_bootstrap, seed):
    """One shared set of P bootstrap index vectors assigned to EVERY MLModelData.

    Because all gastric outcomes share the same patients (rows of X_train),
    resampling rows once and applying it to every outcome makes replicate ``p`` a
    single coherent trial relabeling across all constraints and the objective -
    the bootstrap analogue of CP's shared scenario. Assumes every MLModelData has
    the same number of training rows.
    """
    n = len(instance.constraints[0].models_data[0].y_train)
    shared = generate_bootstrap_samples(n, n_bootstrap, seed)
    cache = {}
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            cache[id(model_data)] = shared
    return cache


def train_bootstrap_ensembles_for_instance(instance,
                                           model_type,
                                           model_params,
                                           n_bootstrap,
                                           seed,
                                           bootstrap_cache=None,
                                           ensembles_cache=None):
    """Train P bootstrap models per MLModelData; optionally reuse index cache.

    If ``ensembles_cache`` (md_id -> trained models) is provided, it is reused
    directly without retraining - handy when calibration evaluates several
    robustness settings that do not change the underlying models.
    """
    if bootstrap_cache is None:
        bootstrap_cache = _get_shared_bootstrap_indices(
            instance, model_type, model_params, n_bootstrap, seed
        )
    if ensembles_cache is not None:
        return ensembles_cache, bootstrap_cache
    trained_ensembles_cache = {}
    config_idx = 0
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            md_id = id(model_data)
            if md_id not in trained_ensembles_cache:
                m_type, m_params = resolve_constraint_config(
                    instance, config_idx, model_type, model_params
                )
                print(
                    f"    [wrapper] Training {n_bootstrap} bootstrap {m_type} "
                    f"models for {constraint.name}...",
                    flush=True,
                )
                t0 = time.time()
                trained_ensembles_cache[md_id] = train_bootstrap_models(
                    model_data.X_train, model_data.y_train,
                    m_type, m_params,
                    bootstrap_cache[md_id],
                    seed + config_idx * 100,
                )
                print(
                    f"    [wrapper] {constraint.name} bootstrap done in "
                    f"{time.time() - t0:.1f}s",
                    flush=True,
                )
            config_idx += 1
    return trained_ensembles_cache, bootstrap_cache


def solve_wrapper(instance: ProblemInstance,
                  model_type: str = "rf",
                  model_params: dict = None,
                  n_estimators: int = 20,
                  alpha: float = 0.1,
                  seed: int = 42,
                  rho: float = 0.0,
                  bootstrap_cache=None,
                  ensembles_cache=None) -> SolutionResult:
    """Solve using the Maragno et al. wrapper approach, made coherent across
    constraints: a single shared bootstrap replicate ``p`` must satisfy ALL
    toxicity constraints jointly (one indicator ``z[p]`` shared across
    constraints), and at least ``(1 - alpha)`` of replicates must do so. The
    learned objective (OS) is robustified with a worst-case epigraph over the
    same replicates."""
    start = time.time()
    n_bootstrap = n_estimators

    trained_ensembles_cache, _ = train_bootstrap_ensembles_for_instance(
        instance, model_type, model_params, n_bootstrap, seed,
        bootstrap_cache, ensembles_cache,
    )

    trained_constraints = []
    config_idx = 0
    for c_idx, constraint in enumerate(instance.constraints):
        row = []
        for model_data in constraint.models_data:
            md_id = id(model_data)
            row.append((
                model_data.weight,
                model_data.obj_weight,
                trained_ensembles_cache[md_id],
            ))
            config_idx += 1
        trained_constraints.append(row)

    opt = gp.Model("wrapper")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 1e-4
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    P = n_bootstrap
    M_val = 1e4
    embedded_models_cache = {}
    obj_scenarios = [[] for _ in range(P)]   # per replicate: objective terms
    models_embedded = 0

    def _embed(ml_model, prefix):
        nonlocal models_embedded
        m_id = id(ml_model)
        if m_id not in embedded_models_cache:
            embedded_models_cache[m_id] = embed_model(
                opt, ml_model, x,
                instance.variable_lb, instance.variable_ub,
                name_prefix=prefix, rho=rho,
            )
            models_embedded += 1
        return embedded_models_cache[m_id]

    # Coherent indicator: z[p] = 1 iff replicate p satisfies *every* toxicity
    # constraint at x. Shared across constraints so the chance constraint is over
    # one joint relabeling, not independent per-constraint relabelings.
    has_constraints = any(
        not any(md.obj_weight != 0 for md in instance.constraints[c_idx].models_data)
        for c_idx in range(len(trained_constraints))
    )
    z = opt.addVars(P, vtype=GRB.BINARY, name="z_wrapper_joint") if has_constraints else None

    for c_idx, constraint_ensembles in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)

        if is_obj:
            for p in range(P):
                for m_idx, (weight, obj_weight, ensemble) in enumerate(constraint_ensembles):
                    f_p = _embed(ensemble[p], f"wrapper_c{c_idx}_m{m_idx}_p{p}")
                    obj_scenarios[p].append(obj_weight * weight * f_p)
        else:
            for p in range(P):
                f_pred_vars = []
                for m_idx, (weight, _, ensemble) in enumerate(constraint_ensembles):
                    f_p = _embed(ensemble[p], f"wrapper_c{c_idx}_m{m_idx}_p{p}")
                    f_pred_vars.append(weight * f_p)
                opt.addConstr(
                    gp.quicksum(f_pred_vars) <= constraint.rhs + M_val * (1 - z[p]),
                    name=f"wrapper_indicator_c{c_idx}_p{p}",
                )

    if z is not None:
        opt.addConstr(
            (1.0 / P) * gp.quicksum(z[p] for p in range(P)) >= 1 - alpha,
            name="wrapper_chance_joint",
        )

    add_problem_constraints(opt, x, instance)
    build_and_set_robust_objective(opt, x, instance, obj_scenarios)
    print(
        f"    [wrapper] MIP built ({models_embedded} models embedded); solving...",
        flush=True,
    )
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


def solve_tree_violation_wrapper(instance: ProblemInstance,
                                 model_type: str = "rf",
                                 model_params: dict = None,
                                 alpha: float = 0.25,
                                 rho: float = 0.0) -> SolutionResult:
    """OptiCL chemo-style wrapper with per-tree RF chance constraints."""
    import time
    from src.methods.nominal import train_constraint_models

    start = time.time()
    print(
        f"    [tree_violation] Training models (RF per-tree chance, alpha={alpha})...",
        flush=True,
    )
    trained_constraints = train_constraint_models(instance, model_type, model_params)

    opt = gp.Model("tree_violation_wrapper")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = 1e-4
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    models_embedded, _, obj_terms = embed_constraints(
        opt, x, instance, trained_constraints,
        rho=rho, embedding_mode="tree_violation", rf_alpha=alpha,
        name_prefix="tvw",
    )
    add_problem_constraints(opt, x, instance)
    build_and_set_objective(opt, x, instance, obj_terms)
    print(
        f"    [tree_violation] MIP built ({models_embedded} tree/model embeds); solving...",
        flush=True,
    )
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

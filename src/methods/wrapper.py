"""
Maragno et al. (2025) model wrapper approach.

Train P estimators (bootstrap) on the same data. Require at least
(1 - alpha) * P satisfy the constraint.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from typing import Optional

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    DEFAULT_MIP_GAP,
    SolutionResult,
    resolve_constraint_config,
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    build_and_set_robust_objective,
    embed_constraints,
    model_X_ref,
)
from src.models.train import (
    generate_bootstrap_samples, train_bootstrap_models, train_model,
)
from src.models.embed import embed_model


def _get_shared_bootstrap_indices(instance, model_type, model_params, n_bootstrap, seed,
                                  bootstrap_frac=0.5):
    """Fixed P bootstrap index vectors per MLModelData.

    Consumed only by the legacy ``scenario_source: "bootstrap"`` wrapper path -
    ``robust_reg`` takes ``label_eps`` + the shared set D and never bootstraps.
    ``bootstrap_frac`` is the per-replicate row proportion (Maragno 0.5).
    """
    cache = {}
    config_idx = 0
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            md_id = id(model_data)
            if md_id not in cache:
                n = len(model_data.y_train)
                cache[md_id] = generate_bootstrap_samples(
                    n, n_bootstrap, seed + config_idx * 100, bootstrap_frac
                )
            config_idx += 1
    return cache


def _coherent_bootstrap_indices(instance, n_bootstrap, seed, bootstrap_frac=0.5):
    """One shared set of P bootstrap index vectors assigned to EVERY MLModelData.

    Because all gastric outcomes share the same patients (rows of X_train),
    resampling rows once and applying it to every outcome makes replicate ``p`` a
    single coherent trial relabeling across all constraints and the objective -
    the bootstrap analogue of CP's shared scenario. Assumes every MLModelData has
    the same number of training rows. ``bootstrap_frac`` is the per-replicate row
    proportion (Maragno 0.5).
    """
    n = len(instance.constraints[0].models_data[0].y_train)
    shared = generate_bootstrap_samples(n, n_bootstrap, seed, bootstrap_frac)
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
                                           ensembles_cache=None,
                                           bootstrap_frac=0.5):
    """Train P bootstrap models per MLModelData; optionally reuse index cache.

    If ``ensembles_cache`` (md_id -> trained models) is provided, it is reused
    directly without retraining - handy when calibration evaluates several
    robustness settings that do not change the underlying models.
    ``bootstrap_frac`` applies only when ``bootstrap_cache`` is built here.
    """
    if bootstrap_cache is None:
        bootstrap_cache = _get_shared_bootstrap_indices(
            instance, model_type, model_params, n_bootstrap, seed, bootstrap_frac
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
                  ensembles_cache=None,
                  bootstrap_frac: float = 0.5,
                  scenario_source: str = "noise",
                  uncertainty_set=None,
                  bank=None,
                  robustify_objective: bool = False,
                  coherent: Optional[bool] = None,
                  mip_gap: float = DEFAULT_MIP_GAP) -> SolutionResult:
    """Maragno et al.'s wrapper: at least ``(1 - alpha)`` of P plausible
    relabelings must satisfy the constraints at ``x``.

    ``scenario_source``:
    - ``"noise"`` (default): the P models come from the **same** uncertainty set D
      and the **same** seeded draw sequence CP separates over
      (:class:`~src.methods.uncertainty.ScenarioBank`). Because draw ``b`` is a
      pure function of ``(seed, b)``, the wrapper's P models are a genuine
      *prefix* of CP's bank -- so ``alpha=0`` here and ``tau->0`` in CP face
      identical adversaries and must agree.
    - ``"bootstrap"``: the legacy bootstrap replicates. Each draws
      ``bootstrap_frac`` of the rows with replacement (default 0.5, matching
      Maragno et al. 2025 Sec. 4.4.1); ignored when ``bootstrap_cache`` is
      supplied, since the caller already fixed the indices.

    P is capped by MIP size, because unlike CP -- which embeds one extra scenario
    per iteration and evicts -- the wrapper embeds **all** P models at once.

    ``coherent`` (default: D's setting): coherent requires one shared relabeling
    to satisfy every constraint jointly via a single indicator ``z[p]``.
    Incoherent gives each constraint its own ``z[c, p]``, so different
    constraints may be satisfied by different relabelings.

    ``robustify_objective``: when ``False`` (default) the objective is a single
    nominal model, matching CP's default. When ``True`` the objective is a
    worst-case epigraph over the same P replicates -- the legacy behavior, which
    carried conservatism CP did not, and which costs P extra OS embeddings.
    """
    start = time.time()
    n_bootstrap = n_estimators
    if coherent is None:
        coherent = bool(getattr(uncertainty_set, "coherent", True))

    if scenario_source == "noise":
        if bank is None:
            import dataclasses
            from src.methods.uncertainty import ScenarioBank, UncertaintySet
            uset = uncertainty_set if uncertainty_set is not None else UncertaintySet()
            # One coherence flag: it must drive the DRAWS as well as the indicator
            # structure, or an "incoherent" wrapper would still face coherent
            # relabelings and only the z's would differ.
            uset = dataclasses.replace(uset, coherent=coherent)
            model_config_map = {
                id(md): resolve_constraint_config(instance, i, model_type, model_params)
                for i, md in enumerate(
                    md for c in instance.constraints for md in c.models_data)
            }
            bank = ScenarioBank(instance, model_config_map, uset,
                                n_scenarios=n_bootstrap, seed=seed)
        elif len(bank) < n_bootstrap:
            bank.extend(n_bootstrap)
        # A prefix of CP's bank: nested, not merely identically distributed.
        trained_ensembles_cache = bank.as_ensembles_cache(n_bootstrap)
    else:
        trained_ensembles_cache, _ = train_bootstrap_ensembles_for_instance(
            instance, model_type, model_params, n_bootstrap, seed,
            bootstrap_cache, ensembles_cache, bootstrap_frac,
        )

    trained_constraints = []
    config_idx = 0
    # Nominal (unperturbed) models for the OBJECTIVE outcomes only. Every entry of
    # trained_ensembles_cache is a retrain_on_perturbed output -- draw 0 included,
    # since _draw(0) produces a nonzero delta -- so "embed ensemble[0]" would put an
    # arbitrary perturbed model in the objective, not the nominal one. Measured on
    # gastric: draw 0's OS model differs from nominal by up to 1.08 months. CP under
    # robustify_objective=False maximizes the true nominal OS, so using draw 0 here
    # made the two methods optimize different objectives at the same D.
    nominal_obj_models = {}
    _cfg_idx = 0
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            if model_data.obj_weight != 0.0 and id(model_data) not in nominal_obj_models:
                m_type, m_params = resolve_constraint_config(
                    instance, _cfg_idx, model_type, model_params)
                nominal_obj_models[id(model_data)] = train_model(
                    model_data.X_train, model_data.y_train, m_type, m_params)
            _cfg_idx += 1
    for c_idx, constraint in enumerate(instance.constraints):
        row = []
        for model_data in constraint.models_data:
            md_id = id(model_data)
            row.append((
                model_data.weight,
                model_data.obj_weight,
                trained_ensembles_cache[md_id],
                nominal_obj_models.get(md_id),
            ))
            config_idx += 1
        trained_constraints.append(row)

    opt = gp.Model("wrapper")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = mip_gap
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    P = n_bootstrap
    M_val = 1e4
    embedded_models_cache = {}
    obj_scenarios = [[] for _ in range(P)]   # per replicate: objective terms
    models_embedded = 0

    def _embed(ml_model, prefix, c_idx=None, m_idx=None):
        nonlocal models_embedded
        m_id = id(ml_model)
        if m_id not in embedded_models_cache:
            embedded_models_cache[m_id] = embed_model(
                opt, ml_model, x,
                instance.variable_lb, instance.variable_ub,
                name_prefix=prefix, rho=rho,
                X_ref=(None if c_idx is None
                       else model_X_ref(instance, c_idx, m_idx)),
            )
            models_embedded += 1
        return embedded_models_cache[m_id]

    # Coherent: z[p] = 1 iff replicate p satisfies *every* toxicity constraint at
    # x, so the chance constraint is over one joint relabeling. Incoherent: a
    # separate z[c, p] per constraint, so each constraint may be satisfied by a
    # different relabeling -- the same coherence flag the bank and robust_reg take.
    constraint_idxs = [
        c_idx for c_idx in range(len(trained_constraints))
        if not any(md.obj_weight != 0 for md in instance.constraints[c_idx].models_data)
    ]
    if not constraint_idxs:
        z = None
    elif coherent:
        z = opt.addVars(P, vtype=GRB.BINARY, name="z_wrapper_joint")
    else:
        z = opt.addVars(constraint_idxs, P, vtype=GRB.BINARY, name="z_wrapper")

    def _z(c_idx, p):
        return z[p] if coherent else z[c_idx, p]

    for c_idx, constraint_ensembles in enumerate(trained_constraints):
        constraint = instance.constraints[c_idx]
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)

        if is_obj:
            # Nominal objective (default): embed ONE model, not P, and that model is
            # the NOMINAL fit -- not bank draw 0, which is itself perturbed. CP's
            # default is a nominal objective too, so robustifying here would hand
            # the wrapper extra conservatism CP does not carry -- and cost P OS
            # embeddings.
            if robustify_objective:
                for p in range(P):
                    for m_idx, (weight, obj_weight, ensemble, _nom) in enumerate(
                            constraint_ensembles):
                        f_p = _embed(ensemble[p], f"wrapper_c{c_idx}_m{m_idx}_p{p}",
                                     c_idx, m_idx)
                        obj_scenarios[p].append(obj_weight * weight * f_p)
            else:
                for m_idx, (weight, obj_weight, ensemble, nom) in enumerate(
                        constraint_ensembles):
                    f_nom = _embed(nom if nom is not None else ensemble[0],
                                   f"wrapper_c{c_idx}_m{m_idx}_nominal", c_idx, m_idx)
                    obj_scenarios[0].append(obj_weight * weight * f_nom)
        else:
            for p in range(P):
                f_pred_vars = []
                for m_idx, (weight, _, ensemble, _nom) in enumerate(constraint_ensembles):
                    f_p = _embed(ensemble[p], f"wrapper_c{c_idx}_m{m_idx}_p{p}",
                                 c_idx, m_idx)
                    f_pred_vars.append(weight * f_p)
                opt.addConstr(
                    gp.quicksum(f_pred_vars) <= constraint.rhs + M_val * (1 - _z(c_idx, p)),
                    name=f"wrapper_indicator_c{c_idx}_p{p}",
                )

    if z is not None:
        if coherent:
            opt.addConstr(
                (1.0 / P) * gp.quicksum(z[p] for p in range(P)) >= 1 - alpha,
                name="wrapper_chance_joint",
            )
        else:
            for c_idx in constraint_idxs:
                opt.addConstr(
                    (1.0 / P) * gp.quicksum(z[c_idx, p] for p in range(P)) >= 1 - alpha,
                    name=f"wrapper_chance_c{c_idx}",
                )

    add_problem_constraints(opt, x, instance)
    if robustify_objective:
        build_and_set_robust_objective(opt, x, instance, obj_scenarios)
    else:
        build_and_set_objective(opt, x, instance, obj_scenarios[0])
    opt.update()
    # MIP size is the whole point of the CP-vs-wrapper comparison: the wrapper
    # embeds all P models, CP embeds one cut per iteration over a bank that can be
    # an order of magnitude larger. Log it so the claim is measured, not asserted.
    print(
        f"    [wrapper] MIP built (P={P}, source={scenario_source}, "
        f"coherent={coherent}, {models_embedded} models embedded, "
        f"{opt.NumVars} vars / {opt.NumConstrs} constrs); solving...",
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
                                 rho: float = 0.0,
                                 mip_gap: float = DEFAULT_MIP_GAP) -> SolutionResult:
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
    opt.Params.MIPGap = mip_gap
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

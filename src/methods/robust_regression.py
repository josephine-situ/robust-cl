"""
Robust regression approach (coherent over constraints):
Pick one *shared* bootstrap replicate - a single coherent trial relabeling that
stresses every outcome jointly - by a normalized worst-case OOB score across all
constraint models AND the objective, then embed that replicate's models. A
``conservativeness`` knob in [0, 1] selects how far up the ranking to go (0 =
least, 1 = most adversarial replicate).
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    SolutionResult,
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    embed_constraints,
)
from src.methods.wrapper import (
    _coherent_bootstrap_indices,
    train_bootstrap_ensembles_for_instance,
)
from src.models.train import oob_worst_case_error


def _select_coherent_replicate(instance, ensembles, bootstrap_cache, conservativeness):
    """Rank the P shared replicates by a normalized joint worst-case OOB score and
    return the replicate index at the requested conservativeness quantile.

    Each model's per-replicate worst OOB error is normalized by that model's own
    mean (across replicates) so large-scale outcomes (OS in months) do not
    dominate the small-scale percentile toxicity models, then summed across all
    models - constraints and the objective alike - into one score per replicate.
    """
    P = None
    score = None
    for constraint in instance.constraints:
        for md in constraint.models_data:
            ens = ensembles[id(md)]
            idxs = bootstrap_cache[id(md)]
            _, _, per_worst = oob_worst_case_error(ens, idxs, md.X_train, md.y_train)
            per_worst = np.asarray(per_worst, dtype=float)
            if P is None:
                P = len(per_worst)
                score = np.zeros(P)
            finite = per_worst[np.isfinite(per_worst)]
            scale = float(finite.mean()) if finite.size and finite.mean() > 0 else 1.0
            score += np.where(np.isfinite(per_worst), per_worst / scale, 0.0)

    order = np.argsort(-score)            # most adversarial (highest score) first
    rank = int(round((1.0 - conservativeness) * (P - 1)))
    rank = max(0, min(P - 1, rank))
    return int(order[rank])


def solve_robust_regression(
        instance: ProblemInstance,
        model_type: str = "rf",
        model_params: dict = None,
        n_bootstrap: int = 25,
        seed: int = 42,
        rho: float = 0.0,
        embedding_mode: str = "hard",
        rf_alpha: float = 0.25,
        conservativeness: float = 1.0,
        bootstrap_cache=None,
        ensembles_cache=None) -> SolutionResult:
    """
    Coherent bootstrap robust training:
    1. Train P models per outcome on a *shared* set of bootstrap resamples.
    2. Select one shared replicate by normalized joint worst-case OOB score
       (across all constraints and the objective), at the conservativeness quantile.
    3. Embed that replicate's models (constraints hard, objective via the
       chosen-replicate epigraph).
    """
    start = time.time()

    if bootstrap_cache is None:
        bootstrap_cache = _coherent_bootstrap_indices(instance, n_bootstrap, seed)

    action = "Reusing cached" if ensembles_cache is not None else "Training"
    print(
        f"    [robust_reg] {action} {n_bootstrap} shared-replicate bootstrap "
        f"ensembles (conservativeness={conservativeness:.3f})...",
        flush=True,
    )
    ensembles, _ = train_bootstrap_ensembles_for_instance(
        instance, model_type, model_params, n_bootstrap, seed,
        bootstrap_cache, ensembles_cache,
    )
    p_star = _select_coherent_replicate(
        instance, ensembles, bootstrap_cache, conservativeness
    )
    print(f"    [robust_reg] Selected coherent replicate {p_star}", flush=True)

    trained_constraints = []
    for constraint in instance.constraints:
        row = []
        for model_data in constraint.models_data:
            row.append((
                model_data.weight,
                ensembles[id(model_data)][p_star],
                model_data.obj_weight,
            ))
        trained_constraints.append(row)

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

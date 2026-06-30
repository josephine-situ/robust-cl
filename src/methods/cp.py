"""
Cutting Planes method for robust constraint learning.

One driver (:func:`solve_cp`) handles every scenario and **auto-selects** the
separation strategy from the problem shape:

- **basic** -- a single global LP with a single learned constraint (synthetic):
  plain worst-case localized-bootstrap separation at ``x*``; no chance budget.
- **coherent** -- multiple constraints, multiple optimal solutions ``x*``, or a
  learned objective (e.g. gastric): one *shared* bootstrap relabeling drives
  every constraint (and the objective) jointly and the single worst scenario --
  ranked by **normalized average distance** (mean relative exceedance over all
  ``(x*, outcome)`` cells, range 0–1) -- is cut. We stop when that distance
  falls within ``cp_dist_tol`` or no scenario can be added without pushing more
  than ``cp_alpha`` of the ``x*`` infeasible. With multiple ``x*`` the coverage
  cap ``cp_alpha`` bounds the fraction allowed to go infeasible (feasibility
  only); with a single ``x*`` there is no cap.

The objective uses an **epigraph** reformulation ``min c'x + t``, ``t >= sum of
learned objective terms``, so a learned objective is robustified by the same
worst-case cuts (each raises ``t``). Only the coherent strategy robustifies it.

Both reuse the same scaffolding: train nominal -> build master -> solve for the
optimal solution(s) ``x*`` -> separate -> add cuts -> terminate.
"""
import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import List, Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    SolutionResult,
    resolve_constraint_config,
    add_domain_constraints,
    build_decision_vars,
)
from src.models.train import (
    train_model,
    retrain_on_bootstrap,
    localized_bootstrap_indices,
    resolve_neighbor_pool_size,
)
from src.models.embed import embed_model
from src.utils.trust_region import add_trust_region


@dataclass
class CPHistory:
    """Track CP iteration history."""
    iterations: int = 0
    violations: List[float] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)
    x_solutions: List[np.ndarray] = field(default_factory=list)


@dataclass
class CPMultiAnchorResult:
    """One finalized CP master per training anchor; prescribe via nearest anchor."""
    anchor_rows: np.ndarray
    anchor_results: List[SolutionResult]
    status: str
    solve_time: float
    models_embedded: int
    nearest_distance_feature_indices: Optional[list] = None
    iterations: Optional[int] = None


class IncrementalMaster:
    """Keeps the Gurobi model in memory to add constraints incrementally."""

    def __init__(self, instance: ProblemInstance, obj_terms: list, rho: float = 0.0):
        self.instance = instance
        self.d = instance.n_features
        self.rho = rho
        self.opt = gp.Model("cp_incremental_master")
        self.opt.Params.OutputFlag = 0
        self.opt.Params.MIPGap = 0.01
        self.opt.Params.MIPFocus = 1
        self.opt.Params.Threads = 0

        self.x = build_decision_vars(self.opt, instance)

        self.base_cost = gp.quicksum(
            instance.cost_vector[j] * self.x[j] for j in range(self.d)
        )
        # Epigraph variable for the learned objective (None until set up); the
        # objective is ``min base_cost + t_obj`` with ``t_obj >= sum(obj terms)``.
        self.t_obj = None
        self.obj_expr = self.base_cost
        self.opt.setObjective(self.obj_expr, GRB.MINIMIZE)

        add_domain_constraints(self.opt, self.x, instance)
        add_trust_region(self.opt, self.x, instance)

        self.n_models = 0
        self.scenario_constrs = []
        self.scenario_vars_map = {}
        self.scenario_constrs_map = {}
        self.scenario_model_ids_map = {}   # s -> cache keys this scenario created
        self.embedded_models_cache = {}

        if obj_terms:
            self.set_epigraph_objective(obj_terms)

    def set_epigraph_objective(self, nominal_obj_terms: list):
        """Reformulate the objective as ``min base_cost + t_obj`` with the nominal
        epigraph cut ``t_obj >= sum(nominal_obj_terms)``.

        Worst-case objective cuts ``sum(obj_weight_i f_i^s(x)) <= t_obj`` (added
        as scenarios with ``rhs=self.t_obj``) then robustify the objective: each
        raises the epigraph floor, so the optimum reflects the worst plausible
        relabeling of the objective outcome.
        """
        self.t_obj = self.opt.addVar(lb=-GRB.INFINITY, name="t_obj")
        self.opt.addConstr(
            self.t_obj >= gp.quicksum(nominal_obj_terms), name="epigraph_obj_nominal"
        )
        self.obj_expr = self.base_cost + self.t_obj
        self.opt.setObjective(self.obj_expr, GRB.MINIMIZE)
        self.opt.update()

    def remove_scenario(self, s: int):
        for c in self.scenario_constrs_map.get(s, []):
            self.opt.remove(c)
        for v in self.scenario_vars_map.get(s, []):
            self.opt.remove(v)
        # Evict the embedded-model vars this scenario created: their Gurobi vars
        # are now gone, so a future id() collision must re-embed rather than reuse
        # a stale (removed) variable.
        for m_id in self.scenario_model_ids_map.get(s, []):
            self.embedded_models_cache.pop(m_id, None)
        self.scenario_constrs_map[s] = []
        self.scenario_vars_map[s] = []
        self.scenario_model_ids_map[s] = []
        if s < len(self.scenario_constrs):
            self.scenario_constrs[s] = None

    def add_scenario(self, c_idx: int, constraint_models: List[tuple], rhs: float,
                     rho: float = 0.0):
        prefix = f"cp_c{c_idx}_s{self.n_models}"
        self.opt.update()
        old_constrs = set(self.opt.getConstrs())
        old_vars = set(self.opt.getVars())

        f_pred_vars = []
        created_ids = []
        for m_idx, (weight, ml_model) in enumerate(constraint_models):
            m_prefix = f"{prefix}_m{m_idx}"
            m_id = id(ml_model)
            if m_id not in self.embedded_models_cache:
                f_s = embed_model(
                    self.opt, ml_model, self.x,
                    self.instance.variable_lb, self.instance.variable_ub,
                    name_prefix=m_prefix, rho=rho,
                )
                self.embedded_models_cache[m_id] = f_s
                created_ids.append(m_id)
            f_pred_vars.append(weight * self.embedded_models_cache[m_id])

        main_constr = None
        if f_pred_vars:
            main_constr = self.opt.addConstr(
                gp.quicksum(f_pred_vars) <= rhs,
                name=f"cp_constr_{c_idx}_{self.n_models}",
            )

        self.opt.update()
        new_constrs = list(set(self.opt.getConstrs()) - old_constrs)
        new_vars = list(set(self.opt.getVars()) - old_vars)

        self.scenario_constrs_map[self.n_models] = new_constrs
        self.scenario_vars_map[self.n_models] = new_vars
        self.scenario_model_ids_map[self.n_models] = created_ids
        self.scenario_constrs.append(main_constr)
        self.n_models += 1

    def add_objective_cut(self, obj_val: float, iteration: int):
        self.opt.addConstr(self.obj_expr >= obj_val, name=f"obj_bound_{iteration}")
        self.opt.update()

    def solve(self):
        self.opt.optimize()
        if self.opt.Status != GRB.OPTIMAL:
            return None, np.inf
        return np.array([v.X for v in self.x]), self.opt.ObjVal


def prune_inactive_scenarios(master: IncrementalMaster, slack_threshold: float = 0.1):
    to_remove = []
    total_active = 0
    for s, constr in enumerate(master.scenario_constrs):
        if constr is not None:
            total_active += 1
            if constr.Slack > slack_threshold:
                to_remove.append(s)
    for s in reversed(to_remove):
        master.remove_scenario(s)
    return len(to_remove), total_active


# ---------------------------------------------------------------------------
# Separation primitives
# ---------------------------------------------------------------------------

def _evaluate_real_candidate(args):
    candidate, X_train, y_train, x_2d, model_type, model_params = args
    model = retrain_on_bootstrap(X_train, y_train, candidate, model_type, model_params)
    val = model.predict(x_2d)[0]
    return val, candidate


def localized_bootstrap_separation(model_data, x_current, model_type, model_params,
                                   k_neighbors_frac, n_candidates, seed,
                                   distance_feature_indices: list = None,
                                   k_neighbors_min: int = 1):
    """Localized bootstrap separation: worst-case over the localized ensemble.

    Each localized bootstrap resample is retrained with the **actual** constraint
    model (``model_type``) and scored at ``x_current``; the worst-case (max)
    candidate is returned. We rank with the real model rather than a CART proxy
    so the cut is correct for non-tree constraints (linear / SVM / XGB) too.

    ``distance_feature_indices`` localizes the neighbor pool to a subset of
    columns (e.g. context features) when provided; defaults to full-vector
    distance, preserving the synthetic / decision-only behavior.

    Used by the basic separation strategy (single global LP); the coherent
    strategy resamples shared scenarios directly instead.
    """
    x_2d = np.atleast_2d(x_current)
    candidates = localized_bootstrap_indices(
        model_data.X_train, x_current, k_neighbors_frac, n_candidates, seed,
        distance_feature_indices=distance_feature_indices,
        k_neighbors_min=k_neighbors_min,
    )

    args_list = [
        (cand, model_data.X_train, model_data.y_train, x_2d, model_type, model_params)
        for cand in candidates
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_evaluate_real_candidate, args_list))

    best_value, best_candidate = max(results, key=lambda vc: vc[0])

    # Retrain the worst-case candidate in-process to return the model object
    # (retraining is deterministic given the resample indices and params).
    best_model = retrain_on_bootstrap(
        model_data.X_train, model_data.y_train,
        best_candidate, model_type, model_params,
    )
    return best_candidate, best_value, best_model


def _union_neighbor_pool(X_train, query_points, k_neighbors_frac,
                         distance_feature_indices=None,
                         k_neighbors_min: int = 1):
    """Union of the k-nearest training indices across a set of query points.

    Localizes a single shared pool to the region spanned by the current query
    points. Distances are computed in the **standardized** feature space
    (z-scored by the training-set column mean/std) so features on different
    scales contribute equally. With ``distance_feature_indices=None`` the
    distance uses the full feature vector (context + decision).
    """
    n = X_train.shape[0]
    k = resolve_neighbor_pool_size(n, k_neighbors_frac, k_neighbors_min)
    cols = list(distance_feature_indices) if distance_feature_indices else None
    Xc = X_train[:, cols] if cols is not None else X_train
    # Standardize the training matrix once; query points are standardized with
    # the same training statistics so the distance space is consistent.
    mu = Xc.mean(axis=0)
    sigma = Xc.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xc_std = (Xc - mu) / sigma
    pool = set()
    for x in query_points:
        x = np.asarray(x, dtype=float).ravel()
        xc = x[cols] if cols is not None else x
        xc_std = (xc - mu) / sigma
        dist = np.linalg.norm(Xc_std - xc_std, axis=1)
        pool.update(np.argsort(dist)[:k].tolist())
    return np.array(sorted(pool), dtype=int)


def select_anchor_contexts(X: np.ndarray,
                           context_var_indices: list,
                           n_anchors: Optional[int],
                           method: str = "kmedoids",
                           seed: int = 42) -> np.ndarray:
    """Select representative anchor rows whose context columns span the cohort space.

    For parametric-context problems (gastric), the separation oracle is run once
    per LP (one per context). Rather than separate at every training cohort, we
    pick ``n_anchors`` representatives that cover the context space; the cuts
    found are full embedded models that are valid at every context.

    Parameters
    ----------
    X : (n, n_features) source rows (training or test feature matrix).
    context_var_indices : columns that index the parametric context.
    n_anchors : number of anchors; ``None`` or ``>= n`` returns all rows.
    method : ``"kmedoids"`` (k-means centroids snapped to nearest real rows),
        ``"sample"`` (uniform random rows), or ``"all"``.

    Returns
    -------
    Array of full feature rows; callers read the context columns to fix per-LP
    context during separation.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if (not context_var_indices or method == "all"
            or n_anchors is None or n_anchors >= n):
        return X.copy()

    cols = list(context_var_indices)
    Z = X[:, cols]
    rng = np.random.RandomState(seed)
    k = max(1, min(int(n_anchors), n))

    if method == "sample":
        idx = rng.choice(n, size=k, replace=False)
        return X[np.sort(idx)].copy()

    # kmedoids: cluster context space, snap each centroid to its nearest real row
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    medoid_idx = []
    for c in range(k):
        d = np.linalg.norm(Z - km.cluster_centers_[c], axis=1)
        medoid_idx.append(int(np.argmin(d)))
    medoid_idx = sorted(set(medoid_idx))
    return X[medoid_idx].copy()


def _get_anchor_rows(instance: ProblemInstance,
                     cp_anchors: Optional[np.ndarray],
                     cp_anchor_source: str,
                     cp_n_anchors: Optional[int],
                     cp_anchor_method: str,
                     seed: int) -> Optional[np.ndarray]:
    """Return representative anchor rows, or ``None`` for non-contextual problems."""
    if not instance.context_var_indices:
        return None
    if cp_anchors is not None:
        return np.asarray(cp_anchors, dtype=float)
    if cp_anchor_source == "test":
        source = instance.X_test
    else:
        source = instance.X_train if instance.X_train is not None else instance.X_test
    return select_anchor_contexts(
        source, instance.context_var_indices,
        cp_n_anchors, cp_anchor_method, seed,
    )


def _resolve_nearest_distance(nearest_distance: str, instance: ProblemInstance):
    """Column subset for nearest-anchor assignment at prescribe time."""
    contextual = bool(instance.context_var_indices)
    if nearest_distance == "full":
        return None
    if nearest_distance in ("context", "auto"):
        return list(instance.context_var_indices) if contextual else None
    raise ValueError(f"Unknown nearest_distance: {nearest_distance}")


def nearest_anchor_index(context_row: np.ndarray,
                         anchor_rows: np.ndarray,
                         context_var_indices: list,
                         distance_feature_indices: Optional[list] = None) -> int:
    """Index of the anchor row closest to ``context_row`` (L2 on selected columns)."""
    cols = list(distance_feature_indices) if distance_feature_indices else None
    Z = anchor_rows[:, cols] if cols is not None else anchor_rows
    z = np.asarray(context_row, dtype=float).ravel()
    zc = z[cols] if cols is not None else z
    return int(np.argmin(np.linalg.norm(Z - zc, axis=1)))


def _is_constraint_constraint(constraint) -> bool:
    """True if constraint contributes learned bounds (not objective-only)."""
    return any(md.obj_weight == 0.0 for md in constraint.models_data)


def _train_nominal_with_configs(instance, model_type, model_params):
    trained_cache = {}
    config_idx = 0
    result = []
    for constraint in instance.constraints:
        row = []
        for model_data in constraint.models_data:
            md_id = id(model_data)
            if md_id not in trained_cache:
                m_type, m_params = resolve_constraint_config(
                    instance, config_idx, model_type, model_params
                )
                trained_cache[md_id] = train_model(
                    model_data.X_train, model_data.y_train, m_type, m_params
                )
            row.append((model_data.weight, trained_cache[md_id], model_data.obj_weight))
            config_idx += 1
        result.append(row)
    return result


def _write_cp_trace(history: "CPHistory", path: Optional[str]) -> None:
    """Persist per-iteration CP history (violation, objective) to CSV."""
    if not path:
        return
    import csv
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "max_violation", "objective"])
        for i in range(len(history.violations)):
            viol = history.violations[i]
            viol_out = viol if np.isfinite(viol) else ""
            obj = history.objectives[i] if i < len(history.objectives) else ""
            writer.writerow([i, viol_out, obj])


# ---------------------------------------------------------------------------
# Shared scaffolding (build / anchors / solve / finalize)
# ---------------------------------------------------------------------------

def _build_master_with_nominal(instance, model_type, model_params, rho,
                               robustify_objective: bool = True):
    """Train nominal models, build the master MIP, embed objective + initial cuts.

    Returns ``(master, model_config_map)`` where ``model_config_map`` maps each
    ``model_data`` id to its resolved ``(model_type, model_params)`` for later
    bootstrap retraining during separation.
    """
    trained_constraints = _train_nominal_with_configs(instance, model_type, model_params)
    master = IncrementalMaster(instance, [], rho=rho)

    nominal_obj_terms = []
    for c_idx, constraint_models in enumerate(trained_constraints):
        for m_idx, (weight, ml_model, obj_weight) in enumerate(constraint_models):
            if obj_weight == 0.0:
                continue
            m_id = id(ml_model)
            if m_id not in master.embedded_models_cache:
                master.embedded_models_cache[m_id] = embed_model(
                    master.opt, ml_model, master.x,
                    instance.variable_lb, instance.variable_ub,
                    name_prefix=f"cp_obj_c{c_idx}_m{m_idx}", rho=rho,
                )
            nominal_obj_terms.append(obj_weight * master.embedded_models_cache[m_id])

    if nominal_obj_terms:
        if robustify_objective:
            master.set_epigraph_objective(nominal_obj_terms)
        else:
            master.obj_expr = master.base_cost + gp.quicksum(nominal_obj_terms)
            master.opt.setObjective(master.obj_expr, GRB.MINIMIZE)
            master.opt.update()

    for c_idx, constraint in enumerate(instance.constraints):
        if not _is_constraint_constraint(constraint):
            continue
        constraint_models = [
            (w, m) for w, m, ow in trained_constraints[c_idx] if ow == 0.0
        ]
        master.add_scenario(c_idx, constraint_models, constraint.rhs, rho=rho)

    config_idx = 0
    model_config_map = {}
    for constraint in instance.constraints:
        for model_data in constraint.models_data:
            model_config_map[id(model_data)] = resolve_constraint_config(
                instance, config_idx, model_type, model_params
            )
            config_idx += 1

    return master, model_config_map


def _setup_anchors(instance, master, cp_anchors, cp_anchor_source,
                   cp_n_anchors, cp_anchor_method, seed):
    """Build the anchor set (one LP per representative context) + saved bounds.

    Non-contextual problems (synthetic) get the single degenerate anchor
    ``[None]`` (one global LP). Returns ``(anchors, ctx_bounds)`` where
    ``ctx_bounds`` records the original context-variable bounds so the master
    can be restored to its context-free form before the final solve.
    """
    if not instance.context_var_indices:
        return [None], {}

    anchor_rows = _get_anchor_rows(
        instance, cp_anchors, cp_anchor_source, cp_n_anchors, cp_anchor_method, seed,
    )
    anchors = [anchor_rows[i] for i in range(anchor_rows.shape[0])]
    ctx_bounds = {
        c: (master.x[c].lb, master.x[c].ub)
        for c in instance.context_var_indices
    }
    return anchors, ctx_bounds


def _fix_anchor_context(master, instance, anchor):
    """Pin the master's context variables to one anchor's context (no-op if None)."""
    if anchor is None:
        return
    for c in instance.context_var_indices:
        val = float(anchor[c])
        master.x[c].lb = val
        master.x[c].ub = val
    master.opt.update()


def _restore_context_bounds(master, ctx_bounds):
    for c, (lb, ub) in ctx_bounds.items():
        master.x[c].lb = lb
        master.x[c].ub = ub
    if ctx_bounds:
        master.opt.update()


def _solve_all_anchors(master, instance, anchors, obj_bounds=None,
                       collect_slack=False):
    """Solve the master at every anchor; return feasible ``(idx, x*, obj)`` + p_infeas.

    ``obj_bounds`` -- optional ``{a_idx: lower_bound}``; while solving anchor
    ``a_idx`` a temporary no-deterioration cut ``obj_expr >= lower_bound`` is
    imposed and removed afterwards. Valid for a *fixed* context because the cuts
    only shrink that context's region, so its optimum is monotone non-decreasing.

    ``collect_slack`` -- if True, also return ``{scenario_idx: min slack across
    feasible anchors}`` for the active scenario constraints (for multi-anchor
    pruning: a cut is globally inactive only if it is slack at *every* ``x*``).
    """
    feasible = []
    n_infeas = 0
    min_slack = {} if collect_slack else None
    for a_idx, anchor in enumerate(anchors):
        _fix_anchor_context(master, instance, anchor)

        tmp = None
        if obj_bounds is not None and a_idx in obj_bounds:
            tmp = master.opt.addConstr(
                master.obj_expr >= obj_bounds[a_idx], name=f"obj_bnd_{a_idx}"
            )
            master.opt.update()

        x_q, obj_q = master.solve()

        if collect_slack and x_q is not None:
            for s, constr in enumerate(master.scenario_constrs):
                if constr is not None:
                    sl = constr.Slack
                    if s not in min_slack or sl < min_slack[s]:
                        min_slack[s] = sl

        if tmp is not None:
            master.opt.remove(tmp)
            master.opt.update()

        if x_q is None:
            n_infeas += 1
        else:
            feasible.append((a_idx, x_q, obj_q))

    p_infeas = n_infeas / len(anchors)
    if collect_slack:
        return feasible, p_infeas, min_slack
    return feasible, p_infeas


def _p_infeas_after_cuts(master, instance, anchors, feasible, n_infeas_base, alpha):
    """Check whether p_infeas stays within alpha after recently-added cuts.

    Anchors that were already infeasible before this call remain infeasible
    (cuts only tighten), so ``n_infeas_base`` is a free lower bound on the
    result. We solve only the currently-feasible subset, sorted by **descending
    obj_q**: anchors that achieved a higher objective under the current cuts
    were closer to the feasibility boundary and are the most likely to flip
    infeasible first, maximising the chance of an early exit.

    Returns ``(p_infeas_after, fits)`` where ``fits = p_infeas_after <= alpha``.
    The ``p_infeas_after`` value may be a lower bound (not the true fraction)
    when we exit early -- callers should only use it for rejection (fits=False).
    """
    n_total = len(anchors)
    n_infeas = n_infeas_base
    # Pre-check: already over budget before solving anything.
    if n_infeas > (alpha + 1e-12) * n_total:
        return n_infeas / n_total, False
    for a_idx, _, _ in sorted(feasible, key=lambda t: t[2], reverse=True):
        _fix_anchor_context(master, instance, anchors[a_idx])
        x_q, _ = master.solve()
        if x_q is None:
            n_infeas += 1
            if n_infeas > (alpha + 1e-12) * n_total:
                return n_infeas / n_total, False
    return n_infeas / n_total, True


def _resolve_distance(cp_distance, instance):
    """Resolve the localization columns from the distance policy.

    ``"full"`` (default) -> full-vector distance (context + decision), so the
    bootstrap pool follows the incumbent ``x*``. ``"context"`` / ``"auto"``
    localize on the context columns only when the problem is contextual.
    """
    contextual = bool(instance.context_var_indices)
    if cp_distance in ("context", "auto"):
        return list(instance.context_var_indices) if contextual else None
    return None  # "full"


def _finalize(instance, master, ctx_bounds, history, status, total_start,
              cp_trace_path, last_x=None, last_obj=np.inf, anchors=None):
    """Restore context-free bounds, do a tight final solve, persist trace."""
    d = instance.n_features
    print("Re-solving with default MIP gap...")
    _restore_context_bounds(master, ctx_bounds)
    if anchors and len(anchors) == 1 and anchors[0] is not None:
        _fix_anchor_context(master, instance, anchors[0])
    master.opt.Params.MIPGap = 1e-4
    x_final, obj_final = master.solve()
    _restore_context_bounds(master, ctx_bounds)
    if x_final is not None:
        x_opt, obj_value = x_final, obj_final
    elif last_x is not None:
        x_opt, obj_value = last_x, last_obj
    else:
        x_opt, obj_value = np.zeros(d), np.inf

    _write_cp_trace(history, cp_trace_path)
    return SolutionResult(
        x_opt=x_opt,
        obj_value=obj_value,
        status=status,
        models_embedded=master.n_models,
        solve_time=time.time() - total_start,
        opt=master.opt,
        x=master.x,
        iterations=history.iterations,
    ), history


# ---------------------------------------------------------------------------
# Separation strategies (one iteration each)
# ---------------------------------------------------------------------------

@dataclass
class _SepEnv:
    """Shared, immutable handles the strategies read each iteration."""
    instance: ProblemInstance
    master: IncrementalMaster
    anchors: list
    model_config_map: dict
    distance_feature_indices: Optional[list]
    rho: float
    seed: int
    k_neighbors_min: int = 1


@dataclass
class _StepResult:
    """Outcome of one separation step; the driver records and decides to stop.

    ``obj`` / ``x`` / ``violation`` are appended to history when not ``None``.
    """
    stop: bool
    status: str = "running"
    obj: Optional[float] = None
    x: Optional[np.ndarray] = None
    violation: Optional[float] = None


class _BasicSeparation:
    """Plain worst-case cutting planes for a single global LP.

    Solves the master, takes the worst-case localized-bootstrap model per
    constraint at the optimal solution ``x*``, and cuts every constraint that
    violates -- no chance budget. Terminates when nothing violates (the
    worst-case robust region has converged). Used for the synthetic case (one LP,
    one constraint, no context) and for contextual single-anchor masters
    (``per_anchor_nearest`` with ``dlt_only``): the lone anchor's context is
    pinned before each solve so only treatment variables are optimized.
    """

    def __init__(self, k_neighbors_frac, n_candidates, k_neighbors_min=1):
        self.k_neighbors_frac = k_neighbors_frac
        self.n_candidates = n_candidates
        self.k_neighbors_min = k_neighbors_min

    def step(self, env: _SepEnv, iteration: int) -> _StepResult:
        iter_start = time.time()
        inst, master = env.instance, env.master

        if len(env.anchors) == 1 and env.anchors[0] is not None:
            _fix_anchor_context(master, inst, env.anchors[0])

        x_star, obj_star = master.solve()
        if x_star is None:
            return _StepResult(stop=True, status="infeasible")

        sep_cache = {}
        max_violation = -np.inf
        scenarios_to_add = []
        for c_idx, constraint in enumerate(inst.constraints):
            if not _is_constraint_constraint(constraint):
                continue

            worst_case_models = []
            constraint_val = 0.0
            for m_idx, model_data in enumerate(constraint.models_data):
                if model_data.obj_weight != 0.0:
                    continue
                md_id = id(model_data)
                m_type, m_params = env.model_config_map[md_id]

                if md_id in sep_cache:
                    best_model, best_value = sep_cache[md_id]
                else:
                    _, best_value, best_model = localized_bootstrap_separation(
                        model_data, x_star, m_type, m_params,
                        self.k_neighbors_frac, self.n_candidates,
                        env.seed + iteration * 1000 + c_idx * 10 + m_idx,
                        distance_feature_indices=env.distance_feature_indices,
                        k_neighbors_min=env.k_neighbors_min,
                    )
                    sep_cache[md_id] = (best_model, best_value)

                worst_case_models.append((model_data.weight, best_model))
                constraint_val += model_data.weight * best_value

            violation = constraint_val - constraint.rhs
            max_violation = max(max_violation, violation)
            if violation > 1e-6:
                scenarios_to_add.append((c_idx, worst_case_models, constraint.rhs))

        print(
            f"Iter {iteration}: Obj={obj_star:.4f} "
            f"Max Violation={max_violation:.4f} Time={time.time()-iter_start:.2f}s"
        )

        # Single global LP: prune slack scenarios and add the no-deterioration
        # objective cut (both valid only with one objective / active set).
        if iteration > 0:
            dynamic_slack = max(0.1, max_violation)
            pruned_count, total_active = prune_inactive_scenarios(
                master, slack_threshold=dynamic_slack
            )
            if pruned_count > 0:
                print(
                    f"Iter {iteration}: Pruned {pruned_count}/{total_active} "
                    f"inactive scenarios"
                )
        if master.t_obj is not None:
            master.add_objective_cut(obj_star, iteration)

        for c_idx, worst_case_models, rhs in scenarios_to_add:
            master.add_scenario(c_idx, worst_case_models, rhs, rho=env.rho)

        viol_metric = max_violation if np.isfinite(max_violation) else np.inf
        return _StepResult(
            stop=not scenarios_to_add,
            status="optimal" if not scenarios_to_add else "running",
            obj=obj_star,
            x=x_star.copy(),
            violation=viol_metric,
        )


class _CoherentSeparation:
    """Simultaneous worst-case across all constraints via shared scenarios.

    A scenario is one **shared** localized bootstrap resample (a plausible
    relabeling of the trial) used to train every constraint jointly, so the
    adversary is a single relabeling rather than an independent worst case per
    constraint. Each iteration solves the master at every context, collects the
    optimal solutions ``x*`` (one per context), localizes one pool to the union
    of their neighborhoods, draws ``n_scenarios`` resamples, and trains one model
    per constraint per scenario.

    When the problem has a **learned objective** (epigraph ``min base_cost +
    t_obj`` with ``t_obj >= sum(obj_weight_i f_i)``), the same shared relabeling
    also retrains the objective model(s). The epigraph is treated as one extra
    "constraint" ``sum(obj_weight_i f_i^s(x)) <= t_obj`` evaluated against each
    ``x*``'s epigraph value (``t_val = obj - c'x``); when the worst scenario
    worsens the objective there, its cut is added and ``t_obj`` rises, so the
    objective is robustified by the same worst-case mechanism as the constraints.

    **Worst-case selection (all cases).** Every sampled scenario is scored by its
    **normalized average distance**: each constraint exceedance ``sum w_i f_i^s(x*)
    - rhs`` is divided by ``max(1, |rhs|)`` and each objective exceedance
    ``sum obj_weight_i f_i^s(x*) - t_val`` by ``max(1, |t_val|)``, then the
    normalized exceedances are averaged over all ``(x*, outcome)`` cells.
    Averaging (rather than summing) keeps the metric on a 0–1 scale regardless of
    the number of constraints or patients, so ``dist_tol`` has a consistent
    meaning. The scenario with the largest average distance is the adversary.

    **Termination.** We stop when either (1) the worst scenario's normalized average
    distance is ``<= dist_tol`` (robust enough -- some residual violation is
    allowed), or (2) no sampled scenario can be cut without pushing more than
    ``alpha`` of the optimal solves infeasible (``coverage_cap``).

    **Coverage cap (multiple ``x*`` only).** ``alpha`` bounds the fraction of
    optimal solves (patients) allowed to become infeasible. We add the *most
    adversarial scenario we can afford* (worst-first by total distance, keeping
    ``p_infeas <= alpha``), falling back to the next-worst if the worst
    over-tightens. With a **single ``x*``** we require ``p_infeas == 0`` (same
    rollback loop with ``alpha=0``): only cuts that leave the anchor solve
    feasible are kept.

    Two solver-acceleration tricks are applied (sound because cuts are global
    embedded models that only shrink each fixed context's region):

    - a **per-anchor no-deterioration** objective bound ``obj_expr >= obj_q`` is
      re-imposed while solving each anchor (warm lower bound from its previous
      optimum);
    - **multi-anchor pruning** drops scenario cuts that are slack at *every*
      ``x*`` (so removing them changes no current solution and not ``p_infeas``).
    """

    def __init__(self, k_neighbors_frac, n_scenarios, alpha, single_point, dist_tol,
                 k_neighbors_min=1):
        self.k_neighbors_frac = k_neighbors_frac
        self.n_scenarios = n_scenarios
        self.alpha = alpha               # coverage cap: max fraction of x* infeasible
        self.single_point = single_point
        self.dist_tol = dist_tol         # stop once worst normalized distance <= this
        self.k_neighbors_min = k_neighbors_min
        self.obj_bound = {}          # a_idx -> best (max) objective seen so far
        self.prev_max_exceed = 0.0   # scale for the dynamic pruning threshold (raw)
        self.last_added_ids: List[int] = []  # scenario ids from the last accepted cut

    def step(self, env: _SepEnv, iteration: int) -> _StepResult:
        iter_start = time.time()
        inst, master = env.instance, env.master

        feasible, p_infeas, min_slack = _solve_all_anchors(
            master, inst, env.anchors,
            obj_bounds=self.obj_bound, collect_slack=True,
        )
        if not feasible:
            if iteration > 0 and self.last_added_ids:
                for s in reversed(self.last_added_ids):
                    master.remove_scenario(s)
                master.opt.update()
                print(
                    f"Iter {iteration}: rolled back scenario(s) {self.last_added_ids} "
                    f"(solve infeasible after last cut); stopping.",
                    flush=True,
                )
                self.last_added_ids = []
                return _StepResult(stop=True, status="coverage_cap")
            print(f"Iter {iteration}: all optimal solves infeasible; stopping.")
            return _StepResult(stop=True, status="infeasible")

        # Per-anchor no-deterioration bound: each context's optimum only rises as
        # cuts are added, so store its best objective to warm-start later solves.
        for a_idx, _, obj_q in feasible:
            if obj_q > self.obj_bound.get(a_idx, -np.inf):
                self.obj_bound[a_idx] = obj_q

        # Multi-anchor pruning: drop cuts that are slack at *every* x* (globally
        # inactive). Threshold scales with the previous iteration's worst cut, as
        # in the basic case; skipped on iteration 0.
        if iteration > 0 and min_slack:
            dynamic_slack = max(0.1, self.prev_max_exceed)
            to_remove = [s for s, sl in min_slack.items() if sl > dynamic_slack]
            for s in to_remove:
                master.remove_scenario(s)
            if to_remove:
                master.opt.update()
                n_active = sum(1 for c in master.scenario_constrs if c is not None)
                print(
                    f"Iter {iteration}: Pruned {len(to_remove)} globally-inactive "
                    f"scenario(s) (ids {sorted(to_remove)}); {n_active} active remaining."
                )

        x_stars = [x_q for (_, x_q, _) in feasible]
        last_obj = float(np.mean([obj_q for (_, _, obj_q) in feasible]))

        ref_md = inst.constraints[0].models_data[0]
        n_train = len(ref_md.y_train)

        # Learned objective (epigraph): the same shared relabeling that stresses
        # the constraints also relabels the objective outcome, so we treat the
        # epigraph as one more "constraint" sum(obj_weight_i f_i^s(x)) <= t_obj
        # and rank/cut it jointly. Each x* carries its epigraph value
        # t_val = obj - c'x; the objective's (normalized) exceedance over t_val
        # feeds the worst-case distance, so it influences scenario selection but
        # is never gated by the coverage cap alpha (it is the objective, not a
        # feasibility requirement).
        has_obj = master.t_obj is not None
        obj_specs = []
        obj_c_idx = None
        if has_obj:
            for c_idx, constraint in enumerate(inst.constraints):
                for md in constraint.models_data:
                    if md.obj_weight != 0.0:
                        obj_specs.append((md.obj_weight, md))
                        if obj_c_idx is None:
                            obj_c_idx = c_idx

        points = []
        for (_, x_q, obj_q) in feasible:
            t_val = obj_q - float(np.dot(inst.cost_vector, x_q)) if has_obj else None
            points.append((x_q, t_val))
        n_points = len(points)

        # Shared, decision-dependent localized pool -> B coherent scenarios.
        pool = _union_neighbor_pool(
            ref_md.X_train, x_stars, self.k_neighbors_frac,
            env.distance_feature_indices,
            k_neighbors_min=env.k_neighbors_min,
        )
        pool_frac = len(pool) / n_train
        rng = np.random.RandomState(env.seed + iteration)
        scenarios = [
            rng.choice(pool, size=n_train, replace=True) for _ in range(self.n_scenarios)
        ]

        # Evaluate every sampled scenario; keep *all* that produce a cut so we can
        # fall back to a less aggressive adversary if the worst one over-tightens.
        # The worst case is ranked by **normalized average distance**: each
        # exceedance is divided by its own scale (so no single large-scale outcome
        # dominates) then averaged over all (x*, outcome) cells (so the metric is
        # 0-1 regardless of the number of constraints or patients).
        candidates = []   # (total_dist, raw_max_exceed, cuts)
        for idx in scenarios:
            per_constraint_models = {}
            for c_idx, constraint in enumerate(inst.constraints):
                if not _is_constraint_constraint(constraint):
                    continue
                models = []
                for model_data in constraint.models_data:
                    if model_data.obj_weight != 0.0:
                        continue
                    m_type, m_params = env.model_config_map[id(model_data)]
                    theta = retrain_on_bootstrap(
                        model_data.X_train, model_data.y_train, idx, m_type, m_params
                    )
                    models.append((model_data.weight, theta))
                per_constraint_models[c_idx] = (models, constraint.rhs)

            obj_models = []
            for obj_weight, md in obj_specs:
                m_type, m_params = env.model_config_map[id(md)]
                theta = retrain_on_bootstrap(
                    md.X_train, md.y_train, idx, m_type, m_params
                )
                obj_models.append((obj_weight, theta))

            sum_norm_exceed = 0.0   # sum of normalized exceedances (for averaging)
            raw_max_exceed = 0.0    # largest raw exceedance (pruning-threshold scale)
            n_outcomes = len(per_constraint_models) + (1 if has_obj else 0)
            violated_constraints = set()
            obj_violated = False
            for (x_star, t_val) in points:
                xs2d = np.atleast_2d(x_star)
                for c_idx, (models, rhs) in per_constraint_models.items():
                    exceed = sum(w * th.predict(xs2d)[0] for w, th in models) - rhs
                    if exceed > 1e-6:
                        sum_norm_exceed += exceed / max(1.0, abs(rhs))
                        raw_max_exceed = max(raw_max_exceed, exceed)
                        violated_constraints.add(c_idx)
                if has_obj:
                    obj_pred = sum(ow * th.predict(xs2d)[0] for ow, th in obj_models)
                    exceed = obj_pred - t_val
                    if exceed > 1e-6:
                        sum_norm_exceed += exceed / max(1.0, abs(t_val))
                        raw_max_exceed = max(raw_max_exceed, exceed)
                        obj_violated = True

            # Average over all (x*, outcome) pairs so dist_tol is on a 0-1 scale.
            n_cells = max(1, n_points * n_outcomes)
            total_dist = sum_norm_exceed / n_cells

            cuts = [
                (c_idx, per_constraint_models[c_idx][0], per_constraint_models[c_idx][1])
                for c_idx in sorted(violated_constraints)
            ]
            # Robustify the objective with the same relabeling: raise the epigraph
            # floor t_obj to this scenario's (worse) objective.
            if obj_violated:
                cuts.append((obj_c_idx, obj_models, master.t_obj))
            if cuts:
                candidates.append((total_dist, raw_max_exceed, cuts))

        # Rank adversaries worst-first by normalized average distance.
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_dist = candidates[0][0] if candidates else 0.0
        # Scale next iteration's pruning threshold by the worst raw cut.
        self.prev_max_exceed = candidates[0][1] if candidates else 0.0

        print(
            f"Iter {iteration}: {'Obj' if self.single_point else 'AvgObj'}={last_obj:.4f} "
            f"WorstScenarioNormDist={best_dist:.4f} "
            f"p_infeas={p_infeas*100:.1f}% PoolFrac={pool_frac:.3f} "
            f"Time={time.time()-iter_start:.2f}s",
            flush=True,
        )

        history_viol = best_dist

        # Terminate (1): even the worst sampled relabeling leaves the normalized
        # total distance within the allowance -- robust enough.
        if best_dist <= self.dist_tol:
            return _StepResult(stop=True, status="optimal", obj=last_obj, violation=history_viol)

        # Add the most adversarial scenario we can afford (worst-first). Single
        # x* uses alpha=0: every cut must keep the anchor solve feasible.
        feas_alpha = 0.0 if self.single_point else self.alpha
        n_infeas_base = len(env.anchors) - len(feasible)
        if n_infeas_base > (feas_alpha + 1e-12) * len(env.anchors):
            cap_label = "feasibility" if self.single_point else "coverage"
            print(
                f"Iter {iteration}: {cap_label} cap hit before adding any cuts "
                f"(p_infeas={p_infeas*100:.1f}% > alpha {feas_alpha*100:.1f}%); stopping.",
                flush=True,
            )
            return _StepResult(stop=True, status="coverage_cap", obj=last_obj,
                               violation=history_viol)

        for cand_rank, (_dist, _mx, cuts) in enumerate(candidates):
            added_ids = []
            for c_idx, models, rhs in cuts:
                master.add_scenario(c_idx, models, rhs, rho=env.rho)
                added_ids.append(master.n_models - 1)
            p_infeas_after, fits = _p_infeas_after_cuts(
                master, inst, env.anchors, feasible, n_infeas_base, feas_alpha
            )
            if fits:
                self.last_added_ids = added_ids
                cap_note = (
                    "feasible"
                    if self.single_point
                    else f"p_infeas={p_infeas_after*100:.1f}%"
                )
                print(
                    f"Iter {iteration}: Added scenario(s) {added_ids} "
                    f"(candidate rank {cand_rank + 1}/{len(candidates)}, "
                    f"norm_dist={_dist:.4f}); {cap_note}",
                    flush=True,
                )
                return _StepResult(stop=False, status="running", obj=last_obj,
                                   violation=history_viol)
            for s in reversed(added_ids):
                master.remove_scenario(s)
            master.opt.update()
            reject_msg = (
                "would become infeasible"
                if self.single_point
                else f"p_infeas_after={p_infeas_after*100:.1f}% > alpha {feas_alpha*100:.1f}%"
            )
            print(
                f"Iter {iteration}: Rolled back scenario(s) {added_ids} "
                f"(candidate rank {cand_rank + 1}/{len(candidates)}, "
                f"norm_dist={_dist:.4f}); {reject_msg}",
                flush=True,
            )

        cap_label = "feasibility" if self.single_point else "coverage"
        print(
            f"Iter {iteration}: {cap_label} cap hit (no sampled scenario keeps "
            f"p_infeas <= alpha {feas_alpha*100:.1f}%); stopping.",
            flush=True,
        )
        return _StepResult(stop=True, status="coverage_cap", obj=last_obj,
                           violation=history_viol)


def _run_cp_loop(instance: ProblemInstance,
                 *,
                 model_type: str,
                 model_params: dict,
                 rho: float,
                 max_iterations: int,
                 cp_k_neighbors_frac: float,
                 cp_n_candidates: int,
                 seed: int,
                 cp_alpha: float,
                 cp_distance: str,
                 cp_dist_tol: float,
                 cp_robustify_objective: bool,
                 anchors: list,
                 cp_k_neighbors_min: int = 1,
                 cp_trace_path: Optional[str] = None) -> tuple[SolutionResult, CPHistory]:
    """Run the CP cut loop for a fixed anchor set (one or many contexts)."""
    total_start = time.time()
    history = CPHistory()

    print("    [cp] Training nominal models and building master MIP...", flush=True)
    master, model_config_map = _build_master_with_nominal(
        instance, model_type, model_params, rho,
        robustify_objective=cp_robustify_objective,
    )
    if not instance.context_var_indices:
        ctx_bounds = {}
    else:
        ctx_bounds = {
            c: (master.x[c].lb, master.x[c].ub)
            for c in instance.context_var_indices
        }

    distance_feature_indices = _resolve_distance(cp_distance, instance)
    env = _SepEnv(
        instance=instance, master=master, anchors=anchors,
        model_config_map=model_config_map,
        distance_feature_indices=distance_feature_indices,
        rho=rho, seed=seed,
        k_neighbors_min=cp_k_neighbors_min,
    )

    n_constraints = sum(1 for c in instance.constraints if _is_constraint_constraint(c))
    has_obj_models = any(
        md.obj_weight != 0.0 for c in instance.constraints for md in c.models_data
    )
    if not cp_robustify_objective:
        has_obj_models = False
    single_point = len(anchors) == 1
    use_basic = (n_constraints <= 1) and single_point and not has_obj_models

    if use_basic:
        strategy = _BasicSeparation(
            cp_k_neighbors_frac, cp_n_candidates, cp_k_neighbors_min,
        )
        mode = "basic"
    else:
        strategy = _CoherentSeparation(
            cp_k_neighbors_frac, cp_n_candidates, cp_alpha,
            single_point, cp_dist_tol, cp_k_neighbors_min,
        )
        mode = "coherent (single x*)" if single_point else "coherent (multi x*)"

    n_solves = 1 if single_point else len(anchors)
    extra = f", alpha={cp_alpha}" if (not use_basic and not single_point) else ""
    obj_flag = "robustify_objective" if cp_robustify_objective else "nominal_objective"
    n_pool = resolve_neighbor_pool_size(
        len(instance.constraints[0].models_data[0].y_train),
        cp_k_neighbors_frac, cp_k_neighbors_min,
    )
    print(
        f"    [cp] separation={mode}; constraints={n_constraints}, "
        f"optimal solves={n_solves}, distance={cp_distance}, {obj_flag}, "
        f"neighbor_pool>={cp_k_neighbors_min} (k={n_pool}){extra}",
        flush=True,
    )

    last_x, last_obj = None, np.inf
    status = "max_iterations"
    for iteration in range(max_iterations):
        res = strategy.step(env, iteration)

        if res.obj is not None:
            history.objectives.append(res.obj)
            last_obj = res.obj
        if res.x is not None:
            history.x_solutions.append(res.x)
            last_x = res.x
        if res.violation is not None:
            history.violations.append(res.violation)
        if res.obj is not None or res.violation is not None:
            history.iterations = iteration + 1

        if res.stop:
            status = res.status
            break

    return _finalize(
        instance, master, ctx_bounds, history, status,
        total_start, cp_trace_path, last_x, last_obj, anchors=anchors,
    )


def solve_cp(instance: ProblemInstance,
             model_type: str = "rf",
             model_params: dict = None,
             rho: float = 0.0,
             max_iterations: int = 50,
             cp_k_neighbors_frac: float = 0.1,
             cp_n_candidates: int = 20,
             cp_k_neighbors_min: int = 1,
             seed: int = 42,
             cp_alpha: float = 0.0,
             cp_anchor_source: str = "train",
             cp_n_anchors: Optional[int] = None,
             cp_anchor_method: str = "kmedoids",
             cp_anchors: Optional[np.ndarray] = None,
             cp_distance: str = "full",
             cp_dist_tol: float = 1e-3,
             cp_trace_path: Optional[str] = None,
             cp_robustify_objective: bool = True,
             cp_eval_mode: str = "global",
             cp_nearest_distance: str = "context",
             ) -> tuple:
    """Cutting Planes for robust constraint learning (one driver, auto-selected oracle).

    The loop is identical across scenarios -- train nominal, build the master,
    solve for the optimal solution(s) ``x*``, separate, add cuts, terminate. The
    separation strategy is chosen automatically from the problem shape:

    - **basic** -- a single global LP with a single learned constraint
      (synthetic): plain worst-case localized-bootstrap separation at ``x*``,
      cut whatever violates, stop when nothing does. No ``cp_alpha``.
    - **coherent** -- multiple constraints, multiple ``x*``, and/or a learned
      objective (e.g. gastric): one *shared* bootstrap relabeling drives all
      constraints (and optionally the epigraph objective), and the worst scenario
      -- ranked by **normalized average distance** -- is cut.

    ``cp_eval_mode``:
    - ``"global"`` (default): one shared master; cuts from all anchors.
    - ``"per_anchor_nearest"``: train one CP master per anchor; at prescribe time
      pick the nearest training anchor's master (see ``CPMultiAnchorResult``).

    ``cp_robustify_objective``: when ``False``, embed OS directly (no epigraph
    cuts); only constraint feasibility is robustified.
    """
    model_params = model_params or {}
    loop_kwargs = dict(
        model_type=model_type,
        model_params=model_params,
        rho=rho,
        max_iterations=max_iterations,
        cp_k_neighbors_frac=cp_k_neighbors_frac,
        cp_n_candidates=cp_n_candidates,
        cp_k_neighbors_min=cp_k_neighbors_min,
        seed=seed,
        cp_alpha=cp_alpha,
        cp_distance=cp_distance,
        cp_dist_tol=cp_dist_tol,
        cp_robustify_objective=cp_robustify_objective,
    )

    if cp_eval_mode == "per_anchor_nearest":
        anchor_rows = _get_anchor_rows(
            instance, cp_anchors, cp_anchor_source,
            cp_n_anchors, cp_anchor_method, seed,
        )
        if anchor_rows is None:
            anchors = [None]
            anchor_rows = np.empty((0, instance.n_features))
        else:
            anchors = [anchor_rows[i] for i in range(anchor_rows.shape[0])]

        print(
            f"    [cp] eval_mode=per_anchor_nearest; "
            f"{len(anchors)} anchor-specific masters "
            f"(nearest_distance={cp_nearest_distance})",
            flush=True,
        )
        total_start = time.time()
        anchor_results: List[SolutionResult] = []
        combined = CPHistory()
        statuses = []
        for k, anchor in enumerate(anchors):
            print(f"    [cp] anchor {k + 1}/{len(anchors)}...", flush=True)
            trace = cp_trace_path if k == len(anchors) - 1 else None
            result, hist = _run_cp_loop(
                instance, anchors=[anchor], cp_trace_path=trace, **loop_kwargs,
            )
            anchor_results.append(result)
            statuses.append(result.status)
            combined.iterations = max(combined.iterations, hist.iterations or 0)
            combined.violations.extend(hist.violations)
            combined.objectives.extend(hist.objectives)

        aggregate_status = "optimal" if all(s == "optimal" for s in statuses) else statuses[-1]
        multi = CPMultiAnchorResult(
            anchor_rows=anchor_rows,
            anchor_results=anchor_results,
            status=aggregate_status,
            solve_time=time.time() - total_start,
            models_embedded=sum(r.models_embedded for r in anchor_results),
            nearest_distance_feature_indices=_resolve_nearest_distance(
                cp_nearest_distance, instance,
            ),
            iterations=combined.iterations,
        )
        return multi, combined

    print("    [cp] eval_mode=global", flush=True)
    anchor_rows = _get_anchor_rows(
        instance, cp_anchors, cp_anchor_source,
        cp_n_anchors, cp_anchor_method, seed,
    )
    if anchor_rows is None:
        anchors = [None]
    else:
        anchors = [anchor_rows[i] for i in range(anchor_rows.shape[0])]

    return _run_cp_loop(
        instance, anchors=anchors, cp_trace_path=cp_trace_path, **loop_kwargs,
    )

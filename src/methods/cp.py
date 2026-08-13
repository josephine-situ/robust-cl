"""
Cutting Planes method for robust constraint learning.

One driver (:func:`solve_cp`) handles every scenario and **auto-selects** the
separation strategy from the problem shape:

- **basic** -- non-contextual synthetic only: a single global LP with a single
  learned constraint; plain worst-case localized-bootstrap separation at ``x*``.
- **coherent** -- contextual problems (gastric), multiple constraints, multiple
  optimal solutions ``x*``, or a learned objective: one *shared* bootstrap
  relabeling drives every constraint (and the objective) jointly and the single
  worst scenario -- ranked by **normalized average distance** (mean relative
  exceedance over all ``(x*, outcome)`` cells, range 0–1) -- is cut. We stop
  when that distance falls within ``cp_dist_tol`` or no scenario can be added
  without pushing more than ``cp_alpha`` of the ``x*`` infeasible. With multiple
  ``x*`` the coverage cap ``cp_alpha`` bounds the fraction allowed to go
  infeasible (feasibility only); with a single ``x*`` there is no cap.

The objective uses an **epigraph** reformulation ``min c'x + t``, ``t >= sum of
learned objective terms``, so a learned objective is robustified by the same
worst-case cuts (each raises ``t``). Only the coherent strategy robustifies it.

Both reuse the same scaffolding: train nominal -> build master -> solve for the
optimal solution(s) ``x*`` -> separate -> add cuts -> terminate.
"""
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

    def __init__(self, instance: ProblemInstance, obj_terms: list, rho: float = 0.0,
                 mip_gap: float = 1e-4):
        self.instance = instance
        self.d = instance.n_features
        self.rho = rho
        self.opt = gp.Model("cp_incremental_master")
        self.opt.Params.OutputFlag = 0
        # Gap for the CUT LOOP. 0.01 (1%) was far too loose on gastric: the
        # objective is ~10, so 1% is ~0.1, while the scenario distances being
        # separated are ~0.007 -- cuts an order of magnitude below the solver's own
        # stopping tolerance, so it returned the SAME incumbent for different cut
        # sets and x* stopped moving. Synthetic never hit this (objective ~1.2,
        # distances ~0.1, so cuts sit above the gap). Matches the final solve and
        # the prescribe-time solve, so cuts are generated at the same optimality
        # the prescriptions are made at.
        self.opt.Params.MIPGap = mip_gap
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

        # anchor index -> its last known x*, used as a MIP start when that same
        # anchor is re-solved. Each anchor pins its own context (_fix_anchor_context
        # rewrites variable bounds), so an incumbent is only reusable for the anchor
        # that produced it -- which is why solving them in a loop otherwise starts
        # cold every time. A cut targets one worst scenario, so most anchors' previous
        # solutions survive it and hand Gurobi an immediate incumbent.
        self.anchor_starts = {}

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
        """No-deterioration cut ``obj_expr >= obj_val``: forbid the master from
        returning to a better objective than it has already attained.

        Scenario cuts only tighten the feasible region, so a minimum can only
        rise; this is therefore REDUNDANT while nothing is removed, and bites only
        once cuts can be pruned or evicted -- which is exactly when a previous
        ``x*`` can recur and CP can cycle.

        It constrains ``x`` in both problem settings, despite appearances:
        ``obj_expr`` is ``c'x`` on synthetic (no learned objective) and
        ``base_cost + sum(obj_terms(x))`` = ``-OS(x)`` on gastric under
        ``robustify_objective: false`` (``_build_master_with_nominal``). It would
        be vacuous only under ``robustify_objective: true``, where ``obj_expr``
        becomes the free epigraph variable ``t_obj`` and bounding it merely raises
        ``t_obj`` without moving ``x``. Callers gate on ``objective_monotone``;
        the old ``t_obj is not None`` guard made this dead code on both problems.
        """
        self.opt.addConstr(self.obj_expr >= obj_val, name=f"obj_bound_{iteration}")
        self.opt.update()

    def apply_start(self, key):
        """Warm-start from this anchor's previous ``x*``, if we have one.

        Gurobi silently drops a start that violates current bounds or cuts, so a
        stale one costs nothing; a surviving one skips the search for a first
        incumbent and leaves only the optimality proof.
        """
        vals = self.anchor_starts.get(key)
        if vals is None:
            return
        for var, v in zip(self.x, vals):
            var.Start = v

    def record_start(self, key, x_vals):
        self.anchor_starts[key] = [float(v) for v in x_vals]

    def solve(self, start_key=None):
        if start_key is not None:
            self.apply_start(start_key)
        self.opt.optimize()
        if self.opt.Status != GRB.OPTIMAL:
            return None, np.inf
        x_vals = np.array([v.X for v in self.x])
        if start_key is not None:
            self.record_start(start_key, x_vals)
        return x_vals, self.opt.ObjVal


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
    results = [_evaluate_real_candidate(args) for args in args_list]

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
                               robustify_objective: bool = True,
                               mip_gap: float = 1e-4):
    """Train nominal models, build the master MIP, embed objective + initial cuts.

    Returns ``(master, model_config_map)`` where ``model_config_map`` maps each
    ``model_data`` id to its resolved ``(model_type, model_params)`` for later
    bootstrap retraining during separation.
    """
    trained_constraints = _train_nominal_with_configs(instance, model_type, model_params)
    master = IncrementalMaster(instance, [], rho=rho, mip_gap=mip_gap)

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

    Returns ``(feasible, p_infeas[, min_slack], n_bound_blocked)``.
    ``n_bound_blocked`` counts anchors that were infeasible ONLY because of the
    no-deterioration bound -- attributed by one re-solve without it, purely to
    report the cause. The anchor is still counted infeasible: we flag rather than
    silently relax, so a bound that is doing more harm than good is visible.
    """
    feasible = []
    n_infeas = 0
    n_bound_blocked = 0
    min_slack = {} if collect_slack else None
    for a_idx, anchor in enumerate(anchors):
        _fix_anchor_context(master, instance, anchor)

        tmp = None
        try:
            if obj_bounds is not None and a_idx in obj_bounds:
                tmp = master.opt.addConstr(
                    master.obj_expr >= obj_bounds[a_idx], name=f"obj_bnd_{a_idx}"
                )
                master.opt.update()

            x_q, obj_q = master.solve(start_key=a_idx)

            if collect_slack and x_q is not None:
                for s, constr in enumerate(master.scenario_constrs):
                    if constr is not None:
                        sl = constr.Slack
                        if s not in min_slack or sl < min_slack[s]:
                            min_slack[s] = sl
        finally:
            # Must not survive into the prescribe phase: the master object is
            # handed back and re-solved per test row, so a leaked bound derived
            # from TRAINING anchors would silently distort every prescription.
            if tmp is not None:
                master.opt.remove(tmp)
                master.opt.update()

        if x_q is None:
            n_infeas += 1
            if tmp is not None:
                # Attribute the cause: was it the bound, or the cuts?
                x_free, _ = master.solve(start_key=a_idx)
                if x_free is not None:
                    n_bound_blocked += 1
        else:
            feasible.append((a_idx, x_q, obj_q))

    p_infeas = n_infeas / len(anchors)
    if collect_slack:
        return feasible, p_infeas, min_slack, n_bound_blocked
    return feasible, p_infeas, n_bound_blocked


def _protected_still_feasible(master, instance, anchors, protected, order=None):
    """Do all ``protected`` anchors remain feasible after the cuts just added?

    ``protected`` is the **fixed** set of anchors the NOMINAL fit could serve,
    measured once before any cut. Pinning the reference there, rather than
    recomputing it each iteration, matters for three reasons:

    - It is the meaningful requirement: "CP may not break an anchor the nominal
      model already served". A per-iteration baseline is a ratchet -- once one
      anchor drops out the cap loosens, legitimising the next drop.
    - It is **set-wise**, not count-wise. A count would let CP trade one
      patient's feasibility for another's, which is arbitrary and unstable.
    - It makes rejections **permanent**: the cap no longer moves and the master
      only tightens, so a cut that breaks a protected anchor once breaks it
      always. That is what lets the caller cache rejected scenarios instead of
      re-deriving them every iteration.

    ``order`` optionally sorts the anchors most-likely-to-fail first (anchors
    that achieved a higher objective sit closer to the boundary), maximising the
    chance of an early exit. Returns ``(n_broken, fits)``.
    """
    seq = [a for a in (order or sorted(protected)) if a in protected]
    for a_idx in seq:
        _fix_anchor_context(master, instance, anchors[a_idx])
        # Warm-started: this check is dominated by anchors that come back FEASIBLE
        # (all of them for an accepted candidate, and every one before the breaking
        # anchor in a rejection). A surviving incumbent removes the search for a
        # first solution. It cannot help the decisive infeasible solve -- proving no
        # solution exists means exhausting the search regardless.
        if master.solve(start_key=a_idx)[0] is None:
            return 1, False        # one broken protected anchor is enough
    return 0, True


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
              cp_trace_path, last_x=None, last_obj=np.inf, anchors=None,
              incumbent_x=None, incumbent_obj=np.inf, strategy=None):
    """Restore context-free bounds, do a tight final solve, persist trace.

    ``incumbent_x`` (best feasible iterate) is supplied only by the basic path
    when it exits WITHOUT converging: the accumulated cuts can leave the final
    master over-constrained/degenerate, so we return the best feasible decision
    actually found during iteration rather than that unreliable final solve."""
    d = instance.n_features
    print("Re-solving with default MIP gap...")
    _restore_context_bounds(master, ctx_bounds)
    if anchors and len(anchors) == 1 and anchors[0] is not None:
        _fix_anchor_context(master, instance, anchors[0])
    master.opt.Params.MIPGap = 1e-4
    x_final, obj_final = master.solve()
    _restore_context_bounds(master, ctx_bounds)
    if incumbent_x is not None:
        # Non-converged basic path: prefer the best feasible incumbent over the
        # (possibly infeasible/degenerate) over-cut final master solve.
        x_opt, obj_value = incumbent_x, incumbent_obj
    elif x_final is not None:
        x_opt, obj_value = x_final, obj_final
    elif last_x is not None:
        x_opt, obj_value = last_x, last_obj
    else:
        x_opt, obj_value = np.zeros(d), np.inf

    # The CP-vs-wrapper size claim, measured: CP embeds one cut per iteration over
    # a bank the wrapper could not embed in full, and evicts/prunes as it goes.
    n_active = sum(1 for c in master.scenario_constrs if c is not None)
    print(
        f"    [cp] final master: {n_active} active cuts, "
        f"{master.opt.NumVars} vars / {master.opt.NumConstrs} constrs",
        flush=True,
    )
    if strategy is not None:
        _report_cp_diagnostics(strategy, status)
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
    # Fixed scenario bank drawn from the shared uncertainty set D
    # (src/methods/uncertainty.ScenarioBank). When present, separation is a lookup
    # over pre-trained models instead of a per-iteration resample-and-retrain --
    # which is what makes the worst-violation trace monotone and tau meaningful.
    # None restores the legacy localized-bootstrap path.
    bank: Optional[object] = None
    d0_quantile: float = 0.9
    # Smallest normalized distance worth separating. The master is solved to
    # `mip_gap` relative optimality, so cuts whose effect falls below that leave
    # x* unmoved -- separating there burns iterations for nothing. Expressed in
    # NORMALIZED distance units: the coherent path's distances are already
    # normalized so it applies this as-is, while the basic path keeps its
    # violations raw and converts this floor into constraint units first.
    resolution_floor: float = 0.0


def _scenario_models(inst: ProblemInstance, model_map: dict, obj_specs: list):
    """Split one scenario's ``{md_id -> model}`` into per-constraint model lists and
    objective terms, in the shape the scoring loop expects. Shared by the bank and
    the legacy bootstrap paths so they are scored identically."""
    per_constraint = {}
    for c_idx, constraint in enumerate(inst.constraints):
        if not _is_constraint_constraint(constraint):
            continue
        models = [
            (md.weight, model_map[id(md)])
            for md in constraint.models_data if md.obj_weight == 0.0
        ]
        per_constraint[c_idx] = (models, constraint.rhs)
    obj_models = [(w, model_map[id(md)]) for w, md in obj_specs]
    return per_constraint, obj_models


def _report_cp_diagnostics(strategy, status: str) -> None:
    """Run-level health of the cut loop: cycling, objective regressions, and how
    often anchors were infeasible. Reported, never acted on -- a run that needs a
    different policy should say so in the log rather than quietly change itself."""
    solves = getattr(strategy, "n_anchor_solves", 0)
    if not solves:
        return
    infeas = strategy.n_anchor_infeasible
    blocked = strategy.n_bound_blocked
    print(
        f"    [cp] diagnostics: status={status}, "
        f"objective regressions={strategy.n_obj_regressions}, "
        f"permanent rejections={getattr(strategy, 'n_rejections', 0)}, "
        f"anchor solves infeasible={infeas}/{solves} ({100 * infeas / solves:.1f}%), "
        f"bound-blocked={blocked}/{solves} ({100 * blocked / solves:.1f}%)",
        flush=True,
    )
    if blocked > 0.25 * solves:
        print(
            "    [cp] WARNING: the no-deterioration bound blocked >25% of anchor "
            "solves -- it is over-constraining. Consider objective_monotone=false.",
            flush=True,
        )
    if status == "cycle_detected":
        print(
            "    [cp] WARNING: cut loop cycled. tau cannot bind in a cycle, so any "
            "tau grid over this configuration is uninformative.",
            flush=True,
        )


def _resolve_d0(distances, quantile: float) -> float:
    """``d0`` = a high quantile of the iteration-0 scenario distances, not their max.

    The max over B draws grows with B, so a max-based ``d0`` would stop tau from
    transferring across bank sizes -- and CP and the wrapper run at different B by
    design. A quantile is stable in B and less seed-noisy than a max.

    The quantile is over the **draws**. Callers pass one distance per draw (the
    coherent path has already collapsed anchors x outcomes by a mean; the basic
    path has no anchor dimension), so anchors never enter this order statistic.

    Note the asymmetry this creates, which is deliberate and documented at
    ``config.yaml``'s ``d0_quantile``: both callers compare the resulting
    tolerance against the **max** over the bank, not against this quantile. So
    ``tau=1`` does not reduce to nominal -- iteration 0 fails its own stopping
    test, since max > q0.9 whenever the top decile is non-degenerate.
    """
    d = np.asarray([v for v in distances if np.isfinite(v)], dtype=float)
    if d.size == 0:
        return 0.0
    # The basic path scores signed violations, so a bank where nothing violates
    # yields a negative quantile. A negative distance is not a distance: clamp to
    # 0 so the tolerance collapses to its floor rather than going negative.
    return max(0.0, float(np.quantile(d, float(quantile))))


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

    def __init__(self, k_neighbors_frac, n_candidates, k_neighbors_min=1,
                 dist_tol_rel=None, prune_slack_cuts=True, objective_monotone=False):
        self.k_neighbors_frac = k_neighbors_frac
        self.n_candidates = n_candidates
        self.k_neighbors_min = k_neighbors_min
        # Same two policies the coherent path takes, so the methods differ only
        # where the PROBLEMS differ (one context vs ten, one constraint vs five).
        self.prune_slack_cuts = prune_slack_cuts
        self.objective_monotone = objective_monotone
        # Same diagnostics as the coherent path (one "anchor" here: the single x*).
        self.prev_min_obj = None
        self.n_obj_regressions = 0
        self.n_bound_blocked = 0
        self.n_anchor_solves = 0
        self.n_anchor_infeasible = 0
        self.seen_states: dict = {}
        # Same problem-agnostic knob as the coherent path: tolerance = tau * d0,
        # with d0 = the iteration-0 worst violation. Without it this path cut every
        # violation > 1e-6, i.e. it had no robustness lever at all.
        self.dist_tol_rel = dist_tol_rel
        self._tol = None
        self.cut_draws: dict = {}   # scenario id -> bank draw (see _CoherentSeparation)

    def step(self, env: _SepEnv, iteration: int) -> _StepResult:
        iter_start = time.time()
        inst, master = env.instance, env.master

        if len(env.anchors) == 1 and env.anchors[0] is not None:
            _fix_anchor_context(master, inst, env.anchors[0])

        x_star, obj_star = master.solve()
        self.n_anchor_solves += 1
        if x_star is None:
            self.n_anchor_infeasible += 1
            return _StepResult(stop=True, status="infeasible")

        if env.bank is not None:
            return self._step_bank(env, iteration, x_star, obj_star, iter_start)

        sep_cache = {}
        max_violation = -np.inf
        candidate_cuts = []  # (violation, c_idx, models, rhs) -- filtered by _tol below
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
            candidate_cuts.append((violation, c_idx, worst_case_models, constraint.rhs))

        # Resolve the effective tolerance once from this problem's own iteration-0
        # worst violation (d0), then cut only what exceeds it. tau->0 approaches the
        # legacy "cut everything > 1e-6".
        #
        # This is the legacy `scenario_source: "bootstrap"` path, and it is the ONE
        # place where tau=1 really does stop at iteration 0 (~nominal): d0 is the
        # max here, the same statistic the filter tests, and the filter is strict.
        # No quantile is taken because there is none available -- candidate_cuts
        # holds one entry per learned constraint (synthetic has exactly one), and
        # localized_bootstrap_separation has already collapsed its candidate draws
        # to the argmax internally, so the per-draw spread never reaches us.
        # The default "noise" path (_step_bank) uses the quantile and does NOT
        # share this tau=1 behaviour.
        if self._tol is None:
            if self.dist_tol_rel is not None and np.isfinite(max_violation):
                self._tol = max(float(self.dist_tol_rel) * float(max_violation), 1e-6)
                print(
                    f"    [cp] d0={max_violation:.4f} (iter-0 worst violation); "
                    f"tau={self.dist_tol_rel:g} -> tol={self._tol:.6f}",
                    flush=True,
                )
            else:
                self._tol = 1e-6
        scenarios_to_add = [(c, m, r) for (v, c, m, r) in candidate_cuts if v > self._tol]

        print(
            f"Iter {iteration}: Obj={obj_star:.4f} "
            f"Max Violation={max_violation:.4f} Time={time.time()-iter_start:.2f}s"
        )

        # Single global LP: prune slack scenarios and add the no-deterioration
        # objective cut. Both are off by default -- see _CoherentSeparation.step
        # and IncrementalMaster.add_objective_cut for why.
        if self.prune_slack_cuts and iteration > 0:
            dynamic_slack = max(0.1, max_violation)
            pruned_count, total_active = prune_inactive_scenarios(
                master, slack_threshold=dynamic_slack
            )
            if pruned_count > 0:
                print(
                    f"Iter {iteration}: Pruned {pruned_count}/{total_active} "
                    f"inactive scenarios"
                )
        if self.objective_monotone:
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

    def _step_bank(self, env, iteration, x_star, obj_star, iter_start) -> _StepResult:
        """One separation step over the fixed scenario bank.

        Scores every draw not currently embedded as an active cut at ``x*``, and
        cuts the single worst. The bank is a separation *pool*, not an embedded
        ensemble: only the argmax becomes a cut, which is why CP's master stays
        small at bank sizes the wrapper could never embed.
        """
        inst, master = env.instance, env.master
        xs2d = np.atleast_2d(x_star)

        # Same two diagnostics the coherent path runs, on its single x*.
        if self.prev_min_obj is not None and obj_star < self.prev_min_obj - 1e-9:
            self.n_obj_regressions += 1
            print(
                f"Iter {iteration}: objective REGRESSED "
                f"{self.prev_min_obj:.6f} -> {obj_star:.6f} (a cut was removed).",
                flush=True,
            )
        self.prev_min_obj = obj_star
        state = frozenset(
            s for s, c in enumerate(master.scenario_constrs) if c is not None
        )
        if state in self.seen_states:
            first = self.seen_states[state]
            print(
                f"Iter {iteration}: CYCLE DETECTED -- active cut set repeats "
                f"iteration {first} (period {iteration - first}); stopping.",
                flush=True,
            )
            return _StepResult(stop=True, status="cycle_detected",
                               obj=obj_star, x=x_star.copy())
        self.seen_states[state] = iteration

        active = {
            b for sid, b in self.cut_draws.items()
            if sid < len(master.scenario_constrs)
            and master.scenario_constrs[sid] is not None
        }
        draws = [b for b in range(len(env.bank)) if b not in active]

        best = None          # (violation, c_idx, models, rhs, draw)
        all_viol = []
        # Raw-unit equivalent of one normalized distance unit. The coherent path
        # divides each exceedance by `max(1.0, abs(rhs))`; this path keeps its
        # violations RAW (so d0, the logged "Max Violation", and the history stay in
        # constraint units), so the same factor is applied to the FLOOR instead --
        # see the tolerance block below. Single-constraint path (`use_basic`), so the
        # max over scanned constraints is just that constraint's scale.
        tol_scale = 1.0
        for b in draws:
            model_map = env.bank.models_for(b)
            for c_idx, constraint in enumerate(inst.constraints):
                if not _is_constraint_constraint(constraint):
                    continue
                models = [(md.weight, model_map[id(md)])
                          for md in constraint.models_data if md.obj_weight == 0.0]
                val = sum(w * th.predict(xs2d)[0] for w, th in models)
                violation = val - constraint.rhs
                all_viol.append(violation)
                tol_scale = max(tol_scale, abs(float(constraint.rhs)))
                if best is None or violation > best[0]:
                    best = (violation, c_idx, models, constraint.rhs, b)

        max_violation = best[0] if best is not None else -np.inf

        # tolerance = tau * d0, with d0 a high QUANTILE of the iteration-0 draws --
        # stable in bank size, unlike the max. The stopping statistic below is the
        # MAX, though, so tau=1 does not stop at iteration 0 on this path (unlike
        # the legacy path above); it separates the bank's worst ~decile.
        #
        # `all_viol` holds RAW SIGNED violations (val - rhs, no normalization),
        # unlike the coherent path's normalized anchor-averaged distances. So
        # `resolution_floor` -- documented in _CPEnv as normalized units, and set
        # from cp_mip_gap -- is converted into this path's raw units by `tol_scale`
        # rather than compared across units. Inert at the default synthetic config
        # (rhs = 0.5 * n_features = 1.0, so tol_scale = 1.0), and live as soon as
        # data.synthetic.n_features > 2 makes rhs exceed 1.
        if self._tol is None:
            if self.dist_tol_rel is not None and all_viol:
                d0 = _resolve_d0(all_viol, env.d0_quantile)
                raw = float(self.dist_tol_rel) * d0
                # Never separate below what the master is solved to: the solver
                # returns any incumbent within `mip_gap` of optimal, so a cut whose
                # effect is smaller leaves x* unmoved. 1e-6 is the absolute backstop
                # (the legacy "cut everything > 1e-6" semantics).
                floor = max(env.resolution_floor * tol_scale, 1e-6)
                self._tol = max(raw, floor)
                # Report a floored tolerance explicitly, as the coherent path does:
                # it means every tau below that point gives the SAME run, so a tau
                # grid must be spanned above it to be informative.
                note = "" if raw >= floor else (
                    f"  [FLOORED at solver resolution {floor:.6f}; "
                    f"tau<={raw / d0 if d0 else 0:.3g} all give this run]"
                )
                print(
                    f"    [cp] d0={d0:.4f} (q{env.d0_quantile:g} of {len(all_viol)} "
                    f"iter-0 scenario violations; max={max_violation:.4f}); "
                    f"tau={self.dist_tol_rel:g} -> tol={self._tol:.6f}{note}",
                    flush=True,
                )
            else:
                self._tol = 1e-6

        print(
            f"Iter {iteration}: Obj={obj_star:.4f} Max Violation={max_violation:.4f} "
            f"scanned={len(draws)}/{len(env.bank)} Time={time.time()-iter_start:.2f}s",
            flush=True,
        )

        if self.prune_slack_cuts and iteration > 0:
            pruned_count, total_active = prune_inactive_scenarios(
                master, slack_threshold=max(0.1, max_violation)
            )
            if pruned_count > 0:
                print(f"Iter {iteration}: Pruned {pruned_count}/{total_active} "
                      f"inactive scenarios")
        if self.objective_monotone:
            master.add_objective_cut(obj_star, iteration)

        add_cut = best is not None and max_violation > self._tol
        if add_cut:
            _viol, c_idx, models, rhs, draw = best
            master.add_scenario(c_idx, models, rhs, rho=env.rho)
            self.cut_draws[master.n_models - 1] = draw

        return _StepResult(
            stop=not add_cut,
            status="optimal" if not add_cut else "running",
            obj=obj_star,
            x=x_star.copy(),
            violation=max_violation if np.isfinite(max_violation) else np.inf,
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
                 k_neighbors_min=1, cut_eviction="reject", base_scenario_ids=None,
                 dist_tol_rel=None, prune_slack_cuts=True, objective_monotone=False,
                 cut_whole_scenario=True):
        self.k_neighbors_frac = k_neighbors_frac
        self.n_scenarios = n_scenarios
        self.alpha = alpha               # coverage cap: max fraction of x* infeasible
        self.single_point = single_point
        self.dist_tol = dist_tol         # stop once worst normalized distance <= this
        # Problem-agnostic knob: when set, the stopping tolerance is tau * d0, where
        # d0 is a QUANTILE (env.d0_quantile) of THIS problem's iteration-0 distances,
        # measured before any cut. tau->0 cuts maximally. Absolute dist_tol values
        # don't transfer across datasets because d0 varies with the data's noise
        # scale; tau does -- but only as a RATIO to each problem's own d0, since the
        # basic and coherent paths measure distance in different units. Resolved
        # once, at iteration 0.
        #
        # tau=1 does NOT stop immediately here: step() compares the MAX distance
        # over the bank against this q0.9-derived tolerance, so iteration 0 fails
        # its own test and tau=1 separates the bank's worst ~decile. Kept as is;
        # see config.yaml's d0_quantile for the full statement and the alternatives.
        self.dist_tol_rel = dist_tol_rel
        self._tol = None                 # effective absolute tolerance (set at iter 0)
        self.k_neighbors_min = k_neighbors_min
        # "evict_slack": on infeasibility keep the (relevant) new cut and evict the
        # most-slack ACTIVE non-base scenario until feasible; "reject": drop the new
        # cut (legacy). Under evict_slack the working set is non-monotone, so the
        # no-deterioration obj bound is disabled.
        self.cut_eviction = cut_eviction
        # Slack-pruning is a cycling hazard under a fixed bank (see step()); the
        # driver turns it off there. Eviction is NOT optional -- it is forced by MIP
        # infeasibility, and an evicted scenario stays re-separable on purpose.
        self.prune_slack_cuts = prune_slack_cuts
        # Per-anchor no-deterioration bounds (the coherent analogue of the basic
        # path's single global cut -- ten contexts, so ten bounds, each imposed
        # transiently while its own anchor is solved). Gated by the SAME flag.
        self.objective_monotone = objective_monotone
        # Enforce every constraint of an accepted scenario, not just the breached
        # subset -- see step(). This is what makes per-scenario exclusion sound.
        self.cut_whole_scenario = cut_whole_scenario
        self.base_scenario_ids = set(base_scenario_ids or ())  # nominal cuts, never evicted
        self.obj_bound = {}          # a_idx -> best (max) objective seen so far
        self.prev_max_exceed = 0.0   # scale for the dynamic pruning threshold (raw)
        self.last_added_ids: List[int] = []  # scenario ids from the last accepted cut
        # scenario id -> bank draw it came from. The exclusion set is derived from
        # the entries whose cut is still ACTIVE, never from "ever cut": eviction and
        # pruning retire cuts, and a retired scenario must stay re-separable.
        # (With nothing removed, the two definitions coincide.)
        self.cut_draws: dict = {}
        # Diagnostics -- counted, reported, never acted on automatically.
        self.prev_min_obj = None
        self.n_obj_regressions = 0
        self.n_bound_blocked = 0
        self.n_anchor_solves = 0
        self.n_anchor_infeasible = 0
        self.seen_states: dict = {}   # active-cut-set hash -> iteration first seen
        # Fixed at the nominal-feasible anchors on the first step, never recomputed.
        self.protected_anchors = None
        # Bank draws whose cut broke a protected anchor. Permanent -- see
        # _protected_still_feasible. Disabled under eviction (non-monotone master).
        self.rejected_draws: set = set()
        self.n_rejections = 0

    def _evict_to_fit(self, master, inst, anchors, added_ids, min_slack, baseline_infeas):
        """Evict the most-slack ACTIVE non-base scenario (keeping the just-added,
        current-x*-relevant cut and the nominal base) until the infeasible-anchor
        count is back to ``baseline_infeas`` (no NEW infeasibility). Returns
        ``(fits, evicted_ids)``. Uses full anchor re-solves because eviction relaxes
        the master (the monotone early-exit of ``_p_infeas_after_cuts`` is invalid
        once cuts are removed)."""
        protected = self.base_scenario_ids | set(added_ids)
        n = len(anchors)
        evicted: List[int] = []
        while True:
            _, p_infeas, _ = _solve_all_anchors(master, inst, anchors)
            if p_infeas * n <= baseline_infeas + 1e-9:
                return True, evicted
            cand = [s for s, c in enumerate(master.scenario_constrs)
                    if c is not None and s not in protected]
            if not cand:
                return False, evicted           # only nominal base left, still worse
            victim = max(cand, key=lambda s: min_slack.get(s, np.inf))  # most stale
            master.remove_scenario(victim)
            master.opt.update()
            evicted.append(victim)
            print(f"      [cp] evicted stale scenario {victim} "
                  f"(slack={min_slack.get(victim, float('inf')):.3f}) to admit new cut",
                  flush=True)

    def step(self, env: _SepEnv, iteration: int) -> _StepResult:
        iter_start = time.time()
        inst, master = env.instance, env.master

        evicting = self.cut_eviction == "evict_slack"
        feasible, p_infeas, min_slack, n_bound_blocked = _solve_all_anchors(
            master, inst, env.anchors,
            obj_bounds=self.obj_bound if self.objective_monotone else None,
            collect_slack=True,
        )
        self.n_bound_blocked += n_bound_blocked
        self.n_anchor_solves += len(env.anchors)
        self.n_anchor_infeasible += len(env.anchors) - len(feasible)
        if n_bound_blocked:
            print(
                f"Iter {iteration}: {n_bound_blocked} anchor solve(s) infeasible "
                f"ONLY under the no-deterioration bound (bound-blocked).",
                flush=True,
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
        # cuts are added, so record its best objective for later solves.
        if self.objective_monotone:
            for a_idx, _, obj_q in feasible:
                if obj_q > self.obj_bound.get(a_idx, -np.inf):
                    self.obj_bound[a_idx] = obj_q

        # Objective-regression detector (always on, never changes behaviour).
        # Scenario cuts only tighten, so min_a v_a can fall ONLY if a cut was
        # removed -- which is the regression that lets a previous x* recur. This
        # is the cheap diagnostic for cycling, independent of any bound.
        m_now = min(obj_q for _, _, obj_q in feasible)
        if self.prev_min_obj is not None and m_now < self.prev_min_obj - 1e-9:
            self.n_obj_regressions += 1
            print(
                f"Iter {iteration}: objective REGRESSED "
                f"{self.prev_min_obj:.6f} -> {m_now:.6f} "
                f"(a cut was removed; this is what permits cycling).",
                flush=True,
            )
        self.prev_min_obj = m_now

        # Cycle detection: the master is fully described by its ACTIVE cut set, so
        # a repeated set means a repeated state and every later iteration replays
        # the same sequence. Terminate with a diagnostic instead of burning the
        # remaining iterations -- gastric ran an exact period-4 cycle to iteration
        # 19 this way, which left tau inert across its whole grid.
        state = frozenset(
            s for s, c in enumerate(master.scenario_constrs) if c is not None
        )
        if state in self.seen_states:
            first = self.seen_states[state]
            print(
                f"Iter {iteration}: CYCLE DETECTED -- active cut set repeats "
                f"iteration {first} (period {iteration - first}); stopping.",
                flush=True,
            )
            return _StepResult(
                stop=True, status="cycle_detected",
                obj=float(np.mean([o for (_, _, o) in feasible])),
            )
        self.seen_states[state] = iteration

        # Multi-anchor pruning: drop cuts that are slack at *every* x* (globally
        # inactive). Threshold scales with the previous iteration's worst cut, as
        # in the basic case; skipped on iteration 0.
        #
        # OFF by default under a fixed bank, because it makes CP cycle. Pruning a
        # slack cut returns its scenario to the eligible pool (exclusion follows
        # ACTIVE cuts); x* then drifts back, the same scenario is again the worst,
        # and it is re-cut -- gastric showed an exact period-4 cycle through
        # iteration 19, which left tau inert across its whole grid. The old
        # localized-bootstrap path hid this by redrawing scenarios every iteration,
        # so re-separating the identical scenario was measure-zero.
        #
        # Permanently excluding a pruned scenario would break the cycle but is
        # WORSE: the cut is still gone, so the master is still relaxed, and now a
        # later x* that violates that scenario can never be cut again. Keeping the
        # cut is what preserves robustness, and it also makes CP terminate by
        # construction -- a scenario whose cut stays in the master is satisfied at
        # every future x*, so the eligible set strictly shrinks (<= B iterations).
        #
        # Cost is bounded by max_iterations (one embed per iteration), not by B.
        if self.prune_slack_cuts and iteration > 0 and min_slack:
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

        # Scenario source. With a bank, the draws are FIXED: theta(delta_b) no
        # longer depends on x*, so the worst violation over the bank is monotone
        # across iterations. Redrawing every iteration (the legacy path below) is
        # what made the trace oscillate -- iteration k's sample was compared
        # against iteration 0's d0, with sampling noise the size of the signal.
        # Draws that are ALREADY an active cut are excluded; "already cut" would be
        # wrong, since eviction and pruning retire cuts and an evicted scenario
        # must remain re-separable.
        if env.bank is not None:
            active = {
                b for sid, b in self.cut_draws.items()
                if sid < len(master.scenario_constrs)
                and master.scenario_constrs[sid] is not None
            }
            # Skip permanently-rejected draws here too, not just at acceptance:
            # scoring them means retraining nothing but re-predicting at every
            # (anchor x outcome) cell for a scenario we already know we cannot use.
            skip = active | self.rejected_draws
            scenarios = [b for b in range(len(env.bank)) if b not in skip]
            pool_frac = float("nan")
        else:
            # Legacy: shared, decision-dependent localized pool -> B coherent scenarios.
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
        candidates = []   # (total_dist, raw_max_exceed, cuts, scenario_key)
        all_dists = []    # EVERY scanned scenario, including the non-violating ones,
                          # so d0's quantile describes the bank and not just its tail
        for scenario in scenarios:
            if env.bank is not None:
                model_map = env.bank.models_for(scenario)
            else:
                model_map = {}
                for constraint in inst.constraints:
                    for md in constraint.models_data:
                        m_type, m_params = env.model_config_map[id(md)]
                        model_map[id(md)] = retrain_on_bootstrap(
                            md.X_train, md.y_train, scenario, m_type, m_params
                        )
            per_constraint_models, obj_models = _scenario_models(
                inst, model_map, obj_specs
            )

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

            # Cut the WHOLE scenario, not just the constraints that happen to be
            # violated at the current x*. A scenario is one coherent relabeling of
            # the trial: if we accept it as plausible, every constraint under it
            # should hold, not the subset that was breached at this particular x*.
            #
            # Three things follow. It matches what the wrapper enforces per
            # replicate -- one shared indicator gated on ALL constraints holding
            # (wrapper.py) -- so CP at tau->0 and the wrapper at alpha=0 are the
            # same object on multi-constraint problems, not just on synthetic
            # where the distinction is vacuous. It makes per-scenario exclusion
            # CORRECT: a fully-cut scenario can never be violated again, whereas a
            # partially-cut one could breach a different constraint at a moved x*
            # and never be re-separated. And it makes the stopping rule a genuine
            # statement about the whole bank rather than the scannable remainder.
            #
            # Cost: ~5 embedded models per accepted scenario instead of ~1.4.
            # Set cut_whole_scenario=False for the old lazy-subset behaviour.
            if self.cut_whole_scenario:
                cuts = [(c_idx, models, rhs)
                        for c_idx, (models, rhs) in sorted(per_constraint_models.items())]
                # Only a genuinely violating scenario is a candidate -- otherwise
                # every draw would qualify with distance 0.
                if not violated_constraints and not obj_violated:
                    cuts = []
            else:
                cuts = [
                    (c_idx, per_constraint_models[c_idx][0], per_constraint_models[c_idx][1])
                    for c_idx in sorted(violated_constraints)
                ]
            # Robustify the objective with the same relabeling: raise the epigraph
            # floor t_obj to this scenario's (worse) objective.
            if obj_violated:
                cuts.append((obj_c_idx, obj_models, master.t_obj))
            all_dists.append(total_dist)
            if cuts:
                candidates.append((total_dist, raw_max_exceed, cuts, scenario))

        # Rank adversaries worst-first by normalized average distance.
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_dist = candidates[0][0] if candidates else 0.0
        # Scale next iteration's pruning threshold by the worst raw cut.
        self.prev_max_exceed = candidates[0][1] if candidates else 0.0

        # Resolve the effective tolerance once, from THIS problem's own iteration-0
        # distances, so a single tau grid transfers across datasets AND bank sizes.
        if self._tol is None:
            if self.dist_tol_rel is not None:
                d0 = _resolve_d0(all_dists, env.d0_quantile)
                raw = float(self.dist_tol_rel) * d0
                # Never separate below what the master is solved to. The solver
                # returns any incumbent within `mip_gap` of optimal, so a cut whose
                # effect is smaller than that leaves x* unmoved and the iteration
                # accomplishes nothing. Reported explicitly: a floored tolerance
                # means every tau below the floor gives the SAME run, so a tau grid
                # must be spanned above it to be informative.
                floor = env.resolution_floor
                self._tol = max(raw, floor)
                note = "" if raw >= floor else (
                    f"  [FLOORED at solver resolution {floor:.5f}; "
                    f"tau<={raw / d0 if d0 else 0:.3g} all give this run]"
                )
                print(
                    f"    [cp] d0={d0:.4f} (q{env.d0_quantile:g} of {len(all_dists)} "
                    f"iter-0 scenario distances; max={best_dist:.4f}); "
                    f"tau={self.dist_tol_rel:g} -> dist_tol={self._tol:.5f}{note}",
                    flush=True,
                )
            else:
                self._tol = self.dist_tol

        pool_note = "" if np.isnan(pool_frac) else f"PoolFrac={pool_frac:.3f} "
        print(
            f"Iter {iteration}: {'Obj' if self.single_point else 'AvgObj'}={last_obj:.4f} "
            f"WorstScenarioNormDist={best_dist:.4f} "
            f"p_infeas={p_infeas*100:.1f}% {pool_note}"
            f"scanned={len(scenarios)} Time={time.time()-iter_start:.2f}s",
            flush=True,
        )

        history_viol = best_dist

        # Terminate (1): even the worst sampled relabeling leaves the normalized
        # total distance within the allowance -- robust enough.
        if best_dist <= self._tol:
            return _StepResult(stop=True, status="optimal", obj=last_obj, violation=history_viol)

        # Single-lever CP: the protected set is FIXED at the anchors the nominal
        # fit could serve (captured at iteration 0, before any cut). A cut may not
        # break any of them. Anchors the nominal fit already failed are tolerated
        # -- nominal is not guaranteed feasible for every context.
        if self.protected_anchors is None:
            self.protected_anchors = frozenset(a_idx for a_idx, _, _ in feasible)
            print(
                f"    [cp] protected anchors fixed at the nominal-feasible set: "
                f"{len(self.protected_anchors)}/{len(env.anchors)}",
                flush=True,
            )
        # Anchors closest to the boundary first -> earliest possible exit.
        order = [a for a, _, _ in sorted(feasible, key=lambda t: t[2], reverse=True)]

        for cand_rank, (_dist, _mx, cuts, draw) in enumerate(candidates):
            # Rejections are permanent: the protected set is fixed and the master
            # only tightens, so a cut that broke a protected anchor once always
            # will. Skipping them is exact, and it is where the time goes --
            # rollbacks were growing 8 -> 44 per iteration, each costing a pass
            # over the anchors.
            if draw in self.rejected_draws:
                continue
            added_ids = []
            for c_idx, models, rhs in cuts:
                master.add_scenario(c_idx, models, rhs, rho=env.rho)
                added_ids.append(master.n_models - 1)
            n_broken, fits = _protected_still_feasible(
                master, inst, env.anchors, self.protected_anchors, order
            )
            p_infeas_after = (
                len(env.anchors) - len(self.protected_anchors) + n_broken
            ) / len(env.anchors)
            # evict_slack: keep the relevant new cut and evict stale scenarios to
            # restore feasibility instead of rejecting the new cut outright.
            evicted = []
            if not fits and evicting:
                fits, evicted = self._evict_to_fit(
                    master, inst, env.anchors, added_ids, min_slack,
                    len(env.anchors) - len(self.protected_anchors),
                )
            if fits:
                self.last_added_ids = added_ids
                if env.bank is not None:
                    # Remember which bank draw each cut came from, so the exclusion
                    # set can follow the ACTIVE cuts rather than the ever-cut ones.
                    for sid in added_ids:
                        self.cut_draws[sid] = draw
                n_active = sum(1 for c in master.scenario_constrs if c is not None)
                cap_note = (
                    "feasible" if self.single_point
                    else f"p_infeas={p_infeas_after*100:.1f}%"
                )
                evict_note = f", evicted {evicted}" if evicted else ""
                print(
                    f"Iter {iteration}: Added scenario(s) {added_ids} "
                    f"(candidate rank {cand_rank + 1}/{len(candidates)}, "
                    f"norm_dist={_dist:.4f}); {cap_note}{evict_note}; "
                    f"working-set={n_active} active cuts",
                    flush=True,
                )
                return _StepResult(stop=False, status="running", obj=last_obj,
                                   violation=history_viol)
            for s in reversed(added_ids):
                master.remove_scenario(s)
            master.opt.update()
            # Permanent under a fixed protected set + monotone master (see
            # _protected_still_feasible); never retried, so the per-iteration
            # rollback storm cannot recur.
            if not evicting:
                self.rejected_draws.add(draw)
                self.n_rejections += 1
            reject_msg = (
                "would become infeasible"
                if self.single_point
                else f"breaks a protected anchor "
                     f"(p_infeas would be {p_infeas_after*100:.1f}%)"
            )
            evict_note = f" (also evicted {evicted}, kept removed)" if evicted else ""
            print(
                f"Iter {iteration}: Rolled back scenario(s) {added_ids} "
                f"(candidate rank {cand_rank + 1}/{len(candidates)}, "
                f"norm_dist={_dist:.4f}); {reject_msg}{evict_note}",
                flush=True,
            )

        cap_label = "feasibility" if self.single_point else "coverage"
        print(
            f"Iter {iteration}: {cap_label} cap hit (no sampled scenario keeps "
            f"p_infeas <= alpha {self.alpha*100:.1f}%); stopping.",
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
                 cp_cut_eviction: str = "reject",
                 cp_dist_tol_rel: float = None,
                 cp_scenario_source: str = "noise",
                 cp_n_scenarios: int = 200,
                 cp_d0_quantile: float = 0.9,
                 cp_objective_monotone: bool = False,
                 cp_cut_whole_scenario: bool = True,
                 cp_mip_gap: float = 1e-4,
                 cp_uncertainty=None,
                 cp_bank=None,
                 cp_trace_path: Optional[str] = None) -> tuple[SolutionResult, CPHistory]:
    """Run the CP cut loop for a fixed anchor set (one or many contexts)."""
    total_start = time.time()
    history = CPHistory()

    print("    [cp] Training nominal models and building master MIP...", flush=True)
    master, model_config_map = _build_master_with_nominal(
        instance, model_type, model_params, rho,
        robustify_objective=cp_robustify_objective,
        mip_gap=cp_mip_gap,
    )
    # Scenarios added during the build are the NOMINAL (point-estimate) cuts; they are
    # the feasible base and must never be evicted.
    base_scenario_ids = set(range(master.n_models))
    if not instance.context_var_indices:
        ctx_bounds = {}
    else:
        ctx_bounds = {
            c: (master.x[c].lb, master.x[c].ub)
            for c in instance.context_var_indices
        }

    distance_feature_indices = _resolve_distance(cp_distance, instance)

    # The bank is independent of tau, so a caller sweeping the tau grid should build
    # it ONCE and pass it in (cp_bank); rebuilding per knob multiplies stage-1 CV
    # cost by the grid size for no reason.
    bank = cp_bank
    if bank is None and cp_scenario_source == "noise":
        from src.methods.uncertainty import ScenarioBank, UncertaintySet
        uset = cp_uncertainty if cp_uncertainty is not None else UncertaintySet()
        bank = ScenarioBank(
            instance, model_config_map, uset,
            n_scenarios=cp_n_scenarios, seed=seed,
        )

    env = _SepEnv(
        instance=instance, master=master, anchors=anchors,
        model_config_map=model_config_map,
        distance_feature_indices=distance_feature_indices,
        rho=rho, seed=seed,
        k_neighbors_min=cp_k_neighbors_min,
        bank=bank, d0_quantile=cp_d0_quantile,
        resolution_floor=cp_mip_gap,
    )

    n_constraints = sum(1 for c in instance.constraints if _is_constraint_constraint(c))
    has_obj_models = any(
        md.obj_weight != 0.0 for c in instance.constraints for md in c.models_data
    )
    if not cp_robustify_objective:
        has_obj_models = False
    single_point = len(anchors) == 1
    contextual = bool(instance.context_var_indices)
    # Basic separation is for the non-contextual synthetic case only. Contextual
    # gastric (including dlt_only + per_anchor_nearest) uses coherent separation
    # so anchors are pinned via _solve_all_anchors and separation stays in-process.
    use_basic = (
        (n_constraints <= 1) and single_point and not has_obj_models
        and not contextual
    )

    # One retention policy for both problems: under a fixed bank nothing is
    # removed from the master, so the eligible set strictly shrinks and CP
    # terminates in at most B iterations on either problem. The methods then
    # differ only where the PROBLEMS differ (one context vs ten, one constraint
    # vs five), not in bookkeeping.
    keep_all = bank is not None
    if use_basic:
        strategy = _BasicSeparation(
            cp_k_neighbors_frac, cp_n_candidates, cp_k_neighbors_min,
            dist_tol_rel=cp_dist_tol_rel,
            prune_slack_cuts=not keep_all,
            objective_monotone=cp_objective_monotone,
        )
        mode = "basic"
    else:
        # With a bank, the scenario count IS the bank size, not cp_n_candidates.
        n_scen = len(bank) if bank is not None else cp_n_candidates
        strategy = _CoherentSeparation(
            cp_k_neighbors_frac, n_scen, cp_alpha,
            single_point, cp_dist_tol, cp_k_neighbors_min,
            cut_eviction=cp_cut_eviction, base_scenario_ids=base_scenario_ids,
            dist_tol_rel=cp_dist_tol_rel,
            prune_slack_cuts=not keep_all,
            objective_monotone=cp_objective_monotone,
            cut_whole_scenario=cp_cut_whole_scenario,
        )
        mode = "coherent (single x*)" if single_point else "coherent (multi x*)"

    n_solves = 1 if single_point else len(anchors)
    extra = f", alpha={cp_alpha}" if (not use_basic and not single_point) else ""
    obj_flag = "robustify_objective" if cp_robustify_objective else "nominal_objective"
    if bank is not None:
        src_note = (f"scenarios=noise bank (B={len(bank)}, "
                    f"coherent={bank.coherent}, d0_q={cp_d0_quantile:g}, "
                    f"keep_all_cuts={keep_all}, "
                    f"objective_monotone={cp_objective_monotone})")
    else:
        n_pool = resolve_neighbor_pool_size(
            len(instance.constraints[0].models_data[0].y_train),
            cp_k_neighbors_frac, cp_k_neighbors_min,
        )
        src_note = (f"scenarios=localized bootstrap, "
                    f"neighbor_pool>={cp_k_neighbors_min} (k={n_pool})")
    print(
        f"    [cp] separation={mode}; constraints={n_constraints}, "
        f"optimal solves={n_solves}, distance={cp_distance}, {obj_flag}, "
        f"{src_note}{extra}",
        flush=True,
    )

    last_x, last_obj = None, np.inf
    # Best feasible incumbent (basic path only): the iterate with the smallest
    # worst-case violation, used as a safeguard when the loop exhausts its
    # iterations without converging (the over-cut final master is unreliable).
    is_basic = isinstance(strategy, _BasicSeparation)
    best_x, best_obj, best_viol = None, np.inf, np.inf
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

        if (is_basic and res.x is not None and res.violation is not None
                and res.violation < best_viol):
            best_viol = res.violation
            best_x = res.x
            best_obj = res.obj if res.obj is not None else np.inf

        if res.stop:
            status = res.status
            break

    # Only fall back to the incumbent for a non-converged basic run; a converged
    # ("optimal") run's tight final solve is trustworthy, and the coherent path
    # (gastric) never sets an incumbent.
    incumbent_x = best_x if (is_basic and status != "optimal") else None
    incumbent_obj = best_obj if incumbent_x is not None else np.inf
    return _finalize(
        instance, master, ctx_bounds, history, status,
        total_start, cp_trace_path, last_x, last_obj, anchors=anchors,
        incumbent_x=incumbent_x, incumbent_obj=incumbent_obj, strategy=strategy,
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
             cp_cut_eviction: str = "reject",
             cp_dist_tol_rel: Optional[float] = None,
             cp_scenario_source: str = "noise",
             cp_n_scenarios: int = 200,
             cp_d0_quantile: float = 0.9,
             cp_objective_monotone: bool = False,
             cp_cut_whole_scenario: bool = True,
             cp_mip_gap: float = 1e-4,
             cp_uncertainty=None,
             cp_bank=None,
             ) -> tuple:
    """Cutting Planes for robust constraint learning (one driver, auto-selected oracle).

    The loop is identical across scenarios -- train nominal, build the master,
    solve for the optimal solution(s) ``x*``, separate, add cuts, terminate. The
    separation strategy is chosen automatically from the problem shape:

    - **basic** -- non-contextual synthetic only: a single global LP with a single
      learned constraint; worst-case separation at ``x*``. No ``cp_alpha``.
    - **coherent** -- contextual problems (gastric), multiple constraints,
      multiple ``x*``, and/or a learned objective: one *shared* relabeling drives
      all constraints (and optionally the epigraph objective), and the worst
      scenario -- ranked by **normalized average distance** -- is cut.

    ``cp_scenario_source``:
    - ``"noise"`` (default): separate over a FIXED :class:`ScenarioBank` of
      ``cp_n_scenarios`` draws from the shared uncertainty set D. Because the
      draws do not move between iterations, the worst violation over the bank is
      monotone and ``tau`` has a stable meaning. Pass ``cp_bank`` to reuse one
      bank across a tau grid, or ``cp_uncertainty`` to set D's shape.
    - ``"bootstrap"``: the legacy localized bootstrap resample, redrawn each
      iteration. Kept as an ablation and to reproduce earlier results; this is the
      path whose fresh-sample-vs-frozen-d0 mismatch made the trace oscillate.

    Only the argmax scenario becomes a cut, so the bank is a separation *pool*,
    not an embedded ensemble -- CP's master stays small at bank sizes the wrapper
    could not embed.

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
        cp_dist_tol_rel=cp_dist_tol_rel,
        cp_robustify_objective=cp_robustify_objective,
        cp_cut_eviction=cp_cut_eviction,
        cp_scenario_source=cp_scenario_source,
        cp_n_scenarios=cp_n_scenarios,
        cp_d0_quantile=cp_d0_quantile,
        cp_objective_monotone=cp_objective_monotone,
        cp_cut_whole_scenario=cp_cut_whole_scenario,
        cp_mip_gap=cp_mip_gap,
        cp_uncertainty=cp_uncertainty,
        cp_bank=cp_bank,
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

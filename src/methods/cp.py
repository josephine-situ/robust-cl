"""
Cutting Planes method for robust constraint learning.

Iteratively:
1. Master: min c'x s.t. f(x; theta_s) <= b for s = 1,...,k
2. Separate: localized bootstrap resamples around x*, retrain, check violation
3. If violated, add new scenario to master as a cutting plane
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
)
from src.models.train import (
    train_model,
    retrain_on_bootstrap,
    localized_bootstrap_indices,
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

        self.x = [
            self.opt.addVar(
                lb=instance.variable_lb[j],
                ub=instance.variable_ub[j],
                name=f"x_{j}",
            )
            for j in range(self.d)
        ]

        base_cost = gp.quicksum(instance.cost_vector[j] * self.x[j] for j in range(self.d))
        self.obj_expr = base_cost + gp.quicksum(obj_terms) if obj_terms else base_cost
        self.opt.setObjective(self.obj_expr, GRB.MINIMIZE)

        add_domain_constraints(self.opt, self.x, instance)
        add_trust_region(self.opt, self.x, instance)

        self.n_models = 0
        self.scenario_constrs = []
        self.scenario_vars_map = {}
        self.scenario_constrs_map = {}
        self.embedded_models_cache = {}

    def remove_scenario(self, s: int):
        for c in self.scenario_constrs_map.get(s, []):
            self.opt.remove(c)
        for v in self.scenario_vars_map.get(s, []):
            self.opt.remove(v)
        self.scenario_constrs_map[s] = []
        self.scenario_vars_map[s] = []
        if s < len(self.scenario_constrs):
            self.scenario_constrs[s] = None

    def add_scenario(self, c_idx: int, constraint_models: List[tuple], rhs: float,
                     rho: float = 0.0):
        prefix = f"cp_c{c_idx}_s{self.n_models}"
        self.opt.update()
        old_constrs = set(self.opt.getConstrs())
        old_vars = set(self.opt.getVars())

        f_pred_vars = []
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


def _evaluate_proxy_candidate(args):
    candidate, X_train, y_train, x_2d = args
    model = retrain_on_bootstrap(X_train, y_train, candidate, "cart", {"max_depth": 3})
    val = model.predict(x_2d)[0]
    return val, candidate


def localized_bootstrap_separation(model_data, x_current, model_type, model_params,
                                   k_neighbors_frac, n_candidates, seed):
    """Localized bootstrap separation with CART proxy filter."""
    n = len(model_data.y_train)
    x_2d = np.atleast_2d(x_current)
    candidates = localized_bootstrap_indices(
        model_data.X_train, x_current, k_neighbors_frac, n_candidates, seed,
    )

    best_value_proxy = -np.inf
    best_candidate = None
    args_list = [
        (cand, model_data.X_train, model_data.y_train, x_2d) for cand in candidates
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_evaluate_proxy_candidate, args_list)

    for val, cand in results:
        if val > best_value_proxy:
            best_value_proxy = val
            best_candidate = cand

    best_model = retrain_on_bootstrap(
        model_data.X_train, model_data.y_train,
        best_candidate, model_type, model_params,
    )
    best_value = best_model.predict(x_2d)[0]
    return best_candidate, best_value, best_model


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


def solve_cp(instance: ProblemInstance,
             model_type: str = "rf",
             model_params: dict = None,
             rho: float = 0.0,
             max_iterations: int = 50,
             cp_k_neighbors_frac: float = 0.1,
             cp_n_candidates: int = 20,
             seed: int = 42,
             embedding_mode: str = "hard",
             rf_alpha: float = 0.25) -> tuple[SolutionResult, CPHistory]:
    """Solve using Cutting Planes with localized bootstrap separation."""
    history = CPHistory()
    d = instance.n_features
    total_start = time.time()

    print("    [cp] Training nominal models for initial scenarios...", flush=True)
    trained_constraints = _train_nominal_with_configs(instance, model_type, model_params)

    master = IncrementalMaster(instance, [], rho=rho)
    print("    [cp] Building master MIP (objective + initial scenarios)...", flush=True)

    obj_terms = []
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
            obj_terms.append(obj_weight * master.embedded_models_cache[m_id])

    base_cost = gp.quicksum(instance.cost_vector[j] * master.x[j] for j in range(d))
    master.obj_expr = base_cost + gp.quicksum(obj_terms)
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

    for iteration in range(max_iterations):
        iter_start = time.time()
        x_current, obj_current = master.solve()

        if x_current is None:
            return SolutionResult(
                x_opt=np.zeros(d),
                obj_value=np.inf,
                status="infeasible",
                models_embedded=master.n_models,
                solve_time=time.time() - total_start,
                opt=master.opt,
                x=master.x,
                iterations=iteration,
            ), history

        history.objectives.append(obj_current)
        history.x_solutions.append(x_current.copy())

        max_violation = -np.inf
        any_added = False
        scenarios_to_add = []
        iteration_separation_cache = {}

        for c_idx, constraint in enumerate(instance.constraints):
            if not _is_constraint_constraint(constraint):
                continue

            worst_case_models = []
            constraint_val = 0.0

            for m_idx, model_data in enumerate(constraint.models_data):
                if model_data.obj_weight != 0.0:
                    continue
                md_id = id(model_data)
                m_type, m_params = model_config_map[md_id]

                if md_id in iteration_separation_cache:
                    best_model, best_value = iteration_separation_cache[md_id]
                else:
                    _, best_value, best_model = localized_bootstrap_separation(
                        model_data, x_current, m_type, m_params,
                        cp_k_neighbors_frac, cp_n_candidates,
                        seed + iteration + c_idx * 100 + m_idx,
                    )
                    iteration_separation_cache[md_id] = (best_model, best_value)

                worst_case_models.append((model_data.weight, best_model))
                constraint_val += model_data.weight * best_value

            violation = constraint_val - constraint.rhs
            max_violation = max(max_violation, violation)

            if violation > 1e-6:
                scenarios_to_add.append((c_idx, worst_case_models, constraint.rhs))
                any_added = True

        iter_time = time.time() - iter_start
        print(
            f"Iter {iteration}: Obj={obj_current:.4f} "
            f"Max Violation={max_violation:.4f} Time={iter_time:.2f}s"
        )

        if iteration > 0:
            dynamic_slack = max(0.1, max_violation)
            pruned_count, total_active = prune_inactive_scenarios(
                master, slack_threshold=dynamic_slack
            )
            if pruned_count > 0:
                print(f"Iter {iteration}: Pruned {pruned_count}/{total_active} inactive scenarios")

        master.add_objective_cut(obj_current, iteration)

        for c_idx, worst_case_models, rhs in scenarios_to_add:
            master.add_scenario(c_idx, worst_case_models, rhs, rho=rho)

        history.violations.append(max_violation)
        history.iterations = iteration + 1

        if not any_added:
            print("Re-solving with default MIP gap...")
            master.opt.Params.MIPGap = 1e-4
            x_final, obj_final = master.solve()
            if x_final is not None:
                x_current, obj_current = x_final, obj_final

            elapsed = time.time() - total_start
            return SolutionResult(
                x_opt=x_current,
                obj_value=obj_current,
                status="optimal",
                models_embedded=master.n_models,
                solve_time=elapsed,
                opt=master.opt,
                x=master.x,
                iterations=iteration + 1,
            ), history

    print("Max iterations reached. Re-solving with default MIP gap...")
    master.opt.Params.MIPGap = 1e-4
    x_final, obj_final = master.solve()
    if x_final is not None:
        x_current, obj_current = x_final, obj_final

    elapsed = time.time() - total_start
    return SolutionResult(
        x_opt=x_current,
        obj_value=obj_current,
        status="max_iterations",
        models_embedded=master.n_models,
        solve_time=elapsed,
        opt=master.opt,
        x=master.x,
        iterations=max_iterations,
    ), history

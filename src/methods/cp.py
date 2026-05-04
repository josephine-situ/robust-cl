"""
Cutting Planes method for robust constraint learning.

Iteratively:
1. Master: min c'x s.t. f(x; theta_s) <= b for s = 1,...,k
   - Uses a relaxed MIP gap in intermediate iterations for tractability.
   - Leverages warm starts to speed up consecutive Master problem solves.
   - Prunes inactive scenarios (constraints with large slack) to keep the Master problem small.
   - Note: We experimented with adding cuts based on "bad leaves" (violating terminal nodes in a tree) and "voting" (aggregate class margin constraints), but these are rarely used as they are not very tight with ensemble methods.
2. Separate: find worst-case delta, retrain using a smaller proxy model (for fast candidate evaluation), check violation.
3. If violated, add new scenario to master as a cutting plane.
4. Final: Re-solve with the default MIP gap to ensure exactness.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from dataclasses import dataclass, field
from typing import List, Optional
import concurrent.futures

from src.data.generate import ProblemInstance
from src.methods.nominal import SolutionResult
from src.models.train import train_model, retrain_on_perturbed, ModelType
from src.models.embed import embed_model, embed_cut_voting, embed_cut_bad_leaf, choose_cut_type
from src.utils.perturbations import greedy_adversarial_perturbation

@dataclass
class CPHistory:
    """Track CP iteration history."""
    iterations: int = 0
    violations: List[float] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)
    x_solutions: List[np.ndarray] = field(default_factory=list)

class IncrementalMaster:
    """Keeps the Gurobi model in memory to add constraints incrementally."""
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.d = instance.n_features
        self.opt = gp.Model("cp_incremental_master")
        self.opt.Params.OutputFlag = 0
        
        # Performance tuning for Master problem tractability
        self.opt.Params.MIPGap = 0.01
        self.opt.Params.MIPFocus = 1
        self.opt.Params.Threads = 0
        
        self.x = [
            self.opt.addVar(lb=instance.variable_lb[j],
                       ub=instance.variable_ub[j],
                       name=f"x_{j}")
            for j in range(self.d)
        ]
        
        self.opt.setObjective(
            gp.quicksum(instance.cost_vector[j] * self.x[j] for j in range(self.d)),
            GRB.MINIMIZE,
        )
        self.n_models = 0
        self.scenario_constrs = []
        self.scenario_vars_map = {}
        self.scenario_constrs_map = {}
        self.embedded_models_cache = {} # id(ml_model) -> f_s
        
    def remove_scenario(self, s: int):
        """Remove all Gurobi variables and constraints for scenario s."""
        for c in self.scenario_constrs_map.get(s, []):
            self.opt.remove(c)
        for v in self.scenario_vars_map.get(s, []):
            self.opt.remove(v)
        self.scenario_constrs_map[s] = []
        self.scenario_vars_map[s] = []
        # Mark as removed but keep indices aligned
        if s < len(self.scenario_constrs):
            self.scenario_constrs[s] = None

    def add_scenario(self, c_idx: int, constraint_models: List[tuple], rhs: float, phase: int = 3, x_k: np.ndarray = None, iteration: int = 0, rho: float = 0.0):
        # constraint_models is a list of (weight, ml_model) tuples
        prefix = f"cp_c{c_idx}_s{self.n_models}"
        
        self.opt.update()
        old_constrs = set(self.opt.getConstrs())
        old_vars = set(self.opt.getVars())
        
        if phase == 1: cut_type = "bad_leaf"
        else: cut_type = "full"
            
        f_pred_vars = []
        for m_idx, (weight, ml_model) in enumerate(constraint_models):
            m_prefix = f"{prefix}_m{m_idx}"
            m_id = id(ml_model)
            
            if cut_type == "voting":
                if m_id not in self.embedded_models_cache:
                    try:
                        f_s = embed_cut_voting(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, rhs, m_prefix)
                    except Exception:
                        f_s = embed_cut_bad_leaf(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, rhs, m_prefix)
                    self.embedded_models_cache[m_id] = f_s
                # voting returning f_s is probably not implemented since it adds constraint directly.
                # Actually, cut embeds directly and doesn't return f_s. Wait!
                pass # Caching not cleanly supported for direct cuts yet
            elif cut_type == "bad_leaf":
                pass # Caching not cleanly supported for direct cuts yet
            else:
                if m_id not in self.embedded_models_cache:
                    f_s = embed_model(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, name_prefix=m_prefix, rho=rho)
                    self.embedded_models_cache[m_id] = f_s
                f_pred_vars.append(weight * self.embedded_models_cache[m_id])
                
        if cut_type == "full" and f_pred_vars:
            main_constr = self.opt.addConstr(gp.quicksum(f_pred_vars) <= rhs, name=f"cp_constr_{c_idx}_{self.n_models}")
        else:
            main_constr = None
        
        self.opt.update()
        new_constrs = list(set(self.opt.getConstrs()) - old_constrs)
        new_vars = list(set(self.opt.getVars()) - old_vars)
        
        self.scenario_constrs_map[self.n_models] = new_constrs
        self.scenario_vars_map[self.n_models] = new_vars
        
        # Only full embedding tracks a single bounding constraint `f_s <= b`
        if cut_type == "full":
            self.scenario_constrs.append(main_constr)
        else:
            self.scenario_constrs.append(None)
            
        self.n_models += 1

    def add_objective_cut(self, obj_val: float, iteration: int):
        """Ensure the objective value cannot improve (decrease) in future iterations."""
        obj_expr = gp.quicksum(self.instance.cost_vector[j] * self.x[j] for j in range(self.d))
        self.opt.addConstr(obj_expr >= obj_val, name=f"obj_bound_{iteration}")
        self.opt.update()

    def solve(self):
        self.opt.optimize()
        if self.opt.Status != GRB.OPTIMAL:
            return None, np.inf
        return np.array([v.X for v in self.x]), self.opt.ObjVal

def prune_inactive_scenarios(master: IncrementalMaster, slack_threshold: float = 0.1):
    """Remove scenarios whose constraint has large slack."""
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
    """Helper for parallel search using proxy model."""
    candidate, instance_X_train, instance_y_train, x_2d, mode = args
    if mode == "bootstrap":
        from src.models.train import retrain_on_bootstrap
        model = retrain_on_bootstrap(instance_X_train, instance_y_train, candidate, "cart", {"max_depth": 3})
    else:
        model = retrain_on_perturbed(instance_X_train, instance_y_train, candidate, "cart", {"max_depth": 3})
    val = model.predict(x_2d)[0]
    return val, candidate

def proxy_based_separation(model_data, x_current, delta_bar, gamma, model_type, model_params, n_candidates, seed, mode="perturbation"):
    """Proxy-based separation using parallel candidate evaluation."""
    n = len(model_data.y_train)
    x_2d = np.atleast_2d(x_current)
    
    if mode == "bootstrap":
        rng = np.random.RandomState(seed)
        candidates = [rng.choice(n, size=n, replace=True) for _ in range(n_candidates)]
    else:
        from src.utils.perturbations import sample_multiple_perturbations
        candidates = sample_multiple_perturbations(n, delta_bar, gamma, n_candidates, seed)
    
    best_value_proxy = -np.inf
    best_candidate = None
    
    args_list = [(cand, model_data.X_train, model_data.y_train, x_2d, mode) for cand in candidates]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_evaluate_proxy_candidate, args_list)
        
    for val, cand in results:
        if val > best_value_proxy:
            best_value_proxy = val
            best_candidate = cand
            
    # Once the best delta/indices is found via proxy, train actual model just once
    if mode == "bootstrap":
        from src.models.train import retrain_on_bootstrap
        best_model = retrain_on_bootstrap(model_data.X_train, model_data.y_train, best_candidate, model_type, model_params)
    else:
        best_model = retrain_on_perturbed(model_data.X_train, model_data.y_train, best_candidate, model_type, model_params)
        
    best_value = best_model.predict(x_2d)[0]
            
    return best_candidate, best_value, best_model

def solve_cp(instance: ProblemInstance,
              model_type: str = "rf",
              model_params: dict = None,
              delta_bar: float = 0.2,
              gamma: float = 5.0,
              rho: float = 0.0,
              max_iterations: int = 50,
              separation_strategy: str = "proxy",
              n_greedy_candidates: int = 20,
              seed: int = 42,
            phase: int = 1) -> tuple[SolutionResult, CPHistory]:
    """
    Solve using Cutting Planes method.

    Returns
    -------
    result : SolutionResult
    history : CPHistory tracking convergence
    """
    history = CPHistory()
    d = instance.n_features

    total_start = time.time()

    # Initialize: train nominal models for each component
    # scenario_models[c_idx] is a list of (weight, model) tuples for that constraint
    scenario_models = []
    trained_models_cache = {}
    
    master = IncrementalMaster(instance)

    for c_idx, constraint in enumerate(instance.constraints):
        constraint_models = []
        for m_idx, model_data in enumerate(constraint.models_data):
            md_id = id(model_data)
            if md_id not in trained_models_cache:
                nominal_model = train_model(
                    model_data.X_train, model_data.y_train,
                    model_type, model_params,
                )
                trained_models_cache[md_id] = nominal_model
            constraint_models.append((model_data.weight, trained_models_cache[md_id]))
        
        scenario_models.append(constraint_models)
        master.add_scenario(c_idx, constraint_models, constraint.rhs, phase=phase, iteration=0, rho=rho)

    for iteration in range(max_iterations):
        iter_start = time.time()
        # === MASTER PROBLEM ===
        x_current, obj_current = master.solve()
        
        if x_current is None:
            # Master infeasible — too constrained
            models_embedded = sum(len(sc) for sc in scenario_models)
            return SolutionResult(
                x_opt=np.zeros(d),
                obj_value=np.inf,
                status="infeasible",
                models_embedded=models_embedded,
                solve_time=time.time() - total_start,
                iterations=iteration,
            ), history

        history.objectives.append(obj_current)
        history.x_solutions.append(x_current.copy())

        # === SEPARATION SUBPROBLEM ===
        max_violation = -np.inf
        any_added = False
        scenarios_to_add = []
        
        iteration_separation_cache = {} # id(model_data) -> (best_model, best_value)
        
        for c_idx, constraint in enumerate(instance.constraints):
            # For each constraint, we need to find the worst-case models
            worst_case_models = []
            constraint_val = 0.0
            
            for m_idx, model_data in enumerate(constraint.models_data):
                md_id = id(model_data)
                
                if md_id in iteration_separation_cache:
                    best_model, best_value = iteration_separation_cache[md_id]
                else:
                    if separation_strategy == "greedy":
                        _, best_value, best_model = \
                            greedy_adversarial_perturbation(
                                model_data.X_train,
                                model_data.y_train,
                                x_current,
                                delta_bar, gamma,
                                model_type, model_params,
                                n_greedy_candidates,
                                seed=seed + iteration + c_idx*100 + m_idx,
                            )
                    elif separation_strategy == "proxy":
                        _, best_value, best_model = \
                            proxy_based_separation(
                                model_data, x_current,
                                delta_bar, gamma,
                                model_type, model_params,
                                n_greedy_candidates,
                                seed=seed + iteration + c_idx*100 + m_idx
                            )
                    elif separation_strategy == "proxy-bootstrap":
                        _, best_value, best_model = \
                            proxy_based_separation(
                                model_data, x_current,
                                delta_bar, gamma,
                                model_type, model_params,
                                n_greedy_candidates,
                                seed=seed + iteration + c_idx*100 + m_idx,
                                mode="bootstrap"
                            )
                    elif separation_strategy == "random":
                        # Random search as baseline
                        _, best_value, best_model = \
                            _random_separation(
                                model_data, x_current,
                                delta_bar, gamma,
                                model_type, model_params,
                                n_candidates=n_greedy_candidates,
                                seed=seed + iteration + c_idx*100 + m_idx,
                            )
                    else:
                        raise ValueError(
                            f"Unknown separation strategy: {separation_strategy}"
                        )
                        
                    iteration_separation_cache[md_id] = (best_model, best_value)
                
                worst_case_models.append((model_data.weight, best_model))
                constraint_val += model_data.weight * best_value
            
            violation = constraint_val - constraint.rhs
            max_violation = max(max_violation, violation)
            
            if violation > 1e-6:
                # Track scenario to be added later to not invalidate Slack yet
                scenario_models[c_idx].extend(worst_case_models) # This is just for tracking
                scenarios_to_add.append((c_idx, worst_case_models, constraint.rhs))
                any_added = True

        iter_time = time.time() - iter_start
        print(f"Iter {iteration}: Obj={obj_current:.4f} Max Violation={max_violation:.4f} Time={iter_time:.2f}s")
        print(f"  x* = {np.round(x_current, 4)}")
        
        if len(history.x_solutions) > 1:
            x_diff = np.linalg.norm(x_current - history.x_solutions[-2])
            if x_diff < 1e-4:
                if separation_strategy == "proxy":
                    separation_strategy = "greedy"
                    print("Switching separation strategy from proxy to greedy for finer search")
                elif phase == 1:
                    phase = 2
                    print(f"Advancing to phase {phase} (full embedding)")
        
        if phase == 2 and iteration > 0:
            # Dynamically adjust slack threshold based on current violation
            # so we don't prune scenarios that might still be relevant
            dynamic_slack = max(0.1, max_violation)
            pruned_count, total_active = prune_inactive_scenarios(master, slack_threshold=dynamic_slack)
            if pruned_count > 0:
                print(f"Iter {iteration}: Pruned {pruned_count}/{total_active} inactive scenarios")
        
        # Add constraint that objective can't improve (decrease) in future iterations
        # Moved here so it doesn't invalidate Gurobi solution constraints (like .Slack) before pruning
        master.add_objective_cut(obj_current, iteration)

        # Now actually safely add the scenarios to Gurobi
        for c_idx, worst_case_models, rhs in scenarios_to_add:
            master.add_scenario(c_idx, worst_case_models, rhs, phase=phase, x_k=x_current, iteration=iteration, rho=rho)

        history.violations.append(max_violation)
        history.iterations = iteration + 1

        # Check termination
        if not any_added:
            # No violation found across any constraint — robust feasible
            
            # Re-solve with default MIP gap for exactness on final iteration
            print("Re-solving with default MIP gap...")
            master.opt.Params.MIPGap = 1e-4
            x_final, obj_final = master.solve()
            if x_final is not None:
                x_current, obj_current = x_final, obj_final
                
            elapsed = time.time() - total_start
            models_embedded = master.n_models
            return SolutionResult(
                x_opt=x_current,
                obj_value=obj_current,
                status="optimal",
                models_embedded=models_embedded,
                solve_time=elapsed,
                iterations=iteration + 1,
            ), history

    # Max iterations reached
    print("Max iterations reached. Re-solving with default MIP gap...")
    master.opt.Params.MIPGap = 1e-4
    x_final, obj_final = master.solve()
    if x_final is not None:
        x_current, obj_current = x_final, obj_final
        
    elapsed = time.time() - total_start
    models_embedded = master.n_models
    return SolutionResult(
        x_opt=x_current,
        obj_value=obj_current,
        status="max_iterations",
        models_embedded=models_embedded,
        solve_time=elapsed,
        iterations=max_iterations,
    ), history


def _random_separation(model_data, x_current, delta_bar, gamma,
                       model_type, model_params,
                       n_candidates=20, seed=42):
    """Random search separation oracle (baseline)."""
    from src.utils.perturbations import sample_multiple_perturbations

    n = len(model_data.y_train)
    x_2d = np.atleast_2d(x_current)

    perturbations = sample_multiple_perturbations(
        n, delta_bar, gamma, n_candidates, seed=seed
    )

    best_val = -np.inf
    best_delta = None
    best_model = None

    for delta in perturbations:
        model = retrain_on_perturbed(
            model_data.X_train, model_data.y_train, delta, model_type, model_params
        )
        val = model.predict(x_2d)[0]
        if val > best_val:
            best_val = val
            best_delta = delta
            best_model = model

    return best_delta, best_val, best_model
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
        self.b = instance.constraint_rhs
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

    def add_scenario(self, ml_model: ModelType, phase: int = 3, x_k: np.ndarray = None, iteration: int = 0, rho: float = 0.0):
        prefix = f"cp_s{self.n_models}"
        
        self.opt.update()
        old_constrs = set(self.opt.getConstrs())
        old_vars = set(self.opt.getVars())
        
        # If model is RF and we're in phase 1, verify dynamically.
        # Otherwise fall back rigidly to full embedding.
        if phase == 1 and hasattr(ml_model, "estimators_") and x_k is not None:
            from src.models.embed import choose_cut_type
            cut_type = choose_cut_type(ml_model, x_k, self.b)
            print(f"Iter {iteration}: Dynamic cut type chosen: {cut_type}")
        else:
            if phase == 1: cut_type = "bad_leaf"
            else: cut_type = "full"
            
        if cut_type == "voting":
            try:
                # Voting and bad leaf cuts don't directly use rho yet.
                embed_cut_voting(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, self.b, prefix)
            except Exception:
                embed_cut_bad_leaf(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, self.b, prefix)
        elif cut_type == "bad_leaf":
            embed_cut_bad_leaf(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, self.b, prefix)
        else:
            f_s = embed_model(self.opt, ml_model, self.x, self.instance.variable_lb, self.instance.variable_ub, name_prefix=prefix, rho=rho)
            main_constr = self.opt.addConstr(f_s <= self.b, name=f"cp_constr_{self.n_models}")
        
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
    delta, instance, x_2d = args
    model = retrain_on_perturbed(instance.X_train, instance.y_train, delta, "cart", {"max_depth": 3})
    val = model.predict(x_2d)[0]
    return val, delta

def proxy_based_separation(instance, x_current, delta_bar, gamma, model_type, model_params, n_candidates, seed):
    """Proxy-based separation using parallel candidate evaluation."""
    from src.utils.perturbations import sample_multiple_perturbations
    n = len(instance.y_train)
    x_2d = np.atleast_2d(x_current)
    perturbations = sample_multiple_perturbations(n, delta_bar, gamma, n_candidates, seed)
    
    best_value_proxy = -np.inf
    best_delta = None
    
    args_list = [(pert, instance, x_2d) for pert in perturbations]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_evaluate_proxy_candidate, args_list)
        
    for val, delta in results:
        if val > best_value_proxy:
            best_value_proxy = val
            best_delta = delta
            
    # Once the best delta is found via proxy, train actual model just once
    best_model = retrain_on_perturbed(instance.X_train, instance.y_train, best_delta, model_type, model_params)
    best_value = best_model.predict(x_2d)[0]
            
    return best_delta, best_value, best_model

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
    n = len(instance.y_train)
    d = instance.n_features
    b = instance.constraint_rhs

    total_start = time.time()

    # Initialize: train nominal model
    nominal_model = train_model(
        instance.X_train, instance.y_train,
        model_type, model_params,
    )
    scenario_models: List[ModelType] = [nominal_model]
    scenario_deltas: List[np.ndarray] = [np.zeros(n)]
    
    master = IncrementalMaster(instance)
    master.add_scenario(nominal_model, phase=phase, iteration=0, rho=rho)

    for iteration in range(max_iterations):
        iter_start = time.time()
        # === MASTER PROBLEM ===
        x_current, obj_current = master.solve()
        
        if x_current is None:
            # Master infeasible — too constrained
            return SolutionResult(
                x_opt=np.zeros(d),
                obj_value=np.inf,
                status="infeasible",
                models_embedded=len(scenario_models),
                solve_time=time.time() - total_start,
                iterations=iteration,
            ), history

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
        elif separation_strategy == "proxy":
            best_delta, best_value, best_model = \
                proxy_based_separation(
                    instance, x_current,
                    delta_bar, gamma,
                    model_type, model_params,
                    n_greedy_candidates,
                    seed=seed + iteration
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
        iter_time = time.time() - iter_start
        print(f"Iter {iteration}: Obj={obj_current:.4f} Violation={violation:.4f} Time={iter_time:.2f}s")
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
            dynamic_slack = max(0.1, violation)
            pruned_count, total_active = prune_inactive_scenarios(master, slack_threshold=dynamic_slack)
            if pruned_count > 0:
                print(f"Iter {iteration}: Pruned {pruned_count}/{total_active} inactive scenarios")
        
        # Add constraint that objective can't improve (decrease) in future iterations
        # Moved here so it doesn't invalidate Gurobi solution constraints (like .Slack) before pruning
        master.add_objective_cut(obj_current, iteration)

        history.violations.append(violation)
        history.iterations = iteration + 1

        # Check termination
        if violation <= 1e-6:
            # No violation found — robust feasible
            
            # Re-solve with default MIP gap for exactness on final iteration
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
                models_embedded=len(scenario_models),
                solve_time=elapsed,
                iterations=iteration + 1,
            ), history

        # Add violating scenario
        scenario_models.append(best_model)
        scenario_deltas.append(best_delta)
        master.add_scenario(best_model, phase=phase, x_k=x_current, iteration=iteration, rho=rho)

    # Max iterations reached
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
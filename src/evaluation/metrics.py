"""
Evaluation metrics for robust constraint learning.

For each solution x*, evaluate:
1. True feasibility (if ground truth available)
2. Feasibility rate over held-out perturbations
3. Worst-case violation over held-out perturbations
4. Objective cost
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Callable, Any

from src.data.generate import ProblemInstance
from src.models.train import retrain_on_perturbed
from src.utils.perturbations import sample_multiple_perturbations


@dataclass
class EvaluationResult:
    """Evaluation of a single solution over prescriptive test set."""
    method: str
    mean_obj_value: float            # average GT objective over test set
    models_embedded: int
    mean_solve_time: float           # average time per prescriptive solve
    mean_iterations: Optional[float] # average iterations per solve (for CP)

    # Feasibility metrics
    feasibility_rate: float        # fraction of test set that is fully feasible (all constraints)
    constraint_violation_rates: List[float] # violation rate per constraint on test set
    mean_constraint_violations: List[float] # average violation magnitude (max(0, output-rhs)) per constraint
    
    true_feasible: Optional[bool] = None # Added for syntactic convenience when n_test=1
    true_constraint_value: Optional[float] = None # Added for syntactic convenience when n_test=1
    worst_case_violation: Optional[float] = None # Added for syntactic convenience when n_test=1


def evaluate_prescriptive_performance(solver_fn: Callable,
                                      instance: ProblemInstance,
                                      method_name: str,
                                      **solver_kwargs) -> EvaluationResult:
    """
    Evaluates a solver using prescriptive evaluation over a test set.
    For each test row, the contextual variables are fixed, and the solver
    optimizes the decision variables. The resulting prescription is
    evaluated against Ground Truth (GT) models.
    """
    import time
    
    n_test = instance.X_test.shape[0] if instance.X_test.size > 0 else 1
    
    obj_values = []
    solve_times = []
    iterations_list = []
    
    # Feasibility tracking
    n_constraints = len(instance.constraints)
    all_feasible_count = 0
    constraint_violations = np.zeros((n_test, n_constraints))

    # Keep original bounds
    orig_lb = instance.variable_lb.copy()
    orig_ub = instance.variable_ub.copy()

    for i in range(n_test):
        # 1. Update bounds for context variables
        if instance.X_test.size > 0:
            context = instance.X_test[i]
            for j, c_idx in enumerate(instance.context_var_indices):
                instance.variable_lb[c_idx] = context[j]
                instance.variable_ub[c_idx] = context[j]
        
        # 2. Call solver
        start_time = time.time()
        result = solver_fn(instance, **solver_kwargs)
        if isinstance(result, tuple):
            result = result[0]
        solve_time = time.time() - start_time
        
        solve_times.append(solve_time)
        
        if getattr(result, 'iterations', None) is not None:
            iterations_list.append(result.iterations)

        # 3. Evaluate Ground Truth
        if result.status == "optimal":
            x_opt = result.x_opt
            
            # Objective
            if instance.gt_objective is not None:
                obj_val = instance.gt_objective(x_opt)
                if isinstance(obj_val, np.ndarray):
                    obj_val = obj_val[0] if obj_val.size == 1 else obj_val.item()
                obj_values.append(float(obj_val))
            else:
                obj_values.append(result.obj_value)
                
            # Constraints
            all_c_feasible = True
            for c_idx, constraint in enumerate(instance.constraints):
                gt_model = instance.gt_constraints[c_idx]
                
                c_val = gt_model(x_opt)
                if isinstance(c_val, np.ndarray):
                    c_val = c_val[0] if c_val.size == 1 else c_val.item()
                
                violation = max(0.0, float(c_val) - constraint.rhs)
                constraint_violations[i, c_idx] = violation
                
                if violation > 1e-4:
                    all_c_feasible = False
                    
            if all_c_feasible:
                all_feasible_count += 1
                
        else:
            # Infeasible problem
            # Depending on how we evaluate, maybe we just penalize
            obj_values.append(np.inf)
            constraint_violations[i, :] = np.inf
            
    # Restore original bounds
    instance.variable_lb = orig_lb
    instance.variable_ub = orig_ub

    # Remove infs for mean calculation of obj if there are valid ones
    valid_objs = [o for o in obj_values if o != np.inf]
    mean_obj = np.mean(valid_objs) if valid_objs else np.inf
    
    # Calculate constraint violation rates
    violation_rates = []
    mean_violations = []
    for c_idx in range(n_constraints):
        c_vis = constraint_violations[:, c_idx]
        valid_c_vis = c_vis[c_vis != np.inf]
        
        if len(valid_c_vis) > 0:
            violation_rates.append(np.mean(valid_c_vis > 1e-4))
            mean_violations.append(np.mean(valid_c_vis))
        else:
            violation_rates.append(1.0)
            mean_violations.append(np.inf)

    true_feasible = (all_feasible_count == 1) if n_test == 1 else None
    true_constraint_value = float(c_val) if n_test == 1 and all_c_feasible is not None else None # Last c_val
    worst_case_violation = float(np.max(constraint_violations[constraint_violations != np.inf])) if np.any(constraint_violations != np.inf) else np.inf

    return EvaluationResult(
        method=method_name,
        mean_obj_value=float(mean_obj),
        models_embedded=sum(len(c.models_data) for c in instance.constraints), # Assuming this represents structural complexity
        mean_solve_time=float(np.mean(solve_times)),
        mean_iterations=float(np.mean(iterations_list)) if iterations_list else None,
        feasibility_rate=float(all_feasible_count / n_test),
        constraint_violation_rates=[float(v) for v in violation_rates],
        mean_constraint_violations=[float(v) for v in mean_violations],
        true_feasible=true_feasible,
        true_constraint_value=true_constraint_value,
        worst_case_violation=worst_case_violation
    )


def evaluate_all(solver_fns: dict,
                 instance: ProblemInstance,
                 **solver_kwargs) -> List[EvaluationResult]:
    """
    Evaluate multiple solvers using prescriptive performance.

    Parameters
    ----------
    solver_fns : dict mapping method_name -> solver callable
    solver_kwargs : common kwargs strictly passed to solvers
    """
    evaluations = []
    for method_name, solver_fn in solver_fns.items():
        print(f"Evaluating solving method: {method_name.upper()}...")
        ev = evaluate_prescriptive_performance(
            solver_fn, instance, method_name, **solver_kwargs
        )
        evaluations.append(ev)

    return evaluations
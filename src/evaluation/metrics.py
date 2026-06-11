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
    models_embedded: int
    mean_solve_time: float           # time to embed/solve initial model
    mean_iterations: Optional[float] # iterations for CP

    # Out-of-sample metrics (Test)
    mean_obj_value: float            # average GT objective over test set
    feasibility_rate: float          # fraction of test set that is fully feasible
    constraint_violation_rates: List[float] # violation rate per constraint on test set
    mean_constraint_violations: List[float] # average violation magnitude per constraint
    worst_case_violation: Optional[float] = None
    
    # In-sample metrics (Train)
    mean_obj_value_train: Optional[float] = None
    feasibility_rate_train: Optional[float] = None
    constraint_violation_rates_train: Optional[List[float]] = None
    mean_constraint_violations_train: Optional[List[float]] = None
    worst_case_violation_train: Optional[float] = None

    true_feasible: Optional[bool] = None # Added for syntactic convenience when n_test=1
    true_constraint_value: Optional[float] = None # Added for syntactic convenience when n_test=1


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
    
    # 1. Call solver ONCE to construct and solve the initial model
    start_time = time.time()
    result = solver_fn(instance, **solver_kwargs)
    if isinstance(result, tuple):
        result = result[0]
    solve_time = time.time() - start_time
    
    models_embedded = result.models_embedded
    iterations = getattr(result, 'iterations', None)
    
    def eval_dataset(X_data, is_single_eval=False):
        n_obs = X_data.shape[0] if X_data is not None and X_data.size > 0 else (1 if is_single_eval else 0)
        if n_obs == 0:
            return None
            
        obj_values = []
        n_constraints = len(instance.constraints)
        all_feasible_count = 0
        constraint_violations = np.full((n_obs, n_constraints), np.nan)
        
        last_all_c_feasible = None
        last_c_val = None

        for i in range(n_obs):
            if X_data is not None and X_data.size > 0:
                # Reset context bounds, then fix to this row's covariates
                # (indices 0..8 of X are drug features, not context)
                for c_idx in instance.context_var_indices:
                    result.x[c_idx].lb = instance.variable_lb[c_idx]
                    result.x[c_idx].ub = instance.variable_ub[c_idx]
                for c_idx in instance.context_var_indices:
                    val = float(X_data[i, c_idx])
                    result.x[c_idx].lb = val
                    result.x[c_idx].ub = val

                # Re-optimize for new context bounds
                result.opt.Params.DualReductions = 0
                if n_obs > 1:
                    result.opt.Params.MIPGap = 0.01
                result.opt.update()
                result.opt.optimize()
                
                if n_obs > 1 and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n_obs):
                    print(f"  {method_name}: test row {i + 1}/{n_obs}", flush=True)
                
                if result.opt.Status != 2: # GRB.OPTIMAL is 2
                    continue
                x_opt = np.array([v.X for v in result.x])
                obj_val_opt = result.opt.ObjVal
            else:
                if result.status != "optimal":
                    continue
                x_opt = result.x_opt
                obj_val_opt = result.obj_value

            # Evaluate Ground Truth
            if instance.gt_objective is not None:
                obj_val = instance.gt_objective(x_opt)
                if isinstance(obj_val, np.ndarray):
                    obj_val = obj_val[0] if obj_val.size == 1 else obj_val.item()
                obj_values.append(float(obj_val))
            else:
                obj_values.append(obj_val_opt)
                
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
            
            last_all_c_feasible = all_c_feasible
            last_c_val = c_val
                
        # Aggregate metrics
        valid_objs = [o for o in obj_values if np.isfinite(o)]
        mean_obj = np.mean(valid_objs) if valid_objs else np.nan
        
        violation_rates = []
        mean_violations = []
        for c_idx in range(n_constraints):
            c_vis = constraint_violations[:, c_idx]
            valid_c_vis = c_vis[np.isfinite(c_vis)]
            if len(valid_c_vis) > 0:
                violation_rates.append(np.mean(valid_c_vis > 1e-4))
                mean_violations.append(np.mean(valid_c_vis))
            else:
                violation_rates.append(np.nan)
                mean_violations.append(np.nan)

        finite_violations = constraint_violations[np.isfinite(constraint_violations)]
        worst_case_violation = float(np.max(finite_violations)) if finite_violations.size > 0 else np.nan

        return {
            "n_obs": n_obs,
            "mean_obj": float(mean_obj),
            "feasibility_rate": float(all_feasible_count / n_obs),
            "violation_rates": [float(v) for v in violation_rates],
            "mean_violations": [float(v) for v in mean_violations],
            "worst_case_violation": worst_case_violation,
            "true_feasible": (all_feasible_count == 1) if n_obs == 1 else None,
            "true_constraint_value": float(last_c_val) if n_obs == 1 and last_all_c_feasible is not None else None
        }

    # Evaluate test and train sets
    eval_test = eval_dataset(instance.X_test, is_single_eval=True)  # Always try to evaluate at least 1 obs (synthetic matches n_test=1)
    eval_train = eval_dataset(instance.X_train, is_single_eval=False)

    # Use original values for evaluation limits if possible
    res = EvaluationResult(
        method=method_name,
        models_embedded=models_embedded,
        mean_solve_time=float(solve_time),
        mean_iterations=float(iterations) if iterations is not None else None,
        
        mean_obj_value=eval_test["mean_obj"] if eval_test else np.nan,
        feasibility_rate=eval_test["feasibility_rate"] if eval_test else np.nan,
        constraint_violation_rates=eval_test["violation_rates"] if eval_test else [],
        mean_constraint_violations=eval_test["mean_violations"] if eval_test else [],
        worst_case_violation=eval_test["worst_case_violation"] if eval_test else np.nan,
        true_feasible=eval_test["true_feasible"] if eval_test else None,
        true_constraint_value=eval_test["true_constraint_value"] if eval_test else None,
    )
    
    if eval_train:
        res.mean_obj_value_train = eval_train["mean_obj"]
        res.feasibility_rate_train = eval_train["feasibility_rate"]
        res.constraint_violation_rates_train = eval_train["violation_rates"]
        res.mean_constraint_violations_train = eval_train["mean_violations"]
        res.worst_case_violation_train = eval_train["worst_case_violation"]

    return res


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


def evaluate_given_treatments(instance: ProblemInstance,
                              method_name: str = "given") -> EvaluationResult:
    """Score observed test treatments against GT (no optimization)."""
    X_data = instance.X_test
    n_obs = X_data.shape[0]
    n_constraints = len(instance.constraints)

    obj_values = []
    all_feasible_count = 0
    constraint_violations = np.full((n_obs, n_constraints), np.nan)

    for i in range(n_obs):
        x_opt = X_data[i].copy()
        obj_val = instance.gt_objective(x_opt)
        if isinstance(obj_val, np.ndarray):
            obj_val = float(obj_val.flat[0])
        obj_values.append(float(obj_val))

        all_c_feasible = True
        for c_idx, constraint in enumerate(instance.constraints):
            gt_fn = instance.gt_constraints[c_idx]
            c_val = gt_fn(x_opt)
            if isinstance(c_val, np.ndarray):
                c_val = float(c_val.flat[0])
            violation = max(0.0, c_val - constraint.rhs)
            constraint_violations[i, c_idx] = violation
            if violation > 1e-4:
                all_c_feasible = False
        if all_c_feasible:
            all_feasible_count += 1

    violation_rates = []
    for c_idx in range(n_constraints):
        c_vis = constraint_violations[:, c_idx]
        violation_rates.append(float(np.mean(c_vis > 1e-4)))

    finite_violations = constraint_violations[np.isfinite(constraint_violations)]
    worst = float(np.max(finite_violations)) if finite_violations.size > 0 else np.nan

    return EvaluationResult(
        method=method_name,
        models_embedded=0,
        mean_solve_time=0.0,
        mean_iterations=None,
        mean_obj_value=float(np.mean(obj_values)),
        feasibility_rate=float(all_feasible_count / n_obs),
        constraint_violation_rates=violation_rates,
        mean_constraint_violations=[float(np.mean(constraint_violations[:, c])) for c in range(n_constraints)],
        worst_case_violation=worst,
    )
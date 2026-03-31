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
from typing import Optional, List

from src.data.generate import ProblemInstance
from src.models.train import retrain_on_perturbed
from src.utils.perturbations import sample_multiple_perturbations


@dataclass
class EvaluationResult:
    """Evaluation of a single solution."""
    method: str
    obj_value: float
    models_embedded: int
    solve_time: float
    iterations: Optional[int]

    # Ground truth (if available)
    true_constraint_value: Optional[float]  # f_true(x*)
    true_feasible: Optional[bool]           # f_true(x*) <= b

    # Held-out perturbation evaluation
    feasibility_rate: float        # fraction feasible
    worst_case_violation: float    # max_m f(x*; theta*(delta_m)) - b
    mean_prediction: float         # average prediction across scenarios
    prediction_std: float          # std of predictions


def evaluate_solution(x_opt: np.ndarray,
                      instance: ProblemInstance,
                      method_name: str,
                      obj_value: float,
                      models_embedded: int,
                      solve_time: float,
                      model_type: str = "rf",
                      model_params: dict = None,
                      delta_bar: float = 0.2,
                      gamma: float = 5.0,
                      n_held_out: int = 100,
                      seed: int = 42,
                      iterations: Optional[int] = None,
                      ) -> EvaluationResult:
    """
    Evaluate a solution x_opt against held-out perturbations
    and (optionally) ground truth.
    """
    n = len(instance.y_train)
    b = instance.constraint_rhs
    x_2d = np.atleast_2d(x_opt)

    # --- Ground truth evaluation ---
    true_val = None
    true_feas = None
    if instance.f_true is not None:
        true_val = instance.f_true(x_opt)[0] if x_opt.ndim == 1 \
            else instance.f_true(x_opt)
        true_feas = bool(true_val <= b)

    # --- Held-out perturbation evaluation ---
    held_out_deltas = sample_multiple_perturbations(
        n, delta_bar, gamma, n_held_out, seed=seed + 9999,
    )

    predictions = []
    for delta in held_out_deltas:
        m = retrain_on_perturbed(
            instance.X_train, instance.y_train, delta,
            model_type, model_params,
        )
        pred = m.predict(x_2d)[0]
        predictions.append(pred)

    predictions = np.array(predictions)
    feasible_flags = predictions <= b

    return EvaluationResult(
        method=method_name,
        obj_value=obj_value,
        models_embedded=models_embedded,
        solve_time=solve_time,
        iterations=iterations,
        true_constraint_value=true_val,
        true_feasible=true_feas,
        feasibility_rate=np.mean(feasible_flags),
        worst_case_violation=np.max(predictions) - b,
        mean_prediction=np.mean(predictions),
        prediction_std=np.std(predictions),
    )


def evaluate_all(results: dict,
                 instance: ProblemInstance,
                 model_type: str = "rf",
                 model_params: dict = None,
                 delta_bar: float = 0.2,
                 gamma: float = 5.0,
                 n_held_out: int = 100,
                 seed: int = 42) -> List[EvaluationResult]:
    """
    Evaluate all method results.

    Parameters
    ----------
    results : dict mapping method_name -> SolutionResult
    """
    evaluations = []
    for method_name, sol in results.items():
        if sol.status == "infeasible":
            evaluations.append(EvaluationResult(
                method=method_name,
                obj_value=np.inf,
                models_embedded=sol.models_embedded,
                solve_time=sol.solve_time,
                iterations=sol.iterations,
                true_constraint_value=None,
                true_feasible=None,
                feasibility_rate=0.0,
                worst_case_violation=np.inf,
                mean_prediction=np.inf,
                prediction_std=0.0,
            ))
            continue

        ev = evaluate_solution(
            sol.x_opt, instance, method_name,
            sol.obj_value, sol.models_embedded, sol.solve_time,
            model_type, model_params,
            delta_bar, gamma, n_held_out, seed,
            iterations=sol.iterations,
        )
        evaluations.append(ev)

    return evaluations
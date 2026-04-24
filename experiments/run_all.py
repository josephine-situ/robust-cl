"""
Main experiment runner.

Runs all five methods on a problem instance, evaluates, and
saves results.
"""

import yaml
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generate import synthetic_nonlinear
from src.methods.nominal import solve_nominal
from src.methods.robust_classification import solve_robust_classification
from src.methods.wrapper import solve_wrapper
from src.methods.cp import solve_cp
from src.evaluation.metrics import evaluate_all


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config):
    """Run all methods and evaluate."""

    print("=" * 60)
    print("ROBUST CONSTRAINT LEARNING EXPERIMENT")
    print("=" * 60)

    # --- Generate data ---
    print("\n[1] Generating problem instance...")
    instance = synthetic_nonlinear(
        n_train=config["data"]["n_train"],
        n_features=config["data"]["n_features"],
        noise_std=config["data"]["noise_std"],
    )
    print(f"    n_train={len(instance.y_train)}, "
          f"d={instance.n_features}, "
          f"noise_std={config['data']['noise_std']}")

    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    delta_bar = config["uncertainty"]["delta_bar"]
    gamma = config["uncertainty"]["gamma"]

    results = {}

    # --- Method 1: Nominal ---
    print("\n[2] Solving NOMINAL...")
    results["nominal"] = solve_nominal(
        instance, model_type, model_params,
    )
    print(f"    obj={results['nominal'].obj_value:.4f}, "
          f"status={results['nominal'].status}")

    # # --- Method 2: Robust Classification ---
    # print("\n[3] Solving ROBUST CLASSIFICATION...")
    # results["robust_cls"] = solve_robust_classification(
    #     instance, model_type, model_params,
    #     delta_bar=delta_bar, gamma=gamma,
    #     n_perturbations=config["methods"]["robust_classification"]
    #         .get("n_perturbations", 50),
    # )
    # print(f"    obj={results['robust_cls'].obj_value:.4f}, "
    #       f"status={results['robust_cls'].status}")

    # # --- Method 3: Wrapper ---
    # print("\n[4] Solving WRAPPER...")
    # wrapper_cfg = config["methods"]["wrapper"]
    # results["wrapper"] = solve_wrapper(
    #     instance, model_type, model_params,
    #     n_estimators=wrapper_cfg["n_estimators"],
    #     alpha=wrapper_cfg["alpha"],
    # )
    # print(f"    obj={results['wrapper'].obj_value:.4f}, "
    #       f"status={results['wrapper'].status}, "
    #       f"models={results['wrapper'].models_embedded}")

    # --- Method 5: CP ---
    print("\n[6] Solving CP...")
    cp_cfg = config["methods"]["cp"]
    cp_result, cp_trace = solve_cp(
        instance, model_type, model_params,
        delta_bar=delta_bar, gamma=gamma,
        max_iterations=cp_cfg["max_iterations"],
        separation_strategy=cp_cfg["separation_strategy"],
        n_greedy_candidates=cp_cfg["n_greedy_candidates"],
    )
    results["cp"] = cp_result
    print(f"    obj={results['cp'].obj_value:.4f}, "
          f"status={results['cp'].status}, "
          f"models={results['cp'].models_embedded}, "
          f"iters={results['cp'].iterations}")

    # --- Evaluate all ---
    print("\n[7] Evaluating all methods on held-out perturbations...")
    eval_cfg = config["evaluation"]
    evaluations = evaluate_all(
        results, instance,
        model_type, model_params,
        delta_bar, gamma,
        n_held_out=eval_cfg["n_held_out"],
        seed=eval_cfg["seed"],
    )

    # --- Results table ---
    rows = []
    for ev in evaluations:
        rows.append({
            "method": ev.method,
            "objective": ev.obj_value,
            "models_embedded": ev.models_embedded,
            "solve_time": ev.solve_time,
            "iterations": ev.iterations,
            "true_feasible": ev.true_feasible,
            "true_constraint_val": ev.true_constraint_value,
            "feasibility_rate": ev.feasibility_rate,
            "worst_violation": ev.worst_case_violation,
            "pred_mean": ev.mean_prediction,
            "pred_std": ev.prediction_std,
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # --- Save ---
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results.csv", index=False)

    # Save CP trace
    if cp_trace.violations:
        trace_df = pd.DataFrame({
            "iteration": list(range(1, len(cp_trace.violations) + 1)),
            "obj_value": cp_trace.objectives,
            "violation": cp_trace.violations,
        })
        trace_df.to_csv("results/cp_trace.csv", index=False)

    return df, cp_trace


if __name__ == "__main__":
    config = load_config()
    run_experiment(config)
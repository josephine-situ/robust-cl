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

from src.data.generate import synthetic_nonlinear, gastric_cancer
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
    print(f"\n[1] Generating problem instance ({config['data']['type']})...")
    if config["data"]["type"] == "gastric_cancer":
        instance = gastric_cancer()
    else:
        instance = synthetic_nonlinear(
            n_train=config["data"]["n_train"],
            n_features=config["data"]["n_features"],
            noise_std=config["data"]["noise_std"],
        )
    print(f"    n_train (model 1)={len(instance.constraints[0].models_data[0].y_train)}, "
          f"d={instance.n_features}, "
          f"noise_std={config['data']['noise_std']}")

    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    delta_bar = config["uncertainty"]["delta_bar"]
    gamma = config["uncertainty"]["gamma"]
    
    from functools import partial
    
    solver_fns = {}

    # --- Method 1: Nominal ---
    solver_fns["nominal"] = partial(
        solve_nominal, model_type=model_type, model_params=model_params, rho=0.0
    )
          
    # --- Method 1.5: Robust Param ---
    robust_param_cfg = config["methods"].get("robust_param", {})
    robust_rho = robust_param_cfg.get("rho", 0.0)
    solver_fns["robust_param"] = partial(
        solve_nominal, model_type=model_type, model_params=model_params, rho=robust_rho
    )

    # --- Method 2: Robust Classification ---
    solver_fns["robust_cls"] = partial(
        solve_robust_classification, model_type=model_type, model_params=model_params,
        delta_bar=delta_bar, gamma=gamma, rho=0.0,
        n_perturbations=config["methods"]["robust_classification"].get("n_perturbations", 50)
    )

    # --- Method 3: Wrapper ---
    wrapper_cfg = config["methods"]["wrapper"]
    solver_fns["wrapper"] = partial(
        solve_wrapper, model_type=model_type, model_params=model_params, rho=0.0,
        n_estimators=wrapper_cfg["n_estimators"],
        alpha=wrapper_cfg["alpha"]
    )

    # --- Method 5: CP ---
    cp_cfg = config["methods"]["cp"]
    solver_fns["cp"] = partial(
        solve_cp, model_type=model_type, model_params=model_params,
        delta_bar=delta_bar, gamma=gamma, rho=0.0,
        max_iterations=cp_cfg["max_iterations"],
        separation_strategy=cp_cfg["separation_strategy"],
        n_greedy_candidates=cp_cfg["n_greedy_candidates"]
    )

    # --- Evaluate all ---
    print("\n[Evaluating all methods prescriptively...]")
    evaluations = evaluate_all(
        solver_fns, instance
    )

    # --- Results table ---
    rows = []
    for ev in evaluations:
        rows.append({
            "method": ev.method,
            "objective": ev.mean_obj_value,
            "models_embedded": ev.models_embedded,
            "solve_time": ev.mean_solve_time,
            "iterations": ev.mean_iterations,
            "feasibility_rate": ev.feasibility_rate,
            "constraint_violation_rates": ev.constraint_violation_rates,
            "worst_violation": ev.mean_constraint_violations,
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # --- Save ---
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results.csv", index=False)
    
    return df, None

if __name__ == "__main__":
    config = load_config()
    run_experiment(config)
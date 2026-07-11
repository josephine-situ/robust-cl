"""
Main experiment runner.

Runs all five methods on a problem instance, evaluates, and
saves results.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --cv-configs results/cv/synthetic_selected_configs.json
"""

import argparse
import json
import yaml
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generate import synthetic_nonlinear, gastric_cancer
from src.methods.nominal import solve_nominal
from src.methods.robust_regression import solve_robust_regression
from src.methods.wrapper import solve_wrapper, _get_shared_bootstrap_indices
from src.methods.cp import solve_cp
from src.evaluation.metrics import evaluate_all


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config, cv_configs=None):
    """Run all methods and evaluate.

    Parameters
    ----------
    cv_configs : optional dict loaded from a ``*_selected_configs.json`` produced
        by ``run_cv.py``.  For the synthetic problem the entry
        ``"synthetic_constraint"`` overrides the global model type/params from
        config.  For gastric the dict is passed directly to
        ``gastric_cancer(fixed_constraint_configs=...)``.
    """

    print("=" * 60)
    print("ROBUST CONSTRAINT LEARNING EXPERIMENT")
    print("=" * 60)

    print(f"\n[1] Generating problem instance ({config['data']['type']})...")
    if config["data"]["type"] == "gastric_cancer":
        instance = gastric_cancer(
            fixed_constraint_configs=cv_configs if cv_configs else None
        )
    else:
        instance = synthetic_nonlinear(
            n_train=config["data"]["n_train"],
            n_features=config["data"]["n_features"],
            noise_std=config["data"]["noise_std"],
        )
    print(f"    n_train (model 1)={len(instance.constraints[0].models_data[0].y_train)}, "
          f"d={instance.n_features}, "
          f"noise_std={config['data'].get('noise_std', 'n/a')}")

    # Resolve model type/params: CV-selected override > config default
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    if cv_configs and config["data"]["type"] != "gastric_cancer":
        # For synthetic, override global model from the single constraint entry
        synth_cfg = cv_configs.get("synthetic_constraint")
        if synth_cfg:
            model_type = synth_cfg["model_type"]
            model_params = synth_cfg.get("model_params", {})
            print(f"    CV-selected model: {model_type}  params={model_params}")
    unc = config["uncertainty"]
    n_bootstrap = unc.get("n_bootstrap", 25)
    bootstrap_seed = unc.get("bootstrap_seed", 42)
    cp_k_neighbors_frac = unc.get("cp_k_neighbors_frac", 0.1)
    cp_k_neighbors_min = unc.get("cp_k_neighbors_min", 1)
    cp_n_candidates = unc.get("cp_n_candidates", 20)

    from functools import partial

    bootstrap_cache = _get_shared_bootstrap_indices(
        instance, model_type, model_params, n_bootstrap, bootstrap_seed
    )

    solver_fns = {}

    solver_fns["nominal"] = partial(
        solve_nominal, model_type=model_type, model_params=model_params, rho=0.0
    )

    robust_param_cfg = config["methods"].get("robust_param", {})
    robust_rho = robust_param_cfg.get("rho", 0.0)
    solver_fns["robust_param"] = partial(
        solve_nominal, model_type=model_type, model_params=model_params, rho=robust_rho
    )

    robust_reg_cfg = config["methods"].get("robust_reg", {})
    solver_fns["robust_reg"] = partial(
        solve_robust_regression,
        model_type=model_type,
        model_params=model_params,
        label_eps=robust_reg_cfg.get("label_eps", 0.1),
        budget_frac=robust_reg_cfg.get("budget_frac", 0.5),
        K=robust_reg_cfg.get("K", 5),
        seed=bootstrap_seed,
        rho=0.0,
    )

    wrapper_cfg = config["methods"]["wrapper"]
    solver_fns["wrapper"] = partial(
        solve_wrapper,
        model_type=model_type,
        model_params=model_params,
        rho=0.0,
        n_estimators=n_bootstrap,
        alpha=wrapper_cfg["alpha"],
        seed=bootstrap_seed,
        bootstrap_cache=bootstrap_cache,
    )

    cp_cfg = config["methods"]["cp"]
    solver_fns["cp"] = partial(
        solve_cp,
        model_type=model_type,
        model_params=model_params,
        rho=0.0,
        max_iterations=cp_cfg["max_iterations"],
        cp_k_neighbors_frac=cp_k_neighbors_frac,
        cp_k_neighbors_min=cp_k_neighbors_min,
        cp_n_candidates=cp_n_candidates,
        seed=bootstrap_seed,
    )

    print("\n[Evaluating all methods prescriptively...]")
    evaluations = evaluate_all(solver_fns, instance)

    rows = []
    for ev in evaluations:
        row = {
            "method": ev.method,
            "objective": ev.mean_obj_value,
            "models_embedded": ev.models_embedded,
            "solve_time": ev.mean_solve_time,
            "iterations": ev.mean_iterations,
            "feasibility_rate": ev.feasibility_rate,
            "constraint_violation_rates": ev.constraint_violation_rates,
            "worst_violation": ev.worst_case_violation,
        }
        if ev.mean_obj_value_train is not None:
            row["objective_train"] = ev.mean_obj_value_train
            row["feasibility_rate_train"] = ev.feasibility_rate_train
            row["worst_violation_train"] = ev.worst_case_violation_train
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    os.makedirs("results/synthetic", exist_ok=True)
    out_path = "results/synthetic/results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return df, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robust CL experiment")
    parser.add_argument(
        "--cv-configs",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a *_selected_configs.json from run_cv.py. "
            "Overrides the model type/params from config.yaml with CV-selected models."
        ),
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cv_configs = None
    if args.cv_configs:
        with open(args.cv_configs, "r") as f:
            cv_configs = json.load(f)
        print(f"Loaded CV configs from {args.cv_configs}")

    run_experiment(cfg, cv_configs=cv_configs)

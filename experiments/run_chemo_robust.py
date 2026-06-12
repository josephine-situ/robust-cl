"""
Compare robust CL methods on gastric cancer chemotherapy (Table 6 metrics).

Usage:
  python experiments/run_chemo_robust.py --quick   # local smoke run
  python experiments/run_chemo_robust.py             # full comparison
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generate import gastric_cancer, filter_constraints
from src.methods.nominal import solve_nominal
from src.methods.robust_classification import solve_robust_classification
from src.methods.wrapper import (
    solve_wrapper,
    solve_tree_violation_wrapper,
    _get_shared_bootstrap_indices,
)
from src.methods.cp import solve_cp
from src.evaluation.chemo_metrics import (
    evaluate_given_table6,
    evaluate_prescribed_table6,
    build_table6_rows,
)

ALL_CONSTRAINTS = [
    "dlt_constraint", "blood_constraint", "constitutional_constraint",
    "infection_constraint", "gi_constraint", "os_constraint",
]
DLT_ONLY = ["dlt_constraint", "os_constraint"]

ALL_METHODS = [
    "nominal", "tree_violation", "robust_param", "robust_cls", "wrapper", "cp",
]


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_run_settings(config, args):
    chemo_cfg = config["methods"].get("chemo", {})
    quick_cfg = chemo_cfg.get("quick", {})
    unc = config["uncertainty"]

    if args.quick:
        settings = {
            "max_test_rows": quick_cfg.get("max_test_rows", 5),
            "methods_to_run": quick_cfg.get(
                "methods_to_run", ["nominal", "wrapper", "cp"]
            ),
            "constraint_modes": quick_cfg.get("constraint_modes", ["all_constraints"]),
            "n_bootstrap": quick_cfg.get("n_bootstrap", 5),
            "cp_max_iterations": quick_cfg.get("cp_max_iterations", 5),
            "cp_n_candidates": quick_cfg.get("cp_n_candidates", 5),
            "cp_k_neighbors_frac": quick_cfg.get("cp_k_neighbors_frac", 0.05),
            "output_path": "results/chemo_robust_table6_quick.csv",
        }
    else:
        settings = {
            "max_test_rows": None,
            "methods_to_run": chemo_cfg.get("methods_to_run", ALL_METHODS),
            "constraint_modes": chemo_cfg.get(
                "constraint_modes", ["all_constraints", "dlt_only"]
            ),
            "n_bootstrap": unc.get("n_bootstrap", 25),
            "cp_max_iterations": config["methods"]["cp"].get("max_iterations", 20),
            "cp_n_candidates": unc.get("cp_n_candidates", 20),
            "cp_k_neighbors_frac": unc.get("cp_k_neighbors_frac", 0.1),
            "output_path": "results/chemo_robust_table6.csv",
        }

    if args.max_test_rows is not None:
        settings["max_test_rows"] = args.max_test_rows
    if args.methods:
        settings["methods_to_run"] = args.methods
    if args.output:
        settings["output_path"] = args.output

    settings["bootstrap_seed"] = unc.get("bootstrap_seed", 42)
    settings["embedding_mode"] = config["methods"].get("embedding_mode", "hard")
    settings["rf_alpha"] = config["methods"].get("chemo_wrapper", {}).get("alpha", 0.25)
    settings["wrapper_alpha"] = config["methods"]["wrapper"].get("alpha", 0.1)
    settings["robust_rho"] = config["methods"].get("robust_param", {}).get("rho", 0.05)
    return settings


def _build_solvers(config, settings, instance):
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    n_bootstrap = settings["n_bootstrap"]
    seed = settings["bootstrap_seed"]
    embedding_mode = settings["embedding_mode"]
    rf_alpha = settings["rf_alpha"]

    bootstrap_cache = _get_shared_bootstrap_indices(
        instance, model_type, model_params, n_bootstrap, seed
    )

    solvers = {
        "nominal": partial(
            solve_nominal,
            model_type=model_type,
            model_params=model_params,
            rho=0.0,
            embedding_mode=embedding_mode,
            rf_alpha=rf_alpha,
        ),
        "tree_violation": partial(
            solve_tree_violation_wrapper,
            model_type=model_type,
            model_params=model_params,
            alpha=rf_alpha,
            rho=0.0,
        ),
        "robust_param": partial(
            solve_nominal,
            model_type=model_type,
            model_params=model_params,
            rho=settings["robust_rho"],
            embedding_mode=embedding_mode,
            rf_alpha=rf_alpha,
        ),
        "robust_cls": partial(
            solve_robust_classification,
            model_type=model_type,
            model_params=model_params,
            n_bootstrap=n_bootstrap,
            seed=seed,
            rho=0.0,
            embedding_mode=embedding_mode,
            rf_alpha=rf_alpha,
            bootstrap_cache=bootstrap_cache,
        ),
        "wrapper": partial(
            solve_wrapper,
            model_type=model_type,
            model_params=model_params,
            n_estimators=n_bootstrap,
            alpha=settings["wrapper_alpha"],
            seed=seed,
            rho=0.0,
            bootstrap_cache=bootstrap_cache,
        ),
        "cp": partial(
            solve_cp,
            model_type=model_type,
            model_params=model_params,
            rho=0.0,
            max_iterations=settings["cp_max_iterations"],
            cp_k_neighbors_frac=settings["cp_k_neighbors_frac"],
            cp_n_candidates=settings["cp_n_candidates"],
            seed=seed,
            embedding_mode=embedding_mode,
            rf_alpha=rf_alpha,
        ),
    }
    return solvers


def run_chemo_robust(config, args):
    settings = _resolve_run_settings(config, args)
    instance = gastric_cancer()
    n_test = instance.X_test.shape[0]
    n_train = instance.X_train.shape[0]

    print("=" * 60)
    print("CHEMO ROBUST METHOD COMPARISON (Table 6 metrics)")
    print("=" * 60)
    print(f"Train: {n_train}, Test: {n_test}")
    print(f"Methods: {settings['methods_to_run']}")
    print(f"Constraint modes: {settings['constraint_modes']}")
    if settings["max_test_rows"]:
        print(f"Max test rows: {settings['max_test_rows']}")

    solvers = _build_solvers(config, settings, instance)

    all_rows = []
    eval_mask = None
    given_values = None

    for method in settings["methods_to_run"]:
        if method not in solvers:
            print(f"Skipping unknown method: {method}")
            continue

        solver_fn = solvers[method]
        print(f"\n{'=' * 40}\nMethod: {method}\n{'=' * 40}")

        for constraint_mode in settings["constraint_modes"]:
            if constraint_mode == "all_constraints":
                names = ALL_CONSTRAINTS
            elif constraint_mode == "dlt_only":
                names = DLT_ONLY
            else:
                raise ValueError(f"Unknown constraint mode: {constraint_mode}")

            sub = filter_constraints(instance, names)
            print(f"\n  constraint_mode={constraint_mode}")

            prescribed, feasible_mask, mean_time, sd_time = evaluate_prescribed_table6(
                solver_fn,
                sub,
                eval_mask=eval_mask,
                max_test_rows=settings["max_test_rows"],
                method_name=method,
                constraint_mode=constraint_mode,
            )
            n_feasible = int(feasible_mask.sum())
            print(f"  Feasible prescriptions: {n_feasible}/{n_test}")

            if constraint_mode == "all_constraints" and eval_mask is None:
                eval_mask = feasible_mask.copy()
                given_values = evaluate_given_table6(instance, eval_mask)
                n_eval = int(eval_mask.sum())
                print(f"  Shared evaluation cohort: {n_eval} test rows")

            report_mask = eval_mask if eval_mask is not None else feasible_mask
            n_prescribed = int((feasible_mask & report_mask).sum()) if report_mask is not None else n_feasible

            rows = build_table6_rows(
                instance,
                constraint_mode=constraint_mode,
                given_values=given_values or evaluate_given_table6(instance, report_mask),
                prescribed_values=prescribed,
                n_test=n_test,
                n_prescribed=n_prescribed,
                mean_solve_time=mean_time,
                solve_time_sd=sd_time,
            )
            for row in rows:
                row_dict = row.__dict__.copy()
                row_dict["method"] = method
                all_rows.append(row_dict)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(settings["output_path"], index=False)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved to {settings['output_path']}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Chemo robust method comparison")
    parser.add_argument("--quick", action="store_true", help="Small local smoke run")
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_chemo_robust(config, args)


if __name__ == "__main__":
    main()

"""
Replicate OptiCL chemotherapy Table 6 (Maragno et al. 2025, Section 5.5).

Compares observed (given) regimens to prescriptions from the paper's full model
(RF tree-violation wrapper with alpha=0.25) under All Constraints vs DLT Only.
"""

import os
import sys
from functools import partial

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generate import gastric_cancer, filter_constraints
from src.methods.wrapper import solve_tree_violation_wrapper
from src.evaluation.chemo_metrics import (
    evaluate_given_table6,
    evaluate_prescribed_table6,
    build_table6_rows,
    table6_results_to_dataframe,
)


ALL_CONSTRAINTS = [
    "dlt_constraint", "blood_constraint", "constitutional_constraint",
    "infection_constraint", "gi_constraint", "os_constraint",
]
DLT_ONLY = ["dlt_constraint", "os_constraint"]


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_chemo_replication(config):
    print("=" * 60)
    print("OPTICL CHEMOTHERAPY REPLICATION (Table 6)")
    print("=" * 60)

    instance = gastric_cancer()
    n_test = instance.X_test.shape[0]
    n_train = instance.X_train.shape[0]
    print(f"Train: {n_train}, Test: {n_test} (paper target: 320, 96)")

    wrapper_cfg = config["methods"].get("chemo_wrapper", config["methods"]["wrapper"])
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]

    solver_fn = partial(
        solve_tree_violation_wrapper,
        model_type=model_type,
        model_params=model_params,
        rho=0.0,
        alpha=wrapper_cfg["alpha"],
    )

    rows = []

    for mode, names in [("all_constraints", ALL_CONSTRAINTS), ("dlt_only", DLT_ONLY)]:
        sub = filter_constraints(instance, names)
        print(f"\nOptimizing prescriptions ({mode})...")
        prescribed_values, feasible_mask, mean_time, sd_time = evaluate_prescribed_table6(
            solver_fn, sub,
        )
        n_prescribed = int(feasible_mask.sum())
        print(f"  Feasible prescriptions: {n_prescribed}/{n_test}")

        given_values = evaluate_given_table6(instance, feasible_mask)

        rows.extend(build_table6_rows(
            instance,
            constraint_mode=mode,
            given_values=given_values,
            prescribed_values=prescribed_values,
            n_test=n_test,
            n_prescribed=n_prescribed,
            mean_solve_time=mean_time,
            solve_time_sd=sd_time,
        ))

    df = table6_results_to_dataframe(rows)
    os.makedirs("results", exist_ok=True)
    out_path = "results/chemo_table6.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("TABLE 6 RESULTS (paper-aligned metrics)")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    config = load_config()
    run_chemo_replication(config)

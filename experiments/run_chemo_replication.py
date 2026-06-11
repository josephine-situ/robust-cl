"""
Replicate OptiCL chemotherapy Table 6: Given vs Nominal vs Wrapper,
All constraints vs DLT-only.
"""

import os
import sys
from functools import partial

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generate import gastric_cancer, filter_constraints
from src.methods.nominal import solve_nominal
from src.methods.wrapper import solve_tree_violation_wrapper
from src.evaluation.metrics import evaluate_prescriptive_performance, evaluate_given_treatments


ALL_CONSTRAINTS = [
    "dlt_constraint", "blood_constraint", "constitutional_constraint",
    "infection_constraint", "gi_constraint", "os_constraint",
]
DLT_ONLY = ["dlt_constraint", "os_constraint"]

TOXICITY_NAMES = [
    "dlt_constraint", "blood_constraint", "constitutional_constraint",
    "infection_constraint", "gi_constraint",
]


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _eval_to_row(ev, constraint_mode, instance):
    row = {
        "method": ev.method,
        "constraint_mode": constraint_mode,
        "mean_os": ev.mean_obj_value,
        "feasibility_rate": ev.feasibility_rate,
        "worst_violation": ev.worst_case_violation,
        "solve_time": ev.mean_solve_time,
        "models_embedded": ev.models_embedded,
    }
    for name, rate in zip(
        [c.name for c in instance.constraints],
        ev.constraint_violation_rates,
    ):
        row[f"violation_rate_{name}"] = rate
    return row


def run_chemo_replication(config):
    print("=" * 60)
    print("OPTICL CHEMOTHERAPY REPLICATION (Table 6)")
    print("=" * 60)

    instance = gastric_cancer()
    instance.X_train = None  # Table 6 evaluates test cohorts only
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    wrapper_cfg = config["methods"].get("chemo_wrapper", config["methods"]["wrapper"])

    rows = []

    print("\nEvaluating GIVEN (observed regimens)...")
    for mode, names in [("all", ALL_CONSTRAINTS), ("dlt_only", DLT_ONLY)]:
        sub = filter_constraints(instance, names)
        ev = evaluate_given_treatments(sub, method_name="given")
        rows.append(_eval_to_row(ev, mode, sub))

    solver_configs = [
        ("nominal", partial(solve_nominal, model_type=model_type, model_params=model_params, rho=0.0)),
        ("wrapper", partial(
            solve_tree_violation_wrapper, model_type=model_type, model_params=model_params, rho=0.0,
            alpha=wrapper_cfg["alpha"],
        )),
    ]

    for method_name, solver_fn in solver_configs:
        for mode, names in [("all", ALL_CONSTRAINTS), ("dlt_only", DLT_ONLY)]:
            sub = filter_constraints(instance, names)
            print(f"\nEvaluating {method_name.upper()} ({mode})...")
            ev = evaluate_prescriptive_performance(solver_fn, sub, method_name)
            rows.append(_eval_to_row(ev, mode, sub))

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_path = "results/chemo_table6.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("TABLE 6 RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    config = load_config()
    run_chemo_replication(config)

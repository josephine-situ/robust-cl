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
from src.methods.robust_regression import solve_robust_regression
from src.methods.wrapper import (
    solve_wrapper,
    solve_tree_violation_wrapper,
    _coherent_bootstrap_indices,
    train_bootstrap_ensembles_for_instance,
)
from src.methods.cp import solve_cp, select_anchor_contexts
from src.methods.calibrate import calibrate_strength
from src.evaluation.chemo_metrics import (
    evaluate_given_table6,
    evaluate_prescribed_table6,
    build_table6_rows,
    samestore_eval_mask,
    subset_table6_outcomes,
)

# Baselines whose single robustness knob is calibrated to the shared alpha.
# cp self-regulates via its p_infeas cap; nominal has no robustness knob.
CALIBRATED_METHODS = ["wrapper", "tree_violation", "robust_param", "robust_reg"]

ALL_CONSTRAINTS = [
    "dlt_constraint", "blood_constraint", "constitutional_constraint",
    "infection_constraint", "gi_constraint", "os_constraint",
]
DLT_ONLY = ["dlt_constraint", "os_constraint"]

ALL_METHODS = [
    "nominal", "tree_violation", "robust_param", "robust_reg", "wrapper", "cp",
]


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_run_settings(config, args):
    chemo_cfg = config["methods"].get("chemo", {})
    quick_cfg = chemo_cfg.get("quick", {})
    unc = config["uncertainty"]

    cp_cfg = config["methods"].get("cp", {})

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
            "alpha": quick_cfg.get("alpha", unc.get("alpha", 0.0)),
            "cp_n_anchors": quick_cfg.get("cp_n_anchors", cp_cfg.get("n_anchors", 4)),
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
            "cp_max_iterations": cp_cfg.get("max_iterations", 20),
            "cp_n_candidates": unc.get("cp_n_candidates", 20),
            "cp_k_neighbors_frac": unc.get("cp_k_neighbors_frac", 0.1),
            "alpha": unc.get("alpha", 0.0),
            "cp_n_anchors": cp_cfg.get("n_anchors", 15),
            "output_path": "results/chemo_robust_table6.csv",
        }

    settings["cp_anchor_source"] = cp_cfg.get("anchor_source", "train")
    settings["cp_anchor_method"] = cp_cfg.get("anchor_method", "kmedoids")
    settings["cp_trace_path"] = "results/cp_trace.csv"
    settings["cp_distance"] = cp_cfg.get("distance", "full")
    settings["cp_dist_tol"] = cp_cfg.get("dist_tol", 1e-3)

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

    calib_cfg = config.get("calibration", {})
    settings["calibrate_to_alpha"] = calib_cfg.get("enabled", False)
    settings["calib_n_grid"] = calib_cfg.get("n_grid", 5)
    settings["calib_wrapper_alpha_max"] = calib_cfg.get("wrapper_alpha_max", 0.5)
    settings["calib_tree_alpha_max"] = calib_cfg.get("tree_alpha_max", 0.5)
    settings["calib_rho_min"] = calib_cfg.get("rho_min", 0.01)
    settings["calib_rho_max"] = calib_cfg.get("rho_max", 0.05)
    return settings


def _build_solvers(config, settings, instance, bootstrap_cache):
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    n_bootstrap = settings["n_bootstrap"]
    seed = settings["bootstrap_seed"]
    embedding_mode = settings["embedding_mode"]
    rf_alpha = settings["rf_alpha"]

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
        "robust_reg": partial(
            solve_robust_regression,
            model_type=model_type,
            model_params=model_params,
            n_bootstrap=n_bootstrap,
            seed=seed,
            rho=0.0,
            embedding_mode=embedding_mode,
            rf_alpha=rf_alpha,
            conservativeness=1.0,
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
        # Single driver; gastric (multiple toxicity constraints over many
        # patients) auto-selects coherent separation with the shared alpha as the
        # feasibility coverage cap.
        "cp": partial(
            solve_cp,
            model_type=model_type,
            model_params=model_params,
            rho=0.0,
            max_iterations=settings["cp_max_iterations"],
            cp_k_neighbors_frac=settings["cp_k_neighbors_frac"],
            cp_n_candidates=settings["cp_n_candidates"],
            seed=seed,
            cp_alpha=settings["alpha"],
            cp_dist_tol=settings["cp_dist_tol"],
            cp_anchor_source=settings["cp_anchor_source"],
            cp_n_anchors=settings["cp_n_anchors"],
            cp_anchor_method=settings["cp_anchor_method"],
            cp_distance=settings["cp_distance"],
            cp_trace_path=settings["cp_trace_path"],
        ),
    }
    return solvers


def _constraint_names(constraint_mode):
    if constraint_mode == "all_constraints":
        return ALL_CONSTRAINTS
    if constraint_mode == "dlt_only":
        return DLT_ONLY
    raise ValueError(f"Unknown constraint mode: {constraint_mode}")


def _make_calibrated_solver(method, sub, settings, calib_contexts, model_type,
                            model_params, bootstrap_cache, ensembles_cache):
    """Calibrate a baseline's robustness knob on ``sub`` training contexts and
    return a solver_fn with that knob baked in."""
    nb = settings["n_bootstrap"]
    seed = settings["bootstrap_seed"]
    em = settings["embedding_mode"]
    rf_alpha = settings["rf_alpha"]
    target = settings["alpha"]

    if method == "wrapper":
        amax = settings["calib_wrapper_alpha_max"]
        strength_to_knob = lambda s: amax * (1.0 - s)  # s=1 strongest -> alpha_w=0
        build = lambda knob: partial(
            solve_wrapper, model_type=model_type, model_params=model_params,
            n_estimators=nb, alpha=knob, seed=seed, rho=0.0,
            bootstrap_cache=bootstrap_cache, ensembles_cache=ensembles_cache,
        )
    elif method == "tree_violation":
        amax = settings["calib_tree_alpha_max"]
        strength_to_knob = lambda s: amax * (1.0 - s)  # s=1 strongest -> alpha_t=0
        build = lambda knob: partial(
            solve_tree_violation_wrapper, model_type=model_type,
            model_params=model_params, alpha=knob, rho=0.0,
        )
    elif method == "robust_param":
        rho_min = settings["calib_rho_min"]
        rho_max = settings["calib_rho_max"]
        if rho_min >= rho_max:
            raise ValueError(
                f"calibration.rho_min ({rho_min}) must be < rho_max ({rho_max})"
            )
        strength_to_knob = lambda s: rho_min + (rho_max - rho_min) * s
        build = lambda knob: partial(
            solve_nominal, model_type=model_type, model_params=model_params,
            rho=knob, embedding_mode=em, rf_alpha=rf_alpha,
        )
    elif method == "robust_reg":
        strength_to_knob = lambda s: s                 # conservativeness quantile
        build = lambda knob: partial(
            solve_robust_regression, model_type=model_type, model_params=model_params,
            n_bootstrap=nb, seed=seed, rho=0.0, embedding_mode=em, rf_alpha=rf_alpha,
            conservativeness=knob, bootstrap_cache=bootstrap_cache,
            ensembles_cache=ensembles_cache,
        )
    else:
        raise ValueError(f"Method {method} is not calibratable")

    knob, frac = calibrate_strength(
        build, strength_to_knob, sub, calib_contexts, target,
        n_grid=settings["calib_n_grid"], label=method,
    )
    print(
        f"    [calib] {method}: chosen knob={knob:.4f} "
        f"(train_infeasible={frac * 100:.1f}%, target<={target * 100:.1f}%)",
        flush=True,
    )
    return build(knob)


def run_chemo_robust(config, args):
    settings = _resolve_run_settings(config, args)
    instance = gastric_cancer()
    n_test = instance.X_test.shape[0]
    n_train = instance.X_train.shape[0]

    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    calibrate = settings["calibrate_to_alpha"]

    print("=" * 60)
    print("CHEMO ROBUST METHOD COMPARISON (Table 6 metrics)")
    print("=" * 60)
    print(f"Train: {n_train}, Test: {n_test}")
    print(f"Methods: {settings['methods_to_run']}")
    print(f"Constraint modes: {settings['constraint_modes']}")
    print(f"Shared alpha: {settings['alpha']}  calibrate_to_alpha: {calibrate}")
    if settings["max_test_rows"]:
        print(f"Max test rows: {settings['max_test_rows']}")

    # Shared coherent bootstrap relabelings (one set of resamples applied to every
    # outcome) drive the coherent wrapper and robust_reg.
    coherent_cache = _coherent_bootstrap_indices(
        instance, settings["n_bootstrap"], settings["bootstrap_seed"]
    )
    solvers = _build_solvers(config, settings, instance, coherent_cache)

    calib_contexts = None
    ensembles_cache = None
    needs_calib = calibrate and any(
        m in CALIBRATED_METHODS for m in settings["methods_to_run"]
    )
    if needs_calib:
        calib_contexts = select_anchor_contexts(
            instance.X_train, instance.context_var_indices,
            settings["cp_n_anchors"], settings["cp_anchor_method"],
            settings["bootstrap_seed"],
        )
        print(f"Calibration contexts: {len(calib_contexts)} training anchors")
        if any(m in ("wrapper", "robust_reg") for m in settings["methods_to_run"]):
            # Pre-train the shared bootstrap ensembles once so calibration grid
            # evaluations (which change only the knob, not the models) reuse them.
            ensembles_cache, _ = train_bootstrap_ensembles_for_instance(
                instance, model_type, model_params,
                settings["n_bootstrap"], settings["bootstrap_seed"], coherent_cache,
            )

    def _resolve_solver(method, sub):
        if method in ("cp", "nominal") or not calibrate:
            return solvers[method]
        if method in CALIBRATED_METHODS:
            return _make_calibrated_solver(
                method, sub, settings, calib_contexts, model_type, model_params,
                coherent_cache, ensembles_cache,
            )
        return solvers[method]

    # Pass 1: optimize every (method, mode); store masks/outcomes/times.
    collected = {}
    # Collect per-mode feasibility masks so each mode's samestore only requires
    # patients to be feasible across methods within that mode (not globally
    # across both modes simultaneously).
    mode_masks: dict[str, list] = {m: [] for m in settings["constraint_modes"]}
    for method in settings["methods_to_run"]:
        if method not in solvers:
            print(f"Skipping unknown method: {method}")
            continue
        print(f"\n{'=' * 40}\nMethod: {method}\n{'=' * 40}")
        for constraint_mode in settings["constraint_modes"]:
            sub = filter_constraints(instance, _constraint_names(constraint_mode))
            print(f"\n  constraint_mode={constraint_mode}")
            solver_fn = _resolve_solver(method, sub)
            _, feasible_mask, mean_time, sd_time, full_outcomes = evaluate_prescribed_table6(
                solver_fn,
                sub,
                max_test_rows=settings["max_test_rows"],
                method_name=method,
                constraint_mode=constraint_mode,
            )
            n_feasible = int(feasible_mask.sum())
            print(f"  Feasible prescriptions: {n_feasible}/{n_test}")
            collected[(method, constraint_mode)] = {
                "mean_time": mean_time,
                "sd_time": sd_time,
                "full_outcomes": full_outcomes,
                "n_method_feasible": n_feasible,
            }
            mode_masks[constraint_mode].append(feasible_mask)

    # Per-mode samestore: patients feasible across all methods for that mode only.
    # This allows each table to use a larger, mode-appropriate evaluation cohort.
    per_mode_mask: dict[str, object] = {}
    per_mode_given: dict[str, dict] = {}
    for constraint_mode, masks in mode_masks.items():
        if not masks:
            continue
        mask = samestore_eval_mask(*masks)
        n_eval = int(mask.sum())
        print(
            f"\nSamestore cohort ({constraint_mode}): {n_eval}/{n_test} test rows"
        )
        per_mode_mask[constraint_mode] = mask
        per_mode_given[constraint_mode] = evaluate_given_table6(instance, mask)

    # Pass 2: build Table 6 rows on the per-mode samestore cohort.
    all_rows = []
    for method in settings["methods_to_run"]:
        for constraint_mode in settings["constraint_modes"]:
            key = (method, constraint_mode)
            if key not in collected:
                continue
            if constraint_mode not in per_mode_mask:
                continue
            res = collected[key]
            mode_mask = per_mode_mask[constraint_mode]
            n_eval = int(mode_mask.sum())
            prescribed = subset_table6_outcomes(res["full_outcomes"], mode_mask)
            rows = build_table6_rows(
                instance,
                constraint_mode=constraint_mode,
                given_values=per_mode_given[constraint_mode],
                prescribed_values=prescribed,
                n_test=n_test,
                n_prescribed=n_eval,
                mean_solve_time=res["mean_time"],
                solve_time_sd=res["sd_time"],
            )
            for row in rows:
                row_dict = row.__dict__.copy()
                row_dict["method"] = method
                row_dict["n_method_feasible"] = res["n_method_feasible"]
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

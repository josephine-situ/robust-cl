"""
Compare robust CL methods on gastric cancer chemotherapy (Table 6 metrics).

Usage:
  python experiments/run_chemo_robust.py --quick                                  # smoke run
  python experiments/run_chemo_robust.py                                           # full run
  python experiments/run_chemo_robust.py --cv-configs results/cv/gastric_selected_configs.json
"""

from __future__ import annotations

import argparse
import json
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
from src.methods.cv_calibrate import lookup_knob
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
            "cp_k_neighbors_min": quick_cfg.get(
                "cp_k_neighbors_min", unc.get("cp_k_neighbors_min", 1)
            ),
            "alpha": quick_cfg.get("alpha", unc.get("alpha", 0.0)),
            "cp_n_anchors": quick_cfg.get("cp_n_anchors", cp_cfg.get("n_anchors", 4)),
            "output_path": "results/gastric/chemo_robust_table6_quick.csv",
            "prescriptions_dir": "results/gastric/prescriptions",
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
            "cp_k_neighbors_min": unc.get("cp_k_neighbors_min", 1),
            "alpha": unc.get("alpha", 0.0),
            "cp_n_anchors": cp_cfg.get("n_anchors", 15),
            "output_path": "results/gastric/chemo_robust_table6.csv",
            "prescriptions_dir": "results/gastric/prescriptions",
        }

    settings["cp_anchor_source"] = cp_cfg.get("anchor_source", "train")
    settings["cp_anchor_method"] = cp_cfg.get("anchor_method", "kmedoids")
    settings["cp_trace_path"] = "results/gastric/cp_trace.csv"
    settings["cp_distance"] = cp_cfg.get("distance", "full")
    settings["cp_dist_tol"] = cp_cfg.get("dist_tol", 1e-3)
    settings["cp_robustify_objective"] = cp_cfg.get("robustify_objective", True)
    settings["cp_eval_mode"] = cp_cfg.get("eval_mode", "global")
    settings["cp_nearest_distance"] = cp_cfg.get("nearest_distance", "context")
    settings["cp_cut_eviction"] = cp_cfg.get("cut_eviction", "reject")
    settings["cp_scenario_source"] = cp_cfg.get("scenario_source", "noise")
    settings["cp_d0_quantile"] = cp_cfg.get("d0_quantile", 0.9)
    settings["cp_objective_monotone"] = cp_cfg.get("objective_monotone", False)
    settings["cp_mip_gap"] = float(cp_cfg.get("mip_gap", 1e-4))
    settings["cp_cut_whole_scenario"] = cp_cfg.get("cut_whole_scenario", True)
    # B: CP embeds one extra scenario per iteration, so it can afford a bank far
    # larger than the wrapper's P (which is embedded in full). --quick shrinks it.
    settings["cp_n_scenarios"] = (
        quick_cfg.get("cp_n_scenarios", 10) if args.quick
        else cp_cfg.get("n_scenarios", 200)
    )
    # The shared uncertainty set D -- one object handed to cp, wrapper and
    # robust_reg, so a difference between them is a difference in METHOD.
    from src.methods.uncertainty import uncertainty_set_from_config
    settings["uncertainty_set"] = uncertainty_set_from_config(config)
    settings["calibration_method"] = config.get("calibration", {}).get("method", "alpha")
    settings["pareto_center_factors"] = config.get("cv_calibration", {}).get(
        "pareto_center_factors", [0.5, 0.75, 1.0, 1.5, 2.0])

    if args.max_test_rows is not None:
        settings["max_test_rows"] = args.max_test_rows
    if args.methods:
        settings["methods_to_run"] = args.methods
    if args.output:
        settings["output_path"] = args.output
    # CP ablation overrides (default None -> use config.yaml values).
    if getattr(args, "cp_robustify_objective", None) is not None:
        settings["cp_robustify_objective"] = (args.cp_robustify_objective == "true")
    if getattr(args, "cp_eval_mode", None) is not None:
        settings["cp_eval_mode"] = args.cp_eval_mode

    settings["bootstrap_seed"] = unc.get("bootstrap_seed", 42)
    settings["embedding_mode"] = config["methods"].get("embedding_mode", "hard")
    settings["rf_alpha"] = config["methods"].get("chemo_wrapper", {}).get("alpha", 0.25)
    _wrap_cfg = config["methods"]["wrapper"]
    settings["wrapper_alpha"] = _wrap_cfg.get("alpha", 0.1)
    settings["wrapper_scenario_source"] = _wrap_cfg.get("scenario_source", "noise")
    settings["wrapper_robustify_objective"] = _wrap_cfg.get("robustify_objective", False)
    settings["robust_rho"] = config["methods"].get("robust_param", {}).get("rho", 0.05)
    rr_cfg = config["methods"].get("robust_reg", {})
    settings["robust_reg_label_eps"] = rr_cfg.get("label_eps", 0.1)
    settings["robust_reg_budget_frac"] = rr_cfg.get("budget_frac", 0.5)
    settings["robust_reg_K"] = rr_cfg.get("K", 5)

    calib_cfg = config.get("calibration", {})
    settings["calibrate_to_alpha"] = calib_cfg.get("enabled", False)
    settings["calib_n_grid"] = calib_cfg.get("n_grid", 5)
    settings["calib_wrapper_alpha_max"] = calib_cfg.get("wrapper_alpha_max", 0.5)
    settings["calib_tree_alpha_max"] = calib_cfg.get("tree_alpha_max", 0.5)
    settings["calib_rho_min"] = calib_cfg.get("rho_min", 0.01)
    settings["calib_rho_max"] = calib_cfg.get("rho_max", 0.05)
    settings["calib_robust_reg_eps_max"] = calib_cfg.get("robust_reg_eps_max", 0.3)

    cs_cfg = config.get("conservativeness_sweep", {})
    settings["cs_robust_param_rho_max"] = cs_cfg.get("robust_param_rho_max", 0.1)
    settings["cs_cp_alpha_max"] = cs_cfg.get("cp_alpha_max", 0.3)
    # CP knob is now RELATIVE (tau = fraction of the problem's iter-0 distance d0).
    settings["cs_cp_dist_tol_rel_max"] = cs_cfg.get("cp_dist_tol_rel_max", 1.0)
    settings["cs_cp_dist_tol_rel_min"] = cs_cfg.get("cp_dist_tol_rel_min", 0.1)
    settings["cs_robust_reg_eps_max"] = cs_cfg.get("robust_reg_eps_max", 0.5)
    settings["cs_wrapper_alpha_max"] = cs_cfg.get("wrapper_alpha_max", 0.5)
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
            label_eps=settings["robust_reg_label_eps"],
            budget_frac=settings["robust_reg_budget_frac"],
            K=settings["robust_reg_K"],
            seed=seed,
            rho=0.0,
            embedding_mode=embedding_mode,
            rf_alpha=rf_alpha,
            uncertainty_set=settings["uncertainty_set"],
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
            scenario_source=settings["wrapper_scenario_source"],
            uncertainty_set=settings["uncertainty_set"],
            robustify_objective=settings["wrapper_robustify_objective"],
        ),
        # Single driver; gastric (multiple toxicity constraints over many
        # patients) auto-selects coherent separation with the shared alpha as the
        # feasibility coverage cap.
        "cp": _cp_solver(settings, model_type, model_params, settings["alpha"]),
    }
    return solvers


def _cp_solver(settings, model_type, model_params, cp_alpha=0.0,
               cp_dist_tol_override=None, cp_dist_tol_rel=None):
    """Build the CP solver partial. Single-lever CP: ``cp_alpha`` is pinned at 0 (the
    coverage cap is not a tunable) -- cuts that would break a training anchor are
    evicted/rolled back (``cut_eviction``), keeping the training set feasible, so
    the distance tolerance is the ONLY robustness knob.

    Prefer ``cp_dist_tol_rel`` (tau): the tolerance becomes tau * d0, where d0 is the
    problem's own iteration-0 worst distance, so ONE tau grid transfers across
    datasets/problems. ``cp_dist_tol_override`` sets the absolute value instead."""
    cp_dist_tol = (settings["cp_dist_tol"] if cp_dist_tol_override is None
                   else cp_dist_tol_override)
    return partial(
        solve_cp, model_type=model_type, model_params=model_params, rho=0.0,
        max_iterations=settings["cp_max_iterations"],
        cp_k_neighbors_frac=settings["cp_k_neighbors_frac"],
        cp_k_neighbors_min=settings["cp_k_neighbors_min"],
        cp_n_candidates=settings["cp_n_candidates"],
        seed=settings["bootstrap_seed"], cp_alpha=0.0,  # single-lever: pinned
        cp_dist_tol=cp_dist_tol, cp_dist_tol_rel=cp_dist_tol_rel,
        cp_anchor_source=settings["cp_anchor_source"],
        cp_n_anchors=settings["cp_n_anchors"],
        cp_anchor_method=settings["cp_anchor_method"],
        cp_distance=settings["cp_distance"],
        cp_trace_path=settings["cp_trace_path"],
        cp_robustify_objective=settings["cp_robustify_objective"],
        cp_eval_mode=settings["cp_eval_mode"],
        cp_nearest_distance=settings["cp_nearest_distance"],
        cp_cut_eviction=settings["cp_cut_eviction"],
        cp_scenario_source=settings["cp_scenario_source"],
        cp_n_scenarios=settings["cp_n_scenarios"],
        cp_d0_quantile=settings["cp_d0_quantile"],
        cp_objective_monotone=settings["cp_objective_monotone"],
        cp_mip_gap=settings["cp_mip_gap"],
        cp_cut_whole_scenario=settings["cp_cut_whole_scenario"],
        cp_uncertainty=settings["uncertainty_set"],
    )


def _constraint_names(constraint_mode):
    if constraint_mode == "all_constraints":
        return ALL_CONSTRAINTS
    if constraint_mode == "dlt_only":
        return DLT_ONLY
    raise ValueError(f"Unknown constraint mode: {constraint_mode}")


def _method_build_map(method, settings, ranges, model_type, model_params,
                      bootstrap_cache, ensembles_cache):
    """Return ``(build, strength_to_knob)`` for a method: ``build(knob)`` -> solver_fn,
    ``strength_to_knob(s in [0,1])`` -> knob (0 = weakest ~ nominal, 1 = strongest).
    ``ranges`` supplies the knob endpoints, so the same mapping serves both
    calibration (calibration ranges) and the conservativeness sweep (wider ranges)."""
    nb = settings["n_bootstrap"]
    seed = settings["bootstrap_seed"]
    em = settings["embedding_mode"]
    rf_alpha = settings["rf_alpha"]

    if method == "wrapper":
        amax = ranges["wrapper_alpha_max"]
        strength_to_knob = lambda s: amax * (1.0 - s)  # s=1 strongest -> alpha_w=0
        build = lambda knob: partial(
            solve_wrapper, model_type=model_type, model_params=model_params,
            n_estimators=nb, alpha=knob, seed=seed, rho=0.0,
            bootstrap_cache=bootstrap_cache, ensembles_cache=ensembles_cache,
            scenario_source=settings["wrapper_scenario_source"],
            uncertainty_set=settings["uncertainty_set"],
            robustify_objective=settings["wrapper_robustify_objective"],
        )
    elif method == "tree_violation":
        amax = ranges["tree_alpha_max"]
        strength_to_knob = lambda s: amax * (1.0 - s)  # s=1 strongest -> alpha_t=0
        build = lambda knob: partial(
            solve_tree_violation_wrapper, model_type=model_type,
            model_params=model_params, alpha=knob, rho=0.0,
        )
    elif method == "robust_param":
        rho_min = ranges.get("rho_min", 0.0)
        rho_max = ranges["rho_max"]
        strength_to_knob = lambda s: rho_min + (rho_max - rho_min) * s
        build = lambda knob: partial(
            solve_nominal, model_type=model_type, model_params=model_params,
            rho=knob, embedding_mode=em, rf_alpha=rf_alpha,
        )
    elif method == "nominal":
        # No robustness knob: a single reference point (same at every strength).
        strength_to_knob = lambda s: 0.0
        build = lambda knob: partial(
            solve_nominal, model_type=model_type, model_params=model_params,
            rho=0.0, embedding_mode=em, rf_alpha=rf_alpha,
        )
    elif method == "robust_reg":
        eps_max = ranges["robust_reg_eps_max"]
        strength_to_knob = lambda s: eps_max * s        # label-uncertainty radius
        build = lambda knob: partial(
            solve_robust_regression, model_type=model_type, model_params=model_params,
            label_eps=knob, budget_frac=settings["robust_reg_budget_frac"],
            K=settings["robust_reg_K"], seed=seed, rho=0.0,
            embedding_mode=em, rf_alpha=rf_alpha,
            uncertainty_set=settings["uncertainty_set"],
        )
    elif method == "cp":
        # CP's knob is the RELATIVE distance tolerance tau: tolerance = tau * d0, with
        # d0 the problem's own iteration-0 worst distance. tau=1 stops at iteration 0
        # (~nominal, the weak end); tau->0 cuts maximally. Relative units make one grid
        # valid across datasets -- absolute dist_tol does not transfer, because
        # anything above d0 (~0.017 on gastric) is a silent no-op that ties nominal.
        tmax = ranges.get("cp_dist_tol_rel_max", 1.0)
        tmin = ranges.get("cp_dist_tol_rel_min", 0.1)
        strength_to_knob = lambda s: tmax * (1.0 - s) + tmin * s  # s=1 -> tmin (strongest)
        build = lambda knob: _cp_solver(
            settings, model_type, model_params, settings["alpha"],
            cp_dist_tol_rel=knob,
        )
    else:
        raise ValueError(f"Unknown method for knob map: {method}")
    return build, strength_to_knob


def _solver_at_strength(method, strength, settings, model_type, model_params,
                        bootstrap_cache, ensembles_cache):
    """Build a solver at a fixed conservativeness ``strength`` in [0,1] (no
    calibration), using the wider conservativeness-sweep knob ranges."""
    ranges = {
        "wrapper_alpha_max": settings["cs_wrapper_alpha_max"],
        "tree_alpha_max": settings["calib_tree_alpha_max"],
        "rho_min": 0.0, "rho_max": settings["cs_robust_param_rho_max"],
        "robust_reg_eps_max": settings["cs_robust_reg_eps_max"],
        "cp_alpha_max": settings["cs_cp_alpha_max"],
        "cp_dist_tol_rel_max": settings["cs_cp_dist_tol_rel_max"],
        "cp_dist_tol_rel_min": settings["cs_cp_dist_tol_rel_min"],
    }
    build, s2k = _method_build_map(
        method, settings, ranges, model_type, model_params,
        bootstrap_cache, ensembles_cache,
    )
    return build(s2k(strength))


def _solver_at_knob(method, knob, settings, model_type, model_params,
                    bootstrap_cache, ensembles_cache):
    """Build a solver at a fixed NATIVE robustness knob (CV theta*, or theta*xfactor
    for the centered Pareto). nominal ignores the knob."""
    ranges = {
        "wrapper_alpha_max": settings["cs_wrapper_alpha_max"],
        "tree_alpha_max": settings["calib_tree_alpha_max"],
        "rho_min": 0.0, "rho_max": settings["cs_robust_param_rho_max"],
        "robust_reg_eps_max": settings["cs_robust_reg_eps_max"],
        "cp_alpha_max": settings["cs_cp_alpha_max"],
        "cp_dist_tol_rel_max": settings["cs_cp_dist_tol_rel_max"],
        "cp_dist_tol_rel_min": settings["cs_cp_dist_tol_rel_min"],
    }
    build, _ = _method_build_map(
        method, settings, ranges, model_type, model_params,
        bootstrap_cache, ensembles_cache,
    )
    return build(knob)


def _make_calibrated_solver(method, sub, settings, calib_contexts, model_type,
                            model_params, bootstrap_cache, ensembles_cache):
    """Calibrate a baseline's robustness knob on ``sub`` training contexts and
    return a solver_fn with that knob baked in."""
    if method == "robust_param" and settings["calib_rho_min"] >= settings["calib_rho_max"]:
        raise ValueError(
            f"calibration.rho_min ({settings['calib_rho_min']}) must be "
            f"< rho_max ({settings['calib_rho_max']})"
        )
    ranges = {
        "wrapper_alpha_max": settings["calib_wrapper_alpha_max"],
        "tree_alpha_max": settings["calib_tree_alpha_max"],
        "rho_min": settings["calib_rho_min"], "rho_max": settings["calib_rho_max"],
        "robust_reg_eps_max": settings["calib_robust_reg_eps_max"],
        "cp_alpha_max": settings["cs_cp_alpha_max"],
        "cp_dist_tol_rel_max": settings["cs_cp_dist_tol_rel_max"],
        "cp_dist_tol_rel_min": settings["cs_cp_dist_tol_rel_min"],
    }
    build, strength_to_knob = _method_build_map(
        method, settings, ranges, model_type, model_params,
        bootstrap_cache, ensembles_cache,
    )

    target = settings["alpha"]
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


def build_table6_df(instance, collected, per_mode_mask, per_mode_given,
                    methods, modes, n_test):
    """Build the Table-6 DataFrame from already-collected outcomes on a supplied
    per-mode evaluation cohort. Factored out of ``run_chemo_robust`` pass 2 so the
    conservativeness sweep can score every (method, strength) cell on ONE shared
    samestore cohort instead of a cohort recomputed per cell."""
    all_rows = []
    for method in methods:
        for constraint_mode in modes:
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
    return pd.DataFrame(all_rows)


def run_chemo_robust(config, args, cv_configs=None, gt_configs=None,
                     subsample_frac=None, subsample_seed=None,
                     write_output=True, tox_ub=None, conservativeness=None,
                     collect_only=False, cv_knobs=None, pareto_center_cv=False):
    settings = _resolve_run_settings(config, args)
    instance = gastric_cancer(
        fixed_constraint_configs=cv_configs if cv_configs else None,
        fixed_gt_ensemble_configs=gt_configs if gt_configs else None,
        train_subsample_frac=subsample_frac,
        subsample_seed=subsample_seed,
        tox_ub=tox_ub,
    )
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
    # A fixed-strength conservativeness sweep bypasses calibration entirely.
    if conservativeness is not None:
        calibrate = False
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
        if "wrapper" in settings["methods_to_run"]:
            # Pre-train the shared bootstrap ensembles once so calibration grid
            # evaluations (which change only the knob, not the models) reuse them.
            # (robust_reg no longer uses bootstrap ensembles.)
            ensembles_cache, _ = train_bootstrap_ensembles_for_instance(
                instance, model_type, model_params,
                settings["n_bootstrap"], settings["bootstrap_seed"], coherent_cache,
            )

    # theta* is stored per (method, coherence) cell; pick the cell matching the
    # uncertainty set this run is actually using. lookup_knob falls back to a bare
    # `method` key so pre-per-cell knobs JSONs still load.
    _coherent = bool(getattr(settings.get("uncertainty_set"), "coherent", True))

    def _resolve_solver(method, sub):
        if conservativeness is not None:
            if pareto_center_cv and cv_knobs is not None:
                # Centered Pareto: knob = CV theta* x factor (nominal has no knob).
                theta = lookup_knob(cv_knobs, method, _coherent, 0.0)
                return _solver_at_knob(
                    method, theta * conservativeness, settings, model_type,
                    model_params, coherent_cache, ensembles_cache,
                )
            # Fixed-strength sweep: set each method's own knob directly (no calibration).
            return _solver_at_strength(
                method, conservativeness, settings, model_type, model_params,
                coherent_cache, ensembles_cache,
            )
        # CV-calibrated comparison: build each method at its CV-selected theta*.
        if settings["calibration_method"] == "cv" and cv_knobs is not None:
            if method == "nominal":
                return solvers[method]
            theta = lookup_knob(cv_knobs, method, _coherent)
            if theta is not None:
                return _solver_at_knob(
                    method, theta, settings, model_type, model_params,
                    coherent_cache, ensembles_cache,
                )
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
            _, feasible_mask, mean_time, sd_time, full_outcomes, _ = evaluate_prescribed_table6(
                solver_fn,
                sub,
                max_test_rows=settings["max_test_rows"],
                method_name=method,
                constraint_mode=constraint_mode,
                prescriptions_dir=settings["prescriptions_dir"],
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

    # The conservativeness sweep computes ONE shared samestore across all
    # (method, strength) cells, so return the raw collected outcomes + per-method
    # masks and let the caller intersect + build rows (no per-cell cohort).
    if collect_only:
        return instance, collected, mode_masks, n_test

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
    df = build_table6_df(
        instance, collected, per_mode_mask, per_mode_given,
        settings["methods_to_run"], settings["constraint_modes"], n_test,
    )
    if write_output:
        os.makedirs("results/gastric", exist_ok=True)
        df.to_csv(settings["output_path"], index=False)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\nSaved to {settings['output_path']}")
    return df


def run_chemo_robust_realizations(config, args, cv_configs=None, gt_configs=None,
                                  cv_knobs=None):
    """Label-noise robustness probe: repeat the whole prescribe/evaluate pipeline
    over R independent training subsamples (m-out-of-n, without replacement) and
    report the *distribution* of GT-feasibility per method.

    The GT ensemble oracle is unchanged across realizations (it is fit on the full
    clean cohort); only the rows that FIT the embedded/robust constraint models are
    resampled. Robust methods should show a higher worst-case and lower SD on the
    ``all_constraints`` conjunction row than ``nominal``.
    """
    import pandas as pd
    from src.evaluation.chemo_metrics import aggregate_realizations

    settings = _resolve_run_settings(config, args)  # for shared-cohort row building
    n_real = args.n_realizations
    base_seed = config["uncertainty"].get("bootstrap_seed", 42)
    # Two sweep axes, both with common random numbers -- the subsample seed depends
    # only on the realization, not on rhs/frac -- so every cell is paired across the
    # same training draws. rhs: toxicity upper bound; frac: subsample fraction
    # (scarcity). Either can be a single value (default) or a grid.
    rhs_grid = args.rhs_grid if args.rhs_grid else [None]
    frac_grid = args.frac_grid if args.frac_grid else [args.subsample_frac]
    # Centered Pareto: the "conservativeness" axis is a set of multiplicative factors
    # applied to each method's CV theta* (knob = theta* x factor), instead of the
    # shared [0,1] strength grid. Reuses the shared-samestore cons-sweep machinery.
    center_cv = getattr(args, "pareto_center_cv", False) and cv_knobs is not None
    if center_cv:
        cons_grid = settings["pareto_center_factors"]
    else:
        cons_grid = args.conservativeness_grid if args.conservativeness_grid else [None]
    sweeping_rhs = args.rhs_grid is not None
    sweeping_frac = args.frac_grid is not None
    sweeping_cons = args.conservativeness_grid is not None or center_cv
    sweeping = sweeping_rhs or sweeping_frac or sweeping_cons

    print("=" * 60)
    print(
        f"LABEL-NOISE ROBUSTNESS PROBE: {n_real} realizations, "
        f"frac_grid={frac_grid}, rhs_grid={rhs_grid}, cons_grid={cons_grid}"
    )
    print("=" * 60)

    def _tag(df_r, r, sub_seed, rhs, frac, cons):
        df_r = df_r.copy()
        df_r["realization"] = r
        df_r["subsample_seed"] = sub_seed
        df_r["rhs"] = rhs if rhs is not None else "default"
        df_r["frac"] = frac if frac is not None else "full"
        df_r["strength"] = cons if cons is not None else "calibrated"
        return df_r

    rows = []
    for frac in frac_grid:
        for rhs in rhs_grid:
            for r in range(n_real):
                sub_seed = base_seed + 1000 * (r + 1)  # CRN: same across all axes
                if sweeping_cons:
                    # Two passes so every (method, strength) cell is scored on ONE
                    # shared samestore cohort (intersection over all method x strength
                    # cells): equal strength is NOT the same frontier point, so a
                    # per-cell cohort would make the frontiers incomparable.
                    stash = {}  # cons -> (instance, collected, mode_masks, n_test)
                    for cons in cons_grid:
                        print(f"\n{'#' * 60}\n# [collect] strength={cons} frac={frac} "
                              f"rhs={rhs} realization {r + 1}/{n_real} "
                              f"(subsample_seed={sub_seed})\n{'#' * 60}")
                        stash[cons] = run_chemo_robust(
                            config, args, cv_configs=cv_configs, gt_configs=gt_configs,
                            subsample_frac=frac, subsample_seed=sub_seed,
                            write_output=False, tox_ub=rhs, conservativeness=cons,
                            collect_only=True, cv_knobs=cv_knobs,
                            pareto_center_cv=center_cv,
                        )
                    instance0, _, _, n_test0 = stash[cons_grid[0]]
                    modes = settings["constraint_modes"]
                    per_mode_mask, per_mode_given = {}, {}
                    for mode in modes:
                        all_masks = []
                        for cons in cons_grid:
                            all_masks.extend(stash[cons][2].get(mode, []))
                        if not all_masks:
                            continue
                        shared = samestore_eval_mask(*all_masks)
                        n_eval = int(shared.sum())
                        print(f"Shared samestore cohort ({mode}, across all "
                              f"strengths): {n_eval}/{n_test0} test rows")
                        if n_eval == 0:
                            print("  WARNING: empty shared cohort; skipping this "
                                  "mode for this realization.")
                            continue
                        per_mode_mask[mode] = shared
                        per_mode_given[mode] = evaluate_given_table6(instance0, shared)
                    for cons in cons_grid:
                        inst_c, collected_c, _, n_test_c = stash[cons]
                        df_c = build_table6_df(
                            inst_c, collected_c, per_mode_mask, per_mode_given,
                            settings["methods_to_run"], modes, n_test_c,
                        )
                        rows.append(_tag(df_c, r, sub_seed, rhs, frac, cons))
                else:
                    print(f"\n{'#' * 60}\n# frac={frac} rhs={rhs} "
                          f"realization {r + 1}/{n_real} "
                          f"(subsample_seed={sub_seed})\n{'#' * 60}")
                    df_r = run_chemo_robust(
                        config, args, cv_configs=cv_configs, gt_configs=gt_configs,
                        subsample_frac=frac, subsample_seed=sub_seed,
                        write_output=False, tox_ub=rhs, conservativeness=None,
                        cv_knobs=cv_knobs,
                    )
                    rows.append(_tag(df_r, r, sub_seed, rhs, frac, None))

    df_long = pd.concat(rows, ignore_index=True)
    os.makedirs("results/gastric", exist_ok=True)
    tag = f"_{args.output_tag}" if getattr(args, "output_tag", None) else ""
    suffix = tag + ("_sweep" if sweeping else "")
    long_path = f"results/gastric/chemo_robust_realizations{suffix}.csv"
    df_long.to_csv(long_path, index=False)

    group_cols = ((["frac"] if sweeping_frac else [])
                  + (["rhs"] if sweeping_rhs else [])
                  + (["strength"] if sweeping_cons else [])) or None
    summary = aggregate_realizations(df_long, extra_group_cols=group_cols)
    summary_path = f"results/gastric/chemo_robust_robustness_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY (distribution across realizations)")
    print("=" * 60)
    # Highlight the headline conjunction row.
    joint = summary[summary["outcome"] == "all_constraints"]
    if not joint.empty:
        print("\nJoint toxicity feasibility (all_constraints conjunction):")
        print(joint.to_string(index=False))
    print(f"\nSaved per-realization rows to {long_path}")
    print(f"Saved robustness summary to {summary_path}")
    return summary


def _cs_ranges(settings):
    """Knob-range dict for ``_method_build_map`` (only ``build(knob)`` is used in CV;
    the ranges just need to exist so construction doesn't KeyError)."""
    return {
        "wrapper_alpha_max": settings["cs_wrapper_alpha_max"],
        "tree_alpha_max": settings["calib_tree_alpha_max"],
        "rho_min": 0.0, "rho_max": settings["cs_robust_param_rho_max"],
        "robust_reg_eps_max": settings["cs_robust_reg_eps_max"],
        "cp_alpha_max": settings["cs_cp_alpha_max"],
        "cp_dist_tol_rel_max": settings["cs_cp_dist_tol_rel_max"],
        "cp_dist_tol_rel_min": settings["cs_cp_dist_tol_rel_min"],
    }


def run_cv_calibration(config, args, cv_configs=None, gt_configs=None):
    """Stage 1: select each method's single robustness knob by held-out CV (temporal
    folds for gastric, KFold for synthetic), writing ``*_robustness_knobs.json`` +
    a resumable scores checkpoint. See ``src/methods/cv_calibrate``."""
    import dataclasses
    from src.methods.cv_calibrate import (
        make_folds, make_cv_oracle, cv_score_knob, select_knob_cv,
        load_score_checkpoint, append_score, write_knobs, knob_key,
    )
    settings = _resolve_run_settings(config, args)
    cvc = config.get("cv_calibration", {})
    # run_chemo_robust.py is the gastric script (it always builds gastric, regardless
    # of config.data.type); the synthetic robustness-parameter CV lives in run_sweep.py.
    prefix = "gastric"
    scores_path = f"results/cv/{prefix}_robustness_cv_scores.csv"
    knobs_path = f"results/cv/{prefix}_robustness_knobs.json"
    if getattr(args, "refresh_cv", False):
        for p in (scores_path, knobs_path):
            if os.path.exists(p):
                os.remove(p)
                print(f"[cv] removed {p} (--refresh-cv)")

    instance = gastric_cancer(
        fixed_constraint_configs=cv_configs, fixed_gt_ensemble_configs=gt_configs,
    )
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]

    folds = make_folds(
        instance, cvc.get("fold_scheme", "auto"),
        tuple(cvc.get("fold_cutoffs", (2004, 2005, 2006, 2007))),
        int(cvc.get("n_kfold", 4)), settings["bootstrap_seed"],
    )
    oracle = make_cv_oracle(instance, gt_specs=gt_configs)
    contextual = bool(instance.context_var_indices)
    constraint_names = ALL_CONSTRAINTS if contextual else None
    os_tol = float(cvc.get("os_tolerance_frac", 0.1))
    ranges = _cs_ranges(settings)
    print(f"\n[cv] problem={prefix}, folds={len(folds)}, contextual={contextual}, "
          f"os_tolerance_frac={os_tol}, oracle_sense={oracle.objective_sense}", flush=True)

    ckpt = load_score_checkpoint(scores_path)

    def make_scorer(method, build):
        def _score(knob):
            key = (method, float(knob))
            if key in ckpt:
                return ckpt[key]
            feas, obj = cv_score_knob(build, knob, folds, oracle, instance,
                                      constraint_names, contextual)
            append_score(scores_path, method, knob, feas, obj)
            ckpt[key] = (feas, obj)
            return feas, obj
        return _score

    # Nominal CV baseline (no robustness knob) -> objective budget reference.
    nom_build, _ = _method_build_map("nominal", settings, ranges, model_type,
                                     model_params, None, None)
    nom_feas, nom_obj = make_scorer("nominal", nom_build)(0.0)
    print(f"[cv] nominal: feas={nom_feas:.3f} obj={nom_obj:.3f}", flush=True)

    knobs = {"nominal": 0.0}
    grids = cvc.get("knob_grids", {})
    # theta* is calibrated PER (method, coherence) cell -- see cv_calibrate.knob_key.
    # The scores CSV is keyed by a free-text `method` column and is resumable, so
    # this doubles stage-1 work but never redoes a cell.
    cells = cvc.get("coherence_cells", [True, False])
    # CP first (slowest / most likely to fail), then the rest.
    order = [m for m in ("cp", "robust_reg", "wrapper")
             if m in settings["methods_to_run"] and m in grids]
    for coherent in cells:
        cell_settings = dict(settings)
        cell_settings["uncertainty_set"] = dataclasses.replace(
            settings["uncertainty_set"], coherent=bool(coherent)
        )
        print(f"\n[cv] === coherence cell: coherent={coherent} ===", flush=True)
        for method in order:
            key = knob_key(method, coherent)
            build, _ = _method_build_map(method, cell_settings, ranges, model_type,
                                         model_params, None, None)
            theta, _rows = select_knob_cv(
                build, grids[method], folds, oracle, instance, os_tol, nom_obj,
                constraint_names=constraint_names, contextual=contextual, method=key,
                score_fn=make_scorer(key, build),
            )
            knobs[key] = float(theta)

    write_knobs(knobs_path, knobs)
    return knobs


_DEFAULT_CV_CONFIGS = "results/cv/gastric_selected_configs.json"
_DEFAULT_GT_CONFIGS = "results/cv/gastric_gt_ensemble_configs.json"


def main():
    parser = argparse.ArgumentParser(description="Chemo robust method comparison")
    parser.add_argument("--quick", action="store_true", help="Small local smoke run")
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=1,
        help=(
            "Number of training-subsample realizations for the label-noise "
            "robustness probe. 1 (default) runs the standard single Table 6."
        ),
    )
    parser.add_argument(
        "--subsample-frac",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Fraction of training rows drawn WITHOUT replacement to fit the "
            "embedded/robust constraint models each realization (m-out-of-n). "
            "Default None = full training set. Only the fit data is resampled; "
            "the GT ensemble oracle is unchanged."
        ),
    )
    parser.add_argument(
        "--rhs-grid",
        type=float,
        nargs="+",
        default=None,
        metavar="UB",
        help=(
            "Toxicity upper-bound values to sweep (overrides the paper's 0.6 for "
            "both the embedded constraint and the GT threshold). Crossed with the "
            "realization loop using common random numbers. E.g. --rhs-grid 0.3 0.4 0.5 0.6."
        ),
    )
    parser.add_argument(
        "--frac-grid",
        type=float,
        nargs="+",
        default=None,
        metavar="FRAC",
        help=(
            "Subsample fractions to sweep (the scarcity axis), crossed with the "
            "realization loop using common random numbers. Overrides --subsample-frac. "
            "E.g. --frac-grid 0.3 0.4 0.5 0.6 0.7 0.8."
        ),
    )
    parser.add_argument(
        "--conservativeness-grid",
        type=float,
        nargs="+",
        default=None,
        metavar="S",
        help=(
            "Fixed-threshold Pareto sweep: strengths in [0,1] (0=weakest ~nominal, "
            "1=strongest). Each method's OWN knob is set per strength (no "
            "calibration): robust_param->rho, cp->alpha, robust_reg->label_eps, "
            "wrapper->alpha. Traces per-method OS vs worst-case-feasibility frontiers "
            "to separate robustness from mere conservatism. E.g. 0 0.25 0.5 0.75 1."
        ),
    )
    parser.add_argument(
        "--cp-robustify-objective",
        choices=["true", "false"],
        default=None,
        help="CP ablation: override methods.cp.robustify_objective (default from config).",
    )
    parser.add_argument(
        "--cp-eval-mode",
        choices=["global", "per_anchor_nearest"],
        default=None,
        help="CP ablation: override methods.cp.eval_mode (default from config).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        metavar="TAG",
        help="Suffix for realization/summary output filenames (keeps ablation variants separate).",
    )
    parser.add_argument(
        "--calibrate-cv",
        action="store_true",
        help=(
            "Stage 1: select each method's robustness knob by held-out CV (temporal "
            "folds gastric / KFold synthetic) and write results/cv/*_robustness_knobs.json "
            "+ a resumable scores checkpoint. Does not run the Table 6 comparison."
        ),
    )
    parser.add_argument(
        "--refresh-cv",
        action="store_true",
        help="With --calibrate-cv, delete the scores checkpoint + knobs JSON first (clean recompute).",
    )
    parser.add_argument(
        "--pareto-center-cv",
        action="store_true",
        help=(
            "Stage 2 Pareto: center each method's knob grid on its CV-selected theta* "
            "(results/cv/*_robustness_knobs.json), scaled by cv_calibration.pareto_center_factors."
        ),
    )
    parser.add_argument(
        "--cv-configs",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Path to gastric_selected_configs.json from run_cv.py. "
            f"Defaults to {_DEFAULT_CV_CONFIGS} if that file exists."
        ),
    )
    parser.add_argument(
        "--gt-cv-configs",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Path to gastric_gt_ensemble_configs.json from run_cv.py --ensemble. "
            f"Defaults to {_DEFAULT_GT_CONFIGS} if that file exists."
        ),
    )
    parser.add_argument(
        "--no-cv-configs",
        action="store_true",
        help=(
            "Disable auto-loading of CV configs. Uses fixed paper constraint models "
            "(GASTRIC_EMBED_CONFIGS) and GT ensemble specs (GT_ENSEMBLE_SPECS) instead."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # --- Resolve constraint CV configs ---
    cv_configs = None
    if args.no_cv_configs:
        print("CV configs disabled (--no-cv-configs); using fixed paper models.")
    elif args.cv_configs:
        with open(args.cv_configs, "r") as f:
            cv_configs = json.load(f)
        print(f"Loaded constraint CV configs from {args.cv_configs}")
    elif os.path.exists(_DEFAULT_CV_CONFIGS):
        with open(_DEFAULT_CV_CONFIGS, "r") as f:
            cv_configs = json.load(f)
        print(f"Auto-loaded constraint CV configs from {_DEFAULT_CV_CONFIGS}")
    else:
        print(
            f"No CV configs found at {_DEFAULT_CV_CONFIGS}; "
            "using fixed paper constraint models. Run experiments/run_cv.py first."
        )

    # --- Resolve GT ensemble configs ---
    gt_configs = None
    if not args.no_cv_configs:
        gt_path = args.gt_cv_configs or _DEFAULT_GT_CONFIGS
        if args.gt_cv_configs and os.path.exists(args.gt_cv_configs):
            with open(args.gt_cv_configs, "r") as f:
                gt_configs = json.load(f)
            print(f"Loaded GT ensemble configs from {args.gt_cv_configs}")
        elif not args.gt_cv_configs and os.path.exists(_DEFAULT_GT_CONFIGS):
            with open(_DEFAULT_GT_CONFIGS, "r") as f:
                gt_configs = json.load(f)
            print(f"Auto-loaded GT ensemble configs from {_DEFAULT_GT_CONFIGS}")
        else:
            print(
                f"No GT ensemble configs at {gt_path}; "
                "using fixed paper GT ensemble (GT_ENSEMBLE_SPECS). "
                "Run experiments/run_cv.py --ensemble to generate them."
            )

    # --- Auto-load CV-selected robustness knobs (stage 2 consumes them) ---
    cv_knobs = None
    _knobs_path = "results/cv/gastric_robustness_knobs.json"
    if os.path.exists(_knobs_path):
        with open(_knobs_path) as f:
            cv_knobs = json.load(f)
        print(f"Auto-loaded CV robustness knobs from {_knobs_path}: {cv_knobs}")

    if args.calibrate_cv:
        run_cv_calibration(config, args, cv_configs=cv_configs, gt_configs=gt_configs)
    elif (args.n_realizations > 1 or args.subsample_frac is not None
            or args.rhs_grid or args.frac_grid or args.conservativeness_grid
            or args.pareto_center_cv):
        run_chemo_robust_realizations(
            config, args, cv_configs=cv_configs, gt_configs=gt_configs, cv_knobs=cv_knobs,
        )
    else:
        run_chemo_robust(config, args, cv_configs=cv_configs, gt_configs=gt_configs,
                         cv_knobs=cv_knobs)


if __name__ == "__main__":
    main()

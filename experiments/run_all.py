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
from src.methods.uncertainty import uncertainty_set_from_config
from src.evaluation.metrics import evaluate_all


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config, cv_configs=None, seed=None, knobs=None):
    """Run all methods and evaluate.

    Parameters
    ----------
    seed : optional int seeding the synthetic data draw. ``None`` reuses
        ``uncertainty.bootstrap_seed``, which is what made every "realization" of
        the noise sweep the *same* dataset -- pass distinct seeds for genuine
        independent draws.
    knobs : optional ``{method: theta*}`` from
        ``results/cv/synthetic_robustness_knobs.json``. Each method's robustness
        knob is taken from here when present, so the synthetic runs sit at their
        CV-calibrated operating points rather than at config defaults (gastric
        already works this way).
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
            seed=(seed if seed is not None
                  else config["uncertainty"].get("bootstrap_seed", 42)),
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

    # CV-calibrated operating points override the config defaults, so every method
    # is compared at the knob its own held-out CV picked rather than at whatever
    # config.yaml happens to hold.
    knobs = knobs or {}
    if knobs:
        print(f"    CV-calibrated knobs: "
              f"{ {k: round(v, 4) for k, v in knobs.items()} }")

    solver_fns = {}

    solver_fns["nominal"] = partial(
        solve_nominal, model_type=model_type, model_params=model_params, rho=0.0
    )

    # robust_param is intentionally NOT run: it is commented out of the gastric
    # methods_to_run (config.yaml) and absent from make_paper_figures.METHODS, so
    # its rows were computed and then silently dropped from every figure.
    # methods.robust_param.rho is still read by run_chemo_robust.py.

    robust_reg_cfg = config["methods"].get("robust_reg", {})
    solver_fns["robust_reg"] = partial(
        solve_robust_regression,
        model_type=model_type,
        model_params=model_params,
        label_eps=knobs.get("robust_reg", robust_reg_cfg.get("label_eps", 0.1)),
        budget_frac=robust_reg_cfg.get("budget_frac", 0.5),
        K=robust_reg_cfg.get("K", 5),
        seed=bootstrap_seed,
        rho=0.0,
        uncertainty_set=uncertainty_set_from_config(config),
    )

    wrapper_cfg = config["methods"]["wrapper"]
    solver_fns["wrapper"] = partial(
        solve_wrapper,
        model_type=model_type,
        model_params=model_params,
        rho=0.0,
        n_estimators=wrapper_cfg.get("n_estimators", n_bootstrap),
        alpha=knobs.get("wrapper", wrapper_cfg["alpha"]),
        seed=bootstrap_seed,
        bootstrap_cache=bootstrap_cache,
        # Same D, same seeded draw sequence CP separates over -- the wrapper's P
        # models are a prefix of CP's bank, so alpha=0 and tau->0 are comparable.
        scenario_source=wrapper_cfg.get("scenario_source", "noise"),
        uncertainty_set=uncertainty_set_from_config(config),
        robustify_objective=wrapper_cfg.get("robustify_objective", False),
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
        # Relative distance tolerance tau -- CP's robustness knob on the basic
        # (synthetic) path too. Without it the basic separation cuts every
        # violation > 1e-6, so CP has no lever and over-cuts to no solution.
        cp_dist_tol_rel=knobs.get("cp", cp_cfg.get("dist_tol_rel")),
        cp_alpha=0.0,
        cp_cut_eviction=cp_cfg.get("cut_eviction", "evict_slack"),
        # Separate over a FIXED bank of draws from the shared uncertainty set D,
        # so the worst violation over the bank is monotone across iterations.
        cp_scenario_source=cp_cfg.get("scenario_source", "noise"),
        cp_n_scenarios=cp_cfg.get("n_scenarios", 200),
        cp_d0_quantile=cp_cfg.get("d0_quantile", 0.9),
        cp_objective_monotone=cp_cfg.get("objective_monotone", False),
        cp_mip_gap=float(cp_cfg.get("mip_gap", 1e-4)),
        cp_cut_whole_scenario=cp_cfg.get("cut_whole_scenario", True),
        cp_uncertainty=uncertainty_set_from_config(config),
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
            # Raw f_true(x*). The signed margin is this minus the constraint rhs;
            # unlike worst_violation it is NOT clipped at 0, so it retains slack on
            # feasible points. metrics.py only fills it for the single-x* case
            # (n_obs == 1), i.e. non-contextual synthetic -- None on gastric.
            "true_constraint_value": ev.true_constraint_value,
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
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the synthetic data draw (default: bootstrap_seed)")
    parser.add_argument(
        "--knobs", type=str, default=None, metavar="PATH",
        help=("Path to a *_robustness_knobs.json from --calibrate-cv. Defaults to "
              "results/cv/synthetic_robustness_knobs.json when it exists; pass "
              "'none' to force config.yaml defaults."),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cv_configs = None
    if args.cv_configs:
        with open(args.cv_configs, "r") as f:
            cv_configs = json.load(f)
        print(f"Loaded CV configs from {args.cv_configs}")

    # Auto-load the CV-calibrated knobs the way the gastric runner does, so a bare
    # `run_all.py` compares methods at their calibrated operating points.
    knobs = None
    knobs_path = args.knobs or "results/cv/synthetic_robustness_knobs.json"
    if args.knobs != "none" and os.path.exists(knobs_path):
        with open(knobs_path, "r") as f:
            knobs = json.load(f)
        print(f"Loaded CV knobs from {knobs_path}: {knobs}")

    run_experiment(cfg, cv_configs=cv_configs, seed=args.seed, knobs=knobs)

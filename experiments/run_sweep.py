"""
Sweep over Gamma values to generate price-of-robustness curves.

Also hosts the SYNTHETIC robustness-parameter CV (--calibrate-cv) and the synthetic
CV-centered Pareto (--pareto), the synthetic counterparts of the gastric pipeline in
run_chemo_robust.py.
"""

import json
import yaml
import numpy as np
import pandas as pd
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_all import load_config, run_experiment


# ---------------------------------------------------------------------------
# Synthetic robustness-parameter CV + CV-centered Pareto
# ---------------------------------------------------------------------------
def _synth_build(method, config, model_type, model_params, seed):
    """Return ``build(knob) -> solver_fn`` for a synthetic method. Single knob per
    method (CP dist_tol, robust_reg label_eps, wrapper alpha); nominal ignores it.
    CP is single-lever (cp_alpha=0) like gastric."""
    from src.methods.nominal import solve_nominal
    from src.methods.robust_regression import solve_robust_regression
    from src.methods.wrapper import solve_wrapper
    from src.methods.cp import solve_cp
    unc = config["uncertainty"]
    rr = config["methods"].get("robust_reg", {})
    cp = config["methods"].get("cp", {})
    if method == "nominal":
        return lambda knob: partial(solve_nominal, model_type=model_type,
                                    model_params=model_params, rho=0.0)
    if method == "robust_reg":
        return lambda knob: partial(
            solve_robust_regression, model_type=model_type, model_params=model_params,
            label_eps=knob, budget_frac=rr.get("budget_frac", 0.5), K=rr.get("K", 5),
            seed=seed, rho=0.0)
    if method == "wrapper":
        return lambda knob: partial(
            solve_wrapper, model_type=model_type, model_params=model_params,
            n_estimators=unc.get("n_bootstrap", 20), alpha=knob, seed=seed, rho=0.0)
    if method == "cp":
        return lambda knob: partial(
            solve_cp, model_type=model_type, model_params=model_params, rho=0.0,
            max_iterations=cp.get("max_iterations", 20),
            cp_k_neighbors_frac=unc.get("cp_k_neighbors_frac", 0.1),
            cp_k_neighbors_min=unc.get("cp_k_neighbors_min", 1),
            cp_n_candidates=unc.get("cp_n_candidates", 20), seed=seed,
            # Relative knob (tau): tolerance = tau * d0 from this problem's own
            # iter-0 worst violation, so the SAME grid works here and on gastric.
            # (The basic path previously ignored dist_tol entirely -- it cut every
            # violation > 1e-6 -- so synthetic CP had no robustness lever at all.)
            cp_alpha=0.0, cp_dist_tol_rel=knob,
            cp_cut_eviction=cp.get("cut_eviction", "evict_slack"))
    raise ValueError(f"unknown synthetic method {method}")


def _synth_instance(config, seed=None):
    from src.data.generate import synthetic_nonlinear
    d = config["data"]
    return synthetic_nonlinear(
        n_train=d["n_train"], n_features=d["n_features"], noise_std=d["noise_std"],
        seed=seed if seed is not None else config["uncertainty"].get("bootstrap_seed", 42),
    )


def run_cv_calibration_synthetic(config, methods=None, refresh=False):
    """Stage 1 for synthetic: KFold folds + proxy-ensemble oracle -> theta* per
    method. Writes results/cv/synthetic_robustness_knobs.json (+ scores checkpoint)."""
    from src.methods.cv_calibrate import (
        make_folds, make_cv_oracle, select_knob_cv, cv_score_knob,
        load_score_checkpoint, append_score, write_knobs,
    )
    os.makedirs("results/cv", exist_ok=True)
    scores_path = "results/cv/synthetic_robustness_cv_scores.csv"
    knobs_path = "results/cv/synthetic_robustness_knobs.json"
    if refresh:
        for p in (scores_path, knobs_path):
            if os.path.exists(p):
                os.remove(p)
    cvc = config.get("cv_calibration", {})
    seed = config["uncertainty"].get("bootstrap_seed", 42)
    inst = _synth_instance(config)
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    folds = make_folds(inst, "kfold", n_kfold=int(cvc.get("n_kfold", 4)), seed=seed)
    oracle = make_cv_oracle(inst)          # proxy ensemble on training labels
    os_tol = float(cvc.get("os_tolerance_frac", 0.1))
    grids = cvc.get("knob_grids", {})
    methods = methods or ["cp", "robust_reg"]
    print(f"[cv-synth] folds={len(folds)}, os_tol={os_tol}, sense={oracle.objective_sense}",
          flush=True)

    ckpt = load_score_checkpoint(scores_path)

    def make_scorer(method, build):
        def _score(knob):
            key = (method, float(knob))
            if key in ckpt:
                return ckpt[key]
            feas, obj = cv_score_knob(build, knob, folds, oracle, inst,
                                      constraint_names=None, contextual=False)
            append_score(scores_path, method, knob, feas, obj)
            ckpt[key] = (feas, obj)
            return feas, obj
        return _score

    nom_build = _synth_build("nominal", config, model_type, model_params, seed)
    _, nom_obj = make_scorer("nominal", nom_build)(0.0)
    knobs = {"nominal": 0.0}
    for method in [m for m in ("cp", "robust_reg", "wrapper") if m in methods and m in grids]:
        build = _synth_build(method, config, model_type, model_params, seed)
        theta, _ = select_knob_cv(build, grids[method], folds, oracle, inst, os_tol,
                                  nom_obj, constraint_names=None, contextual=False,
                                  method=method, score_fn=make_scorer(method, build))
        knobs[method] = float(theta)
    write_knobs(knobs_path, knobs)
    return knobs


def run_synthetic_centered_pareto(config, methods=None, n_real=8):
    """Synthetic CV-centered Pareto: per method sweep knob = theta* x factor over
    noise realizations; score feasibility + objective vs the ANALYTIC truth (final
    eval, honest here). Writes results/synthetic/synthetic_pareto.csv."""
    from src.evaluation.metrics import evaluate_all
    knobs_path = "results/cv/synthetic_robustness_knobs.json"
    if not os.path.exists(knobs_path):
        raise SystemExit("run --calibrate-cv first (no synthetic_robustness_knobs.json)")
    knobs = json.load(open(knobs_path))
    cvc = config.get("cv_calibration", {})
    factors = cvc.get("pareto_center_factors", [0.5, 0.75, 1.0, 1.5, 2.0])
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    base_seed = config["uncertainty"].get("bootstrap_seed", 42)
    methods = methods or ["nominal", "robust_reg", "cp"]

    rows = []
    for method in methods:
        theta = knobs.get(method, 0.0)
        grid = [0.0] if method == "nominal" else [theta * f for f in factors]
        for knob in grid:
            build = _synth_build(method, config, model_type, model_params, base_seed)
            feas_draws, obj_draws = [], []
            for r in range(n_real):
                inst = _synth_instance(config, seed=base_seed + 1000 * (r + 1))
                ev = evaluate_all({method: build(knob)}, inst)[0]
                feas_draws.append(ev.feasibility_rate)
                if ev.mean_obj_value is not None and ev.mean_obj_value < 1e6:
                    obj_draws.append(ev.mean_obj_value)
            rows.append({
                "method": method, "knob": knob,
                "worst_case_feas": float(np.min(feas_draws)),
                "mean_feas": float(np.mean(feas_draws)),
                "objective": float(np.mean(obj_draws)) if obj_draws else float("nan"),
            })
            print(f"[synth-pareto] {method} knob={knob:.4g}: worst_feas="
                  f"{rows[-1]['worst_case_feas']:.3f} obj={rows[-1]['objective']:.3f}",
                  flush=True)
    os.makedirs("results/synthetic", exist_ok=True)
    out = "results/synthetic/synthetic_pareto.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved synthetic CV-centered Pareto -> {out}")
    return out


def run_gamma_sweep(gamma_values=None):
    """
    Run the full experiment for each Gamma value.
    Produces a table of results indexed by (method, Gamma).
    """
    if gamma_values is None:
        gamma_values = [1.0, 2.0, 5.0, 10.0, 20.0]

    config = load_config()
    all_rows = []

    for gamma in gamma_values:
        print(f"\n{'#' * 60}")
        print(f"# GAMMA = {gamma}")
        print(f"{'#' * 60}")

        config["uncertainty"]["gamma"] = gamma
        df, _ = run_experiment(config)
        df["gamma"] = gamma
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs("results/synthetic", exist_ok=True)
    combined.to_csv("results/synthetic/sweep_results.csv", index=False)
    print(f"\nSaved sweep results to results/synthetic/sweep_results.csv")

    return combined


def run_noise_sweep(noise_values=None):
    """
    Sweep over label noise levels sigma.
    Shows how each method degrades as noise increases.
    """
    if noise_values is None:
        noise_values = [0.0, 0.05, 0.1, 0.2, 0.5]

    config = load_config()
    all_rows = []

    for sigma in noise_values:
        print(f"\n{'#' * 60}")
        print(f"# NOISE_STD = {sigma}")
        print(f"{'#' * 60}")

        config["data"]["noise_std"] = sigma
        df, _ = run_experiment(config)
        df["noise_std"] = sigma
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs("results/synthetic", exist_ok=True)
    combined.to_csv("results/synthetic/noise_sweep_results.csv", index=False)
    print(f"\nSaved noise sweep to results/synthetic/noise_sweep_results.csv")

    return combined


def plot_gamma_sweep(csv_path="results/synthetic/sweep_results.csv",
                     save_dir="results/synthetic"):
    """Plot price of robustness from Gamma sweep."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    methods = df["method"].unique()
    colors = sns.color_palette("colorblind", len(methods))
    method_colors = dict(zip(methods, colors))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Objective vs Gamma ---
    ax = axes[0]
    for method in methods:
        sub = df[df["method"] == method]
        sub = sub[sub["objective"] < 1e6]  # filter infeasible
        if len(sub) > 0:
            ax.plot(sub["gamma"], sub["objective"],
                    "o-", label=method, color=method_colors[method])
    ax.set_xlabel("$\\Gamma$ (uncertainty budget)")
    ax.set_ylabel("Objective $c^\\top x^*$")
    ax.set_title("Price of Robustness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Feasibility vs Gamma ---
    ax = axes[1]
    for method in methods:
        sub = df[df["method"] == method]
        ax.plot(sub["gamma"], sub["feasibility_rate"],
                "o-", label=method, color=method_colors[method])
    ax.set_xlabel("$\\Gamma$ (uncertainty budget)")
    ax.set_ylabel("Held-out feasibility rate")
    ax.set_title("Robustness vs. Uncertainty Budget")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Feasibility vs Objective (Pareto) ---
    ax = axes[2]
    for method in methods:
        sub = df[df["method"] == method]
        sub = sub[sub["objective"] < 1e6]
        if len(sub) > 0:
            ax.scatter(sub["objective"], sub["feasibility_rate"],
                       label=method, color=method_colors[method],
                       s=60, zorder=3)
            # Connect points in Gamma order
            sub_sorted = sub.sort_values("gamma")
            ax.plot(sub_sorted["objective"],
                    sub_sorted["feasibility_rate"],
                    "-", color=method_colors[method], alpha=0.5)
    ax.set_xlabel("Objective $c^\\top x^*$")
    ax.set_ylabel("Held-out feasibility rate")
    ax.set_title("Robustness--Cost Tradeoff")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gamma_sweep.png"), dpi=150)
    plt.close()
    print(f"Saved gamma sweep plot to {save_dir}/gamma_sweep.png")


def plot_noise_sweep(csv_path="results/synthetic/noise_sweep_results.csv",
                     save_dir="results/synthetic"):
    """Plot degradation under increasing label noise."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    methods = df["method"].unique()
    colors = sns.color_palette("colorblind", len(methods))
    method_colors = dict(zip(methods, colors))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- True feasibility vs noise (only if the column exists) ---
    ax = axes[0]
    if "true_feasible" in df.columns:
        for method in methods:
            sub = df[df["method"] == method]
            if sub["true_feasible"].notna().any():
                ax.plot(sub["noise_std"],
                        sub["true_feasible"].astype(float),
                        "o-", label=method, color=method_colors[method])
        ax.set_ylabel("True feasibility")
        ax.set_title("Ground Truth Feasibility vs. Noise")
        ax.legend(fontsize=8)
    else:
        # Fall back to worst-case violation when no separate GT feasibility column.
        for method in methods:
            sub = df[df["method"] == method]
            ax.plot(sub["noise_std"], sub["worst_violation"].astype(float),
                    "o-", label=method, color=method_colors[method])
        ax.set_ylabel("Worst-case violation")
        ax.set_title("Worst-case Violation vs. Noise")
        ax.legend(fontsize=8)
    ax.set_xlabel("Label noise $\\sigma$")
    ax.grid(alpha=0.3)

    # --- Held-out feasibility vs noise ---
    ax = axes[1]
    for method in methods:
        sub = df[df["method"] == method]
        ax.plot(sub["noise_std"], sub["feasibility_rate"],
                "o-", label=method, color=method_colors[method])
    ax.set_xlabel("Label noise $\\sigma$")
    ax.set_ylabel("Held-out feasibility rate")
    ax.set_title("Empirical Feasibility vs. Noise")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Objective vs noise ---
    ax = axes[2]
    for method in methods:
        sub = df[df["method"] == method]
        sub = sub[sub["objective"] < 1e6]
        if len(sub) > 0:
            ax.plot(sub["noise_std"], sub["objective"],
                    "o-", label=method, color=method_colors[method])
    ax.set_xlabel("Label noise $\\sigma$")
    ax.set_ylabel("Objective $c^\\top x^*$")
    ax.set_title("Objective Cost vs. Noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "noise_sweep.png"), dpi=150)
    plt.close()
    print(f"Saved noise sweep plot to {save_dir}/noise_sweep.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["gamma", "noise",
                                            "all"],
                        default="all")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot from existing CSVs")
    parser.add_argument("--calibrate-cv", action="store_true",
                        help="Synthetic robustness-parameter CV (KFold) -> synthetic_robustness_knobs.json")
    parser.add_argument("--refresh-cv", action="store_true",
                        help="With --calibrate-cv, delete the checkpoint + knobs first")
    parser.add_argument("--pareto", action="store_true",
                        help="Synthetic CV-centered Pareto -> results/synthetic/synthetic_pareto.csv")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--n-real", type=int, default=8, help="Pareto noise realizations")
    args = parser.parse_args()

    os.makedirs("results/synthetic", exist_ok=True)

    if args.calibrate_cv:
        run_cv_calibration_synthetic(load_config(), methods=args.methods, refresh=args.refresh_cv)
    elif args.pareto:
        run_synthetic_centered_pareto(load_config(), methods=args.methods, n_real=args.n_real)
    elif args.plot_only:
        if args.sweep in ["gamma", "all"]:
            plot_gamma_sweep()
        if args.sweep in ["noise", "all"]:
            plot_noise_sweep()
    else:
        if args.sweep in ["gamma", "all"]:
            run_gamma_sweep()
            plot_gamma_sweep()
        if args.sweep in ["noise", "all"]:
            run_noise_sweep()
            plot_noise_sweep()
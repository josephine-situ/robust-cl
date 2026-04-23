"""
Sweep over Gamma values to generate price-of-robustness curves.
"""

import yaml
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_all import load_config, run_experiment


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
    os.makedirs("results", exist_ok=True)
    combined.to_csv("results/sweep_results.csv", index=False)
    print(f"\nSaved sweep results to results/sweep_results.csv")

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
    os.makedirs("results", exist_ok=True)
    combined.to_csv("results/noise_sweep_results.csv", index=False)
    print(f"\nSaved noise sweep to results/noise_sweep_results.csv")

    return combined


def plot_gamma_sweep(csv_path="results/sweep_results.csv",
                     save_dir="results"):
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


def plot_noise_sweep(csv_path="results/noise_sweep_results.csv",
                     save_dir="results"):
    """Plot degradation under increasing label noise."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    methods = df["method"].unique()
    colors = sns.color_palette("colorblind", len(methods))
    method_colors = dict(zip(methods, colors))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- True feasibility vs noise ---
    ax = axes[0]
    for method in methods:
        sub = df[df["method"] == method]
        if sub["true_feasible"].notna().any():
            ax.plot(sub["noise_std"],
                    sub["true_feasible"].astype(float),
                    "o-", label=method, color=method_colors[method])
    ax.set_xlabel("Label noise $\\sigma$")
    ax.set_ylabel("True feasibility")
    ax.set_title("Ground Truth Feasibility vs. Noise")
    ax.legend(fontsize=8)
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


def plot_efficiency_comparison(save_dir="results"):
    """
    Compare C&CG vs random scenarios at equal number of
    embedded models. Requires running both with varying k.
    """
    import matplotlib.pyplot as plt
    import yaml

    config = load_config()
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]
    delta_bar = config["uncertainty"]["delta_bar"]
    gamma = config["uncertainty"]["gamma"]

    from src.data.generate import synthetic_nonlinear
    from src.methods.random_scenarios import solve_random_scenarios
    from src.methods.cp import solve_cp
    from src.evaluation.metrics import evaluate_solution

    instance = synthetic_nonlinear(
        n_train=config["data"]["n_train"],
        n_features=config["data"]["n_features"],
        noise_std=config["data"]["noise_std"],
    )

    k_values = [1, 2, 5, 10, 15, 20]
    random_feas = []
    ccg_feas = []

    for k in k_values:
        # Random scenarios with k models
        sol_rand = solve_random_scenarios(
            instance, model_type, model_params,
            delta_bar=delta_bar, gamma=gamma,
            n_scenarios=k,
        )
        if sol_rand.status != "infeasible":
            ev = evaluate_solution(
                sol_rand.x_opt, instance, "random",
                sol_rand.obj_value, sol_rand.models_embedded,
                sol_rand.solve_time,
                model_type, model_params,
                delta_bar, gamma,
                n_held_out=config["evaluation"]["n_held_out"],
            )
            random_feas.append(ev.feasibility_rate)
        else:
            random_feas.append(0.0)

        # C&CG with max k iterations
        sol_cp, _ = solve_cp(
            instance, model_type, model_params,
            delta_bar=delta_bar, gamma=gamma,
            max_iterations=k,
        )
        if sol_cp.status != "infeasible":
            ev = evaluate_solution(
                sol_cp.x_opt, instance, "ccg",
                sol_cp.obj_value, sol_cp.models_embedded,
                sol_cp.solve_time,
                model_type, model_params,
                delta_bar, gamma,
                n_held_out=config["evaluation"]["n_held_out"],
            )
            ccg_feas.append(ev.feasibility_rate)
        else:
            ccg_feas.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, random_feas, "o--", label="Random scenarios",
            color="orange")
    ax.plot(k_values, ccg_feas, "s-", label="C&CG (adversarial)",
            color="blue")
    ax.set_xlabel("Number of embedded models $k$")
    ax.set_ylabel("Held-out feasibility rate")
    ax.set_title("Efficiency: Adversarial vs. Random Scenario Selection")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "efficiency.png"), dpi=150)
    plt.close()
    print(f"Saved efficiency plot to {save_dir}/efficiency.png")

    # Save data
    eff_df = pd.DataFrame({
        "k": k_values,
        "random_feasibility": random_feas,
        "ccg_feasibility": ccg_feas,
    })
    eff_df.to_csv(os.path.join(save_dir, "efficiency.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["gamma", "noise",
                                            "efficiency", "all"],
                        default="all")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot from existing CSVs")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.plot_only:
        if args.sweep in ["gamma", "all"]:
            plot_gamma_sweep()
        if args.sweep in ["noise", "all"]:
            plot_noise_sweep()
        if args.sweep in ["efficiency", "all"]:
            # efficiency needs re-running
            print("Efficiency plot requires running experiments.")
    else:
        if args.sweep in ["gamma", "all"]:
            run_gamma_sweep()
            plot_gamma_sweep()
        if args.sweep in ["noise", "all"]:
            run_noise_sweep()
            plot_noise_sweep()
        if args.sweep in ["efficiency", "all"]:
            plot_efficiency_comparison()
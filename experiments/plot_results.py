"""
Plotting utilities for experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def plot_comparison_bar(df: pd.DataFrame, save_dir: str = "results"):
    """Bar chart comparing methods on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Objective cost
    ax = axes[0]
    mask = df["objective"] < 1e6
    sns.barplot(data=df[mask], x="method", y="objective", ax=ax)
    ax.set_title("Objective Cost (lower = better)")
    ax.set_ylabel("$$c^\\top x^*$$")
    ax.tick_params(axis="x", rotation=30)

    # Feasibility rate
    ax = axes[1]
    sns.barplot(data=df, x="method", y="feasibility_rate", ax=ax)
    ax.set_title("Held-Out Feasibility Rate")
    ax.set_ylabel("Fraction feasible")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=30)

    # Worst-case violation
    ax = axes[2]
    sns.barplot(data=df, x="method", y="worst_violation", ax=ax)
    ax.set_title("Worst-Case Violation")
    ax.set_ylabel("max $$f(x^*; \\theta^*(\\delta)) - b$$")
    ax.axhline(y=0, color="green", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_dir}/comparison.png")


def plot_ccg_convergence(trace_path: str = "results/ccg_trace.csv",
                         save_dir: str = "results"):
    """Plot C&CG convergence: violation and objective per iteration."""
    trace = pd.read_csv(trace_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violation over iterations
    ax = axes[0]
    ax.plot(trace["iteration"], trace["violation"], "o-", color="red")
    ax.axhline(y=0, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Iteration $$k$$")
    ax.set_ylabel("Violation $$v^k - b$$")
    ax.set_title("C\\&CG: Constraint Violation")

    # Objective over iterations
    ax = axes[1]
    ax.plot(trace["iteration"], trace["obj_value"], "s-", color="blue")
    ax.set_xlabel("Iteration $$k$$")
    ax.set_ylabel("$$c^\\top x^k$$")
    ax.set_title("C\\&CG: Objective Value (Lower Bound)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ccg_convergence.png"), dpi=150)
    plt.close()
    print(f"Saved convergence plot to {save_dir}/ccg_convergence.png")


def plot_price_of_robustness(save_dir: str = "results"):
    """
    Placeholder for price-of-robustness sweep.
    Run experiments with varying Gamma, collect results,
    then plot objective vs Gamma for each method.
    """
    # This requires running the experiment multiple times
    # with different Gamma values. See run_sweep() below.
    pass


def plot_efficiency(save_dir: str = "results"):
    """
    Placeholder for efficiency comparison.
    At equal number of models k, compare feasibility
    of C&CG vs random scenarios.
    """
    pass


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("results/results.csv")
    plot_comparison_bar(df)

    if os.path.exists("results/ccg_trace.csv"):
        plot_ccg_convergence()
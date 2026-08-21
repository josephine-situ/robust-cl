"""
Sweep over Gamma values to generate price-of-robustness curves.

Also hosts the SYNTHETIC robustness-parameter CV (--calibrate-cv) and the synthetic
CV-centered Pareto (--pareto), the synthetic counterparts of the gastric pipeline in
run_chemo_robust.py.
"""

import json
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_all import load_config, run_experiment


# ---------------------------------------------------------------------------
# Synthetic robustness-parameter CV + CV-centered Pareto
# ---------------------------------------------------------------------------
def _synth_build(method, config, model_type, model_params, seed):
    """Return ``build(knob) -> solver_fn`` for a non-contextual problem (synthetic,
    reactor). Single knob per method (CP tau, robust_reg label_eps, wrapper alpha);
    nominal ignores it. CP is single-lever (cp_alpha=0) like gastric.

    The argument lists themselves live in ``experiments/method_builders.py``, shared
    with the gastric builder (``run_chemo_robust._method_build_map``), so a change to
    a solver's signature or to a cross-cutting decision -- the one ``mip_gap``, the
    pinned ``cp_alpha``, the shared ``uncertainty_set`` -- cannot reach one problem
    and miss the other.

    ``config`` is read fresh on every call: the rho sweep hands in a config whose
    ``uncertainty.rho`` it has just overwritten.
    """
    from experiments.method_builders import build_method, synth_settings
    settings = synth_settings(config, seed)
    return lambda knob: build_method(method, knob, model_type, model_params,
                                     settings)


# Written by `run_cv.py --problem synthetic`; the CV-selected embedded model.
SYNTH_CV_CONFIGS = os.path.join("results", "cv", "synthetic_selected_configs.json")
SYNTH_OUTCOME = "synthetic_constraint"


def synth_model_spec(config, path=None, verbose=False):
    """``(model_type, model_params, from_cv)`` for the synthetic embedded model.

    The CV selection wins over ``config.yaml``'s ``model`` block when present; the
    synthetic model was hard-coded ``rf`` (50 trees, depth 5) and had never been
    cross-validated (2026-08-19 deck, next step 2). Returns ``from_cv`` so callers
    can SAY which one they used -- the two train different models on the same data,
    and a resumable score checkpoint keyed only by ``(method@rho, knob)`` would
    otherwise merge them silently (see ``run_rho_sweep._variant_suffix``).
    """
    path = path or SYNTH_CV_CONFIGS
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f).get(SYNTH_OUTCOME)
        if cfg:
            mt = cfg["model_type"]
            mp = cfg.get("model_params", cfg.get("params", {}))
            if verbose:
                print(f"    [synth] CV-selected embedded model: {mt} {mp} "
                      f"(from {path})", flush=True)
            return mt, dict(mp), True
    if verbose:
        print(f"    [synth] no {path}; embedded model from config.yaml: "
              f"{config['model']['type']} {config['model']['params']}", flush=True)
    return config["model"]["type"], dict(config["model"]["params"]), False


# Written by `run_cv.py --problem reactor`; the CV-selected embedded model.
REACTOR_CV_CONFIGS = os.path.join("results", "cv", "reactor_selected_configs.json")
REACTOR_OUTCOME = "benzene_constraint"


def reactor_model_spec(config, path=None, verbose=False):
    """``(model_type, model_params, from_cv)`` for the reactor embedded model.

    Same contract as :func:`synth_model_spec`: the CV selection wins over the
    ``reactor.model`` block in ``config.yaml`` when present, and ``from_cv`` is
    returned so the caller can scope the sweep cell by which model is in force.
    """
    path = path or REACTOR_CV_CONFIGS
    rc = config.get("reactor", {})
    default_t = rc.get("model", {}).get("type", config["model"]["type"])
    default_p = dict(rc.get("model", {}).get("params", config["model"]["params"]))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f).get(REACTOR_OUTCOME)
        if cfg:
            mt = cfg["model_type"]
            mp = cfg.get("model_params", cfg.get("params", {}))
            if verbose:
                print(f"    [reactor] CV-selected embedded model: {mt} {mp} "
                      f"(from {path})", flush=True)
            return mt, dict(mp), True
    if verbose:
        print(f"    [reactor] no {path}; embedded model from config.yaml: "
              f"{default_t} {default_p}", flush=True)
    return default_t, default_p, False


def _reactor_instance(config, cv_path=None, verbose=False):
    """The DMA-MR instance, carrying the CV-selected embedded model if there is one.

    The ODE dataset is cached on disk (see ``generate._reactor_dataset``), so this
    is cheap after the first call even though each oracle evaluation is a stiff
    ODE solve.
    """
    from src.data.generate import reactor_micl
    rc = config.get("reactor", {})
    mt, mp, from_cv = reactor_model_spec(config, cv_path, verbose=verbose)
    return reactor_micl(
        n_train=int(rc.get("n_train", 1000)),
        noise_std=float(rc.get("noise_std", 2.0)),
        seed=int(config["uncertainty"].get("bootstrap_seed", 42)),
        fixed_constraint_config=({"model_type": mt, "model_params": mp}
                                 if from_cv else None),
    )


def _synth_instance(config, seed=None, cv_path=None, verbose=False):
    """The synthetic instance, carrying the CV-selected embedded model if there is one.

    Setting ``constraint_model_configs`` is what makes the selection reach every
    method at once: ``nominal.resolve_constraint_config`` prefers it over the
    ``model_type``/``model_params`` arguments, and ``ScenarioBank`` resolves its
    per-draw refits through the same map.
    """
    from src.data.generate import synthetic_nonlinear
    d = config["data"]
    mt, mp, from_cv = synth_model_spec(config, cv_path, verbose=verbose)
    return synthetic_nonlinear(
        n_train=d["n_train"], n_features=d["n_features"], noise_std=d["noise_std"],
        seed=seed if seed is not None else config["uncertainty"].get("bootstrap_seed", 42),
        fixed_constraint_config=({"model_type": mt, "model_params": mp}
                                 if from_cv else None),
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
    inst = _synth_instance(config, verbose=True)
    # Same source as the instance, so the fallback path (no CV file) still trains
    # what config.yaml asks for and the CV path is used consistently.
    model_type, model_params, _ = synth_model_spec(config)
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
            feas, obj, solved = cv_score_knob(build, knob, folds, oracle, inst,
                                      constraint_names=None, contextual=False)
            append_score(scores_path, method, knob, feas, obj, solved)
            ckpt[key] = (feas, obj, solved)
            return feas, obj, solved
        return _score

    nom_build = _synth_build("nominal", config, model_type, model_params, seed)
    _, nom_obj, _ = make_scorer("nominal", nom_build)(0.0)
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
    model_type, model_params, _ = synth_model_spec(config, verbose=True)
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


def run_noise_sweep(noise_values=None, refresh=False, n_real=1):
    """
    Sweep over label noise levels sigma, across ``n_real`` independent data draws.
    Shows how each method degrades as noise increases.

    Each draw gets seed ``base + 1000 * r``, so a "realization" is a genuinely
    different dataset. Previously ``run_experiment`` was called without a seed, so
    every repeat rebuilt the *same* synthetic instance and the spread across
    realizations was identically zero.

    Methods run at their CV-calibrated knobs when
    ``results/cv/synthetic_robustness_knobs.json`` exists.

    Written INCREMENTALLY: the CSV is rewritten after each (sigma, draw) cell, and
    cells already present are skipped on re-entry (pass ``refresh=True`` to start
    over). A single end-of-run write loses the whole sweep if the process is killed
    partway -- which it was.
    """
    if noise_values is None:
        noise_values = [0.0, 0.05, 0.1, 0.2, 0.5]

    config = load_config()
    os.makedirs("results/synthetic", exist_ok=True)
    out_path = "results/synthetic/noise_sweep_results.csv"
    base_seed = config["uncertainty"].get("bootstrap_seed", 42)

    knobs = _load_synth_knobs()

    all_rows, done = [], set()
    if refresh and os.path.exists(out_path):
        os.remove(out_path)
    elif os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        # Only resume from a CSV matching the current method set; a stale run with
        # different methods must not be silently spliced into the new one.
        if "noise_std" in prev.columns and len(prev):
            all_rows.append(prev)
            # Pre-`draw` CSVs are treated as draw 0 so old sweeps still resume.
            draws = (prev["draw"] if "draw" in prev.columns else 0)
            done = set(zip(prev["noise_std"].astype(float),
                           pd.Series(draws, index=prev.index).astype(int)))
            print(f"[noise-sweep] resuming; {len(done)} (sigma, draw) cells already done",
                  flush=True)

    for sigma in noise_values:
        for r in range(int(n_real)):
            if (float(sigma), r) in done:
                print(f"[noise-sweep] skip sigma={sigma} draw={r} (already in {out_path})",
                      flush=True)
                continue
            print(f"\n{'#' * 60}")
            print(f"# NOISE_STD = {sigma}   DRAW {r + 1}/{n_real}")
            print(f"{'#' * 60}")

            config["data"]["noise_std"] = sigma
            df, _ = run_experiment(config, seed=base_seed + 1000 * r, knobs=knobs)
            df["noise_std"] = sigma
            df["draw"] = r
            all_rows.append(df)

            pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
            print(f"[noise-sweep] checkpointed sigma={sigma} draw={r} -> {out_path}",
                  flush=True)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved noise sweep to {out_path}")

    return combined


def _load_synth_knobs(path="results/cv/synthetic_robustness_knobs.json"):
    """CV-calibrated theta* per method, or ``None`` if --calibrate-cv hasn't run."""
    if not os.path.exists(path):
        print(f"[noise-sweep] no {path}; using config.yaml knobs "
              f"(run --calibrate-cv for calibrated operating points)", flush=True)
        return None
    with open(path) as f:
        knobs = json.load(f)
    print(f"[noise-sweep] CV knobs from {path}: {knobs}", flush=True)
    return knobs


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

    n_draws = df["draw"].nunique() if "draw" in df.columns else 1

    def _band(ax, method, col, worst="max"):
        """Mean line + worst-case band over the independent draws.

        A mean alone hides the tail, and the tail is the whole claim -- robustness
        is about the bad draw, not the average one. With a single draw the band
        collapses onto the line.
        """
        sub = df[df["method"] == method].dropna(subset=[col])
        if sub.empty:
            return
        g = sub.groupby("noise_std")[col]
        mean, lo, hi = g.mean(), g.min(), g.max()
        ax.plot(mean.index, mean.values, "o-", label=method,
                color=method_colors[method])
        if n_draws > 1:
            ax.fill_between(mean.index, lo.values, hi.values,
                            color=method_colors[method], alpha=0.15, lw=0)
            edge = hi if worst == "max" else lo
            ax.plot(edge.index, edge.values, ls=":", lw=1.2,
                    color=method_colors[method])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    band_note = f" (mean, band = min-max over {n_draws} draws)" if n_draws > 1 else ""

    # --- True feasibility vs noise (only if the column exists) ---
    ax = axes[0]
    if "true_feasible" in df.columns and df["true_feasible"].notna().any():
        for method in methods:
            _band(ax, method, "true_feasible", worst="min")
        ax.set_ylabel("True feasibility")
        ax.set_title("Ground Truth Feasibility vs. Noise" + band_note)
    else:
        # Fall back to worst-case violation when no separate GT feasibility column.
        for method in methods:
            _band(ax, method, "worst_violation")
        ax.set_ylabel("Worst-case violation")
        ax.set_title("Worst-case Violation vs. Noise" + band_note)
    ax.legend(fontsize=8)
    ax.set_xlabel("Label noise $\\sigma$")
    ax.grid(alpha=0.3)

    # --- Held-out feasibility vs noise ---
    ax = axes[1]
    for method in methods:
        _band(ax, method, "feasibility_rate", worst="min")
    ax.set_xlabel("Label noise $\\sigma$")
    ax.set_ylabel("Held-out feasibility rate")
    ax.set_title("Empirical Feasibility vs. Noise" + band_note)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Objective vs noise ---
    ax = axes[2]
    df = df[df["objective"].isna() | (df["objective"] < 1e6)]
    for method in methods:
        _band(ax, method, "objective")
    ax.set_xlabel("Label noise $\\sigma$")
    ax.set_ylabel("Objective $c^\\top x^*$")
    ax.set_title("Objective Cost vs. Noise" + band_note)
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
    parser.add_argument("--refresh-sweep", action="store_true",
                        help="Discard an existing noise_sweep_results.csv instead of resuming it")
    parser.add_argument("--calibrate-cv", action="store_true",
                        help="Synthetic robustness-parameter CV (KFold) -> synthetic_robustness_knobs.json")
    parser.add_argument("--refresh-cv", action="store_true",
                        help="With --calibrate-cv, delete the checkpoint + knobs first")
    parser.add_argument("--pareto", action="store_true",
                        help="Synthetic CV-centered Pareto -> results/synthetic/synthetic_pareto.csv")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--n-real", type=int, default=8,
                        help="Independent data draws (Pareto and --sweep noise)")
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
            run_noise_sweep(refresh=args.refresh_sweep, n_real=args.n_real)
            plot_noise_sweep()
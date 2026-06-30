#!/usr/bin/env python3
"""
Cross-validate model type and hyperparameters for constraint learning.

Two CV passes are available:

  Constraint CV  (default, fast)
      Shallow grids; selects ONE best model type per outcome for MIO embedding.
      Outputs: *_cv_scores.{csv,tex}, *_best_models.{csv,tex},
               *_selected_configs.json  (used by --cv-configs)

  GT Ensemble CV  (--ensemble flag, slow)
      Tunes the six paper GT model types (linear, svm, cart, rf, gbm, xgb) on the
      full train+test cohort; all six are averaged for ground-truth evaluation.
      Outputs: *_gt_cv_scores.{csv,tex}, *_gt_insample_r2.{csv,tex},
               *_gt_ensemble_configs.json  (used by --gt-cv-configs)

Usage:
    python experiments/run_cv.py                            # constraint CV, both problems
    python experiments/run_cv.py --ensemble                 # also run GT ensemble CV
    python experiments/run_cv.py --problem gastric          # gastric only
    python experiments/run_cv.py --problem gastric --ensemble
    python experiments/run_cv.py --scoring neg_mse
    python experiments/run_cv.py --output-dir results/cv

run_chemo_robust.py auto-loads both JSON files if present in results/cv/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.train import train_best_model_cv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Models that can be embedded into a Gurobi MIP via src/models/embed.py.
EMBEDDABLE_TYPES = {"linear", "svm", "cart", "rf", "gbm", "xgb", "mlp"}

# Shared CV parameter grids for constraint (embedded) models.
# Uniform increments: alpha/C use 10× multiples; max_depth uses +2;
# n_estimators uses ~2× steps; learning_rate uses ~5× steps.
CV_PARAM_GRIDS = {
    "linear": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],      # 10× multiples
        "l1_ratio": [0.01, 0.25, 0.5, 0.75, 1.0],      # +0.25
    },
    "svm": {
        "C": [0.01, 0.1, 1.0, 10.0],           # 10× multiples
        "epsilon": [0.01, 0.05, 0.1],
    },
    "cart": {
        "max_depth": [2, 4, 6, 8],                     # +2
        "min_samples_leaf": [0.02, 0.05, 0.1],         # ~2.5×
        "max_features": [0.5, 0.75, 1.0],              # +0.25
    },
    "rf": {
        "n_estimators": [5, 10, 20],                  # ~2.5×
        "max_depth": [2, 3, 4],                     # +1
        "max_features": ["sqrt", 0.1, 0.2],
    },
    "gbm": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],      # ~5× then 2×
        "max_depth": [2, 3, 4],                     # +1
        "n_estimators": [5, 10, 20],                 # 2×
        "subsample": [0.7, 0.9],
    },
    "xgb": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "n_estimators": [5, 10, 20],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.3, 0.5],
    },
    "mlp": {
        "hidden_layer_sizes": [
                (25,), (50,),               # Your core 1-layer workhorses
                (25, 10), (25, 25),         # 2-layer options for slightly more complexity
                (10, 5, 2)                  # The single 3-layer test (expect this to perform poorly)
            ],
        "solver": ["lbfgs"],
        "alpha": [1e-4, 1e-3, 0.01],                     # Strong L2 regularization
    }
}

# GT ensemble CV grids — deeper and richer than the embedded grids.
# Each grid is a superset of the corresponding CV_PARAM_GRIDS entry so that
# simpler embedded-model configurations are also evaluated as ensemble members.
GT_CV_PARAM_GRIDS = {
    "linear": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "l1_ratio": [0.01, 0.25, 0.5, 0.75, 1.0],
    },
    "svm": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.05, 0.1],
    },
    "cart": {
        "max_depth": [2, 4, 6, 8, 10],                # +2, one deeper level
        "min_samples_leaf": [0.02, 0.05, 0.1],
        "max_features": [0.5, 0.75, 1.0],
    },
    "rf": {
        "n_estimators": [10, 50, 100, 250],      # includes simpler + deeper
        "max_depth": [2, 4, 6, 8],
        "max_features": ["sqrt", 0.1, 0.2],
    },
    "gbm": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],                    # +2, deeper range
        "n_estimators": [10, 50, 100, 250],           # 2×
        "subsample": [0.7, 0.9],
    },
    "xgb": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],                       # +2
        "n_estimators": [10, 50, 100, 250],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.3, 0.5],
    },
}

# Display order for model type columns in CV scores table
MODEL_ORDER = ["linear", "svm", "cart", "rf", "gbm", "xgb", "mlp"]

# Paper GT ensemble members (Table EC.12): average of these six model types
GT_MODEL_ORDER = ["linear", "svm", "cart", "rf", "gbm", "xgb"]

# Human-readable labels for outcome/constraint names
OUTCOME_LABELS = {
    "synthetic_constraint": "Synthetic",
    "dlt_constraint": "Any DLT",
    "blood_constraint": "Blood",
    "constitutional_constraint": "Constitutional",
    "infection_constraint": "Infection",
    "gi_constraint": "Gastrointestinal",
    "os_constraint": "Overall Survival",
}

# Short outcome keys used in GT_ENSEMBLE_SPECS (no "_constraint" suffix)
GT_OUTCOME_LABELS = {
    "dlt": "Any DLT",
    "blood": "Blood",
    "constitutional": "Constitutional",
    "infection": "Infection",
    "gi": "Gastrointestinal",
    "os": "Overall Survival",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_embeddable(model_type: str, outcome: str) -> None:
    """Warn if the CV winner cannot be embedded into a Gurobi MIP."""
    if model_type not in EMBEDDABLE_TYPES:
        warnings.warn(
            f"CV selected '{model_type}' for '{outcome}' which is not yet supported "
            f"for MIO embedding in embed.py (supported: {sorted(EMBEDDABLE_TYPES)}). "
            "This config cannot be used directly in run_chemo_robust.py / run_all.py "
            "until embed.py is extended.",
            stacklevel=3,
        )


def _params_str(params: dict) -> str:
    """Compact parameter string for display in tables."""
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _write_cv_scores_tex(
    df_scores: pd.DataFrame,
    path: Path,
    caption: str,
    model_order: list | None = None,
) -> None:
    """
    Write CV scores table as booktabs LaTeX.

    Rows = outcomes, columns = model types, values = 5-fold CV R².
    Best score per row is bolded.
    """
    order = model_order if model_order is not None else MODEL_ORDER
    model_cols = [c for c in order if c in df_scores.columns]
    col_header = " & ".join(["Outcome"] + [c.upper() for c in model_cols])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{l" + "r" * len(model_cols) + "}",
        r"\toprule",
        col_header + r" \\",
        r"\midrule",
    ]

    for _, row in df_scores.iterrows():
        vals = [row.get(c, float("nan")) for c in model_cols]
        valid_vals = [v for v in vals if not np.isnan(v)]
        best_val = max(valid_vals) if valid_vals else float("nan")
        cells = []
        for v in vals:
            if np.isnan(v):
                cells.append("---")
            elif not np.isnan(best_val) and abs(v - best_val) < 1e-9:
                cells.append(rf"\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        lines.append(str(row["outcome_label"]) + " & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_best_models_tex(df_best: pd.DataFrame, path: Path, caption: str) -> None:
    """
    Write best-models table as booktabs LaTeX (EC.7 / EC.10 style).

    Columns: Outcome | Best Model | CV R² | Test R² (if available) | Best Parameters
    """
    has_test_r2 = (
        "test_r2" in df_best.columns
        and df_best["test_r2"].notna().any()
    )
    col_keys = ["outcome_label", "best_model", "cv_r2"]
    headers = ["Outcome", "Best Model", "CV R$^2$"]
    if has_test_r2:
        col_keys.append("test_r2")
        headers.append("Test R$^2$")
    col_keys.append("best_params_str")
    headers.append("Best Parameters")

    col_fmt = "ll" + "r" * (2 + int(has_test_r2)) + "l"
    header_str = " & ".join(headers)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        header_str + r" \\",
        r"\midrule",
    ]

    for _, row in df_best.iterrows():
        cv_r2_val = row.get("cv_r2", float("nan"))
        cells = [
            str(row.get("outcome_label", "")),
            str(row.get("best_model", "")),
            f"{cv_r2_val:.3f}" if not np.isnan(float(cv_r2_val)) else "---",
        ]
        if has_test_r2:
            v = row.get("test_r2", float("nan"))
            cells.append(f"{float(v):.3f}" if not np.isnan(float(v)) else "---")
        cells.append(str(row.get("best_params_str", "")))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# In-sample R² comparison table writer
# ---------------------------------------------------------------------------

def _write_insample_r2_tex(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    model_order: list | None = None,
) -> None:
    """
    Write an in-sample R² comparison table as booktabs LaTeX.

    Rows = outcomes.  Columns = individual model types, then ensemble.
    Best value per row is bolded.
    """
    order = model_order if model_order is not None else MODEL_ORDER
    model_cols = [c for c in order if c in df.columns]
    extra_cols = [c for c in ["ensemble"] if c in df.columns]
    all_val_cols = model_cols + extra_cols

    col_headers = (
        ["Outcome"]
        + [c.upper() for c in model_cols]
        + ["Ensemble"][: len(extra_cols)]
    )
    col_header_str = " & ".join(col_headers)
    col_fmt = "l" + "r" * len(all_val_cols)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        col_header_str + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        vals = [row.get(c, float("nan")) for c in all_val_cols]
        valid_vals = [v for v in vals if not np.isnan(float(v))]
        best_val = max(valid_vals) if valid_vals else float("nan")
        cells = []
        for v in vals:
            fv = float(v)
            if np.isnan(fv):
                cells.append("---")
            elif not np.isnan(best_val) and abs(fv - best_val) < 1e-9:
                cells.append(rf"\textbf{{{fv:.3f}}}")
            else:
                cells.append(f"{fv:.3f}")
        lines.append(str(row["outcome_label"]) + " & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Base model factory (shared by constraint CV and ensemble CV)
# ---------------------------------------------------------------------------

def _make_base_model(model_type: str, seed: int):
    """Return a fresh sklearn estimator for the given model type, or None if unavailable."""
    from sklearn.linear_model import ElasticNet
    from sklearn.svm import LinearSVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor

    if model_type == "linear":
        return ElasticNet(random_state=seed, max_iter=10_000)
    elif model_type == "svm":
        return LinearSVR(max_iter=100_000, dual=False, loss="squared_epsilon_insensitive")
    elif model_type == "cart":
        return DecisionTreeRegressor(random_state=seed)
    elif model_type == "rf":
        return RandomForestRegressor(random_state=seed)
    elif model_type == "gbm":
        return GradientBoostingRegressor(random_state=seed)
    elif model_type == "xgb":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                verbosity=0,
                tree_method="exact",
            )
        except ImportError:
            warnings.warn("xgboost not installed; skipping xgb in ensemble CV.")
            return None
    elif model_type == "mlp":
        return MLPRegressor(random_state=seed, max_iter=10_000)
    return None


# ---------------------------------------------------------------------------
# In-sample R² comparison helper
# ---------------------------------------------------------------------------

def _compute_insample_r2(
    outcomes: list,
    gt_configs: dict,
    df_cv_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train each model type on the full cohort and compute in-sample R².

    Also trains the ensemble (mean of all tuned members) and reports its
    in-sample R².

    Parameters
    ----------
    outcomes     : list of dicts with keys name, label, X_train, y_train
                   (X_train / y_train are the *full* cohort arrays)
    gt_configs   : {outcome_name: [{model_type, params}, ...]}
    df_cv_scores : CV R² DataFrame (used only for row ordering / labels)

    Returns
    -------
    pd.DataFrame with columns: name, outcome_label, <model types>, ensemble
    """
    from sklearn.metrics import r2_score as _r2
    from src.models.train import train_model

    rows = []
    for item in outcomes:
        name = item["name"]
        label = item["label"]
        X = item["X_train"]
        y = item["y_train"]

        row = {"name": name, "outcome_label": label}

        member_preds = []
        specs = gt_configs.get(name, [])

        for spec in specs:
            mtype = spec["model_type"]
            params = {k: v for k, v in spec["params"].items() if k != "random_state"}
            params["random_state"] = spec["params"].get("random_state", 1)
            try:
                model = train_model(X, y, mtype, params, normalize=True)
                preds = model.predict(X)
                r2_val = float(_r2(y, preds))
                row[mtype] = r2_val
                member_preds.append(preds)
            except Exception as exc:
                warnings.warn(f"  In-sample R² failed for {name}/{mtype}: {exc}")
                row[mtype] = float("nan")

        if member_preds:
            ens_pred = np.mean(member_preds, axis=0)
            row["ensemble"] = float(_r2(y, ens_pred))
        else:
            row["ensemble"] = float("nan")

        print(
            f"  [{label}]  in-sample R²: "
            + "  ".join(
                f"{k}={v:.3f}" for k, v in row.items()
                if k not in ("name", "outcome_label")
            ),
            flush=True,
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAP feature importance for constraint CV best models
# ---------------------------------------------------------------------------

def _run_feature_importance_gastric(
    configs: dict,
    X_train: np.ndarray,
    outcomes_y: dict,
    feature_names: list,
    out_dir: Path,
    seed: int = 1,
) -> None:
    """
    Compute SHAP values for each outcome's best constraint model and save outputs.

    For each outcome:
      - Trains the best model (from constraint CV configs) on X_train / y_train.
      - Selects an appropriate SHAP explainer:
          * tree models  (cart, rf, gbm, xgb) → shap.TreeExplainer
          * linear/svm                         → shap.LinearExplainer
          * mlp                                → shap.KernelExplainer (slow fallback)
      - Saves a beeswarm plot (top 20 features) as gastric_shap_{outcome}.png.
      - Assembles mean |SHAP| per feature across outcomes into a CSV table.

    Parameters
    ----------
    configs       : {constraint_name: {"model_type": ..., "model_params": ...}}
                    from gastric_selected_configs.json
    X_train       : (n_train, d) feature matrix
    outcomes_y    : {constraint_name: y_train array}
    feature_names : list of d feature name strings
    out_dir       : directory where outputs are saved
    seed          : random state (used when training models)
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("shap or matplotlib not installed; skipping feature importance.")
        return

    from src.models.train import train_model

    importance_records = {}  # {outcome_short_name: array of mean |SHAP|}

    for cname, cfg in configs.items():
        mtype = cfg["model_type"]
        mparams = {**cfg.get("model_params", {}), "random_state": seed}
        y = outcomes_y.get(cname)
        if y is None:
            continue

        short_name = cname.replace("_constraint", "")
        label = OUTCOME_LABELS.get(cname, short_name)
        print(f"\n  SHAP [{label}]  model={mtype}", flush=True)

        try:
            model = train_model(X_train, y, mtype, mparams, normalize=True)
        except Exception as exc:
            warnings.warn(f"  Could not train {mtype} for {cname}: {exc}")
            continue

        scaler = model.named_steps["scaler"]
        inner = model.named_steps["model"]
        X_scaled = scaler.transform(X_train)

        try:
            if mtype in ("cart", "rf", "gbm", "xgb"):
                explainer = shap.TreeExplainer(inner)
                shap_values = explainer.shap_values(X_scaled)
            elif mtype in ("linear", "svm"):
                explainer = shap.LinearExplainer(inner, X_scaled)
                shap_values = explainer.shap_values(X_scaled)
            else:
                # MLP fallback: KernelExplainer with a small background summary
                background = shap.kmeans(X_scaled, min(50, len(X_scaled)))
                explainer = shap.KernelExplainer(inner.predict, background)
                shap_values = explainer.shap_values(X_scaled, nsamples=100)
        except Exception as exc:
            warnings.warn(f"  SHAP failed for {cname}/{mtype}: {exc}")
            continue

        shap_values = np.asarray(shap_values)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_records[short_name] = mean_abs_shap

        # Beeswarm plot — top 20 features by mean |SHAP|
        top_idx = np.argsort(mean_abs_shap)[::-1][:20]
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            shap_values[:, top_idx],
            X_scaled[:, top_idx],
            feature_names=[feature_names[i] for i in top_idx],
            plot_type="dot",
            show=False,
            plot_size=None,
        )
        ax = plt.gca()
        ax.set_title(f"SHAP — {label} ({mtype.upper()})", fontsize=12)
        plt.tight_layout()
        plot_path = out_dir / f"gastric_shap_{short_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"    Saved SHAP plot: {plot_path.name}", flush=True)

    # Summary CSV: rows = features, cols = outcome short names
    if importance_records:
        df_imp = pd.DataFrame(
            importance_records,
            index=feature_names,
        )
        df_imp.index.name = "feature"
        imp_csv = out_dir / "gastric_feature_importance.csv"
        df_imp.to_csv(imp_csv)
        print(f"\n  Saved feature importance CSV: {imp_csv.name}", flush=True)


# ---------------------------------------------------------------------------
# Core CV runner
# ---------------------------------------------------------------------------

def run_cv_for_outcomes(
    outcomes: list,
    cv_param_grids: dict,
    scoring: str = "r2",
    cv_folds: int = 5,
    seed: int = 1,
) -> tuple:
    """
    Run GridSearchCV for each outcome and collect results.

    Parameters
    ----------
    outcomes : list of dicts, each with keys:
        name       : constraint name (used as JSON key)
        label      : human-readable display label
        X_train    : (n_train, d) features
        y_train    : (n_train,) targets
        X_test     : (n_test, d) or None  — for test R² only
        y_test     : (n_test,) or None

    Returns
    -------
    df_scores : pd.DataFrame — CV R² pivot (rows=outcomes, cols=model types)
    df_best   : pd.DataFrame — best-model summary per outcome
    configs   : dict — {constraint_name: {"model_type": ..., "model_params": ...}}
    """
    sk_scoring = "r2" if scoring == "r2" else "neg_mean_squared_error"

    score_rows = []
    best_rows = []
    configs = {}

    for item in outcomes:
        name = item["name"]
        label = item["label"]
        X_tr = item["X_train"]
        y_tr = item["y_train"]
        X_te = item.get("X_test")
        y_te = item.get("y_test")

        print(f"\n  [{label}]  n_train={len(y_tr)}", flush=True)

        best_model, best_type, best_params, all_scores = train_best_model_cv(
            X_tr,
            y_tr,
            param_grids=cv_param_grids,
            random_state=seed,
            scoring=sk_scoring,
            cv_folds=cv_folds,
            return_params=True,
            return_all_scores=True,
        )

        cv_r2 = float(all_scores.get(best_type, float("nan")))
        print(f"    Winner: {best_type}  CV R\u00b2={cv_r2:.3f}  params={best_params}")

        _check_embeddable(best_type, name)

        # Compute held-out test R²
        test_r2 = float("nan")
        if X_te is not None and y_te is not None and len(y_te) > 0:
            test_r2 = float(r2_score(y_te, best_model.predict(X_te)))
            print(f"    Test R\u00b2={test_r2:.3f}")

        score_row = {"name": name, "outcome_label": label}
        score_row.update(all_scores)
        score_rows.append(score_row)

        best_rows.append({
            "name": name,
            "outcome_label": label,
            "best_model": best_type,
            "cv_r2": cv_r2,
            "test_r2": test_r2,
            "best_params_str": _params_str(best_params),
        })

        configs[name] = {
            "model_type": best_type,
            "model_params": {**best_params, "random_state": seed},
        }

    df_scores = pd.DataFrame(score_rows)
    df_best = pd.DataFrame(best_rows)
    return df_scores, df_best, configs


def run_cv_for_ensemble(
    outcomes: list,
    cv_param_grids: dict,
    scoring: str = "r2",
    cv_folds: int = 5,
    seed: int = 1,
) -> tuple:
    """
    GT ensemble CV: tune ALL model types per outcome using a deep parameter grid.

    Unlike ``run_cv_for_outcomes`` (which picks ONE winner per outcome for MIO
    embedding), this function tunes every model type independently because they
    are all used as ensemble members for ground-truth evaluation.

    Parameters
    ----------
    outcomes : list of dicts with keys: name (short, e.g. "dlt"), label, X_train, y_train

    Returns
    -------
    df_scores  : pd.DataFrame — CV R² pivot (rows=outcomes, cols=model types)
    gt_configs : dict — {outcome_name: [{model_type, params}, ...]}
                 Same format as ``GT_ENSEMBLE_SPECS`` in gastric_model_specs.py.
    """
    sk_scoring = "r2" if scoring == "r2" else "neg_mean_squared_error"
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    score_rows = []
    gt_configs: dict = {}

    for item in outcomes:
        name = item["name"]
        label = item["label"]
        X_tr = item["X_train"]
        y_tr = item["y_train"]

        print(f"\n  [{label}]  n_train={len(y_tr)}", flush=True)

        score_row = {"name": name, "outcome_label": label}
        specs = []

        for model_type, grid in cv_param_grids.items():
            base = _make_base_model(model_type, seed)
            if base is None:
                continue

            search = GridSearchCV(base, grid, cv=kf, scoring=sk_scoring, n_jobs=-1)
            search.fit(X_tr, y_tr)

            cv_score = float(search.best_score_)
            best_params = dict(search.best_params_)
            score_row[model_type] = cv_score

            print(
                f"    {model_type:8s}: CV R\u00b2={cv_score:.3f}  params={best_params}",
                flush=True,
            )

            specs.append({
                "model_type": model_type,
                "params": {**best_params, "random_state": seed},
            })

        score_rows.append(score_row)
        gt_configs[name] = specs

    df_scores = pd.DataFrame(score_rows)
    return df_scores, gt_configs


# ---------------------------------------------------------------------------
# Problem-specific runners
# ---------------------------------------------------------------------------

def run_cv_synthetic(args, out_dir: Path) -> None:
    """Run CV for the synthetic nonlinear problem."""
    from src.data.generate import synthetic_nonlinear

    print("\n" + "=" * 60)
    print("SYNTHETIC CROSS-VALIDATION")
    print("=" * 60)

    seed = args.seed
    n_train = 200
    n_test_held_out = 50

    # Generate a larger dataset and split off a held-out test portion.
    # y_true (noiseless) is used as the test target for a noise-free test R².
    instance = synthetic_nonlinear(n_train=n_train + n_test_held_out, seed=seed)
    md = instance.constraints[0].models_data[0]

    X_tr = md.X_train[:n_train]
    y_tr = md.y_train[:n_train]
    X_te = md.X_train[n_train:]
    # Prefer noiseless ground truth for test evaluation; fall back to noisy labels.
    y_te = md.y_true[n_train:] if md.y_true is not None else md.y_train[n_train:]

    outcomes = [
        {
            "name": "synthetic_constraint",
            "label": OUTCOME_LABELS["synthetic_constraint"],
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
        }
    ]

    df_scores, df_best, configs = run_cv_for_outcomes(
        outcomes,
        CV_PARAM_GRIDS,
        scoring=args.scoring,
        cv_folds=args.cv_folds,
        seed=seed,
    )

    _save_cv_outputs(
        prefix="synthetic",
        out_dir=out_dir,
        df_scores=df_scores,
        df_best=df_best,
        configs=configs,
        scores_caption=(
            "Synthetic: 5-Fold CV R\\textsuperscript{2} by Model Type"
        ),
        best_caption=(
            "Synthetic: Best Constraint Model (CV Selection)"
        ),
    )


def run_cv_gastric(args, out_dir: Path) -> None:
    """Run CV for the gastric cancer chemotherapy problem."""
    from src.data.generate import gastric_cancer
    from src.data.gastric_v11 import train_percentile_scores

    print("\n" + "=" * 60)
    print("GASTRIC CANCER — CONSTRAINT MODEL CV  (shallow grid)")
    print("=" * 60)

    seed = args.seed
    instance = gastric_cancer()

    X_tr = instance.X_train   # 320 train arms
    X_te = instance.X_test    # 96 test arms
    obs = instance.observed_test_outcomes or {}

    outcomes = []
    for c in instance.constraints:
        cname = c.name
        label = OUTCOME_LABELS.get(cname, cname)
        md = c.models_data[0]
        y_tr = md.y_train  # percentile-transformed (toxicities) or raw (OS)

        # Build held-out test targets (apply same transform as training)
        outcome_key = cname.replace("_constraint", "")
        y_te = None
        if outcome_key in obs and obs[outcome_key] is not None:
            raw_te = np.asarray(obs[outcome_key], dtype=float)
            if outcome_key == "os":
                y_te = raw_te
            elif md.y_true is not None:
                # Map test outcomes to train-percentile scale using raw train values
                y_te = train_percentile_scores(md.y_true, raw_te)

        outcomes.append({
            "name": cname,
            "label": label,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te if y_te is not None else None,
            "y_test": y_te,
        })

    df_scores, df_best, configs = run_cv_for_outcomes(
        outcomes,
        CV_PARAM_GRIDS,
        scoring=args.scoring,
        cv_folds=args.cv_folds,
        seed=seed,
    )

    _save_cv_outputs(
        prefix="gastric",
        out_dir=out_dir,
        df_scores=df_scores,
        df_best=df_best,
        configs=configs,
        scores_caption=(
            "Gastric Cancer: 5-Fold CV R\\textsuperscript{2} by Model Type "
            "(Embedded Constraint Models, Table EC.6 Style)"
        ),
        best_caption=(
            "Gastric Cancer: Best Constraint Model per Outcome "
            "(CV Selection, Table EC.10 Style)"
        ),
    )

    # SHAP feature importance for each best constraint model
    if instance.feature_names:
        print("\n" + "-" * 60)
        print("  Computing SHAP feature importance for best constraint models...")
        outcomes_y = {
            o["name"]: o["y_train"] for o in outcomes
        }
        _run_feature_importance_gastric(
            configs=configs,
            X_train=X_tr,
            outcomes_y=outcomes_y,
            feature_names=instance.feature_names,
            out_dir=out_dir,
            seed=seed,
        )

    if getattr(args, "ensemble", False):
        run_cv_gastric_ensemble(args, out_dir, instance=instance)


def run_cv_gastric_ensemble(args, out_dir: Path, instance=None) -> None:
    """
    GT ensemble CV for gastric: tune all model types with a deep grid.

    CV is run on the whole cohort (train + test arms) so hyperparameters are
    selected on all available labeled data, matching the full 461-arm training
    used for the GT ensemble itself.
    """
    from src.data.generate import gastric_cancer
    from src.data.gastric_v11 import train_percentile_scores

    print("\n" + "=" * 60)
    print("GASTRIC CANCER — GT ENSEMBLE CV  (6 paper model types)")
    print("=" * 60)
    print("  CV is run on the whole cohort (train + test arms, all 461 arms).")

    seed = args.seed
    if instance is None:
        instance = gastric_cancer()

    X_tr = instance.X_train
    X_te = instance.X_test
    obs = instance.observed_test_outcomes or {}

    X_all = np.vstack([X_tr, X_te]) if len(X_te) > 0 else X_tr

    # Build per-outcome arrays for all 461 arms using short outcome keys.
    outcomes = []
    for cname, label in GT_OUTCOME_LABELS.items():
        constraint = next(
            (c for c in instance.constraints if c.name == f"{cname}_constraint"),
            None,
        )
        if constraint is None:
            continue
        md = constraint.models_data[0]
        y_tr = md.y_train  # percentile-transformed train labels

        # Build test labels on the same percentile scale as training
        y_te = None
        if cname in obs and obs[cname] is not None:
            raw_te = np.asarray(obs[cname], dtype=float)
            if cname == "os":
                y_te = raw_te
            elif md.y_true is not None:
                y_te = train_percentile_scores(md.y_true, raw_te)

        if y_te is not None and len(y_te) == len(X_te):
            y_all = np.concatenate([y_tr, y_te])
            X_cv = X_all
            n_total = len(y_all)
        else:
            y_all = y_tr
            X_cv = X_tr
            n_total = len(y_all)
            print(f"    Warning: no test labels for '{label}'; using train-only cohort.")

        print(f"  [{label}] CV cohort size: {n_total}", flush=True)
        outcomes.append({
            "name": cname,
            "label": label,
            "X_train": X_cv,
            "y_train": y_all,
        })

    df_scores, gt_configs = run_cv_for_ensemble(
        outcomes,
        GT_CV_PARAM_GRIDS,
        scoring=args.scoring,
        cv_folds=args.cv_folds,
        seed=seed,
    )

    # In-sample R² comparison
    print("\n" + "-" * 60)
    print("  Computing in-sample R² (training on full cohort)...")
    df_insample = _compute_insample_r2(outcomes, gt_configs, df_scores)

    # Save outputs
    scores_csv = out_dir / "gastric_gt_cv_scores.csv"
    scores_tex = out_dir / "gastric_gt_cv_scores.tex"
    insample_csv = out_dir / "gastric_gt_insample_r2.csv"
    insample_tex = out_dir / "gastric_gt_insample_r2.tex"
    configs_json = out_dir / "gastric_gt_ensemble_configs.json"

    df_scores.to_csv(scores_csv, index=False)
    df_insample.to_csv(insample_csv, index=False)

    with open(configs_json, "w", encoding="utf-8") as f:
        json.dump(gt_configs, f, indent=2, default=str)

    _write_cv_scores_tex(
        df_scores,
        scores_tex,
        "Gastric Cancer: GT Ensemble CV R\\textsuperscript{2} by Model Type "
        "(Deep Grid, Table EC.11 Style)",
        model_order=GT_MODEL_ORDER,
    )
    _write_insample_r2_tex(
        df_insample,
        insample_tex,
        "Gastric Cancer: In-Sample R\\textsuperscript{2} — Individual Models and Ensemble",
        model_order=GT_MODEL_ORDER,
    )

    print(f"\n  Outputs saved to {out_dir}/")
    print(f"    {scores_csv.name}, {scores_tex.name}")
    print(f"    {insample_csv.name}, {insample_tex.name}")
    print(f"    {configs_json.name}")

    print("\n  GT Ensemble CV Scores (5-fold R\u00b2):")
    model_cols = [c for c in GT_MODEL_ORDER if c in df_scores.columns]
    display_cols = ["outcome_label"] + model_cols
    print(df_scores[[c for c in display_cols if c in df_scores.columns]].to_string(index=False))

    print("\n  In-sample R\u00b2 (full cohort):")
    insample_display = ["outcome_label"] + model_cols + ["ensemble"]
    print(df_insample[[c for c in insample_display if c in df_insample.columns]].to_string(index=False))


# ---------------------------------------------------------------------------
# Shared output helper
# ---------------------------------------------------------------------------

def _save_cv_outputs(
    prefix: str,
    out_dir: Path,
    df_scores: pd.DataFrame,
    df_best: pd.DataFrame,
    configs: dict,
    scores_caption: str,
    best_caption: str,
) -> None:
    """Save CSV, LaTeX, and JSON outputs for one problem."""
    scores_csv = out_dir / f"{prefix}_cv_scores.csv"
    scores_tex = out_dir / f"{prefix}_cv_scores.tex"
    best_csv = out_dir / f"{prefix}_best_models.csv"
    best_tex = out_dir / f"{prefix}_best_models.tex"
    configs_json = out_dir / f"{prefix}_selected_configs.json"

    df_scores.to_csv(scores_csv, index=False)
    df_best.to_csv(best_csv, index=False)

    with open(configs_json, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, default=str)

    _write_cv_scores_tex(df_scores, scores_tex, scores_caption)
    _write_best_models_tex(df_best, best_tex, best_caption)

    print(f"\n  Outputs saved to {out_dir}/")
    print(f"    {scores_csv.name}, {scores_tex.name}")
    print(f"    {best_csv.name}, {best_tex.name}")
    print(f"    {configs_json.name}")

    print("\n  CV Scores (5-fold R\u00b2):")
    model_cols = [c for c in MODEL_ORDER if c in df_scores.columns]
    display_cols = ["outcome_label"] + model_cols
    print(df_scores[[c for c in display_cols if c in df_scores.columns]].to_string(index=False))

    print("\n  Best Models:")
    best_display = ["outcome_label", "best_model", "cv_r2"]
    if "test_r2" in df_best.columns:
        best_display.append("test_r2")
    best_display.append("best_params_str")
    print(df_best[[c for c in best_display if c in df_best.columns]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validate constraint models for synthetic and gastric problems. "
            "Outputs EC.6/EC.7-style tables and a JSON config for downstream experiments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--problem",
        choices=["synthetic", "gastric", "both"],
        default="both",
        help="Which problem to run CV for (default: both)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1, matching paper's run_MLmodels.py)",
    )
    parser.add_argument(
        "--scoring",
        choices=["r2", "neg_mse"],
        default="r2",
        help="CV scoring metric (default: r2, matching paper)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cv",
        metavar="DIR",
        help="Output directory for tables and JSON (default: results/cv)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "Also run GT ensemble CV with a deep grid (deeper trees, more estimators). "
            "Outputs gastric_gt_cv_scores.{csv,tex} and gastric_gt_ensemble_configs.json. "
            "Slow — omit for a quick constraint-model-only run."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONSTRAINT MODEL CROSS-VALIDATION")
    print("=" * 60)
    print(f"  problem   : {args.problem}")
    print(f"  ensemble  : {args.ensemble}")
    print(f"  cv_folds  : {args.cv_folds}")
    print(f"  seed      : {args.seed}")
    print(f"  scoring   : {args.scoring}")
    print(f"  output_dir: {out_dir}")
    print(f"  constraint models : {list(CV_PARAM_GRIDS.keys())}")
    if args.ensemble:
        print(f"  ensemble models   : {list(GT_CV_PARAM_GRIDS.keys())}")
    print(f"  embeddable: {sorted(EMBEDDABLE_TYPES)}")

    if args.problem in ("synthetic", "both"):
        run_cv_synthetic(args, out_dir)

    if args.problem in ("gastric", "both"):
        run_cv_gastric(args, out_dir)

    print("\n" + "=" * 60)
    print(f"CV complete. All outputs in: {out_dir}")
    print("=" * 60)
    print("\nTo use CV configs in downstream experiments:")
    if args.problem in ("synthetic", "both"):
        print(
            f"  python experiments/run_all.py "
            f"--cv-configs {out_dir}/synthetic_selected_configs.json"
        )
    if args.problem in ("gastric", "both"):
        print(
            f"  python experiments/run_chemo_robust.py  "
            f"# auto-loads {out_dir}/gastric_selected_configs.json if present"
        )
        if args.ensemble:
            print(
                f"  # also auto-loads {out_dir}/gastric_gt_ensemble_configs.json for GT models"
            )


if __name__ == "__main__":
    main()

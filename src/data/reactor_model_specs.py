"""Reactor-problem model specs: the mixed-type ensemble used as the CV oracle.

Two different judges serve the reactor, and they must not be confused:

  FINAL EVALUATION uses the ODEs (:func:`src.data.dma_mr.benzene_flow`). That is
  real ground truth and is what C-MICL reports as the "empirical ground-truth
  feasibility rate", so our numbers are readable against their Figure 1.

  RHO TUNING uses the ensemble configured here -- a train-only PROXY. Using the
  ODE oracle to pick rho would calibrate the uncertainty set against the exact
  truth, the same error as pinning D to synthetic's known ``noise_std``, and CP
  would win by construction. The rule in this repo is that rho is never fitted
  against the judge that scores it.

The payoff of having both is an AUDIT that is impossible on gastric: the proxy's
verdicts can be checked against the ODE, which is how we quantify how much of a
reported feasibility is method and how much is judge error. On synthetic that
audit (2026-08-21) found the proxy disagreeing with the analytic truth on 5 of 5
nominal decisions, because a constrained optimum sits on the boundary where the
judge's error decides the verdict.

Fallback specs only; ``run_cv.py --problem reactor --ensemble`` writes CV-tuned
ones to ``results/cv/reactor_gt_ensemble_configs.json``, which takes precedence.
Sized for n=1000, d=5 -- richer than the synthetic fallbacks, which are for n=2500
but only two features.
"""

from __future__ import annotations

from typing import Any, Dict, List

REACTOR_ML_SEED = 1

_REACTOR_GT_RAW: List[Dict[str, Any]] = [
    {"model_type": "linear", "params": {"alpha": 0.01, "l1_ratio": 0.5}},
    {"model_type": "svm", "params": {"C": 10.0, "epsilon": 0.05}},
    {"model_type": "cart", "params": {
        "max_depth": 8, "min_samples_leaf": 0.01, "max_features": 1.0,
    }},
    {"model_type": "rf", "params": {
        "n_estimators": 250, "max_depth": 12, "max_features": 0.6,
    }},
    {"model_type": "gbm", "params": {
        "learning_rate": 0.05, "max_depth": 4, "n_estimators": 250,
    }},
    {"model_type": "xgb", "params": {
        "learning_rate": 0.05, "max_depth": 4, "n_estimators": 250,
        "subsample": 0.9, "colsample_bytree": 0.8,
    }},
]

REACTOR_GT_ENSEMBLE_SPECS: List[Dict[str, Any]] = [
    {"model_type": s["model_type"],
     "params": {**s["params"], "random_state": REACTOR_ML_SEED}}
    for s in _REACTOR_GT_RAW
]

REACTOR_OUTCOME = "benzene_constraint"

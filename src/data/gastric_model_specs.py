"""
Gastric cancer ML hyperparameters aligned with constraint-learning v11.

Sources (Maragno et al. 2025 / constraint-learning repo):
- Table EC.10 / ``v11_performance.csv``: embedded constraint models (train split CV).
- Table EC.12 / ``v11_full_performance.csv``: ground-truth ensemble (full 461 arms CV).

Training seed matches CL ``run_MLmodels.py`` (seed=1).

- ``svm`` entries use ``LinearSVR`` (dual=False, squared-epsilon loss), not kernel SVR.
- RF ``max_features='auto'`` is mapped to ``1.0`` for modern sklearn (CL regression RF used all features).
- XGB entries omit ``learning_rate`` so xgboost uses its default (0.3), matching CL grids.
  ``tree_method='exact'`` matches legacy xgboost 1.x behavior under xgboost 3.x.
"""

from __future__ import annotations

from typing import Any, Dict, List

# CL run_MLmodels default seed
GASTRIC_ML_SEED = 1

# Table EC.10 — one winner per toxicity + OS (v11 train-split CV)
_GASTRIC_EMBED_RAW: List[Dict[str, Any]] = [
    {"model_type": "gbm", "model_params": {
        "learning_rate": 0.2, "max_depth": 2, "n_estimators": 20,
    }},  # DLT_PROP — gbm
    {"model_type": "linear", "model_params": {
        "alpha": 0.1, "l1_ratio": 0.7,
    }},  # BLOOD_4 — linear
    {"model_type": "rf", "model_params": {
        "n_estimators": 25, "max_depth": 4, "max_features": "auto",
    }},  # CONSTITUTIONAL_34 — rf
    {"model_type": "linear", "model_params": {
        "alpha": 1.0, "l1_ratio": 0.5,
    }},  # INFECTION_34 — linear
    {"model_type": "gbm", "model_params": {
        "learning_rate": 0.1, "max_depth": 4, "n_estimators": 20,
    }},  # GI_34 — gbm
    {"model_type": "gbm", "model_params": {
        "learning_rate": 0.1, "max_depth": 3, "n_estimators": 20,
    }},  # OS — gbm
]

# Table EC.12 — GT ensemble members (v11_full, full-cohort CV)
# Order per outcome: linear, svm, cart, rf, gbm, xgb
_GT_SPECS_RAW: Dict[str, List[Dict[str, Any]]] = {
    "dlt": [
        {"model_type": "linear", "params": {"alpha": 0.1, "l1_ratio": 0.6}},
        {"model_type": "svm", "params": {"C": 10}},
        {"model_type": "cart", "params": {
            "max_depth": 3, "min_samples_leaf": 0.04, "max_features": 1.0,
        }},
        {"model_type": "rf", "params": {
            "n_estimators": 500, "max_depth": 6, "max_features": "auto",
        }},
        {"model_type": "gbm", "params": {
            "learning_rate": 0.01, "max_depth": 5, "n_estimators": 250,
        }},
        {"model_type": "xgb", "params": {
            "colsample_bytree": 0.8, "gamma": 0.5, "max_depth": 4,
            "min_child_weight": 10, "n_estimators": 250, "subsample": 1.0,
        }},
    ],
    "blood": [
        {"model_type": "linear", "params": {"alpha": 0.1, "l1_ratio": 0.5}},
        {"model_type": "svm", "params": {"C": 100}},
        {"model_type": "cart", "params": {
            "max_depth": 3, "min_samples_leaf": 0.06, "max_features": 1.0,
        }},
        {"model_type": "rf", "params": {
            "n_estimators": 500, "max_depth": 8, "max_features": "auto",
        }},
        {"model_type": "gbm", "params": {
            "learning_rate": 0.025, "max_depth": 5, "n_estimators": 250,
        }},
        {"model_type": "xgb", "params": {
            "colsample_bytree": 1.0, "gamma": 0.5, "max_depth": 5,
            "min_child_weight": 1, "n_estimators": 250, "subsample": 0.8,
        }},
    ],
    "constitutional": [
        {"model_type": "linear", "params": {"alpha": 1.0, "l1_ratio": 0.4}},
        {"model_type": "svm", "params": {"C": 1}},
        {"model_type": "cart", "params": {
            "max_depth": 4, "min_samples_leaf": 0.06, "max_features": 0.6,
        }},
        {"model_type": "rf", "params": {
            "n_estimators": 500, "max_depth": 6, "max_features": "auto",
        }},
        {"model_type": "gbm", "params": {
            "learning_rate": 0.01, "max_depth": 5, "n_estimators": 250,
        }},
        {"model_type": "xgb", "params": {
            "colsample_bytree": 0.8, "gamma": 1.0, "max_depth": 4,
            "min_child_weight": 10, "n_estimators": 250, "subsample": 0.8,
        }},
    ],
    "infection": [
        {"model_type": "linear", "params": {"alpha": 1.0, "l1_ratio": 0.3}},
        {"model_type": "svm", "params": {"C": 10}},
        {"model_type": "cart", "params": {
            "max_depth": 3, "min_samples_leaf": 0.06, "max_features": 0.6,
        }},
        {"model_type": "rf", "params": {
            "n_estimators": 250, "max_depth": 6, "max_features": "auto",
        }},
        {"model_type": "gbm", "params": {
            "learning_rate": 0.01, "max_depth": 5, "n_estimators": 250,
        }},
        {"model_type": "xgb", "params": {
            "colsample_bytree": 1.0, "gamma": 1.0, "max_depth": 4,
            "min_child_weight": 10, "n_estimators": 250, "subsample": 0.8,
        }},
    ],
    "gi": [
        {"model_type": "linear", "params": {"alpha": 1.0, "l1_ratio": 0.7}},
        {"model_type": "svm", "params": {"C": 100}},
        {"model_type": "cart", "params": {
            "max_depth": 5, "min_samples_leaf": 0.06, "max_features": 0.6,
        }},
        {"model_type": "rf", "params": {
            "n_estimators": 250, "max_depth": 6, "max_features": "auto",
        }},
        {"model_type": "gbm", "params": {
            "learning_rate": 0.01, "max_depth": 6, "n_estimators": 250,
        }},
        {"model_type": "xgb", "params": {
            "colsample_bytree": 0.8, "gamma": 0.5, "max_depth": 5,
            "min_child_weight": 1, "n_estimators": 250, "subsample": 0.8,
        }},
    ],
    "os": [
        {"model_type": "linear", "params": {"alpha": 0.1, "l1_ratio": 0.8}},
        {"model_type": "svm", "params": {"C": 0.1}},
        {"model_type": "cart", "params": {
            "max_depth": 3, "min_samples_leaf": 0.02, "max_features": 0.8,
        }},
        {"model_type": "rf", "params": {
            "n_estimators": 250, "max_depth": 8, "max_features": "auto",
        }},
        {"model_type": "gbm", "params": {
            "learning_rate": 0.01, "max_depth": 5, "n_estimators": 250,
        }},
        {"model_type": "xgb", "params": {
            "colsample_bytree": 1.0, "gamma": 10.0, "max_depth": 4,
            "min_child_weight": 10, "n_estimators": 250, "subsample": 1.0,
        }},
    ],
}


def _with_seed(specs: List[Dict[str, Any]], *, param_key: str) -> List[Dict[str, Any]]:
    out = []
    for spec in specs:
        entry = {"model_type": spec["model_type"], param_key: dict(spec[param_key])}
        entry[param_key].setdefault("random_state", GASTRIC_ML_SEED)
        out.append(entry)
    return out


GASTRIC_EMBED_CONFIGS: List[Dict[str, Any]] = _with_seed(
    _GASTRIC_EMBED_RAW, param_key="model_params",
)
GT_ENSEMBLE_SPECS: Dict[str, List[Dict[str, Any]]] = {
    outcome: _with_seed(specs, param_key="params")
    for outcome, specs in _GT_SPECS_RAW.items()
}

# Table EC.11 in-sample R² (full cohort, percentile tox targets except OS)
GT_ENSEMBLE_PAPER_R2: Dict[str, Dict[str, float]] = {
    "dlt": {"linear": 0.301, "svm": 0.330, "cart": 0.250, "rf": 0.573, "gbm": 0.670, "xgb": 0.323},
    "blood": {"linear": 0.287, "svm": 0.351, "cart": 0.211, "rf": 0.701, "gbm": 0.813, "xgb": 0.446},
    "constitutional": {"linear": 0.139, "svm": 0.224, "cart": 0.246, "rf": 0.602, "gbm": 0.682, "xgb": 0.285},
    "infection": {"linear": 0.217, "svm": 0.303, "cart": 0.139, "rf": 0.514, "gbm": 0.588, "xgb": 0.247},
    "gi": {"linear": 0.201, "svm": 0.328, "cart": 0.238, "rf": 0.563, "gbm": 0.733, "xgb": 0.475},
    "os": {"linear": 0.528, "svm": 0.469, "cart": 0.421, "rf": 0.815, "gbm": 0.827, "xgb": 0.756},
}

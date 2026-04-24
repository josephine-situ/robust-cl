"""
Train ML models on (possibly perturbed) data.
Supports CART, Random Forest, XGBoost.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Union

ModelType = Union[
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
]


def train_model(X: np.ndarray,
                y: np.ndarray,
                model_type: str = "rf",
                params: dict = None) -> ModelType:
    """
    Train a tree-based model.

    Parameters
    ----------
    X : (n, d) training features
    y : (n,) training labels (possibly perturbed)
    model_type : "cart", "rf", or "xgb"
    params : model hyperparameters

    Returns
    -------
    Trained sklearn model
    """
    params = params or {}

    if model_type == "cart":
        model = DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
        )
    elif model_type == "xgb":
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, y)
    return model


def retrain_on_perturbed(X: np.ndarray,
                         y: np.ndarray,
                         delta: np.ndarray,
                         model_type: str = "rf",
                         params: dict = None) -> ModelType:
    """
    Retrain model on perturbed labels y + delta.
    """
    return train_model(X, y + delta, model_type, params)

def retrain_on_bootstrap(X: np.ndarray,
                         y: np.ndarray,
                         indices: np.ndarray,
                         model_type: str = "rf",
                         params: dict = None) -> ModelType:
    """
    Retrain model on bootstrap sample specified by indices.
    """
    return train_model(X[indices], y[indices], model_type, params)
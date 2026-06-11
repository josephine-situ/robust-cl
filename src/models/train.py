"""
Train ML models on (possibly perturbed) data.
Supports Linear, SVM, CART, Random Forest, XGB/GBM, and MLP.
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from typing import Union, Dict, Any

ModelType = Union[
    ElasticNet,
    SVR,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    MLPRegressor
]


def train_model(X: np.ndarray,
                y: np.ndarray,
                model_type: str = "rf",
                params: dict = None) -> ModelType:
    """
    Train a specified ML model.

    Parameters
    ----------
    X : (n, d) training features
    y : (n,) training labels (possibly perturbed)
    model_type : "linear", "svm", "cart", "rf", "gbm", or "mlp"
    params : model hyperparameters

    Returns
    -------
    Trained sklearn model
    """
    params = params or {}
    random_state = params.get("random_state", 42)

    if model_type == "linear":
        model = ElasticNet(
            alpha=params.get("alpha", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            random_state=random_state
        )
    elif model_type == "svm":
        model = SVR(
            C=params.get("C", 1.0)
        )
    elif model_type == "cart":
        model = DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", None),
            random_state=random_state,
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 5),
            max_features=params.get("max_features", 1.0),
            random_state=random_state,
        )
    elif model_type in ["xgb", "gbm"]:
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 50),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=random_state,
        )
    elif model_type == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
            random_state=random_state,
            max_iter=500
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, y)
    return model


def train_best_model_cv(X: np.ndarray,
                        y: np.ndarray,
                        param_grids: Dict[str, Dict[str, Any]],
                        random_state: int = 42,
                        return_params: bool = False) -> Union[ModelType, tuple]:
    """
    Train multiple models using 5-fold CV and return the best one across all types.
    """
    best_model = None
    best_score = -np.inf
    best_model_type = None
    best_params = None

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_type, grid in param_grids.items():
        if model_type == "linear":
            base_model = ElasticNet(random_state=random_state)
        elif model_type == "svm":
            base_model = SVR()
        elif model_type == "cart":
            base_model = DecisionTreeRegressor(random_state=random_state)
        elif model_type == "rf":
            base_model = RandomForestRegressor(random_state=random_state)
        elif model_type in ["xgb", "gbm"]:
            base_model = GradientBoostingRegressor(random_state=random_state)
        elif model_type == "mlp":
            base_model = MLPRegressor(random_state=random_state, max_iter=500)
        else:
            continue
            
        search = GridSearchCV(base_model, grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        search.fit(X, y)
        
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_model_type = model_type
            best_params = search.best_params_

    if return_params:
        return best_model, best_model_type, best_params
    return best_model


class EnsembleModel:
    """An ensemble model that averages the predictions of multiple base models."""
    def __init__(self, models):
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)


def train_fixed_ensemble(X: np.ndarray, y: np.ndarray, specs: list) -> EnsembleModel:
    """Train one model per spec and average predictions."""
    models = [train_model(X, y, s["model_type"], s.get("params", {})) for s in specs]
    return EnsembleModel(models)


def train_ensemble_model_cv(X: np.ndarray,
                            y: np.ndarray,
                            param_grids: Dict[str, Dict[str, Any]],
                            random_state: int = 42) -> EnsembleModel:
    """
    Train an ensemble model by finding the best parameter combination for each model class,
    and averaging their predictions.
    """
    best_models = []
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_type, grid in param_grids.items():
        if model_type == "linear":
            base_model = ElasticNet(random_state=random_state)
        elif model_type == "svm":
            base_model = SVR()
        elif model_type == "cart":
            base_model = DecisionTreeRegressor(random_state=random_state)
        elif model_type == "rf":
            base_model = RandomForestRegressor(random_state=random_state)
        elif model_type in ["xgb", "gbm"]:
            base_model = GradientBoostingRegressor(random_state=random_state)
        elif model_type == "mlp":
            base_model = MLPRegressor(random_state=random_state, max_iter=500)
        else:
            continue
            
        search = GridSearchCV(base_model, grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        search.fit(X, y)
        best_models.append(search.best_estimator_)

    return EnsembleModel(best_models)


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
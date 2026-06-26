"""
Train ML models on (possibly perturbed) data.
Supports Linear, SVM, CART, Random Forest, XGB/GBM, and MLP.
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from typing import Union, Dict, Any

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

ModelType = Union[
    ElasticNet,
    LinearSVR,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    MLPRegressor,
    Any,
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
    random_state = params.get("random_state", 1)

    if model_type == "linear":
        model = ElasticNet(
            alpha=params.get("alpha", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            random_state=random_state
        )
    elif model_type == "svm":
        model = LinearSVR(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 100_000),
            dual=params.get("dual", False),
            loss=params.get("loss", "squared_epsilon_insensitive"),
        )
    elif model_type == "cart":
        model = DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", None),
            random_state=random_state,
        )
    elif model_type == "rf":
        max_features = params.get("max_features", 1.0)
        if max_features == "auto":
            # CL v11 / legacy sklearn: regression RF used all features
            max_features = 1.0
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 5),
            max_features=max_features,
            random_state=random_state,
        )
    elif model_type == "gbm":
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 50),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=random_state,
        )
    elif model_type == "xgb":
        xgb_kwargs = {
            "n_estimators": params.get("n_estimators", 250),
            "max_depth": params.get("max_depth", 4),
            "colsample_bytree": params.get("colsample_bytree", 1.0),
            "gamma": params.get("gamma", 0.0),
            "min_child_weight": params.get("min_child_weight", 1),
            "subsample": params.get("subsample", 1.0),
            "tree_method": params.get("tree_method", "exact"),
            "objective": "reg:squarederror",
            "random_state": random_state,
            "verbosity": 0,
        }
        if "learning_rate" in params:
            xgb_kwargs["learning_rate"] = params["learning_rate"]
        if "reg_lambda" in params:
            xgb_kwargs["reg_lambda"] = params["reg_lambda"]
        if "reg_alpha" in params:
            xgb_kwargs["reg_alpha"] = params["reg_alpha"]
        model = XGBRegressor(**xgb_kwargs)
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
            base_model = LinearSVR(max_iter=100_000, dual=False, loss="squared_epsilon_insensitive")
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
            base_model = LinearSVR(max_iter=100_000, dual=False, loss="squared_epsilon_insensitive")
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


def generate_bootstrap_samples(n: int, P: int, seed: int = 42) -> list:
    """Generate P fixed bootstrap index vectors of length n."""
    rng = np.random.RandomState(seed)
    return [rng.choice(n, size=n, replace=True) for _ in range(P)]


def train_bootstrap_models(X: np.ndarray,
                           y: np.ndarray,
                           model_type: str,
                           params: dict,
                           bootstrap_indices: list,
                           seed: int = 42) -> list:
    """Train one model per bootstrap index vector."""
    models = []
    for p, idx in enumerate(bootstrap_indices):
        pparams = (params or {}).copy()
        pparams["random_state"] = seed + p
        models.append(retrain_on_bootstrap(X, y, idx, model_type, pparams))
    return models


def oob_worst_case_error(models: list,
                          bootstrap_indices: list,
                          X: np.ndarray,
                          y: np.ndarray) -> tuple:
    """
    For each model, compute worst OOB absolute error on training labels.
    Returns (best_model_index, best_worst_error, per_model_worst_errors).
    """
    n = len(y)
    n_models = len(models)
    oob_counts = np.zeros(n)
    oob_sum = np.zeros(n)
    oob_sq_sum = np.zeros(n)

    for model, idx in zip(models, bootstrap_indices):
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        if not oob_mask.any():
            continue
        preds = model.predict(X[oob_mask])
        oob_sum[oob_mask] += preds
        oob_sq_sum[oob_mask] += preds
        oob_counts[oob_mask] += 1

    per_model_worst = []
    for p, (model, idx) in enumerate(zip(models, bootstrap_indices)):
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        if not oob_mask.any():
            per_model_worst.append(np.inf)
            continue
        preds = model.predict(X[oob_mask])
        errors = np.abs(y[oob_mask] - preds)
        per_model_worst.append(float(np.max(errors)))

    best_idx = int(np.argmin(per_model_worst))
    return best_idx, per_model_worst[best_idx], per_model_worst


def _standardize(X_ref: np.ndarray, X_query: np.ndarray):
    """Z-score X_query using column-wise mean/std of X_ref.

    Columns with zero variance are left unchanged (avoid divide-by-zero).
    Returns the standardized versions of both X_ref and X_query.
    """
    mu = X_ref.mean(axis=0)
    sigma = X_ref.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_ref - mu) / sigma, (X_query - mu) / sigma


def localized_bootstrap_indices(X: np.ndarray,
                                x_star: np.ndarray,
                                k_neighbors_frac: float,
                                n_candidates: int,
                                seed: int = 42,
                                distance_feature_indices: list = None) -> list:
    """Generate bootstrap candidates resampling only from training points
    nearest to x_star in feature space.

    Distances are computed in the **standardized** feature space (z-scored by
    the training-set column mean/std) so that features on different scales
    contribute equally to the neighbour search.

    Parameters
    ----------
    distance_feature_indices : optional list of column indices used to compute
        the nearest-neighbor distance. When provided, only these features
        (e.g. context columns) drive localization; the remaining decision
        features are ignored. Defaults to all columns (full-vector distance).
    """
    n = X.shape[0]
    k = max(1, int(round(k_neighbors_frac * n)))
    x_star = np.asarray(x_star, dtype=float).ravel()
    cols = list(distance_feature_indices) if distance_feature_indices else None
    Xc = X[:, cols] if cols is not None else X
    xc = x_star[cols] if cols is not None else x_star
    Xc_std, xc_std = _standardize(Xc, xc.reshape(1, -1))
    distances = np.linalg.norm(Xc_std - xc_std, axis=1)
    pool = np.argsort(distances)[:k]
    rng = np.random.RandomState(seed)
    return [rng.choice(pool, size=n, replace=True) for _ in range(n_candidates)]
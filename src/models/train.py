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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
                params: dict = None,
                normalize: bool = True) -> ModelType:
    """
    Train a specified ML model.

    Parameters
    ----------
    X : (n, d) training features
    y : (n,) training labels (possibly perturbed)
    model_type : "linear", "svm", "cart", "rf", "gbm", "xgb", or "mlp"
    params : model hyperparameters
    normalize : if True (default), prepend a StandardScaler so the returned
        object is a Pipeline(StandardScaler, model).  The scaler is fitted on X.
        Linear and SVM models benefit most; tree models are scale-invariant but
        normalization ensures consistent behaviour when the model is embedded
        into a Gurobi MIP via embed.py.

    Returns
    -------
    Fitted sklearn estimator (or Pipeline when normalize=True)
    """
    params = params or {}
    random_state = params.get("random_state", 1)
    if isinstance(params.get("hidden_layer_sizes"), list):
        # JSON round-trips tuples to lists, and *_selected_configs.json is JSON.
        params = {**params, "hidden_layer_sizes": tuple(params["hidden_layer_sizes"])}

    if model_type == "linear":
        estimator = ElasticNet(
            alpha=params.get("alpha", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            random_state=random_state,
            max_iter=10_000,
        )
    elif model_type == "svm":
        estimator = LinearSVR(
            C=params.get("C", 1.0),
            # Read, not dropped: `epsilon` is in run_cv.py's grid, so CV selects a
            # value that this function used to ignore -- the fitted model was then
            # not the model that was cross-validated (fixed 2026-08-21).
            epsilon=params.get("epsilon", 0.0),
            max_iter=params.get("max_iter", 100_000),
            dual=params.get("dual", False),
            loss=params.get("loss", "squared_epsilon_insensitive"),
        )
    elif model_type == "cart":
        estimator = DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", None),
            random_state=random_state,
        )
    elif model_type == "rf":
        max_features = params.get("max_features", 1.0)
        if max_features == "auto":
            max_features = 1.0
        estimator = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 5),
            max_features=max_features,
            random_state=random_state,
        )
    elif model_type == "gbm":
        estimator = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 50),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 1.0),   # in the CV grid; see svm note
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
        estimator = XGBRegressor(**xgb_kwargs)
    elif model_type == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
            # `solver` and `alpha` are BOTH in run_cv.py's mlp grid, and both were
            # dropped here until 2026-08-21 -- so CV chose lbfgs/alpha=0.01 while
            # every downstream fit silently used adam/alpha=1e-4. That is not only
            # the wrong model, it is far slower on an unscaled target: one reactor
            # refit was 19.7s under adam against 1.4s under the selected lbfgs,
            # which is what made CP's B=200 bank look intractable there.
            solver=params.get("solver", "adam"),
            alpha=params.get("alpha", 1e-4),
            random_state=random_state,
            max_iter=params.get("max_iter", 10_000),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if normalize:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipe.fit(X, y)
        return pipe

    estimator.fit(X, y)
    return estimator


def _make_base_estimator(model_type: str, random_state: int):
    """Return a fresh, un-fitted sklearn estimator for the given model type."""
    if model_type == "linear":
        return ElasticNet(random_state=random_state, max_iter=10_000)
    elif model_type == "svm":
        return LinearSVR(max_iter=100_000, dual=False, loss="squared_epsilon_insensitive")
    elif model_type == "cart":
        return DecisionTreeRegressor(random_state=random_state)
    elif model_type == "rf":
        return RandomForestRegressor(random_state=random_state)
    elif model_type == "gbm":
        return GradientBoostingRegressor(random_state=random_state)
    elif model_type == "xgb":
        if XGBRegressor is None:
            return None
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=0,
        )
    elif model_type == "mlp":
        return MLPRegressor(random_state=random_state, max_iter=10_000)
    return None


def train_best_model_cv(X: np.ndarray,
                        y: np.ndarray,
                        param_grids: Dict[str, Dict[str, Any]],
                        random_state: int = 42,
                        return_params: bool = False,
                        scoring: str = "r2",
                        cv_folds: int = 5,
                        return_all_scores: bool = False) -> Union[ModelType, tuple]:
    """
    Train multiple models using k-fold CV and return the best one across all types.

    Each model type is wrapped in a Pipeline(StandardScaler, estimator) so that
    features are normalised before fitting.  The ``model__`` prefix is stripped
    from GridSearchCV's ``best_params_`` before returning, so callers receive
    clean hyperparameter dicts (e.g. ``{"alpha": 1.0}`` not ``{"model__alpha": 1.0}``).

    Parameters
    ----------
    scoring : sklearn scoring string, e.g. ``'r2'`` or ``'neg_mean_squared_error'``.
    return_all_scores : if True, also return a dict mapping model_type -> best CV score
        for that type. Useful for producing EC.6-style comparison tables.
    """
    best_pipeline = None
    best_score = -np.inf
    best_model_type = None
    best_params = None
    all_type_scores: Dict[str, float] = {}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for model_type, grid in param_grids.items():
        base_est = _make_base_estimator(model_type, random_state)
        if base_est is None:
            continue

        pipe = Pipeline([("scaler", StandardScaler()), ("model", base_est)])
        prefixed_grid = {f"model__{k}": v for k, v in grid.items()}

        search = GridSearchCV(pipe, prefixed_grid, cv=kf, scoring=scoring, n_jobs=-1)
        search.fit(X, y)

        cv_score = float(search.best_score_)
        all_type_scores[model_type] = cv_score

        if cv_score > best_score:
            best_score = cv_score
            best_pipeline = search.best_estimator_
            best_model_type = model_type
            # Strip the "model__" prefix so callers get clean param dicts
            best_params = {
                k.replace("model__", ""): v
                for k, v in search.best_params_.items()
            }

    if return_all_scores:
        if return_params:
            return best_pipeline, best_model_type, best_params, all_type_scores
        return best_pipeline, all_type_scores

    if return_params:
        return best_pipeline, best_model_type, best_params
    return best_pipeline


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
                            random_state: int = 42,
                            scoring: str = "r2",
                            cv_folds: int = 5) -> EnsembleModel:
    """
    Train an ensemble model by finding the best parameter combination for each model
    class and averaging their predictions.  Each model type is fitted inside a
    Pipeline(StandardScaler, estimator) for consistent normalisation.
    """
    best_pipelines = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for model_type, grid in param_grids.items():
        base_est = _make_base_estimator(model_type, random_state)
        if base_est is None:
            continue

        pipe = Pipeline([("scaler", StandardScaler()), ("model", base_est)])
        prefixed_grid = {f"model__{k}": v for k, v in grid.items()}

        search = GridSearchCV(pipe, prefixed_grid, cv=kf, scoring=scoring, n_jobs=-1)
        search.fit(X, y)
        best_pipelines.append(search.best_estimator_)

    return EnsembleModel(best_pipelines)


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


def generate_bootstrap_samples(n: int, P: int, seed: int = 42,
                               frac: float = 0.5) -> list:
    """Generate P fixed bootstrap index vectors, each of length ``round(frac*n)``.

    ``frac`` is the proportion of the training rows drawn per replicate, with
    replacement. The default 0.5 matches Maragno et al. (2025) Sec. 4.4.1, which
    specifies "a bootstrap sample (proportion = 0.5) of the underlying data" for
    the model-wrapper ensemble; ``frac=1.0`` is the classical n-out-of-n
    bootstrap. Half-size replicates overlap less (an n-out-of-n draw carries only
    ~63.2% unique rows), so the P models spread wider and the (1-alpha)-of-P
    constraint binds harder at matched (P, alpha).

    Only the legacy ``scenario_source: "bootstrap"`` path reaches this; the
    production wrapper draws from the shared set D via ``ScenarioBank``.
    """
    rng = np.random.RandomState(seed)
    m = max(1, int(round(frac * n)))
    return [rng.choice(n, size=m, replace=True) for _ in range(P)]


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


def resolve_neighbor_pool_size(n: int,
                               k_neighbors_frac: float,
                               k_neighbors_min: int = 1) -> int:
    """Number of training arms in the localized bootstrap pool."""
    k = max(k_neighbors_min, int(round(k_neighbors_frac * n)))
    return min(k, n)


def localized_bootstrap_indices(X: np.ndarray,
                                x_star: np.ndarray,
                                k_neighbors_frac: float,
                                n_candidates: int,
                                seed: int = 42,
                                distance_feature_indices: list = None,
                                k_neighbors_min: int = 1) -> list:
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
    k = resolve_neighbor_pool_size(n, k_neighbors_frac, k_neighbors_min)
    x_star = np.asarray(x_star, dtype=float).ravel()
    cols = list(distance_feature_indices) if distance_feature_indices else None
    Xc = X[:, cols] if cols is not None else X
    xc = x_star[cols] if cols is not None else x_star
    Xc_std, xc_std = _standardize(Xc, xc.reshape(1, -1))
    distances = np.linalg.norm(Xc_std - xc_std, axis=1)
    pool = np.argsort(distances)[:k]
    rng = np.random.RandomState(seed)
    return [rng.choice(pool, size=n, replace=True) for _ in range(n_candidates)]
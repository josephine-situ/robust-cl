"""
Embed trained tree-based models into Gurobi MIO.

For a single decision tree with leaves L:
    f(x) = sum_l mu_l * z_l
    sum_l z_l = 1
    z_l = 1 => x in R_l  (leaf region)
    z_l in {0,1}

For ensembles (RF, XGB):
    f(x) = (1/T) sum_t f_t(x)   [RF]
    f(x) = sum_t f_t(x)          [XGB]
    Each f_t embedded separately, predictions aggregated.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from typing import Union

ModelType = Union[
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
]


def _extract_tree_structure(tree):
    """
    Extract leaves and their defining split conditions.

    Returns
    -------
    leaves : list of dict, each with:
        'value': prediction value at this leaf
        'bounds_lower': (d,) lower bounds on x for this leaf
        'bounds_upper': (d,) upper bounds on x for this leaf
    """
    tree_ = tree.tree_
    n_features = tree_.n_features

    leaves = []

    def recurse(node, lb, ub):
        if tree_.children_left[node] == tree_.children_right[node]:
            # Leaf node
            leaves.append({
                "value": tree_.value[node].flatten()[0],
                "bounds_lower": lb.copy(),
                "bounds_upper": ub.copy(),
            })
            return

        feature = tree_.feature[node]
        threshold = tree_.threshold[node]

        # Left child: x[feature] <= threshold
        ub_left = ub.copy()
        ub_left[feature] = min(ub_left[feature], threshold)
        recurse(tree_.children_left[node], lb, ub_left)

        # Right child: x[feature] > threshold
        lb_right = lb.copy()
        lb_right[feature] = max(lb_right[feature], threshold)
        recurse(tree_.children_right[node], lb_right, ub)

    lb_init = np.full(n_features, -np.inf)
    ub_init = np.full(n_features, np.inf)
    recurse(0, lb_init, ub_init)

    return leaves


def embed_single_tree(model: gp.Model,
                      tree: DecisionTreeRegressor,
                      x_vars: list,
                      var_lb: np.ndarray,
                      var_ub: np.ndarray,
                      name_prefix: str = "tree") -> gp.Var:
    """
    Embed a single decision tree into a Gurobi model.

    Parameters
    ----------
    model : Gurobi model
    tree : trained DecisionTreeRegressor (or a single
        tree from an ensemble)
    x_vars : list of Gurobi variables for x
    var_lb, var_ub : variable bounds
    name_prefix : prefix for variable/constraint names

    Returns
    -------
    f_var : Gurobi variable representing f(x; tree)
    """
    leaves = _extract_tree_structure(tree)
    n_leaves = len(leaves)
    d = len(x_vars)

    # Binary variables: which leaf is x in?
    z = model.addVars(
        n_leaves, vtype=GRB.BINARY, name=f"{name_prefix}_z"
    )

    # Exactly one leaf
    model.addConstr(
        gp.quicksum(z[l] for l in range(n_leaves)) == 1,
        name=f"{name_prefix}_one_leaf",
    )

    # Leaf region constraints (big-M)
    for l, leaf in enumerate(leaves):
        for j in range(d):
            lb_leaf = max(leaf["bounds_lower"][j], var_lb[j])
            ub_leaf = min(leaf["bounds_upper"][j], var_ub[j])

            M_lower = var_lb[j] - lb_leaf
            M_upper = ub_leaf - var_ub[j]

            # x[j] >= lb_leaf - M * (1 - z[l])
            if lb_leaf > var_lb[j]:
                model.addConstr(
                    x_vars[j] >= lb_leaf - (lb_leaf - var_lb[j]) * (1 - z[l]),
                    name=f"{name_prefix}_lb_{l}_{j}",
                )

            # x[j] <= ub_leaf + M * (1 - z[l])
            if ub_leaf < var_ub[j]:
                model.addConstr(
                    x_vars[j] <= ub_leaf + (var_ub[j] - ub_leaf) * (1 - z[l]),
                    name=f"{name_prefix}_ub_{l}_{j}",
                )

    # Prediction variable
    f_var = model.addVar(
        lb=-GRB.INFINITY, name=f"{name_prefix}_pred"
    )
    model.addConstr(
        f_var == gp.quicksum(
            leaves[l]["value"] * z[l] for l in range(n_leaves)
        ),
        name=f"{name_prefix}_pred_def",
    )

    return f_var


def embed_model(model: gp.Model,
                ml_model: ModelType,
                x_vars: list,
                var_lb: np.ndarray,
                var_ub: np.ndarray,
                name_prefix: str = "model") -> gp.Var:
    """
    Embed any supported model into Gurobi.

    Returns a Gurobi variable representing f(x; ml_model).
    """
    if isinstance(ml_model, DecisionTreeRegressor):
        return embed_single_tree(
            model, ml_model, x_vars, var_lb, var_ub, name_prefix
        )

    elif isinstance(ml_model, RandomForestRegressor):
        tree_preds = []
        for t, estimator in enumerate(ml_model.estimators_):
            f_t = embed_single_tree(
                model, estimator, x_vars, var_lb, var_ub,
                name_prefix=f"{name_prefix}_t{t}",
            )
            tree_preds.append(f_t)

        # Average
        f_var = model.addVar(
            lb=-GRB.INFINITY, name=f"{name_prefix}_pred"
        )
        T = len(tree_preds)
        model.addConstr(
            f_var == (1.0 / T) * gp.quicksum(tree_preds),
            name=f"{name_prefix}_avg",
        )
        return f_var

    elif isinstance(ml_model, GradientBoostingRegressor):
        tree_preds = []
        for t, estimator_arr in enumerate(ml_model.estimators_):
            estimator = estimator_arr[0]  # single output
            f_t = embed_single_tree(
                model, estimator, x_vars, var_lb, var_ub,
                name_prefix=f"{name_prefix}_t{t}",
            )
            tree_preds.append(f_t)

        # Sum (with learning rate) + init
        f_var = model.addVar(
            lb=-GRB.INFINITY, name=f"{name_prefix}_pred"
        )
        lr = ml_model.learning_rate
        init = ml_model.init_.constant_[0][0]
        model.addConstr(
            f_var == init + lr * gp.quicksum(tree_preds),
            name=f"{name_prefix}_sum",
        )
        return f_var

    else:
        raise ValueError(f"Unsupported model type: {type(ml_model)}")
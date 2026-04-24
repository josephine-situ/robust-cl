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
from typing import Union, List, Dict

ModelType = Union[
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
]


def _extract_tree_structure(tree):
    """
    Extract leaves and their defining split conditions.
    """
    if hasattr(tree, "tree_"):
        tree_ = tree.tree_
    else:
        tree_ = tree

    n_features = tree_.n_features

    leaves = []

    def recurse(node, lb, ub):
        if tree_.children_left[node] == tree_.children_right[node]:
            # Leaf node
            leaves.append({
                "id": node,
                "value": tree_.value[node].flatten()[0] if hasattr(tree_.value, "flatten") else tree_.value[node][0][0],
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


def compute_valid_alpha(scenario_model, b):
    """
    Compute the maximum alpha such that the voting cut
    is valid (doesn't exclude any feasible x).
    
    Returns alpha_valid.
    """
    if not hasattr(scenario_model, "estimators_"):
        return 0.0

    T = len(scenario_model.estimators_)
    
    # For each tree, find min leaf prediction and min "bad" prediction
    tree_info = []
    for t, estimator in enumerate(scenario_model.estimators_):
        leaves = _extract_tree_structure(estimator)
        leaf_values = [leaf['value'] for leaf in leaves]
        
        mu_min = min(leaf_values)
        bad_leaves = [v for v in leaf_values if v > b]
        
        if bad_leaves:
            mu_bad_min = min(bad_leaves)
            cost = mu_bad_min - mu_min
            tree_info.append({
                'tree': t,
                'mu_min': mu_min,
                'mu_bad_min': mu_bad_min,
                'cost': cost,
            })
        else:
            # This tree can never be "bad" — no leaf exceeds b
            tree_info.append({
                'tree': t,
                'mu_min': mu_min,
                'mu_bad_min': None,
                'cost': float('inf'),
            })
    
    # Total budget: how much room if all trees at minimum
    total_min = sum(info['mu_min'] for info in tree_info)
    budget = T * b - total_min
    
    if budget < 0:
        # No feasible x exists for this model — any alpha works
        return 0.0
    
    # Greedily assign trees to "bad", cheapest first
    candidates = [info for info in tree_info 
                  if info['cost'] < float('inf')]
    candidates.sort(key=lambda x: x['cost'])
    
    k_max = 0
    remaining_budget = budget
    for info in candidates:
        if info['cost'] <= remaining_budget:
            remaining_budget -= info['cost']
            k_max += 1
        else:
            break
    
    alpha_valid = k_max / T
    return alpha_valid

def choose_cut_type(scenario_model, x_k, b):
    """
    Decide which cut to add based on the scenario structure.
    """
    alpha_valid = compute_valid_alpha(scenario_model, b)
    
    # Count bad trees at x_k
    preds = [est.predict(x_k.reshape(1, -1))[0] 
             for est in scenario_model.estimators_]
    bad_fraction = sum(1 for p in preds if p > b) / len(preds)
    
    gap = bad_fraction - alpha_valid
    
    # Use thresholds as detailed in the prompt
    if gap > 0.1:
        return "voting"
    elif gap > 0:
        return "bad_leaf"
    else:
        return "full"

# --- Phase cuts logic ---

def embed_cut_voting(model: gp.Model, ml_model: ModelType, x_vars: list, var_lb: np.ndarray, var_ub: np.ndarray, b: float, name_prefix: str):
    """Phase 1: Voting Cuts. Approximate the ensemble prediction by counting dangerous trees."""
    # Simplified Voting cut: sum of indicators
    if not isinstance(ml_model, RandomForestRegressor):
        return  # Only support RF for voting right now
    
    n_trees = len(ml_model.estimators_)
    z_tree = list(model.addVars(n_trees, vtype=GRB.BINARY, name=f"{name_prefix}_v").values())
    alpha_valid = compute_valid_alpha(ml_model, b)
    
    for t, tree in enumerate(ml_model.estimators_):
        leaves = _extract_tree_structure(tree)
        bad_leaves = [l for l in leaves if l['value'] > b]
        
        # If x is in any bad leaf, z_tree[t] can be 1
        for i, leaf in enumerate(bad_leaves):
            z_out = model.addVars(len(x_vars), 2, vtype=GRB.BINARY, name=f"{name_prefix}_bl_{t}_{i}")
            for j in range(len(x_vars)):
                if leaf["bounds_lower"][j] > var_lb[j]:
                    # To relax the constraint x_j <= L_j - e, we need it to hold up to x_j = var_ub[j]
                    M_lower = var_ub[j] - leaf["bounds_lower"][j] + 1e-4
                    model.addConstr(x_vars[j] <= leaf["bounds_lower"][j] - 1e-4 + M_lower * (1 - z_out[j, 0] + z_tree[t]))
                else:
                    model.addConstr(z_out[j, 0] == 0)
                    
                if leaf["bounds_upper"][j] < var_ub[j]:
                    # To relax x_j >= U_j + e, we need it to hold down to x_j = var_lb[j]
                    M_upper = leaf["bounds_upper"][j] - var_lb[j] + 1e-4
                    model.addConstr(x_vars[j] >= leaf["bounds_upper"][j] + 1e-4 - M_upper * (1 - z_out[j, 1] + z_tree[t]))
                else:
                    model.addConstr(z_out[j, 1] == 0)
                    
            model.addConstr(gp.quicksum(z_out[j, k] for j in range(len(x_vars)) for k in range(2)) >= 1)
            
    model.addConstr(gp.quicksum(z_tree) <= int(alpha_valid * len(z_tree)))


def embed_cut_bad_leaf(model: gp.Model, ml_model: ModelType, x_vars: list, var_lb: np.ndarray, var_ub: np.ndarray, b: float, name_prefix: str):
    """Phase 2: Bad-leaf Cuts. Add no-good cuts for leaves whose values are strictly unreachable in a feasible solution."""
    if isinstance(ml_model, DecisionTreeRegressor):
        estimators = [ml_model]
        b_valid = [b]
    elif hasattr(ml_model, "estimators_"):
        if isinstance(ml_model, GradientBoostingRegressor):
            estimators = [e[0] for e in ml_model.estimators_]
            lr = ml_model.learning_rate
            init = ml_model.init_.constant_[0][0]
            # Sum of trees <= (b - init) / lr
            margin = (b - init) / lr
            
            mu_mins = []
            for t in estimators:
                leaves = _extract_tree_structure(t)
                mu_mins.append(min([l["value"] for l in leaves]))
            total_min = sum(mu_mins)
            b_valid = [margin - (total_min - m) for m in mu_mins]
            
        elif isinstance(ml_model, RandomForestRegressor):
            estimators = ml_model.estimators_
            T = len(estimators)
            
            mu_mins = []
            for t in estimators:
                leaves = _extract_tree_structure(t)
                mu_mins.append(min([l["value"] for l in leaves]))
            total_min = sum(mu_mins)
            b_valid = [T * b - (total_min - m) for m in mu_mins]
        else:
            return
    else:
        return
        
    for t, tree in enumerate(estimators):
        leaves = _extract_tree_structure(tree)
        thresh = b_valid[t]
        for i, leaf in enumerate(leaves):
            if leaf["value"] > thresh:
                # Add a cut to prevent x from falling exactly in this leaf's bounds
                # We need at least one feature to be outside the leaf bounds
                z_out = model.addVars(len(x_vars), 2, vtype=GRB.BINARY, name=f"{name_prefix}_bl_{t}_{i}")
                for j in range(len(x_vars)):
                    if leaf["bounds_lower"][j] > var_lb[j]:
                        M_lower = var_ub[j] - leaf["bounds_lower"][j] + 1e-4
                        model.addConstr(x_vars[j] <= leaf["bounds_lower"][j] - 1e-4 + M_lower * (1 - z_out[j, 0]))
                    else:
                        model.addConstr(z_out[j, 0] == 0)
                        
                    if leaf["bounds_upper"][j] < var_ub[j]:
                        M_upper = leaf["bounds_upper"][j] - var_lb[j] + 1e-4
                        model.addConstr(x_vars[j] >= leaf["bounds_upper"][j] + 1e-4 - M_upper * (1 - z_out[j, 1]))
                    else:
                        model.addConstr(z_out[j, 1] == 0)
                        
                model.addConstr(gp.quicksum(z_out[j, k] for j in range(len(x_vars)) for k in range(2)) >= 1)

# --- Full Embedding (Phase 3) ---

def embed_single_tree(model: gp.Model,
                      tree: DecisionTreeRegressor,
                      x_vars: list,
                      var_lb: np.ndarray,
                      var_ub: np.ndarray,
                      name_prefix: str = "tree",
                      rho: float = 0.0) -> gp.Var:
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
            lb_orig = leaf["bounds_lower"][j]
            ub_orig = leaf["bounds_upper"][j]

            # Shrink bounds for parameter robustness according to rho ||a_j x||_q
            # Simplified version for when we have axis-aligned splits.
            if rho > 0:
                if lb_orig > -np.inf:
                    lb_leaf_tight = lb_orig / (1 - rho) if lb_orig >= 0 else lb_orig / (1 + rho)
                else:
                    lb_leaf_tight = -np.inf

                if ub_orig < np.inf:
                    ub_leaf_tight = ub_orig / (1 + rho) if ub_orig >= 0 else ub_orig / (1 - rho)
                else:
                    ub_leaf_tight = np.inf
            else:
                lb_leaf_tight = lb_orig
                ub_leaf_tight = ub_orig

            lb_leaf = max(lb_leaf_tight, var_lb[j])
            ub_leaf = min(ub_leaf_tight, var_ub[j])

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
                name_prefix: str = "model",
                rho: float = 0.0) -> gp.Var:
    """
    Embed any supported model into Gurobi.

    Returns a Gurobi variable representing f(x; ml_model).
    """
    if isinstance(ml_model, DecisionTreeRegressor):
        return embed_single_tree(
            model, ml_model, x_vars, var_lb, var_ub, name_prefix, rho
        )

    elif isinstance(ml_model, RandomForestRegressor):
        tree_preds = []
        for t, estimator in enumerate(ml_model.estimators_):
            f_t = embed_single_tree(
                model, estimator, x_vars, var_lb, var_ub,
                name_prefix=f"{name_prefix}_t{t}", rho=rho
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
                name_prefix=f"{name_prefix}_t{t}", rho=rho
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
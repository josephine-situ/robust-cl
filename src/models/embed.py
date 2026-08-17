"""
Embed trained ML models into Gurobi MIO.

Supported model types
---------------------
- ElasticNet (linear)  : f(x) = intercept + coef'x
- LinearSVR  (svm)     : f(x) = intercept + coef'x
- DecisionTreeRegressor: leaf selection via binary z variables (big-M)
- RandomForestRegressor: average of embedded trees
- GradientBoostingRegressor: sum of embedded trees
- XGBRegressor         : sum of embedded trees (parsed from JSON dump)
- MLPRegressor         : ReLU network via binary activation variables (big-M)
- Pipeline             : normalization constraints for the inner StandardScaler
                         followed by recursive embedding of the wrapped estimator

For decision trees and ensembles:
    z_l = 1  =>  x in leaf region R_l   (big-M constraints)
    sum_l z_l = 1                        (exactly one leaf)
    f(x) = sum_l mu_l * z_l

For MLP (relu):
    h_j^l = ReLU(W^l h^{l-1} + b^l)   (modelled with binary indicator per neuron)
"""

import json
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from typing import Union, List, Dict

try:
    from xgboost import XGBRegressor as _XGBRegressor
except ImportError:
    _XGBRegressor = None

ModelType = Union[
    ElasticNet,
    LinearSVR,
    MLPRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
]


def _get_inner_model(ml_model):
    """Return the underlying estimator, unwrapping a sklearn Pipeline if needed."""
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(ml_model, Pipeline):
            return ml_model[-1]
    except ImportError:
        pass
    return ml_model


# SPLIT_EPS -- width of the band excluded between sibling leaf boxes.
#
# Every split is a strict inequality on one side (sklearn sends x > threshold
# right, XGBoost sends x < threshold to "yes"), but a box can only encode a
# closed bound. Encoding both sides closed makes x exactly ON a threshold
# admissible in two sibling leaves at once, and the solver then takes whichever
# leaf value relaxes the constraint -- measured gaps up to 5.4e-02 against
# sklearn on gastric, on 5 of 13 anchors. It is the optimizer's preferred point,
# so it is systematic, not measure-zero. Excluding a band around the threshold
# makes the boxes genuinely disjoint.
#
# BOTH sides are pushed off the threshold, not just the strict one. A bound
# left sitting exactly at theta is satisfied by Gurobi down to theta - tol, so
# points just below theta get routed to the >= branch while the model routes
# them to the < branch -- a misrouting rather than an exclusion, and the
# optimizer will find it. Margining only the strict side left gastric infection
# at 1.31e-02. Each side therefore needs its own margin > FeasibilityTol.
#
# The width is bounded on both sides:
#
#   lower -- must exceed FeasibilityTol per side (so > 2 * FeasibilityTol
#     between the boxes). Gurobi may violate EACH box by up to FeasibilityTol.
#     At the default 1e-6 that is 1e-6 per side. A narrower band is not merely
#     weak, it is invisible. (Tightening FeasibilityTol as well made things
#     worse, so it is left at Gurobi's default; IntFeasTol is a separate matter
#     -- see EMBED_INTFEASTOL below.)
#
#   upper -- must stay below the smallest distance from a training row to a
#     threshold, on EITHER side, or a real row falls in the band and no leaf
#     admits it. Measured over all (row, tree, node) triples on gastric the tail
#     is empty between 1e-6 and 1e-4, so any width in that range costs the same
#     -- see COST below for what it costs at 1e-5.
#
# 1e-5 sits 5x above the lower bound and inside the empty stretch below the
# upper one. Synthetic has more room still (min strict-side distance 7.33e-05).
#
# This depends on gastric_v11.CTX_DECIMALS: before that rounding, FRAC_MALE
# carried float32 round-trip duplicates (0.689999998 alongside 0.69) and trees
# split between them, leaving gaps of ~1e-7 that no band could clear.
#
# The band is CLAMPED against the training rows reaching each node (see
# _split_edges), so it never costs a row its leaf:
#
#     ub_low = max(lo, thr - SPLIT_EPS)      lb_high = min(hi, thr + SPLIT_EPS)
#
# This is needed because XGBoost derives thresholds from histogram bin
# boundaries and stores them in float32, so a threshold hugs an adjacent data
# value to within ~1e-8, on EITHER side depending on the node (e.g. Pub_Year at
# thr - lo = 8.06e-10). Whichever side the hugged value sits on has no room for
# an unclamped band. The straddling gaps are wide (hi - lo runs 0.16-0.22), so
# this is an artifact of where the threshold was stored, not of dense data.
# Unclamped, 6 rows of 320 lost their leaf on gastric.
#
# It must be the clamp, NOT ub_low = lo / lb_high = hi: the latter would exclude
# the whole inter-data gap at every one of ~220 nodes per model, collapsing the
# feasible set onto the data.
#
# Clamping is sound only while min(hi - lo) > MIN_SPLIT_SEP across all nodes --
# a threshold hugging lo AND hi would leave the boxes overlapping. That is the
# degenerate-split condition gastric_v11.CTX_DECIMALS removed (count now 0);
# _split_edges falls back to the plain band if it ever recurs.
#
# The branch a row takes must be decided the way the LIBRARY decides it, not
# just the way the threshold reads. BOTH libraries traverse in float32 --
# XGBoost holds features as float32 in the DMatrix, and sklearn casts X to
# tree._tree.DTYPE (float32) inside apply/predict -- so a row sitting a float32
# ulp from a threshold (float32 eps is 1.2e-7 near 1.0; the gastric minimum gap
# is 8.06e-10) routes as if equal to it. Deciding that in float64 put one
# infection row in the wrong leaf (2.2e-03 off XGBoost's predict, against
# 1.6e-07 elsewhere) and misrouted rows in a default-parameter RF on gastric
# (1.3e-02). XGBoost compares float32 against float32; sklearn compares the
# float32 value against a float64 threshold, hence the two casts differ.
# Checked by routing all 320 training rows through the parsed boxes and summing
# the selected leaves: max |diff| vs predict() is 1.9e-07 over the four nominal
# xgb outcomes and 2.6e-07 over 32 refits on perturbed labels (the models CP
# cuts and the wrapper replicates actually embed), with nothing unrouted or
# ambiguous. Synthetic's RF is exact at 0 in both. The residual is XGBoost's own
# float32 leaf summation, not a routing difference. MLP, LinearSVR, single CART
# and GBM are supported here but exercised by neither problem, so untested.
#
# X_ref costs ~9.7 ms per constraint-model, about 1 s over a full CP or wrapper
# run. One matrix serves every scenario under scenario_source "noise", since
# delta perturbs labels and leaves X untouched. Passing a stale or resampled
# matrix is safe: float32 rounding is monotone, so low.max() < high.min()
# whatever rows are supplied, and a wrong X can cost a row its leaf but can
# never misroute one.
SPLIT_EPS = 1e-5

# Minimum separation the clamped bands may leave between sibling boxes, in
# scaled units. Must stay above 2 * FeasibilityTol or the boxes overlap within
# tolerance and the leaf tie returns.
MIN_SPLIT_SEP = 2e-6

def _split_edges(low, high, threshold):
    """Band edges around `threshold`, clamped so no reaching row is excluded.

    Returns (ub_low, lb_high): the upper bound of the low child's box and the
    lower bound of the high child's. Without rows this is the plain
    +/- SPLIT_EPS band; with them each edge is pulled back toward the threshold
    just far enough to keep the nearest reaching row admissible.

    low / high are the split feature's values for the rows the LIBRARY routes to
    each branch. The caller applies that comparison, so this stays agnostic
    about < vs <= and about the precision it is decided in -- which matters:
    XGBoost holds features as float32 and compares in float32, so a row can be
    below the threshold in float64 and still be routed as if equal to it.
    """
    ub = threshold - SPLIT_EPS
    lb = threshold + SPLIT_EPS
    c_ub = ub if low is None or low.size == 0 else max(ub, float(low.max()))
    c_lb = lb if high is None or high.size == 0 else min(lb, float(high.min()))
    # float32 rounding is monotone, so low.max() < high.min() however the branch
    # was decided, and neither edge can cross to the wrong side of the split.
    # What the clamp can do is bring the two edges too close together, when a
    # threshold hugs BOTH its neighbours -- then fall back to the plain band and
    # accept losing the row, rather than letting the boxes overlap again.
    if c_lb - c_ub < MIN_SPLIT_SEP:
        return ub, lb
    return c_ub, c_lb


def _tol_from_env(name, default=None):
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


# Solver tolerances, applied to any model that receives an embedded tree (see
# _embed_leaves) and overridable per run by the env vars below. None leaves
# Gurobi's own default in place.
#
# FeasibilityTol: left alone. Its default 1e-6 is what SPLIT_EPS is sized
# against, and tightening it to 1e-9 measurably made fidelity WORSE (gastric
# infection went 7.4e-08 -> 3.3e-02).
#
# IntFeasTol: pinned to the floor. A leaf box is big-M, so integrality slack
# becomes x-slack of M * IntFeasTol, and M reaches ~46 here -- at Gurobi's
# default 1e-5 that is ~4.6e-04 of room to sit outside the selected leaf, enough
# to flip it. With x free over the whole box a solver maximising an embedded
# production model extracted 1.618e-02 on gastric infection; at 1e-9 the same
# search yields 1.397e-08. It does NOT show up inside the anchored master (both
# settings read 1.066e-07 there) because pinning the context block leaves too
# little freedom -- so this was measured with x free, which is the adversarial
# regime CP's master approaches.
EMBED_FEASTOL = _tol_from_env("ROBCL_EMBED_FEASTOL")
EMBED_INTFEASTOL = _tol_from_env("ROBCL_EMBED_INTFEASTOL", default=1e-9)


def _extract_tree_structure(tree, Z=None):
    """
    Extract leaves and their defining split conditions.

    Z, when given, is the SCALED training matrix; split bands are then clamped
    against the rows reaching each node so no training row loses its leaf.
    """
    if hasattr(tree, "tree_"):
        tree_ = tree.tree_
    else:
        tree_ = tree

    n_features = tree_.n_features

    leaves = []

    def recurse(node, lb, ub, idx):
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

        # sklearn sends x <= threshold left and x > threshold right. BOTH bounds
        # are pushed off the threshold -- see the note at the top for why
        # margining only the strict side is not enough.
        left_idx = right_idx = None
        low = high = None
        if Z is not None:
            v = Z[idx, feature]
            # sklearn casts X to float32 (tree._tree.DTYPE) before traversing and
            # compares that against the float64 threshold, so route the same way
            # -- deciding it in float64 sent rows within a float32 ulp of a
            # threshold down the wrong branch.
            goes_left = v.astype(np.float32) <= threshold
            left_idx, right_idx = idx[goes_left], idx[~goes_left]
            low, high = v[goes_left], v[~goes_left]
        ub_edge, lb_edge = _split_edges(low, high, threshold)

        ub_left = ub.copy()
        ub_left[feature] = min(ub_left[feature], ub_edge)
        recurse(tree_.children_left[node], lb, ub_left, left_idx)

        lb_right = lb.copy()
        lb_right[feature] = max(lb_right[feature], lb_edge)
        recurse(tree_.children_right[node], lb_right, ub, right_idx)

    lb_init = np.full(n_features, -np.inf)
    ub_init = np.full(n_features, np.inf)
    recurse(0, lb_init, ub_init, None if Z is None else np.arange(Z.shape[0]))

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
            
            # Keep track of which variable conditions are active (not fixed)
            active_z = []
            
            for j in range(len(x_vars)):
                # If variable is fixed, we can just skip adding logical big-M constraints
                if var_lb[j] == var_ub[j]:
                    model.addConstr(z_out[j, 0] == 0)
                    model.addConstr(z_out[j, 1] == 0)
                    continue

                if leaf["bounds_lower"][j] > var_lb[j]:
                    # To relax the constraint x_j <= L_j - e, we need it to hold up to x_j = var_ub[j]
                    M_lower = var_ub[j] - leaf["bounds_lower"][j] + 1e-4
                    model.addConstr(x_vars[j] <= leaf["bounds_lower"][j] - 1e-4 + M_lower * (1 - z_out[j, 0] + z_tree[t]))
                    active_z.append(z_out[j, 0])
                else:
                    model.addConstr(z_out[j, 0] == 0)
                    
                if leaf["bounds_upper"][j] < var_ub[j]:
                    # To relax x_j >= U_j + e, we need it to hold down to x_j = var_lb[j]
                    M_upper = leaf["bounds_upper"][j] - var_lb[j] + 1e-4
                    model.addConstr(x_vars[j] >= leaf["bounds_upper"][j] + 1e-4 - M_upper * (1 - z_out[j, 1] + z_tree[t]))
                    active_z.append(z_out[j, 1])
                else:
                    model.addConstr(z_out[j, 1] == 0)
            
            if active_z:
                model.addConstr(gp.quicksum(active_z) >= 1)
            
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
                
                active_z = []
                
                for j in range(len(x_vars)):
                    if var_lb[j] == var_ub[j]:
                        model.addConstr(z_out[j, 0] == 0)
                        model.addConstr(z_out[j, 1] == 0)
                        continue

                    if leaf["bounds_lower"][j] > var_lb[j]:
                        M_lower = var_ub[j] - leaf["bounds_lower"][j] + 1e-4
                        model.addConstr(x_vars[j] <= leaf["bounds_lower"][j] - 1e-4 + M_lower * (1 - z_out[j, 0]))
                        active_z.append(z_out[j, 0])
                    else:
                        model.addConstr(z_out[j, 0] == 0)
                        
                    if leaf["bounds_upper"][j] < var_ub[j]:
                        M_upper = leaf["bounds_upper"][j] - var_lb[j] + 1e-4
                        model.addConstr(x_vars[j] >= leaf["bounds_upper"][j] + 1e-4 - M_upper * (1 - z_out[j, 1]))
                        active_z.append(z_out[j, 1])
                    else:
                        model.addConstr(z_out[j, 1] == 0)
                        
                if active_z:
                    model.addConstr(gp.quicksum(active_z) >= 1)

# --- Full Embedding (Phase 3) ---

def _embed_leaves(model: gp.Model,
                  leaves: list,
                  x_vars: list,
                  var_lb: np.ndarray,
                  var_ub: np.ndarray,
                  name_prefix: str = "tree",
                  rho: float = 0.0) -> gp.Var:
    """
    Embed a pre-extracted list of leaf dicts into Gurobi.

    Each leaf dict must have keys: 'value', 'bounds_lower', 'bounds_upper'.
    This is the core embedding shared by embed_single_tree and embed_xgb.
    """
    # Set here rather than at each call site so every path that embeds a tree
    # (nominal, CP master, wrapper, prediction checks) stays consistent.
    if EMBED_FEASTOL is not None and model.Params.FeasibilityTol > EMBED_FEASTOL:
        model.Params.FeasibilityTol = EMBED_FEASTOL
    if EMBED_INTFEASTOL is not None and model.Params.IntFeasTol > EMBED_INTFEASTOL:
        model.Params.IntFeasTol = EMBED_INTFEASTOL

    d = len(x_vars)

    z = {}
    valid_leaves = []

    for l, leaf in enumerate(leaves):
        feasible_leaf = True
        for j in range(d):
            lb_orig = leaf["bounds_lower"][j]
            ub_orig = leaf["bounds_upper"][j]

            if rho > 0:
                lb_tight = (lb_orig / (1 - rho) if lb_orig >= 0 else lb_orig / (1 + rho)) if lb_orig > -np.inf else -np.inf
                ub_tight = (ub_orig / (1 + rho) if ub_orig >= 0 else ub_orig / (1 - rho)) if ub_orig < np.inf else np.inf
            else:
                lb_tight = lb_orig
                ub_tight = ub_orig

            if var_lb[j] > ub_tight + 1e-6 or var_ub[j] < lb_tight - 1e-6:
                feasible_leaf = False
                break

        if feasible_leaf:
            valid_leaves.append(l)

    if len(valid_leaves) == 0:
        f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
        model.addConstr(f_var == 0, name=f"{name_prefix}_pred_inf")
        return f_var

    for l in valid_leaves:
        z[l] = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_z{l}")

    model.addConstr(
        gp.quicksum(z[l] for l in valid_leaves) == 1,
        name=f"{name_prefix}_one_leaf",
    )

    for l in valid_leaves:
        leaf = leaves[l]
        for j in range(d):
            if var_lb[j] == var_ub[j]:
                continue

            lb_orig = leaf["bounds_lower"][j]
            ub_orig = leaf["bounds_upper"][j]

            if rho > 0:
                lb_leaf_tight = (lb_orig / (1 - rho) if lb_orig >= 0 else lb_orig / (1 + rho)) if lb_orig > -np.inf else -np.inf
                ub_leaf_tight = (ub_orig / (1 + rho) if ub_orig >= 0 else ub_orig / (1 - rho)) if ub_orig < np.inf else np.inf
            else:
                lb_leaf_tight = lb_orig
                ub_leaf_tight = ub_orig

            lb_leaf = max(lb_leaf_tight, var_lb[j])
            ub_leaf = min(ub_leaf_tight, var_ub[j])

            if lb_leaf > var_lb[j]:
                model.addConstr(
                    x_vars[j] >= lb_leaf - (lb_leaf - var_lb[j]) * (1 - z[l]),
                    name=f"{name_prefix}_lb{l}_{j}",
                )
            if ub_leaf < var_ub[j]:
                model.addConstr(
                    x_vars[j] <= ub_leaf + (var_ub[j] - ub_leaf) * (1 - z[l]),
                    name=f"{name_prefix}_ub{l}_{j}",
                )

    f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    model.addConstr(
        f_var == gp.quicksum(leaves[l]["value"] * z[l] for l in valid_leaves),
        name=f"{name_prefix}_pred_def",
    )
    return f_var


def embed_single_tree(model: gp.Model,
                      tree: DecisionTreeRegressor,
                      x_vars: list,
                      var_lb: np.ndarray,
                      var_ub: np.ndarray,
                      name_prefix: str = "tree",
                      rho: float = 0.0,
                      Z: np.ndarray = None) -> gp.Var:
    """
    Embed a single sklearn DecisionTreeRegressor into a Gurobi model.

    Returns a Gurobi variable representing f(x; tree).
    """
    leaves = _extract_tree_structure(tree, Z)
    return _embed_leaves(model, leaves, x_vars, var_lb, var_ub, name_prefix, rho)


def embed_linear(model: gp.Model,
                 ml_model: ElasticNet,
                 x_vars: list,
                 name_prefix: str = "linear") -> gp.Var:
    """Embed a fitted ElasticNet: f(x) = intercept + coef'x."""
    coef = np.asarray(ml_model.coef_).ravel()
    intercept = float(np.asarray(ml_model.intercept_).ravel()[0])
    f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    model.addConstr(
        f_var == intercept + gp.quicksum(float(coef[j]) * x_vars[j] for j in range(len(coef))),
        name=f"{name_prefix}_pred_def",
    )
    return f_var


def embed_svm(model: gp.Model,
              ml_model: LinearSVR,
              x_vars: list,
              name_prefix: str = "svm") -> gp.Var:
    """Embed a fitted LinearSVR: f(x) = intercept + coef'x."""
    coef = np.asarray(ml_model.coef_).ravel()
    intercept = float(np.asarray(ml_model.intercept_).ravel()[0])
    f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    model.addConstr(
        f_var == intercept + gp.quicksum(float(coef[j]) * x_vars[j] for j in range(len(coef))),
        name=f"{name_prefix}_pred_def",
    )
    return f_var


def _compute_mlp_big_m(ml_model: MLPRegressor,
                       var_lb: np.ndarray,
                       var_ub: np.ndarray) -> list:
    """
    Compute per-hidden-layer big-M bounds for MLP embedding.

    Uses L1 weight-norm propagation: given an element-wise upper bound on
    |h| at each layer, the maximum absolute pre-activation at the next layer
    is bounded by |W|' * h_bound + |b|.  After ReLU, h_bound is updated to
    the new pre-activation bound (ReLU clips negatives, so max(0, pre) <= pre).
    """
    coefs = ml_model.coefs_       # list of (n_prev, n_curr) weight matrices
    biases = ml_model.intercepts_ # list of (n_curr,) bias vectors

    h_bound = np.maximum(np.abs(var_lb), np.abs(var_ub))  # element-wise input bound

    M_vals = []
    for l in range(len(coefs) - 1):  # hidden layers only, not the output layer
        W = coefs[l]   # (n_prev, n_curr)
        b = biases[l]  # (n_curr,)
        pre_bound = np.abs(W).T @ h_bound + np.abs(b)  # (n_curr,)
        M_vals.append(float(pre_bound.max()) + 1.0)
        h_bound = pre_bound  # post-ReLU bound (h <= pre for active neurons)
    return M_vals


def embed_mlp(model: gp.Model,
              ml_model: MLPRegressor,
              x_vars: list,
              var_lb: np.ndarray,
              var_ub: np.ndarray,
              name_prefix: str = "mlp") -> gp.Var:
    """
    Embed a fitted MLPRegressor (relu activation) as MIP constraints.

    Each hidden neuron j in layer l is encoded with a binary indicator s_j^l:
        s_j = 1  =>  h_j = pre_j  (neuron active)
        s_j = 0  =>  h_j = 0      (neuron inactive / clipped by ReLU)
    The big-M value per layer is computed from L1 weight-norm propagation.
    """
    if ml_model.activation != "relu":
        raise ValueError(
            f"MLP embedding only supports 'relu' activation, got '{ml_model.activation}'"
        )

    coefs = ml_model.coefs_
    biases = ml_model.intercepts_
    n_hidden = len(coefs) - 1

    M_vals = _compute_mlp_big_m(ml_model, var_lb, var_ub)

    h = list(x_vars)  # current-layer outputs (Gurobi vars)

    for l in range(n_hidden):
        W = coefs[l]    # (n_prev, n_curr)
        b = biases[l]   # (n_curr,)
        n_curr = W.shape[1]
        M = M_vals[l]

        h_next = []
        for j in range(n_curr):
            # Pre-activation
            pre_j = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_l{l}_pre{j}")
            model.addConstr(
                pre_j == float(b[j])
                + gp.quicksum(float(W[i, j]) * h[i] for i in range(len(h))),
                name=f"{name_prefix}_l{l}_predef{j}",
            )
            # ReLU: h_j = max(0, pre_j) via binary s_j
            s_j = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_l{l}_s{j}")
            h_j = model.addVar(lb=0.0, name=f"{name_prefix}_l{l}_h{j}")
            model.addConstr(h_j >= pre_j,              name=f"{name_prefix}_l{l}_relu_lo{j}")
            model.addConstr(h_j <= M * s_j,            name=f"{name_prefix}_l{l}_relu_M1{j}")
            model.addConstr(h_j <= pre_j + M*(1-s_j),  name=f"{name_prefix}_l{l}_relu_M2{j}")
            h_next.append(h_j)

        h = h_next

    # Output layer (linear, no activation)
    W_out = coefs[-1]   # (n_last, n_out)
    b_out = biases[-1]  # (n_out,)
    f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    model.addConstr(
        f_var == float(b_out[0])
        + gp.quicksum(float(W_out[i, 0]) * h[i] for i in range(len(h))),
        name=f"{name_prefix}_pred_def",
    )
    return f_var


# ---------------------------------------------------------------------------
# XGBoost embedding helpers
# ---------------------------------------------------------------------------

def _parse_xgb_json_tree(node: dict,
                          lb: np.ndarray,
                          ub: np.ndarray,
                          n_features: int,
                          leaves: list,
                          Z: np.ndarray = None,
                          idx: np.ndarray = None) -> None:
    """
    Recursively parse one XGBoost JSON tree node, collecting leaves with
    their tight feature-space bounds.

    XGBoost convention: "yes" branch satisfies  x[feat] < split_condition,
                        "no"  branch satisfies  x[feat] >= split_condition.

    Z, when given, is the SCALED training matrix; split bands are then clamped
    against the rows reaching each node so no training row loses its leaf. idx
    carries that row subset down the recursion and is derived from Z at the root.
    """
    if Z is not None and idx is None:
        idx = np.arange(Z.shape[0])
    if "leaf" in node:
        leaves.append({
            "value": float(node["leaf"]),
            "bounds_lower": lb.copy(),
            "bounds_upper": ub.copy(),
        })
        return

    feat_str = str(node["split"])
    try:
        feat_idx = int(feat_str[1:]) if feat_str.startswith("f") else int(feat_str)
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse XGBoost feature name '{feat_str}'")

    threshold = float(node["split_condition"])
    yes_id = int(node["yes"])
    no_id = int(node["no"])

    children_by_id = {int(c["nodeid"]): c for c in node.get("children", [])}

    # XGBoost sends x < threshold to "yes" and x >= threshold to "no". BOTH
    # bounds are pushed off the threshold -- see the note at the top of this
    # module for why margining only the strict side is not enough.
    yes_idx = no_idx = None
    low = high = None
    if Z is not None:
        v = Z[idx, feat_idx]
        # XGBoost's DMatrix is float32 and the comparison happens there, so a
        # row within a float32 ulp below the threshold routes as if equal to it.
        # Deciding this in float64 sent one gastric row to the wrong leaf.
        goes_yes = v.astype(np.float32) < np.float32(threshold)
        yes_idx, no_idx = idx[goes_yes], idx[~goes_yes]
        low, high = v[goes_yes], v[~goes_yes]
    ub_edge, lb_edge = _split_edges(low, high, threshold)

    ub_yes = ub.copy()
    ub_yes[feat_idx] = min(ub_yes[feat_idx], ub_edge)
    _parse_xgb_json_tree(children_by_id[yes_id], lb.copy(), ub_yes, n_features,
                         leaves, Z, yes_idx)

    lb_no = lb.copy()
    lb_no[feat_idx] = max(lb_no[feat_idx], lb_edge)
    _parse_xgb_json_tree(children_by_id[no_id], lb_no, ub.copy(), n_features,
                         leaves, Z, no_idx)


def embed_xgb(model: gp.Model,
              ml_model,
              x_vars: list,
              var_lb: np.ndarray,
              var_ub: np.ndarray,
              name_prefix: str = "xgb",
              rho: float = 0.0,
              Z: np.ndarray = None) -> gp.Var:
    """
    Embed a fitted XGBRegressor as a sum of embedded decision trees.

    Prediction: f(x) = base_score + sum_t leaf_t(x)

    Each tree is parsed from XGBoost's JSON dump and embedded with the same
    big-M leaf-region constraints as sklearn trees.
    """
    booster = ml_model.get_booster()
    dump = booster.get_dump(dump_format="json")
    n_features = len(x_vars)

    # The intercept. XGBoost >= 2.0 leaves the sklearn attribute None and keeps
    # the LEARNED intercept (the label mean for squared loss) in the booster
    # config, as a bracketed string like '[5.015625E-1]'. Falling back to 0.5
    # biased every embedded xgb prediction by |mean(y) - 0.5|; the old
    # ``getattr(...) or 0.5`` also silently replaced a legitimate base_score=0.0.
    base_score = None
    try:
        cfg = json.loads(booster.save_config())
        raw = cfg["learner"]["learner_model_param"]["base_score"]
        base_score = float(str(raw).strip("[]"))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        pass
    if base_score is None:
        attr = getattr(ml_model, "base_score", None)
        base_score = 0.5 if attr is None else float(attr)

    tree_pred_vars = []
    for t_idx, tree_str in enumerate(dump):
        tree_dict = json.loads(tree_str)
        leaves = []
        _parse_xgb_json_tree(
            tree_dict,
            var_lb.copy(), var_ub.copy(),
            n_features, leaves, Z,
        )
        f_t = _embed_leaves(
            model, leaves, x_vars, var_lb, var_ub,
            name_prefix=f"{name_prefix}_t{t_idx}", rho=rho,
        )
        tree_pred_vars.append(f_t)

    f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    model.addConstr(
        f_var == float(base_score) + gp.quicksum(tree_pred_vars),
        name=f"{name_prefix}_sum",
    )
    return f_var


def embed_model(model: gp.Model,
                ml_model,
                x_vars: list,
                var_lb: np.ndarray,
                var_ub: np.ndarray,
                name_prefix: str = "model",
                rho: float = 0.0,
                X_ref: np.ndarray = None) -> gp.Var:
    """
    Embed any supported model into Gurobi.

    Supported types: Pipeline (with StandardScaler), ElasticNet, LinearSVR,
    DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor,
    XGBRegressor, MLPRegressor.

    X_ref is the model's own training matrix in the SAME units as x_vars. When
    given, tree split bands are clamped against it so no training row is left
    without an admissible leaf (see the SPLIT_EPS note at the top). It is scaled
    alongside var_lb / var_ub on the way through a Pipeline, and ignored by
    non-tree models.

    Returns a Gurobi variable representing f(x; ml_model).
    """
    # --- Pipeline: add normalisation constraints then embed inner model ---
    try:
        from sklearn.pipeline import Pipeline as _Pipeline
        if isinstance(ml_model, _Pipeline):
            scaler = ml_model.named_steps.get("scaler")
            inner = ml_model.named_steps.get("model", ml_model[-1])
            if scaler is not None and hasattr(scaler, "mean_"):
                mu = scaler.mean_
                sigma = scaler.scale_
                d = len(x_vars)
                z_vars = [
                    model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_scl{j}")
                    for j in range(d)
                ]
                for j in range(d):
                    model.addConstr(
                        z_vars[j] == (x_vars[j] - float(mu[j])) / float(sigma[j]),
                        name=f"{name_prefix}_norm{j}",
                    )
                z_lb = (var_lb - mu) / sigma
                z_ub = (var_ub - mu) / sigma
                z_ref = None if X_ref is None else (np.asarray(X_ref) - mu) / sigma
                return embed_model(
                    model, inner, z_vars, z_lb, z_ub,
                    name_prefix=f"{name_prefix}_m", rho=rho, X_ref=z_ref,
                )
            else:
                return embed_model(
                    model, inner, x_vars, var_lb, var_ub,
                    name_prefix=name_prefix, rho=rho, X_ref=X_ref,
                )
    except ImportError:
        pass

    # --- Linear models ---
    if isinstance(ml_model, ElasticNet):
        return embed_linear(model, ml_model, x_vars, name_prefix)

    elif isinstance(ml_model, LinearSVR):
        return embed_svm(model, ml_model, x_vars, name_prefix)

    # --- Neural network ---
    elif isinstance(ml_model, MLPRegressor):
        return embed_mlp(model, ml_model, x_vars, var_lb, var_ub, name_prefix)

    # --- Tree-based models ---
    elif isinstance(ml_model, DecisionTreeRegressor):
        return embed_single_tree(model, ml_model, x_vars, var_lb, var_ub,
                                 name_prefix, rho, X_ref)

    elif isinstance(ml_model, RandomForestRegressor):
        tree_preds = []
        for t, estimator in enumerate(ml_model.estimators_):
            f_t = embed_single_tree(
                model, estimator, x_vars, var_lb, var_ub,
                name_prefix=f"{name_prefix}_t{t}", rho=rho, Z=X_ref,
            )
            tree_preds.append(f_t)
        T = len(tree_preds)
        f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
        model.addConstr(
            f_var == (1.0 / T) * gp.quicksum(tree_preds),
            name=f"{name_prefix}_avg",
        )
        return f_var

    elif isinstance(ml_model, GradientBoostingRegressor):
        tree_preds = []
        for t, estimator_arr in enumerate(ml_model.estimators_):
            estimator = estimator_arr[0]
            f_t = embed_single_tree(
                model, estimator, x_vars, var_lb, var_ub,
                name_prefix=f"{name_prefix}_t{t}", rho=rho, Z=X_ref,
            )
            tree_preds.append(f_t)
        lr = ml_model.learning_rate
        init = ml_model.init_.constant_[0][0]
        f_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
        model.addConstr(
            f_var == init + lr * gp.quicksum(tree_preds),
            name=f"{name_prefix}_sum",
        )
        return f_var

    # --- XGBoost ---
    elif _XGBRegressor is not None and isinstance(ml_model, _XGBRegressor):
        return embed_xgb(model, ml_model, x_vars, var_lb, var_ub, name_prefix,
                         rho, X_ref)

    else:
        raise ValueError(f"Unsupported model type: {type(ml_model)}")


def embedded_prediction_at_point(
    ml_model: ModelType,
    x_point: np.ndarray,
    var_lb: np.ndarray,
    var_ub: np.ndarray,
    rho: float = 0.0,
) -> float:
    """Evaluate an embedded model at a fixed x by fixing all decision variables."""
    x_point = np.asarray(x_point, dtype=float).ravel()
    d = len(x_point)
    m = gp.Model("embed_check")
    m.Params.OutputFlag = 0
    x_vars = [
        m.addVar(lb=var_lb[j], ub=var_ub[j], name=f"x_{j}")
        for j in range(d)
    ]
    f_var = embed_model(m, ml_model, x_vars, var_lb, var_ub, name_prefix="chk", rho=rho)
    for j in range(d):
        m.addConstr(x_vars[j] == float(x_point[j]), name=f"fix_{j}")
    m.setObjective(f_var, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Embedding check failed with Gurobi status {m.Status}")
    return float(f_var.X)


def verify_embedded_predictions(instance, configs, n_points: int = 5, seed: int = 0):
    """
    Confirm embedded predictions match sklearn at random feasible points.
    Uses optimizer model configs (Table EC.10), not GT ensemble models.
    """
    try:
        from ..models.train import train_model
    except ImportError:
        from src.models.train import train_model

    rng = np.random.RandomState(seed)
    X = instance.X_train
    y = instance.constraints[0].models_data[0].y_train
    lines = []
    max_err = 0.0

    for cfg in configs:
        mtype = cfg["model_type"]
        params = dict(cfg.get("model_params", {}))
        params.setdefault("random_state", 42)
        model = train_model(X, y, mtype, params)
        errs = []
        for _ in range(n_points):
            idx = rng.randint(0, len(X))
            x_pt = X[idx].copy()
            sk = float(model.predict(x_pt.reshape(1, -1))[0])
            gu = embedded_prediction_at_point(
                model, x_pt, instance.variable_lb, instance.variable_ub,
            )
            errs.append(abs(sk - gu))
        max_err = max(max_err, max(errs))
        lines.append(
            f"{mtype:6s} max|sklearn-embedded|={max(errs):.2e}  mean={np.mean(errs):.2e}"
        )
    lines.append(f"overall max error: {max_err:.2e}")
    return lines
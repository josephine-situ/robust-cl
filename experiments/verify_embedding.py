"""Verify that embedded models agree with sklearn, including ON split thresholds.

The pre-existing check (`embed.verify_embedded_predictions`) evaluates at random
training rows and minimises f_var. That misses the failure mode a MIP actually
exercises: a tree split is a strict inequality on one side, and if both child
boxes are closed at the threshold then x == threshold satisfies TWO leaves and
the optimizer picks whichever is cheaper. Real data points essentially never sit
on a threshold, so the bug is invisible there and systematic in the master.

An MLP has the same shape of failure at a different place: `embed_mlp` encodes
each ReLU with a big-M binary, and the analogue of "x sits on a split" is a
hidden unit whose PRE-ACTIVATION is exactly zero, where that binary is free. The
tie is benign in principle (both branches give h = 0), but only exactly at zero
-- integrality slack turns it into `M * IntFeasTol` of h-slack on either side,
and M is propagated from the input box, so it is the quantity worth measuring.
Since CV selects `mlp` for BOTH synthetic and reactor (2026-08-21), that encoding
is now load-bearing for every number those two instances produce.

This script therefore evaluates at BOTH:
  - random training rows, and
  - points constructed to lie exactly ON a decision boundary of the model --
    a split threshold for a tree/ensemble, a zero pre-activation for an MLP,
each time solving the embedded model twice (minimise and maximise f_var). Any
gap between those two is leaf/branch ambiguity; any offset from sklearn is an
encoding error. A correct embedding gives min == max == sklearn.

The model verified is the one the experiments actually embed: on synthetic and
reactor that is the CV selection (`results/cv/{problem}_selected_configs.json`,
reached through the same `run_sweep` instance builders the sweep uses), NOT
`config.yaml`'s `model` block. Linear and SVM models have no boundary case and
report "(n/a)".

    python experiments/verify_embedding.py                 # gastric constraint models
    python experiments/verify_embedding.py --problem synthetic
    python experiments/verify_embedding.py --problem reactor
"""

import argparse
import json
import os
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neural_network import MLPRegressor

from src.models.embed import embed_model, _extract_tree_structure, _parse_xgb_json_tree
from src.models.train import train_model
from experiments.run_adversary_probe import build_instance, outcome_rows, _unwrap
from src.data.instances import load_config


def embedded_range_at(ml_model, x_point, var_lb, var_ub):
    """(min, max) of the embedded prediction with x pinned -- equal iff unambiguous."""
    out = []
    for sense in (GRB.MINIMIZE, GRB.MAXIMIZE):
        m = gp.Model("chk")
        m.Params.OutputFlag = 0
        d = len(x_point)
        xv = [m.addVar(lb=var_lb[j], ub=var_ub[j], name=f"x{j}") for j in range(d)]
        f = embed_model(m, ml_model, xv, var_lb, var_ub, name_prefix="chk")
        for j in range(d):
            m.addConstr(xv[j] == float(x_point[j]))
        m.setObjective(f, sense)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return None, None
        out.append(float(f.X))
    return out[0], out[1]


def threshold_points(ml_model, X, var_lb, var_ub, n_points, rng):
    """Training rows with one coordinate moved exactly onto a split threshold.

    Thresholds live in the SCALED space the trees see, so they are mapped back
    through the scaler to raw x. These are the points the optimizer gravitates to.
    """
    scaler, est = _unwrap(ml_model)
    if scaler is not None:
        sc = scaler.named_steps["scaler"]
        mu, sigma = sc.mean_, sc.scale_
    else:
        mu, sigma = np.zeros(X.shape[1]), np.ones(X.shape[1])

    splits = []          # (feature, threshold_in_scaled_space)
    if hasattr(est, "get_booster"):
        for ts in est.get_booster().get_dump(dump_format="json"):
            stack = [json.loads(ts)]
            while stack:
                nd = stack.pop()
                if "leaf" in nd:
                    continue
                fs = str(nd["split"])
                splits.append((int(fs[1:]) if fs.startswith("f") else int(fs),
                               float(nd["split_condition"])))
                stack.extend(nd.get("children", []))
    else:
        trees = [t for t in np.asarray(getattr(est, "estimators_", [est])).ravel()
                 if hasattr(t, "tree_")]
        for t in trees[:10]:
            tr = t.tree_
            for nd in range(tr.node_count):
                if tr.children_left[nd] != tr.children_right[nd]:
                    splits.append((int(tr.feature[nd]), float(tr.threshold[nd])))
    if not splits:
        return []

    pts = []
    for _ in range(n_points):
        x = X[rng.randint(len(X))].astype(float).copy()
        j, thr = splits[rng.randint(len(splits))]
        x[j] = thr * sigma[j] + mu[j]          # exactly on the split, in raw units
        if var_lb[j] <= x[j] <= var_ub[j]:
            pts.append(x)
    return pts


def _mlp_preactivations(est, Xs):
    """Per-hidden-layer pre-activation matrices for a fitted MLPRegressor.

    Mirrors `embed_mlp`: z_l = h_{l-1} W_l + b_l, h_l = max(z_l, 0), skipping the
    linear output layer. Inputs are in the SCALED space the net sees.
    """
    a = np.asarray(Xs, dtype=float)
    zs = []
    for W, b in zip(est.coefs_[:-1], est.intercepts_[:-1]):
        z = a @ W + b
        zs.append(z)
        a = np.maximum(z, 0.0)
    return zs


def mlp_kink_points(ml_model, X, var_lb, var_ub, n_points, rng):
    """Points where some hidden unit's pre-activation is exactly zero.

    Constructed by bisection along the segment between two training rows: every
    pre-activation is continuous and piecewise-linear along that segment, so a
    sign change between the endpoints brackets a kink at ANY layer -- which a
    closed-form solve would only reach for layer 1. 80 halvings put t at machine
    precision, so |z| lands near 1e-16 relative: the tie the big-M binary sees.
    """
    scaler, est = _unwrap(ml_model)
    if not isinstance(est, MLPRegressor):
        return []
    if scaler is not None:
        sc = scaler.named_steps["scaler"]
        mu, sigma = sc.mean_, sc.scale_
    else:
        mu, sigma = np.zeros(X.shape[1]), np.ones(X.shape[1])
    Xs = (np.asarray(X, dtype=float) - mu) / sigma

    pts = []
    for _ in range(40 * n_points):
        if len(pts) >= n_points:
            break
        a, b = Xs[rng.randint(len(Xs))], Xs[rng.randint(len(Xs))]
        za = _mlp_preactivations(est, a[None, :])
        zb = _mlp_preactivations(est, b[None, :])
        cand = [(l, k) for l in range(len(za)) for k in range(za[l].shape[1])
                if (za[l][0, k] > 0) != (zb[l][0, k] > 0)]
        if not cand:
            continue
        l, k = cand[rng.randint(len(cand))]
        lo, hi, pos_lo = 0.0, 1.0, za[l][0, k] > 0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            zm = _mlp_preactivations(est, (a + mid * (b - a))[None, :])[l][0, k]
            if (zm > 0) == pos_lo:
                lo = mid
            else:
                hi = mid
        x = (a + 0.5 * (lo + hi) * (b - a)) * sigma + mu
        if np.all(x >= var_lb) and np.all(x <= var_ub):
            pts.append(x)
    return pts


def boundary_points(ml_model, X, var_lb, var_ub, n_points, rng):
    """(case_label, points) -- the model family's own ambiguous set."""
    _, est = _unwrap(ml_model)
    if isinstance(est, MLPRegressor):
        return "on-kink", mlp_kink_points(ml_model, X, var_lb, var_ub,
                                          n_points, rng)
    return "on-threshold", threshold_points(ml_model, X, var_lb, var_ub,
                                            n_points, rng)


def resolve_instance(config, args):
    """(instance, fallback_model_type, fallback_params, provenance_note).

    Synthetic and reactor go through `src.data.instances` so the model verified
    is the one every method embeds. `run_adversary_probe.build_instance` cannot be
    used for them: it reads a `data.synthetic` key config.yaml does not have (so it
    silently falls back to n_train=200) and never loads the CV selection at all.
    """
    if args.problem == "synthetic":
        from src.data.instances import synth_instance, synth_model_spec
        mt, mp, from_cv = synth_model_spec(config)
        return (synth_instance(config), mt, mp,
                "synthetic_selected_configs.json" if from_cv else "config.yaml")
    if args.problem == "reactor":
        from src.data.instances import reactor_instance, reactor_model_spec
        mt, mp, from_cv = reactor_model_spec(config)
        return (reactor_instance(config), mt, mp,
                "reactor_selected_configs.json" if from_cv else "config.yaml")
    return (build_instance(config, args), config["default_model"]["type"],
            config["default_model"]["params"], "gastric_selected_configs.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=["gastric", "synthetic", "reactor"],
                   default="gastric")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--n-rows", type=int, default=25)
    p.add_argument("--n-thresholds", type=int, default=25,
                   help="Boundary points per outcome: split thresholds, or ReLU "
                        "kinks for an MLP")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json",
                   help="Gastric only; synthetic and reactor resolve their own "
                        "results/cv/{problem}_selected_configs.json")
    p.add_argument("--gt-cv-configs",
                   default="results/cv/gastric_gt_ensemble_configs.json")
    args = p.parse_args()

    config = load_config(args.config)
    inst, mt, mp, provenance = resolve_instance(config, args)
    rows = outcome_rows(inst, mt, mp)
    lb, ub = inst.variable_lb.astype(float), inst.variable_ub.astype(float)
    rng = np.random.RandomState(0)

    print("=" * 78)
    print("EMBEDDING VERIFICATION  |sklearn - embedded|, and the embedded min/max "
          "spread")
    print(f"problem={args.problem}   embedded models from: {provenance}")
    print("=" * 78)
    print(f"{'outcome':<28}{'type':<8}{'case':<12}{'max err':>12}{'max spread':>13}")

    worst_err = worst_spread = 0.0
    for (c_idx, name, md, m_type, m_params, is_con, rhs) in rows:
        # trained exactly as _train_nominal_with_configs does
        model = train_model(md.X_train, md.y_train, m_type, m_params)
        b_label, b_pts = boundary_points(model, md.X_train, lb, ub,
                                         args.n_thresholds, rng)
        cases = {
            "data rows": [md.X_train[i] for i in
                          rng.choice(len(md.X_train), args.n_rows, replace=False)],
            b_label: b_pts,
        }
        for case, pts in cases.items():
            errs, spreads = [], []
            for x in pts:
                lo, hi = embedded_range_at(model, x, lb, ub)
                if lo is None:
                    continue
                sk = float(model.predict(np.atleast_2d(x))[0])
                errs.append(max(abs(sk - lo), abs(sk - hi)))
                spreads.append(hi - lo)
            if not errs:
                print(f"{name:<28}{m_type:<8}{case:<12}{'(n/a)':>12}")
                continue
            e, s = max(errs), max(spreads)
            worst_err, worst_spread = max(worst_err, e), max(worst_spread, s)
            flag = "  <-- FAIL" if (e > 1e-6 or s > 1e-6) else ""
            print(f"{name:<28}{m_type:<8}{case:<12}{e:>12.2e}{s:>13.2e}{flag}")

    print("-" * 78)
    print(f"{'OVERALL':<48}{worst_err:>12.2e}{worst_spread:>13.2e}")
    ok = worst_err <= 1e-6 and worst_spread <= 1e-6
    print("PASS: embedding is exact at both data rows and model boundaries." if ok
          else "FAIL: embedding disagrees with sklearn or admits ambiguous leaves.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

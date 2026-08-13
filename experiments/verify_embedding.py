"""Verify that embedded models agree with sklearn, including ON split thresholds.

The pre-existing check (`embed.verify_embedded_predictions`) evaluates at random
training rows and minimises f_var. That misses the failure mode a MIP actually
exercises: a tree split is a strict inequality on one side, and if both child
boxes are closed at the threshold then x == threshold satisfies TWO leaves and
the optimizer picks whichever is cheaper. Real data points essentially never sit
on a threshold, so the bug is invisible there and systematic in the master.

This script therefore evaluates at BOTH:
  - random training rows, and
  - points constructed to lie exactly ON a split threshold,
each time solving the embedded model twice (minimise and maximise f_var). Any
gap between those two is leaf ambiguity; any offset from sklearn is an encoding
error. A correct embedding gives min == max == sklearn.

    python experiments/verify_embedding.py                 # gastric constraint models
    python experiments/verify_embedding.py --problem synthetic
"""

import argparse
import json
import os
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.embed import embed_model, _extract_tree_structure, _parse_xgb_json_tree
from src.models.train import train_model
from experiments.run_adversary_probe import build_instance, outcome_rows, _unwrap
from experiments.run_chemo_robust import load_config


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=["gastric", "synthetic"], default="gastric")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--n-rows", type=int, default=25)
    p.add_argument("--n-thresholds", type=int, default=25)
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    p.add_argument("--gt-cv-configs",
                   default="results/cv/gastric_gt_ensemble_configs.json")
    args = p.parse_args()

    config = load_config(args.config)
    mt, mp = config["model"]["type"], config["model"]["params"]
    inst = build_instance(config, args)
    rows = outcome_rows(inst, mt, mp)
    lb, ub = inst.variable_lb.astype(float), inst.variable_ub.astype(float)
    rng = np.random.RandomState(0)

    print("=" * 78)
    print("EMBEDDING VERIFICATION  |sklearn - embedded|, and the embedded min/max "
          "spread")
    print("=" * 78)
    print(f"{'outcome':<28}{'type':<8}{'case':<12}{'max err':>12}{'max spread':>13}")

    worst_err = worst_spread = 0.0
    for (c_idx, name, md, m_type, m_params, is_con, rhs) in rows:
        # trained exactly as _train_nominal_with_configs does
        model = train_model(md.X_train, md.y_train, m_type, m_params)
        cases = {
            "data rows": [md.X_train[i] for i in
                          rng.choice(len(md.X_train), args.n_rows, replace=False)],
            "on-threshold": threshold_points(model, md.X_train, lb, ub,
                                             args.n_thresholds, rng),
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
                print(f"{name:<28}{m_type:<8}{case:<12}{'(no points)':>12}")
                continue
            e, s = max(errs), max(spreads)
            worst_err, worst_spread = max(worst_err, e), max(worst_spread, s)
            flag = "  <-- FAIL" if (e > 1e-6 or s > 1e-6) else ""
            print(f"{name:<28}{m_type:<8}{case:<12}{e:>12.2e}{s:>13.2e}{flag}")

    print("-" * 78)
    print(f"{'OVERALL':<48}{worst_err:>12.2e}{worst_spread:>13.2e}")
    ok = worst_err <= 1e-6 and worst_spread <= 1e-6
    print("PASS: embedding is exact at both data rows and split thresholds." if ok
          else "FAIL: embedding disagrees with sklearn or admits ambiguous leaves.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

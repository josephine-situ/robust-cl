"""WFP palatable diet (OptiCL's case study) as a semi-synthetic instance, and the
RC on Ovalle et al.'s tree surrogates there (paper_plan.md, E3; first version).

A probe for Paper A: it imports no runner and writes only under
results/paper_a/prelim/. Data in data/wfp/ (SOURCE.md).

INSTANCE (the released OptiCL notebook's diet-only problem):
    x_k >= 0, 25 commodities in units of 100 g per person per day, bounded above
    by the largest amount in the 5,000 public baskets; salt 0.05, sugar 0.2 fixed
    min c'x  s.t.  nutrients: sum_k nutval_kl x_k >= req_l (12 nutrients),
                   learned palatability h(x) >= T = 0.5
TRUTH: the palatability function recovered from the public labels
(data/wfp/SOURCE.md), h(x) = a - b * D(x), exact to the CSV's rounding. Since
{h >= T} is a second-order cone in the group sums, the true optimum is an SOCP
and is solved here per cost vector as the reference.

SEMI-SYNTHETIC DESIGN (our choices, 2026-11-28; to confirm):
    training baskets  a fixed random 1,000 of the 5,000 public rows (the notebook
                      trains on 1,000)
    labels            y = h(x) + sigma e, sigma = SIGMA (0.05, about a fifth of
                      the labels' sd 0.226), redrawn per run
    costs             100 vectors: Amman's $/mt times an independent U(0.5, 1.5)
                      factor per commodity (the paper's own cost draws are
                      undocumented); run k uses vector k
TRUST REGION (default on, as OptiCL): x in the convex hull of the 1,000 training
baskets. Without it the optimizer leaves the data and every tree family's
decision has true palatability near 0.05 where its clean-label model reads 0.53
(smoke test, redraw 0): misfit, which no label-noise margin can see.
Surrogates and methods as probe_rc_trees.py (families at Ovalle et al.'s reactor
settings, standard or honest splits): nominal, the RC at Theorem 1's simulated
radius and on a grid, a constant margin grid, split conformal at alpha = 0.1.
Judged against the clean-label model (same splits, leaves on h(x)) and h itself.

    python experiments/probe_wfp.py --family rf --variant standard --first 0 --n-redraws 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_rc_trees import TreeMap, build_map, check_map, fit_family, sim_radius  # noqa: E402
from src.methods.cmicl import conformal_quantile  # noqa: E402

OUT_DIR = os.path.join("results", "paper_a", "prelim")
DATA = os.path.join("data", "wfp")
T_PAL = 0.5
SIGMA = 0.05
N_TRAIN = 1000
TIME_LIMIT = 120
RC_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
MARGINS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]
CAL_FRAC = 0.2

GROUPS = {
    "cereals": (["Bulgur", "Maize", "Maize meal", "Rice", "Sorghum", "Wheat", "Wheat flour"], 400.0, 1.0),
    "pulses": (["Beans", "Lentils", "Chickpeas", "Dates"], 65.0, 5.7),
    "oil": (["Oil"], 27.0, 16.6),
    "mixed": (["CSB", "WSB", "Soya-fortified bulgur wheat", "Soya-fortified maize meal",
               "Soya-fortified sorghum grits", "Soya-fortified wheat flour"], 45.0, 4.4),
    "mfd": (["Cheese", "Fish", "Meat", "DSM", "Milk"], 30.0, 6.6),
}
H_A, H_B = 1.19314169, 0.0026836779


class WFP:
    def __init__(self):
        df = pd.read_csv(os.path.join(DATA, "WFP_dataset.csv"))
        self.foods = [c for c in df.columns if c != "label"]
        self.X_all = df[self.foods].to_numpy(float)
        self.label = df["label"].to_numpy(float)
        nv = pd.read_csv(os.path.join(DATA, "nutr_val.csv")).set_index("Food")
        assert set(nv.index) == set(self.foods), "nutrient table and dataset name different foods"
        self.nutval = nv.loc[self.foods].to_numpy(float)          # 25 x 12, per 100 g
        req = pd.read_csv(os.path.join(DATA, "nutr_req.csv"))
        self.req = req[nv.columns].to_numpy(float)[0]
        fc = pd.read_csv(os.path.join(DATA, "FoodCost.csv")).set_index("Supplier")
        self.amman = fc.loc["Amman S", self.foods].to_numpy(float)
        self.d = len(self.foods)
        self.lb = np.zeros(self.d)
        self.ub = self.X_all.max(axis=0)
        for f, v in (("Salt", 0.05), ("Sugar", 0.2)):
            j = self.foods.index(f)
            self.lb[j] = self.ub[j] = v
        self.G = np.zeros((len(GROUPS), self.d))                   # group sums, in grams
        self.opt = np.zeros(len(GROUPS))
        self.gam = np.zeros(len(GROUPS))
        for i, (foods, o, g) in enumerate(GROUPS.values()):
            for f in foods:
                self.G[i, self.foods.index(f)] = 100.0
            self.opt[i], self.gam[i] = o, g
        err = np.abs(self.truth(self.X_all) - self.label).max()
        assert err < 1e-5, f"recovered palatability misses the public labels by {err:.1e}"
        self.train_rows = np.sort(np.random.default_rng(2026).choice(len(df), N_TRAIN, replace=False))

    def truth(self, X):
        X = np.atleast_2d(X)
        D = np.linalg.norm(self.gam * (X @ self.G.T - self.opt), axis=1)
        return H_A - H_B * D

    def costs(self, n):
        return np.array([self.amman * np.random.default_rng([77, k]).uniform(0.5, 1.5, self.d)
                         for k in range(n)])

    def master(self, hull=None):
        """The diet polytope; with hull (an array of baskets), OptiCL's trust
        region: x in the convex hull of those baskets."""
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = TIME_LIMIT
        if os.environ.get("GRB_THREADS"):
            m.Params.Threads = int(os.environ["GRB_THREADS"])
        x = [m.addVar(lb=self.lb[j], ub=self.ub[j]) for j in range(self.d)]
        for l in range(len(self.req)):
            m.addConstr(gp.quicksum(float(self.nutval[j, l]) * x[j] for j in range(self.d)) >= self.req[l])
        if hull is not None:
            lam = m.addVars(len(hull), lb=0.0)
            m.addConstr(lam.sum() == 1)
            for j in range(self.d):
                if self.lb[j] < self.ub[j]:
                    m.addConstr(x[j] == gp.quicksum(float(hull[i, j]) * lam[i]
                                                    for i in range(len(hull)) if hull[i, j] != 0))
        return m, x

    def solve(self, m, x, cost):
        m.setObjective(gp.quicksum(float(cost[j]) * x[j] for j in range(self.d)), GRB.MINIMIZE)
        t0 = time.time()
        m.optimize()
        out = {"status": int(m.Status), "time_s": time.time() - t0, "x": None}
        if m.SolCount > 0:
            out["x"] = np.clip(np.array([v.X for v in x]), self.lb, self.ub)
            if m.IsMIP:
                out["mip_gap"] = float(m.MIPGap)
        return out

    def true_opt(self, cost):
        """min c'x over the nutrients and the TRUE h(x) >= T (an SOCP)."""
        m, x = self.master()
        r = (H_A - T_PAL) / H_B
        v = m.addVars(len(GROUPS), lb=-GRB.INFINITY)
        for i in range(len(GROUPS)):
            m.addConstr(v[i] == self.gam[i] * (gp.quicksum(float(self.G[i, j]) * x[j]
                                                           for j in range(self.d)) - self.opt[i]))
        m.addConstr(gp.quicksum(v[i] * v[i] for i in range(len(GROUPS))) <= r * r)
        return self.solve(m, x, cost)


def run(args):
    w = WFP()
    X = w.X_all[w.train_rows]
    ystar = w.truth(X)
    n = len(X)
    last = args.first + args.n_redraws
    costs = w.costs(last)
    rng = np.random.default_rng(5)
    Xs = np.vstack([w.X_all, w.lb + (w.ub - w.lb) * rng.random((args.n_sample, w.d))])
    tag = f"wfp_{args.family}_{args.variant}{'' if args.trust_region else '_notr'}_{args.first:03d}"
    path = os.path.join(OUT_DIR, tag + ".json")
    out = json.load(open(path)) if os.path.exists(path) else {"runs": {}}
    out.update(family=args.family, variant=args.variant, sigma=SIGMA, T=T_PAL, n_train=n,
               trust_region=bool(args.trust_region))

    for k in range(args.first, last):
        rec = out["runs"].get(str(k), {"rows": []})
        if rec.get("done"):
            continue
        done = {(r["kind"], r["param"]) for r in rec["rows"]}
        ybar = ystar + SIGMA * np.random.default_rng([args.seed, k]).standard_normal(n)
        cost = costs[k]
        t0 = time.time()
        tm = build_map(args.family, args.variant, X, ybar, args.seed, k)
        if args.variant == "standard":
            rec["map_check"] = check_map(tm, X, ybar)
        yR, ysR = ybar[tm.rows], ystar[tm.rows]
        if "R" not in rec:
            rec["R"], rec["n_cells"] = sim_radius(tm, Xs, args.seed, k, args.n_draws, args.eps)
        if "true_opt" not in rec:
            to = w.true_opt(cost)
            rec["true_opt"] = {"obj": float(cost @ to["x"]), "status": to["status"],
                               "truth": float(w.truth(to["x"])[0])}
        rec.update(empty_leaves=int(tm.empty), fit_s=time.time() - t0)
        R = rec["R"]

        def judge(xx):
            return {"x": xx.tolist(), "obj": float(cost @ xx),
                    "h": float(tm.predict(xx, yR)[0]), "clean": float(tm.predict(xx, ysR)[0]),
                    "snorm": float(np.linalg.norm(tm.S(xx)[0])), "truth": float(w.truth(xx)[0])}

        def add(kind, param, res, **extra):
            row = {"kind": kind, "param": param, "status": res["status"], "time_s": res["time_s"],
                   "obj": float("nan"), "mip_gap": res.get("mip_gap")}
            if res["x"] is not None:
                row.update(judge(res["x"]))
            row.update(extra)
            rec["rows"].append(row)
            done.add((kind, param))
            out["runs"][str(k)] = rec
            _save(tag + ".json", out)

        hull = X if args.trust_region else None

        def solve_tree(tmap, yy, shift=0.0, g=0.0, prefix="h"):
            m, x = w.master(hull)
            f, t = tmap.embed(m, x, w.lb, w.ub, yy, need_norm=g > 0, prefix=prefix)
            m.addConstr((f - g * R * SIGMA * t if g > 0 else f) >= T_PAL + shift)
            return w.solve(m, x, cost)

        if ("nominal", 0.0) not in done:
            add("nominal", 0.0, solve_tree(tm, yR))
        for g in RC_GRID:
            if ("rc", g) not in done:
                add("rc", g, solve_tree(tm, yR, g=g), R=R)
        for mg in MARGINS:
            if ("margin", mg) not in done:
                add("margin", mg, solve_tree(tm, yR, shift=mg))
        if args.variant == "standard" and ("conformal", 0.1) not in done:
            perm = np.random.default_rng([args.seed, k, 2]).permutation(n)
            n_cal = int(round(CAL_FRAC * n))
            cal, tr = perm[:n_cal], perm[n_cal:]
            hmap = TreeMap(args.family, fit_family(args.family, X[tr], ybar[tr], args.seed), X, tr)
            q = conformal_quantile(np.abs(ybar[cal] - hmap.predict(X[cal], ybar[tr])), 0.1)
            add("conformal", 0.1, solve_tree(hmap, ybar[tr], shift=q, prefix="c"), q=q)
        rec["done"] = True
        out["runs"][str(k)] = rec
        _save(tag + ".json", out)
        nom = next(r for r in rec["rows"] if r["kind"] == "nominal")
        rc1 = next(r for r in rec["rows"] if r["kind"] == "rc" and r["param"] == 1.0)
        print(f"{tag} redraw {k}: R {R:.2f} sigma ({rec['n_cells']} cells); true opt {rec['true_opt']['obj']:.1f}; "
              f"nominal obj {nom['obj']:.1f} truth {nom.get('truth', np.nan):.3f}; RC obj {rc1['obj']:.1f} "
              f"clean {rc1.get('clean', np.nan):.3f} truth {rc1.get('truth', np.nan):.3f} "
              f"({rc1['time_s']:.0f}s); {(time.time() - t0) / 60:.1f} min", flush=True)


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["rf", "gbm", "lineardt"], required=True)
    ap.add_argument("--variant", choices=["standard", "honest"], default="standard")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--n-redraws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-sample", type=int, default=15000)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--no-trust-region", dest="trust_region", action="store_false",
                    help="drop OptiCL's convex-hull trust region (default: on)")
    run(ap.parse_args())

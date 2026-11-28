"""WFP palatable diet as a semi-synthetic REGRESSION instance (paper_plan.md, E3),
on Ovalle et al.'s design, with the RC on tree surrogates and FO-CP on the network.

A probe for Paper A: it imports no runner and writes only under
results/paper_a/prelim/. Data in data/wfp/ (SOURCE.md; gitignored, copied to the
cluster by hand).

INSTANCE (the diet problem of OptiCL's notebook and of Ovalle et al.'s
classification.py):
    x_k, 25 commodities in units of 100 g per person per day, within the
    training data's bounds; salt 0.05 and sugar 0.2 fixed
    min c'x  s.t.  sum_k nutval_kl x_k >= req_l (12 nutrients),
                   learned palatability >= T = 0.5
TRUTH: the palatability function recovered exactly from the public labels
(SOURCE.md). The true optimum is an SOCP, solved per cost vector as the reference.

DESIGN (Ovalle et al.'s classification.py, 2026-11-28), except where noted:
    split      the 5,000 rows shuffled (their shuffle is unseeded; ours seed 0),
               half set aside ("unseen", unused), the other 2,500 ("rest") split
               92/8 into 2,300 train and 200 calibration (train_test_split,
               random_state 42, as theirs)
    models     nominal, RC and FO-CP models on the 2,500 rest rows (their MICL's
               rows); C-MICL and split conformal: h on the 2,300, calibrated on
               the 200 (their C-MICL's rows)
    bounds     min and max of the rest rows per commodity (their ensembles'; their
               C-MICL uses the 2,300's, nearly identical); no trust region
    costs      their 100 vectors exactly: np.random.seed(0), then per run
               uniform(0.1, 4) per commodity, rounded to 0.01
    OURS       regression on the exact h (theirs: 4-class classification judged by
               a fitted MLP oracle), and labels h(x) + N(0, SIGMA^2) redrawn per
               run (theirs: noise-free)

Stages:
    --stage trees  Ovalle et al.'s RF / GBM / LinearDT as in probe_rc_trees.py
                   (linear maps of the labels for fixed splits): nominal, the RC at
                   Theorem 1's simulated radius and on a grid, margins, C-MICL with
                   the family's h, split conformal at alpha 0.1
    --stage cv     the network's L2 penalty by 5-fold CV on one noisy draw, one
                   alpha per call (as probe_focp's cv_alpha)
    --stage focp   the (32, 32) network trained end to end: radius (c), the FO-CP
                   loop (multistart separation, tau = 0.05 sigma, cap 20), a light a
                   posteriori attack, and on the same theta0: nominal, margins,
                   C-MICL with the network as h, split conformal at alpha 0.1

    python experiments/probe_wfp.py --stage trees --family rf --first 0 --n-redraws 10
    python experiments/probe_wfp.py --stage cv --alpha 1
    python experiments/probe_wfp.py --stage focp --redraw 0
"""
import argparse
import glob
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
TIME_LIMIT = 120
RC_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
MARGINS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]
CONF_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
CV_ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]

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
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(os.path.join(DATA, "WFP_dataset.csv"))
        self.foods = [c for c in df.columns if c != "label"]
        self.X_all = df[self.foods].to_numpy(float)
        self.label = df["label"].to_numpy(float)
        nv = pd.read_csv(os.path.join(DATA, "nutr_val.csv")).set_index("Food")
        assert set(nv.index) == set(self.foods), "nutrient table and dataset name different foods"
        self.nutval = nv.loc[self.foods].to_numpy(float)          # 25 x 12, per 100 g
        self.req = pd.read_csv(os.path.join(DATA, "nutr_req.csv"))[nv.columns].to_numpy(float)[0]
        fc = pd.read_csv(os.path.join(DATA, "FoodCost.csv")).set_index("Supplier")
        self.cost_order = list(fc.columns)
        self.d = len(self.foods)
        self.G = np.zeros((len(GROUPS), self.d))                   # group sums, in grams
        self.opt, self.gam = np.zeros(len(GROUPS)), np.zeros(len(GROUPS))
        for i, (foods, o, g) in enumerate(GROUPS.values()):
            for f in foods:
                self.G[i, self.foods.index(f)] = 100.0
            self.opt[i], self.gam[i] = o, g
        err = np.abs(self.truth(self.X_all) - self.label).max()
        assert err < 1e-5, f"recovered palatability misses the public labels by {err:.1e}"
        # Ovalle et al.'s split, on a seeded shuffle
        order = np.random.default_rng(0).permutation(len(df))
        _, self.rest = train_test_split(order, test_size=0.5, random_state=42)
        self.train, self.cal = train_test_split(self.rest, test_size=0.08, random_state=42)
        Xr = self.X_all[self.rest]
        self.lb, self.ub = Xr.min(axis=0), Xr.max(axis=0)
        self.fixed = self.lb == self.ub                            # salt, sugar
        self.ub_s = np.where(self.fixed, self.lb + 1.0, self.ub)   # scaling bounds for the network

    def truth(self, X):
        X = np.atleast_2d(X)
        D = np.linalg.norm(self.gam * (X @ self.G.T - self.opt), axis=1)
        return H_A - H_B * D

    def costs(self, n):
        rs = np.random.RandomState(0)                              # their np.random.seed(0)
        out = []
        for _ in range(n):
            v = dict(zip(self.cost_order, rs.uniform(0.1, 4, size=len(self.cost_order)).round(2)))
            out.append(np.array([v[f] for f in self.foods]))
        return np.array(out)

    def domain_sample(self, n, seed=0):
        """Uniform points in the bounds that meet the nutrient requirements."""
        rng = np.random.default_rng(seed)
        out, have = [], 0
        while have < n:
            Z = self.lb + (self.ub - self.lb) * rng.random((4 * n, self.d))
            Z = Z[(Z @ self.nutval >= self.req).all(axis=1)]
            out.append(Z)
            have += len(Z)
        return np.vstack(out)[:n]

    def master(self, hull=None):
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
                if not self.fixed[j]:
                    m.addConstr(x[j] == gp.quicksum(float(hull[i, j]) * lam[i]
                                                    for i in range(len(hull)) if hull[i, j] != 0))
        return m, x

    def u_of(self, x):
        """The network's scaled inputs as expressions in x (in [-1, 1] on the bounds)."""
        return [2.0 * (x[j] - self.lb[j]) / (self.ub_s[j] - self.lb[j]) - 1.0 for j in range(self.d)]

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
        m, x = self.master()
        r = (H_A - T_PAL) / H_B
        v = m.addVars(len(GROUPS), lb=-GRB.INFINITY)
        for i in range(len(GROUPS)):
            m.addConstr(v[i] == self.gam[i] * (gp.quicksum(float(self.G[i, j]) * x[j]
                                                           for j in range(self.d)) - self.opt[i]))
        m.addConstr(gp.quicksum(v[i] * v[i] for i in range(len(GROUPS))) <= r * r)
        res = self.solve(m, x, cost)
        return {"obj": float(cost @ res["x"]) if res["x"] is not None else None,
                "status": res["status"],
                "truth": float(self.truth(res["x"])[0]) if res["x"] is not None else None}


def labels(w, rows, seed, k):
    """Noisy labels for redraw k on the rest rows (one stream over all 5,000)."""
    ystar = w.truth(w.X_all)
    yb = ystar + SIGMA * np.random.default_rng([seed, k]).standard_normal(len(ystar))
    return yb[rows], ystar[rows]


class Recorder:
    def __init__(self, tag, header):
        self.tag, self.path = tag, os.path.join(OUT_DIR, tag + ".json")
        self.out = json.load(open(self.path)) if os.path.exists(self.path) else {"runs": {}}
        self.out.update(header)

    def run(self, k):
        return self.out["runs"].setdefault(str(k), {"rows": []})

    def save(self):
        _save(self.tag + ".json", self.out)


def rows_adder(rec, store, judge, k):
    done = {(r["kind"], r["param"]) for r in rec["rows"]}

    def add(kind, param, res, **extra):
        row = {"kind": kind, "param": param, "status": res["status"], "time_s": res["time_s"],
               "obj": float("nan"), "mip_gap": res.get("mip_gap")}
        if res["x"] is not None:
            row.update(judge(res["x"]))
        row.update(extra)
        rec["rows"].append(row)
        done.add((kind, param))
        store.save()
    return done, add


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------
def stage_trees(args):
    from probe_focp import width_net
    w = WFP()
    X = w.X_all[w.rest]
    n = len(X)
    last = args.first + args.n_redraws
    costs = w.costs(last)
    Xs = np.vstack([X, w.domain_sample(args.n_sample)])
    hull = X if args.trust_region else None
    st = Recorder(f"wfp_{args.family}_{args.variant}{'_tr' if args.trust_region else ''}_{args.first:03d}",
                  dict(family=args.family, variant=args.variant, sigma=SIGMA, T=T_PAL, n_rows=n,
                       design="ovalle", trust_region=bool(args.trust_region)))
    pos = {r: i for i, r in enumerate(w.rest)}
    tr_i, cal_i = np.array([pos[r] for r in w.train]), np.array([pos[r] for r in w.cal])
    for k in range(args.first, last):
        rec = st.run(k)
        if rec.get("done"):
            continue
        ybar, ystar = labels(w, w.rest, args.seed, k)
        cost = costs[k]
        t0 = time.time()
        tm = build_map(args.family, args.variant, X, ybar, args.seed, k)
        if args.variant == "standard":
            rec["map_check"] = check_map(tm, X, ybar)
        yR, ysR = ybar[tm.rows], ystar[tm.rows]
        if "R" not in rec:
            rec["R"], rec["n_cells"] = sim_radius(tm, Xs, args.seed, k, args.n_draws, args.eps)
        if "true_opt" not in rec:
            rec["true_opt"] = w.true_opt(cost)
        R = rec["R"]

        def judge(xx):
            return {"x": xx.tolist(), "obj": float(cost @ xx), "h": float(tm.predict(xx, yR)[0]),
                    "clean": float(tm.predict(xx, ysR)[0]), "truth": float(w.truth(xx)[0])}
        done, add = rows_adder(rec, st, judge, k)

        def solve_tree(tmap, yy, shift=0.0, g=0.0, prefix="h", extra=None):
            m, x = w.master(hull)
            f, t = tmap.embed(m, x, w.lb, w.ub, yy, need_norm=g > 0, prefix=prefix)
            lhs = f - g * R * SIGMA * t if g > 0 else f
            if extra is not None:
                lhs = extra(m, x, lhs)
            m.addConstr(lhs >= T_PAL + shift)
            return w.solve(m, x, cost)

        if ("nominal", 0.0) not in done:
            add("nominal", 0.0, solve_tree(tm, yR))
        for g in RC_GRID:
            if ("rc", g) not in done:
                add("rc", g, solve_tree(tm, yR, g=g), R=R)
        for mg in MARGINS:
            if ("margin", mg) not in done:
                add("margin", mg, solve_tree(tm, yR, shift=mg))
        if args.variant == "standard":
            Xt, yt = X[tr_i], ybar[tr_i]
            hmap = TreeMap(args.family, fit_family(args.family, Xt, yt, args.seed), X[tr_i], np.arange(len(tr_i)))
            r_cal = np.abs(ybar[cal_i] - hmap.predict(X[cal_i], yt))
            resid = np.abs(yt - hmap.predict(Xt, yt))
            um, th_u = width_net(Xt, resid, w.lb, w.ub_s, seed=k)
            u_cal = um.predict(X[cal_i], th_u)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(np.isnan(r_cal / u_cal), np.inf, r_cal / u_cal)
            for a in CONF_ALPHAS:
                if ("cmicl", a) in done:
                    continue
                q = conformal_quantile(scores, a)

                def with_u(m, x, lhs, q=q):
                    ue = um.gurobi_out(m, w.u_of(x), th_u)
                    m.addConstr(ue >= 0.0)
                    return lhs - q * ue
                res = (solve_tree(hmap, yt, prefix="c", extra=with_u) if np.isfinite(q)
                       else {"status": -1, "time_s": 0.0, "x": None})
                ux = float(um.predict(res["x"][None], th_u)[0]) if res["x"] is not None else None
                add("cmicl", a, res, q=q, u_at_x=ux)
            if ("conformal", 0.1) not in done:
                q = conformal_quantile(r_cal, 0.1)
                add("conformal", 0.1, solve_tree(hmap, yt, shift=q, prefix="c"), q=q)
        rec.update(done=True, fit_s=time.time() - t0)
        st.save()
        _report(st.tag, k, rec)


# --------------------------------------------------------------------------
# The network: CV, FO-CP and its baselines
# --------------------------------------------------------------------------
def stage_cv(args):
    from probe_focp import MLPModel
    w = WFP()
    X = w.X_all[w.rest]
    y, _ = labels(w, w.rest, 1, 9999)                 # one fixed draw for the choice
    folds = np.array_split(np.random.default_rng(2026).permutation(len(X)), 5)
    mse = []
    for f, te in enumerate(folds):
        trn = np.setdiff1d(np.arange(len(X)), te)
        m = MLPModel(X[trn], w.lb, w.ub_s, y[trn], alpha=args.alpha)
        th, info = m.cold_fit(y[trn], seed=f)
        mse.append(float(np.mean((m.predict(X[te], th) - y[te]) ** 2)))
        print(f"alpha {args.alpha:g} fold {f}: MSE {mse[-1]:.5f} ({info['time']:.0f}s)", flush=True)
    _save(f"wfp_cv_a{args.alpha:g}.json", {"alpha": args.alpha, "cv_mse": float(np.mean(mse)),
                                            "fold_mse": mse, "sigma2": SIGMA ** 2})


def chosen_alpha_wfp():
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(OUT_DIR, "wfp_cv_a*.json"))]
    if len(rows) < len(CV_ALPHAS):
        raise SystemExit(f"run --stage cv for every alpha in {CV_ALPHAS} first ({len(rows)} done)")
    return min(rows, key=lambda r: r["cv_mse"])["alpha"]


def stage_focp(args):
    import torch
    from probe_focp import MLPModel, Sensitivity, ascend, width_net
    torch.set_num_threads(args.threads)
    w = WFP()
    alpha = chosen_alpha_wfp()
    X = w.X_all[w.rest]
    n = len(X)
    k = args.redraw
    ybar, ystar = labels(w, w.rest, args.seed, k)
    cost = w.costs(k + 1)[k]
    model = MLPModel(X, w.lb, w.ub_s, ybar, alpha=alpha)
    st = Recorder(f"wfp_focp_r{k}_a{alpha:g}", dict(redraw=k, alpha=alpha, sigma=SIGMA, T=T_PAL,
                                                     n_rows=n, design="ovalle"))
    rec = st.run(k)
    # fits, stored once (the cold fit is not reproducible across nodes)
    fpath = os.path.join(OUT_DIR, f"wfp_fits_r{k}_a{alpha:g}.npz")
    if os.path.exists(fpath):
        z = np.load(fpath)
        theta0, th_clean = z["theta0"], z["th_clean"]
    else:
        theta0, info0 = model.cold_fit(ybar, seed=k)
        th_clean, _ = model.train(ystar, theta0)
        np.savez(fpath, theta0=theta0, th_clean=th_clean)
        rec["theta0_fit"] = {k_: info0[k_] for k_ in ("nit", "ok", "time")}
    if "true_opt" not in rec:
        rec["true_opt"] = w.true_opt(cost)

    # radius (c): W = max over the domain of [h(x; T(ybar + e')) - h(x; theta0)] / ||s(x)||
    Xs = np.vstack([X, w.domain_sample(args.n_sample)])
    rad = rec.setdefault("radius", {"W": []})
    if "R" not in rad:
        sens = Sensitivity(model, theta0, ybar)
        snorm = np.concatenate([SIGMA * np.linalg.norm(sens.dh_dy(Xs[i:i + 2000], args.lam_rel), axis=0)
                                for i in range(0, len(Xs), 2000)])
        h0 = model.predict(Xs, theta0)
        t0 = time.time()
        for i in range(len(rad["W"]), args.n_draws):
            e = SIGMA * np.random.default_rng([args.seed, k, 11, i]).standard_normal(n)
            th, _ = model.train(ybar + e, theta0)
            rad["W"].append(float(((model.predict(Xs, th) - h0) / snorm).max()))
            if (i + 1) % 20 == 0:
                st.save()
                print(f"radius: {i + 1}/{args.n_draws} draws, {(time.time() - t0) / 60:.1f} min", flush=True)
        kk = int(np.ceil((args.n_draws + 1) * (1 - args.eps)))
        rad["R"] = float(np.sort(rad["W"])[min(kk, args.n_draws) - 1])
        rad["snorm_median"] = float(np.median(snorm))
        st.save()
    R = rad["R"]
    tau = 0.05 * SIGMA

    def judge(xx):
        return {"x": xx.tolist(), "obj": float(cost @ xx), "h": float(model.predict(xx[None], theta0)[0]),
                "clean": float(model.predict(xx[None], th_clean)[0]), "truth": float(w.truth(xx)[0])}
    done, add = rows_adder(rec, st, judge, k)

    def net_master(shift=0.0, cuts=()):
        m, x = w.master()
        u = w.u_of(x)
        m.addConstr(model.gurobi_out(m, u, theta0) >= T_PAL + shift)
        for th in cuts:
            m.addConstr(model.gurobi_out(m, u, th) >= T_PAL + tau)
        return w.solve(m, x, cost)

    if ("nominal", 0.0) not in done:
        add("nominal", 0.0, net_master())
    for mg in MARGINS:
        if ("margin", mg) not in done:
            add("margin", mg, net_master(shift=mg))

    # FO-CP loop (multistart separation), checkpointed per iteration
    if ("focp", 1.0) not in done:
        ck = os.path.join(OUT_DIR, f"wfp_focp_r{k}_a{alpha:g}_cuts.npz")
        loop = rec.setdefault("loop", {"iters": []})
        cuts, zs = [], []
        if os.path.exists(ck) and loop["iters"]:
            z_ = np.load(ck)
            cuts, zs = list(z_["cuts"]), list(z_["zs"])
        rng = np.random.default_rng([args.seed, k, 13])
        t_loop, conv, res = time.time(), False, None
        while len(loop["iters"]) < args.max_iter:
            res = net_master(cuts=cuts)
            if res["x"] is None:
                break
            x = res["x"]
            starts = [("zero", np.zeros(n))]
            if cuts:
                hs = [float(model.predict(x[None], th)[0]) for th in cuts]
                starts.append(("warm", zs[int(np.argmin(hs))]))
            for j in range(4):
                g = rng.standard_normal(n)
                starts.append((f"random{j}", R * g / np.linalg.norm(g)))
            t1 = time.time()
            best = ascend(model, x, ybar, SIGMA, R, theta0, starts, args.lam_rel, max_steps=20)
            worst = float(model.predict(x[None], best["theta"])[0])
            loop["iters"].append({"obj": float(cost @ x), "worst": worst, "master_s": res["time_s"],
                                  "sep_s": time.time() - t1, "status": res["status"], "n_cuts": len(cuts)})
            print(f"focp r{k} iter {len(loop['iters'])}: obj {cost @ x:.2f} h "
                  f"{model.predict(x[None], theta0)[0]:.4f} worst {worst:.4f} "
                  f"(sep {time.time() - t1:.0f}s)", flush=True)
            if worst >= T_PAL:
                conv = True
                break
            cuts.append(best["theta"])
            zs.append(best["z"])
            np.savez(ck, cuts=np.array(cuts), zs=np.array(zs))
            st.save()
        if res is not None and res["x"] is not None:
            x = res["x"]
            starts = [("zero", np.zeros(n))] + [
                (f"random{j}", R * g / np.linalg.norm(g))
                for j, g in enumerate(rng.standard_normal((args.attack_starts, n)))]
            att = ascend(model, x, ybar, SIGMA, R, theta0, starts, args.lam_rel,
                         max_steps=args.attack_steps, halvings=8)
            a_worst = float(model.predict(x[None], att["theta"])[0])
            res = dict(res, time_s=time.time() - t_loop)
            add("focp", 1.0, res, R=R, iters=len(loop["iters"]), converged=conv,
                attack_worst=a_worst, attack_gap=(T_PAL - a_worst) / SIGMA)
        else:
            add("focp", 1.0, {"status": res["status"] if res else -1, "time_s": time.time() - t_loop,
                              "x": None}, R=R, converged=False)

    # C-MICL with the network as h (Ovalle et al.'s 2,300 / 200), split conformal
    pos = {r: i for i, r in enumerate(w.rest)}
    tr_i, cal_i = np.array([pos[r] for r in w.train]), np.array([pos[r] for r in w.cal])
    if not {("cmicl", a) for a in CONF_ALPHAS} | {("conformal", 0.1)} <= done:
        hm = MLPModel(X[tr_i], w.lb, w.ub_s, ybar[tr_i], alpha=alpha)
        hpath = os.path.join(OUT_DIR, f"wfp_fits_h80_r{k}_a{alpha:g}.npz")
        if os.path.exists(hpath):
            th_h = np.load(hpath)["th_h"]
        else:
            th_h, _ = hm.cold_fit(ybar[tr_i], seed=k)
            np.savez(hpath, th_h=th_h)
        resid = np.abs(ybar[tr_i] - hm.predict(X[tr_i], th_h))
        um, th_u = width_net(X[tr_i], resid, w.lb, w.ub_s, seed=k)
        u_cal = um.predict(X[cal_i], th_u)
        r_cal = np.abs(ybar[cal_i] - hm.predict(X[cal_i], th_h))
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(np.isnan(r_cal / u_cal), np.inf, r_cal / u_cal)
        for a in CONF_ALPHAS:
            if ("cmicl", a) in done:
                continue
            q = conformal_quantile(scores, a)
            if np.isfinite(q):
                m, x = w.master()
                u = w.u_of(x)
                he, ue = hm.gurobi_out(m, u, th_h), um.gurobi_out(m, u, th_u)
                m.addConstr(he - q * ue >= T_PAL)
                m.addConstr(ue >= 0.0)
                res = w.solve(m, x, cost)
            else:
                res = {"status": -1, "time_s": 0.0, "x": None}
            ux = float(um.predict(res["x"][None], th_u)[0]) if res["x"] is not None else None
            add("cmicl", a, res, q=q, u_at_x=ux)
        if ("conformal", 0.1) not in done:
            q = conformal_quantile(r_cal, 0.1)
            m, x = w.master()
            m.addConstr(hm.gurobi_out(m, w.u_of(x), th_h) >= T_PAL + q)
            add("conformal", 0.1, w.solve(m, x, cost), q=q)
    rec["done"] = True
    st.save()
    _report(st.tag, k, rec)


def _report(tag, k, rec):
    rows = {(r["kind"], r["param"]): r for r in rec["rows"]}
    nom = rows.get(("nominal", 0.0), {})
    main = rows.get(("focp", 1.0)) or rows.get(("rc", 1.0)) or {}
    to = rec.get("true_opt", {})
    print(f"{tag} redraw {k}: true opt {to.get('obj', float('nan')):.2f}; nominal obj "
          f"{nom.get('obj', float('nan')):.2f} truth {nom.get('truth', float('nan')):.3f}; "
          f"{main.get('kind', '?')} obj {main.get('obj', float('nan')):.2f} clean "
          f"{main.get('clean', float('nan')):.3f} truth {main.get('truth', float('nan')):.3f}", flush=True)


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["trees", "cv", "focp"], required=True)
    ap.add_argument("--family", choices=["rf", "gbm", "lineardt"], default="rf")
    ap.add_argument("--variant", choices=["standard", "honest"], default="standard")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--n-redraws", type=int, default=10)
    ap.add_argument("--redraw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-sample", type=int, default=15000)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0, help="cv: the L2 penalty to score")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lam-rel", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--attack-starts", type=int, default=5)
    ap.add_argument("--attack-steps", type=int, default=20)
    ap.add_argument("--trust-region", action="store_true", help="trees: OptiCL's convex hull")
    args = ap.parse_args()
    {"trees": stage_trees, "cv": stage_cv, "focp": stage_focp}[args.stage](args)

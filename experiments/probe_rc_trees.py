"""The RC on Ovalle et al.'s tree surrogates (SS1, both arms), beside its baselines.

A probe for Paper A: it imports no runner and writes only under
results/paper_a/prelim/. Same SS1 redraws, noise stream, cost vectors and judges
as probe_focp.py's F2 (redraw k uses Ovalle et al.'s cost vector k).

Families, at Ovalle et al.'s grid points (src/methods/cmicl.py, DIFFERENCES #2):
    lineardt  linear-tree's LinearTreeRegressor(LinearRegression), depth 5,
              min_samples_split 10, max_bins 40 (other settings: the package's)
    rf        RandomForestRegressor, 15 trees, depth 5, min_samples_split 3,
              max_features 0.6
    gbm       GradientBoostingRegressor, 15 trees, learning rate 0.2, depth 5,
              min_samples_split 5, max_features 0.6
Trees are fit on the physical inputs (they are invariant to Ovalle et al.'s /100
scaling up to tie-breaking).

WHY THE RC APPLIES. With its splits fixed, each family predicts LINEARLY in the
labels it is fitted to, f(x) = s(x)'y:
    rf        leaf value = (bootstrap-weighted) mean of the leaf's labels; the
              forest averages its trees
    gbm       init = mean label; stage t's leaf value = mean residual in the leaf,
              and the residuals y - F_{t-1} are linear in y (squared loss)
    lineardt  leaf prediction = [1, x]'beta_l, beta_l the leaf's least squares
The maps are rebuilt here and checked against the library's own predictions.

Split variants:
    standard  splits and leaves fitted to the same noisy labels (as practitioners
              fit them); Theorem 1 ignores the splits' dependence on the noise,
              as for the network's standard fit (R3)
    honest    splits fitted on a random half A, leaves refitted on the other
              half B (Theorem 1 holds exactly given the splits)

Per (family, variant, arm, redraw):
    nominal      f(x) >= 50
    rc g         f(x) - g R sigma ||s(x)|| >= 50, R = Theorem 1's simulated radius
                 (enumeration, radius_c's noise stream, N = 200, eps = 0.1;
                 understated where small cells are missed), g in RC_GRID
    margin m     f(x) >= 50 + m over probe_rc_ridge's frontier grid
    cmicl a      (standard only) Ovalle et al.'s C-MICL with this family's h on
                 a random 80% and the config's width net u: h - q u >= 50, u >= 0
    conformal    (standard only) split conformal at alpha = 0.1 (as prescribed)
Judged against the clean-label model (the same splits, leaves refitted on the
noise-free labels y*) and the arm's truth.

    python experiments/probe_rc_trees.py --family rf --variant standard --arm ode --first 0 --n-redraws 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_rc_ridge import (CAL_FRAC, CONF_ALPHAS, FRONTIER_MARGIN, OUT_DIR,  # noqa: E402
                            cost_vectors, domain_sample, setup)
from probe_focp import (MLPModel, _master, _solve, arm_truth, width_net)  # noqa: E402
from src.data.dma_mr import PRODUCT_FLOOR  # noqa: E402
from src.methods.cmicl import conformal_quantile  # noqa: E402
from src.models.embed import _embed_leaves, _extract_tree_structure  # noqa: E402

RC_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
LDT_EPS = 1e-6        # strict side of a LinearDT split (x > threshold), relative


# --------------------------------------------------------------------------
# Fitting, with the label map rebuilt for fixed splits
# --------------------------------------------------------------------------
def _lineartree():
    """linear-tree 0.3.5 calls BaseEstimator._validate_data (removed in sklearn
    1.6) with force_all_finite (renamed ensure_all_finite): route both."""
    import lineartree
    from sklearn.utils.validation import validate_data

    def _vd(self, *a, **k):
        if "force_all_finite" in k:
            k["ensure_all_finite"] = k.pop("force_all_finite")
        return validate_data(self, *a, **k)
    lineartree.LinearTreeRegressor._validate_data = _vd
    return lineartree.LinearTreeRegressor


def fit_family(family, X, y, seed):
    if family == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=15, max_depth=5, min_samples_split=3,
                                  max_features=0.6, random_state=seed)
    elif family == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(n_estimators=15, learning_rate=0.2, max_depth=5,
                                      min_samples_split=5, max_features=0.6, random_state=seed)
    else:
        from sklearn.linear_model import LinearRegression
        m = _lineartree()(base_estimator=LinearRegression(), max_depth=5,
                          min_samples_split=10, max_bins=40)
    return m.fit(X, y)


class TreeMap:
    """A fitted tree surrogate with its splits frozen, as a map from the labels
    y_R of the leaf rows R to predictions: f(x) = s(x)' y_R.

    Constant-leaf ensembles (rf, gbm): s(x) = c0 + sum_t A_t[leaf_t(x)].
    lineardt: s(x) = G_l' [1, x] in leaf l."""

    def __init__(self, family, model, X, rows, Xsplit_rows=None):
        self.family, self.model = family, model
        self.rows = np.asarray(rows)
        self.nR = len(self.rows)
        XR = X[self.rows]
        if family == "lineardt":
            self.leaves = []
            for key, leaf in model._leaves.items():
                mask = self._in_leaf(XR, leaf.threshold)
                Z = np.c_[np.ones(mask.sum()), XR[mask]]
                G = np.zeros((Z.shape[1], self.nR))
                if mask.sum() > 0:
                    # least squares on centred, scaled columns (raw T ~ 1000 is
                    # ill-conditioned), mapped back to [1, x]
                    mu, sd = Z[:, 1:].mean(axis=0), Z[:, 1:].std(axis=0)
                    sd = np.where(sd < 1e-9, 1.0, sd)            # constant columns (WFP's salt, sugar)
                    Zs = np.c_[np.ones(len(Z)), (Z[:, 1:] - mu) / sd]
                    Gs = np.linalg.pinv(Zs)                      # coefficients on Zs
                    G[1:, mask] = Gs[1:] / sd[:, None]
                    G[0, mask] = Gs[0] - (mu / sd) @ Gs[1:]
                self.leaves.append({"thr": leaf.threshold, "G": G, "n": int(mask.sum())})
            self.empty = sum(lf["n"] == 0 for lf in self.leaves)
            return
        trees = model.estimators_ if family == "rf" else [e[0] for e in model.estimators_]
        self.trees = trees
        T = len(trees)
        self.c0 = np.zeros(self.nR)
        self.A = []          # per tree: {node id: a (nR,)}
        self.empty = 0
        if family == "rf":
            for tr in trees:
                leaf_of = tr.apply(XR.astype(np.float32))
                if Xsplit_rows is None:
                    from sklearn.ensemble._forest import (_generate_sample_indices,
                                                          _get_n_samples_bootstrap)
                    nb = _get_n_samples_bootstrap(self.nR, model.max_samples)
                    w = np.bincount(_generate_sample_indices(tr.random_state, self.nR, nb),
                                    minlength=self.nR).astype(float)
                else:
                    w = np.ones(self.nR)                  # honest leaves: rows of B, unweighted
                amap = {}
                for node in np.flatnonzero(tr.tree_.children_left == -1):
                    ww = w * (leaf_of == node)
                    if ww.sum() > 0:
                        amap[node] = ww / ww.sum() / T
                    else:
                        amap[node] = np.full(self.nR, 1.0 / self.nR / T)
                        self.empty += 1
                self.A.append(amap)
        else:   # gbm, squared loss, subsample 1: leaf value = mean residual
            lr = model.learning_rate
            self.c0 = np.full(self.nR, 1.0 / self.nR)
            P = np.tile(self.c0, (self.nR, 1))                # in-sample prediction map
            for tr in trees:
                leaf_of = tr.apply(XR.astype(np.float32))
                Rres = np.eye(self.nR) - P                     # residual map
                amap = {}
                for node in np.flatnonzero(tr.tree_.children_left == -1):
                    msk = leaf_of == node
                    if msk.any():
                        v = Rres[msk].mean(axis=0)
                    else:
                        v = np.zeros(self.nR)
                        self.empty += 1
                    amap[node] = lr * v
                    P[msk] += lr * v
                self.A.append(amap)

    @staticmethod
    def _in_leaf(X, thr):
        mask = np.ones(len(X), bool)
        for f, side, t in thr:
            mask &= (X[:, f] <= t) if side == "L" else (X[:, f] > t)
        return mask

    def S(self, Xq):
        """s(x) for each row of Xq: m x nR."""
        Xq = np.atleast_2d(Xq)
        if self.family == "lineardt":
            out = np.zeros((len(Xq), self.nR))
            Z = np.c_[np.ones(len(Xq)), Xq]
            for lf in self.leaves:
                msk = self._in_leaf(Xq, lf["thr"])
                if msk.any():
                    out[msk] = Z[msk] @ lf["G"]
            return out
        out = np.tile(self.c0, (len(Xq), 1))
        for tr, amap in zip(self.trees, self.A):
            leaf_of = tr.apply(Xq.astype(np.float32))
            for node in np.unique(leaf_of):
                out[leaf_of == node] += amap[node]
        return out

    def predict(self, Xq, yR):
        return self.S(Xq) @ yR

    # ---- MIP --------------------------------------------------------------
    def embed(self, m, x, lb, ub, yR, need_norm, prefix="h"):
        """Add the surrogate to model m over the expressions x. Returns (f, t):
        f an expression with f <= f(x) (exact for constant leaves), t a variable
        with t >= ||s(x)|| (None unless need_norm). Fine for constraints of the
        form f - c t >= rhs."""
        if self.family == "lineardt":
            return self._embed_ldt(m, x, lb, ub, yR, need_norm, prefix)
        cols, coef = [self.c0], [float(self.c0 @ yR)]
        zs = []
        for i, (tr, amap) in enumerate(zip(self.trees, self.A)):
            leaves = _extract_tree_structure(tr)
            _embed_leaves(m, leaves, x, lb, ub, name_prefix=f"{prefix}{i}")
            m.update()
            for l, leaf in enumerate(leaves):
                z = m.getVarByName(f"{prefix}{i}_z{l}")
                if z is None:
                    continue
                a = amap[leaf["id"]]
                zs.append(z)
                cols.append(a)
                coef.append(float(a @ yR))
        f = coef[0] + gp.quicksum(c * z for c, z in zip(coef[1:], zs))
        t = None
        if need_norm:
            M = np.column_stack(cols)                        # nR x (1 + L)
            Rq = np.linalg.qr(M, mode="r")
            Rq = Rq[np.abs(Rq).sum(axis=1) > 1e-12]
            v = m.addVars(Rq.shape[0], lb=-GRB.INFINITY)
            for r in range(Rq.shape[0]):
                m.addConstr(v[r] == Rq[r, 0] + gp.quicksum(float(Rq[r, j + 1]) * zs[j]
                                                           for j in range(len(zs)) if Rq[r, j + 1] != 0))
            t = m.addVar(lb=0.0)
            m.addConstr(gp.quicksum(v[r] * v[r] for r in range(Rq.shape[0])) <= t * t)
        return f, t

    def _embed_ldt(self, m, x, lb, ub, yR, need_norm, prefix):
        d = len(lb)
        span = ub - lb
        z = m.addVars(len(self.leaves), vtype=GRB.BINARY)
        m.addConstr(z.sum() == 1)
        m.Params.IntFeasTol = 1e-9
        preds, lo, hi = [], [], []
        for l, lf in enumerate(self.leaves):
            for f_, side, thr in lf["thr"]:
                if side == "L":   # x <= thr when z = 1
                    m.addConstr(x[f_] <= thr + (ub[f_] - thr) * (1 - z[l]))
                else:             # x >= thr + eps when z = 1
                    e = thr + LDT_EPS * span[f_]
                    m.addConstr(x[f_] >= e - (e - lb[f_]) * (1 - z[l]))
            beta = lf["G"] @ yR                              # [intercept, coef]
            preds.append(beta)
            lo.append(beta[0] + np.minimum(beta[1:] * lb, beta[1:] * ub).sum())
            hi.append(beta[0] + np.maximum(beta[1:] * lb, beta[1:] * ub).sum())
        fv = m.addVar(lb=-GRB.INFINITY, ub=max(hi))
        for l, beta in enumerate(preds):
            pl = beta[0] + gp.quicksum(float(beta[j + 1]) * x[j] for j in range(d))
            m.addConstr(fv <= pl + (max(hi) - lo[l]) * (1 - z[l]))
        t = None
        if need_norm:
            t = m.addVar(lb=0.0)
            xmax = np.maximum(np.abs(lb), np.abs(ub))
            for l, lf in enumerate(self.leaves):
                K = np.linalg.cholesky(lf["G"] @ lf["G"].T + 1e-12 * np.eye(d + 1)).T  # K'K = G G'
                N = np.linalg.norm(K[:, 0]) + sum(xmax[j] * np.linalg.norm(K[:, j + 1]) for j in range(d))
                v = m.addVars(d + 1, lb=-GRB.INFINITY)
                for r in range(d + 1):
                    m.addConstr(v[r] == float(K[r, 0]) + gp.quicksum(float(K[r, j + 1]) * x[j] for j in range(d)))
                u_ = m.addVar(lb=0.0)
                m.addConstr(u_ == t + N * (1 - z[l]))
                m.addConstr(gp.quicksum(v[r] * v[r] for r in range(d + 1)) <= u_ * u_)
        return fv, t


def build_map(family, variant, X, ybar, seed, k):
    """(TreeMap, the rows whose labels it uses)."""
    n = len(ybar)
    if variant == "standard":
        model = fit_family(family, X, ybar, seed)
        tm = TreeMap(family, model, X, np.arange(n))
    else:
        perm = np.random.default_rng([seed, k, 1]).permutation(n)
        A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
        model = fit_family(family, X[A], ybar[A], seed)
        tm = TreeMap(family, model, X, B, Xsplit_rows=A)
    return tm


def check_map(tm, X, ybar):
    """Standard fit: the rebuilt map must reproduce the library's predictions."""
    err = float(np.abs(tm.predict(X, ybar[tm.rows]) - tm.model.predict(X)).max())
    # rf, gbm: exact (~1e-14). lineardt: ~1e-4, and the rebuilt leaves are the
    # MORE accurate least squares (lower SSE in 7 of 8 leaves on redraw 0;
    # linear-tree's own fit is the less precise one), so its tolerance is 1e-3.
    tol = 1e-3 if tm.family == "lineardt" else 1e-6
    assert err < tol, f"{tm.family}: rebuilt map disagrees with the library by {err:.2e}"
    return err


# --------------------------------------------------------------------------
# Radius (Theorem 1, simulated) by enumeration
# --------------------------------------------------------------------------
def sim_radius(tm, Xs, seed, k, n_draws, eps):
    S = tm.S(Xs)
    S, idx = np.unique(np.round(S, 12), axis=0, return_index=True)  # one row per cell (constant leaves)
    sn = np.linalg.norm(S, axis=1)
    keep = sn > 0
    S, sn = S[keep], sn[keep]
    W = []
    for i in range(n_draws):
        e = np.random.default_rng([seed, k, 11, i]).standard_normal(tm.nR)
        W.append(float((S @ e / sn).max()))
    kk = int(np.ceil((n_draws + 1) * (1 - eps)))
    return float(np.sort(W)[min(kk, n_draws) - 1]), int(len(S))


# --------------------------------------------------------------------------
# One (family, variant, arm) chunk of redraws
# --------------------------------------------------------------------------
def run(args):
    inst, md, sigma = setup()
    lb, ub = inst.variable_lb, inst.variable_ub
    X = md.X_train
    n = len(X)
    ystar, truth = arm_truth(args.arm, inst, md)
    last = args.first + args.n_redraws
    costs = cost_vectors("paper", last, inst)
    Xs = domain_sample(lb, ub, inst, args.n_sample, seed=0)
    tag = f"trees_{args.family}_{args.variant}_{args.arm}_{args.first:03d}"
    path = os.path.join(OUT_DIR, tag + ".json")
    out = json.load(open(path)) if os.path.exists(path) else {"runs": {}}
    out.update(family=args.family, variant=args.variant, arm=args.arm, sigma=sigma, eps=args.eps)

    for k in range(args.first, last):
        rec = out["runs"].get(str(k), {"rows": []})
        if rec.get("done"):
            continue
        done = {(r["kind"], r["param"]) for r in rec["rows"]}
        ybar = ystar + sigma * np.random.default_rng([args.seed, k]).standard_normal(n)
        cost = costs[k]
        obj_range = float(np.abs(cost) @ (ub - lb))
        t0 = time.time()
        tm = build_map(args.family, args.variant, X, ybar, args.seed, k)
        if args.variant == "standard":
            rec["map_check"] = check_map(tm, X, ybar)
        yR, ysR = ybar[tm.rows], ystar[tm.rows]
        if "R" not in rec:
            rec["R"], rec["n_cells"] = sim_radius(tm, Xs, args.seed, k, args.n_draws, args.eps)
        rec.update(obj_range=obj_range, empty_leaves=int(tm.empty), fit_s=time.time() - t0)
        R = rec["R"]

        def judge(xx):
            return {"x": xx.tolist(), "obj": float(cost @ xx),
                    "h": float(tm.predict(xx, yR)[0]),
                    "clean": float(tm.predict(xx, ysR)[0]),
                    "snorm": float(np.linalg.norm(tm.S(xx)[0])),
                    "truth": float(truth(xx))}

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

        def solve_tree(shift=0.0, g=0.0):
            m, u, x = _master(inst, lb, ub)
            f, t = tm.embed(m, x, lb, ub, yR, need_norm=g > 0)
            m.addConstr((f - g * R * sigma * t if g > 0 else f) >= PRODUCT_FLOOR + shift)
            return _solve(m, u, x, cost, lb, ub)

        if ("nominal", 0.0) not in done:
            add("nominal", 0.0, solve_tree())
        for g in RC_GRID:
            if ("rc", g) not in done:
                add("rc", g, solve_tree(g=g), R=R)
        for mg in FRONTIER_MARGIN:
            if ("margin", mg) not in done:
                add("margin", mg, solve_tree(shift=mg))

        if args.variant == "standard":
            perm = np.random.default_rng([args.seed, k, 2]).permutation(n)
            n_cal = int(round(CAL_FRAC * n))
            cal, tr = perm[:n_cal], perm[n_cal:]
            hmap = TreeMap(args.family, fit_family(args.family, X[tr], ybar[tr], args.seed), X, tr)
            yT = ybar[tr]
            resid = np.abs(ybar[tr] - hmap.predict(X[tr], yT))
            um, th_u = width_net(X[tr], resid, lb, ub, seed=k)
            u_cal = um.predict(X[cal], th_u)
            r_cal = np.abs(ybar[cal] - hmap.predict(X[cal], yT))
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(np.isnan(r_cal / u_cal), np.inf, r_cal / u_cal)
            rec["cmicl_info"] = {"n_neg_u": int((u_cal <= 0).sum()), "mean_u": float(u_cal.mean()),
                                 "cal_mae": float(r_cal.mean())}
            for a in CONF_ALPHAS:
                if ("cmicl", a) in done:
                    continue
                q = conformal_quantile(scores, a)
                if np.isfinite(q):
                    m, u, x = _master(inst, lb, ub)
                    f, _ = hmap.embed(m, x, lb, ub, yT, need_norm=False, prefix="c")
                    ue = um.gurobi_out(m, u, th_u)
                    m.addConstr(f - q * ue >= PRODUCT_FLOOR)
                    m.addConstr(ue >= 0.0)
                    res = _solve(m, u, x, cost, lb, ub)
                else:
                    res = {"status": -1, "time_s": 0.0, "x": None}
                ux = float(um.predict(res["x"][None], th_u)[0]) if res["x"] is not None else None
                add("cmicl", a, res, q=q, u_at_x=ux)
            if ("conformal", 0.1) not in done:
                q = conformal_quantile(r_cal, 0.1)
                m, u, x = _master(inst, lb, ub)
                f, _ = hmap.embed(m, x, lb, ub, yT, need_norm=False, prefix="c")
                m.addConstr(f >= PRODUCT_FLOOR + q)
                add("conformal", 0.1, _solve(m, u, x, cost, lb, ub), q=q)
        rec["done"] = True
        out["runs"][str(k)] = rec
        _save(tag + ".json", out)
        nom = next(r for r in rec["rows"] if r["kind"] == "nominal")
        rc1 = next(r for r in rec["rows"] if r["kind"] == "rc" and r["param"] == 1.0)
        print(f"{tag} redraw {k}: R {R:.2f} sigma ({rec['n_cells']} cells), nominal clean "
              f"{nom.get('clean', np.nan):.2f} truth {nom.get('truth', np.nan):.2f}; RC clean "
              f"{rc1.get('clean', np.nan):.2f} truth {rc1.get('truth', np.nan):.2f} cost "
              f"{100 * (rc1['obj'] - nom['obj']) / obj_range:+.2f}% ({rc1['time_s']:.0f}s); "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["rf", "gbm", "lineardt"], required=True)
    ap.add_argument("--variant", choices=["standard", "honest"], required=True)
    ap.add_argument("--arm", choices=["ode", "net"], default="ode")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--n-redraws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--eps", type=float, default=0.1)
    run(ap.parse_args())

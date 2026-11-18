"""Solver test for the robust counterpart (RC) on the reactor.

A probe for Paper A (paper/paper_plan.md), not part of the dial/rho pipeline: it
imports no runner and writes only under results/paper_a/prelim/.

Every model here is RIDGE ON A FEATURE MAP phi(x) (penalty LAM on the centred
features, unpenalised intercept), so the prediction is affine in the labels,

    f(x; theta(y)) = s(x)' y,   s(x) = 1/n + Zc M (phi(x) - phibar),

and the RC of the reactor's LOWER-bound constraint F_C6H6 >= 50 is

    f(x; theta_hat) - R * ||s(x)||_2 >= 50,

a second-order cone in phi(x). Three feature maps (``--model``):

    cubic         all monomials of degree <= 3 in the five design variables.
                  Fixed before any label is read: the theorem holds exactly.
    net_frozen    the last hidden layer of a 32x32 ReLU network (Ovalle et al.'s
                  architecture). Hidden layers trained ONCE on an independent
                  noise draw and frozen; each redraw refits the last layer only.
                  The theorem holds exactly, against the clean-label model that
                  shares those hidden layers.
    net_standard  the same network with the hidden layers retrained on EACH
                  redraw's own noisy labels, then the last layer refit on the
                  same labels. This is how the model would be fitted in
                  practice; the theorem's condition (s(x) independent of the
                  noise) fails, and the run measures by how much.
    net_honest    sample-split: on each redraw the hidden layers are trained on
                  a random half of the rows and the last layer is refit on the
                  other half. The theorem holds exactly with ONE dataset, at
                  the cost of half the rows in the last layer. net_frozen's
                  second noise draw on the same inputs is something a
                  practitioner does not have; this is the version they can run.

Two cost designs (``--costs``): ``balanced`` (the repo's fixed 1/span cost) and
``paper`` (Ovalle et al.'s draw, their exact random stream: U(-4,4), negative
components /10, against their /100-scaled inputs), one cost vector per redraw.

No trust region is added: the design is uniform over the box.

Stages:
    --stage solve    one noise draw: nominal + RC at each radius, checked against
                     enumeration over a random sample of the domain
    --stage radius   simulated radius through the solver (Dinkelbach), N draws
    --stage redraw   redraws: refit, solve at each radius, judge against the
                     clean-label model and the ODE
    --stage modelfree  probe F-1: the same redraws (same noise, same cost
                     vectors) at the MODEL-FREE radius sigma*sqrt(chi2(n, 1-eps)),
                     the norm of the whole noise vector. For a model affine in
                     its labels this is FO-CP at radius (a). Writes its own
                     files; never touches the committed redraw files, which
                     --stage modelfree_collect pairs it with.
"""
import argparse
import json
import os
import time
from itertools import combinations_with_replacement

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy import stats

from src.data.instances import load_config, reactor_instance
from src.data.dma_mr import PRODUCT_FLOOR

DEGREE = 3
LAM = 1.0            # fixed in advance; not selected on the labels
NET_HIDDEN = (32, 32)
NET_ALPHA = 0.01
OUT_DIR = os.path.join("results", "paper_a", "prelim")
TIME_LIMIT = 120     # guard only; a hit is reported as a status, never hidden
ENUM_INFLATE = 0.1   # sigma units added to an enumerated radius: the solver's W
                     # ran 0.03-0.1 above enumeration on the cubic (2026-11-14)
PAPER_INPUT_SCALE = np.array([0.01, 0.01, 0.01, 1.0, 0.01])


# --------------------------------------------------------------------------
# Feature maps on the scaled design u in [-1, 1]^d
# --------------------------------------------------------------------------
class CubicFeatures:
    def __init__(self, d):
        self.d = d
        self.monos = [m for k in range(1, DEGREE + 1)
                      for m in combinations_with_replacement(range(d), k)]

    def F(self, U):
        return np.column_stack([np.prod(U[:, list(m)], axis=1) for m in self.monos])

    def gurobi(self, m, u):
        mono = {(i,): u[i] for i in range(self.d)}
        for mm in self.monos:
            if len(mm) == 1:
                continue
            v = m.addVar(lb=-1.0, ub=1.0)
            m.addConstr(v == mono[mm[:-1]] * u[mm[-1]])
            mono[mm] = v
        return [mono[mm] for mm in self.monos]


class NetFeatures:
    """Activations of the last hidden layer of a ReLU network trained on (U, y)."""

    def __init__(self, U, y, seed, weights=None):
        self.d = U.shape[1]
        if weights is not None:          # saved by --stage prep
            self.Ws, self.bs = weights
            return
        from sklearn.neural_network import MLPRegressor
        mu, sd = float(np.mean(y)), float(np.std(y))
        net = MLPRegressor(hidden_layer_sizes=NET_HIDDEN, activation="relu",
                           solver="lbfgs", alpha=NET_ALPHA, max_iter=2000,
                           random_state=seed)
        net.fit(U, (y - mu) / sd)
        self.Ws = [w.copy() for w in net.coefs_[:-1]]
        self.bs = [b.copy() for b in net.intercepts_[:-1]]

    def F(self, U):
        a = U
        for W, b in zip(self.Ws, self.bs):
            a = np.maximum(a @ W + b, 0.0)
        return a

    def gurobi(self, m, u):
        # big-M ReLU embedding with interval bounds; binaries need a tight
        # integrality tolerance because big-M scales the slack (see CLAUDE.md).
        m.Params.IntFeasTol = 1e-9
        prev = [u[i] for i in range(self.d)]
        lo, hi = -np.ones(self.d), np.ones(self.d)
        for W, b in zip(self.Ws, self.bs):
            Wp, Wn = np.clip(W, 0, None), np.clip(W, None, 0)
            l = lo @ Wp + hi @ Wn + b
            h = hi @ Wp + lo @ Wn + b
            nxt = []
            for j in range(W.shape[1]):
                pre = gp.quicksum(float(W[i, j]) * prev[i]
                                  for i in range(W.shape[0])) + float(b[j])
                if h[j] <= 0:
                    a = m.addVar(lb=0.0, ub=0.0)
                elif l[j] >= 0:
                    a = m.addVar(lb=0.0, ub=float(h[j]))
                    m.addConstr(a == pre)
                else:
                    a = m.addVar(lb=0.0, ub=float(h[j]))
                    z = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(a >= pre)
                    m.addConstr(a <= pre - float(l[j]) * (1 - z))
                    m.addConstr(a <= float(h[j]) * z)
                nxt.append(a)
            prev = nxt
            lo, hi = np.maximum(l, 0.0), np.maximum(h, 0.0)
        return prev


# --------------------------------------------------------------------------
# Ridge on a feature map: everything that depends on the INPUTS only
# --------------------------------------------------------------------------
class FeatureRidge:
    def __init__(self, feat, X, lb, ub):
        self.feat, self.lb, self.ub = feat, lb, ub
        self.n, self.d = X.shape
        Z = self.features(X)
        self.p = Z.shape[1]
        self.phibar = Z.mean(axis=0)
        Zc = Z - self.phibar
        G = Zc.T @ Zc
        self.M = np.linalg.inv(G + LAM * np.eye(self.p))
        self.H = self.M @ Zc.T                 # p x n: beta = H y
        A = self.M @ G @ self.M                # ||s||^2 = 1/n + dphi' A dphi
        w, V = np.linalg.eigh(A)
        keep = w > 1e-12 * max(w.max(), 1e-300)
        self.Lt = (V[:, keep] * np.sqrt(w[keep])).T
        self.dim = int(np.linalg.matrix_rank(Zc)) + 1   # subspace dimension of s

    def scale(self, X):
        return 2.0 * (np.atleast_2d(X) - self.lb) / (self.ub - self.lb) - 1.0

    def features(self, X):
        return self.feat.F(self.scale(X))

    def fit(self, y):
        return float(np.mean(y)), self.H @ y

    def predict(self, X, coef):
        b0, beta = coef
        return b0 + (self.features(X) - self.phibar) @ beta

    def snorm(self, X):
        dphi = self.features(X) - self.phibar
        return np.sqrt(1.0 / self.n + np.sum((dphi @ self.Lt.T) ** 2, axis=1))

    def s_dot(self, X, eps):
        return float(np.mean(eps)) + (self.features(X) - self.phibar) @ (self.H @ eps)


# --------------------------------------------------------------------------
# Gurobi model: the domain, phi(x) and t >= ||s(x)||
# --------------------------------------------------------------------------
def base_model(cr, instance):
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = TIME_LIMIT
    if os.environ.get("GRB_THREADS"):    # on SLURM: stay inside the allocation
        m.Params.Threads = int(os.environ["GRB_THREADS"])
    u = m.addVars(cr.d, lb=-1.0, ub=1.0, name="u")
    span = cr.ub - cr.lb
    x = [cr.lb[i] + 0.5 * span[i] * (u[i] + 1.0) for i in range(cr.d)]
    for dc in instance.domain_constraints:
        m.addConstr(gp.quicksum(float(dc.coeffs[i]) * x[i] for i in range(cr.d))
                    <= dc.rhs)
    phi = cr.feat.gurobi(m, u)
    dphi = [phi[j] - float(cr.phibar[j]) for j in range(cr.p)]
    w = m.addVars(cr.Lt.shape[0], lb=-GRB.INFINITY)
    for k in range(cr.Lt.shape[0]):
        m.addConstr(w[k] == gp.quicksum(float(cr.Lt[k, j]) * dphi[j]
                                        for j in range(cr.p)))
    t = m.addVar(lb=0.0)
    m.addConstr(gp.quicksum(w[k] * w[k] for k in range(cr.Lt.shape[0]))
                + 1.0 / cr.n <= t * t)
    return m, u, x, dphi, t


def _x_of(cr, u):
    uu = np.clip(np.array([u[i].X for i in range(cr.d)]), -1.0, 1.0)
    return cr.lb + 0.5 * (cr.ub - cr.lb) * (uu + 1.0)


def solve_rc(cr, instance, coef, R, cost, shift=0.0):
    """min c'x  s.t.  f(x) - R ||s(x)|| >= 50 + shift.  R = 0 is nominal.

    ``shift`` > 0 with R = 0 is the CONSTANT margin baseline (the same
    tightening at every x); the frontier stage walks it beside the RC.
    """
    b0, beta = coef
    m, u, x, dphi, t = base_model(cr, instance)
    f = b0 + gp.quicksum(float(beta[j]) * dphi[j] for j in range(cr.p))
    m.addConstr(f - R * t >= PRODUCT_FLOOR + shift)
    m.setObjective(gp.quicksum(float(cost[i]) * x[i] for i in range(cr.d)),
                   GRB.MINIMIZE)
    t0 = time.time()
    m.optimize()
    out = {"R": R, "status": int(m.Status), "time": time.time() - t0,
           "x": None, "obj": np.nan, "bound": np.nan}
    if m.SolCount > 0:
        xs = _x_of(cr, u)
        out.update(x=xs, obj=float(cost @ xs), bound=float(m.ObjBound))
    return out


def solve_W(cr, instance, eps, tol=1e-4, max_iter=15):
    """W = sup_x s(x)'eps / ||s(x)|| over the domain, by Dinkelbach iterations.

    Returns the best ratio attained (W_lo) and a certified upper bound (W_hi):
    that ratio plus the last solve's bound on the residual over the smallest
    possible ||s||.
    """
    a = cr.H @ eps
    m, u, x, dphi, t = base_model(cr, instance)
    g = float(np.mean(eps)) + gp.quicksum(float(a[j]) * dphi[j] for j in range(cr.p))
    R, n_solves, t0 = 0.0, 0, time.time()
    smin = 1.0 / np.sqrt(cr.n)
    resid_bound = np.inf
    for _ in range(max_iter):
        m.setObjective(g - R * t, GRB.MAXIMIZE)
        m.optimize()
        n_solves += 1
        if m.SolCount == 0:
            break
        xs = _x_of(cr, u)
        resid_bound = float(m.ObjBound)
        ratio = float(cr.s_dot(xs, eps)[0] / cr.snorm(xs)[0])
        if m.ObjVal <= tol or ratio <= R + tol:
            R = max(R, ratio)
            break
        R = ratio
    return {"W_lo": R, "W_hi": R + max(resid_bound, 0.0) / smin,
            "n_solves": n_solves, "time": time.time() - t0, "status": int(m.Status)}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def domain_sample(lb, ub, instance, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(n, len(lb)))
    ok = np.ones(n, dtype=bool)
    for dc in instance.domain_constraints:
        ok &= X @ dc.coeffs <= dc.rhs
    return X[ok]


def enum_radius(cr, Xs, levels, n_draws, seed):
    """Simulated radius by enumeration, in sigma units, per level (inflated)."""
    rng = np.random.default_rng(seed)
    dF = cr.features(Xs) - cr.phibar
    sn = cr.snorm(Xs)
    W = []
    for start in range(0, n_draws, 100):
        E = rng.standard_normal((cr.n, min(100, n_draws - start)))
        num = E.mean(axis=0)[None, :] + dF @ (cr.H @ E)
        W.extend((num / sn[:, None]).max(axis=0))
    W = np.sort(W)
    out = {}
    for e in levels:
        k = int(np.ceil((n_draws + 1) * (1 - e)))
        out[e] = float(W[min(k, n_draws) - 1]) + ENUM_INFLATE
    return out


def enum_rc(cr, coef, R, Xs, cost):
    ok = cr.predict(Xs, coef) - R * cr.snorm(Xs) >= PRODUCT_FLOOR
    return float((Xs[ok] @ cost).min()) if ok.any() else np.nan


def ode_flow(instance, x):
    return -float(instance.gt_constraints[0](x))


def setup():
    config = load_config()
    inst = reactor_instance(config)
    md = inst.constraints[0].models_data[0]
    sigma = float(config["reactor"].get("noise_std", 2.0))
    return inst, md, sigma


def build(model, inst, md, sigma, y_struct=None, seed=0,
          struct_rows=None, fit_rows=None):
    """FeatureRidge for ``model``. ``y_struct`` are the labels the hidden layers
    are trained on (net models only), already restricted to ``struct_rows``;
    the last layer is refit on ``fit_rows`` (default: every row)."""
    lb, ub = inst.variable_lb, inst.variable_ub
    sr = slice(None) if struct_rows is None else struct_rows
    fr = slice(None) if fit_rows is None else fit_rows
    if model == "cubic":
        feat = CubicFeatures(md.X_train.shape[1])
    else:
        U = 2.0 * (md.X_train[sr] - lb) / (ub - lb) - 1.0
        weights = None
        if model == "net_frozen" and os.path.exists(FROZEN_NET):
            # one saved network for every array task: an L-BFGS fit is not
            # guaranteed bit-identical across nodes
            z = np.load(FROZEN_NET)
            k = len([f for f in z.files if f.startswith("W")])
            weights = ([z[f"W{i}"] for i in range(k)], [z[f"b{i}"] for i in range(k)])
        feat = NetFeatures(U, y_struct, seed, weights)
    return FeatureRidge(feat, md.X_train[fr], lb, ub)


FROZEN_NET = os.path.join(OUT_DIR, "net_frozen_weights.npz")


def stage_prep(args):
    """Train the frozen hidden layers once and save them."""
    inst, md, sigma = setup()
    if os.path.exists(FROZEN_NET):
        os.remove(FROZEN_NET)
    cr = build("net_frozen", inst, md, sigma, frozen_labels(md, sigma))
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(FROZEN_NET, **{f"W{i}": w for i, w in enumerate(cr.feat.Ws)},
             **{f"b{i}": b for i, b in enumerate(cr.feat.bs)})
    print(f"saved {FROZEN_NET}: dim(s)={cr.dim}")


def frozen_labels(md, sigma):
    """An independent noise draw for the frozen hidden layers (never reused)."""
    return md.y_true + sigma * np.random.default_rng(987654321).standard_normal(len(md.y_true))


def radii(cr, sigma, eps_level, sim=None):
    r = {"nominal": 0.0,
         "pointwise": sigma * stats.norm.ppf(1 - eps_level),
         "chi2": sigma * np.sqrt(stats.chi2.ppf(1 - eps_level, cr.dim))}
    if sim is not None:
        r["simulated"] = sigma * sim
    return r


def cost_vectors(kind, n, inst):
    if kind == "balanced":
        return np.tile(inst.cost_vector, (n, 1))
    c = np.random.RandomState(0).uniform(-4.0, 4.0, size=(n, len(inst.cost_vector)))
    c[c < 0] /= 10.0
    return c * PAPER_INPUT_SCALE


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_solve(args):
    inst, md, sigma = setup()
    y_struct = frozen_labels(md, sigma) if args.model == "net_frozen" else md.y_train
    cr = build(args.model, inst, md, sigma, y_struct)
    coef, clean = cr.fit(md.y_train), cr.fit(md.y_true)
    resid = md.y_train - cr.predict(md.X_train, coef)
    Xs = domain_sample(cr.lb, cr.ub, inst, args.n_sample, seed=0)
    sn = cr.snorm(Xs)
    near = np.abs(cr.predict(Xs, clean) - PRODUCT_FLOOR) < 5
    print(f"{args.model}: p={cr.p}, dim(s)={cr.dim}, lam={LAM}, n={cr.n}, "
          f"train R2={1 - resid.var() / md.y_train.var():.4f}")
    print(f"domain sample: {len(Xs)} points; ||s|| median {np.median(sn):.3f}, "
          f"max {sn.max():.3f}; near-boundary points {int(near.sum())}")
    sim = enum_radius(cr, Xs, [args.eps], args.n_sim, seed=1)[args.eps]
    cost = cost_vectors(args.costs, 1, inst)[0]
    rows = []
    for name, R in radii(cr, sigma, args.eps, sim=sim).items():
        r = solve_rc(cr, inst, coef, R, cost)
        row = {"radius": name, "R": R, "status": r["status"], "time_s": r["time"],
               "obj": r["obj"], "bound": r["bound"],
               "obj_enum": enum_rc(cr, coef, R, Xs, cost)}
        if r["x"] is not None:
            xs = r["x"]
            row.update(margin=float(R * cr.snorm(xs)[0]),
                       f_hat=float(cr.predict(xs, coef)[0]),
                       f_clean=float(cr.predict(xs, clean)[0]),
                       f_ode=ode_flow(inst, xs), x=xs.tolist())
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items() if k != "x"}), flush=True)
    _save(f"solve_{args.model}_{args.costs}.json", rows)


def stage_radius(args):
    inst, md, sigma = setup()
    cr = build(args.model, inst, md, sigma, frozen_labels(md, sigma))
    rng = np.random.default_rng(args.seed)
    Xs = domain_sample(cr.lb, cr.ub, inst, args.n_sample, seed=0)
    rows = []
    for k in range(args.n_draws):
        eps = rng.standard_normal(cr.n)
        r = solve_W(cr, inst, eps)
        r["W_enum"] = float((cr.s_dot(Xs, eps) / cr.snorm(Xs)).max())
        rows.append(r)
        print(f"draw {k+1}: W_lo={r['W_lo']:.3f} W_hi={r['W_hi']:.3f} "
              f"enum={r['W_enum']:.3f} solves={r['n_solves']} "
              f"time={r['time']:.1f}s status={r['status']}", flush=True)
    _save(f"radius_{args.model}.json", rows)


def stage_redraw(args):
    inst, md, sigma = setup()
    last = args.first + args.n_redraws
    costs = cost_vectors(args.costs, last, inst)
    span = inst.variable_ub - inst.variable_lb
    Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, args.n_sample, seed=0)
    fixed = args.model in ("cubic", "net_frozen")
    if fixed:
        cr = build(args.model, inst, md, sigma, frozen_labels(md, sigma))
        sim = enum_radius(cr, Xs, [args.eps], args.n_sim, seed=1)[args.eps]
        print(f"{args.model}: dim(s)={cr.dim}, simulated radius {sim:.2f} sigma "
              f"(enumeration + {ENUM_INFLATE}), chi2 "
              f"{np.sqrt(stats.chi2.ppf(1 - args.eps, cr.dim)):.2f} sigma", flush=True)
    tag = f"redraw_{args.model}_{args.costs}_eps{args.eps}_{args.first:03d}"
    path = os.path.join(OUT_DIR, tag + ".json")
    # resume: a requeued or restarted task keeps the redraws it finished
    rows = json.load(open(path)) if os.path.exists(path) else []
    n_rad = 4
    done = {k for k in {r["redraw"] for r in rows}
            if sum(r["redraw"] == k for r in rows) == n_rad}
    rows = [r for r in rows if r["redraw"] in done]
    for k in range(args.first, last):
        if k in done:
            continue
        # redraw k is the same noise whichever task or chunk runs it
        rng = np.random.default_rng([args.seed, k])
        y = md.y_true + sigma * rng.standard_normal(len(md.y_true))
        y_fit, y_clean = y, md.y_true
        if args.model == "net_honest":
            # own stream, so the noise of redraw k is the same in every arm
            perm = np.random.default_rng([args.seed, k, 1]).permutation(len(y))
            A, B = perm[: len(y) // 2], perm[len(y) // 2:]
            cr = build(args.model, inst, md, sigma, y[A], seed=k,
                       struct_rows=A, fit_rows=B)
            y_fit, y_clean = y[B], md.y_true[B]
        elif not fixed:
            cr = build(args.model, inst, md, sigma, y, seed=k)
        if not fixed:
            sim = enum_radius(cr, Xs, [args.eps], args.n_sim, seed=1)[args.eps]
        coef, clean = cr.fit(y_fit), cr.fit(y_clean)
        c = costs[k]
        obj_range = float(np.abs(c) @ span)
        for name, R in radii(cr, sigma, args.eps, sim=sim).items():
            r = solve_rc(cr, inst, coef, R, c)
            row = {"redraw": k, "radius": name, "R": R, "status": r["status"],
                   "time_s": r["time"], "obj": r["obj"], "obj_range": obj_range}
            if r["x"] is not None:
                xs = r["x"]
                row.update(f_clean=float(cr.predict(xs, clean)[0]),
                           f_ode=ode_flow(inst, xs),
                           margin=float(R * cr.snorm(xs)[0]), x=xs.tolist())
            rows.append(row)
        print(f"redraw {k} done", flush=True)
        _save(tag + ".json", rows)
    _save(tag + ".json", rows)
    summarize(rows, tag)


def stage_modelfree(args):
    """Probe F-1 (paper_plan.md): is the model-free radius affordable?

    Same redraws as --stage redraw (noise from default_rng([seed, k]), cost
    vector k), one solve each at R = sigma * sqrt(chi2.ppf(1 - eps, n)): a
    label ball containing the whole noise vector with probability 1 - eps,
    valid for ANY training map. Fixed-structure models only, so the rows pair
    with the committed redraw files for the nominal and simulated rows.
    """
    if args.model not in ("cubic", "net_frozen"):
        raise SystemExit("--stage modelfree: cubic or net_frozen only")
    inst, md, sigma = setup()
    last = args.first + args.n_redraws
    costs = cost_vectors(args.costs, last, inst)
    span = inst.variable_ub - inst.variable_lb
    cr = build(args.model, inst, md, sigma, frozen_labels(md, sigma))
    g = float(np.sqrt(stats.chi2.ppf(1 - args.eps, cr.n)))
    R = sigma * g
    print(f"{args.model}: n={cr.n}, model-free radius {g:.2f} sigma "
          f"(R = {R:.2f}), chi2 {np.sqrt(stats.chi2.ppf(1 - args.eps, cr.dim)):.2f} "
          f"sigma", flush=True)
    tag = f"modelfree_{args.model}_{args.costs}_eps{args.eps}_{args.first:03d}"
    path = os.path.join(OUT_DIR, tag + ".json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {r["redraw"] for r in rows}
    for k in range(args.first, last):
        if k in done:
            continue
        rng = np.random.default_rng([args.seed, k])
        y = md.y_true + sigma * rng.standard_normal(len(md.y_true))
        coef, clean = cr.fit(y), cr.fit(md.y_true)
        c = costs[k]
        r = solve_rc(cr, inst, coef, R, c)
        row = {"redraw": k, "radius": "modelfree", "R": R, "status": r["status"],
               "time_s": r["time"], "obj": r["obj"],
               "obj_range": float(np.abs(c) @ span)}
        if r["x"] is not None:
            xs = r["x"]
            row.update(f_clean=float(cr.predict(xs, clean)[0]),
                       f_ode=ode_flow(inst, xs),
                       margin=float(R * cr.snorm(xs)[0]), x=xs.tolist())
        rows.append(row)
        print(f"redraw {k}: status {r['status']}, obj {r['obj']:.4f}, "
              f"{r['time']:.1f}s", flush=True)
        _save(tag + ".json", rows)
    _save(tag + ".json", rows)


def stage_modelfree_collect(args):
    """Pair the model-free rows with the committed redraw rows and summarize."""
    import glob
    for model in ("cubic", "net_frozen"):
        pat = f"_{model}_{args.costs}_eps{args.eps}_*.json"
        base = [r for f in sorted(glob.glob(os.path.join(OUT_DIR, "redraw" + pat)))
                for r in json.load(open(f))]
        mf = [r for f in sorted(glob.glob(os.path.join(OUT_DIR, "modelfree" + pat)))
              for r in json.load(open(f))]
        if not mf:
            continue
        ks = {r["redraw"] for r in mf}
        rows = [r for r in base if r["redraw"] in ks] + mf
        summarize(rows, f"F-1 {model}, {args.costs} costs, eps={args.eps}, "
                        f"{len(ks)} redraws")


# Frontier grids. The RC's radius is in SIGMA units (R = sigma * g), the
# constant margin in FLOW units (F_C6H6 >= 50 + m). Both start at the nominal
# decision (one shared row, kind="nominal"). The pointwise and simulated radii
# are added to the RC grid so the protocol points sit on the curve.
FRONTIER_RC = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
FRONTIER_MARGIN = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
                   2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


DESIGN_SEED = 2026    # fixes WHICH rows a design keeps; the noise is redrawn
THIN_ABOVE = 45.0     # flow above which "thin" keeps only a fraction of rows


def design_rows(md, args):
    """Training rows for the frontier stage, and the tag suffix naming them.

    full   every row (R1's design; no suffix, so its files are R1-paired)
    sub    a uniform random subset of --n-train rows
    thin   every row with clean flow <= THIN_ABOVE, and a --thin-keep fraction
           of the rest: historical runs rarely near the high-yield region where
           the optimizer lands. Selection is on F(x), a function of x alone, so
           s(x) stays independent of the noise and Theorem 1 holds unchanged.
    """
    n = len(md.y_true)
    rng = np.random.default_rng(DESIGN_SEED)
    if args.design == "full":
        return np.arange(n), ""
    if args.design == "sub":
        return np.sort(rng.choice(n, args.n_train, replace=False)), f"_n{args.n_train}"
    high = md.y_true > THIN_ABOVE
    keep = ~high | (rng.random(n) < args.thin_keep)
    return np.flatnonzero(keep), f"_thin{args.thin_keep:g}"


def stage_frontier(args):
    """H2 probe: the RC over a radius grid against a constant margin over an m
    grid, on the SAME redraws as stage_redraw (same noise stream, same cost
    vector per redraw), so the curves are paired with each other and with R1.
    Fixed feature maps only: the radius is enumerated once, as in R1.

    The margin is given its whole grid and is read at whichever m first
    delivers -- i.e. tuned against the ODE, which no practitioner can do. That
    favours the baseline; the RC is read at its own (untuned) simulated radius.
    """
    if args.model not in ("cubic", "net_frozen"):
        raise SystemExit("frontier: fixed feature maps only (cubic, net_frozen)")
    inst, md, sigma = setup()
    last = args.first + args.n_redraws
    costs = cost_vectors(args.costs, last, inst)
    span = inst.variable_ub - inst.variable_lb
    Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, args.n_sample, seed=0)
    rows, suffix = design_rows(md, args)
    cr = build(args.model, inst, md, sigma, frozen_labels(md, sigma), fit_rows=rows)
    sim = enum_radius(cr, Xs, [args.eps], args.n_sim, seed=1)[args.eps]
    point = float(stats.norm.ppf(1 - args.eps))
    print(f"{args.model}{suffix}: n={len(rows)} ({int((md.y_true[rows] > THIN_ABOVE).sum())} "
          f"with flow > {THIN_ABOVE:g}), dim(s)={cr.dim}, simulated radius {sim:.2f} "
          f"sigma, pointwise {point:.2f} sigma", flush=True)
    grid = [("nominal", "none", 0.0, 0.0, 0.0)]
    grid += [("rc", f"{g:g}", g, sigma * g, 0.0) for g in FRONTIER_RC]
    grid += [("rc", "pointwise", point, sigma * point, 0.0),
             ("rc", "simulated", sim, sigma * sim, 0.0)]
    grid += [("margin", f"{m:g}", m, 0.0, m) for m in FRONTIER_MARGIN]

    tag = f"frontier_{args.model}{suffix}_{args.costs}_eps{args.eps}_{args.first:03d}"
    path = os.path.join(OUT_DIR, tag + ".json")
    out = json.load(open(path)) if os.path.exists(path) else []
    done = {k for k in {r["redraw"] for r in out}
            if sum(r["redraw"] == k for r in out) == len(grid)}
    out = [r for r in out if r["redraw"] in done]
    for k in range(args.first, last):
        if k in done:
            continue
        # == stage_redraw's stream over ALL rows, then restricted: a row kept by
        # a design carries the same noise as in R1, so designs are paired
        rng = np.random.default_rng([args.seed, k])
        y = md.y_true + sigma * rng.standard_normal(len(md.y_true))
        coef, clean = cr.fit(y[rows]), cr.fit(md.y_true[rows])
        c = costs[k]
        obj_range = float(np.abs(c) @ span)
        for kind, label, dial, R, shift in grid:
            r = solve_rc(cr, inst, coef, R, c, shift=shift)
            row = {"redraw": k, "kind": kind, "label": label, "dial": dial,
                   "R": R, "shift": shift, "status": r["status"],
                   "time_s": r["time"], "obj": r["obj"], "obj_range": obj_range}
            if r["x"] is not None:
                xs = r["x"]
                row.update(f_clean=float(cr.predict(xs, clean)[0]),
                           f_ode=ode_flow(inst, xs),
                           margin=float(R * cr.snorm(xs)[0] + shift),
                           x=xs.tolist())
            out.append(row)
        print(f"redraw {k} done", flush=True)
        _save(tag + ".json", out)
    _save(tag + ".json", out)


CONF_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
CAL_FRAC = 0.2        # Ovalle et al.'s 80/20 split (cmicl.py)


def solve_lin(cr, instance, coef, cost, nonneg=None):
    """min c'x  s.t.  coef(x) >= 50 [and nonneg(x) >= 0], both affine in phi(x)."""
    m, u, x, dphi, t = base_model(cr, instance)
    lin = lambda cf: cf[0] + gp.quicksum(float(cf[1][j]) * dphi[j] for j in range(cr.p))
    m.addConstr(lin(coef) >= PRODUCT_FLOOR)
    if nonneg is not None:
        m.addConstr(lin(nonneg) >= 0.0)
    m.setObjective(gp.quicksum(float(cost[i]) * x[i] for i in range(cr.d)), GRB.MINIMIZE)
    t0 = time.time()
    m.optimize()
    out = {"status": int(m.Status), "time": time.time() - t0, "x": None, "obj": np.nan}
    if m.SolCount > 0:
        xs = _x_of(cr, u)
        out.update(x=xs, obj=float(cost @ xs))
    return out


def stage_conformal(args):
    """QUICK C-MICL on cubic ridge, beside the split-conformal constant margin.

    Per redraw: h = cubic ridge on a random 80% of the design's rows; score the
    other 20% with |y - h| / u. Two width models:
      cmicl_poly  u = cubic ridge on h's ABSOLUTE in-sample residuals, used raw
                  in the score and constrained u(x) >= 0 in the MIP (Ovalle et
                  al.'s semantics, as in cmicl.py) -- but u is a POLYNOMIAL,
                  not their 32x32 network, so h - q u is linear in phi(x).
      conformal   u = 1: split conformal with a constant width (a constant
                  margin q set by held-out residuals, no oracle).
    q = s_(k), k = ceil((n_cal+1)(1-alpha)); k > n_cal is recorded unsolved.
    Same noise stream and cost vectors as R1 and the frontier stage; the
    nominal row is h itself (fit on 80%), so its cost baseline is C-MICL's own.
    """
    inst, md, sigma = setup()
    last = args.first + args.n_redraws
    costs = cost_vectors(args.costs, last, inst)
    span = inst.variable_ub - inst.variable_lb
    rows, suffix = design_rows(md, args)
    tag = f"conformal_{args.model}{suffix}_{args.costs}_{args.first:03d}"
    path = os.path.join(OUT_DIR, tag + ".json")
    out = json.load(open(path)) if os.path.exists(path) else []
    n_per = 1 + 2 * len(CONF_ALPHAS)
    done = {k for k in {r["redraw"] for r in out}
            if sum(r["redraw"] == k for r in out) == n_per}
    out = [r for r in out if r["redraw"] in done]
    for k in range(args.first, last):
        if k in done:
            continue
        rng = np.random.default_rng([args.seed, k])
        y = md.y_true + sigma * rng.standard_normal(len(md.y_true))
        perm = np.random.default_rng([args.seed, k, 2]).permutation(rows)
        n_cal = int(round(CAL_FRAC * len(rows)))
        cal, tr = perm[:n_cal], perm[n_cal:]
        cr = build("cubic", inst, md, sigma, fit_rows=tr)
        h = cr.fit(y[tr])
        clean = build("cubic", inst, md, sigma).fit(md.y_true)   # all rows, as R1
        clean_cr = build("cubic", inst, md, sigma)
        res_tr = np.abs(y[tr] - cr.predict(md.X_train[tr], h))
        uc = cr.fit(res_tr)                                       # width model
        u_cal = cr.predict(md.X_train[cal], uc)
        r_cal = np.abs(y[cal] - cr.predict(md.X_train[cal], h))
        c = costs[k]
        obj_range = float(np.abs(c) @ span)
        plan = [("nominal", 0.0, None)]
        for a in CONF_ALPHAS:
            plan += [("cmicl_poly", a, r_cal / u_cal), ("conformal", a, r_cal)]
        for kind, a, scores in plan:
            if scores is None:
                q = 0.0
            else:
                kk = int(np.ceil((n_cal + 1) * (1 - a)))
                q = float(np.sort(scores)[kk - 1]) if kk <= n_cal else np.inf
            row = {"redraw": k, "kind": kind, "alpha": a, "q": q, "n_cal": n_cal,
                   "obj_range": obj_range, "obj": np.nan, "status": -1}
            if np.isfinite(q):
                if kind == "cmicl_poly":
                    coef = (h[0] - q * uc[0], h[1] - q * uc[1])
                    r = solve_lin(cr, inst, coef, c, nonneg=uc)
                else:
                    r = solve_lin(cr, inst, (h[0] - q, h[1]), c)
                row.update(status=r["status"], obj=r["obj"], time_s=r["time"])
                if r["x"] is not None:
                    xs = r["x"]
                    row.update(f_clean=float(clean_cr.predict(xs, clean)[0]),
                               f_ode=ode_flow(inst, xs),
                               u_at_x=float(cr.predict(xs, uc)[0]) if kind == "cmicl_poly" else 1.0,
                               x=xs.tolist())
            out.append(row)
        print(f"redraw {k} done", flush=True)
        _save(tag + ".json", out)
    _save(tag + ".json", out)


def stage_conformal_collect(args):
    import glob
    import pandas as pd
    _, suffix = design_rows(setup()[1], args)
    files = sorted(glob.glob(os.path.join(
        OUT_DIR, f"conformal_{args.model}{suffix}_{args.costs}_*.json")))
    df = pd.DataFrame([r for f in files for r in json.load(open(f))])
    # cost against the FRONTIER's nominal (h on all of the design's rows, same
    # cost vector), so these rows read on the RC/margin frontier's axis; the
    # 80%-row h of this stage is kept as kind="nominal" for its own feasibility
    ff = glob.glob(os.path.join(OUT_DIR, f"frontier_{args.model}{suffix}_{args.costs}_eps0.1_*.json"))
    fr = [r for f in ff for r in json.load(open(f)) if r["kind"] == "nominal"]
    nom = (pd.DataFrame(fr).set_index("redraw")["obj"] if fr else
           df[df.kind == "nominal"].set_index("redraw")["obj"])
    print("cost baseline:", "frontier nominal (all rows)" if fr else "this stage's 80% nominal")
    df["cost_pct"] = 100 * (df.obj - df.redraw.map(nom)) / df.obj_range
    sv = df.f_ode.notna()
    df["feas_ode"] = np.where(sv, df.f_ode >= PRODUCT_FLOOR, np.nan)
    df["feas_clean"] = np.where(sv, df.f_clean >= PRODUCT_FLOOR - 1e-6, np.nan)
    g = df.groupby(["kind", "alpha"], sort=False)
    tab = pd.DataFrame({"n": g.redraw.nunique(), "q": g.q.median(),
                        "solved": g.f_ode.apply(lambda s: s.notna().mean()),
                        "feas_clean": g.feas_clean.mean(), "feas_ode": g.feas_ode.mean(),
                        "cost_pct": g.cost_pct.mean(), "u_at_x": g.u_at_x.mean(),
                        "time_s": g.time_s.mean()}).reset_index()
    tag = f"conformal_{args.model}{suffix}_{args.costs}"
    print(f"\n{tag}: {df.redraw.nunique()} redraws, n_cal {int(df.n_cal.iloc[0])}")
    with pd.option_context("display.width", 200):
        print(tab.round(3).to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, tag + ".csv"), index=False)


def stage_frontier_collect(args):
    """Merge frontier chunks; one row per (kind, setting), and the cheapest
    setting of each kind reaching the target against the ODE."""
    import glob
    import pandas as pd
    _, suffix = design_rows(setup()[1], args)
    files = sorted(glob.glob(os.path.join(
        OUT_DIR, f"frontier_{args.model}{suffix}_{args.costs}_eps{args.eps}_*.json")))
    rows = [r for f in files for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    nom = df[df.kind == "nominal"].set_index("redraw")["obj"]
    df["cost_pct"] = 100 * (df.obj - df.redraw.map(nom)) / df.obj_range
    solved = df.f_ode.notna()
    df["feas_ode"] = np.where(solved, df.f_ode >= PRODUCT_FLOOR, np.nan)
    df["feas_clean"] = np.where(solved, df.f_clean >= PRODUCT_FLOOR - 1e-6, np.nan)
    df["short_ode"] = np.where(solved & (df.f_ode < PRODUCT_FLOOR),
                               PRODUCT_FLOOR - df.f_ode, np.nan)
    g = df.groupby(["kind", "label"], sort=False)
    tab = pd.DataFrame({
        "dial": g.dial.first(), "R": g.R.mean(), "n": g.redraw.nunique(),
        "solved": g.f_ode.apply(lambda s: s.notna().mean()),
        "feas_clean": g.feas_clean.mean(), "feas_ode": g.feas_ode.mean(),
        "se_ode": g.feas_ode.apply(lambda s: np.sqrt(s.mean() * (1 - s.mean()) / s.notna().sum())),
        "short_ode": g.short_ode.mean(), "worst_ode": g.short_ode.max(),
        "cost_pct": g.cost_pct.mean(), "margin_at_x": g.margin.mean(),
        "time_s": g.time_s.mean()}).reset_index()
    tab = tab.sort_values(["kind", "dial"], kind="stable")
    tag = f"frontier_{args.model}{suffix}_{args.costs}_eps{args.eps}"
    print(f"\n{tag}: {df.redraw.nunique()} redraws from {len(files)} files")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(tab.round(3).to_string(index=False))
    target = 1 - args.eps
    print(f"\ncheapest setting with ODE feasibility >= {target}:")
    for kind in ("rc", "margin"):
        ok = tab[(tab.kind == kind) & (tab.feas_ode >= target) & (tab.solved >= 0.99)]
        if ok.empty:
            print(f"  {kind:7s} none on the grid")
        else:
            b = ok.loc[ok.cost_pct.idxmin()]
            print(f"  {kind:7s} {b.label:>9s}  feas_ode {b.feas_ode:.2f}  "
                  f"cost {b.cost_pct:.2f}%  mean margin at x* {b.margin_at_x:.2f}")
    tab.to_csv(os.path.join(OUT_DIR, tag + ".csv"), index=False)


def stage_collect(args):
    """Merge the chunk files of each model and print one table per model."""
    import glob
    for model in ("cubic", "net_frozen", "net_standard", "net_honest"):
        files = sorted(glob.glob(os.path.join(
            OUT_DIR, f"redraw_{model}_{args.costs}_eps{args.eps}_*.json")))
        rows = [r for f in files for r in json.load(open(f))]
        if rows:
            n = len({r["redraw"] for r in rows})
            summarize(rows, f"{model}, {args.costs} costs, eps={args.eps}, "
                            f"{n} redraws from {len(files)} files")


def summarize(rows, tag):
    names = list(dict.fromkeys(r["radius"] for r in rows))
    nom = {r["redraw"]: r["obj"] for r in rows if r["radius"] == "nominal"}
    print(f"\n{tag}")
    # short_*: mean shortfall below the floor among the VIOLATING runs only, so
    # a rate of 0.85 can be read as "missed by 0.2" or "missed by 5"
    print(f"{'radius':10s} {'R':>6s} {'solved':>7s} {'opt':>5s} {'feas_clean':>10s} "
          f"{'feas_ode':>9s} {'short_cl':>8s} {'short_ode':>9s} {'worst_ode':>9s} "
          f"{'cost+%':>7s} {'margin':>7s} {'time':>6s}")
    for name in names:
        rr = [r for r in rows if r["radius"] == name]
        sv = [r for r in rr if "f_ode" in r]
        if not sv:
            print(f"{name:10s} solved 0 of {len(rr)}")
            continue
        extra = [100 * (r["obj"] - nom[r["redraw"]]) / r["obj_range"] for r in sv
                 if np.isfinite(nom.get(r["redraw"], np.nan))]
        gap_cl = [PRODUCT_FLOOR - r["f_clean"] for r in sv
                  if r["f_clean"] < PRODUCT_FLOOR - 1e-6]
        gap_ode = [PRODUCT_FLOOR - r["f_ode"] for r in sv if r["f_ode"] < PRODUCT_FLOOR]
        nanmean = lambda v: float(np.mean(v)) if v else 0.0
        print(f"{name:10s} {np.mean([r['R'] for r in rr]):6.2f} "
              f"{len(sv)/len(rr):7.2f} "
              f"{np.mean([r['status'] == 2 for r in rr]):5.2f} "
              f"{np.mean([r['f_clean'] >= PRODUCT_FLOOR - 1e-6 for r in sv]):10.3f} "
              f"{np.mean([r['f_ode'] >= PRODUCT_FLOOR for r in sv]):9.3f} "
              f"{nanmean(gap_cl):8.2f} {nanmean(gap_ode):9.2f} "
              f"{max(gap_ode, default=0.0):9.2f} "
              f"{np.mean(extra):7.2f} "
              f"{np.mean([r['margin'] for r in sv]):7.2f} "
              f"{np.mean([r['time_s'] for r in rr]):6.1f}")


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["solve", "radius", "redraw", "prep", "collect",
                             "frontier", "frontier_collect", "conformal",
                             "conformal_collect", "modelfree",
                             "modelfree_collect"])
    ap.add_argument("--first", type=int, default=0, help="first redraw index")
    ap.add_argument("--model",
                    choices=["cubic", "net_frozen", "net_standard", "net_honest"],
                    default="cubic")
    ap.add_argument("--costs", choices=["balanced", "paper"], default="balanced")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n-sample", type=int, default=100000)
    ap.add_argument("--n-sim", type=int, default=500,
                    help="noise draws for the enumerated simulated radius")
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--n-redraws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", choices=["full", "sub", "thin"], default="full",
                    help="frontier: which training rows (see design_rows)")
    ap.add_argument("--n-train", type=int, default=200, help="--design sub")
    ap.add_argument("--thin-keep", type=float, default=0.2, help="--design thin")
    args = ap.parse_args()
    {"solve": stage_solve, "radius": stage_radius, "redraw": stage_redraw,
     "prep": stage_prep, "collect": stage_collect, "frontier": stage_frontier,
     "frontier_collect": stage_frontier_collect, "conformal": stage_conformal,
     "conformal_collect": stage_conformal_collect, "modelfree": stage_modelfree,
     "modelfree_collect": stage_modelfree_collect}[args.stage](args)

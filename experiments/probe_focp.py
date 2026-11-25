"""FO-CP machinery (I11, I12) and probe F0 (paper/paper_plan.md, "FO-CP").

A probe for Paper A: it imports no runner and writes only under
results/paper_a/prelim/. Reactor data, network architecture, noise stream and
cost vectors are probe_rc_ridge.py's, so redraw k here is redraw k there.

I11, the training map T(y). The (32, 32) ReLU network of probe_rc_ridge.py,
trained END TO END on its exact sklearn objective

    L(theta; y) = 0.5 mean_i (f(u_i; theta) - (y_i - mu)/sd)^2
                  + 0.5 alpha sum ||W||^2 / n,

by scipy's L-BFGS-B to a GRADIENT tolerance (gtol 1e-7, ftol 0: no stop on a
small loss change), always from one fixed theta0, with mu and sd fixed from
ybar so that T is a function of the labels alone. theta0 is sklearn's cold
fit on ybar, polished by the same L-BFGS-B, so T(ybar) = theta0. Gradients,
the Hessian and the mixed derivative d(grad_theta L)/dy come from torch
autograd on that same objective (float64).

I12, the separation. The implicit gradient of the retrained prediction in the
labels,

    d h(x; T(y)) / dy = - L_theta_y' (H + lam I)^{-1} grad_theta h(x; theta),

with lam = lam_rel * tr(H)/p, and monotone projected gradient ascent over the
whitened label ball ||z|| <= R, Delta y = sigma z.

The reactor constraint is a LOWER bound (F_C6H6 >= 50), so the separation
pushes the retrained prediction DOWN: it ascends sign * h with sign = -1.

Stages:
    --stage f0_affine  F0(i). Ridge on fixed features (cubic, net_frozen): the
                       generic implicit gradient must equal s(x), one ascent
                       step from zero must move the prediction by exactly
                       R sigma ||s(x)||, and no further step may be accepted.
    --stage f0_grad    F0(ii). End-to-end network: the implicit gradient
                       against central finite differences of retraining, row by
                       row, at the nominal decision, a tightened decision and
                       random points near the boundary, for a grid of lam_rel.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_rc_ridge import (NET_ALPHA, NET_HIDDEN, LAM, OUT_DIR, TIME_LIMIT,  # noqa: E402
                            NetFeatures, build, cost_vectors, domain_sample,
                            frozen_labels, ode_flow, setup)
from src.data.dma_mr import PRODUCT_FLOOR  # noqa: E402

torch.set_default_dtype(torch.float64)

GTOL = 1e-7
MAX_ITER = 20000
MAX_FUN = 50000
LAM_GRID = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]   # x tr(H)/p
SIGN = -1.0            # lower-bound constraint: the adversary lowers h


def scale(X, lb, ub):
    return 2.0 * (np.atleast_2d(X) - lb) / (ub - lb) - 1.0


def noisy_labels(md, sigma, seed, k):
    """Redraw k's labels: probe_rc_ridge.stage_redraw's stream."""
    rng = np.random.default_rng([seed, k])
    return md.y_true + sigma * rng.standard_normal(len(md.y_true))


# --------------------------------------------------------------------------
# Models: train(y, theta_init), loss_t(theta, y), pred_t(Z, theta)
# --------------------------------------------------------------------------
class MLPModel:
    """The end-to-end (32, 32) ReLU network; theta packs [W1, b1, W2, b2, W3, b3]."""

    def __init__(self, X, lb, ub, ybar, alpha=NET_ALPHA, hidden=NET_HIDDEN):
        self.lb, self.ub, self.alpha = lb, ub, alpha
        self.U = scale(X, lb, ub)
        self.n, self.d = self.U.shape
        self.mu, self.sd = float(np.mean(ybar)), float(np.std(ybar))
        self.sizes = [self.d, *hidden, 1]
        self.shapes = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.shapes += [(a, b), (b,)]
        self.p = int(sum(np.prod(s) for s in self.shapes))
        self.U_t = torch.tensor(self.U)

    def unpack(self, theta):
        out, i = [], 0
        for s in self.shapes:
            k = int(np.prod(s))
            out.append(theta[i:i + k].reshape(s))
            i += k
        return out

    def pack(self, Ws, bs):
        return np.concatenate([a.ravel() for W, b in zip(Ws, bs) for a in (W, b)])

    def net_t(self, U_t, theta_t):
        parts = self.unpack(theta_t)
        a = U_t
        for j in range(0, len(parts), 2):
            a = a @ parts[j] + parts[j + 1]
            if j < len(parts) - 2:
                a = torch.relu(a)
        return a[:, 0]

    def loss_t(self, theta_t, y_t):
        ys = (y_t - self.mu) / self.sd
        r = self.net_t(self.U_t, theta_t) - ys
        parts = self.unpack(theta_t)
        l2 = sum((parts[j] ** 2).sum() for j in range(0, len(parts), 2))
        return 0.5 * (r ** 2).mean() + 0.5 * self.alpha * l2 / self.n

    def features_t(self, X):
        return torch.tensor(scale(X, self.lb, self.ub))

    def pred_t(self, Z_t, theta_t):
        return self.mu + self.sd * self.net_t(Z_t, theta_t)

    def predict(self, X, theta):
        with torch.no_grad():
            return self.pred_t(self.features_t(X), torch.tensor(theta)).numpy()

    def train(self, y, theta_init):
        """T(y): L-BFGS-B from theta_init until the loss stops decreasing.

        A ReLU network's minimizer typically sits on a kink (training points
        pinned at a unit's activation boundary), where no one-sided gradient
        vanishes: max |grad| stalls near 1e-3 while the loss is flat to 8
        digits (diagnosed 2026-11-19). So the stop is scipy's relative loss
        change (ftol, default) or GTOL, never the iteration cap; a fit that
        hits the cap is flagged."""
        y_t = torch.tensor(y)

        def fg(th):
            t = torch.tensor(th, requires_grad=True)
            loss = self.loss_t(t, y_t)
            (g,) = torch.autograd.grad(loss, t)
            return float(loss.detach()), g.numpy().copy()

        t0 = time.time()
        res = minimize(fg, theta_init, jac=True, method="L-BFGS-B",
                       options={"maxiter": MAX_ITER, "maxfun": MAX_FUN, "gtol": GTOL})
        gmax = float(np.abs(fg(res.x)[1]).max())
        info = {"nit": int(res.nit), "gmax": gmax, "ok": bool(res.status == 0),
                "msg": str(res.message), "time": time.time() - t0,
                "pinned": self.pinned(res.x)}
        return res.x, info

    def pinned(self, theta, tol=1e-8):
        """Number of (training row, hidden unit) pre-activations within tol of 0."""
        with torch.no_grad():
            parts = self.unpack(torch.tensor(theta))
            a, k = self.U_t, 0
            for j in range(0, len(parts) - 2, 2):
                z = a @ parts[j] + parts[j + 1]
                k += int((z.abs() < tol).sum())
                a = torch.relu(z)
        return k

    def cold_fit(self, ybar, seed):
        """sklearn's fit (probe_rc_ridge's settings), polished to GTOL: theta0."""
        from sklearn.neural_network import MLPRegressor
        net = MLPRegressor(hidden_layer_sizes=NET_HIDDEN, activation="relu",
                           solver="lbfgs", alpha=self.alpha, max_iter=2000,
                           random_state=seed)
        net.fit(self.U, (ybar - self.mu) / self.sd)
        theta = self.pack(net.coefs_, [b.ravel() for b in net.intercepts_])
        return self.train(ybar, theta)

    def gurobi_out(self, m, u, theta):
        """h(x) for the embedded network, as a linear expression in the MIP."""
        parts = self.unpack(theta)
        Ws = [parts[j] for j in range(0, len(parts), 2)]
        bs = [parts[j + 1] for j in range(0, len(parts), 2)]
        a = _Hidden(self.d, Ws[:-1], bs[:-1]).gurobi(m, u)
        out = float(bs[-1][0]) + gp.quicksum(float(Ws[-1][j, 0]) * a[j]
                                             for j in range(len(a)))
        return self.mu + self.sd * out


class _Hidden(NetFeatures):
    """probe_rc_ridge's big-M ReLU embedding, for given hidden-layer weights."""

    def __init__(self, d, Ws, bs):
        self.d, self.Ws, self.bs = d, Ws, bs


class RidgeModel:
    """Ridge on a fixed feature map (probe_rc_ridge.FeatureRidge), written as a
    generic model so F0(i) runs the SAME implicit-gradient and ascent code."""

    def __init__(self, cr, X):
        self.cr = cr
        self.n, self.p = cr.n, cr.p + 1
        self.Zc_t = torch.tensor(cr.features(X) - cr.phibar)

    def loss_t(self, theta_t, y_t):
        r = y_t - theta_t[0] - self.Zc_t @ theta_t[1:]
        return 0.5 * (r ** 2).sum() + 0.5 * LAM * (theta_t[1:] ** 2).sum()

    def features_t(self, X):
        return torch.tensor(self.cr.features(X) - self.cr.phibar)

    def pred_t(self, Z_t, theta_t):
        return theta_t[0] + Z_t @ theta_t[1:]

    def predict(self, X, theta):
        return self.cr.predict(X, (theta[0], theta[1:]))

    def train(self, y, theta_init=None):
        b0, beta = self.cr.fit(y)
        return np.concatenate([[b0], beta]), {"ok": True, "nit": 0, "gmax": 0.0,
                                              "time": 0.0}


# --------------------------------------------------------------------------
# I12: implicit gradient and projected ascent
# --------------------------------------------------------------------------
class Sensitivity:
    """Everything at one theta that does not depend on lam or x: the Hessian's
    eigendecomposition and the mixed derivative L_theta_y (p x n)."""

    def __init__(self, model, theta, y):
        th = torch.tensor(theta)
        y_t = torch.tensor(y)
        t0 = time.time()
        H = torch.func.hessian(lambda t: model.loss_t(t, y_t))(th)
        gfun = torch.func.grad(lambda t, yy: model.loss_t(t, yy), argnums=0)
        Lty = torch.func.jacrev(gfun, argnums=1)(th, y_t)        # p x n
        self.w, self.Q = np.linalg.eigh(H.numpy())
        self.Lty = Lty.numpy()
        self.trace_p = float(np.trace(H.numpy())) / len(theta)
        self.model, self.theta = model, theta
        self.time = time.time() - t0

    def grad_h(self, X):
        """grad_theta h(x; theta) for each row of X: m x p."""
        th = torch.tensor(self.theta)
        Z = self.model.features_t(X)
        return torch.func.jacrev(lambda t: self.model.pred_t(Z, t))(th).numpy()

    def auto_lam_rel(self):
        """Twice the damping that makes H + lam I positive definite (H is
        indefinite at a network's theta0), and at least 1e-8: F0's choice."""
        return max(1e-8, -2.0 * self.w.min() / self.trace_p)

    def dh_dy(self, X, lam_rel):
        """d h(x; T(y)) / dy for each row of X: n x m, at damping lam_rel
        (None: auto_lam_rel)."""
        if lam_rel is None:
            lam_rel = self.auto_lam_rel()
        lam = lam_rel * self.trace_p
        Gx = self.grad_h(X)                                    # m x p
        V = self.Q @ ((self.Q.T @ Gx.T) / (self.w + lam)[:, None])
        return -self.Lty.T @ V


def ascend(model, x, ybar, sigma, R, theta0, starts, lam_rel, sign=SIGN,
           max_steps=20, halvings=5, min_gain=1e-3, log=None):
    """Monotone projected gradient ascent of sign * h(x; T(ybar + sigma z)) over
    ||z|| <= R. Steps: z' = Proj(z + l d), l = inf (the jump R d), R, R/2, ...;
    accept the first z' whose value beats the current by min_gain * sigma."""
    X = np.atleast_2d(x)
    best = None
    for name, z0 in starts:
        z = np.array(z0, dtype=float)
        if np.linalg.norm(z) == 0:
            theta, info = theta0, {"ok": True}
        else:
            theta, info = model.train(ybar + sigma * z, theta0)
        val = sign * float(model.predict(X, theta)[0])
        path = [{"step": 0, "val": val, "norm_z": float(np.linalg.norm(z)),
                 "ok": info["ok"]}]
        first_theta = theta
        for step in range(1, max_steps + 1):
            g = sign * sigma * Sensitivity(model, theta, ybar + sigma * z).dh_dy(X, lam_rel)[:, 0]
            d = g / np.linalg.norm(g)
            accepted = False
            for ell in [np.inf] + [R / 2 ** j for j in range(halvings + 1)]:
                zn = R * d if np.isinf(ell) else z + ell * d
                nz = np.linalg.norm(zn)
                if nz > R:
                    zn = zn * (R / nz)
                tn, info = model.train(ybar + sigma * zn, theta)
                if not info["ok"]:
                    continue
                vn = sign * float(model.predict(X, tn)[0])
                if vn >= val + min_gain * sigma:
                    z, theta, val, accepted = zn, tn, vn, True
                    if step == 1:
                        first_theta = theta
                    path.append({"step": step, "ell": float(ell), "val": val,
                                 "norm_z": float(np.linalg.norm(z))})
                    break
            if not accepted:
                break
        rec = {"start": name, "val": val, "z": z, "theta": theta, "path": path,
               "first_theta": first_theta}
        if log is not None:
            log(rec)
        if best is None or val > best["val"]:
            best = rec
    return best


# --------------------------------------------------------------------------
# The nominal problem with the end-to-end network embedded (part of I13)
# --------------------------------------------------------------------------
def solve_nominal(inst, model, theta, cost, shift=0.0, cuts=(), tau=0.0):
    """min c'x s.t. h(x; theta) >= 50 + shift and, for each cut model,
    h(x; theta_cut) >= 50 + tau (the reactor's floor, restricted by tau)."""
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = TIME_LIMIT
    if os.environ.get("GRB_THREADS"):
        m.Params.Threads = int(os.environ["GRB_THREADS"])
    d = len(model.lb)
    u = m.addVars(d, lb=-1.0, ub=1.0)
    span = model.ub - model.lb
    x = [model.lb[i] + 0.5 * span[i] * (u[i] + 1.0) for i in range(d)]
    for dc in inst.domain_constraints:
        m.addConstr(gp.quicksum(float(dc.coeffs[i]) * x[i] for i in range(d)) <= dc.rhs)
    m.addConstr(model.gurobi_out(m, u, theta) >= PRODUCT_FLOOR + shift)
    for th in cuts:
        m.addConstr(model.gurobi_out(m, u, th) >= PRODUCT_FLOOR + tau)
    m.setObjective(gp.quicksum(float(cost[i]) * x[i] for i in range(d)), GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        return None, int(m.Status)
    uu = np.clip(np.array([u[i].X for i in range(d)]), -1, 1)
    return model.lb + 0.5 * span * (uu + 1.0), int(m.Status)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_f0_affine(args):
    """F0(i): on ridge with fixed features the machinery must reproduce the RC."""
    inst, md, sigma = setup()
    ybar = noisy_labels(md, sigma, args.seed, args.redraw)
    Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, 2000, seed=0)
    pts = Xs[np.random.default_rng(1).choice(len(Xs), 5, replace=False)]
    out = []
    for name in ("cubic", "net_frozen"):
        cr = build(name, inst, md, sigma, frozen_labels(md, sigma))
        model = RidgeModel(cr, md.X_train)
        theta0, _ = model.train(ybar)
        sens = Sensitivity(model, theta0, ybar)
        G = sens.dh_dy(pts, 0.0)                                # n x 5
        dphi = cr.features(pts) - cr.phibar
        S = 1.0 / cr.n + (dphi @ cr.H).T                         # s(x), n x 5
        rel = float(np.abs(G - S).max() / np.abs(S).max())
        print(f"{name}: max |implicit - s(x)| / max |s(x)| = {rel:.2e} "
              f"(Hessian {sens.time:.1f}s)", flush=True)
        R = args.R
        for j, x in enumerate(pts):
            snorm = float(cr.snorm(x)[0])
            best = ascend(model, x, ybar, sigma, R, theta0, [("zero", np.zeros(cr.n))],
                          lam_rel=0.0)
            h0 = float(model.predict(x[None], theta0)[0])
            drop = h0 - float(model.predict(x[None], best["theta"])[0])
            steps = len(best["path"]) - 1
            row = {"model": name, "point": j, "R_sigma": R, "rel_grad_err": rel,
                   "drop": drop, "rc_margin": R * sigma * snorm,
                   "drop_minus_margin": drop - R * sigma * snorm,
                   "accepted_steps": steps,
                   "first_ell": best["path"][1]["ell"] if steps else None}
            out.append(row)
            print(f"  point {j}: drop {drop:.6f}  R*sigma*||s|| {R*sigma*snorm:.6f}  "
                  f"diff {drop - R*sigma*snorm:.2e}  accepted steps {steps}", flush=True)
    _save(f"f0_affine_r{args.redraw}.json", out)


def stage_f0_grad(args):
    """F0(ii): implicit gradient against finite differences of retraining."""
    inst, md, sigma = setup()
    ybar = noisy_labels(md, sigma, args.seed, args.redraw)
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=args.alpha)
    theta0, info0 = model.cold_fit(ybar, seed=args.redraw)
    print(f"theta0: p={model.p}, nit {info0['nit']}, max|grad| {info0['gmax']:.1e}, "
          f"ok {info0['ok']}, pinned {info0['pinned']}, {info0['time']:.1f}s "
          f"({info0['msg']})", flush=True)
    if not info0["ok"]:
        raise SystemExit("theta0: L-BFGS-B hit its iteration cap")

    cost = cost_vectors("paper", args.redraw + 1, inst)[args.redraw]
    x_nom, st = solve_nominal(inst, model, theta0, cost)
    x_tight, st2 = solve_nominal(inst, model, theta0, cost, shift=args.tighten)
    Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, 20000, seed=0)
    hs = model.predict(Xs, theta0)
    near = Xs[np.abs(hs - PRODUCT_FLOOR) < 1.0]
    rnd = near[np.random.default_rng(2).choice(len(near), 3, replace=False)]
    pts = np.vstack([x_nom, x_tight, rnd])
    labels = ["nominal", f"tightened_{args.tighten:g}", "near_1", "near_2", "near_3"]
    print(f"nominal status {st}, h={model.predict(x_nom[None], theta0)[0]:.3f}, "
          f"ODE {ode_flow(inst, x_nom):.3f}; tightened status {st2}", flush=True)

    sens = Sensitivity(model, theta0, ybar)
    print(f"Hessian + mixed derivative {sens.time:.1f}s; eigenvalues min "
          f"{sens.w.min():.2e}, max {sens.w.max():.2e}, tr/p {sens.trace_p:.2e}, "
          f"# below 1e-10*max: {(sens.w < 1e-10 * sens.w.max()).sum()}", flush=True)
    G = {lr: sens.dh_dy(pts, lr) for lr in LAM_GRID if (sens.w + lr * sens.trace_p).min() > 0}

    rows = np.sort(np.random.default_rng(3).choice(model.n, args.fd_rows, replace=False)) \
        if args.fd_rows < model.n else np.arange(model.n)
    FD = np.zeros((len(rows), len(pts)))
    times, bad = [], 0
    for a, i in enumerate(rows):
        vals = []
        for s in (+1, -1):
            y = ybar.copy()
            y[i] += s * args.h
            th, info = model.train(y, theta0)
            times.append(info["time"])
            bad += not info["ok"]
            vals.append(model.predict(pts, th))
        FD[a] = (vals[0] - vals[1]) / (2 * args.h)
        if (a + 1) % 50 == 0:
            print(f"  FD {a+1}/{len(rows)} rows, {np.mean(times):.2f}s per fit, "
                  f"{bad} fits above tolerance", flush=True)

    res = {"redraw": args.redraw, "h": args.h, "fd_rows": len(rows), "p": model.p,
           "theta0": info0, "fit_time_mean": float(np.mean(times)),
           "fits_above_tol": bad, "eig_min": float(sens.w.min()),
           "eig_max": float(sens.w.max()), "trace_over_p": sens.trace_p,
           "points": labels, "x": pts.tolist(), "lam": []}
    print(f"\nh = {args.h} flow units, {len(rows)} rows; cosine / norm ratio "
          f"(implicit / FD) per point")
    for lr, Gl in G.items():
        Gs = Gl[rows]
        cos = [float(Gs[:, j] @ FD[:, j] / (np.linalg.norm(Gs[:, j]) * np.linalg.norm(FD[:, j])))
               for j in range(len(pts))]
        ratio = [float(np.linalg.norm(Gs[:, j]) / np.linalg.norm(FD[:, j]))
                 for j in range(len(pts))]
        ok = all(c >= 0.99 for c in cos) and all(abs(r - 1) <= 0.05 for r in ratio)
        res["lam"].append({"lam_rel": lr, "cos": cos, "ratio": ratio, "pass": ok})
        print(f"  lam_rel {lr:7.0e}: cos " + " ".join(f"{c:.4f}" for c in cos)
              + " | ratio " + " ".join(f"{r:.3f}" for r in ratio) + f"  {'PASS' if ok else ''}")
    res["fd_norm"] = [float(np.linalg.norm(FD[:, j])) for j in range(len(pts))]
    _save(f"f0_grad_r{args.redraw}_h{args.h:g}_n{len(rows)}.json", res)
    np.savez(os.path.join(OUT_DIR, f"f0_grad_r{args.redraw}_h{args.h:g}_n{len(rows)}.npz"),
             rows=rows, FD=FD, theta0=theta0,
             **{f"G_{lr:g}": Gl[rows] for lr, Gl in G.items()})


def stage_f0_dir(args):
    """F0(ii), directional form. Per-row finite differences are below the
    training map's own noise (one label of 1,000 moves h less than retraining
    on the SAME labels does), so test the implicit gradient where the ascent
    uses it: retrain along the direction that should lower h(x*) fastest, at
    whitened steps t (Delta y = sigma t d), against -t sigma ||g||, and along
    random directions at t = R."""
    inst, md, sigma = setup()
    ybar = noisy_labels(md, sigma, args.seed, args.redraw)
    cost = cost_vectors("paper", args.redraw + 1, inst)[args.redraw]
    cr = build("net_frozen", inst, md, sigma, frozen_labels(md, sigma))
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=args.alpha)
    theta0, info0 = model.cold_fit(ybar, seed=args.redraw)
    x, st = solve_nominal(inst, model, theta0, cost)
    X = x[None]
    h0 = float(model.predict(X, theta0)[0])
    tz, _ = model.train(ybar, theta0)
    sens = Sensitivity(model, theta0, ybar)
    lam_rel = max(1e-8, -2.0 * sens.w.min() / sens.trace_p)   # smallest that makes H + lam I PD, x2
    g = sens.dh_dy(X, lam_rel)[:, 0]
    d = -g / np.linalg.norm(g)
    res = {"redraw": args.redraw, "alpha": args.alpha, "theta0": info0, "x": x.tolist(),
           "h0": h0, "ode": ode_flow(inst, x), "drift": float(model.predict(X, tz)[0]) - h0,
           "eig_min": float(sens.w.min()), "lam_rel": lam_rel,
           "sigma_gnorm": float(sigma * np.linalg.norm(g)),
           "frozen_rc_margin_R": float(args.R * sigma * cr.snorm(x)[0]),
           "along": [], "random": []}
    print(f"alpha {args.alpha}: h(x*) {h0:.3f}, ODE {res['ode']:.3f}, drift {res['drift']:+.4f}, "
          f"eig min {res['eig_min']:.1e}, sigma||g|| {res['sigma_gnorm']:.4f}, "
          f"frozen-net RC margin {res['frozen_rc_margin_R']:.4f}", flush=True)
    for t in [0.5, 1.0, 2.0, args.R]:
        th, inf = model.train(ybar + sigma * t * d, theta0)
        dh = float(model.predict(X, th)[0]) - h0
        res["along"].append({"t": t, "dh": dh, "linear": -t * res["sigma_gnorm"], "ok": inf["ok"]})
        print(f"  along -g, t={t}: dh {dh:+.4f}, linear {-t * res['sigma_gnorm']:+.4f}", flush=True)
    rng = np.random.default_rng(5)
    for _ in range(6):
        v = rng.standard_normal(model.n)
        th, _ = model.train(ybar + sigma * args.R * v / np.linalg.norm(v), theta0)
        res["random"].append(float(model.predict(X, th)[0]) - h0)
    print(f"  random directions, t={args.R}: " + " ".join(f"{v:+.4f}" for v in res["random"]))
    _save(f"f0_dir_r{args.redraw}_a{args.alpha:g}.json", res)


def chosen_alpha():
    path = os.path.join(OUT_DIR, "cv_alpha.json")
    if not os.path.exists(path):
        raise SystemExit("run --stage cv_alpha first (the network's alpha is chosen by CV)")
    return float(json.load(open(path))["chosen"])


def stage_f1(args):
    """F1: does ascent beyond the first step find worse relabelings?

    End-to-end network at the CV-chosen alpha, one redraw, at the nominal
    decision x0 and at one robust iterate x1 (the master with the cut from
    x0's separation). At each point, the drop of h(x) in sigma units under:
      linearized  one step from zero, z = R d (d from the implicit gradient)
      ascent      I12 from zero + 4 random starts on the sphere
      bank        the worst of B random relabelings on the sphere ||z|| = R
      true        retraining on the clean labels, Delta y = -e (outside the
                  ball: ||e||/sigma ~ sqrt(n)), the noise's actual effect
      cold        sklearn cold fits with other seeds on the same labels:
                  training randomness, against which the rest is read
    plus d_eff, the numerical rank of the linearized sensitivities over the
    domain. Resumable per point."""
    inst, md, sigma = setup()
    alpha = args.alpha if args.alpha_set else chosen_alpha()
    torch.set_num_threads(args.threads)
    k = args.redraw
    ybar = noisy_labels(md, sigma, args.seed, k)
    cost = cost_vectors("paper", k + 1, inst)[k]
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=alpha)
    theta0, info0 = model.cold_fit(ybar, seed=k)
    tag = f"f1_r{k}_a{alpha:g}"
    path = os.path.join(OUT_DIR, tag + ".json")
    res = json.load(open(path)) if os.path.exists(path) else {"points": []}
    res.update(redraw=k, alpha=alpha, R=args.R, B=args.bank, theta0=info0)
    R, tau = args.R, 0.05 * sigma
    rng = np.random.default_rng([args.seed, k, 7])

    def drop(x, th):
        return (float(model.predict(x[None], theta0)[0])
                - float(model.predict(x[None], th)[0])) / sigma

    x0, st0 = solve_nominal(inst, model, theta0, cost)
    xs, best0 = [("x0", x0)], None
    done = {p["name"] for p in res["points"]}
    for name in ("x0", "x1"):
        if name == "x1":
            if best0 is None:      # x0 came from the file: redo its separation from zero
                best0 = ascend(model, x0, ybar, sigma, R, theta0,
                               [("zero", np.zeros(model.n))], args.lam_rel)
                res["x1_cut_from"] = "zero start only (resumed run)"
            x1, st1 = solve_nominal(inst, model, theta0, cost, cuts=[best0["theta"]], tau=tau)
            if x1 is None:
                res["x1_status"] = st1
                break
            xs.append(("x1", x1))
        x = dict(xs)[name]
        if name in done:
            continue
        t0 = time.time()
        p = {"name": name, "x": x.tolist(), "h": float(model.predict(x[None], theta0)[0]),
             "ode": ode_flow(inst, x)}
        log = []
        starts = [("zero", np.zeros(model.n))]
        for j in range(args.n_random):
            v = rng.standard_normal(model.n)
            starts.append((f"random{j}", R * v / np.linalg.norm(v)))
        best = ascend(model, x, ybar, sigma, R, theta0, starts, args.lam_rel, max_steps=args.max_steps,
                      log=lambda r: log.append(r))
        zero = next(r for r in log if r["start"] == "zero")
        # from zero the first accepted step is always the jump z = R d0
        p["linearized"] = drop(x, zero["first_theta"])
        p["ascent"] = drop(x, best["theta"])
        p["starts"] = [{"start": r["start"], "drop": drop(x, r["theta"]),
                        "steps": len(r["path"]) - 1} for r in log]
        if name == "x0":
            best0 = best
        bank = []
        for _ in range(args.bank):
            v = rng.standard_normal(model.n)
            th, inf = model.train(ybar + sigma * R * v / np.linalg.norm(v), theta0)
            if inf["ok"]:
                bank.append(drop(x, th))
        p["bank_max"] = float(np.max(bank))
        p["bank_q"] = np.quantile(bank, [0.5, 0.9, 0.99]).tolist()
        th_true, _ = model.train(md.y_true, theta0)
        p["true"] = drop(x, th_true)
        p["true_norm_sigma"] = float(np.linalg.norm(md.y_true - ybar) / sigma)
        p["cold"] = []
        for s in range(1, args.cold + 1):
            th_c, _ = model.cold_fit(ybar, seed=k + 1000 * s)
            p["cold"].append(drop(x, th_c))
        p["time_s"] = time.time() - t0
        res["points"].append(p)
        print(f"{tag} {name}: h {p['h']:.2f} ODE {p['ode']:.2f} | drop/sigma: linearized "
              f"{p['linearized']:.3f}, ascent {p['ascent']:.3f}, bank max {p['bank_max']:.3f}, "
              f"true {p['true']:.3f}, cold {np.round(p['cold'], 3)} ({p['time_s']/60:.0f} min)",
              flush=True)
        _save(tag + ".json", res)
    if "d_eff" not in res:
        Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, 5000, seed=0)[:500]
        S = Sensitivity(model, theta0, ybar).dh_dy(Xs, args.lam_rel)     # n x 500
        sv = np.linalg.svd(S, compute_uv=False)
        res["d_eff_rank"] = int((sv > 1e-6 * sv[0]).sum())
        res["d_eff"] = float(sv.sum() ** 2 / (sv ** 2).sum())   # participation ratio
        res["d_eff_90"] = int(np.searchsorted(np.cumsum(sv ** 2) / (sv ** 2).sum(), 0.9) + 1)
        print(f"{tag}: d_eff rank {res['d_eff_rank']}, participation {res['d_eff']:.1f}, "
              f"90% energy {res['d_eff_90']}", flush=True)
    _save(tag + ".json", res)


def stage_f0_loop(args):
    """F0(iii): the cutting-plane loop on ridge with fixed features. Separation
    is exact in one step there (F0(i)) and each cut is linear in phi(x), so as
    tau -> 0 the loop's decision and objective must converge to the RC's own
    cone solution at the same radius. Master: the nominal row h(x) >= 50 and
    every cut h(x; theta_cut) >= 50 + tau (restriction); separation tests
    against 50; cuts accumulate; at most 100 master solves."""
    from probe_rc_ridge import base_model, solve_rc, _x_of
    inst, md, sigma = setup()
    k = args.redraw
    ybar = noisy_labels(md, sigma, args.seed, k)
    cost = cost_vectors("paper", k + 1, inst)[k]
    R = args.R
    out = []
    for name in ("net_frozen", "cubic"):
        cr = build(name, inst, md, sigma, frozen_labels(md, sigma))
        model = RidgeModel(cr, md.X_train)
        theta0, _ = model.train(ybar)
        rc = solve_rc(cr, inst, (theta0[0], theta0[1:]), R * sigma, cost)
        print(f"{name}: RC at R={R} sigma: obj {rc['obj']:.5f} status {rc['status']}", flush=True)
        for tau_s in [0.5, 0.2, 0.05, 0.01, 0.001]:
            tau = tau_s * sigma
            cuts, it, t0 = [], 0, time.time()
            while it < 100:
                it += 1
                m, u, xv, dphi, _ = base_model(cr, inst)
                lin = lambda th: th[0] + gp.quicksum(float(th[1 + j]) * dphi[j] for j in range(cr.p))
                m.addConstr(lin(theta0) >= PRODUCT_FLOOR)
                for th in cuts:
                    m.addConstr(lin(th) >= PRODUCT_FLOOR + tau)
                m.setObjective(gp.quicksum(float(cost[i]) * xv[i] for i in range(cr.d)), GRB.MINIMIZE)
                m.optimize()
                if m.SolCount == 0:
                    break
                x = _x_of(cr, u)
                best = ascend(model, x, ybar, sigma, R, theta0, [("zero", np.zeros(cr.n))],
                              lam_rel=0.0)
                worst = float(model.predict(x[None], best["theta"])[0])
                if worst >= PRODUCT_FLOOR - 1e-9:
                    break
                cuts.append(best["theta"])
            obj = float(cost @ x)
            rc_slack = float(model.predict(x[None], theta0)[0] - R * sigma * cr.snorm(x)[0]) - PRODUCT_FLOOR
            row = {"model": name, "redraw": k, "R_sigma": R, "tau_sigma": tau_s, "iters": it,
                   "cuts": len(cuts), "obj": obj, "obj_rc": rc["obj"],
                   "rel_obj_gap": (obj - rc["obj"]) / abs(rc["obj"]),
                   "x_dist": float(np.linalg.norm((x - rc["x"]) / (inst.variable_ub - inst.variable_lb))),
                   "rc_slack_at_x": rc_slack, "time_s": time.time() - t0}
            out.append(row)
            print(f"  tau {tau_s:g} sigma: {it} solves, {len(cuts)} cuts, obj {obj:.5f} "
                  f"(gap {row['rel_obj_gap']:+.2e}), |x - x_RC| {row['x_dist']:.2e}, "
                  f"RC slack at x {rc_slack:+.4f}", flush=True)
    _save(f"f0_loop_r{k}.json", out)


TRUTH_NET = os.path.join(OUT_DIR, "ss1_truth_net.npz")
TRUTH_SEED = 12345


class TruthNet:
    """SS1 arm (ii)'s truth: a (32, 32) network fitted once to the noise-free
    ODE labels, at the model's alpha, so the model class contains the truth."""

    def __init__(self, inst, md):
        z = np.load(TRUTH_NET)
        self.model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, md.y_true,
                              alpha=float(z["alpha"]))
        self.model.mu, self.model.sd = float(z["mu"]), float(z["sd"])
        self.theta = z["theta"]

    def __call__(self, X):
        return self.model.predict(np.atleast_2d(X), self.theta)


def stage_truth_net(args):
    """I14 (SS1 arm ii): fit and save the truth network."""
    inst, md, sigma = setup()
    alpha = chosen_alpha()
    m = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, md.y_true, alpha=alpha)
    theta, info = m.cold_fit(md.y_true, seed=TRUTH_SEED)
    pred = m.predict(md.X_train, theta)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(TRUTH_NET, theta=theta, mu=m.mu, sd=m.sd, alpha=alpha, seed=TRUTH_SEED)
    Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, 2000, seed=0)
    ode = np.array([ode_flow(inst, x) for x in Xs[:300]])
    print(f"truth net (alpha {alpha:g}): train RMSE vs ODE {np.sqrt(np.mean((pred - md.y_true)**2)):.3f}, "
          f"domain RMSE vs ODE (300 pts) {np.sqrt(np.mean((m.predict(Xs[:300], theta) - ode)**2)):.3f}, "
          f"{info['msg']}", flush=True)


def arm_truth(arm, inst, md):
    """(clean training labels y*, truth function x -> flow) for SS1's arm."""
    if arm == "ode":
        return md.y_true, lambda x: ode_flow(inst, x)
    t = TruthNet(inst, md)
    return t(md.X_train), lambda x: float(t(x)[0])


def fits_for(model, arm, k, ybar, ystar):
    """theta0 = cold fit on ybar, the clean-label reference T(y*) from theta0 and a
    cold-start clean fit, computed once per (arm, redraw) and stored. The cold fit
    is not reproducible across cluster nodes (BLAS kernels; chaotic L-BFGS-B
    paths), so a restarted task, or a later stage on the same redraw, must reload
    them rather than refit: job 30384811 refit theta0 after preemption and 9
    runs mixed two nominal networks."""
    path = os.path.join(OUT_DIR, f"fits_{arm}_r{k}_a{model.alpha:g}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["theta0"], z["th_clean"], z["th_cold"]
    import socket
    theta0, _ = model.cold_fit(ybar, seed=k)
    th_clean, _ = model.train(ystar, theta0)            # the guarantee's reference
    th_cold, _ = model.cold_fit(ystar, seed=k + 500)    # beside it: a cold-start clean fit
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(path, theta0=theta0, th_clean=th_clean, th_cold=th_cold,
             host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID", "local"))
    return theta0, th_clean, th_cold


def stage_radius_c(args):
    """Radius (c) by ENUMERATION (a lower bound on each draw's maximum).

    For N noise draws e' ~ N(0, sigma^2 I): retrain on ybar + e' from theta0 and
    W = max over a domain sample of [h(x; T(ybar + e')) - h(x; theta0)] / ||s(x)||,
    s(x) = sigma * d h(x) / dy at theta0 (whitened linearized sensitivity). R is
    the ceil((N+1)(1-eps))-th smallest W. The sign is "noisy minus clean": the
    direction in which an RC-feasible decision violates the clean-label model of
    a lower-bound constraint. Resumable per draw; the Gurobi bound on a few
    draws (enumeration gap) is a separate stage."""
    inst, md, sigma = setup()
    alpha = chosen_alpha()
    torch.set_num_threads(args.threads)
    k = args.redraw
    ystar, _ = arm_truth(args.arm, inst, md)
    ybar = ystar + sigma * np.random.default_rng([args.seed, k]).standard_normal(len(ystar))
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=alpha)
    theta0, _, _ = fits_for(model, args.arm, k, ybar, ystar)
    Xs = domain_sample(inst.variable_lb, inst.variable_ub, inst, args.n_sample, seed=0)
    sens = Sensitivity(model, theta0, ybar)
    snorm = np.concatenate([sigma * np.linalg.norm(sens.dh_dy(Xs[i:i + 2000], args.lam_rel), axis=0)
                            for i in range(0, len(Xs), 2000)])
    h0 = model.predict(Xs, theta0)
    tag = f"radius_c_{args.arm}_r{k}_a{alpha:g}"
    path = os.path.join(OUT_DIR, tag + ".json")
    res = json.load(open(path)) if os.path.exists(path) else {"W": [], "argmax": [], "fit": []}
    res.update(arm=args.arm, redraw=k, alpha=alpha, n_sample=len(Xs), eps=args.eps,
               snorm_median=float(np.median(snorm)), snorm_max=float(snorm.max()),
               lam_rel=sens.auto_lam_rel() if args.lam_rel is None else args.lam_rel)
    t0 = time.time()
    for i in range(len(res["W"]), args.n_draws):
        e = sigma * np.random.default_rng([args.seed, k, 11, i]).standard_normal(model.n)
        th, info = model.train(ybar + e, theta0)
        ratio = (model.predict(Xs, th) - h0) / snorm
        j = int(np.argmax(ratio))
        res["W"].append(float(ratio[j]))
        res["argmax"].append(Xs[j].tolist())
        res["fit"].append({"nit": info["nit"], "ok": info["ok"], "time": info["time"]})
        if (i + 1) % 10 == 0 or i + 1 == args.n_draws:
            _save(tag + ".json", res)
            print(f"{tag}: {i+1}/{args.n_draws} draws, {(time.time()-t0)/60:.1f} min", flush=True)
    W = np.sort(res["W"])
    kk = int(np.ceil((len(W) + 1) * (1 - args.eps)))
    res["R"] = float(W[min(kk, len(W)) - 1])
    _save(tag + ".json", res)
    print(f"{tag}: R = {res['R']:.3f} sigma from {len(W)} draws (median W {np.median(W):.3f}); "
          f"||s|| median {res['snorm_median']:.3f}", flush=True)


def stage_radius_refine(args):
    """How much does enumeration understate radius (c)? For the draws ranked
    highest by enumeration (the order statistic R is their 20th largest at
    N = 200, eps = 0.1), retrain on the same noise and maximize the ratio over
    x by local search (SLSQP on the scaled design with the domain constraints,
    from the 10 best enumerated points), then recompute R. A better LOWER
    bound, not a certificate: it measures the gap enumeration leaves."""
    from scipy.optimize import minimize as sp_min
    inst, md, sigma = setup()
    alpha = chosen_alpha()
    torch.set_num_threads(args.threads)
    k = args.redraw
    tag = f"radius_c_{args.arm}_r{k}_a{alpha:g}"
    rad = json.load(open(os.path.join(OUT_DIR, tag + ".json")))
    ystar, _ = arm_truth(args.arm, inst, md)
    ybar = ystar + sigma * np.random.default_rng([args.seed, k]).standard_normal(len(ystar))
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=alpha)
    theta0, _, _ = fits_for(model, args.arm, k, ybar, ystar)
    sens = Sensitivity(model, theta0, ybar)
    lam = (sens.auto_lam_rel() if args.lam_rel is None else args.lam_rel) * sens.trace_p
    # whitened sensitivity s(x) = sigma * A grad_theta h(x), A = -L_ty' (H + lam I)^-1
    A = -sigma * (sens.Lty.T @ sens.Q) / (sens.w + lam)[None, :] @ sens.Q.T     # n x p
    lb, ub = inst.variable_lb, inst.variable_ub
    th0_t = torch.tensor(theta0)
    cons = [{"type": "ineq", "fun": (lambda u, dc=dc: dc.rhs - dc.coeffs @ (lb + 0.5 * (ub - lb) * (u + 1)))}
            for dc in inst.domain_constraints]
    Xs = domain_sample(lb, ub, inst, rad["n_sample"], seed=0)
    h0 = model.predict(Xs, theta0)
    snorm_all = np.concatenate([sigma * np.linalg.norm(sens.dh_dy(Xs[i:i + 2000], args.lam_rel), axis=0)
                                for i in range(0, len(Xs), 2000)])

    def ratio_fn(theta_e):
        the_t = torch.tensor(theta_e)

        def f(u):
            x = lb + 0.5 * (ub - lb) * (np.clip(u, -1, 1) + 1)
            Z = model.features_t(x[None])
            num = float(model.pred_t(Z, the_t)[0] - model.pred_t(Z, th0_t)[0])
            g = torch.func.jacrev(lambda t: model.pred_t(Z, t))(th0_t).numpy()[0]
            return -num / np.linalg.norm(A @ g)
        return f

    W = np.array(rad["W"])
    top = np.argsort(-W)[:args.refine_top]
    refined = W.copy()
    rows = []
    for i in top:
        e = sigma * np.random.default_rng([args.seed, k, 11, int(i)]).standard_normal(model.n)
        th, _ = model.train(ybar + e, theta0)
        r_all = (model.predict(Xs, th) - h0) / snorm_all
        f = ratio_fn(th)
        best = -np.inf
        for j in np.argsort(-r_all)[:10]:
            u0 = scale(Xs[j], lb, ub)[0]
            out = sp_min(f, u0, method="SLSQP", bounds=[(-1, 1)] * len(lb), constraints=cons,
                         options={"maxiter": 200, "ftol": 1e-10})
            best = max(best, -out.fun if out.success or np.isfinite(out.fun) else -np.inf)
        refined[i] = max(W[i], best)
        rows.append({"draw": int(i), "W_enum": float(W[i]), "W_refined": float(refined[i]),
                     "W_enum_recomputed": float(r_all.max())})
        print(f"{tag} draw {i}: enum {W[i]:.4f} (recomputed {r_all.max():.4f}) -> refined {refined[i]:.4f}",
              flush=True)
    kk = int(np.ceil((len(W) + 1) * (1 - args.eps)))
    R_enum, R_ref = float(np.sort(W)[kk - 1]), float(np.sort(refined)[kk - 1])
    out = {"arm": args.arm, "redraw": k, "R_enum": R_enum, "R_refined": R_ref,
           "gap": R_ref - R_enum, "top": args.refine_top, "rows": rows}
    _save(f"radius_refine_{args.arm}_r{k}_a{alpha:g}.json", out)
    print(f"{tag}: R enum {R_enum:.4f} -> refined {R_ref:.4f} (gap {R_ref - R_enum:+.4f} sigma)", flush=True)


def stage_timing(args):
    """Fixed separation workload (arm ode, redraw 0): one Sensitivity (Hessian
    + mixed derivative), 10 retrains at the radius, one 3-step ascent from zero.
    Run at the thread count given by --threads; prints wall seconds per piece."""
    import socket
    inst, md, sigma = setup()
    torch.set_num_threads(args.threads)
    ystar, _ = arm_truth("ode", inst, md)
    ybar = ystar + sigma * np.random.default_rng([0, 0]).standard_normal(len(ystar))
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=chosen_alpha())
    theta0, _ = model.cold_fit(ybar, seed=0)
    x, _ = solve_nominal(inst, model, theta0, cost_vectors("paper", 1, inst)[0])
    t0 = time.time(); Sensitivity(model, theta0, ybar); t_sens = time.time() - t0
    t0 = time.time()
    for j in range(10):
        g = np.random.default_rng([5, j]).standard_normal(model.n)
        model.train(ybar + sigma * 4.65 * g / np.linalg.norm(g), theta0)
    t_ret = (time.time() - t0) / 10
    t0 = time.time()
    ascend(model, x, ybar, sigma, 4.65, theta0, [("zero", np.zeros(model.n))], None, max_steps=3)
    t_asc = time.time() - t0
    print(f"TIMING host {socket.gethostname()} threads {args.threads} torch {torch.get_num_threads()} "
          f"OMP {os.environ.get('OMP_NUM_THREADS')}: sensitivity {t_sens:.1f}s, retrain {t_ret:.2f}s, "
          f"ascent(3 steps) {t_asc:.1f}s", flush=True)


def radius_for(arm, k, alpha):
    path = os.path.join(OUT_DIR, f"radius_c_{arm}_r{k}_a{alpha:g}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no radius for arm {arm} redraw {k}: run --stage radius_c first")
    r = json.load(open(path))
    if "R" not in r:
        raise SystemExit(f"{path} is incomplete ({len(r['W'])} draws)")
    return float(r["R"])


def stage_f2(args):
    """F2 (pilot): the FO-CP loop on SS1 with the end-to-end network.

    Per (arm, redraw): radius (c) from radius_c (per redraw, enumeration),
    tau = 0.05 sigma, cuts enforced at 50 + tau and separation against 50
    (restriction of the right-hand side), cuts cumulative, at most 20 master
    solves, then the a posteriori attack (zero + 20 random starts, 50 steps,
    step down to R/256). Two separations: "multi" (zero, warm start from the
    tightest cut's relabeling, 4 random starts) and "one" (the linearized step
    alone). The final decision and the nominal one are judged against
    the clean-label model T(y*) from theta0 (the guarantee's reference), a
    cold-start clean fit (training randomness), and the arm's truth.
    Checkpointed per iteration (cuts saved), for mit_preemptable --requeue."""
    inst, md, sigma = setup()
    alpha = chosen_alpha()
    torch.set_num_threads(args.threads)
    k, arm = args.redraw, args.arm
    if not args.R_set:
        rpath = os.path.join(OUT_DIR, f"radius_c_{arm}_r{k}_a{alpha:g}.json")
        if not os.path.exists(rpath) or "R" not in json.load(open(rpath)):
            stage_radius_c(args)     # resumable; N = --n-draws, enumeration
    R = args.R if args.R_set else radius_for(arm, k, alpha)
    tau = 0.05 * sigma
    ystar, truth = arm_truth(arm, inst, md)
    ybar = ystar + sigma * np.random.default_rng([args.seed, k]).standard_normal(len(ystar))
    cost = cost_vectors("paper", k + 1, inst)[k]
    obj_range = float(np.abs(cost) @ (inst.variable_ub - inst.variable_lb))
    model = MLPModel(md.X_train, inst.variable_lb, inst.variable_ub, ybar, alpha=alpha)
    theta0, th_clean, th_cold = fits_for(model, arm, k, ybar, ystar)
    tag = f"f2_{arm}_r{k}_a{alpha:g}"
    path = os.path.join(OUT_DIR, tag + ".json")
    res = json.load(open(path)) if os.path.exists(path) else {"variants": {}}
    res.update(arm=arm, redraw=k, alpha=alpha, R=R, tau=tau, obj_range=obj_range)

    def judge(x):
        return {"x": x.tolist(), "obj": float(cost @ x),
                "h": float(model.predict(x[None], theta0)[0]),
                "clean": float(model.predict(x[None], th_clean)[0]),
                "clean_cold": float(model.predict(x[None], th_cold)[0]),
                "truth": float(truth(x))}

    x_nom, st = solve_nominal(inst, model, theta0, cost)
    res["nominal"] = judge(x_nom)
    rng = np.random.default_rng([args.seed, k, 13])
    for variant in ("one", "multi"):
        v = res["variants"].get(variant, {"iters": [], "done": False})
        if v["done"]:
            continue
        ck = os.path.join(OUT_DIR, f"{tag}_{variant}_cuts.npz")
        cuts, zs = [], []
        if os.path.exists(ck) and v["iters"]:
            z_ = np.load(ck)
            cuts, zs = list(z_["cuts"]), list(z_["zs"])
        x = None
        if len(v["iters"]) >= args.max_iter and not v["iters"][-1].get("infeasible"):
            # preempted after the last master solve but before the attack: re-solve
            # that master with the cuts it had (the checkpoint holds one more)
            x, _ = solve_nominal(inst, model, theta0, cost, cuts=cuts[:v["iters"][-1]["n_cuts"]], tau=tau)
        while len(v["iters"]) < args.max_iter:
            t0 = time.time()
            x, st = solve_nominal(inst, model, theta0, cost, cuts=cuts, tau=tau)
            t_master = time.time() - t0
            if x is None:
                v["iters"].append({"status": st, "infeasible": True})
                break
            starts = [("zero", np.zeros(model.n))]
            if variant == "multi":
                if cuts:
                    hs = [float(model.predict(x[None], th)[0]) for th in cuts]
                    starts.append(("warm", zs[int(np.argmin(hs))]))
                for j in range(4):
                    g = rng.standard_normal(model.n)
                    starts.append((f"random{j}", R * g / np.linalg.norm(g)))
            t1 = time.time()
            best = ascend(model, x, ybar, sigma, R, theta0, starts, args.lam_rel,
                          max_steps=1 if variant == "one" else 20)
            worst = float(model.predict(x[None], best["theta"])[0])
            it = {"master_s": t_master, "sep_s": time.time() - t1, "obj": float(cost @ x),
                  "h": float(model.predict(x[None], theta0)[0]), "worst": worst,
                  "best_start": best["start"], "n_cuts": len(cuts), "master_status": st}
            v["iters"].append(it)
            print(f"{tag} {variant} iter {len(v['iters'])}: obj {it['obj']:.4f} h {it['h']:.3f} "
                  f"worst {worst:.3f} (master {t_master:.0f}s, sep {it['sep_s']:.0f}s)", flush=True)
            if worst >= PRODUCT_FLOOR:
                v["converged"] = True
                break
            cuts.append(best["theta"])
            zs.append(best["z"])
            np.savez(ck, cuts=np.array(cuts), zs=np.array(zs))
            res["variants"][variant] = v
            _save(tag + ".json", res)
        v.setdefault("converged", False)
        if x is not None:
            v["final"] = judge(x)
            t2 = time.time()
            starts = [("zero", np.zeros(model.n))]
            for j in range(args.attack_starts):
                g = rng.standard_normal(model.n)
                starts.append((f"random{j}", R * g / np.linalg.norm(g)))
            att = ascend(model, x, ybar, sigma, R, theta0, starts, args.lam_rel,
                         max_steps=args.attack_steps, halvings=8)
            a_worst = float(model.predict(x[None], att["theta"])[0])
            last = v["iters"][-1]["worst"] if v["iters"] else np.nan
            v["attack"] = {"worst": a_worst, "violation_gap": (PRODUCT_FLOOR - a_worst) / sigma,
                           "separation_gap": (last - a_worst) / sigma, "time_s": time.time() - t2}
            print(f"{tag} {variant}: final obj {v['final']['obj']:.4f}, clean {v['final']['clean']:.3f}, "
                  f"truth {v['final']['truth']:.3f}; attack worst {a_worst:.3f} "
                  f"({v['attack']['time_s']/60:.0f} min)", flush=True)
        v["done"] = True
        res["variants"][variant] = v
        _save(tag + ".json", res)


CV_ALPHAS = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def stage_cv_alpha(args):
    """Choose the end-to-end network's L2 penalty once, before any robust run:
    5-fold CV on the dataset's own noisy labels (md.y_train, not a redraw),
    each fit by the training map's procedure (sklearn, then L-BFGS-B to loss
    convergence), alpha = argmin held-out MSE. The error against the clean
    labels is recorded as a diagnostic only; it plays no part in the choice."""
    inst, md, sigma = setup()
    n = len(md.y_train)
    folds = np.array_split(np.random.default_rng(2026).permutation(n), 5)
    rows = []
    for a in CV_ALPHAS:
        mse, clean, nits = [], [], []
        for f, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), te)
            m = MLPModel(md.X_train[tr], inst.variable_lb, inst.variable_ub,
                         md.y_train[tr], alpha=a)
            th, info = m.cold_fit(md.y_train[tr], seed=f)
            pred = m.predict(md.X_train[te], th)
            mse.append(float(np.mean((pred - md.y_train[te]) ** 2)))
            clean.append(float(np.mean((pred - md.y_true[te]) ** 2)))
            nits.append(info["nit"])
        rows.append({"alpha": a, "cv_mse": float(np.mean(mse)),
                     "cv_se": float(np.std(mse, ddof=1) / np.sqrt(len(mse))),
                     "clean_mse_diag": float(np.mean(clean)), "fold_mse": mse,
                     "mean_nit": float(np.mean(nits))})
        print(f"alpha {a:7g}: CV MSE {rows[-1]['cv_mse']:.3f} (se {rows[-1]['cv_se']:.3f})  "
              f"[diag: MSE vs clean {rows[-1]['clean_mse_diag']:.3f}]", flush=True)
    best = min(rows, key=lambda r: r["cv_mse"])
    print(f"chosen alpha = {best['alpha']:g} (noise variance sigma^2 = {sigma**2:g})")
    _save("cv_alpha.json", {"rows": rows, "chosen": best["alpha"], "folds": 5,
                            "labels": "md.y_train", "rule": "argmin CV MSE"})


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["f0_affine", "f0_grad", "f0_dir", "cv_alpha", "f1", "f0_loop",
                             "truth_net", "radius_c", "f2", "radius_refine", "timing"])
    ap.add_argument("--alpha", type=float, default=None,
                    help="network L2 penalty (f1 default: the CV choice; else NET_ALPHA)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads (f1)")
    ap.add_argument("--n-random", type=int, default=4, help="f1: random ascent starts")
    ap.add_argument("--bank", type=int, default=200, help="f1: random-bank size B")
    ap.add_argument("--cold", type=int, default=5, help="f1: cold restarts")
    ap.add_argument("--max-steps", type=int, default=20, help="f1: accepted steps per start")
    ap.add_argument("--arm", choices=["ode", "net"], default="ode", help="SS1 arm: truth = ODE or truth net")
    ap.add_argument("--n-draws", type=int, default=200, help="radius_c: noise draws N")
    ap.add_argument("--n-sample", type=int, default=20000, help="radius_c: domain points")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--refine-top", type=int, default=40, help="radius_refine: draws refined")
    ap.add_argument("--max-iter", type=int, default=20, help="f2: master solves")
    ap.add_argument("--attack-starts", type=int, default=20, help="f2: random starts of the a posteriori attack")
    ap.add_argument("--attack-steps", type=int, default=50, help="f2: steps per attack start")
    ap.add_argument("--lam-rel", type=float, default=None,
                    help="damping x tr(H)/p; default: auto (2x the smallest PD value)")
    ap.add_argument("--redraw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=float, default=None, help="radius, sigma units (f2 default: radius_c)")
    ap.add_argument("--h", type=float, default=0.5, help="FD step, flow units")
    ap.add_argument("--fd-rows", type=int, default=1000)
    ap.add_argument("--tighten", type=float, default=2.0,
                    help="flow units added to the floor for the tightened point")
    args = ap.parse_args()
    args.alpha_set = args.alpha is not None
    args.R_set = args.R is not None
    if args.R is None:
        args.R = 3.59
    if args.alpha is None:
        args.alpha = NET_ALPHA
    {"f0_affine": stage_f0_affine, "f0_grad": stage_f0_grad,
     "f0_dir": stage_f0_dir, "cv_alpha": stage_cv_alpha, "f1": stage_f1, "f0_loop": stage_f0_loop,
     "truth_net": stage_truth_net, "radius_c": stage_radius_c, "f2": stage_f2,
     "radius_refine": stage_radius_refine, "timing": stage_timing}[args.stage](args)

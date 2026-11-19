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

    def dh_dy(self, X, lam_rel):
        """d h(x; T(y)) / dy for each row of X: n x m, at damping lam_rel."""
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
                    path.append({"step": step, "ell": float(ell), "val": val,
                                 "norm_z": float(np.linalg.norm(z))})
                    break
            if not accepted:
                break
        rec = {"start": name, "val": val, "z": z, "theta": theta, "path": path}
        if log is not None:
            log(rec)
        if best is None or val > best["val"]:
            best = rec
    return best


# --------------------------------------------------------------------------
# The nominal problem with the end-to-end network embedded (part of I13)
# --------------------------------------------------------------------------
def solve_nominal(inst, model, theta, cost, shift=0.0):
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


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["f0_affine", "f0_grad", "f0_dir"])
    ap.add_argument("--alpha", type=float, default=NET_ALPHA, help="network L2 penalty")
    ap.add_argument("--redraw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=float, default=3.59, help="radius, sigma units")
    ap.add_argument("--h", type=float, default=0.5, help="FD step, flow units")
    ap.add_argument("--fd-rows", type=int, default=1000)
    ap.add_argument("--tighten", type=float, default=2.0,
                    help="flow units added to the floor for the tightened point")
    args = ap.parse_args()
    {"f0_affine": stage_f0_affine, "f0_grad": stage_f0_grad,
     "f0_dir": stage_f0_dir}[args.stage](args)

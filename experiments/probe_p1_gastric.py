"""Probe P1 (paper/paper_plan.md): does the RC certify gastric cohorts?

A go/no-go for H3, not a result for the paper. It imports no runner and writes
only under results/paper_a/prelim/.

Models: ridge (penalty LAM on standardized features, unpenalized intercept) on
each of the five toxicity labels (percentile ranks in [0, 1], limit 0.6), fit
on the 320 training arms. Affine in the labels:

    f_j(x) = s_j(x)' y_j,   s_j(x) = 1/n + H' (z(x) - zbar),

so the RC for outcome j at radius t is f_j(x) + t ||Sigma_j^1/2 s_j(x)|| <= 0.6.

For each of the 96 test cohorts (context fixed), the CERTIFIED RADIUS

    r_hat = max over the cohort's regimens x of  min_j (0.6 - f_j(x)) / ||Sigma_j^1/2 s_j(x)||

(the largest radius at which the RC still has a regimen), solved exactly by
Dinkelbach iterations for max-min ratios; each step is a MISOCP with the
domain constraint (at most 3 drugs) and the convex-hull trust region, as in the
existing pipeline. A cohort whose nominal problem is infeasible has no r_hat.

Joint radius by ENUMERATION (good enough for a go/no-go; it understates the
solver's radius): W = max over outcomes j and over candidate regimens (each
test cohort's context with every training regimen that satisfies the domain
constraint) of s_j(x)'e_j / ||Sigma_j^1/2 s_j(x)||, with e_j ~ N(0, Sigma_j)
independent across outcomes; the radius at level eps is the
ceil((N+1)(1-eps))-th smallest of N draws. A cohort is certified at eps if
r_hat >= that radius, and its level is P(W > r_hat).

Two choices of sigma (``--sigma``):
    binomial  per arm: sqrt(p(1-p)/N_Patient) on the raw proportion, with
              p shrunk to (pN+1)/(N+2) so a 0/N arm has a nonzero sigma, mapped
              to the percentile scale by the delta method (times the KDE
              density of the training proportions at p)
    oof       one sigma per outcome: the sd of 5-fold out-of-fold residuals
"""
import argparse
import json
import os
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy import stats

from src.data.instances import gastric_instance
from src.methods.nominal import build_decision_vars, add_problem_constraints

OUT_DIR = os.path.join("results", "paper_a", "prelim")
TOX = ["dlt", "blood", "constitutional", "infection", "gi"]
LAM = 1.0            # fixed in advance; not selected on the labels
LEVELS = [0.05, 0.1, 0.2]


class Ridge:
    """Ridge on standardized features; everything that depends on inputs only."""

    def __init__(self, X, sigma):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mu) / self.sd
        self.n, self.p = Z.shape
        self.zbar = Z.mean(axis=0)
        Zc = Z - self.zbar
        M = np.linalg.inv(Zc.T @ Zc + LAM * np.eye(self.p))
        self.H = M @ Zc.T                                     # p x n
        # sigma_i s_i(x) = A[i] @ [1; dz]; ||A v|| = ||R v||
        self.A = np.column_stack([sigma / self.n, sigma[:, None] * self.H.T])
        self.R = np.linalg.qr(self.A, mode="r")               # (p+1) x (p+1)

    def fit(self, y):
        return float(np.mean(y)), self.H @ y

    def design(self, X):
        """[1, z(x) - zbar] for each row."""
        dz = (np.atleast_2d(X) - self.mu) / self.sd - self.zbar
        return np.column_stack([np.ones(len(dz)), dz])

    def predict(self, X, coef):
        b0, beta = coef
        return b0 + self.design(X)[:, 1:] @ beta

    def wnorm(self, X):
        return np.linalg.norm(self.design(X) @ self.R.T, axis=1)


def sigma_binomial(inst, j):
    md = inst.constraints[j].models_data[0]
    N = inst.X_train[:, inst.feature_names.index("N_Patient")]
    p = (md.y_true * N + 1.0) / (N + 2.0)
    s_raw = np.sqrt(p * (1.0 - p) / N)
    dens = stats.gaussian_kde(md.y_true)(md.y_true)
    return dens * s_raw


def sigma_oof(X, y, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    res = np.empty(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        r = Ridge(X[tr], np.ones(len(tr)))
        res[te] = y[te] - r.predict(X[te], r.fit(y[tr]))
    return np.full(len(y), float(np.std(res)))


def build_model(inst, ridges, coefs, rhs):
    """Feasible regimens of one cohort (context pinned later), with each
    outcome's prediction f_j and u_j >= ||Sigma^1/2 s_j(x)||."""
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = build_decision_vars(m, inst)
    add_problem_constraints(m, x, inst)
    d = inst.n_features
    f, u = [], []
    for r, (b0, beta) in zip(ridges, coefs):
        dz = [(x[k] - float(r.mu[k])) / float(r.sd[k]) - float(r.zbar[k]) for k in range(d)]
        f.append(b0 + gp.quicksum(float(beta[k]) * dz[k] for k in range(d)))
        w = m.addVars(r.R.shape[0], lb=-GRB.INFINITY)
        for a in range(r.R.shape[0]):
            m.addConstr(w[a] == float(r.R[a, 0])
                        + gp.quicksum(float(r.R[a, k + 1]) * dz[k] for k in range(d)))
        uj = m.addVar(lb=0.0)
        m.addConstr(gp.quicksum(w[a] * w[a] for a in range(r.R.shape[0])) <= uj * uj)
        u.append(uj)
    eta = m.addVar(lb=-GRB.INFINITY)
    rows = [m.addConstr(eta <= rhs - f[j]) for j in range(len(ridges))]
    m.setObjective(eta, GRB.MAXIMIZE)
    return m, x, eta, rows, u


def certified_radius(m, x, eta, rows, u, ridges, coefs, rhs, tol=1e-5, max_iter=30):
    """Dinkelbach for max_x min_j (rhs - f_j) / u_j. Returns (r_hat, x, iters)."""
    s, xs = 0.0, None
    for it in range(max_iter):
        for j, row in enumerate(rows):
            m.chgCoeff(row, u[j], s)                  # eta <= rhs - f_j - s u_j
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return None, None, it + 1, int(m.Status)
        xs = np.array([v.X for v in x])
        if it == 0 and eta.X < -1e-9:
            return None, xs, 1, int(m.Status)        # nominal infeasible
        ratios = [(rhs - r.predict(xs[None], c)[0]) / r.wnorm(xs[None])[0]
                  for r, c in zip(ridges, coefs)]
        s_new = float(min(ratios))
        if eta.X <= tol or s_new <= s + tol:
            return max(s, s_new), xs, it + 1, int(m.Status)
        s = s_new
    return s, xs, max_iter, int(m.Status)


def enum_W(inst, ridges, n_draws, seed):
    """Joint W draws over each test context x every domain-feasible training regimen."""
    dec, ctx = inst.decision_var_indices, inst.context_var_indices
    dc = inst.domain_constraints[0]
    regs = inst.trust_region_points
    regs = regs[regs @ dc.coeffs[dec] <= dc.rhs + 1e-9]
    P = np.repeat(inst.X_test, len(regs), axis=0)
    P[:, dec] = np.tile(regs, (len(inst.X_test), 1))
    rng = np.random.default_rng(seed)
    W = np.full(n_draws, -np.inf)
    for r in ridges:
        D = r.design(P)
        den = np.linalg.norm(D @ r.R.T, axis=1)
        for start in range(0, n_draws, 200):
            xi = rng.standard_normal((r.n, min(200, n_draws - start)))
            num = D @ (r.A.T @ xi)
            W[start:start + xi.shape[1]] = np.maximum(
                W[start:start + xi.shape[1]], (num / den[:, None]).max(axis=0))
    return np.sort(W), len(regs)


def summarize(kind, chunk_rows, meta):
    """Print and return the summary for one sigma choice over the given cohorts."""
    radius = {float(e): v for e, v in meta["joint_radius_enum"].items()}
    rh = np.array([c["r_hat"] for c in chunk_rows if c["r_hat"] is not None])
    n = len(chunk_rows)
    summ = dict(meta)
    summ.update({
        "n_cohorts": n,
        "nominal_feasible_frac": len(rh) / n,
        "certified_frac": {e: float(np.sum(rh >= radius[e]) / n) for e in LEVELS},
        "r_hat_quantiles": ([float(q) for q in np.quantile(rh, [0, .25, .5, .75, 1])]
                            if len(rh) else []),
        "mean_iters": float(np.mean([c["iters"] for c in chunk_rows])),
        "mean_time_s": float(np.mean([c["time_s"] for c in chunk_rows])),
    })
    sg = summ["sigma"]
    print(f"\n[{kind}] sigma median per outcome: "
          + ", ".join(f"{t} {v['sigma_median']:.3f}" for t, v in sg.items()))
    print(f"[{kind}] ridge train R2: "
          + ", ".join(f"{t} {v['train_R2']:.2f}" for t, v in sg.items()))
    print(f"[{kind}] joint radius (enumeration, {summ['n_draws']} draws, "
          f"{summ['n_regimens_enumerated']} regimens x 96 contexts): "
          + ", ".join(f"eps={e}: {radius[e]:.2f}" for e in LEVELS)
          + f"; closed-form (union, eps=0.1) {summ['closed_form_union']:.2f}")
    print(f"[{kind}] {n} cohorts: nominal feasible {summ['nominal_feasible_frac']:.2f}; "
          "certified " + ", ".join(f"eps={e}: {v:.2f}"
                                   for e, v in summ["certified_frac"].items()))
    print(f"[{kind}] r_hat quantiles (0/25/50/75/100%): "
          + " ".join(f"{q:.2f}" for q in summ["r_hat_quantiles"])
          + f"; Dinkelbach iters {summ['mean_iters']:.1f}, "
          f"{summ['mean_time_s']:.2f}s/cohort", flush=True)
    return summ


def main(args):
    t0 = time.time()
    inst = gastric_instance()
    X = inst.X_train
    js = [next(i for i, c in enumerate(inst.constraints) if c.name == f"{t}_constraint")
          for t in TOX]
    rhs = float(inst.constraints[js[0]].rhs)
    last = len(inst.X_test) if args.n_cohorts is None else min(
        args.first + args.n_cohorts, len(inst.X_test))
    for kind in args.sigma:
        ridges, coefs, sig_summary = [], [], {}
        for t, j in zip(TOX, js):
            y = inst.constraints[j].models_data[0].y_train
            sig = sigma_binomial(inst, j) if kind == "binomial" else sigma_oof(X, y)
            r = Ridge(X, sig)
            c = r.fit(y)
            ridges.append(r)
            coefs.append(c)
            resid = y - r.predict(X, c)
            sig_summary[t] = {"sigma_median": float(np.median(sig)),
                              "train_R2": float(1 - resid.var() / y.var())}
        # W depends only on inputs and its seed: every chunk computes the same draws
        W, n_regs = enum_W(inst, ridges, args.n_draws, seed=1)
        N = len(W)
        radius = {e: float(W[min(int(np.ceil((N + 1) * (1 - e))), N) - 1]) for e in LEVELS}
        meta = {"sigma": sig_summary, "n_draws": N, "n_regimens_enumerated": n_regs,
                "joint_radius_enum": radius,
                "closed_form_union": float(np.sqrt(stats.chi2.ppf(
                    1 - 0.1 / len(TOX), ridges[0].p + 1)))}
        m, x, eta, rows, u = build_model(inst, ridges, coefs, rhs)
        cohorts = []
        for i in range(args.first, last):
            for k in inst.context_var_indices:
                x[k].LB = x[k].UB = float(inst.X_test[i, k])
            ts = time.time()
            r_hat, xs, iters, status = certified_radius(m, x, eta, rows, u, ridges, coefs, rhs)
            row = {"cohort": i, "r_hat": r_hat, "iters": iters, "status": status,
                   "time_s": time.time() - ts, "nominal_feasible": r_hat is not None}
            if r_hat is not None:
                row["level"] = float(np.mean(W > r_hat))
                row["x"] = xs.tolist()
            cohorts.append(row)
            print(f"[{kind}] cohort {i}: r_hat {r_hat}, {row['time_s']:.1f}s", flush=True)
        summarize(kind, cohorts, meta)
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = f"p1_gastric_{kind}_{args.first:03d}"
        with open(os.path.join(OUT_DIR, tag + ".json"), "w") as fh:
            json.dump({"meta": meta, "cohorts": cohorts}, fh, indent=1)
    print(f"\ndone in {time.time() - t0:.0f}s")


def collect(args):
    import glob
    out = {}
    for kind in args.sigma:
        files = sorted(glob.glob(os.path.join(OUT_DIR, f"p1_gastric_{kind}_[0-9]*.json")))
        if not files:
            continue
        chunks = [json.load(open(f)) for f in files]
        cohorts = sorted((c for ch in chunks for c in ch["cohorts"]),
                         key=lambda c: c["cohort"])
        out[kind] = summarize(kind, cohorts, chunks[0]["meta"])
    with open(os.path.join(OUT_DIR, "p1_gastric_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", nargs="+", choices=["binomial", "oof"],
                    default=["binomial", "oof"])
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--first", type=int, default=0, help="first test cohort")
    ap.add_argument("--n-cohorts", type=int, default=None,
                    help="cohorts in this chunk (default: all from --first)")
    ap.add_argument("--collect", action="store_true",
                    help="merge the chunk files and print the summary")
    a = ap.parse_args()
    collect(a) if a.collect else main(a)

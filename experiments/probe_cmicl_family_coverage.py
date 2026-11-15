"""Does C-MICL's Assumption 4.1 fail more for some surrogate families? (reactor)

A DIAGNOSTIC, not part of any evaluation. It writes only under
results/reactor/diagnostics/cmicl_family/.

THE QUESTION. Ovalle et al.'s Theorem 4.1 is a statement about an exchangeable
test point; it reaches the optimizer's ``x*`` only through Assumption 4.1
(feasibility and conformal coverage conditionally independent given ground-truth
feasibility), which their Sec. 6 says can fail. On the reactor, C-MICL
feasibility EQUALS coverage at ``x*`` to two decimals
(``probe_cmicl_cost_sampling.py``), so the assumption is the whole story there.
The hypothesis tested here: it fails MORE for tree surrogates. A tree's ``h`` is
constant on each cell, a linear cost sends ``x*`` to the cell's corner in the
cost-saving direction, and on the reactor saving cost lowers benzene flow, so the
truth at ``x*`` sits below the cell's own average -- an optimistic error the
calibration points (spread over cell interiors) never show, and which the smooth
32x32 width model ``u`` cannot follow.

WHAT IT MEASURES, per surrogate family and noise redraw. Coverage means
``|F(x) - h(x)| <= q u(x)`` with ``F`` the ODE, and ``u(x) >= 0``:

  marginal   random points, uniform over the design box (the population the
             calibration rows are exchangeable with). With the NOISY label this
             should be ~1-alpha by construction -- the sanity check.
  boundary   random points ON C-MICL's own boundary, by bisection between a
             C-MICL-feasible and an infeasible point of a random sample. Being on
             the boundary, without being chosen by a cost.
  x*         the optimizer's choice under Ovalle et al.'s 100 cost vectors.

boundary -> x* is the selection effect that Assumption 4.1 assumes away;
marginal -> boundary is the effect of the boundary alone. Beside each: the
optimistic error ``h - F`` and the width ``u``.

For the tree families, the CELL CHECK reads the corner mechanism off directly:
the cell is the box on which ``h`` is constant (the intersection of every tree's
leaf), K points are drawn uniformly in it (inside the domain constraints), and
the row records ``mean F(cell) - F(x*)`` (positive: ``x*`` sits where the truth
is lower than its cell's average) and the fraction of cell points CHEAPER than
``x*`` (near 0: ``x*`` is at the cell's cost-saving corner). The error at ``x*``
then splits exactly: ``h - F(x*) = [h - mean F(cell)] + [mean F(cell) - F(x*)]``,
the optimizer choosing a cell whose prediction overstates it, plus the corner. Done for the nominal
``x*`` too, which is the same question for the robust counterpart of Paper A.

DESIGN. The rows are the cached seed-42 designs (``reactor_micl``); only the
label noise is redrawn (``--seed``), which also seeds the calibration split.
Their protocol is one dataset and 100 cost vectors; the redraws keep a family's
result from being one fitted model's quirk without new ODE designs. ``h``, ``u``
and ``q`` are fitted ONCE per (family, seed): ``train_model`` is memoized here,
so ``solve_cmicl``'s per-instance refit returns the identical fitted object.

FAMILIES are Ovalle et al.'s fixed settings (``regression.py``, cmicl.py
DIFFERENCES #2), in sklearn: rf, gbm, mlp; ``cart`` (depth 5) is the plain-tree
extreme. LinearDT is not here: it has no MIP embedding in this repo yet. ``u``
is the production width model (config.yaml), the same for every family, as
theirs is.

Stages:
    --stage prep       reference points + their ODE truth (once)
    --stage run        one family and one noise seed; saves after every cost
    --stage collect    one table per family, pooled over seeds
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.methods.cmicl as cmicl_mod
import src.methods.nominal as nominal_mod
from src.data.dma_mr import benzene_flow
from src.data.generate import reactor_micl
from src.methods.cmicl import calibrate_conformal_model, solve_cmicl
from src.methods.nominal import resolve_mip_gap, solve_nominal
from src.models.train import train_model as _train_model

OUT_DIR = (os.environ.get("CMICL_FAMILY_OUT")      # smoke tests write elsewhere
           or os.path.join("results", "reactor", "diagnostics", "cmicl_family"))
REF_PATH = os.path.join(OUT_DIR, "reference.npz")
FLOOR = 50.0
DATA_SEED = 42                  # the cached designs; only the noise is redrawn
PAPER_INPUT_SCALE = np.array([0.01, 0.01, 0.01, 1.0, 0.01])

FAMILIES = {
    "rf": ("rf", {"n_estimators": 15, "max_depth": 5, "min_samples_split": 3,
                  "max_features": 0.6}),
    "gbm": ("gbm", {"n_estimators": 15, "learning_rate": 0.2, "max_depth": 5,
                    "min_samples_split": 5, "max_features": 0.6}),
    "mlp": ("mlp", {"hidden_layer_sizes": (32, 32), "solver": "adam",
                    "alpha": 0.01, "max_iter": 2000, "batch_size": 32}),
    "cart": ("cart", {"max_depth": 5}),
}
TREES = {"rf", "gbm", "cart"}


# --------------------------------------------------------------------------
# Plumbing: one fit per (family, seed); Gurobi inside the allocation
# --------------------------------------------------------------------------
_FIT_CACHE = {}


def _memo_train(X, y, model_type="rf", params=None, normalize=True):
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    key = (model_type, json.dumps(params or {}, sort_keys=True, default=str),
           normalize, hashlib.sha1(X.tobytes()).hexdigest(),
           hashlib.sha1(y.tobytes()).hexdigest())
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = _train_model(X, y, model_type, params, normalize)
    return _FIT_CACHE[key]


_orig_cfg_log = nominal_mod.configure_solver_log


def _cfg_log(model, tag="solve", note=""):
    _orig_cfg_log(model, tag, note)
    if os.environ.get("GRB_THREADS"):
        model.Params.Threads = int(os.environ["GRB_THREADS"])


for _mod in (cmicl_mod, nominal_mod):
    _mod.train_model = _memo_train
    _mod.configure_solver_log = _cfg_log


def n_jobs():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def ode(X):
    """ODE truth at the rows of X, in parallel (~0.28 s per point)."""
    X = np.atleast_2d(np.asarray(X, float))
    if len(X) == 0:
        return np.zeros(0)
    from joblib import Parallel, delayed
    return np.array(Parallel(n_jobs=n_jobs())(delayed(benzene_flow)(x) for x in X))


# --------------------------------------------------------------------------
# Instance, costs, domain
# --------------------------------------------------------------------------
def load():
    cfg = yaml.safe_load(open("config.yaml"))
    rc = cfg.get("reactor", {})
    inst = reactor_micl(n_train=int(rc.get("n_train", 1000)),
                        noise_std=float(rc.get("noise_std", 2.0)),
                        seed=DATA_SEED)
    return cfg, inst


def paper_costs(n):
    """Ovalle et al.'s exact stream (regression.py:713-719), in our units."""
    c = np.random.RandomState(0).uniform(-4.0, 4.0, size=(n, 5))
    c[c < 0] /= 10.0
    return c * PAPER_INPUT_SCALE


def in_domain(inst, X):
    ok = np.ones(len(X), bool)
    for dc in inst.domain_constraints:
        ok &= X @ dc.coeffs <= dc.rhs + 1e-9
    return ok


def box_sample(inst, n, rng):
    lb, ub = inst.variable_lb, inst.variable_ub
    return lb + (ub - lb) * rng.uniform(size=(n, len(lb)))


# --------------------------------------------------------------------------
# Coverage pieces
# --------------------------------------------------------------------------
def cmicl_feasible(h, u, q, X):
    hx, ux = h.predict(X), u.predict(X)
    return (ux >= 0) & (hx - q * ux >= FLOOR), hx, ux


def covered(F, hx, ux, q):
    return (ux >= 0) & (np.abs(F - hx) <= q * ux)


def boundary_points(inst, h, u, q, n, rng, n_pool=20000, iters=40):
    """Random points on C-MICL's boundary, on its FEASIBLE side (as x* is)."""
    P = box_sample(inst, n_pool, rng)
    P = P[in_domain(inst, P)]
    feas, _, _ = cmicl_feasible(h, u, q, P)
    A, B = P[feas], P[~feas]
    if len(A) == 0 or len(B) == 0:
        return np.zeros((0, P.shape[1]))
    a = A[rng.integers(len(A), size=n)]
    b = B[rng.integers(len(B), size=n)]
    lo, hi = np.zeros(n), np.ones(n)          # lo feasible, hi infeasible
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f, _, _ = cmicl_feasible(h, u, q, a + mid[:, None] * (b - a))
        lo, hi = np.where(f, mid, lo), np.where(f, hi, mid)
    return a + lo[:, None] * (b - a)


def _trees(pipe):
    est = pipe.named_steps["model"]
    if hasattr(est, "tree_"):
        return [est]
    e = est.estimators_
    return list(e.ravel()) if hasattr(e, "ravel") else list(e)


def cell_box(pipe, x, lb, ub):
    """The box on which a tree ensemble's prediction is constant around x."""
    sc = pipe.named_steps["scaler"]
    z = sc.transform(x.reshape(1, -1))
    lo, hi = lb.astype(float).copy(), ub.astype(float).copy()
    for t in _trees(pipe):
        tr = t.tree_
        for node in t.decision_path(z).indices:
            f = tr.feature[node]
            if f < 0:                              # leaf
                continue
            thr = tr.threshold[node] * sc.scale_[f] + sc.mean_[f]
            if z[0, f] <= tr.threshold[node]:
                hi[f] = min(hi[f], thr)
            else:
                lo[f] = max(lo[f], thr)
    return lo, hi


def cell_sample(inst, lo, hi, k, rng, rounds=50):
    got = []
    for _ in range(rounds):
        P = lo + (hi - lo) * rng.uniform(size=(400, len(lo)))
        got.extend(P[in_domain(inst, P)])
        if len(got) >= k:
            break
    return np.array(got[:k]) if got else np.zeros((0, len(lo)))


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_prep(args):
    """Reference points uniform over the design box, with ODE truth."""
    cfg, inst = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    X = box_sample(inst, args.n_ref, np.random.default_rng(7))
    t0 = time.time()
    F = ode(X)
    np.savez(REF_PATH, X=X, F=F, dom=in_domain(inst, X))
    print(f"saved {REF_PATH}: {len(X)} points, {np.isnan(F).sum()} ODE failures, "
          f"{in_domain(inst, X).mean():.2f} in the domain, {time.time()-t0:.0f}s")


def stage_run(args):
    cfg, inst = load()
    cm = cfg["methods"]["cmicl"]
    mip_gap = resolve_mip_gap(cfg)
    sigma = float(cfg["reactor"].get("noise_std", 2.0))
    mt, mp = FAMILIES[args.family]
    md = inst.constraints[0].models_data[0]
    # the noise redraw: same designs, fresh labels
    md.y_train = md.y_true + sigma * np.random.default_rng([args.seed, 11]).standard_normal(len(md.y_true))
    kw = dict(alpha=args.alpha, cal_frac=float(cm.get("cal_frac", 0.2)),
              seed=args.seed, width_model_type=cm.get("width_model_type"),
              width_model_params=cm.get("width_model_params"))
    h, u, q, info = calibrate_conformal_model(md.X_train, md.y_train, mt, mp,
                                              label=args.family, **kw)
    # nominal embeds a model fit on ALL rows (same memo key as solve_nominal's)
    h_nom = _memo_train(md.X_train, md.y_train, mt, mp)
    n_train = int(cfg["reactor"].get("n_train", 1000))

    path = os.path.join(OUT_DIR, f"{args.family}_s{args.seed}_a{args.alpha:g}.json")
    out = json.load(open(path)) if os.path.exists(path) else None
    if out is None:
        rng = np.random.default_rng([args.seed, 3])
        z = np.load(REF_PATH)
        ok = np.isfinite(z["F"])
        Xr, Fr, dom = z["X"][ok], z["F"][ok], z["dom"][ok]
        hr, ur = h.predict(Xr), u.predict(Xr)
        yr = Fr + sigma * rng.standard_normal(len(Fr))
        Xb = boundary_points(inst, h, u, q, args.n_boundary, rng)
        Fb = ode(Xb)
        okb = np.isfinite(Fb)
        Xb, Fb = Xb[okb], Fb[okb]
        hb, ub_ = h.predict(Xb), u.predict(Xb)
        out = {
            "family": args.family, "seed": args.seed, "alpha": args.alpha,
            "model_params": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in mp.items()},
            "fit": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in info.items()},
            "marginal": {
                "n": int(len(Fr)),
                "cov_noisy": float(covered(yr, hr, ur, q).mean()),
                "cov_true": float(covered(Fr, hr, ur, q).mean()),
                "cov_true_domain": float(covered(Fr[dom], hr[dom], ur[dom], q).mean()),
                "err": float(np.mean(hr - Fr)), "abs_err": float(np.mean(np.abs(hr - Fr))),
                "u": float(np.mean(ur)),
            },
            "boundary": {
                "n": int(len(Fb)),
                "cov": float(covered(Fb, hb, ub_, q).mean()),
                "feasible": float(np.mean(Fb >= FLOOR)),
                "err": float(np.mean(hb - Fb)), "u": float(np.mean(ub_)),
                "frac_u_binding": float(np.mean(ub_ < 1e-3)),
            },
            "rows": [],
        }
    done = {r["cost"] for r in out["rows"]}
    costs = paper_costs(args.n_costs)
    rng_cell = np.random.default_rng([args.seed, 5])
    t0 = time.time()
    for i, c in enumerate(costs):
        if i in done:
            continue
        inst_c = reactor_micl(n_train=n_train,
                              noise_std=sigma, seed=DATA_SEED, cost_vector=c)
        md_c = inst_c.constraints[0].models_data[0]
        assert np.array_equal(md_c.X_train, md.X_train)
        md_c.y_train = md.y_train
        res = {
            "cmicl": solve_cmicl(inst_c, model_type=mt, model_params=mp,
                                 multiplicity=cm.get("multiplicity", "none"),
                                 mip_gap=mip_gap, **kw),
            "nominal": solve_nominal(inst_c, model_type=mt, model_params=mp,
                                     mip_gap=mip_gap),
        }
        X_all, parts = [], {}
        for meth, r in res.items():
            if r.status != "optimal":
                continue
            x = np.asarray(r.x_opt, float).ravel()
            pts = [x]
            if args.family in TREES:
                hm = h if meth == "cmicl" else h_nom
                lo, hi = cell_box(hm, x, inst.variable_lb, inst.variable_ub)
                pts += list(cell_sample(inst, lo, hi, args.k_cell, rng_cell))
            parts[meth] = (len(X_all), len(pts), x)
            X_all += pts
        F_all = ode(np.array(X_all)) if X_all else np.zeros(0)
        for meth, r in res.items():
            row = {"cost": i, "method": meth, "status": r.status}
            if meth in parts:
                s, n, x = parts[meth]
                xr = x.reshape(1, -1)
                hm = h if meth == "cmicl" else h_nom
                hx, ux, Fx = float(hm.predict(xr)[0]), float(u.predict(xr)[0]), float(F_all[s])
                row.update(x=x.tolist(), F=Fx, h=hx, u=ux,
                           feasible=bool(Fx >= FLOOR),
                           covered=bool(ux >= 0 and abs(Fx - hx) <= q * ux),
                           err=hx - Fx, obj=float(c @ x))
                if n > 1:
                    Xc, Fc = np.array(X_all[s + 1:s + n]), F_all[s + 1:s + n]
                    okc = np.isfinite(Fc)
                    Xc, Fc = Xc[okc], Fc[okc]
                    hc = hm.predict(Xc)
                    row.update(cell_n=int(len(Fc)),
                               cell_h_sd=float(np.std(hc)),   # ~0, or the cell is wrong
                               cell_F_mean=float(np.mean(Fc)),
                               cell_drop=float(np.mean(Fc) - Fx),
                               cell_frac_cheaper=float(np.mean(Xc @ c < c @ x)))
            out["rows"].append(row)
        _save(path, out)
        print(f"cost {i}: " + ", ".join(
            f"{r['method']} F={r.get('F', np.nan):.2f} cov={r.get('covered')}"
            for r in out["rows"][-2:]) + f"  ({time.time()-t0:.0f}s)", flush=True)
    _save(path, out)


def _save(path, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def stage_collect(args):
    def m(v):
        v = [a for a in v if a is not None and np.isfinite(a)]
        return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
    for fam in FAMILIES:
        files = sorted(glob.glob(os.path.join(OUT_DIR, f"{fam}_s*_a{args.alpha:g}.json")))
        if not files:
            continue
        runs = [json.load(open(f)) for f in files]
        print(f"\n=== {fam}: {len(runs)} noise redraws, alpha={args.alpha:g} "
              f"(mean, and sd over redraws)")
        def line(name, vals):
            mu, sd = m(vals)
            print(f"  {name:38s} {mu:7.3f}  ({sd:.3f})")
        line("marginal coverage, noisy y (~1-alpha)", [r["marginal"]["cov_noisy"] for r in runs])
        line("marginal coverage, ODE", [r["marginal"]["cov_true"] for r in runs])
        line("marginal coverage, ODE, in domain", [r["marginal"]["cov_true_domain"] for r in runs])
        line("boundary coverage, ODE", [r["boundary"]["cov"] for r in runs])
        line("boundary ODE-feasible", [r["boundary"]["feasible"] for r in runs])
        for meth in ("cmicl", "nominal"):
            rr = [[x for x in r["rows"] if x["method"] == meth] for r in runs]
            line(f"{meth}: solved", [np.mean([x["status"] == "optimal" for x in g]) for g in rr])
            sv = [[x for x in g if "F" in x] for g in rr]
            line(f"{meth}: ODE-feasible at x*", [np.mean([x["feasible"] for x in g]) if g else None for g in sv])
            if meth == "cmicl":
                line("cmicl: coverage at x*", [np.mean([x["covered"] for x in g]) if g else None for g in sv])
            line(f"{meth}: h - F at x* (optimistic > 0)", [np.mean([x["err"] for x in g]) if g else None for g in sv])
            if meth == "cmicl":
                line("   h - F at boundary points", [r["boundary"]["err"] for r in runs])
                line("   h - F at marginal points", [r["marginal"]["err"] for r in runs])
                line("   u at x*", [np.mean([x["u"] for x in g]) if g else None for g in sv])
                line("   u at boundary points", [r["boundary"]["u"] for r in runs])
            if fam in TREES:
                cv = [[x for x in g if x.get("cell_n", 0) >= 5] for g in sv]
                # h - F(x*) = [h - mean F(cell)]  +  [mean F(cell) - F(x*)]
                #              which cell is chosen     where in the cell x* sits
                line(f"{meth}: h - mean F(cell): cell choice", [np.mean([x["h"] - x["cell_F_mean"] for x in g]) if g else None for g in cv])
                line(f"{meth}: mean F(cell) - F(x*): corner", [np.mean([x["cell_drop"] for x in g]) if g else None for g in cv])
                line(f"{meth}: cell points cheaper than x*", [np.mean([x["cell_frac_cheaper"] for x in g]) if g else None for g in cv])
                line(f"{meth}: max h sd within cell (~0)", [np.max([x["cell_h_sd"] for x in g]) if g else None for g in cv])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["prep", "run", "collect"])
    ap.add_argument("--family", choices=sorted(FAMILIES), default="rf")
    ap.add_argument("--seed", type=int, default=0, help="noise redraw and calibration split")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-costs", type=int, default=100)
    ap.add_argument("--n-ref", type=int, default=1000)
    ap.add_argument("--n-boundary", type=int, default=200)
    ap.add_argument("--k-cell", type=int, default=20)
    args = ap.parse_args()
    {"prep": stage_prep, "run": stage_run, "collect": stage_collect}[args.stage](args)

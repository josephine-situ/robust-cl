"""Does resampling the cost vector restore C-MICL's feasibility rate? (reactor)

A DIAGNOSTIC, not part of any evaluation. It changes nothing about the
production instance: ``reactor_micl`` already takes an optional ``cost_vector``,
and this script passes one explicitly. ``cost_vector`` stays FIXED at ones
everywhere a result is produced -- see ``src/data/generate.py``, "a new ``c`` is
a different problem, not a new sample of this one".

THE QUESTION. On the reactor we measure ODE feasibility 0.60 at alpha=0.1 where
Ovalle et al. (arXiv:2506.03531, Fig. 1) report >= 0.90, while split conformal's
own marginal coverage comes out at 0.899 -- exactly 1-alpha. So the calibration
is right and the OPTIMUM is what escapes it: coverage at ``x*`` is 0.50. Their
Assumption 4.1 asks feasibility and coverage to be conditionally independent
given ground-truth feasibility, and a fixed ``c`` is a good way to break it --
every training draw sends the argmin to the same corner of the design box, so one
systematic model error there is never averaged away.

They average over **100 randomly generated instances, each defined by a sampled
cost vector**, with the calibration set held fixed. That is a different average
over a different source of randomness, and this script measures it:

    fixed c   (what we report)  -- vary the TRAINING DRAW, 10 folds  -> 0.60
    sampled c (what they report) -- vary c, model held fixed, N draws -> ?

If the sampled-c rate returns to ~0.9, the gap is the protocol and not the
method, and our 0.60 is a statement about a harder average rather than a defect.

WHAT c IS SAMPLED FROM IS OURS, NOT THEIRS. The paper does not give the
distribution (Table 5 says only "operational or design cost coefficient"), so two
readings are run and reported side by side:

  ``unit``   c_i ~ U(0,1). The literal reading. But the five variables are not
             commensurate (T ~ 1e3 against dt ~ 1), so almost every draw is still
             dominated by T, v0 and v_He -- the same corner, and a weak test.
  ``scaled`` c_i ~ U(0,1) / span_i, span_i the design box width. Each variable
             then contributes comparably and different draws genuinely favour
             different variables. ``generate.py`` anticipates exactly this
             ("pass an explicit cost_vector to weight the variables evenly").

Because a feasibility rate is only interesting if the optimum actually moved, the
SPREAD of x* is reported alongside it, in box-width units. A scheme that leaves
x* pinned tells us nothing about the hypothesis either way, and this is how you
can see which happened.

RESULT (2026-08-22, alpha=0.1, N=100/scheme, cost_seed=0, full-data 800/200 fit,
CV-selected mlp base). **The hypothesis is confirmed under one reading of `c` and
destroyed under the other**, which is a more interesting answer than either alone:

    scheme    C-MICL feas   coverage at x*   mean slack   x* mean pairwise
    unit          0.99           0.99          +1.22           0.145
    scaled        0.11           0.11          -1.05           0.330
    (nominal is 0.00 under both)

Read three things off that:

1. **Their >= 0.90 IS reproducible here** -- 0.99 under the literal reading. So
   our 0.60 is not a defect in the implementation; it is a different average.
2. **Feasibility EQUALS coverage-at-x* to two decimals in both schemes** (0.99 /
   0.99 and 0.11 / 0.11). Feasibility here is not "mostly" about coverage at the
   optimum, it IS coverage at the optimum. That is the mechanism, measured.
3. **The rate is a property of the cost distribution, not of the method.** Under
   `unit` the optimum barely moves and never leaves the region the model fits
   well -- note T is pinned to a bound, sd 9e-16. Under `scaled` it roams over
   d_t and L (sd 0.17 each, more than twice the pairwise spread) and coverage
   collapses. Assumption 4.1 holds in the first case by an accident of how the
   variables are scaled, not because the method secured it.

So "does the reactor line up with the paper" has no single answer: under their
(unstated) sampling it does, and a cost distribution that actually explores the
design box takes it to 0.11. Do NOT quote either number as "C-MICL's feasibility
on the reactor" without naming the cost distribution it came from.

The ``fixed_ones`` anchor scheme was added AFTER the committed CSV, so that file
carries only ``unit`` and ``scaled``; the anchor was measured separately and is
feasible (F_true = 51.79 vs the floor of 50, nominal 45.85).

Usage:
    uv run python experiments/probe_cmicl_cost_sampling.py
    uv run python experiments/probe_cmicl_cost_sampling.py --n-instances 100 --alpha 0.1
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dma_mr import benzene_flow
from src.data.generate import reactor_micl
from src.methods.cmicl import calibrate_conformal_model, solve_cmicl
from src.methods.nominal import solve_nominal, resolve_mip_gap

OUT_DIR = os.path.join("results", "reactor", "diagnostics")
PRODUCT_FLOOR = 50.0


def sample_costs(scheme, n, span, rng):
    """N cost vectors under one reading of "a sampled cost vector".

    ``fixed_ones`` is not a sampling scheme: it is the production ``c`` repeated
    once, the anchor every rate here has to be read against. Without it a rate
    over sampled ``c`` has nothing to be a rate *relative to*.
    """
    d = len(span)
    if scheme == "fixed_ones":
        return np.ones((1, d))
    u = rng.uniform(0.0, 1.0, size=(n, d))
    if scheme == "unit":
        return u
    if scheme == "scaled":
        return u / span
    raise ValueError(f"unknown scheme {scheme!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-instances", type=int, default=100,
                   help="cost vectors per scheme (paper: 100)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="conformal miscoverage; the paper's main text is 0.1")
    p.add_argument("--schemes", nargs="+",
                   default=["fixed_ones", "unit", "scaled"],
                   choices=("fixed_ones", "unit", "scaled"),
                   help="fixed_ones is the production c, run once, as the anchor")
    p.add_argument("--cost-seed", type=int, default=0,
                   help="seed for the cost draws ONLY; the data, the split and "
                        "every model keep uncertainty.bootstrap_seed")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cm = cfg["methods"].get("cmicl", {})
    rc = cfg.get("reactor", {})
    seed = int(cfg["uncertainty"].get("bootstrap_seed", 42))
    mip_gap = resolve_mip_gap(cfg)

    # Same model resolution instances.reactor_instance uses, so the probe embeds what the
    # evaluation embeds.
    from src.data.instances import reactor_model_spec
    mt, mp, from_cv = reactor_model_spec(cfg, verbose=True)
    fixed_cfg = {"model_type": mt, "model_params": mp} if from_cv else None

    base = reactor_micl(n_train=int(rc.get("n_train", 1000)),
                        noise_std=float(rc.get("noise_std", 2.0)),
                        seed=seed, fixed_constraint_config=fixed_cfg)
    span = np.asarray(base.variable_ub, float) - np.asarray(base.variable_lb, float)
    md = base.constraints[0].models_data[0]

    # The model is fitted ONCE, exactly as solve_cmicl will refit it per instance:
    # same rows, same cal_frac, same seed, so h/u/q are identical at every c. This
    # copy is only so the probe can evaluate coverage at x* afterwards.
    h, u, q, floor, info = calibrate_conformal_model(
        md.X_train, md.y_train, mt, mp,
        alpha=args.alpha, cal_frac=float(cm.get("cal_frac", 0.2)), seed=seed,
        width_model_type=cm.get("width_model_type") or None,
        width_model_params=cm.get("width_model_params") or None,
        width_floor_frac=float(cm.get("width_floor_frac", 0.05)),
        label="benzene_constraint (probe)",
    )
    print(f"[probe] model fixed across instances: n_fit={info['n_train']} "
          f"n_cal={info['n_cal']} q={q:.4g}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for scheme in args.schemes:
        rng = np.random.default_rng(args.cost_seed)
        costs = sample_costs(scheme, args.n_instances, span, rng)
        n_here = len(costs)
        t0 = time.time()
        for i, c in enumerate(costs):
            inst = reactor_micl(n_train=int(rc.get("n_train", 1000)),
                                noise_std=float(rc.get("noise_std", 2.0)),
                                seed=seed, fixed_constraint_config=fixed_cfg,
                                cost_vector=c)
            for method in ("cmicl", "nominal"):
                if method == "cmicl":
                    r = solve_cmicl(
                        inst, model_type=mt, model_params=mp, alpha=args.alpha,
                        cal_frac=float(cm.get("cal_frac", 0.2)),
                        width_model_type=cm.get("width_model_type") or None,
                        width_model_params=cm.get("width_model_params") or None,
                        width_floor_frac=float(cm.get("width_floor_frac", 0.05)),
                        multiplicity=cm.get("multiplicity", "none"),
                        seed=seed, mip_gap=mip_gap)
                else:
                    r = solve_nominal(inst, model_type=mt, model_params=mp,
                                      mip_gap=mip_gap)
                rec = dict(scheme=scheme, instance=i, method=method,
                           alpha=args.alpha, status=r.status)
                if r.status == "optimal":
                    x = np.asarray(r.x_opt, float).ravel()
                    f_true = float(benzene_flow(x))
                    xr = x.reshape(1, -1)
                    h_star = float(h.predict(xr)[0])
                    hw = float(q * max(float(u.predict(xr)[0]), floor))
                    rec.update(
                        objective=float(np.dot(c, x)),
                        f_true=f_true,
                        feasible=int(f_true >= PRODUCT_FLOOR),
                        covered_at_xstar=int(abs(f_true - h_star) <= hw),
                        slack=f_true - (h_star - hw),
                        **{f"x{j}": float(v) for j, v in enumerate(x)},
                    )
                rows.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  [{scheme}] {i + 1}/{n_here} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"cmicl_cost_sampling_a{args.alpha:g}.csv")
    df.to_csv(path, index=False)

    print("\n" + "=" * 78, flush=True)
    print(f"C-MICL vs a SAMPLED cost vector (alpha={args.alpha:g}, "
          f"N={args.n_instances}/scheme, model held fixed)", flush=True)
    print("Reference, FIXED c=ones over 10 training folds: feasibility 0.60, "
          "coverage-at-x* 0.50", flush=True)
    print("=" * 78, flush=True)
    xcols = [f"x{j}" for j in range(len(span))]
    for scheme in args.schemes:
        for method in ("cmicl", "nominal"):
            g = df[(df.scheme == scheme) & (df.method == method) &
                   (df.status == "optimal")]
            if g.empty:
                print(f"{scheme:>10s} {method:<8s} no solved instances", flush=True)
                continue
            # How far did the optimum actually move? sd per coordinate in box
            # widths, and the mean pairwise distance in the same units. A rate
            # measured over a pinned x* is not evidence about the hypothesis.
            xs = g[xcols].to_numpy() / span
            sd = xs.std(axis=0)
            pairs = list(itertools.islice(
                itertools.combinations(range(len(xs)), 2), 2000))
            dist = (np.mean([np.linalg.norm(xs[a] - xs[b]) for a, b in pairs])
                    if pairs else 0.0)
            extra = ""
            if method == "cmicl":
                extra = (f"  coverage-at-x*={g.covered_at_xstar.mean():.2f}"
                         f"  mean slack={g.slack.mean():+.2f}")
            n_sch = int((df.scheme == scheme).sum() / 2)
            print(f"{scheme:>10s} {method:<8s} feasibility={g.feasible.mean():.2f}"
                  f"  solved={len(g)}/{n_sch}{extra}", flush=True)
            print(f"{'':>10s} {'':<8s} x* spread (box widths): "
                  f"sd={np.array2string(sd, precision=3)} "
                  f"mean pairwise={dist:.3f}", flush=True)
    print(f"\n[probe] wrote {path}", flush=True)


if __name__ == "__main__":
    main()

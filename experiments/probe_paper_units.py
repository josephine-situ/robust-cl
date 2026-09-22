"""Their training regime and their /100 input scaling, our CV-selected h.

A DIAGNOSTIC that stands in for a config change, so the change can be measured
before it is made. Patches this process only, then delegates to
``probe_cmicl_cost_sampling.main``; loop, MIP, cost draw, alpha and split are
identical to every other cell.

WHAT IS ADOPTED, AND WHY. The width net's failure is its FLOOR, not its mean:
the MIP solves for the argmin of cost s.t. h - q*u >= 50, so small u relaxes the
constraint and the optimizer walks to the lowest dip in the fitted surface. A
marginal conformal guarantee says nothing there -- it averages over the
calibration distribution, which is precisely what the optimizer leaves. Measured
across the trainer factorial:

    u fitted with          feas   min u@x*   u<=0 on cal
    baseline (defaults)    0.53      0.000          9/200
    batch_size=32          0.81      0.325          0/200
    all three              0.92      0.740          4/200
    upstream               0.95      1.762          0/200

|residual| here is nearly constant -- a decent h leaves mostly label noise, and
E|N(0,2)| ~ 1.60 against upstream's flat 1.803 -- so the correct surface IS flat.
The linear output layer is free to go negative, and with batch 200 of 800 rows
(4 steps/epoch) sklearn's tol=1e-4 / n_iter_no_change=10 stops early, leaving an
under-converged surface with arbitrary structure. Thin data is where that hurts:
only 216/1000 design rows satisfy the D.1b-e ratios and only 30 reach F >= 50, so
the boundary the optimizer sits on is supported by ~30 rows. u >= 0 in the MIP
then clamps the dip to exactly zero -- binding, no margin.

So batch 32 (25 steps/epoch) plus the full 2000 iterations is a convergence fix,
not a tuning knob. NOTE the interaction: the full-iteration change ALONE made
things worse (0.49), because more epochs on a lumpy batch-200 surface deepens the
dips. It helps only together with the small batch, which is why this combination
had to be measured rather than predicted.

THE /100. Theirs is inputs /100 with dt then multiplied back by 100, i.e. net
(0.01, 0.01, 0.01, 1.0, 0.01) over (v0, v_He, T, dt, L) -- regression.py:562-565.
It is applied here as a StandardScaler with mean_=0 and scale_=(100,100,100,1,100)
rather than a new transformer type: the scaling is affine exactly as
StandardScaler is, so embed.py embeds it unchanged and nothing downstream needs
to learn a new object. Applied to h AND u, because splitting them would fit the
two nets on different geometries.

WHAT IS *NOT* ADOPTED, deliberately, and still divergent:
  - their labels /10. Separable from the input scaling, and it changes how
    alpha=0.01 bites relative to the output scale. Named, not silently half-done.
  - alpha's SCOPE. sklearn penalises every layer, Keras only the hidden kernels;
    no MLPRegressor argument separates them.
  - h's architecture stays the CV selection (mlp [10,5,2], lbfgs). Already
    excluded as a factor -- but note it was SELECTED under StandardScaler, so
    under /100 it is off-distribution and the selection may want re-running.

Also prints each width net's n_iter_, which is what makes the under-convergence
story testable rather than plausible: if the baseline stops in a few dozen
iterations and this runs to the cap, that is the mechanism, measured.

Usage:
    U_MODE=paper    python -u experiments/probe_paper_units.py --schemes paper --out-suffix _paperfit
    U_MODE=baseline python -u experiments/probe_paper_units.py --schemes paper --out-suffix _basefit
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.methods.cmicl as cmicl_mod

MODE = os.environ.get("U_MODE", "paper")
if MODE not in {"paper", "baseline"}:
    raise SystemExit("U_MODE must be paper|baseline; got " + repr(MODE))

# regression.py:562-565, in (v0, v_He, T, dt, L) order. StandardScaler divides by
# scale_, so these are DIVISORS: dt's 1.0 is their /100 undone by the x100.
PAPER_DIVISORS = np.array([100.0, 100.0, 100.0, 1.0, 100.0])
_WIDTH_SHAPE = (32, 32)
_orig_train_model = cmicl_mod.train_model
_stats = {"width_fits": 0, "n_iter": []}


def _paper_scaler(d):
    """A real fitted StandardScaler that performs their /100.

    Not a FunctionTransformer: embed.py already handles StandardScaler, and the
    transform is affine either way, so this needs no embedder change.
    """
    if d != len(PAPER_DIVISORS):
        raise ValueError("expected %d features, got %d" % (len(PAPER_DIVISORS), d))
    sc = StandardScaler()
    sc.mean_ = np.zeros(d)
    sc.scale_ = PAPER_DIVISORS.copy()
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = d
    sc.n_samples_seen_ = 1
    return sc


def _is_width(params):
    h = (params or {}).get("hidden_layer_sizes")
    if h is None:
        return False
    try:
        return tuple(h) == _WIDTH_SHAPE
    except TypeError:
        return False


def _patched_train_model(X, y, model_type="rf", params=None, normalize=True):
    if MODE == "baseline" or model_type != "mlp" or not normalize:
        return _orig_train_model(X, y, model_type, params, normalize)

    p = dict(params or {})
    kw = dict(
        hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", (100,))),
        solver=p.get("solver", "adam"),
        alpha=p.get("alpha", 1e-4),
        random_state=p.get("random_state", 1),
        max_iter=p.get("max_iter", 10000),
    )
    width = _is_width(p)
    if width:
        # Their regime. batch_size is ignored by lbfgs, so this only ever binds
        # on the adam-fitted width net -- h keeps whatever the CV selected.
        kw["batch_size"] = 32
        kw["tol"] = 0.0
        kw["n_iter_no_change"] = kw["max_iter"]

    # Pre-fitted scaler, so the estimator is fit on transformed X and the
    # Pipeline is assembled from parts -- Pipeline.fit would refit the scaler.
    sc = _paper_scaler(X.shape[1])
    est = MLPRegressor(**kw)
    est.fit(sc.transform(X), y)

    if width:
        _stats["width_fits"] += 1
        _stats["n_iter"].append(int(getattr(est, "n_iter_", -1)))
        if _stats["width_fits"] == 1:
            print("    [paper-units] width net: %s | divisors %s | n_iter_=%s"
                  % (kw, PAPER_DIVISORS.tolist(), est.n_iter_), flush=True)

    return Pipeline([("scaler", sc), ("model", est)])


cmicl_mod.train_model = _patched_train_model

from experiments.probe_cmicl_cost_sampling import main  # noqa: E402

if __name__ == "__main__":
    main()
    n = _stats["n_iter"]
    if MODE == "paper":
        if not n:
            print("WARNING: no width fit intercepted -- the patch did nothing", flush=True)
        else:
            print("\n[paper-units] width-net n_iter_ over %d fits: min=%d median=%d max=%d"
                  % (len(n), min(n), sorted(n)[len(n) // 2], max(n)), flush=True)

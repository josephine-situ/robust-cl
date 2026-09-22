"""Is the gap in how `u` is FITTED? (reactor)

A DIAGNOSTIC. Patches this process only, then delegates to
``probe_cmicl_cost_sampling.main`` so loop, MIP, calibration and cost draw stay
identical to every cell this is the A/B against.

WHERE THIS PICKS UP. Four factors are already excluded: the averaging protocol
(their draw gives 0.53 here, not >=0.90), the training rows (their sheet gives
0.19, worse), h's accuracy at the optimum (2.00 upstream vs 1.87 here), and h's
architecture (pinning 32x32 moves 0.53->0.54 and 0.19->0.30). Across all five
cells feasibility is strictly monotone in mean u(x*) -- 0.19 0.30 0.53 0.54 0.95
against 0.210 0.213 0.398 0.539 1.803 -- and nothing else orders them.

WHAT IS LEFT is `cmicl.py` DIFFERENCES #3: same shape of u, differently fitted.
sklearn's MLPRegressor is not Keras, and the reachable divergences are

    batch      sklearn's 'auto' = min(200, n) = 200 here, against their 32
    epochs     max_iter is a CAP that tol=1e-4 / n_iter_no_change=10 stops early,
               against their fixed 2000
    scaling    train_model(normalize=True) standardises every feature, against
               their own /100 (dt net unscaled), which leaves dt and L small
               beside T and v0

`alpha`'s SCOPE is the one divergence not reachable: sklearn penalises every
layer, Keras only the hidden kernels, and no MLPRegressor argument separates
them. So a null result here narrows but does not close.

THE SCALING CELL IS THE ONE TO WATCH. Standardising gives all five variables
equal leverage, so u can learn fine structure in every direction and reach zero
in the corner the optimizer wants; their /100 keeps the raw scale ratios, which
should give a flatter, smoother u -- the flatness their u actually shows
(u(x*) in [1.762, 1.865] across 23 distinct optima). `raw` is that test's
direction, not its replica: it removes the equalisation rather than substituting
their exact /100.

h IS LEFT AT THE CV SELECTION (mlp [10,5,2]) on purpose. It is excluded already,
and it keeps h's params distinguishable from the width model's [32,32], which is
how the patch below knows which of train_model's two calls to intervene on.

BASELINE for every cell here is `_paper`: feasibility 0.53, mean u(x*) 0.398,
55% of optima at u < 0.10.

Usage (VARIANT in baseline|batch32|full2000|raw|all):
    U_VARIANT=raw python -u experiments/probe_u_trainer.py --schemes paper --out-suffix _u_raw
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.methods.cmicl as cmicl_mod

VARIANT = os.environ.get("U_VARIANT", "baseline")
VALID = {"baseline", "batch32", "full2000", "raw", "all", "converge", "batchauto"}
if VARIANT not in VALID:
    raise SystemExit(f"U_VARIANT must be one of {sorted(VALID)}; got {VARIANT!r}")

_orig_train_model = cmicl_mod.train_model
_WIDTH_SHAPE = (32, 32)
_seen = {"width_fits": 0}


def _is_width(params):
    """The width model is the only [32,32] fit in this configuration."""
    h = (params or {}).get("hidden_layer_sizes")
    if h is None:
        return False
    try:
        return tuple(h) == _WIDTH_SHAPE
    except TypeError:
        return False


def _patched_train_model(X, y, model_type="rf", params=None, normalize=True):
    """train_model, except the WIDTH fit is rebuilt with the variant's knobs.

    Rebuilt rather than delegated because train_model forwards only
    hidden_layer_sizes / solver / alpha / random_state / max_iter to
    MLPRegressor -- batch_size and tol are not reachable through `params`.
    Everything it DOES forward is read the same way here, so `baseline` and an
    unpatched run agree.
    """
    if VARIANT == "baseline" or not _is_width(params):
        return _orig_train_model(X, y, model_type, params, normalize)

    p = dict(params or {})
    kw = dict(
        hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", _WIDTH_SHAPE)),
        solver=p.get("solver", "adam"),
        alpha=p.get("alpha", 1e-4),
        random_state=p.get("random_state", 1),
        max_iter=p.get("max_iter", 10_000),
    )
    norm = normalize

    if VARIANT == "batchauto":
        kw["batch_size"] = "auto"
    if VARIANT in ("batch32", "all", "converge"):
        kw["batch_size"] = 32
    if VARIANT in ("full2000", "all", "converge"):
        # tol=0 with n_iter_no_change=max_iter defeats early stopping, so the fit
        # runs the full cap the way their fixed epoch count does.
        kw["tol"] = 0.0
        kw["n_iter_no_change"] = kw["max_iter"]
    if VARIANT in ("raw", "all"):
        norm = False

    est = MLPRegressor(**kw)
    _seen["width_fits"] += 1
    if _seen["width_fits"] == 1:
        print(f"    [u-trainer] variant={VARIANT} normalize={norm} "
              f"{ {k: v for k, v in kw.items() if k != 'random_state'} }", flush=True)

    if norm:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        pipe.fit(X, y)
        return pipe
    est.fit(X, y)
    return est


cmicl_mod.train_model = _patched_train_model

from experiments.probe_cmicl_cost_sampling import main  # noqa: E402

if __name__ == "__main__":
    main()
    if VARIANT != "baseline" and _seen["width_fits"] == 0:
        # The patch keys on [32,32]; if h were ever also [32,32] the detection
        # would be ambiguous and this would be the only warning of it.
        print("WARNING: no width fit was intercepted -- the variant did nothing",
              flush=True)

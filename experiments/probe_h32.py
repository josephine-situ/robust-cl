"""Pin h to a 32x32 MLP: does the width model flatten and feasibility return?

A DIAGNOSTIC. Monkeypatches this process only, then delegates to
``probe_cmicl_cost_sampling.main`` so the loop, MIP, calibration and cost draw
stay byte-identical to the runs this is the A/B against.

THE QUESTION. Under their own cost draw the split is not the protocol, not the
rows, and not the point predictor's accuracy AT the optimum:

    configuration                 feas  cov@x*   u@x* mean   u<0.1
    upstream code + their data    0.95    0.95       1.803      0%
    robust-cl + generated data    0.53    0.38       0.398     55%
    robust-cl + THEIR data        0.19    0.10       0.210     70%

Feasibility tracks coverage-at-x* tracks u@x*, monotonically. Handing this
pipeline their own rows made it WORSE, so the design distribution and the sample
are both excluded -- both designs are uniform over the same box.

WHAT IS LEFT is that `u` was matched to them and `h` was not. `u` is a 32x32 MLP
with a linear output (`config.yaml:341-342`), theirs exactly; `h` is CV-selected
and resolves to `mlp [10, 5, 2]` with lbfgs
(`results/cv/reactor_selected_configs.json`) against their 32x32 Keras net. `u`
trains on |residuals of h|, so h's capacity propagates into the width: an
underfit h leaves systematic, spatially varying error for u to model, while a
well-fit 32x32 h leaves residuals that are near-pure label noise -- a constant
about 1.6, which is nearly their flat u@x* of 1.803 (E|N(0,2)| ~ 1.60).

So pinning h to 32x32 predicts u flattens, u@x* rises toward ~1.6-1.8, the
fraction of optima at u < 0.10 collapses, and feasibility climbs. If it does
not, h's architecture is exonerated and the remaining difference is the
estimator itself (sklearn lbfgs MLPRegressor vs their Keras/Adam fit).

The params mirror the width model's, which is what this repo already uses to
mean "their 32x32": {hidden_layer_sizes: [32, 32], alpha: 0.01, max_iter: 2000}.

TWO CELLS, selected by env var so both share one script:
    CMICL_SHIPPED=0  h=32x32 on our generated design   (vs the 0.53 cell)
    CMICL_SHIPPED=1  h=32x32 on their shipped rows     (vs the 0.19 cell)

Usage:
    python -u experiments/probe_h32.py --schemes paper --out-suffix _h32
    CMICL_SHIPPED=1 python -u experiments/probe_h32.py --schemes paper --out-suffix _h32_shipped
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data.instances as instances

H32 = ("mlp", {"hidden_layer_sizes": [32, 32], "alpha": 0.01, "max_iter": 2000})

if os.environ.get("CMICL_SHIPPED", "0") == "1":
    # reuse the verified loader rather than re-deriving their split here
    import experiments.probe_shipped_data  # noqa: F401  (patches _reactor_dataset on import)
    _SRC = "their shipped rows"
else:
    _SRC = "our generated design"


def _pinned_spec(config, path=None, verbose=False):
    """Their h, not ours. ``from_cv=False`` -- this is pinned, not selected."""
    mt, mp = H32
    if verbose:
        print(f"    [reactor] h PINNED to {mt} {mp} (32x32 diagnostic, "
              f"overriding the CV selection); data: {_SRC}", flush=True)
    return mt, dict(mp), False


instances.reactor_model_spec = _pinned_spec

from experiments.probe_cmicl_cost_sampling import main  # noqa: E402

if __name__ == "__main__":
    main()

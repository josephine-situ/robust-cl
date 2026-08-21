"""Synthetic-problem model specs: the mixed-type GT ensemble used as the CV oracle.

The synthetic problem's *embedded* model is chosen by ``run_cv.py --problem
synthetic`` (``results/cv/synthetic_selected_configs.json``); this module holds the
other half -- the **judge**.

WHY A MIXED-TYPE ENSEMBLE. The robustness CV scores a prescription against a
train-only proxy oracle (``cv_calibrate.make_cv_oracle``). Until 2026-08-21 the
synthetic oracle was a single model of the *same class* the candidate embeds -- an
``rf`` judging an ``rf`` -- so oracle and candidate shared their approximation
error, and a decision that exploited an rf artifact was scored feasible by an rf
that had the same artifact. Averaging many model types is what gastric already does
(``gastric_model_specs.GT_ENSEMBLE_SPECS``, Table EC.12), so this puts the two
problems on the same convention.

WHY MLP IS A MEMBER ANYWAY (2026-08-21). The judge carries all SEVEN types,
including the class the candidate now embeds. Excluding it would buy formal
class-disjointness at the price of a judge that cannot follow an MLP candidate
into the region where that candidate is wrong -- and near a constrained optimum
that region IS the boundary, which is where the verdict is decided. Averaging
seven members dilutes any single member's artifact to 1/7; sharing zero classes
does not help if the remaining six systematically miss the failure. Gastric is the
one exception, and only because its ensemble is a replication of Table EC.12.

WHAT IT DOES NOT FIX. The ensemble is fit on the **noisy** ``y_train`` -- the
analytic ``f_true`` stays reserved for final evaluation (``evaluation/metrics.py``),
because an oracle that knows the data-generating process would score CP's
uncertainty set against the truth D was calibrated to. And it is fit on the FULL
training rows, a superset of any fold's train rows, exactly as gastric's oracle is;
the judge therefore shares rows with the candidate on both problems. That is a
property of the protocol, not of synthetic.

These are FALLBACK specs, deliberately deeper than the embedded grids (the judge
may be richer than the thing it judges). ``run_cv.py --problem synthetic
--ensemble`` writes CV-tuned ones to
``results/cv/synthetic_gt_ensemble_configs.json``, which takes precedence.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Matches run_cv.py's default --seed, and gastric's GASTRIC_ML_SEED.
SYNTHETIC_ML_SEED = 1

# Order mirrors GT_MODEL_ORDER in run_cv.py: linear, svm, cart, rf, gbm, xgb, mlp.
# Tuned by hand for n=200, d=2, sigma=0.1 over [0,1]^2 -- rich enough to track a
# smooth quadratic, not so rich as to interpolate the noise.
_SYNTH_GT_RAW: List[Dict[str, Any]] = [
    {"model_type": "linear", "params": {"alpha": 0.01, "l1_ratio": 0.5}},
    {"model_type": "svm", "params": {"C": 1.0, "epsilon": 0.05}},
    {"model_type": "cart", "params": {
        "max_depth": 6, "min_samples_leaf": 0.02, "max_features": 1.0,
    }},
    {"model_type": "rf", "params": {
        "n_estimators": 250, "max_depth": 8, "max_features": 1.0,
    }},
    {"model_type": "gbm", "params": {
        "learning_rate": 0.05, "max_depth": 3, "n_estimators": 250,
    }},
    {"model_type": "xgb", "params": {
        "learning_rate": 0.05, "max_depth": 3, "n_estimators": 250,
        "subsample": 0.9, "colsample_bytree": 1.0,
    }},
    # Wider and two-layer where the embedded candidate is (50,): the judge shares
    # the candidate's model CLASS here, deliberately, but not its architecture.
    {"model_type": "mlp", "params": {
        "hidden_layer_sizes": (50, 25), "solver": "lbfgs", "alpha": 1e-3,
    }},
]

SYNTHETIC_GT_ENSEMBLE_SPECS: List[Dict[str, Any]] = [
    {"model_type": s["model_type"],
     "params": {**s["params"], "random_state": SYNTHETIC_ML_SEED}}
    for s in _SYNTH_GT_RAW
]

# The JSON key run_cv.py writes for the single synthetic outcome.
SYNTHETIC_OUTCOME = "synthetic_constraint"

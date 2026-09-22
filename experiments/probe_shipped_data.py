"""Does fitting on Ovalle et al.'s SHIPPED rows restore C-MICL's feasibility? (reactor)

A DIAGNOSTIC, not part of any evaluation. It changes nothing in production: it
monkeypatches ``_reactor_dataset`` for this process only and then delegates to
``probe_cmicl_cost_sampling.main`` so the loop, the MIP, the calibration and the
cost draw are byte-identical to the run this is the A/B against.

THE QUESTION. Under their own cost draw (``--schemes paper``) the two
implementations split: upstream c-micl reaches ODE feasibility 0.95 with
coverage-at-x* 0.95, this repo reaches 0.53 with coverage-at-x* 0.38. The point
predictors are equally good AT the optimum (mean |h - f_true| 2.00 vs 1.87), so
the split is not h. It is the WIDTH model: upstream's u@x* sits in
[1.762, 1.865] across 23 distinct optima -- effectively constant, an
unexploitable fixed margin -- while ours runs [0.000, 2.989] with mean 0.398,
4.1x below its own calibration mean of 1.635, and 55/100 optima land at u < 0.10
with the minimum pinned at exactly 0 by the MIP's ``u >= 0``. The optimizer
drives x* to where the width model predicts no uncertainty, which collapses
``h - q*u >= 50`` to ``h >= 50``.

WHAT THIS ISOLATES. Both designs are uniform over the SAME box (their sheet's
per-column min/max match DECISION_RANGES), so the design *distribution* is not
the difference -- only the realized rows, their noise draw, and everything
downstream of them. Holding the pipeline fixed and swapping only the rows
therefore splits the remaining hypotheses cleanly:

    feasibility ~= 0.95 here -> the ROWS explain it (our sample, not our code)
    feasibility ~= 0.53 here -> the PIPELINE explains it (our h/u fit), and the
                                next suspect is h's architecture: theirs is a
                                fixed 32x32 ReLU MLP, ours is CV-selected by
                                ``reactor_model_spec``, and u is trained on
                                |residuals of h|, so a different h moves u.

THEIR ROWS, EXACTLY. ``regression.py:604`` reads
``data/unscaled_noisy_reactor_data.xlsx`` (2000 rows, raw units, columns
``v0,v_He,T,dt,L,F_C6H6``) and ``:620`` throws half of it away as ``X_unseen``,
which is never read again anywhere in that file. The surviving half is what h
and the calibration split are built from, so that is what is handed over here.
Their ``/100`` and ``/10`` scaling is NOT applied: this repo works in raw units
and ``calibrate_conformal_model`` does its own scaling.

``y_clean`` is not in their sheet -- it ships only noisy labels -- so it is
integrated here with THIS repo's ODE and cached. It feeds ``y_true`` on the
instance and no part of the fit; ground-truth feasibility is ``benzene_flow``
at x* either way, so nothing downstream depends on it.

Usage:
    python -u experiments/probe_shipped_data.py --schemes paper --out-suffix _shipped
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data.generate as gen
from src.data.dma_mr import benzene_flow_batch

SHIPPED = os.path.expanduser(
    os.environ.get("CMICL_SHEET", "~/c-micl/data/unscaled_noisy_reactor_data.xlsx"))
COLS = ["v0", "v_He", "T", "dt", "L"]
TARGET = "F_C6H6"

_CACHE = {}


def _shipped_dataset(n_train, noise_std, seed, cache_dir):
    """``(U, y_clean, y_noisy)`` from their sheet, with this repo's contract.

    ``n_train``/``noise_std``/``seed`` are accepted and IGNORED -- the rows and
    their noise are theirs, already realized. Ignoring the seed is the point of
    the diagnostic, not an oversight: varying it would put us back on our own
    sample.
    """
    if "d" not in _CACHE:
        if not os.path.exists(SHIPPED):
            raise FileNotFoundError(
                f"upstream sheet not found at {SHIPPED}; set CMICL_SHEET")
        df = pd.read_excel(SHIPPED)
        missing = [c for c in COLS + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"{SHIPPED} is missing columns {missing}")
        U_all = df[COLS].to_numpy(dtype=float)
        y_all = df[TARGET].to_numpy(dtype=float)
        # regression.py:620 -- the FIRST half is X_unseen and is dead; keep the second.
        _, U, _, y_noisy = train_test_split(
            U_all, y_all, test_size=0.5, random_state=42)

        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, "shipped_clean.npz")
        if os.path.exists(cache):
            y_clean = np.load(cache)["y"]
        else:
            y_clean = benzene_flow_batch(U)          # ~160 ms/row, once
            np.savez(cache, y=y_clean)
        ok = np.isfinite(y_clean)
        U, y_clean, y_noisy = U[ok], y_clean[ok], y_noisy[ok]
        print(f"[shipped] {len(U)} rows from {os.path.basename(SHIPPED)} "
              f"(their surviving half); label resid sd vs our ODE = "
              f"{np.std(y_noisy - y_clean):.3f}", flush=True)
        _CACHE["d"] = (U, y_clean, y_noisy)
    return _CACHE["d"]


gen._reactor_dataset = _shipped_dataset

from experiments.probe_cmicl_cost_sampling import main  # noqa: E402  (after the patch)

if __name__ == "__main__":
    main()

"""
Calibrate a baseline method's robustness to a shared coverage target on the
training set.

Each non-CP baseline exposes a single monotone robustness knob. We pick the
*strongest* setting whose fraction of infeasible training contexts is at most
``alpha`` - the same coverage semantics CP enforces directly via its ``p_infeas``
cap - so every method is compared at a matched level of training-set robustness.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.data.generate import ProblemInstance
from src.evaluation.chemo_metrics import solve_for_context


def infeasible_fraction(solver_fn: Callable, instance: ProblemInstance,
                        contexts: np.ndarray) -> float:
    """Build the MIP once via ``solver_fn(instance)`` and return the fraction of
    ``contexts`` (rows whose context columns are pinned) that solve infeasible."""
    result = solver_fn(instance)
    if isinstance(result, tuple):
        result = result[0]
    n_infeas = 0
    for row in contexts:
        _, x_opt = solve_for_context(result, instance, row)
        if x_opt is None:
            n_infeas += 1
    return n_infeas / max(1, len(contexts))


def calibrate_strength(build_solver: Callable[[float], Callable],
                       strength_to_knob: Callable[[float], float],
                       instance: ProblemInstance,
                       contexts: np.ndarray,
                       target_alpha: float,
                       n_grid: int = 5,
                       label: str = "") -> tuple[float, float]:
    """Grid-scan robustness strength from strongest (1.0) to weakest (0.0).

    ``strength_to_knob`` maps a strength in [0, 1] (1 = most conservative) to the
    method's actual knob value. Returns ``(knob, infeasible_fraction)`` for the
    strongest setting whose infeasible fraction is within ``target_alpha``; if no
    setting qualifies, returns the weakest setting with a warning.
    """
    n_grid = max(2, int(n_grid))
    strengths = [i / (n_grid - 1) for i in range(n_grid - 1, -1, -1)]  # 1.0 -> 0.0
    weakest = None
    for s in strengths:
        knob = strength_to_knob(s)
        frac = infeasible_fraction(build_solver(knob), instance, contexts)
        print(
            f"    [calib] {label}: strength={s:.2f} knob={knob:.4f} "
            f"train_infeasible={frac * 100:.1f}%",
            flush=True,
        )
        weakest = (knob, frac)
        if frac <= target_alpha + 1e-9:
            return knob, frac
    print(
        f"    [calib] {label}: WARNING no setting meets alpha "
        f"{target_alpha * 100:.1f}%; using weakest knob={weakest[0]:.4f} "
        f"(train_infeasible={weakest[1] * 100:.1f}%)",
        flush=True,
    )
    return weakest

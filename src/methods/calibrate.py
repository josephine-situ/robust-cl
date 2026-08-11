"""
Calibrate a baseline method's robustness to a shared coverage target on the
training set.

Each non-CP baseline exposes a single monotone robustness knob. We pick the
*strongest* setting whose fraction of infeasible training contexts is at most
``alpha``, so every method is compared at a matched level of training-set
robustness.

This is the **legacy** calibration path: ``calibration.method`` defaults to
``"cv"`` (:mod:`src.methods.cv_calibrate`), and this module runs only under
``calibration.method: "alpha"``. CP is exempt and never reads
``uncertainty.alpha`` -- its ``cp_alpha`` is pinned at 0 at every call site, so
cuts that would break a protected anchor are rolled back rather than budgeted.

Calibration bounds (weakest at strength=0, strongest at strength=1) are set in
``config.yaml``: ``wrapper_alpha_max`` / ``tree_alpha_max`` for chance-constraint
methods (strongest alpha=0), and ``rho_min`` / ``rho_max`` for robust_param (never
scans rho=0; fallback on failure is ``rho_min``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.data.generate import ProblemInstance
from src.evaluation.chemo_metrics import solve_for_test_cohort


def infeasible_fraction(solver_fn: Callable, instance: ProblemInstance,
                        contexts: np.ndarray,
                        alpha_budget: float | None = None,
                        priority_indices: list[int] | None = None,
                        ) -> tuple[float, list[int]]:
    """Build the MIP once via ``solver_fn(instance)`` and return
    ``(fraction_infeasible, infeasible_indices)``.

    ``priority_indices`` — context row indices to evaluate first.  Solving
    known-infeasible contexts early maximises the chance of triggering an early
    exit when ``alpha_budget`` is set (same ordering idea as
    ``_p_infeas_after_cuts`` in CP).

    If ``alpha_budget`` is given, stop as soon as the infeasible count exceeds
    ``alpha_budget * n_total``; the returned fraction is then a lower bound
    sufficient for rejection.  ``infeasible_indices`` covers only the contexts
    evaluated before the early exit.
    """
    result = solver_fn(instance)
    if isinstance(result, tuple):
        result = result[0]
    n_total = max(1, len(contexts))

    # Build evaluation order: priority indices first, then the rest.
    if priority_indices:
        priority_set = set(priority_indices)
        order = list(priority_indices) + [i for i in range(len(contexts)) if i not in priority_set]
    else:
        order = list(range(len(contexts)))

    n_infeas = 0
    infeasible_indices: list[int] = []
    for i in order:
        _, x_opt = solve_for_test_cohort(result, instance, contexts[i])
        if x_opt is None:
            n_infeas += 1
            infeasible_indices.append(i)
            if alpha_budget is not None and n_infeas > (alpha_budget + 1e-12) * n_total:
                return n_infeas / n_total, infeasible_indices
    return n_infeas / n_total, infeasible_indices


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
    setting qualifies, returns the weakest setting (e.g. ``rho_min`` for
    robust_param, never nominal rho=0) with a warning.
    """
    n_grid = max(2, int(n_grid))
    strengths = [i / (n_grid - 1) for i in range(n_grid - 1, -1, -1)]  # 1.0 -> 0.0
    weakest = None
    # Contexts infeasible at the previous (stronger) setting are solved first at
    # the next step; they are likely still infeasible and trigger early exit fast.
    priority: list[int] = []
    for i, s in enumerate(strengths):
        knob = strength_to_knob(s)
        # For the last (weakest) candidate skip the budget to get the exact
        # fraction needed for the fallback warning message.
        is_last = (i == len(strengths) - 1)
        frac, infeas_idx = infeasible_fraction(
            build_solver(knob), instance, contexts,
            alpha_budget=None if is_last else target_alpha,
            priority_indices=priority,
        )
        print(
            f"    [calib] {label}: strength={s:.2f} knob={knob:.4f} "
            f"train_infeasible={frac * 100:.1f}%",
            flush=True,
        )
        weakest = (knob, frac)
        if frac <= target_alpha + 1e-9:
            return knob, frac
        # Carry forward infeasible indices as priority for the next grid point.
        priority = infeas_idx
    print(
        f"    [calib] {label}: WARNING no setting meets alpha "
        f"{target_alpha * 100:.1f}%; using weakest knob={weakest[0]:.4f} "
        f"(train_infeasible={weakest[1] * 100:.1f}%)",
        flush=True,
    )
    return weakest

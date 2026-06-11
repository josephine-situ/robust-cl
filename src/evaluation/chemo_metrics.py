"""
Paper-aligned evaluation for OptiCL chemotherapy Table 6.

Metrics match Maragno et al. (2025) Section 5.5:
- Constraint satisfaction: binary indicator GT(x) <= threshold
- Overall survival: GT ensemble prediction in months
- Prescribed results averaged over test cohorts with an optimal optimizer solution
- Given baseline averaged over the full test set
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from src.data.generate import ProblemInstance


@dataclass
class ChemoTable6Result:
    outcome: str
    constraint_mode: str
    given_mean: float
    given_sd: float
    prescribed_mean: float
    prescribed_sd: float
    pct_change: float
    n_test: int
    n_prescribed_eval: int
    mean_solve_time: float
    solve_time_sd: float


def _predict_outcome(gt_fn, x: np.ndarray) -> float:
    val = gt_fn.predict(np.atleast_2d(x))
    if isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val)


def _outcome_values(x_rows: np.ndarray, outcome, instance: ProblemInstance) -> np.ndarray:
    values = np.array([_predict_outcome(outcome.gt_fn, x) for x in x_rows], dtype=float)
    if outcome.is_survival:
        return values
    return (values <= outcome.rhs).astype(float)


def evaluate_given_table6(instance: ProblemInstance) -> Dict[str, np.ndarray]:
    """Given-treatment outcomes on the full test set."""
    X_test = instance.X_test
    results = {}
    for outcome in instance.eval_outcomes:
        results[outcome.label] = _outcome_values(X_test, outcome, instance)
    return results


def evaluate_prescribed_table6(
    solver_fn: Callable,
    instance: ProblemInstance,
    **solver_kwargs,
) -> tuple[Dict[str, np.ndarray], np.ndarray, float, float]:
    """
    Optimize a prescription per test cohort; return outcome vectors on feasible cohorts.

    Returns
    -------
    outcomes : dict outcome_label -> values on feasible test indices
    feasible_mask : bool array length n_test
    mean_solve_time, solve_time_sd : per-cohort re-optimization times (seconds)
    """
    result = solver_fn(instance, **solver_kwargs)
    if isinstance(result, tuple):
        result = result[0]

    n_test = instance.X_test.shape[0]
    feasible_mask = np.zeros(n_test, dtype=bool)
    row_times: List[float] = []

    outcome_buffers = {o.label: np.full(n_test, np.nan) for o in instance.eval_outcomes}

    for i in range(n_test):
        for c_idx in instance.context_var_indices:
            result.x[c_idx].lb = instance.variable_lb[c_idx]
            result.x[c_idx].ub = instance.variable_ub[c_idx]
        for c_idx in instance.context_var_indices:
            val = float(instance.X_test[i, c_idx])
            result.x[c_idx].lb = val
            result.x[c_idx].ub = val

        result.opt.Params.DualReductions = 0
        result.opt.Params.MIPGap = 0.01
        result.opt.update()

        t0 = time.time()
        result.opt.optimize()
        row_times.append(time.time() - t0)

        if n_test > 1 and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n_test):
            print(f"  prescribed: test row {i + 1}/{n_test}", flush=True)

        if result.opt.Status != 2:
            continue

        feasible_mask[i] = True
        x_opt = np.array([v.X for v in result.x])
        for outcome in instance.eval_outcomes:
            if outcome.is_survival:
                outcome_buffers[outcome.label][i] = _predict_outcome(outcome.gt_fn, x_opt)
            else:
                val = _predict_outcome(outcome.gt_fn, x_opt)
                outcome_buffers[outcome.label][i] = float(val <= outcome.rhs)

    mean_time = float(np.mean(row_times)) if row_times else np.nan
    sd_time = float(np.std(row_times, ddof=1)) if len(row_times) > 1 else 0.0

    feasible_outcomes = {
        label: values[feasible_mask]
        for label, values in outcome_buffers.items()
    }
    return feasible_outcomes, feasible_mask, mean_time, sd_time


def build_table6_rows(
    instance: ProblemInstance,
    constraint_mode: str,
    given_values: Dict[str, np.ndarray],
    prescribed_values: Dict[str, np.ndarray],
    n_test: int,
    n_prescribed: int,
    mean_solve_time: float,
    solve_time_sd: float,
) -> List[ChemoTable6Result]:
    rows = []
    for outcome in instance.eval_outcomes:
        given_arr = given_values[outcome.label]
        prescribed_arr = prescribed_values[outcome.label]

        given_mean = float(np.mean(given_arr))
        given_sd = float(np.std(given_arr, ddof=1)) if len(given_arr) > 1 else 0.0
        prescribed_mean = float(np.mean(prescribed_arr)) if len(prescribed_arr) else np.nan
        prescribed_sd = (
            float(np.std(prescribed_arr, ddof=1)) if len(prescribed_arr) > 1 else 0.0
        )
        if given_mean != 0 and np.isfinite(prescribed_mean):
            pct_change = 100.0 * (prescribed_mean - given_mean) / given_mean
        else:
            pct_change = np.nan

        rows.append(ChemoTable6Result(
            outcome=outcome.label,
            constraint_mode=constraint_mode,
            given_mean=given_mean,
            given_sd=given_sd,
            prescribed_mean=prescribed_mean,
            prescribed_sd=prescribed_sd,
            pct_change=pct_change,
            n_test=n_test,
            n_prescribed_eval=n_prescribed,
            mean_solve_time=mean_solve_time,
            solve_time_sd=solve_time_sd,
        ))
    return rows


def table6_results_to_dataframe(rows: List[ChemoTable6Result]):
    import pandas as pd

    return pd.DataFrame([row.__dict__ for row in rows])

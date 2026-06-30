"""
Paper-aligned evaluation for OptiCL chemotherapy Table 6.

Metrics match Maragno et al. (2025) Section 5.5:
- All reported given/prescribed outcomes use the **ground-truth (GT) ensemble**
  (6 sklearn models per outcome, Table EC.12), not the embedded MIP models.
- Toxicity satisfaction: binary indicator GT(x) <= 0.6 on percentile-scale predictions
- Overall survival: GT ensemble prediction in months
- Prescribed and given results averaged over a shared samestore cohort
  (feasible under both all-constraints and DLT-only prescriptions)
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

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


def samestore_eval_mask(*feasible_masks: np.ndarray) -> np.ndarray:
    """Intersection of per-mode optimizer feasibility (constraint-learning samestore)."""
    if not feasible_masks:
        raise ValueError("At least one feasibility mask is required")
    out = feasible_masks[0].astype(bool).copy()
    for mask in feasible_masks[1:]:
        out &= mask.astype(bool)
    return out


def subset_table6_outcomes(
    full_outcomes: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {label: values[mask] for label, values in full_outcomes.items()}


def _feature_column_names(instance: ProblemInstance) -> List[str]:
    if instance.feature_names and len(instance.feature_names) == instance.n_features:
        return list(instance.feature_names)
    return [f"x_{j}" for j in range(instance.n_features)]


def save_prescriptions_csv(
    path: str,
    instance: ProblemInstance,
    prescriptions: np.ndarray,
    feasible_mask: np.ndarray,
    solve_times: np.ndarray,
    *,
    method: str | None = None,
    constraint_mode: str | None = None,
) -> str:
    """Write per-test prescription vectors to CSV.

    ``prescriptions`` is ``(n_test, n_features)`` with ``nan`` rows for infeasible
    solves. Returns the path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    feature_cols = _feature_column_names(instance)
    header = ["test_idx", "feasible", "solve_time_s"]
    if method is not None:
        header.append("method")
    if constraint_mode is not None:
        header.append("constraint_mode")
    header.extend(feature_cols)

    n_test = prescriptions.shape[0]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_test):
            row = [
                i,
                int(bool(feasible_mask[i])),
                "" if not np.isfinite(solve_times[i]) else float(solve_times[i]),
            ]
            if method is not None:
                row.append(method)
            if constraint_mode is not None:
                row.append(constraint_mode)
            x_row = prescriptions[i]
            if np.all(np.isnan(x_row)):
                row.extend([""] * len(feature_cols))
            else:
                row.extend(float(v) for v in x_row)
            writer.writerow(row)
    return path


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


def solve_for_context(result, instance: ProblemInstance, context_row: np.ndarray):
    """Pin a built MIP's context columns to ``context_row`` and re-solve.

    Returns ``(status, x_opt)`` with ``x_opt`` the optimized decision vector when
    the solve is optimal (Gurobi status 2), else ``None``. Shared by the
    prescription evaluation and the baseline calibration so both measure
    feasibility the same way.
    """
    for c_idx in instance.context_var_indices:
        val = float(context_row[c_idx])
        result.x[c_idx].lb = val
        result.x[c_idx].ub = val
    result.opt.Params.DualReductions = 0
    result.opt.Params.MIPGap = 1e-4
    result.opt.update()
    result.opt.optimize()
    status = result.opt.Status
    if status == 2:
        return status, np.array([v.X for v in result.x])
    return status, None


def evaluate_given_table6(
    instance: ProblemInstance,
    feasible_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Given-treatment outcomes on observed regimens for the feasible test cohort."""
    X_test = instance.X_test[feasible_mask]
    return {
        outcome.label: _outcome_values(X_test, outcome, instance)
        for outcome in instance.eval_outcomes
    }


def evaluate_prescribed_table6(
    solver_fn: Callable,
    instance: ProblemInstance,
    eval_mask: np.ndarray | None = None,
    max_test_rows: int | None = None,
    method_name: str | None = None,
    constraint_mode: str | None = None,
    prescriptions_dir: str | None = None,
    **solver_kwargs,
) -> tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    float,
    float,
    Dict[str, np.ndarray],
    np.ndarray,
]:
    """
    Optimize a prescription per test cohort; return outcome vectors on feasible cohorts.

    If eval_mask is provided, returned outcome vectors are restricted to
    test indices where both the optimizer is feasible and eval_mask is True.
    The fifth return value is the full-length (n_test) outcome vectors for
    building a samestore cohort across constraint modes.
    The sixth return value is the full-length (n_test, n_features) prescription
    matrix (``nan`` rows when infeasible).

    If ``prescriptions_dir`` is set, writes
    ``{prescriptions_dir}/{method}_{constraint_mode}.csv`` after the prescribe pass.

    Returns
    -------
    outcomes : dict outcome_label -> values on reported test indices
    feasible_mask : bool array length n_test (optimizer feasibility per row)
    mean_solve_time, solve_time_sd : per-cohort re-optimization times (seconds)
    full_outcomes : dict outcome_label -> length n_test arrays (nan if infeasible)
    full_prescriptions : (n_test, n_features) prescription vectors
    """
    label = method_name or getattr(solver_fn, "func", solver_fn).__name__
    if constraint_mode:
        label = f"{label}/{constraint_mode}"

    print(f"  [{label}] Building initial MIP (train + embed)...", flush=True)
    t_build = time.time()
    result = solver_fn(instance, **solver_kwargs)
    if isinstance(result, tuple):
        result = result[0]
    build_time = time.time() - t_build
    print(
        f"  [{label}] Initial solve done in {build_time:.1f}s "
        f"(status={result.status}, models_embedded={result.models_embedded})",
        flush=True,
    )

    n_test = instance.X_test.shape[0]
    n_eval_rows = n_test if max_test_rows is None else min(max_test_rows, n_test)
    feasible_mask = np.zeros(n_test, dtype=bool)
    row_times: List[float] = []

    outcome_buffers = {o.label: np.full(n_test, np.nan) for o in instance.eval_outcomes}
    full_prescriptions = np.full((n_test, instance.n_features), np.nan)
    row_solve_times = np.full(n_test, np.nan)

    print(f"  [{label}] Prescribing per test cohort ({n_eval_rows} rows)...", flush=True)

    for i in range(n_eval_rows):
        print(f"  [{label}] test row {i + 1}/{n_eval_rows}: optimizing...", flush=True)
        t0 = time.time()
        status, x_opt = solve_for_context(result, instance, instance.X_test[i])
        row_elapsed = time.time() - t0
        row_times.append(row_elapsed)
        row_solve_times[i] = row_elapsed

        if status == 2:
            status_str = "optimal"
        elif status == 3:
            status_str = "infeasible"
        elif status == 9:
            status_str = "time_limit"
        else:
            status_str = f"status={status}"
        print(
            f"  [{label}] test row {i + 1}/{n_eval_rows}: "
            f"{status_str} in {row_elapsed:.1f}s",
            flush=True,
        )

        if x_opt is None:
            continue

        feasible_mask[i] = True
        full_prescriptions[i] = x_opt
        for outcome in instance.eval_outcomes:
            if outcome.is_survival:
                outcome_buffers[outcome.label][i] = _predict_outcome(outcome.gt_fn, x_opt)
            else:
                val = _predict_outcome(outcome.gt_fn, x_opt)
                outcome_buffers[outcome.label][i] = float(val <= outcome.rhs)

    mean_time = float(np.mean(row_times)) if row_times else np.nan
    sd_time = float(np.std(row_times, ddof=1)) if len(row_times) > 1 else 0.0
    n_feasible = int(feasible_mask[:n_eval_rows].sum())
    print(
        f"  [{label}] Prescription pass complete: "
        f"{n_feasible}/{n_eval_rows} feasible, "
        f"mean re-solve {mean_time:.2f}s",
        flush=True,
    )

    if prescriptions_dir and method_name and constraint_mode:
        rx_path = os.path.join(
            prescriptions_dir, f"{method_name}_{constraint_mode}.csv"
        )
        save_prescriptions_csv(
            rx_path,
            instance,
            full_prescriptions,
            feasible_mask,
            row_solve_times,
            method=method_name,
            constraint_mode=constraint_mode,
        )
        print(f"  [{label}] Prescriptions saved to {rx_path}", flush=True)

    report_mask = feasible_mask if eval_mask is None else (feasible_mask & eval_mask)
    feasible_outcomes = {
        label: values[report_mask]
        for label, values in outcome_buffers.items()
    }
    return feasible_outcomes, feasible_mask, mean_time, sd_time, outcome_buffers, full_prescriptions


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

    # Conjunction row: fraction of regimens feasible for *all* (toxicity)
    # constraints simultaneously, i.e. the per-regimen AND of the binary GT
    # satisfaction indicators above. Survival outcomes are continuous, not a
    # feasibility indicator, so they are excluded.
    binary_labels = [o.label for o in instance.eval_outcomes if not o.is_survival]
    if binary_labels:
        def _all_feasible(values: Dict[str, np.ndarray]) -> np.ndarray:
            cols = [np.asarray(values[label], dtype=float) for label in binary_labels]
            if not cols or len(cols[0]) == 0:
                return np.array([])
            # Rows are already restricted to the feasible samestore cohort, so
            # every indicator is a finite 0/1.
            return (np.vstack(cols) >= 0.5).all(axis=0).astype(float)

        given_all = _all_feasible(given_values)
        prescribed_all = _all_feasible(prescribed_values)

        given_mean = float(np.mean(given_all)) if len(given_all) else np.nan
        given_sd = float(np.std(given_all, ddof=1)) if len(given_all) > 1 else 0.0
        prescribed_mean = float(np.mean(prescribed_all)) if len(prescribed_all) else np.nan
        prescribed_sd = (
            float(np.std(prescribed_all, ddof=1)) if len(prescribed_all) > 1 else 0.0
        )
        if np.isfinite(given_mean) and given_mean != 0 and np.isfinite(prescribed_mean):
            pct_change = 100.0 * (prescribed_mean - given_mean) / given_mean
        else:
            pct_change = np.nan

        rows.append(ChemoTable6Result(
            outcome="all_constraints",
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

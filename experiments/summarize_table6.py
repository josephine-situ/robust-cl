#!/usr/bin/env python3
"""
Post-processing script: summarize chemo_robust_table6.csv into
presentation-ready tables (.csv and .tex).

Usage:
    uv run python experiments/summarize_table6.py
    uv run python experiments/summarize_table6.py --input results/chemo_robust_table6.csv

Outputs (in results/):
    summary_all_constraints.csv / .tex
    summary_dlt_only.csv / .tex

Table structure
---------------
all_constraints mode
    columns: Method | Joint Toxicity mean (SD) | % Improv | OS mean (SD) | % Improv | Solve Time
dlt_only mode
    columns: Method | Any DLT mean (SD) | % Improv | OS mean (SD) | % Improv | Solve Time

In the .tex output:
  - means/SDs for binary outcomes are expressed as percentages (×100)
  - Mean and SD are combined into a single "mean (SD)" column
  - The best value in each numeric column (among methods, not Given) is bolded
  - Requires the LaTeX packages: booktabs

Samestore note
--------------
The input CSV stores n_prescribed_eval, which reflects the cohort used during
evaluation. If run_chemo_robust.py was updated to compute per-mode samestores
(see the companion change in that file), each mode's table will reflect patients
feasible across all methods for that mode only.  With the old global samestore
both tables share the same n_prescribed_eval cohort.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Path("results/chemo_robust_table6.csv")
OUT_DIR = Path("results")

# Display order for methods (nominal listed first among methods as a baseline)
METHOD_ORDER = ["nominal", "robust_param", "robust_reg", "tree_violation", "wrapper", "cp"]

METHOD_LABELS: dict[str, str] = {
    "nominal": "Nominal",
    "robust_param": "Robust (Param.)",
    "robust_reg": "Robust (Reg.)",
    "tree_violation": "Tree Violation",
    "wrapper": "Wrapper",
    "cp": "CP",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAN = float("nan")


def _isnan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return True


def _fmt_pct_val(val: float, mean_decimals: int = 1, sd_decimals: int = 1, scale: float = 100.0) -> tuple[str, str]:
    """Return (mean_str, sd_str) scaled to percentage."""
    m = f"{val * scale:.{mean_decimals}f}"
    return m


def fmt_mean_sd_pct(mean: float, sd: float, decimals: int = 1) -> str:
    """'XX.X (YY.Y)' for a fraction expressed as %, or '---' if missing."""
    if _isnan(mean):
        return "---"
    m = f"{mean * 100:.{decimals}f}"
    if _isnan(sd):
        return m
    s = f"{sd * 100:.{decimals}f}"
    return f"{m} ({s})"


def fmt_mean_sd_raw(mean: float, sd: float, decimals: int = 2) -> str:
    """'XX.XX (YY.YY)' for a raw value (e.g. months), or '---' if missing."""
    if _isnan(mean):
        return "---"
    m = f"{mean:.{decimals}f}"
    if _isnan(sd):
        return m
    s = f"{sd:.{decimals}f}"
    return f"{m} ({s})"


def fmt_pct_change(val: float, decimals: int = 1) -> str:
    """'+X.X\\%' or '---' if missing."""
    if _isnan(val):
        return "---"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{decimals}f}\\%"


def fmt_time(val: float, decimals: int = 3) -> str:
    if _isnan(val):
        return "---"
    return f"{val:.{decimals}f}"


def bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


# ---------------------------------------------------------------------------
# Build internal summary DataFrame
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame, constraint_mode: str, tox_outcome: str) -> pd.DataFrame:
    """
    Build a per-method summary for one constraint mode.

    Returns a DataFrame with columns:
        method_key, method_label, tox_mean, tox_sd, tox_pct,
        os_mean, os_sd, os_pct, solve_time, solve_time_sd, n_eval
    The first row is always 'Given' (observed treatment).
    """
    sub = df[df["constraint_mode"] == constraint_mode]
    if sub.empty:
        raise ValueError(f"No rows found for constraint_mode={constraint_mode!r}")

    rows: list[dict] = []

    # ---- Given row -------------------------------------------------------
    # given_mean/given_sd are identical across all methods; take from first method.
    tox_ref = sub[sub["outcome"] == tox_outcome]
    os_ref = sub[sub["outcome"] == "Overall_Survival"]
    if tox_ref.empty:
        raise ValueError(f"Outcome {tox_outcome!r} not found in constraint_mode={constraint_mode!r}")
    if os_ref.empty:
        raise ValueError(f"Outcome 'Overall_Survival' not found in constraint_mode={constraint_mode!r}")

    # Use the samestore cohort size as a sanity check (same for all methods in the mode)
    n_eval = int(tox_ref.iloc[0]["n_prescribed_eval"])

    rows.append(dict(
        method_key="given",
        method_label="Given",
        tox_mean=tox_ref.iloc[0]["given_mean"],
        tox_sd=tox_ref.iloc[0]["given_sd"],
        tox_pct=_NAN,
        os_mean=os_ref.iloc[0]["given_mean"],
        os_sd=os_ref.iloc[0]["given_sd"],
        os_pct=_NAN,
        solve_time=_NAN,
        solve_time_sd=_NAN,
        n_eval=n_eval,
    ))

    # ---- Method rows -----------------------------------------------------
    for method in METHOD_ORDER:
        mdf = sub[sub["method"] == method]
        if mdf.empty:
            continue
        tox_row = mdf[mdf["outcome"] == tox_outcome]
        os_row = mdf[mdf["outcome"] == "Overall_Survival"]
        if tox_row.empty or os_row.empty:
            continue
        tox_row = tox_row.iloc[0]
        os_row = os_row.iloc[0]

        rows.append(dict(
            method_key=method,
            method_label=METHOD_LABELS.get(method, method),
            tox_mean=tox_row["prescribed_mean"],
            tox_sd=tox_row["prescribed_sd"],
            tox_pct=tox_row["pct_change"],
            os_mean=os_row["prescribed_mean"],
            os_sd=os_row["prescribed_sd"],
            os_pct=os_row["pct_change"],
            solve_time=tox_row["mean_solve_time"],
            solve_time_sd=tox_row["solve_time_sd"],
            n_eval=int(tox_row["n_prescribed_eval"]),
        ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(summary: pd.DataFrame, tox_label: str, path: Path) -> None:
    out = pd.DataFrame()
    out["Method"] = summary["method_label"]
    out[f"{tox_label} Mean (%)"] = (summary["tox_mean"] * 100).round(1)
    out[f"{tox_label} SD (%)"] = (summary["tox_sd"] * 100).round(1)
    out[f"{tox_label} % Improv"] = summary["tox_pct"].round(1)
    out["OS Mean (mo.)"] = summary["os_mean"].round(2)
    out["OS SD (mo.)"] = summary["os_sd"].round(2)
    out["OS % Improv"] = summary["os_pct"].round(1)
    out["Solve Time (s)"] = summary["solve_time"].round(3)
    out["N Eval"] = summary["n_eval"]
    out.to_csv(path, index=False)
    print(f"  Saved CSV  -> {path}")


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def write_tex(
    summary: pd.DataFrame,
    tox_label: str,
    path: Path,
    caption: str = "",
    label: str = "",
) -> None:
    """
    Write a booktabs LaTeX table.

    Columns: Method | tox mean (SD) | tox % improv | OS mean (SD) | OS % improv | Solve Time
    Best value per column (among method rows only) is bolded.
    Requires: \\usepackage{booktabs}
    """
    method_mask = summary["method_key"] != "given"
    method_rows = summary[method_mask]

    def _best_idxs(col: str, higher_better: bool) -> set:
        """Set of indices tied for the best non-NaN value among method rows."""
        vals = pd.to_numeric(method_rows[col].copy(), errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            return set()
        best_val = valid.max() if higher_better else valid.min()
        return set(valid[valid == best_val].index)

    best = {
        "tox_mean":   _best_idxs("tox_mean", True),
        "tox_pct":    _best_idxs("tox_pct",  True),
        "os_mean":    _best_idxs("os_mean",  True),
        "os_pct":     _best_idxs("os_pct",   True),
        "solve_time": _best_idxs("solve_time", False),
    }

    n_eval_common = summary["n_eval"].iloc[0]  # same for all rows (samestore cohort)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    if caption:
        cap_str = caption
        if n_eval_common:
            cap_str += f" Evaluated on $n={n_eval_common}$ samestore patients."
        lines.append(f"\\caption{{{cap_str}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    # 6 columns: method + 2×(mean(SD) + %improv) + solve time
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")

    # Header row 1: group labels
    lines.append(
        f"& \\multicolumn{{2}}{{c}}{{{tox_label} (\\%)}}"
        f" & \\multicolumn{{2}}{{c}}{{OS (mo.)}}"
        f" & \\\\"
    )
    lines.append(
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}"
    )
    # Header row 2: sub-column labels
    lines.append(
        r"Method & mean (SD) & \% Improv & mean (SD) & \% Improv & Solve Time (s) \\"
    )
    lines.append(r"\midrule")

    for idx, row in summary.iterrows():
        is_given = row["method_key"] == "given"

        # ---- Toxicity mean (SD) -----------------------------------------
        cell_tox_ms = fmt_mean_sd_pct(row["tox_mean"], row["tox_sd"], decimals=1)
        if not is_given and idx in best["tox_mean"]:
            cell_tox_ms = bold(cell_tox_ms)

        # ---- Toxicity % improv ------------------------------------------
        cell_tox_pct = "---" if is_given else fmt_pct_change(row["tox_pct"], decimals=1)
        if not is_given and idx in best["tox_pct"]:
            cell_tox_pct = bold(cell_tox_pct)

        # ---- OS mean (SD) -----------------------------------------------
        cell_os_ms = fmt_mean_sd_raw(row["os_mean"], row["os_sd"], decimals=2)
        if not is_given and idx in best["os_mean"]:
            cell_os_ms = bold(cell_os_ms)

        # ---- OS % improv ------------------------------------------------
        cell_os_pct = "---" if is_given else fmt_pct_change(row["os_pct"], decimals=1)
        if not is_given and idx in best["os_pct"]:
            cell_os_pct = bold(cell_os_pct)

        # ---- Solve time -------------------------------------------------
        cell_time = "---" if is_given else fmt_time(row["solve_time"], decimals=3)
        if not is_given and idx in best["solve_time"]:
            cell_time = bold(cell_time)

        line = (
            f"{row['method_label']} & {cell_tox_ms} & {cell_tox_pct}"
            f" & {cell_os_ms} & {cell_os_pct} & {cell_time} \\\\"
        )
        lines.append(line)

        # Thin rule after Given to separate baseline from methods
        if is_given:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved TeX  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(
        constraint_mode="all_constraints",
        tox_outcome="all_constraints",
        tox_label="Joint Toxicity",
        caption=(
            "Comparison of robust prescriptive methods under all constraints. "
            "Joint toxicity: fraction of regimens satisfying all toxicity "
            "constraints simultaneously (ground-truth ensemble)."
        ),
        tex_label="tab:chemo_all_constraints",
        stem="summary_all_constraints",
    ),
    dict(
        constraint_mode="dlt_only",
        tox_outcome="Any_DLT",
        tox_label="Any DLT",
        caption=(
            "Comparison of robust prescriptive methods under DLT-only constraints. "
            "Any DLT: fraction of regimens satisfying the DLT toxicity constraint "
            "(ground-truth ensemble)."
        ),
        tex_label="tab:chemo_dlt_only",
        stem="summary_dlt_only",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize chemo_robust_table6.csv")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to chemo_robust_table6.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Directory to write output files (default: results/)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Constraint modes: {sorted(df['constraint_mode'].unique())}")
    print()

    for cfg in CONFIGS:
        mode = cfg["constraint_mode"]
        print(f"=== {mode} ===")
        try:
            summary = build_summary(df, cfg["constraint_mode"], cfg["tox_outcome"])
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        print(summary[["method_label", "tox_mean", "tox_pct", "os_mean", "os_pct", "solve_time"]].to_string(index=False))

        write_csv(
            summary,
            tox_label=cfg["tox_label"],
            path=args.out_dir / f"{cfg['stem']}.csv",
        )
        write_tex(
            summary,
            tox_label=cfg["tox_label"],
            path=args.out_dir / f"{cfg['stem']}.tex",
            caption=cfg["caption"],
            label=cfg["tex_label"],
        )
        print()


if __name__ == "__main__":
    main()

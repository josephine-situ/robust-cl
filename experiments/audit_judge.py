"""How much of a reported feasibility is the METHOD, and how much is the JUDGE?

The tuning judge on every problem is a fitted ensemble, and a constrained
optimum sits ON the constraint boundary by construction -- exactly where a
fitted judge's error decides the verdict. Measured on synthetic (2026-08-21):
error sd 0.033 in the band ``|f_true - b| < 0.05`` against decision margins of
0.015-0.020, 26% of verdicts flipped, 5 of 5 nominal decisions misjudged. On the
reactor the proxy under-called every series the ODE then passed 10/10. So a
feasibility of 0.80 under a proxy judge may be a method that violates 2 of 10
decisions, or a method that violates none and a judge that cannot tell.

This script separates the two WITHOUT looking at any ground truth. It reads the
per-decision rows the sweep and the test stage write and reports three things
per cell:

  **The abstention band.** A decision is called feasible only when its slack is
  below ``-kappa * s_c``, violated only when above ``+kappa * s_c``, and
  UNDECIDED in between -- ``s_c`` being the same out-of-fold label scale that
  D's radius, tau and the margin baseline are all quoted in
  (``uncertainty.instance_label_scales``, the one place it is estimated). So
  kappa is in unexplained standard deviations, commensurable with rho and m, and
  the output is an interval ``[feas_lo, feas_hi]`` whose width is the judge's
  own ignorance rather than the method's failure rate.

  **Member instability** (``lomo_*``). The fraction of leave-one-member-out
  sub-ensembles whose verdict differs from the full average's. Needs no truth
  and no refit -- the members are already fit -- and it is the ONLY judge check
  available on gastric, which has no ground truth at all.

  **Where the band and the bit disagree.** ``feas`` is the committed verdict
  rate; ``feas_lo``/``feas_hi`` bracket it. A cell whose bracket straddles
  ``--feas-target`` is a cell whose dial* the judge did not actually earn.

WHAT THIS IS NOT. The band is REPORTING ONLY and must never be fed back into
the sweep: dial* stays defined on the point verdict, or the protocol point
becomes a function of kappa (the same rule that keeps ``CPHistory.iter0_tau`` a
diagnostic). Nothing here re-solves anything, nothing reads the ODE or the
analytic ``f_true``, and no output of this script is a feasibility claim -- the
test stage still carries those.

``feas``, ``feas_lo`` and ``feas_hi`` are all conditional on the decisions that
SOLVED, exactly as the sweep's ``feasibility`` column is; read them beside
``solved_frac`` in the curve, never alone.

Usage (LOCAL, seconds -- it reads results and refits nothing but the scale)::

    uv run python experiments/audit_judge.py --problem reactor --suffix _incoh_f10_mmlp_s42
    uv run python experiments/audit_judge.py --problem gastric --suffix _incoh_s42
    uv run python experiments/audit_judge.py --problem reactor --suffix _incoh_f10_mmlp_s42 --phase test

Reads  ``{problem}_dial_judge{suffix}.csv``            (``--phase tune``, default)
       ``{problem}_dial_test_points{suffix}.csv``      (``--phase test``)
Writes ``{problem}_judge_audit{suffix}.csv`` / ``..._judge_audit_test{suffix}.csv``
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.instances import (
    load_config, synth_instance, reactor_instance, gastric_instance,
    synth_model_spec, reactor_model_spec, load_gastric_cv_configs,
)
from src.methods.nominal import resolve_constraint_config
from src.methods.uncertainty import instance_label_scales

OUT_DIR = "results/rho_sweep"
# Three widths rather than one. A single kappa would have to be defended; the
# three together say how fast the verdict degrades as the judge is given credit
# for its own error, which is the actual question. All of it is post-hoc
# arithmetic on the recorded slacks -- a width costs nothing and re-solves
# nothing.
KAPPAS = (0.25, 0.5, 1.0)


def _label_scales(problem: str, config: dict) -> dict:
    """``{constraint name -> s_c}`` under the SAME estimator D uses.

    Two names per outcome are returned -- ``dlt`` and ``dlt_constraint`` -- so a
    ``binding`` column written by either oracle resolves. Using anything but
    ``instance_label_scales`` here would put kappa on a different axis from rho,
    tau and the margin, and nothing downstream would flag the drift.
    """
    if problem == "synthetic":
        inst = synth_instance(config)
        m_type, m_params, _ = synth_model_spec(config)
    elif problem == "reactor":
        inst = reactor_instance(config)
        m_type, m_params, _ = reactor_model_spec(config)
    else:
        # Gastric resolves per constraint from the frozen CV picks, which
        # `resolve_constraint_config` reads off the instance itself, so the
        # defaults handed in here are never reached.
        inst = gastric_instance(cv_configs=load_gastric_cv_configs())
        m_type = str(config.get("default_model", {}).get("type", "xgb"))
        m_params = {}
    cfg_map, idx = {}, 0
    for c in inst.constraints:
        for md in c.models_data:
            cfg_map[id(md)] = resolve_constraint_config(inst, idx, m_type, m_params)
            idx += 1
    stat = str(config["uncertainty"].get("scale_stat", "oof_sd"))
    seed = int(config["uncertainty"].get("bootstrap_seed", 42))
    scales = instance_label_scales(inst, cfg_map, stat=stat, seed=seed)
    out = {}
    for c in inst.constraints:
        s = float(scales[id(c.models_data[0])])
        out[c.name] = s
        out[c.name.replace("_constraint", "")] = s
    print(f"[audit] label scale ({stat}): "
          + ", ".join(f"{k}={v:.4g}" for k, v in sorted(out.items())
                      if k.endswith("_constraint")), flush=True)
    return out


def _audit(df: pd.DataFrame, scales: dict, keys: list, feas_target: float,
           default_scale=None) -> pd.DataFrame:
    """One row per cell: the point verdict, the band at each kappa, instability."""
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        g = g[np.isfinite(g["slack"])]
        if g.empty:
            continue
        s = g["binding"].map(scales)
        if default_scale is not None:
            s = s.fillna(default_scale)
        if s.isna().any():
            missing = sorted(set(g.loc[s.isna(), "binding"].astype(str)))
            raise KeyError(
                f"no label scale for binding outcome(s) {missing} -- the judge "
                f"CSV was written for a different instance than --problem names")
        z = g["slack"].to_numpy(dtype=float) / s.to_numpy(dtype=float)
        n = len(z)
        row = dict(zip(keys, key))
        row["n_dec"] = n
        # The committed verdict is `slack <= 0`. Recomputed here rather than read
        # off the contexts file, so a disagreement between the two files is
        # visible rather than assumed away.
        row["feas"] = float(np.mean(z <= 0.0))
        for kap in KAPPAS:
            und = float(np.mean(np.abs(z) <= kap))
            lo = float(np.mean(z < -kap))
            row[f"undecided_k{kap:g}"] = und
            row[f"feas_lo_k{kap:g}"] = lo
            row[f"feas_hi_k{kap:g}"] = lo + und
        # `lomo_flip` is already a per-decision FRACTION of sub-ensembles, so its
        # mean is the chance that one dropped member changes the call, and
        # `lomo_any_frac` the share of decisions where any member would.
        fl = g["lomo_flip"].to_numpy(dtype=float)
        finite = np.isfinite(fl)
        row["lomo_flip_mean"] = float(np.mean(fl[finite])) if finite.any() else np.nan
        row["lomo_any_frac"] = float(np.mean(fl[finite] > 0)) if finite.any() else np.nan
        sd = g["lomo_sd"].to_numpy(dtype=float)
        row["lomo_sd_median"] = (float(np.median(sd[np.isfinite(sd)]))
                                 if np.isfinite(sd).any() else np.nan)
        # The verdict a cell cannot defend: at kappa=0.5 the band straddles the
        # target, so "cleared it" and "missed it" both sit inside the judge's own
        # error.
        row["target_inside_band"] = bool(
            row["feas_lo_k0.5"] < feas_target <= row["feas_hi_k0.5"])
        row["median_abs_z"] = float(np.median(np.abs(z)))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "reactor", "gastric"),
                   required=True)
    p.add_argument("--suffix", default="_incoh",
                   help="cell suffix, exactly as passed to the plotters")
    p.add_argument("--phase", choices=("tune", "test"), default="tune",
                   help="tune: the sweep's judge rows. test: run_dial_test's "
                        "points, whose judge may be EXACT -- the instability "
                        "columns are then nan by construction, not missing")
    p.add_argument("--feas-target", type=float, default=0.9)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if args.phase == "tune":
        src = os.path.join(OUT_DIR, f"{args.problem}_dial_judge{args.suffix}.csv")
        keys = ["method", "knob"]
        out = os.path.join(OUT_DIR, f"{args.problem}_judge_audit{args.suffix}.csv")
    else:
        src = os.path.join(OUT_DIR,
                           f"{args.problem}_dial_test_points{args.suffix}.csv")
        keys = ["method", "rho", "dial_star", "phase", "judge"]
        out = os.path.join(OUT_DIR,
                           f"{args.problem}_judge_audit_test{args.suffix}.csv")
    if not os.path.exists(src):
        raise SystemExit(
            f"no {src}. The audit columns are written by runs from 2026-09-02 "
            f"on; a cell scored before that carries the verdict bit only and "
            f"has to be re-run -- without --refresh, since the scoring rule is "
            f"unchanged and the search replays the existing rows free.")
    df = pd.read_csv(src)
    if "slack" not in df.columns:
        raise SystemExit(f"{src} predates the judge audit (no `slack` column).")
    if args.phase == "test":
        df = df[df["solved"] == 1.0]

    config = load_config(args.config)
    scales = _label_scales(args.problem, config)
    # A single-constraint problem writes `binding` as its one constraint name;
    # an older row could carry an empty one, and there is only one scale it
    # could possibly mean. On gastric there are five and guessing is not on.
    default = (None if args.problem == "gastric"
               else next(v for k, v in scales.items()
                         if k.endswith("_constraint")))
    res = _audit(df, scales, keys, float(args.feas_target), default_scale=default)
    res = res.sort_values(keys).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(out, index=False)

    show = keys + ["n_dec", "feas", "feas_lo_k0.5", "feas_hi_k0.5",
                   "undecided_k0.5", "lomo_any_frac", "target_inside_band"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res[show].to_string(index=False))
    n_straddle = int(res["target_inside_band"].sum())
    if n_straddle:
        print(f"\n[audit] {n_straddle}/{len(res)} cells have the "
              f"{args.feas_target:g} target INSIDE their kappa=0.5 band: there "
              f"the judge cannot separate clearing it from missing it.")
    print(f"[audit] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

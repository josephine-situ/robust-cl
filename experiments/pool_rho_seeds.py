"""Pool the per-seed rho curves into one table with a spread.

``run_rho_sweep.py --seed S`` reseeds the DRAWS FROM D only -- CP's bank of B
vertices and the wrapper's prefix of P -- while the instance and the folds stay
bit-identical. Repeating the whole sweep across seeds therefore isolates one
question: how much of a curve is the method, and how much is the bank that
happened to be drawn. This script reads those per-seed files and reports it.

Two tables, from the files the sweep already wrote (no solving):

  1. ``{problem}_rho_curve{cell}_pooled.csv`` -- per (method, rho): mean, sd, min
     and max of feasibility / objective / solved_frac over seeds, plus the seed
     count and how many cells were capped. **The sd is what a single-seed curve
     could not show**; where it is comparable to the gap between two methods at
     the same rho, that gap is not yet evidence.

  2. ``{problem}_rho_star{cell}_pooled.csv`` -- per method: every seed's rho*, its
     min/median/max, and the seed count. rho* is an argmax over a coarse grid, so
     it moves in grid steps and a spread of one step is the smallest one it can
     report. ``bounds`` carries each seed's ``bound`` value, because a rho* that
     is ``grid_max`` on one seed and ``feasibility`` on another is not a spread in
     the same quantity -- the censored ones are lower bounds.

With a handful of seeds this is a RANGE, not a confidence interval, and belongs
in a deck as one. It also says nothing about training-draw uncertainty: the folds
are fixed by construction here, so Known gap #5 is only half closed by it.

Usage:
    python experiments/pool_rho_seeds.py --problem gastric --cell _coh
    python experiments/pool_rho_seeds.py --problem synthetic --cell _coh_f5
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "results/rho_sweep"


def _read_seed_files(problem, kind, cell):
    """Every ``{problem}_{kind}{cell}_s*.csv``, concatenated, with a seed column.

    The seed is recovered from the FILENAME rather than trusted from the rows:
    ``rho_star`` files carry no seed column (they are a per-method summary), and a
    curve written before the column existed would otherwise pool as one seed.
    """
    pat = os.path.join(OUT_DIR, f"{problem}_{kind}{cell}_s*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit(
            f"no files matching {pat}. Run the sweep with --seed first, and pass "
            f"the same cell suffix it used (--cell _coh, _incoh, _coh_matchbank, "
            f"_coh_f5, ...)")
    frames = []
    for p in paths:
        seed = os.path.basename(p).rsplit("_s", 1)[1].removesuffix(".csv")
        d = pd.read_csv(p)
        d["seed"] = int(seed)
        d["source"] = os.path.basename(p)
        frames.append(d)
    return pd.concat(frames, ignore_index=True), paths


def pool_curve(problem, cell):
    df, paths = _read_seed_files(problem, "rho_curve", cell)
    # Ablation rows sit in the same file under a different phase and are swept over
    # tau/alpha, not rho -- pooling them with the rho rows would average different
    # experiments.
    df = df[df.get("phase", "rho_sweep") == "rho_sweep"]
    g = df.groupby(["method", "rho"])
    out = g.agg(
        n_seeds=("seed", "nunique"),
        feas_mean=("feasibility", "mean"), feas_sd=("feasibility", "std"),
        feas_min=("feasibility", "min"), feas_max=("feasibility", "max"),
        obj_mean=("objective", "mean"), obj_sd=("objective", "std"),
        solved_mean=("solved_frac", "mean"), solved_sd=("solved_frac", "std"),
        n_capped=("n_capped", "sum"),
        master_time_s=("master_time_s", "mean"),
        test_time_per_point_s=("test_time_per_point_s", "mean"),
    ).reset_index()
    out.insert(0, "problem", problem)
    out.insert(1, "cell", cell)
    path = os.path.join(OUT_DIR, f"{problem}_rho_curve{cell}_pooled.csv")
    out.to_csv(path, index=False)
    print(f"[pool] {len(paths)} seed files -> {path}")
    print(out.to_string(index=False))
    return out


def pool_rho_star(problem, cell):
    df, paths = _read_seed_files(problem, "rho_star", cell)
    rows = []
    for method, g in df.groupby("method"):
        g = g.sort_values("seed")
        vals = g["rho_star"].dropna()
        rows.append(dict(
            problem=problem, cell=cell, method=method,
            n_seeds=int(g["seed"].nunique()),
            n_reached=int(len(vals)),
            rho_star_min=float(vals.min()) if len(vals) else float("nan"),
            rho_star_median=float(vals.median()) if len(vals) else float("nan"),
            rho_star_max=float(vals.max()) if len(vals) else float("nan"),
            per_seed="; ".join(f"{int(s)}:{v:g}" if v == v else f"{int(s)}:none"
                               for s, v in zip(g["seed"], g["rho_star"])),
            bounds="; ".join(f"{int(s)}:{b}" for s, b in zip(g["seed"], g["bound"])),
        ))
    out = pd.DataFrame(rows).sort_values("rho_star_median", ascending=False)
    path = os.path.join(OUT_DIR, f"{problem}_rho_star{cell}_pooled.csv")
    out.to_csv(path, index=False)
    print(f"\n[pool] {len(paths)} seed files -> {path}")
    print(out.to_string(index=False))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "gastric"), default="gastric")
    p.add_argument("--cell", default="_coh",
                   help="cell suffix the sweep wrote, WITHOUT the _s<seed> part "
                        "(_coh, _incoh, _coh_matchbank, _coh_f5, ...)")
    args = p.parse_args()
    pool_curve(args.problem, args.cell)
    pool_rho_star(args.problem, args.cell)


if __name__ == "__main__":
    main()

"""Summarize probe F2 (paper/paper_plan.md, "FO-CP"): the FO-CP loop on SS1.

Reads results/paper_a/prelim/f2_{net,ode}_r*_a1.json (probe_focp.py --stage f2)
and prints, per arm and separation variant: convergence, master solves,
feasibility of the final decision against the clean-label model T(y*) from
theta0, a cold-start clean fit and the arm's truth (floor 50, a lower bound),
cost over the nominal decision as % of the objective range, and the a
posteriori attack. Read-only.

    python experiments/summarize_f2.py [--dir results/paper_a/prelim] [--redraws 0-99]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

FLOOR = 50.0


def load(d, arm):
    out = {}
    for p in glob.glob(os.path.join(d, f"f2_{arm}_r*_a1.json")):
        m = re.search(r"_r(\d+)_a1\.json$", p)
        out[int(m.group(1))] = json.load(open(p))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rate(vals):
    vals = np.asarray(vals, bool)
    lo, hi = wilson(vals.sum(), len(vals))
    return f"{vals.mean():.2f} [{lo:.2f}, {hi:.2f}]"


def summarize(runs, arm, redraws):
    ks = [k for k in sorted(runs) if k in redraws]
    R = np.array([runs[k]["R"] for k in ks])
    print(f"\n=== arm {arm}: {len(ks)} redraws; R (sigma) {R.min():.2f} to {R.max():.2f}, mean {R.mean():.2f}")
    nom = [runs[k]["nominal"] for k in ks]
    sig = 2.0
    for key in ("h", "clean", "clean_cold", "truth"):
        print(f"  nominal  feasible vs {key:10s}: {rate([n[key] >= FLOOR - 1e-6 for n in nom])}")
    rows = {}
    for var in ("one", "multi"):
        vs = [(k, runs[k]["variants"].get(var)) for k in ks]
        miss = [k for k, v in vs if not v or not v.get("done")]
        vs = [(k, v) for k, v in vs if v and v.get("done") and "final" in v]
        if miss:
            print(f"  [{var}] not done: redraws {miss}")
        fin = [v["final"] for _, v in vs]
        nomv = [runs[k]["nominal"] for k, _ in vs]
        conv = np.array([v["converged"] for _, v in vs])
        its = np.array([len(v["iters"]) for _, v in vs])
        cost = np.array([100 * (f["obj"] - n["obj"]) / runs[k]["obj_range"]
                         for (k, _), f, n in zip(vs, fin, nomv)])
        att = np.array([v["attack"]["worst"] for _, v in vs])
        vgap = np.array([v["attack"]["violation_gap"] for _, v in vs])
        sgap = np.array([v["attack"]["separation_gap"] for _, v in vs])
        sep = np.array([np.mean([i["sep_s"] for i in v["iters"] if "sep_s" in i]) for _, v in vs])
        master = np.array([np.max([i["master_s"] for i in v["iters"] if "master_s" in i]) for _, v in vs])
        wall = np.array([sum(i.get("sep_s", 0) + i.get("master_s", 0) for i in v["iters"])
                         + v["attack"]["time_s"] for _, v in vs])
        clean_gap = np.array([(f["clean"] - FLOOR) / sig for f in fin])
        truth_gap = np.array([(f["truth"] - FLOOR) / sig for f in fin])
        print(f"  [{var}] n = {len(vs)}; converged {conv.sum()}/{len(vs)}; master solves mean {its.mean():.1f}, "
              f"median {np.median(its):.0f}, max {its.max()}")
        for key in ("h", "clean", "clean_cold", "truth"):
            print(f"    feasible vs {key:10s}: {rate([f[key] >= FLOOR - 1e-6 for f in fin])}")
        both = [f["clean"] >= FLOOR and f["truth"] >= FLOOR for f in fin]
        print(f"    feasible vs clean AND truth: {rate(both)}")
        print(f"    converged only: clean {rate([f['clean'] >= FLOOR for f, c in zip(fin, conv) if c])}, "
              f"truth {rate([f['truth'] >= FLOOR for f, c in zip(fin, conv) if c])}")
        print(f"    slack vs clean (sigma): median {np.median(clean_gap):+.2f}, 10th pct {np.percentile(clean_gap, 10):+.2f}, "
              f"min {clean_gap.min():+.2f}; vs truth median {np.median(truth_gap):+.2f}, min {truth_gap.min():+.2f}")
        print(f"    cost over nominal, % of range: mean {cost.mean():.2f}, median {np.median(cost):.2f}, "
              f"IQR {np.percentile(cost, 25):.2f} to {np.percentile(cost, 75):.2f}, max {cost.max():.2f}")
        print(f"    attack below 50: {int((att < FLOOR).sum())}/{len(att)}; violation gap (sigma) among them: "
              f"median {np.median(vgap[att < FLOOR]) if (att < FLOOR).any() else np.nan:.3f}, "
              f"max {vgap.max():.3f}; separation gap median {np.median(sgap):.3f}, max {sgap.max():.3f}")
        print(f"    time: separation per iteration median {np.median(sep):.0f}s (max {sep.max():.0f}s); "
              f"largest master solve median {np.median(master):.0f}s (max {master.max():.0f}s); "
              f"loop + attack median {np.median(wall)/60:.0f} min (max {wall.max()/60:.0f} min)")
        nc = [k for (k, v) in vs if not v["converged"]]
        if nc:
            print(f"    not converged: redraws {nc}; their final feasible vs clean "
                  f"{[bool(runs[k]['variants'][var]['final']['clean'] >= FLOOR) for k in nc]}")
        rows[var] = dict(ks=[k for k, _ in vs], fin=fin, conv=conv, cost=cost, att=att)
    # does the attack predict clean-label violation?
    if "multi" in rows:
        r = rows["multi"]
        viol = np.array([f["clean"] < FLOOR for f in r["fin"]])
        flag = r["att"] < FLOOR
        print(f"  [multi] clean violation | attack flags: {int((viol & flag).sum())}/{int(flag.sum())}; "
              f"| attack clears: {int((viol & ~flag).sum())}/{int((~flag).sum())}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/paper_a/prelim")
    ap.add_argument("--redraws", default="0-99")
    a = ap.parse_args()
    lo, hi = map(int, a.redraws.split("-"))
    for arm in ("net", "ode"):
        summarize(load(a.dir, arm), arm, set(range(lo, hi + 1)))


if __name__ == "__main__":
    main()

"""Summarize the RC on Ovalle et al.'s tree surrogates (probe_rc_trees.py, reactor
SS1) and the WFP first look (probe_wfp.py). Read-only.

Per (family, split variant, arm or trust region) and method setting: runs solved,
feasibility against the clean-label model (same splits, leaves on the noise-free
labels) and the truth, cost over the run's nominal as % of the objective range
(reactor) or of the nominal objective (WFP), and the equal-feasibility read at
0.9 (each method's parameter interpolated with the oracle, as frontier_f2.py).

    python experiments/summarize_trees.py [--wfp]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

D = os.path.join("results", "paper_a", "prelim")


def load(wfp):
    rows = []
    pat = "wfp_*.json" if wfp else "trees_*.json"
    for p in glob.glob(os.path.join(D, pat)):
        f = json.load(open(p))
        thr = f.get("T", 50.0)
        if wfp:
            key = (f["family"], f["variant"], "hull" if f.get("trust_region", True) else "none")
        else:
            key = (f["family"], f["variant"], f["arm"])
        for k, rec in f["runs"].items():
            if not rec.get("done"):
                continue
            nom = next((r for r in rec["rows"] if r["kind"] == "nominal"), None)
            if nom is None or nom.get("obj") != nom.get("obj"):
                continue
            denom = rec.get("obj_range") or abs(nom["obj"])
            for r in rec["rows"]:
                solved = r.get("obj") == r.get("obj") and "truth" in r
                rows.append({"family": key[0], "variant": key[1], "setting": key[2],
                             "kind": r["kind"], "param": r["param"], "redraw": int(k),
                             "solved": solved, "status": r["status"], "time_s": r["time_s"],
                             "clean": (r["clean"] >= thr - 1e-6) if solved else np.nan,
                             "truth": (r["truth"] >= thr - 1e-9) if solved else np.nan,
                             "cost": 100 * (r["obj"] - nom["obj"]) / denom if solved else np.nan,
                             "R": rec.get("R"), "true_opt": (rec.get("true_opt") or {}).get("obj"),
                             "obj": r.get("obj")})
    return pd.DataFrame(rows)


def interp(t, ref, lvl):
    t = t.sort_values("p")
    v, c = t[ref].values, t.cost.values
    if not (v >= lvl).any():
        return np.nan
    i = int(np.argmax(v >= lvl))
    if i == 0:
        return c[0]
    w = (lvl - v[i - 1]) / (v[i] - v[i - 1])
    return c[i - 1] + w * (c[i] - c[i - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfp", action="store_true")
    a = ap.parse_args()
    df = load(a.wfp)
    if df.empty:
        print("no finished runs")
        return
    keys = ["family", "variant", "setting"]
    for key, g in df.groupby(keys):
        n = g.redraw.nunique()
        R = g.drop_duplicates("redraw").R.mean()
        print(f"\n=== {' / '.join(key)}: {n} redraws, simulated radius mean {R:.2f} sigma")
        if a.wfp:
            nom = g[g.kind == "nominal"]
            gap = 100 * (nom.obj.values - g.drop_duplicates("redraw").true_opt.values) / nom.obj.values
            print(f"  nominal cost over the true optimum: mean {np.nanmean(gap):+.1f}%")
        tab = (g.groupby(["kind", "param"], sort=False)
                .agg(n=("redraw", "nunique"), solved=("solved", "mean"), clean=("clean", "mean"),
                     truth=("truth", "mean"), cost=("cost", "mean"), cost_se=("cost", "sem"),
                     time_s=("time_s", "median"), tl=("status", lambda s: (s == 9).mean()))
                .reset_index())
        show = tab[tab.kind.isin(["nominal", "rc", "conformal", "cmicl"]) |
                   ((tab.kind == "margin") & tab.param.isin(tab[tab.kind == "margin"].param.iloc[::3]))]
        with pd.option_context("display.width", 200):
            print(show.round(3).to_string(index=False))
        for ref in ("clean", "truth"):
            out = []
            for kind in ("rc", "margin", "cmicl"):
                t = tab[tab.kind == kind].rename(columns={"param": "p"})
                if kind == "cmicl":
                    t = t.assign(p=-t.p)
                if len(t):
                    c = interp(t, ref, 0.9)
                    out.append(f"{kind} {c:.2f}%" if c == c else f"{kind} never")
            print(f"  cost to reach 0.9 vs {ref}: " + "; ".join(out))


if __name__ == "__main__":
    main()

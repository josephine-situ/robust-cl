"""F2 frontier: cost (% of the objective range over each run's nominal) at which FO-CP
(radius grid, f2_*_s<scale>.json), the constant margin, split conformal and C-MICL
(f2base_*.json) reach a feasibility level against the clean-label model or the truth,
by linear interpolation along each method's parameter. The parameter is read with the
oracle for EVERY method (the "at equal feasibility" device of the outline), and the
run time and convergence of each FO-CP radius. Read-only.

    python experiments/frontier_f2.py
"""
import json, glob, re, numpy as np, pandas as pd
D = "results/paper_a/prelim"
def interp(t, ref, lvl):
    t = t.sort_values("p")
    v, c, p = t[ref].values, t.cost.values, t.p.values
    if (v >= lvl).sum() == 0: return None
    i = int(np.argmax(v >= lvl))
    if i == 0: return c[0], p[0]
    w = (lvl - v[i-1]) / (v[i] - v[i-1])
    return c[i-1] + w * (c[i] - c[i-1]), p[i-1] + w * (p[i] - p[i-1])
for arm in ("net", "ode"):
    rows = []
    for p in glob.glob(f"{D}/f2base_{arm}_r*_a1.json"):
        r = json.load(open(p)); k = r["redraw"]
        nom = next(x for x in r["rows"] if x["kind"] == "nominal")["obj"]
        for x in r["rows"]:
            if x["kind"] in ("margin", "conformal", "cmicl") and x.get("obj") == x.get("obj"):
                rows.append((x["kind"], x["param"], x["clean"] >= 50 - 1e-6, x["truth"] >= 50, 100 * (x["obj"] - nom) / r["obj_range"]))
        for q in glob.glob(f"{D}/f2_{arm}_r{k}_a1_s*.json"):
            g = json.load(open(q)); f = g["variants"]["multi"]["final"]
            rows.append(("focp", g["r_scale"], f["clean"] >= 50, f["truth"] >= 50, 100 * (f["obj"] - g["nominal"]["obj"]) / g["obj_range"]))
    df = pd.DataFrame(rows, columns=["k", "p", "c", "t", "cost"]).groupby(["k", "p"]).mean().reset_index()
    print(f"== arm {arm}")
    for ref in ("c", "t"):
        for lvl in (0.9, 0.95):
            out = []
            for k in ("focp", "margin", "conformal", "cmicl"):
                t = df[df.k == k]
                if k == "conformal" or k == "cmicl": t = t.assign(p=-t.p)  # smaller alpha = more conservative
                r = interp(t, ref, lvl)
                out.append(f"{k} " + (f"{r[0]:.2f}% (param {abs(r[1]):.3g})" if r else "never"))
            print(f"  vs {'clean' if ref=='c' else 'truth'} at {lvl}: " + "; ".join(out))
    # convergence and time per scale
    for s in (0.5, 0.75, 1.0):
        its, conv, wall, att = [], 0, [], 0
        fs = glob.glob(f"{D}/f2_{arm}_r*_a1_s{s:g}.json")
        for q in fs:
            v = json.load(open(q))["variants"]["multi"]
            its.append(len(v["iters"])); conv += v["converged"]
            wall.append(sum(i["master_s"] + i["sep_s"] for i in v["iters"]) + v["attack"]["time_s"])
            att += v["attack"]["worst"] < 50
        print(f"  s={s}: n {len(fs)}, converged {conv}, iters mean {np.mean(its):.1f}, run median {np.median(wall)/60:.0f} min (max {max(wall)/60:.0f}), attack flags {att}")

"""How much of D the label bounds actually take away, per rho.

``ScenarioBank`` clips every draw to each outcome's ``label_bounds`` -- shifting a
percentile rank past 0 or 1 is not a plausible relabeling -- so the set the
methods really face is ``D n [lo, hi]``, not D. Clipping only shrinks
``|delta_i|``, so the shift stays inside D; what it changes is how much of the
assumed radius is REACHABLE, and that is a function of rho.

This measures it. CLAUDE.md carried the numbers for rho = 1.0 and 0.75 only, and
rho = 0.5 was inferred from them rather than read -- which stopped being good
enough when rho = 0.5 became a headline column of the dial sweep rather than a
sensitivity check.

Per (rho, outcome) it reports, over B draws:

  out_frac      fraction of shifted labels that would leave [lo, hi]
  ||raw||       mean L2 norm of the drawn shift, before clipping
  ||clipped||   mean L2 norm of what is actually applied
  reach         ||clipped|| / ||raw|| -- the fraction of the radius that survives
  R_c           the outcome's nominal radius, rho * scale(y_c) * sqrt(n)

Two things make the numbers the production ones rather than an approximation:

- The draws come from ``ScenarioBank._draw`` itself, at ``n_scenarios=0`` so no
  model is trained. Same rng sequence, same coherence grouping, same label link.
- The raw shift is captured by instrumenting ``_clip_to_bounds`` where it is
  actually called, so the DERIVED DLT is measured after the identity has been
  applied to the already-clipped components -- which is what really happens, and
  is not reproducible by re-deriving the draw outside the bank.

Bites only where ``label_bounds`` is set: gastric's five toxicities. Gastric OS,
synthetic and the reactor carry none and report ``out_frac = 0``, ``reach = 1``.

Usage::

    python experiments/measure_clip_fraction.py                    # gastric, the dial-sweep columns
    python experiments/measure_clip_fraction.py --rhos 0.25 0.5 0.75 1.0
    python experiments/measure_clip_fraction.py --coherent
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/rho_sweep/gastric_clip_fraction.csv"


def measure(instance, uset, model_spec, rho, n_draws, seed):
    import dataclasses
    import src.methods.uncertainty as U

    model_type, model_params = model_spec
    u_rho = dataclasses.replace(uset, rho=float(rho))
    # n_scenarios=0: the draws are what is wanted, not the B model fits behind them.
    bank = U.build_bank_for_instance(instance, model_type, model_params, u_rho,
                                     n_scenarios=0, seed=seed, verbose=False)

    # Capture (raw, clipped) exactly where the bank clips, so the derived DLT is
    # measured after the identity has run on the already-clipped components.
    captured = []
    orig = U._clip_to_bounds

    def spy(model_data, delta):
        out = orig(model_data, delta)
        captured.append((id(model_data), np.asarray(delta, float),
                         np.asarray(out, float)))
        return out

    by_md = {}
    U._clip_to_bounds = spy
    try:
        for b in range(n_draws):
            captured.clear()
            bank._draw(b)
            # LAST call per outcome within the draw. A LINKED target is clipped
            # TWICE -- once on the free draw the loop makes and discards (it is
            # kept only so the rng sequence, and under coherence the shared
            # direction, are bit-identical to a bank with the link off), and again
            # on the shift the identity derives. Averaging both would report DLT
            # as a mixture of a draw that never reaches a model and the one that
            # does.
            per_draw = {}
            for md_id, raw, clipped in captured:
                per_draw[md_id] = (raw, clipped)
            for md_id, pair in per_draw.items():
                by_md.setdefault(md_id, []).append(pair)
    finally:
        U._clip_to_bounds = orig

    rows = []
    for md in bank._mds:
        md_id = id(md)
        draws = by_md.get(md_id, [])
        if not draws:
            continue
        bounds = getattr(md, "label_bounds", None)
        y = np.asarray(md.y_train, float)
        n = len(y)
        out_fracs, raw_n, clip_n = [], [], []
        for raw, clipped in draws:
            if bounds is None:
                out_fracs.append(0.0)
            else:
                lo, hi = bounds
                shifted = y + raw
                out_fracs.append(float(np.mean((shifted < lo) | (shifted > hi))))
            raw_n.append(float(np.linalg.norm(raw)))
            clip_n.append(float(np.linalg.norm(clipped)))
        raw_m, clip_m = float(np.mean(raw_n)), float(np.mean(clip_n))
        rows.append(dict(
            rho=float(rho), outcome=bank._md_name[md_id],
            bounded=bounds is not None, n_rows=n, n_draws=len(draws),
            scale=float(bank.scales[md_id]),
            R_c=float(u_rho.radius(bank.scales[md_id], n)),
            out_frac=float(np.mean(out_fracs)),
            raw_norm=raw_m, clipped_norm=clip_m,
            reach=(clip_m / raw_m if raw_m else float("nan")),
        ))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.75, 1.0],
                   help="0.5 and 1.0 are the dial sweep's gastric columns; 0.75 "
                        "reproduces the number already in CLAUDE.md as a check")
    p.add_argument("--n-draws", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coherent", action="store_true",
                   help="the ablation cell; production is incoherent")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs",
                   default="results/cv/gastric_selected_configs.json")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    import dataclasses
    import yaml
    from src.data.generate import gastric_cancer
    from src.methods.uncertainty import uncertainty_set_from_config

    config = yaml.safe_load(open(args.config))
    cv_configs = gt_configs = None
    if os.path.exists(args.cv_configs):
        cv_configs = json.load(open(args.cv_configs))
        gt_path = args.cv_configs.replace("_selected_configs",
                                          "_gt_ensemble_configs")
        if os.path.exists(gt_path):
            gt_configs = json.load(open(gt_path))
    inst = gastric_cancer(fixed_constraint_configs=cv_configs,
                          fixed_gt_ensemble_configs=gt_configs)
    uset = dataclasses.replace(uncertainty_set_from_config(config),
                               geometry="ellipsoid", coherent=bool(args.coherent))
    spec = (config["default_model"]["type"], config["default_model"]["params"])

    print(f"[clip] gastric, geometry=ellipsoid, coherent={args.coherent}, "
          f"B={args.n_draws} draws, seed={args.seed}", flush=True)
    rows = []
    for rho in args.rhos:
        rows += measure(inst, uset, spec, rho, int(args.n_draws), int(args.seed))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[clip] bounded outcomes only, by rho:")
    b = df[df["bounded"]]
    if not b.empty:
        agg = b.groupby("rho").agg(
            out_frac_min=("out_frac", "min"), out_frac_max=("out_frac", "max"),
            reach_min=("reach", "min"), reach_max=("reach", "max"))
        print(agg.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"[clip] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

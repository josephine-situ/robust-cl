"""Diagnostics for the two unresolved features of the rho axis (Known gaps #1).

Two separate questions, one script, selected by ``--which``:

``robust_reg`` (synthetic)
    Why does robust_reg's objective get BETTER than nominal at rho >= 0.5 while
    held-out feasibility falls to 0? A robustification that improves the
    objective is not robustifying: it is loosening the constraint.

    The test needs no MIP. The synthetic problem is
    ``min -(x1+x2) s.t. f(x) <= 1`` on ``[0,1]^2`` with
    ``f_true = x1^2 + x2^2 + 0.5 x1 x2``, so the constraint model IS the feasible
    region and "looser" has a direct meaning: the fitted surface sits LOWER where
    the optimizer looks. We train the nominal and the label-robust model at each
    rho and compare their surfaces on a dense grid -- overall, and restricted to
    the near-boundary band the LP actually optimizes along.

    The hypothesis being tested: the label adversary ``delta = R * r / ||r||_2``
    maximizes TRAINING LOSS, and squared loss is symmetric in the sign of the
    residual. It therefore has no preferred direction in PREDICTION space and
    cannot be relied on to raise the fitted surface. On a flexible model class
    (rf) it instead amplifies deviation from the current fit -- a variance
    inflating, not a conservative, perturbation -- and the optimizer then walks
    into whichever dip it opened.

``cp_dip`` (gastric)
    Why does CP's held-out feasibility dip at rho=0.5 (0.958 at rho=0.3 ->
    0.898) and recover at 0.75? The sweep records only fold MEANS, and
    feasibility is conditional on the context being solvable, so a dip can come
    from a genuine loss of robustness OR from the solved COHORT changing beneath
    the mean. This re-scores the three rho values keeping the per-fold
    feasibility, objective, solved fraction, status and the solved row ids, which
    separates the two.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "results/rho_sweep/diagnostics"


# ---------------------------------------------------------------------------
# synthetic robust_reg
# ---------------------------------------------------------------------------
def diagnose_robust_reg(config, rhos, n_grid=400, model_type=None):
    from src.data.generate import synthetic_nonlinear, _synthetic_f_true
    from src.methods.robust_regression import _train_label_robust_model
    from src.models.train import train_model
    from src.methods.uncertainty import label_scale
    from src.methods.uncertainty import instance_folds

    d = config["synthetic"]
    seed = config["uncertainty"].get("bootstrap_seed", 42)
    inst = synthetic_nonlinear(n_train=d["n_train"], n_features=d["n_features"],
                               noise_std=d["noise_std"], seed=seed)
    md = inst.constraints[0].models_data[0]
    X, y = md.X_train, np.asarray(md.y_train, float)
    n = len(y)
    # --model-type overrides the config so the EXACT arm can be checked too:
    # "linear" takes _label_robust_linear (the closed-form SOCP), everything
    # else takes the alternating retrain loop. If the adversary has no
    # conservative direction, it has none on either arm.
    mt = model_type or config["default_model"]["type"]
    mp = {} if model_type else config["default_model"]["params"]
    rr = config["methods"].get("robust_reg", {})
    K = int(rr.get("K", 5))
    bf = float(rr.get("budget_frac", 0.5))
    folds = instance_folds(inst, seed)
    scale = label_scale(y, stat="oof_sd", X=X, model_type=mt, model_params=mp,
                        folds=folds)
    rhs = float(inst.constraints[0].rhs)

    # Dense grid over the box the optimizer searches. 2 features, so this is the
    # whole decision space and the LP optimum can be read off it directly.
    g = np.linspace(0.0, 1.0, n_grid)
    G1, G2 = np.meshgrid(g, g)
    XG = np.column_stack([G1.ravel(), G2.ravel()])
    f_gt = _synthetic_f_true(XG)
    obj_gt = -XG.sum(axis=1)                     # the LP objective, minimized

    nom = train_model(X, y, mt, mp)
    p_nom = nom.predict(XG)

    def lp_optimum(pred):
        """Best (lowest) objective over grid points the fitted model calls feasible."""
        ok = pred <= rhs
        if not ok.any():
            return np.nan, np.nan
        i = int(np.argmin(np.where(ok, obj_gt, np.inf)))
        return float(obj_gt[i]), float(f_gt[i])

    rows = []
    for rho in rhos:
        R = float(np.sqrt(n)) * rho * scale
        rob = (nom if rho == 0 else
               _train_label_robust_model(X, y, mt, mp, rho, bf, K,
                                         scale_stat="oof_sd", folds=folds,
                                         geometry="ellipsoid"))
        p_rob = rob.predict(XG)
        r = y - nom.predict(X)                    # residuals the adversary sees
        rn = float(np.linalg.norm(r))
        shift = p_rob - p_nom
        # The band the optimizer walks along: where the NOMINAL model sits at its
        # own boundary. A conservative perturbation raises the surface here.
        band = np.abs(p_nom - rhs) <= 0.05
        obj_n, fgt_n = lp_optimum(p_nom)
        obj_r, fgt_r = lp_optimum(p_rob)
        rows.append(dict(
            rho=rho, radius_R=R, resid_norm=rn,
            amplification_c=(R / rn if rn else np.nan),
            mean_shift=float(shift.mean()),
            mean_shift_band=float(shift[band].mean()) if band.any() else np.nan,
            frac_lowered=float((shift < 0).mean()),
            frac_lowered_band=float((shift[band] < 0).mean()) if band.any() else np.nan,
            feasible_area_nom=float((p_nom <= rhs).mean()),
            feasible_area_rob=float((p_rob <= rhs).mean()),
            lp_obj_nom=obj_n, lp_obj_rob=obj_r,
            gt_at_opt_nom=fgt_n, gt_at_opt_rob=fgt_r, gt_rhs=rhs,
            violates_nom=bool(fgt_n > rhs + 1e-9),
            violates_rob=bool(fgt_r > rhs + 1e-9),
            train_rmse_nom=float(np.sqrt(np.mean(r ** 2))),
            train_rmse_rob=float(np.sqrt(np.mean((y - rob.predict(X)) ** 2))),
        ))
        c = rows[-1]["amplification_c"]
        print(f"  rho={rho:<5g} R={R:7.3f} c=R/||r||={c:6.3f} "
              f"feas_area {rows[-1]['feasible_area_nom']:.4f}->"
              f"{rows[-1]['feasible_area_rob']:.4f}  "
              f"lowered_in_band={rows[-1]['frac_lowered_band']:.3f}  "
              f"lp_obj {obj_n:+.4f}->{obj_r:+.4f}  "
              f"gt_at_opt {fgt_n:.4f}->{fgt_r:.4f} (rhs {rhs:g})", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# gastric CP dip
# ---------------------------------------------------------------------------
def diagnose_cp_dip(config, rhos, method="cp"):
    """Re-score (method, rho) keeping the PER-FOLD breakdown the sweep averages away."""
    import time as _time
    from experiments.run_rho_sweep import _setup_gastric, _fixed_knobs
    from src.methods.uncertainty import uncertainty_set_from_config
    from src.methods.cv_calibrate import _fold_instance
    from src.evaluation.chemo_metrics import solve_for_test_cohort
    from src.data.generate import filter_constraints

    args = argparse.Namespace(coherent=True, match_bank=False, n_folds=None,
                              cv_configs="results/cv/gastric_selected_configs.json",
                              knobs=None, knobs_from_cv=False)
    inst, folds, oracle, make_build, cnames, contextual = _setup_gastric(config, args)
    base = dataclasses.replace(uncertainty_set_from_config(config),
                               geometry="ellipsoid", coherent=True)
    knob = _fixed_knobs("gastric", args, config).get(method, 0.0)
    print(f"[cp-dip] method={method} knob={knob} folds={len(folds)}", flush=True)

    rows = []
    for rho in rhos:
        solver = make_build(method, dataclasses.replace(base, rho=rho))(knob)
        for fi_idx, (train_idx, val_idx) in enumerate(folds):
            val_rows = inst.X_train[val_idx]
            fi = filter_constraints(_fold_instance(inst, train_idx, val_rows), cnames)
            t0 = _time.time()
            res = solver(fi)
            master_s = _time.time() - t0
            if isinstance(res, tuple):
                res = res[0]
            solved_ids, feas, objs = [], [], []
            for j, row in enumerate(val_rows):
                _, x_opt = solve_for_test_cohort(res, fi, row)
                if x_opt is None:
                    continue
                solved_ids.append(int(val_idx[j]))
                feas.append(1.0 if oracle.feasible(x_opt) else 0.0)
                objs.append(float(oracle.objective(x_opt)))
            rows.append(dict(
                rho=rho, method=method, fold=fi_idx,
                n_val=len(val_rows), n_solved=len(solved_ids),
                solved_frac=len(solved_ids) / max(1, len(val_rows)),
                feas=float(np.mean(feas)) if feas else np.nan,
                obj=float(np.mean(objs)) if objs else np.nan,
                status=str(getattr(res, "status", "unknown")),
                master_time_s=master_s,
                solved_ids=json.dumps(solved_ids),
                feas_by_id=json.dumps(dict(zip(map(str, solved_ids), feas))),
            ))
            print(f"  rho={rho:<5g} fold={fi_idx} solved={rows[-1]['solved_frac']:.3f} "
                  f"({len(solved_ids)}/{len(val_rows)}) feas={rows[-1]['feas']:.3f} "
                  f"obj={rows[-1]['obj']:+.4f} [{rows[-1]['status']}] "
                  f"{master_s:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=("robust_reg", "cp_dip"), required=True)
    p.add_argument("--rhos", type=float, nargs="+", default=None)
    p.add_argument("--method", default="cp", help="cp_dip only")
    p.add_argument("--model-type", default=None,
                   help="robust_reg only: override the embedded model class "
                        "(e.g. linear, to exercise the exact SOCP arm)")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    import yaml
    config = yaml.safe_load(open(args.config))
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.which == "robust_reg":
        rhos = args.rhos or [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        df = diagnose_robust_reg(config, rhos, model_type=args.model_type)
        tag = f"_{args.model_type}" if args.model_type else ""
        out = os.path.join(OUT_DIR, f"synthetic_robust_reg_surface{tag}.csv")
    else:
        rhos = args.rhos or [0.3, 0.5, 0.75]
        df = diagnose_cp_dip(config, rhos, method=args.method)
        out = os.path.join(OUT_DIR, f"gastric_{args.method}_dip_per_fold.csv")
    df.to_csv(out, index=False)
    print(f"\n[diagnose] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

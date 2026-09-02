"""Adversary probe: how strong is the random scenario bank, really?

Measurement only -- no CP loop, no calibration, no Table 6 evaluation. Answers
three questions at the nominal prescriptions, per outcome:

  A. RANDOM vs DIRECTED. How much of the frozen-topology worst case (a +eps_c
     shift of the prediction at x*) does the best of the B random bank draws
     achieve, and does a directed adversary -- greedy top-m along the influence
     vector g_i = d f(x*) / d delta_i -- beat it?

  B. DOES THE BUDGET BIND? nnz(g) vs m = budget_frac * n. For a single tree the
     support of g is one leaf, so m > nnz(g) and the L1 budget is slack: the
     worst case collapses to a uniform +eps_c tightening. For an ensemble the
     support is the union of x*'s leaves over all trees and the budget may bind,
     which is what makes the greedy sort do real work.

  C. PER-OUTCOME FEASIBILITY. If every constraint is pushed to its OWN worst
     case simultaneously (the product-set adversary, not the shared-b diagonal),
     is the master still feasible at each anchor? Swept over budget fractions t
     so the answer is a threshold, not a yes/no.

(C) is the one that decides a design question: if full-budget product-set
separation is broadly infeasible on gastric, then "what replaces scenario
rollback" is moot and the budget fraction has to carry the load instead.

Influence vectors are exact for `linear` (ElasticNet at a fixed active set) and
for mean-valued tree ensembles (CART/RF), and a support-correct heuristic for
boosted ensembles (xgb/gbm leaf values are shrunk residual sums, not label
means). Every directed delta is verified by an actual retrain, so the frozen-
structure approximation is never trusted -- only used to propose.

Usage:
    python experiments/run_adversary_probe.py --quick        # local smoke test
    python experiments/run_adversary_probe.py                # full, use SLURM
    sbatch experiments/submit_adversary_probe.sh
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline

from src.data.generate import gastric_cancer, synthetic_nonlinear
from src.methods.cp import (
    _build_master_with_nominal,
    _setup_anchors,
    _fix_anchor_context,
    _restore_context_bounds,
    _is_constraint_constraint,
)
from src.methods.nominal import resolve_constraint_config
from dataclasses import replace as _dc_replace

from src.methods.uncertainty import (
    GEOMETRIES,
    ScenarioBank,
    uncertainty_set_from_config,
    _clip_to_bounds,
)
from src.utils.perturbations import l2_worst_case_shift
from src.models.train import retrain_on_perturbed, train_model
from src.data.instances import load_config


# ---------------------------------------------------------------------------
# Influence vectors:  g_i = d f_{theta(delta)}(x*) / d delta_i  at delta = 0
# ---------------------------------------------------------------------------
def _unwrap(model):
    """Return (scaler_or_None, final_estimator) for a Pipeline or bare estimator."""
    if isinstance(model, Pipeline):
        return model[:-1], model[-1]
    return None, model


def _as_leaf_matrix(leaves, n):
    """Normalize the many shapes of sklearn/xgb `.apply()` to (n_rows, n_trees)."""
    L = np.asarray(leaves)
    if L.ndim == 1:                      # single DecisionTree -> (n,)
        return L.reshape(n, 1)
    if L.ndim == 2:                      # RF / xgb -> (n, T)
        return L
    if L.ndim == 3:                      # sklearn GBM -> (n, T, K)
        return L.reshape(n, -1)
    raise ValueError(f"unexpected leaf-index shape {L.shape}")


def _influence_trees(est, Xs, xs_star):
    """g_i = (1/T) sum_t 1[i in leaf_t(x*)] / |leaf_t(x*)|.

    EXACT for models whose leaf value is the mean of its members' labels
    (DecisionTree, RandomForest). For boosted ensembles the leaf value is a
    shrunk sum of stage residuals rather than a label mean, so this is a
    heuristic -- it gets the SUPPORT of g exactly right (only rows co-located
    with x* can move the prediction at all) and the relative weights
    approximately. The retrain-verify step downstream is what makes that safe.
    """
    n = Xs.shape[0]
    L = _as_leaf_matrix(est.apply(Xs), n)
    L_star = _as_leaf_matrix(est.apply(xs_star), 1)
    T = L.shape[1]
    g = np.zeros(n)
    for t in range(T):
        mask = L[:, t] == L_star[0, t]
        size = int(mask.sum())
        if size:
            g[mask] += 1.0 / size
    return g / max(1, T)


def _influence_linear(est, Xs, xs_star):
    """Closed form for ElasticNet at a fixed active set A and fixed signs.

    sklearn minimises  1/(2n)||y - Xw||^2 + a*l1*||w||_1 + 0.5*a*(1-l1)*||w||^2,
    so on A the stationarity condition is affine in y:
        (Xa'Xa/n + a(1-l1) I) w = Xa'y/n - a*l1*s
    hence  dw/dy_i = M^{-1} Xa[i]'/n  and  g = Xa M^{-1} x*_A / n.
    Exact while the active set and coefficient signs hold -- checked post-hoc by
    the retrain-verify step.
    """
    coef = np.asarray(getattr(est, "coef_", None), dtype=float).ravel()
    n = Xs.shape[0]
    active = np.flatnonzero(np.abs(coef) > 1e-12)
    if active.size == 0:
        return np.zeros(n)          # intercept-only fit: labels cannot move f(x*)
    Xa = Xs[:, active]
    alpha = float(getattr(est, "alpha", 0.0))
    l1_ratio = float(getattr(est, "l1_ratio", 1.0))
    M = Xa.T @ Xa / n + alpha * (1.0 - l1_ratio) * np.eye(active.size)
    return (Xa @ np.linalg.pinv(M) @ np.asarray(xs_star).ravel()[active]) / n


def influence_vector(model, X_train, x_star):
    """g for one trained model at one point. Raises on unsupported classes."""
    scaler, est = _unwrap(model)
    xs_star = np.atleast_2d(np.asarray(x_star, dtype=float))
    Xs = np.asarray(X_train, dtype=float)
    if scaler is not None:
        Xs = scaler.transform(Xs)
        xs_star = scaler.transform(xs_star)
    if hasattr(est, "apply"):
        return _influence_trees(est, Xs, xs_star)
    if hasattr(est, "coef_"):
        return _influence_linear(est, Xs, xs_star)
    raise NotImplementedError(
        f"no influence vector for {type(est).__name__}; supported: tree ensembles "
        f"(via .apply) and linear models (via .coef_)"
    )


# ---------------------------------------------------------------------------
# The directed adversary
# ---------------------------------------------------------------------------
def directed_delta(g, eps, gamma, model_data, geometry="box_l1", radius=None):
    """argmax_{delta in D} g'delta, then clipped to the label bounds.

    ``box_l1``: greedy top-m at +/-eps. Exact for the linearised objective -- the
    maximum of a linear function over {|d_i| <= eps, ||d||_1 <= gamma} is at a
    vertex, so sort by |g_i| and spend the budget on the largest entries in their
    own sign direction. Same routine as `worst_case_label_shift`, ranking by
    influence-at-x* instead of residual.

    ``ellipsoid``: the closed form ``radius * g/||g||_2`` -- no sort, no budget
    bookkeeping. At the matched radius ``sqrt(m)*eps`` this achieves at least as
    much as the greedy vertex (Cauchy-Schwarz), with equality only if the top-m
    |g_i| are equal and the rest vanish. Clipping is safe under both: it shrinks
    every |delta_i|, hence both the L1 and the L2 norm, so the result stays in D.
    """
    g = np.asarray(g, dtype=float)
    if geometry == "ellipsoid":
        return _clip_to_bounds(model_data, l2_worst_case_shift(g, radius or 0.0))
    delta = np.zeros(g.shape[0])
    if eps <= 0 or gamma <= 0:
        return delta
    budget = float(gamma)
    for i in np.argsort(-np.abs(g)):
        if budget <= 1e-12 or abs(g[i]) <= 0.0:
            break                              # zero influence: spending here is waste
        shift = min(eps, budget)
        delta[i] = shift * (1.0 if g[i] >= 0 else -1.0)
        budget -= shift
    return _clip_to_bounds(model_data, delta)


def budget_at(uset, eps, n, t=1.0):
    """``(gamma, radius)`` at budget fraction ``t``, matched in L2 length.

    The box path keeps its existing meaning exactly -- ``gamma_t = t * gamma``
    moves ``t*m`` rows at +/-eps, an L2 length of ``sqrt(t*m)*eps``. The ellipsoid
    therefore takes ``R_t = sqrt(t) * R``, NOT ``t * R``, so the two geometries
    spend the same L2 budget at the same ``t`` and the Part C/D sweeps stay
    comparable across the flag. All existing box measurements are unchanged.
    """
    gamma = float(t) * uset.budget_frac * int(n) * float(eps)
    radius = float(np.sqrt(max(0.0, t))) * uset.radius_from_eps(eps, int(n))
    return gamma, radius


def _predict(model, x_star):
    return float(model.predict(np.atleast_2d(x_star))[0])


def frank_wolfe_adversary(model_data, m_type, m_params, x_star, eps, gamma,
                          seed, n_steps=3, geometry="box_l1", radius=None):
    """Re-linearise / re-maximise / retrain, keeping only verified improvements.

    Each step costs one retrain + one influence evaluation + one vertex solve. The
    accept-only-if-improved guard means the returned model is never worse than
    the nominal one, whatever the frozen-structure approximation got wrong.

    Geometry only changes the inner argmax (:func:`directed_delta`); the
    re-linearise / verify / accept loop is identical, so the two arms differ by
    exactly the set and nothing else.
    """
    X, y = model_data.X_train, model_data.y_train
    params = dict(m_params or {})
    params["random_state"] = seed          # match ScenarioBank: delta is the only variation

    best_delta = np.zeros(len(y))
    best_model = retrain_on_perturbed(X, y, best_delta, m_type, params)
    best_val = _predict(best_model, x_star)
    trace = [best_val]

    for _ in range(n_steps):
        g = influence_vector(best_model, X, x_star)
        delta = directed_delta(g, eps, gamma, model_data, geometry, radius)
        if np.allclose(delta, best_delta):
            break
        model = retrain_on_perturbed(X, y, delta, m_type, params)
        val = _predict(model, x_star)
        trace.append(val)
        if val <= best_val + 1e-12:
            break                          # no verified gain: stop rather than drift
        best_delta, best_model, best_val = delta, model, val
    return best_delta, best_model, best_val, trace


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
def build_instance(config, args):
    if args.problem == "synthetic":
        syn = config.get("synthetic", {})
        return synthetic_nonlinear(
            n_train=syn.get("n_train", 200), n_test=syn.get("n_test", 100),
            noise_std=syn.get("noise_std", 0.1), seed=config.get("seed", 42),
        )
    cv_configs = gt_configs = None
    if args.cv_configs and os.path.exists(args.cv_configs):
        with open(args.cv_configs) as fh:
            cv_configs = json.load(fh)
        print(f"Constraint models from {args.cv_configs}")
    if args.gt_cv_configs and os.path.exists(args.gt_cv_configs):
        with open(args.gt_cv_configs) as fh:
            gt_configs = json.load(fh)
    return gastric_cancer(
        fixed_constraint_configs=cv_configs,
        fixed_gt_ensemble_configs=gt_configs,
    )


def outcome_rows(instance, model_type, model_params):
    """[(c_idx, name, model_data, m_type, m_params, is_constraint)] in bank order."""
    rows, seen, config_idx = [], set(), 0
    for c_idx, constraint in enumerate(instance.constraints):
        for md in constraint.models_data:
            m_type, m_params = resolve_constraint_config(
                instance, config_idx, model_type, model_params
            )
            if id(md) not in seen:
                seen.add(id(md))
                rows.append((c_idx, constraint.name, md, m_type, m_params,
                             md.obj_weight == 0.0, constraint.rhs))
            config_idx += 1
    return rows


# ---------------------------------------------------------------------------
# Part A/B -- influence, random vs directed
# ---------------------------------------------------------------------------
def probe_influence(instance, bank, rows, anchor_x, uset, seed, fw_steps):
    records = []
    for (c_idx, name, md, m_type, m_params, is_con, rhs) in rows:
        eps = uset.eps(bank.scales[id(md)])
        n = len(md.y_train)
        m_rows = uset.n_moved(n)
        gamma, radius = budget_at(uset, eps, n, 1.0)
        # Trained the bank's way (random_state pinned), so the only difference
        # between this and any bank member is the label shift.
        nominal_model = retrain_on_perturbed(
            md.X_train, md.y_train, np.zeros(n), m_type,
            {**(m_params or {}), "random_state": seed},
        )
        for a_idx, x_star in enumerate(anchor_x):
            t0 = time.time()
            f0 = _predict(nominal_model, x_star)

            try:
                g = influence_vector(nominal_model, md.X_train, x_star)
            except NotImplementedError as exc:
                print(f"  [skip] {name}: {exc}", flush=True)
                break
            nnz = int(np.count_nonzero(np.abs(g) > 1e-15))

            # frozen-structure prediction of the directed vertex, then the truth
            delta_d = directed_delta(g, eps, gamma, md, uset.geometry, radius)
            f_frozen = f0 + float(g @ delta_d)
            model_d = retrain_on_perturbed(
                md.X_train, md.y_train, delta_d, m_type,
                {**(m_params or {}), "random_state": seed},
            )
            f_directed = _predict(model_d, x_star)

            _, _, f_fw, trace = frank_wolfe_adversary(
                md, m_type, m_params, x_star, eps, gamma, seed, fw_steps,
                uset.geometry, radius,
            )

            preds = [_predict(bank.model(md, b), x_star) for b in range(len(bank))]
            f_bank = float(np.max(preds))

            records.append({
                "constraint": name, "c_idx": c_idx, "is_constraint": is_con,
                "anchor": a_idx, "rhs": rhs, "eps_c": eps, "n_train": n,
                "geometry": uset.geometry, "radius_c": radius,
                "m_budget_rows": m_rows, "nnz_g": nnz,
                # An ellipsoid has no L1 face, so nothing can "bind" in that sense;
                # the radius is always fully spent. Recorded as NA rather than
                # False so the two runs' summaries stay readable side by side.
                "budget_binds": (None if uset.geometry == "ellipsoid"
                                 else bool(nnz > m_rows)),
                "f_nominal": f0,
                "d_bank_best": f_bank - f0,
                "d_directed_frozen": f_frozen - f0,
                "d_directed_retrained": f_directed - f0,
                "d_fw": f_fw - f0,
                # every delta expressed as a fraction of the +eps_c anchor
                "frac_bank": (f_bank - f0) / eps,
                "frac_directed": (f_directed - f0) / eps,
                "frac_fw": (f_fw - f0) / eps,
                "bank_mean_delta": float(np.mean(preds)) - f0,
                "fw_steps_used": len(trace) - 1,
                "seconds": time.time() - t0,
            })
            budget_note = (f"R={radius:.4f}" if uset.geometry == "ellipsoid"
                           else f"m={m_rows}, binds={nnz > m_rows}")
            print(f"  {name} anchor{a_idx}: eps={eps:.4f} nnz(g)={nnz} "
                  f"({budget_note})  "
                  f"bank={f_bank - f0:+.4f} directed={f_directed - f0:+.4f} "
                  f"fw={f_fw - f0:+.4f}  [frac of eps: {(f_bank - f0)/eps:.2f} / "
                  f"{(f_directed - f0)/eps:.2f} / {(f_fw - f0)/eps:.2f}]", flush=True)
    return records


# ---------------------------------------------------------------------------
# Part C -- does the product-set adversary leave anything feasible?
# ---------------------------------------------------------------------------
def probe_feasibility(instance, bank, rows, master, anchors, anchor_x, uset,
                      seed, budget_fracs, fw_steps):
    """Impose every constraint's OWN worst case at once; sweep the budget."""
    records = []
    con_rows = [r for r in rows if r[5]]        # learned inequalities only
    # IncrementalMaster.add_scenario caches embedded models by id(model) and does
    # NOT hold a reference. Production CP is safe because ScenarioBank owns every
    # model; here they would be garbage-collected right after the call, and CPython
    # reuses the id -- so a later cut silently picks up an earlier cut's embedded
    # expression. Keep them alive.
    held = []
    for a_idx, (anchor, x_star) in enumerate(zip(anchors, anchor_x)):
        for t in budget_fracs:
            added = []
            for (c_idx, name, md, m_type, m_params, _, rhs) in con_rows:
                eps = uset.eps(bank.scales[id(md)])
                n = len(md.y_train)
                gamma, radius = budget_at(uset, eps, n, t)
                if t <= 0:
                    model = retrain_on_perturbed(
                        md.X_train, md.y_train, np.zeros(n), m_type,
                        {**(m_params or {}), "random_state": seed},
                    )
                else:
                    _, model, _, _ = frank_wolfe_adversary(
                        md, m_type, m_params, x_star, eps, gamma, seed, fw_steps,
                        uset.geometry, radius,
                    )
                held.append(model)
                master.add_scenario(c_idx, [(md.weight, model)], rhs)
                added.append(master.n_models - 1)

            _fix_anchor_context(master, instance, anchor)
            master.opt.optimize()
            status = int(master.opt.Status)
            feasible = status == 2                     # GRB.OPTIMAL
            obj = float(master.opt.ObjVal) if feasible else float("nan")

            for s in reversed(added):
                master.remove_scenario(s)
            master.opt.update()

            records.append({
                "anchor": a_idx, "budget_frac_t": t, "n_cuts": len(added),
                "status": status, "feasible": feasible, "objective": obj,
            })
            print(f"  anchor{a_idx} t={t:.2f}: "
                  f"{'FEASIBLE obj=%.4f' % obj if feasible else 'INFEASIBLE'}",
                  flush=True)
    return records


# ---------------------------------------------------------------------------
# Part D -- does a directed per-outcome loop survive cut accumulation?
# ---------------------------------------------------------------------------
def probe_loop(instance, rows, master, anchors, ctx_bounds, uset, bank, seed,
               n_iters, budget_frac_t, fw_steps):
    """CP's loop shape with a DIRECTED PER-OUTCOME oracle and no safety net.

    Part C imposes one round of cuts at the nominal x*, which says nothing about
    the regime that actually matters: cuts ACCUMULATE and are never removed, so
    infeasibility (if it comes) comes at iteration 12, not iteration 1.

    Deliberately omitted, because they are the things under debate:
      - tau and the d0 stopping rule -- there is no bank to take a quantile over
        once the adversary is directed rather than sampled;
      - rollback / permanent scenario rejection -- there is no finite pool to
        reject from, which is exactly the open question;
      - the coverage cap cp_alpha.
    So this reports the RAW failure mode: which iteration first loses a protected
    anchor, and whether the objective is still moving when it does.

    Cadence: one cut per OUTCOME per iteration (the per-outcome analogue of
    cutting one whole scenario), each generated at the anchor where that outcome
    is most violated under its own worst case.
    """
    con_rows = [r for r in rows if r[5]]
    protected = set(range(len(anchors)))
    records, total_cuts, prev_x = [], 0, None
    # See probe_feasibility: add_scenario caches by id(model) without holding a
    # reference, so every cut model must outlive the loop or its embedding is
    # silently aliased to a garbage-collected predecessor.
    held = []

    for k in range(n_iters):
        t0 = time.time()

        # -- solve the current master at every anchor -------------------------
        x_stars, feasible_idx = {}, set()
        objs = []
        for a_idx, anchor in enumerate(anchors):
            _fix_anchor_context(master, instance, anchor)
            master.opt.optimize()
            if int(master.opt.Status) == 2:
                feasible_idx.add(a_idx)
                x_stars[a_idx] = np.array([v.X for v in master.x])
                objs.append(float(master.opt.ObjVal))
        _restore_context_bounds(master, ctx_bounds)

        # How far did the cuts actually move the incumbent? A cut that is added
        # but leaves x* untouched is doing nothing, and that is invisible in the
        # objective alone.
        dx = float("nan")
        if prev_x is not None and feasible_idx:
            shared = sorted(feasible_idx & set(prev_x))
            if shared:
                dx = float(max(np.linalg.norm(x_stars[a] - prev_x[a]) for a in shared))
        prev_x = dict(x_stars)

        lost = sorted(protected - feasible_idx)
        if not feasible_idx:
            records.append({
                "iteration": k, "n_feasible_anchors": 0,
                "n_protected": len(protected), "protected_intact": False,
                "max_dx": float("nan"),
                "n_lost_anchors": len(lost), "cuts_added": 0,
                "cuts_total": total_cuts, "max_norm_exceed": float("nan"),
                "mean_objective": float("nan"), "worst_objective": float("nan"),
                "converged": False, "seconds": time.time() - t0,
            })
            print(f"  iter {k}: MASTER INFEASIBLE AT EVERY ANCHOR "
                  f"(cuts={total_cuts}) -- stopping", flush=True)
            break

        # -- one directed cut per outcome, at that outcome's worst anchor -----
        added, max_exceed = 0, 0.0
        for (c_idx, name, md, m_type, m_params, _, rhs) in con_rows:
            eps = uset.eps(bank.scales[id(md)])
            gamma, radius = budget_at(uset, eps, len(md.y_train), budget_frac_t)
            best = None
            for a_idx in sorted(feasible_idx):
                _, model, val, _ = frank_wolfe_adversary(
                    md, m_type, m_params, x_stars[a_idx], eps, gamma, seed,
                    fw_steps, uset.geometry, radius,
                )
                exceed = (md.weight * val - rhs) / max(1.0, abs(rhs))
                if best is None or exceed > best[0]:
                    best = (exceed, model)
            if best is not None and best[0] > 1e-6:
                held.append(best[1])
                master.add_scenario(c_idx, [(md.weight, best[1])], rhs)
                added += 1
                max_exceed = max(max_exceed, best[0])
        total_cuts += added

        converged = added == 0
        records.append({
            "iteration": k, "n_feasible_anchors": len(feasible_idx),
            "n_protected": len(protected),
            "protected_intact": not lost, "n_lost_anchors": len(lost),
            "max_dx": dx,
            "cuts_added": added, "cuts_total": total_cuts,
            "max_norm_exceed": max_exceed,
            "mean_objective": float(np.mean(objs)),
            "worst_objective": float(np.max(objs)),
            "converged": converged, "seconds": time.time() - t0,
        })
        print(f"  iter {k}: anchors {len(feasible_idx)}/{len(protected)} feasible"
              f"{' (LOST %s)' % lost if lost else ''}  cuts +{added} "
              f"(total {total_cuts})  max_exceed={max_exceed:.5f}  "
              f"max_dx={dx:.4g}  "
              f"mean_obj={np.mean(objs):.4f}  [{time.time() - t0:.1f}s]", flush=True)
        if converged:
            print(f"  converged at iteration {k}: no outcome violated at any x*",
                  flush=True)
            break

    return records


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=["gastric", "synthetic"], default="gastric")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--quick", action="store_true",
                   help="B=10, 2 anchors, 1 FW step -- local smoke test")
    p.add_argument("--n-scenarios", type=int, default=None, help="B (default: config)")
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--fw-steps", type=int, default=3)
    p.add_argument("--budget-fracs", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.0])
    p.add_argument("--skip-feasibility", action="store_true",
                   help="Part A/B only (no Gurobi master solves)")
    p.add_argument("--loop-iters", type=int, default=8,
                   help="Part D: directed per-outcome CP loop iterations (0 = skip)")
    p.add_argument("--loop-budget-fracs", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.1],
                   help="Part D: adversary budgets to sweep (fresh master each)")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    p.add_argument("--gt-cv-configs", default="results/cv/gastric_gt_ensemble_configs.json")
    p.add_argument("--geometry", choices=list(GEOMETRIES), default=None,
                   help="override uncertainty.geometry; run once per value to "
                        "compare the two sets at matched L2 size")
    p.add_argument("--outdir", default="results/adversary_probe")
    args = p.parse_args()

    config = load_config(args.config)
    cp_cfg = config.get("methods", {}).get("cp", {})
    from src.methods.nominal import resolve_mip_gap
    mip_gap = resolve_mip_gap(config)
    seed = int(config.get("uncertainty", {}).get("bootstrap_seed", 42))
    model_type = config["default_model"]["type"]
    model_params = config["default_model"]["params"]

    B = args.n_scenarios or int(cp_cfg.get("n_scenarios", 200))
    n_anchors = args.n_anchors or int(cp_cfg.get("n_anchors", 15))
    fw_steps = args.fw_steps
    loop_iters = args.loop_iters
    if args.quick:
        B, n_anchors, fw_steps = 10, 2, 1
        loop_iters = min(loop_iters, 3)

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 70)
    print(f"ADVERSARY PROBE  problem={args.problem}  B={B}  anchors={n_anchors}  "
          f"fw_steps={fw_steps}")
    print("=" * 70, flush=True)

    instance = build_instance(config, args)
    rows = outcome_rows(instance, model_type, model_params)
    uset = uncertainty_set_from_config(config)
    if args.geometry:
        uset = _dc_replace(uset, geometry=args.geometry)
    print(f"D: eps_0={uset.eps_0} budget_frac={uset.budget_frac} "
          f"coherent={uset.coherent} scale_stat={uset.scale_stat} "
          f"geometry={uset.geometry}", flush=True)
    if uset.geometry == "ellipsoid":
        print("   ellipsoid: R_c = sqrt(m)*eps_c, the L2 length of the box vertex, "
              "so the two sets are\n   matched in size. Expect frac_directed/frac_fw "
              "to RISE (Cauchy-Schwarz) while\n   frac_bank stays put -- random "
              "draws are equally uninformative under both.", flush=True)

    print("\nBuilding master (nominal cuts) ...", flush=True)
    master, model_config_map = _build_master_with_nominal(
        instance, model_type, model_params, rho=0.0,
        robustify_objective=bool(cp_cfg.get("robustify_objective", False)),
        mip_gap=mip_gap,
    )
    anchors, ctx_bounds = _setup_anchors(
        instance, master, None, cp_cfg.get("anchor_source", "train"),
        n_anchors, cp_cfg.get("anchor_method", "kmedoids"), seed,
    )

    # x* per anchor from the NOMINAL master -- the incumbent the adversary attacks.
    print(f"Solving nominal at {len(anchors)} anchor(s) ...", flush=True)
    anchor_x, kept, embedded_lhs = [], [], []
    for a_idx, anchor in enumerate(anchors):
        _fix_anchor_context(master, instance, anchor)
        master.opt.optimize()
        if int(master.opt.Status) == 2:
            anchor_x.append(np.array([v.X for v in master.x]))
            kept.append(anchor)
            # LHS the MIP believes, for each nominal cut: rhs - slack. Compared
            # against sklearn's own prediction below -- any gap is embedding
            # error, and a cut smaller than the gap cannot bite.
            embedded_lhs.append([
                (c.RHS - c.Slack) if c is not None else float("nan")
                for c in master.scenario_constrs
            ])
        else:
            print(f"  anchor{a_idx}: nominal infeasible -- dropped "
                  f"(not in the protected set either)", flush=True)
    _restore_context_bounds(master, ctx_bounds)
    anchors = kept
    print(f"  {len(anchor_x)} anchor(s) servable under the nominal fit", flush=True)

    # ---- embedding fidelity -------------------------------------------------
    # embed.py re-encodes each trained model as MIP constraints. If that encoding
    # disagrees with sklearn at x*, then any cut whose sklearn-measured violation
    # is SMALLER than the disagreement is vacuous in MIP space -- it will be added
    # and will not move x*. This bounds how small a separation tolerance can mean
    # anything, so it is measured before anything is read off the cut loop.
    print("\nEmbedding fidelity (|sklearn - embedded| at each nominal x*):",
          flush=True)
    con_only = [r for r in rows if r[5]]
    worst_gap = 0.0
    for j, (c_idx, name, md, m_type, m_params, _, rhs) in enumerate(con_only):
        # MUST match _train_nominal_with_configs exactly -- m_params untouched.
        # Overriding random_state here (as the bank does) trains a DIFFERENT
        # forest for stochastic learners (xgb subsample/colsample), and the
        # comparison then measures seed noise, not embedding error.
        nm = train_model(md.X_train, md.y_train, m_type, m_params)
        gaps = []
        for a_idx, xs in enumerate(anchor_x):
            if j < len(embedded_lhs[a_idx]):
                gaps.append(abs(md.weight * _predict(nm, xs) - embedded_lhs[a_idx][j]))
        if gaps:
            worst_gap = max(worst_gap, max(gaps))
            print(f"  {name}: max={max(gaps):.6f} mean={np.mean(gaps):.6f} "
                  f"(rhs={rhs})", flush=True)
    print(f"  WORST GAP = {worst_gap:.6f}  -- cuts violated by less than this "
          f"cannot bite", flush=True)
    print("  (expect ~1e-7 or less: float32 leaf-value storage in XGBoost. Anything "
          "larger\n   means the embedding regressed -- run experiments/"
          "verify_embedding.py.)", flush=True)

    print(f"\nBuilding scenario bank (B={B}) ...", flush=True)
    bank = ScenarioBank(instance, model_config_map, uset, n_scenarios=B, seed=seed)

    print("\n--- Part A/B: random bank vs directed adversary ---", flush=True)
    inf_records = probe_influence(instance, bank, rows, anchor_x, uset, seed, fw_steps)

    feas_records = []
    if not args.skip_feasibility:
        print("\n--- Part C: all constraints at their own worst case ---", flush=True)
        feas_records = probe_feasibility(
            instance, bank, rows, master, anchors, anchor_x, uset, seed,
            args.budget_fracs, fw_steps,
        )

    loop_records = []
    for t in (args.loop_budget_fracs if loop_iters > 0 else []):
        print(f"\n--- Part D: directed per-outcome CP loop "
              f"({loop_iters} iters, t={t}) ---", flush=True)
        # A FRESH master per budget: Part C added and removed scenarios, and each
        # sweep point must start from the same nominal base CP would.
        loop_master, _ = _build_master_with_nominal(
            instance, model_type, model_params, rho=0.0,
            robustify_objective=bool(cp_cfg.get("robustify_objective", False)),
            mip_gap=mip_gap,
        )
        _, loop_ctx_bounds = _setup_anchors(
            instance, loop_master, None, cp_cfg.get("anchor_source", "train"),
            n_anchors, cp_cfg.get("anchor_method", "kmedoids"), seed,
        )
        for rec in probe_loop(instance, rows, loop_master, anchors,
                              loop_ctx_bounds, uset, bank, seed, loop_iters,
                              t, fw_steps):
            rec["budget_frac_t"] = t
            loop_records.append(rec)

    import pandas as pd
    # Geometry goes in the filename so a side-by-side pair of runs does not
    # overwrite itself. box_l1 keeps the historical names.
    tag = args.problem + ("_quick" if args.quick else "")
    if uset.geometry != "box_l1":
        tag += f"_{uset.geometry}"
    inf_path = os.path.join(args.outdir, f"{tag}_influence.csv")
    pd.DataFrame(inf_records).to_csv(inf_path, index=False)
    print(f"\nWrote {inf_path}")
    if feas_records:
        feas_path = os.path.join(args.outdir, f"{tag}_feasibility.csv")
        pd.DataFrame(feas_records).to_csv(feas_path, index=False)
        print(f"Wrote {feas_path}")
    if loop_records:
        loop_path = os.path.join(args.outdir, f"{tag}_loop.csv")
        pd.DataFrame(loop_records).to_csv(loop_path, index=False)
        print(f"Wrote {loop_path}")

    # ---- summary -----------------------------------------------------------
    df = pd.DataFrame(inf_records)
    if not df.empty:
        print("\n" + "=" * 70)
        print("SUMMARY -- prediction shift at x*, as a fraction of eps_c")
        print("=" * 70)
        agg = df.groupby("constraint").agg(
            eps_c=("eps_c", "first"),
            nnz_g=("nnz_g", "mean"),
            m_rows=("m_budget_rows", "first"),
            binds=("budget_binds", "mean"),
            bank=("frac_bank", "mean"),
            directed=("frac_directed", "mean"),
            fw=("frac_fw", "mean"),
        )
        print(agg.round(4).to_string())
        gain = (df["frac_fw"] - df["frac_bank"]).mean()
        print(f"\nMean directed-over-bank gain: {gain:+.4f} eps_c")
        print("(> 0 means the random bank understates the worst case at x*.)")
    if feas_records:
        fd = pd.DataFrame(feas_records)
        print("\n" + "=" * 70)
        print("SUMMARY -- anchors feasible with every constraint at its own worst case")
        print("=" * 70)
        print(fd.groupby("budget_frac_t")["feasible"].agg(["mean", "sum", "count"])
                .rename(columns={"mean": "frac_feasible", "sum": "n_feasible"})
                .round(4).to_string())
    if loop_records:
        ld = pd.DataFrame(loop_records)
        print("\n" + "=" * 70)
        print("SUMMARY -- directed per-outcome loop (no tau, no rollback)")
        print("=" * 70)
        print(ld[["budget_frac_t", "iteration", "n_feasible_anchors",
                  "protected_intact", "cuts_added", "cuts_total",
                  "max_norm_exceed", "max_dx", "mean_objective"]]
              .round(5).to_string(index=False))
        print("\nPer-budget outcome:")
        for t, grp in ld.groupby("budget_frac_t"):
            broke = grp[~grp["protected_intact"]]
            if not broke.empty:
                verdict = (f"protected set broken at iteration "
                           f"{int(broke['iteration'].iloc[0])} "
                           f"({int(broke['cuts_total'].iloc[0])} cuts)")
            elif bool(grp["converged"].iloc[-1]):
                verdict = (f"CONVERGED at iteration "
                           f"{int(grp['iteration'].iloc[-1])}, protected set intact")
            else:
                verdict = (f"intact through {len(grp)} iters, not converged "
                           f"(max_exceed {grp['max_norm_exceed'].iloc[-1]:.5f})")
            print(f"  t={t:<5} {verdict}")
        print("\nThe largest t that converges with the protected set intact is the\n"
              "budget a directed per-outcome adversary can actually be run at.\n"
              "If no t survives, the product set is too conservative for gastric\n"
              "and the fix is elastic cuts, not a rollback analogue.")


if __name__ == "__main__":
    main()

"""One place where a method name becomes a solver ``partial``.

Every runner used to keep its own copy of the same four builders -- synthetic
fixed-knob (``run_all.run_experiment``), synthetic knob-parameterized
(``run_sweep._synth_build``), gastric fixed-knob
(``run_chemo_robust._build_solvers``) and gastric knob-parameterized
(``run_chemo_robust._method_build_map``, which the rho sweep drives). The four
copies repeated the same cross-cutting decisions -- the one ``mip_gap`` from
``resolve_mip_gap``, ``cp_alpha`` pinned at 0, the shared ``uncertainty_set`` --
so a change to one of them had to land in four places or silently reach only some
problems. The one-MIP-gap fix (2026-08-20) is the worked example: it had to be
made four times.

The problem-specific half stays with the problem. A **settings resolver** turns a
config into the flat dict below (``run_chemo_robust._resolve_run_settings`` for
gastric, :func:`synth_settings` for synthetic/reactor), and the *ranges* /
strength maps stay in ``run_chemo_robust``. Only the argument lists live here.

``settings`` keys read by :func:`build_method` and :func:`cp_solver`::

    shared      mip_gap, bootstrap_seed, embedding_mode, rf_alpha, uncertainty_set
    wrapper     wrapper_n_estimators, bootstrap_frac, wrapper_scenario_source,
                wrapper_robustify_objective
    robust_reg  robust_reg_budget_frac, robust_reg_K
    cp          cp_max_iterations, cp_k_neighbors_frac, cp_k_neighbors_min,
                cp_n_candidates, cp_anchor_source, cp_n_anchors, cp_anchor_method,
                cp_distance, cp_dist_tol, cp_trace_path, cp_robustify_objective,
                cp_eval_mode, cp_nearest_distance, cp_cut_eviction,
                cp_scenario_source, cp_n_scenarios, cp_d0_quantile,
                cp_tolerance_basis, cp_objective_monotone, cp_cut_whole_scenario,
                cp_separation, cp_cut_rollback
    cmicl       cmicl_cal_frac, cmicl_width_model_type, cmicl_width_model_params,
                cmicl_width_floor_frac, cmicl_multiplicity,
                cmicl_robustify_objective
    margin      margin_scale_stat

``cmicl`` and ``margin`` are the two methods that read no ``uncertainty_set``.
C-MICL's tightening comes from held-out residuals and the margin baseline's from
a fitted dial, neither from the shared D (see :mod:`src.methods.cmicl`,
:mod:`src.methods.margin`), so both are flat in rho by construction. Both still
take ``bootstrap_seed``: it moves C-MICL's calibration SPLIT and the margin's
fold scheme for ``scale(y_c)`` -- the same role the seed plays for the other
methods' draws, so a multi-seed sweep varies every method's own randomness.
"""

from functools import partial

from src.methods.nominal import resolve_mip_gap, solve_nominal
from src.methods.robust_regression import solve_robust_regression
from src.methods.wrapper import solve_wrapper, solve_tree_violation_wrapper
from src.methods.cp import solve_cp
from src.methods.cmicl import solve_cmicl
from src.methods.margin import solve_margin
from src.methods.uncertainty import uncertainty_set_from_config


def cp_solver(settings, model_type, model_params, cp_dist_tol_rel=None):
    """Build the CP solver partial -- the ONE place CP's argument list is written.

    Single-lever CP: ``cp_alpha`` is pinned at 0 here (the coverage cap is not a
    tunable) -- cuts that would break a training anchor are evicted/rolled back
    (``cut_eviction``), keeping the training set feasible, so the distance
    tolerance is the ONLY robustness knob.

    ``cp_dist_tol_rel`` (tau) is that knob: the tolerance becomes tau * d0, where
    d0 is the problem's own iteration-0 distance quantile, so ONE tau grid
    transfers across datasets/problems. ``None`` falls back to the ABSOLUTE
    ``methods.cp.dist_tol``, which is what a one-off solve at the config's own
    settings uses.
    """
    return partial(
        solve_cp, model_type=model_type, model_params=model_params, rho=0.0,
        max_iterations=settings["cp_max_iterations"],
        cp_k_neighbors_frac=settings["cp_k_neighbors_frac"],
        cp_k_neighbors_min=settings["cp_k_neighbors_min"],
        cp_n_candidates=settings["cp_n_candidates"],
        seed=settings["bootstrap_seed"], cp_alpha=0.0,  # single-lever: pinned
        cp_dist_tol=settings["cp_dist_tol"],
        cp_dist_tol_rel=cp_dist_tol_rel,
        cp_anchor_source=settings["cp_anchor_source"],
        cp_n_anchors=settings["cp_n_anchors"],
        cp_anchor_method=settings["cp_anchor_method"],
        cp_distance=settings["cp_distance"],
        cp_trace_path=settings["cp_trace_path"],
        cp_robustify_objective=settings["cp_robustify_objective"],
        cp_eval_mode=settings["cp_eval_mode"],
        cp_nearest_distance=settings["cp_nearest_distance"],
        cp_cut_eviction=settings["cp_cut_eviction"],
        cp_scenario_source=settings["cp_scenario_source"],
        cp_n_scenarios=settings["cp_n_scenarios"],
        cp_d0_quantile=settings["cp_d0_quantile"],
        cp_tolerance_basis=settings["cp_tolerance_basis"],
        cp_objective_monotone=settings["cp_objective_monotone"],
        cp_mip_gap=settings["mip_gap"],
        cp_cut_whole_scenario=settings["cp_cut_whole_scenario"],
        cp_separation=settings["cp_separation"],
        cp_cut_rollback=settings["cp_cut_rollback"],
        cp_uncertainty=settings["uncertainty_set"],
    )


def build_method(method, knob, model_type, model_params, settings,
                 bootstrap_cache=None, ensembles_cache=None):
    """Return the solver ``partial`` for ``method`` at its single knob value.

    One knob per method, in its own native units: CP ``tau``
    (``cp_dist_tol_rel``; ``None`` falls back to the absolute
    ``settings["cp_dist_tol"]``), wrapper ``alpha``, robust_reg ``label_eps``,
    cmicl ``alpha`` (conformal miscoverage), margin ``margin`` (RHS tightening
    in unexplained-sd units), robust_param ``rho``, tree_violation ``alpha``.
    ``nominal`` has none and ignores ``knob``.

    ``mip_gap`` is ``settings["mip_gap"]`` for all of them: the methods are
    compared on their objective, so a per-method gap would confound that
    comparison (see CLAUDE.md, Conventions).
    """
    seed = settings["bootstrap_seed"]
    em = settings["embedding_mode"]
    rf_alpha = settings["rf_alpha"]
    mip_gap = settings["mip_gap"]

    if method == "nominal":
        return partial(
            solve_nominal, model_type=model_type, model_params=model_params,
            rho=0.0, embedding_mode=em, rf_alpha=rf_alpha, mip_gap=mip_gap,
        )
    if method == "robust_param":
        # Nominal plus leaf-margin tightening: the knob IS rho.
        return partial(
            solve_nominal, model_type=model_type, model_params=model_params,
            rho=knob, embedding_mode=em, rf_alpha=rf_alpha, mip_gap=mip_gap,
        )
    if method == "tree_violation":
        return partial(
            solve_tree_violation_wrapper, model_type=model_type,
            model_params=model_params, alpha=knob, rho=0.0, mip_gap=mip_gap,
        )
    if method == "robust_reg":
        return partial(
            solve_robust_regression, model_type=model_type,
            model_params=model_params, label_eps=knob,
            budget_frac=settings["robust_reg_budget_frac"],
            K=settings["robust_reg_K"], seed=seed, rho=0.0,
            embedding_mode=em, rf_alpha=rf_alpha,
            uncertainty_set=settings["uncertainty_set"], mip_gap=mip_gap,
        )
    if method == "wrapper":
        return partial(
            solve_wrapper, model_type=model_type, model_params=model_params,
            n_estimators=settings["wrapper_n_estimators"], alpha=knob, seed=seed,
            rho=0.0, bootstrap_cache=bootstrap_cache,
            ensembles_cache=ensembles_cache,
            bootstrap_frac=settings["bootstrap_frac"],
            # Same D and the same seeded draw sequence CP separates over -- the
            # wrapper's P models are a prefix of CP's bank, so alpha=0 here and
            # tau->0 in CP face identical adversaries.
            scenario_source=settings["wrapper_scenario_source"],
            uncertainty_set=settings["uncertainty_set"],
            robustify_objective=settings["wrapper_robustify_objective"],
            mip_gap=mip_gap,
        )
    if method == "cmicl":
        # The one method with no uncertainty_set argument: C-MICL's tightening
        # comes from held-out residuals, not from D. Its knob is the conformal
        # miscoverage level alpha.
        return partial(
            solve_cmicl, model_type=model_type, model_params=model_params,
            alpha=knob, cal_frac=settings["cmicl_cal_frac"],
            width_model_type=settings["cmicl_width_model_type"],
            width_model_params=settings["cmicl_width_model_params"],
            width_floor_frac=settings["cmicl_width_floor_frac"],
            multiplicity=settings["cmicl_multiplicity"],
            seed=seed, rho=0.0,
            robustify_objective=settings["cmicl_robustify_objective"],
            mip_gap=mip_gap,
        )
    if method == "margin":
        # Feasibility-tuned nominal: same fit, same MIP, rhs moved in by
        # knob * scale(y_c) per constraint. No uncertainty_set -- it faces no D,
        # so its rho-sweep curve is flat by construction and it is read as a
        # reference line. `seed` reaches only the fold scheme behind scale(y_c).
        return partial(
            solve_margin, model_type=model_type, model_params=model_params,
            margin=knob, scale_stat=settings["margin_scale_stat"], seed=seed,
            rho=0.0, embedding_mode=em, rf_alpha=rf_alpha, mip_gap=mip_gap,
        )
    if method == "cp":
        return cp_solver(settings, model_type, model_params,
                         cp_dist_tol_rel=knob)
    raise ValueError(f"Unknown method for the build map: {method}")


def synth_settings(config, seed=None):
    """The flat settings dict for the non-contextual problems (synthetic, reactor).

    The gastric counterpart is ``run_chemo_robust._resolve_run_settings``; this one
    has no ``--quick`` overrides and no bootstrap caches to thread through.

    ``seed`` overrides ``uncertainty.bootstrap_seed`` -- the rho sweep's bank axis
    reseeds the ScenarioBank and every model's ``random_state`` through it.

    Three CP settings now come from ``config.yaml`` where the synthetic builder
    used to leave them at ``solve_cp``'s defaults -- ``cp_n_anchors`` (None -> 10),
    ``cp_dist_tol`` (1e-3 -> 0.01) and ``cp_robustify_objective`` (True -> False).
    All three are **inert on the basic separation path**, so no synthetic or
    reactor number moves. Checked on both instances (2026-08-21), not merely
    argued: ``context_var_indices`` is empty, so ``_get_anchor_rows`` returns
    ``None`` and ``anchors=[None]`` at either anchor count; ``_BasicSeparation``
    has no ``dist_tol`` parameter to receive; and no model carries
    ``obj_weight != 0``, so the epigraph branch is unreachable. Every OTHER
    argument of every method is byte-identical to the pre-merge builders.
    """
    unc = config["uncertainty"]
    cp_cfg = config["methods"].get("cp", {})
    wrap_cfg = config["methods"].get("wrapper", {})
    rr_cfg = config["methods"].get("robust_reg", {})
    cm_cfg = config["methods"].get("cmicl", {})
    mg_cfg = config["methods"].get("margin", {})

    return {
        "mip_gap": resolve_mip_gap(config),      # one gap for every method
        "bootstrap_seed": (unc.get("bootstrap_seed", 42) if seed is None
                           else seed),
        "embedding_mode": config["methods"].get("embedding_mode", "hard"),
        "rf_alpha": config["methods"].get("chemo_wrapper", {}).get("alpha", 0.25),
        # The shared uncertainty set D -- one object handed to cp, wrapper and
        # robust_reg, so a difference between them is a difference in METHOD.
        "uncertainty_set": uncertainty_set_from_config(config),
        "wrapper_n_estimators": wrap_cfg.get("n_estimators", 20),
        "bootstrap_frac": unc.get("bootstrap_frac", 0.5),
        "wrapper_scenario_source": wrap_cfg.get("scenario_source", "noise"),
        "wrapper_robustify_objective": wrap_cfg.get("robustify_objective", False),
        "robust_reg_budget_frac": rr_cfg.get("budget_frac", 0.5),
        "robust_reg_K": rr_cfg.get("K", 5),
        "cp_max_iterations": cp_cfg.get("max_iterations", 20),
        "cp_k_neighbors_frac": unc.get("cp_k_neighbors_frac", 0.1),
        "cp_k_neighbors_min": unc.get("cp_k_neighbors_min", 100),
        "cp_n_candidates": unc.get("cp_n_candidates", 20),
        "cp_anchor_source": cp_cfg.get("anchor_source", "train"),
        "cp_n_anchors": cp_cfg.get("n_anchors", 10),
        "cp_anchor_method": cp_cfg.get("anchor_method", "kmedoids"),
        "cp_distance": cp_cfg.get("distance", "full"),
        "cp_dist_tol": cp_cfg.get("dist_tol", 1e-3),
        "cp_trace_path": cp_cfg.get("trace_path") or None,
        "cp_robustify_objective": cp_cfg.get("robustify_objective", False),
        "cp_eval_mode": cp_cfg.get("eval_mode", "global"),
        "cp_nearest_distance": cp_cfg.get("nearest_distance", "context"),
        "cp_cut_eviction": cp_cfg.get("cut_eviction", "reject"),
        "cp_scenario_source": cp_cfg.get("scenario_source", "noise"),
        "cp_n_scenarios": cp_cfg.get("n_scenarios", 200),
        "cp_d0_quantile": cp_cfg.get("d0_quantile", 0.9),
        "cp_tolerance_basis": cp_cfg.get("tolerance_basis", "scale"),
        "cp_objective_monotone": cp_cfg.get("objective_monotone", False),
        "cp_cut_whole_scenario": cp_cfg.get("cut_whole_scenario", True),
        "cp_separation": cp_cfg.get("separation", "auto"),
        "cp_cut_rollback": cp_cfg.get("cut_rollback", "forward"),
        "cmicl_cal_frac": cm_cfg.get("cal_frac", 0.25),
        "cmicl_width_model_type": cm_cfg.get("width_model_type") or None,
        "cmicl_width_model_params": cm_cfg.get("width_model_params") or None,
        "cmicl_width_floor_frac": cm_cfg.get("width_floor_frac", 0.05),
        "cmicl_multiplicity": cm_cfg.get("multiplicity", "none"),
        "cmicl_robustify_objective": cm_cfg.get("robustify_objective", False),
        # Defaults to uncertainty.scale_stat: the margin is quoted in the same
        # units as rho and tau, so it must be the same estimator.
        "margin_scale_stat": mg_cfg.get("scale_stat")
                             or unc.get("scale_stat", "oof_sd"),
    }

"""One place where a config becomes a solver ``partial``.

Every runner used to keep its own copy of the same four builders -- synthetic
fixed-knob (``run_all.run_experiment``), synthetic knob-parameterized
(``run_sweep._synth_build``), gastric fixed-knob
(``run_chemo_robust._build_solvers``) and gastric knob-parameterized
(``run_chemo_robust._method_build_map``, which the sweeps drive). The four copies
repeated the same cross-cutting decisions -- the one ``mip_gap`` from
``resolve_mip_gap``, ``cp_alpha`` pinned at 0, the shared ``uncertainty_set`` --
so a change to one of them had to land in four places or silently reach only some
problems. The one-MIP-gap fix (2026-08-20) is the worked example: it had to be
made four times.

**Both halves now live here** (it was ``experiments/method_builders.py``, holding
only the argument lists, while the two settings resolvers sat one in this file and
one in ``run_chemo_robust``). The sweeps needed the gastric resolver, so they
imported a *runner* to reach it and could not run without it. Nothing moved but
the text: :func:`gastric_settings` is ``run_chemo_robust._resolve_run_settings``
and :func:`synth_build` is ``run_sweep._synth_build``, key for key. Only the knob
*ranges* and the strength->knob maps stay in ``run_chemo_robust``, which is the
one caller that calibrates rather than being handed a dial.

Three layers, in call order::

    load_config          src/data/instances.py -- config.yaml as plain data
    *_settings(config)   here -- the flat settings dict below
    *_build(...)         here -- ``build(knob) -> solver_fn``
    build_method         here -- the argument lists

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
                cmicl_multiplicity, cmicl_robustify_objective
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


def cp_solver(settings, model_type, model_params, cp_dist_tol_rel=None,
              cp_alpha=None):
    """Build the CP solver partial -- the ONE place CP's argument list is written.

    Single-lever CP: ``cp_alpha`` is 0 for every result -- cuts that would break a
    protected training anchor are rolled back, keeping the training set feasible,
    so the distance tolerance is the ONLY robustness knob. It is no longer
    hard-coded here, because there is now one experiment that has to move it: the
    coverage-cap ABLATION, which holds tau at tau* and walks cp_alpha to ask
    whether relaxing the cap lifts CP's feasibility ceiling and what that costs in
    solved fraction. ``None`` (the default, and what every runner passes) keeps the
    pinned 0, so no other call site changes.

    Above 0 the cap admits a cut that breaks up to ``alpha`` of the anchors -- CP
    trading one patient's feasibility for another's, which is exactly the property
    the ablation exists to price. It is structurally inert on the single-decision
    problems (synthetic, reactor): those take the basic separation path, which has
    no protected-anchor test to relax.

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
        seed=settings["bootstrap_seed"],
        # Pinned at 0 unless a caller deliberately ablates the coverage cap.
        cp_alpha=0.0 if cp_alpha is None else float(cp_alpha),
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
                 bootstrap_cache=None, ensembles_cache=None, cp_alpha=None):
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

    ``cp_alpha`` reaches CP only, and only the coverage-cap ablation passes it.
    Everywhere else it is ``None`` -> the pinned 0; see :func:`cp_solver`.
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
                         cp_dist_tol_rel=knob, cp_alpha=cp_alpha)
    raise ValueError(f"Unknown method for the build map: {method}")


def synth_settings(config, seed=None):
    """The flat settings dict for the non-contextual problems (synthetic, reactor).

    The gastric counterpart is :func:`gastric_settings`; this one has no
    ``--quick`` overrides and no bootstrap caches to thread through.

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
        "cmicl_multiplicity": cm_cfg.get("multiplicity", "none"),
        "cmicl_robustify_objective": cm_cfg.get("robustify_objective", False),
        # Defaults to uncertainty.scale_stat: the margin is quoted in the same
        # units as rho and tau, so it must be the same estimator.
        "margin_scale_stat": mg_cfg.get("scale_stat")
                             or unc.get("scale_stat", "oof_sd"),
    }


# Every method the gastric runner knows how to build, and `methods_to_run`'s
# default. `cp` self-regulates via its protected-anchor cap; `nominal` has no
# robustness knob at all.
ALL_METHODS = [
    "nominal", "tree_violation", "robust_param", "robust_reg", "wrapper", "cp",
    "cmicl", "margin",
]


def default_gastric_args():
    """The flags :func:`gastric_settings` reads, at FULL-RUN values.

    Each sweep's parser is its own; none of them shares a flag with the gastric
    runner, and their ``--methods`` means "methods to sweep", not
    ``methods_to_run`` (a sweep builds one solver per method by name, so that key
    goes unused). Passing a sweep namespace straight through therefore both crashed
    on ``args.quick`` and would have silently repurposed ``--methods``.

    Full-run settings are what a sweep wants anyway: ``--quick`` shrinks B, the
    anchor count and the iteration cap, which would change what each cell measures.
    """
    import argparse
    return argparse.Namespace(
        quick=False,
        max_test_rows=None,
        methods=None,
        output=None,
        cp_robustify_objective=None,
        cp_eval_mode=None,
    )


def gastric_settings(config, args=None):
    """The flat settings dict for the contextual problem (gastric).

    The counterpart of :func:`synth_settings`, which has no ``--quick`` overrides
    and no bootstrap caches to thread through. ``args=None`` means
    :func:`default_gastric_args` -- the full-run values every sweep uses.
    """
    args = default_gastric_args() if args is None else args
    chemo_cfg = config["methods"].get("chemo", {})
    quick_cfg = chemo_cfg.get("quick", {})
    unc = config["uncertainty"]

    cp_cfg = config["methods"].get("cp", {})

    if args.quick:
        settings = {
            "max_test_rows": quick_cfg.get("max_test_rows", 5),
            "methods_to_run": quick_cfg.get(
                "methods_to_run", ["nominal", "wrapper", "cp"]
            ),
            "constraint_modes": quick_cfg.get("constraint_modes", ["all_constraints"]),
            "n_bootstrap": quick_cfg.get("n_bootstrap", 5),
            "cp_max_iterations": quick_cfg.get("cp_max_iterations", 5),
            "cp_n_candidates": quick_cfg.get("cp_n_candidates", 5),
            "cp_k_neighbors_frac": quick_cfg.get("cp_k_neighbors_frac", 0.05),
            "cp_k_neighbors_min": quick_cfg.get(
                "cp_k_neighbors_min", unc.get("cp_k_neighbors_min", 100)
            ),
            "alpha": quick_cfg.get("alpha", unc.get("alpha", 0.0)),
            "cp_n_anchors": quick_cfg.get("cp_n_anchors", cp_cfg.get("n_anchors", 4)),
            "output_path": "results/gastric/chemo_robust_table6_quick.csv",
            "prescriptions_dir": "results/gastric/prescriptions",
        }
    else:
        settings = {
            "max_test_rows": None,
            "methods_to_run": chemo_cfg.get("methods_to_run", ALL_METHODS),
            "constraint_modes": chemo_cfg.get(
                "constraint_modes", ["all_constraints", "dlt_only"]
            ),
            "n_bootstrap": unc.get("n_bootstrap", 25),
            "cp_max_iterations": cp_cfg.get("max_iterations", 20),
            "cp_n_candidates": unc.get("cp_n_candidates", 20),
            "cp_k_neighbors_frac": unc.get("cp_k_neighbors_frac", 0.1),
            "cp_k_neighbors_min": unc.get("cp_k_neighbors_min", 100),
            "alpha": unc.get("alpha", 0.0),
            "cp_n_anchors": cp_cfg.get("n_anchors", 15),
            "output_path": "results/gastric/chemo_robust_table6.csv",
            "prescriptions_dir": "results/gastric/prescriptions",
        }

    settings["cp_anchor_source"] = cp_cfg.get("anchor_source", "train")
    settings["cp_anchor_method"] = cp_cfg.get("anchor_method", "kmedoids")
    # Off unless methods.cp.trace_path is set: every CP solve rewrites the same
    # file, so over a robustness run only the last loop survives and nothing in it
    # says which realization/RHS/mode it came from. The stdout log carries the same
    # per-iteration numbers in context.
    settings["cp_trace_path"] = cp_cfg.get("trace_path") or None
    settings["cp_distance"] = cp_cfg.get("distance", "full")
    settings["cp_dist_tol"] = cp_cfg.get("dist_tol", 1e-3)
    settings["cp_robustify_objective"] = cp_cfg.get("robustify_objective", True)
    settings["cp_eval_mode"] = cp_cfg.get("eval_mode", "global")
    settings["cp_nearest_distance"] = cp_cfg.get("nearest_distance", "context")
    settings["cp_cut_eviction"] = cp_cfg.get("cut_eviction", "reject")
    settings["cp_scenario_source"] = cp_cfg.get("scenario_source", "noise")
    settings["cp_d0_quantile"] = cp_cfg.get("d0_quantile", 0.9)
    settings["cp_tolerance_basis"] = cp_cfg.get("tolerance_basis", "scale")
    settings["cp_objective_monotone"] = cp_cfg.get("objective_monotone", False)
    # One optimality tolerance for the whole run: CP's cut loop and final solve,
    # the wrapper, robust_reg, nominal, and the prescribe-time re-solve. The
    # methods are compared on their objective, so a per-method gap confounds it.
    settings["mip_gap"] = resolve_mip_gap(config)
    settings["cp_cut_whole_scenario"] = cp_cfg.get("cut_whole_scenario", True)
    # Separation path: "auto" reads it off the bank's coherence (coherent -> one
    # shared draw cut per iteration; incoherent -> the draws ranked per constraint,
    # one model admitted for each). Gastric is where the two differ -- 5 constraints.
    settings["cp_separation"] = cp_cfg.get("separation", "auto")
    settings["cp_cut_rollback"] = cp_cfg.get("cut_rollback", "forward")
    # B: CP embeds one extra scenario per iteration, so it can afford a bank far
    # larger than the wrapper's P (which is embedded in full). --quick shrinks it.
    settings["cp_n_scenarios"] = (
        quick_cfg.get("cp_n_scenarios", 10) if args.quick
        else cp_cfg.get("n_scenarios", 200)
    )
    # The shared uncertainty set D -- one object handed to cp, wrapper and
    # robust_reg, so a difference between them is a difference in METHOD.
    settings["uncertainty_set"] = uncertainty_set_from_config(config)
    settings["calibration_method"] = config.get("calibration", {}).get("method", "cv")
    settings["pareto_center_factors"] = config.get("cv_calibration", {}).get(
        "pareto_center_factors", [0.5, 0.75, 1.0, 1.5, 2.0])

    if args.max_test_rows is not None:
        settings["max_test_rows"] = args.max_test_rows
    if args.methods:
        settings["methods_to_run"] = args.methods
    if args.output:
        settings["output_path"] = args.output
    # CP ablation overrides (default None -> use config.yaml values).
    if getattr(args, "cp_robustify_objective", None) is not None:
        settings["cp_robustify_objective"] = (args.cp_robustify_objective == "true")
    if getattr(args, "cp_eval_mode", None) is not None:
        settings["cp_eval_mode"] = args.cp_eval_mode

    settings["bootstrap_seed"] = unc.get("bootstrap_seed", 42)
    settings["bootstrap_frac"] = unc.get("bootstrap_frac", 0.5)
    settings["embedding_mode"] = config["methods"].get("embedding_mode", "hard")
    settings["rf_alpha"] = config["methods"].get("chemo_wrapper", {}).get("alpha", 0.25)
    _wrap_cfg = config["methods"]["wrapper"]
    # P, under the name the shared builder reads. Gastric takes it from
    # uncertainty.n_bootstrap (--quick shrinks it); the non-contextual problems take
    # methods.wrapper.n_estimators -- see `synth_settings`.
    settings["wrapper_n_estimators"] = settings["n_bootstrap"]
    settings["wrapper_alpha"] = _wrap_cfg.get("alpha", 0.1)
    settings["wrapper_scenario_source"] = _wrap_cfg.get("scenario_source", "noise")
    settings["wrapper_robustify_objective"] = _wrap_cfg.get("robustify_objective", False)
    settings["robust_rho"] = config["methods"].get("robust_param", {}).get("rho", 0.05)
    rr_cfg = config["methods"].get("robust_reg", {})
    settings["robust_reg_label_eps"] = rr_cfg.get("label_eps", 0.1)
    settings["robust_reg_budget_frac"] = rr_cfg.get("budget_frac", 0.5)
    settings["robust_reg_K"] = rr_cfg.get("K", 5)
    # C-MICL (src/methods/cmicl.py). Its dial is the conformal miscoverage alpha;
    # everything below is structural and has one production value. It reads no
    # uncertainty_set -- the only method here that does not face the shared D.
    cm_cfg = config["methods"].get("cmicl", {})
    settings["cmicl_alpha"] = cm_cfg.get("alpha", 0.1)
    settings["cmicl_cal_frac"] = cm_cfg.get("cal_frac", 0.25)
    settings["cmicl_width_model_type"] = cm_cfg.get("width_model_type") or None
    settings["cmicl_width_model_params"] = cm_cfg.get("width_model_params") or None
    settings["cmicl_multiplicity"] = cm_cfg.get("multiplicity", "none")
    settings["cmicl_robustify_objective"] = cm_cfg.get("robustify_objective", False)

    mg_cfg = config["methods"].get("margin", {})
    settings["margin"] = mg_cfg.get("margin", 0.5)
    # The margin is quoted in the same units as rho and tau, so it reads the same
    # label-scale estimator D's radius does unless told otherwise.
    settings["margin_scale_stat"] = (mg_cfg.get("scale_stat")
                                     or unc.get("scale_stat", "oof_sd"))

    calib_cfg = config.get("calibration", {})
    settings["calibrate_to_alpha"] = calib_cfg.get("enabled", True)
    settings["calib_n_grid"] = calib_cfg.get("n_grid", 5)
    settings["calib_wrapper_alpha_max"] = calib_cfg.get("wrapper_alpha_max", 0.5)
    settings["calib_tree_alpha_max"] = calib_cfg.get("tree_alpha_max", 0.5)
    settings["calib_rho_min"] = calib_cfg.get("rho_min", 0.001)
    settings["calib_rho_max"] = calib_cfg.get("rho_max", 0.005)
    settings["calib_robust_reg_eps_max"] = calib_cfg.get("robust_reg_eps_max", 0.3)

    cs_cfg = config.get("conservativeness_sweep", {})
    settings["cs_robust_param_rho_max"] = cs_cfg.get("robust_param_rho_max", 0.03)
    settings["cs_cp_alpha_max"] = cs_cfg.get("cp_alpha_max", 0.3)
    # CP knob is now RELATIVE (tau = fraction of the problem's iter-0 distance d0).
    settings["cs_cp_dist_tol_rel_max"] = cs_cfg.get("cp_dist_tol_rel_max", 1.0)
    settings["cs_cp_dist_tol_rel_min"] = cs_cfg.get("cp_dist_tol_rel_min", 0.1)
    settings["cs_robust_reg_eps_max"] = cs_cfg.get("robust_reg_eps_max", 1.0)
    settings["cs_wrapper_alpha_max"] = cs_cfg.get("wrapper_alpha_max", 0.5)
    return settings


# ---------------------------------------------------------------------------
# build(knob) -> solver_fn
# ---------------------------------------------------------------------------
def synth_build(method, config, model_type, model_params, seed, cp_alpha=None):
    """``build(knob) -> solver_fn`` for a non-contextual problem (synthetic, reactor).

    Single knob per method (CP tau, robust_reg label_eps, wrapper alpha); nominal
    ignores it. CP is single-lever (``cp_alpha=0``) like gastric.

    ``config`` is read fresh on every call: the sweeps hand in a config whose
    ``uncertainty.rho`` they have just overwritten.
    """
    settings = synth_settings(config, seed)
    # cp_alpha stays None (the pinned 0) here: the coverage cap lives in the
    # protected-anchor test on the CONTEXTUAL separation path, and these problems
    # take the basic path, which has no such test. Threaded anyway so the two
    # builders keep the same shape.
    return lambda knob: build_method(method, knob, model_type, model_params,
                                     settings, cp_alpha=cp_alpha)


def gastric_build(method, settings, model_type, model_params,
                  bootstrap_cache=None, ensembles_cache=None):
    """``build(knob) -> solver_fn`` for gastric.

    The counterpart of :func:`synth_build`, differing only in the two caches (the
    bootstrap index sets and the trained replicate ensembles are shared across the
    methods that need them) and in taking a resolved ``settings`` rather than a
    config -- callers mutate a copy of it per cell (the swept ``uncertainty_set``,
    the bank seed, ``cp_alpha``).

    ``cp_alpha`` comes off ``settings`` and is ``None`` -- the pinned 0 -- for every
    runner except the coverage-cap ablation, which sets it to walk it.
    """
    return lambda knob: build_method(
        method, knob, model_type, model_params, settings,
        bootstrap_cache=bootstrap_cache, ensembles_cache=ensembles_cache,
        cp_alpha=settings.get("cp_alpha"),
    )

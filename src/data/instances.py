"""One place where a config becomes a :class:`ProblemInstance`.

Every runner used to build its own. The three instances and the two
CV-selected-model readers lived in ``experiments/run_sweep.py`` (synthetic,
reactor) and ``experiments/run_chemo_robust.py`` (gastric), so the sweeps -- the
current experiments -- imported two *runners* to reach them and could not run
without carrying both. That is what this module removes: the sweeps now depend on
``src/`` alone, and ``run_chemo_robust.py`` is a consumer of this file rather than
a provider to the sweeps.

Nothing here is new. Every default, every seed and every ``fixed_*_config``
argument is byte-identical to the code it was lifted from, so the CV stages and
every committed cell still reproduce.

The pattern each builder follows: the CV selection in
``results/cv/*_selected_configs.json`` wins over ``config.yaml``'s model block
when the file exists, and the winner is passed to the *generator* as
``fixed_constraint_config(s)`` rather than to a solver. That is what makes the
selection reach every method at once -- ``nominal.resolve_constraint_config``
prefers the instance's map over the ``model_type``/``model_params`` arguments, and
``ScenarioBank`` resolves its per-draw refits through the same map.
"""

import json
import os

import yaml


def load_config(path="config.yaml"):
    """The repo's one config reader. Plain data -- callers deep-copy it freely."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Gastric outcome names
# ---------------------------------------------------------------------------
# A property of the instance, not of a runner: these are the learned models
# `gastric_cancer()` builds, in the order `filter_constraints` expects.
ALL_CONSTRAINTS = [
    "dlt_constraint", "blood_constraint", "constitutional_constraint",
    "infection_constraint", "gi_constraint", "os_constraint",
]
DLT_ONLY = ["dlt_constraint", "os_constraint"]


def constraint_names(constraint_mode):
    if constraint_mode == "all_constraints":
        return ALL_CONSTRAINTS
    if constraint_mode == "dlt_only":
        return DLT_ONLY
    raise ValueError(f"Unknown constraint mode: {constraint_mode}")


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
# Written by `run_cv.py --problem synthetic`; the CV-selected embedded model.
SYNTH_CV_CONFIGS = os.path.join("results", "cv", "synthetic_selected_configs.json")
SYNTH_OUTCOME = "synthetic_constraint"


def synth_model_spec(config, path=None, verbose=False):
    """``(model_type, model_params, from_cv)`` for the synthetic embedded model.

    The CV selection wins over ``config.yaml``'s ``model`` block when present; the
    synthetic model was hard-coded ``rf`` (50 trees, depth 5) and had never been
    cross-validated (2026-08-19 deck, next step 2). Returns ``from_cv`` so callers
    can SAY which one they used -- the two train different models on the same data,
    and a resumable score checkpoint keyed only by ``(method@rho, knob)`` would
    otherwise merge them silently (see ``run_rho_sweep._variant_suffix``).
    """
    path = path or SYNTH_CV_CONFIGS
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f).get(SYNTH_OUTCOME)
        if cfg:
            mt = cfg["model_type"]
            mp = cfg.get("model_params", cfg.get("params", {}))
            if verbose:
                print(f"    [synth] CV-selected embedded model: {mt} {mp} "
                      f"(from {path})", flush=True)
            return mt, dict(mp), True
    if verbose:
        print(f"    [synth] no {path}; embedded model from config.yaml: "
              f"{config['default_model']['type']} "
              f"{config['default_model']['params']}", flush=True)
    return config["default_model"]["type"], dict(config["default_model"]["params"]), False


def synth_instance(config, seed=None, cv_path=None, verbose=False):
    """The synthetic instance, carrying the CV-selected embedded model if there is one."""
    from src.data.generate import synthetic_nonlinear
    d = config["synthetic"]
    mt, mp, from_cv = synth_model_spec(config, cv_path, verbose=verbose)
    return synthetic_nonlinear(
        n_train=d["n_train"], n_features=d["n_features"], noise_std=d["noise_std"],
        seed=seed if seed is not None else config["uncertainty"].get("bootstrap_seed", 42),
        fixed_constraint_config=({"model_type": mt, "model_params": mp}
                                 if from_cv else None),
    )


# ---------------------------------------------------------------------------
# Reactor (C-MICL DMA-MR)
# ---------------------------------------------------------------------------
# Written by `run_cv.py --problem reactor`; the CV-selected embedded model.
REACTOR_CV_CONFIGS = os.path.join("results", "cv", "reactor_selected_configs.json")
REACTOR_OUTCOME = "benzene_constraint"


def reactor_model_spec(config, path=None, verbose=False):
    """``(model_type, model_params, from_cv)`` for the reactor embedded model.

    Same contract as :func:`synth_model_spec`: the CV selection wins over the
    ``reactor.model`` block in ``config.yaml`` when present, and ``from_cv`` is
    returned so the caller can scope the sweep cell by which model is in force.
    """
    path = path or REACTOR_CV_CONFIGS
    rc = config.get("reactor", {})
    default_t = rc.get("model", {}).get("type", config["default_model"]["type"])
    default_p = dict(rc.get("model", {}).get("params", config["default_model"]["params"]))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f).get(REACTOR_OUTCOME)
        if cfg:
            mt = cfg["model_type"]
            mp = cfg.get("model_params", cfg.get("params", {}))
            if verbose:
                print(f"    [reactor] CV-selected embedded model: {mt} {mp} "
                      f"(from {path})", flush=True)
            return mt, dict(mp), True
    if verbose:
        print(f"    [reactor] no {path}; embedded model from config.yaml: "
              f"{default_t} {default_p}", flush=True)
    return default_t, default_p, False


def reactor_instance(config, cv_path=None, verbose=False):
    """The DMA-MR instance, carrying the CV-selected embedded model if there is one.

    The ODE dataset is cached on disk (see ``generate._reactor_dataset``), so this
    is cheap after the first call even though each oracle evaluation is a stiff ODE
    solve.
    """
    from src.data.generate import reactor_micl
    rc = config.get("reactor", {})
    mt, mp, from_cv = reactor_model_spec(config, cv_path, verbose=verbose)
    return reactor_micl(
        n_train=int(rc.get("n_train", 1000)),
        noise_std=float(rc.get("noise_std", 2.0)),
        seed=int(config["uncertainty"].get("bootstrap_seed", 42)),
        fixed_constraint_config=({"model_type": mt, "model_params": mp}
                                 if from_cv else None),
    )


# ---------------------------------------------------------------------------
# Gastric
# ---------------------------------------------------------------------------
GASTRIC_CV_CONFIGS = os.path.join("results", "cv", "gastric_selected_configs.json")
GASTRIC_GT_CONFIGS = os.path.join("results", "cv", "gastric_gt_ensemble_configs.json")


def load_gastric_cv_configs(path=None):
    """``(constraint_configs, gt_ensemble_configs)`` -- the FROZEN CV picks.

    The GT path is derived from the constraint path by name substitution, which is
    how every runner has read the pair: ``--cv-configs`` names one file and the
    ensemble file beside it is picked up automatically. A missing file gives
    ``None``, i.e. "let the generator choose", so a fresh clone still runs.
    """
    cv_configs = gt_configs = None
    # `None` means "the default path"; an explicitly EMPTY path means "no configs,
    # let the generator choose", which is what a falsy `--cv-configs` has always
    # meant to the sweeps.
    base = GASTRIC_CV_CONFIGS if path is None else path
    if base and os.path.exists(base):
        with open(base, "r", encoding="utf-8") as f:
            cv_configs = json.load(f)
        gt_path = base.replace("_selected_configs", "_gt_ensemble_configs")
        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_configs = json.load(f)
    return cv_configs, gt_configs


def gastric_instance(cv_configs=None, gt_configs=None):
    """The 416-arm gastric instance under the frozen CV picks.

    Unlike the two single-decision instances this reads no ``config`` at all: the
    cohort, the temporal split and the treatment columns are properties of the
    processed dataset (``src/data/gastric_v11.py``), and the only choices are the
    two model maps.
    """
    from src.data.generate import gastric_cancer
    return gastric_cancer(fixed_constraint_configs=cv_configs,
                          fixed_gt_ensemble_configs=gt_configs)

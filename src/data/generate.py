"""
Data generation for constraint learning experiments.

We need:
1. Training data (X, y) where y = f_true(X) + noise
2. A known ground truth f_true for synthetic experiments
3. A downstream optimization problem: min c'x s.t. f(x) <= b
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any, Dict


@dataclass
class DomainConstraint:
    """Linear constraint: sum(coeffs[j] * x[j]) <= rhs."""
    coeffs: np.ndarray
    rhs: float


@dataclass
class MLModelData:
    """Data for a single ML model used inside a constraint."""
    X_train: np.ndarray        # (n, d_features_for_this_model)
    y_train: np.ndarray        # (n,)
    y_true: Optional[np.ndarray] # (n,) - True noiseless values if available
    weight: float = 1.0        # Coefficient for this model in the constraint (w_i in sum(w_i * f_i(x)) <= b)
    obj_weight: float = 0.0    # Coefficient for this model in the objective
    # Range y_train can plausibly take, used to clip label perturbations (see
    # src/methods/uncertainty.py). Percentile-scored outcomes are (0.0, 1.0);
    # None (synthetic, raw-scale OS) leaves perturbations unclipped.
    label_bounds: Optional[tuple] = None

@dataclass
class LearnedConstraint:
    """A single constraint modeled via one or more ML models that are linearly combined: sum(w_i * f_i(x)) <= rhs."""
    name: str
    models_data: List[MLModelData] # List of datasets/weights, one for each ML model in this constraint
    rhs: float                 # b in sum_i(w_i * f_i(x)) <= b
    f_true: Optional[Callable] = None # Or list of callables, if known


@dataclass
class LabelLink:
    """A label that is a deterministic function of OTHER outcomes' labels.

    Gastric is the only instance with one: ``DLT_PROP = 1 - prod(1 - tox)`` holds
    exactly (2e-16) over the four modeled toxicities, so DLT carries no degree of
    freedom of its own. Perturbing all five labels independently -- which is what
    a per-outcome draw from D does -- produces a relabeling that is not a
    relabeling of any trial: no ``delta in D`` leaves perturbed DLT equal to the
    identity applied to the perturbed components.

    ``derive`` maps ``{source constraint name -> perturbed labels}`` to the target's
    labels, in the target's own label space (percentile ranks on gastric), which
    means it has to undo the source percentile transform, apply the identity on the
    raw scale, and re-percentile. ``baseline`` is ``derive`` at the *unperturbed*
    sources: the shift handed to the target is ``derive(perturbed) - baseline``, not
    ``derive(perturbed) - y_train``, so a zero source shift derives a zero target
    shift **exactly** rather than to within the identity's own float error (which a
    tie in ``percentileofscore`` would otherwise amplify to 1/n).

    ``derive`` must be **row-wise** -- every map inside it elementwise against a
    fixed full-train reference -- so that a fold or subsample can carry the link by
    subsetting ``baseline`` (``cv_calibrate._fold_instance``) rather than rebuilding
    it.

    Consumed by ``ScenarioBank._draw`` under ``uncertainty.derive_linked_labels``.
    """
    target: str                                              # constraint name derived
    sources: List[str]                                       # constraint names it derives from
    derive: Callable[[Dict[str, np.ndarray]], np.ndarray]
    baseline: np.ndarray                                     # derive() at the unperturbed sources
    identity: str = ""                                       # human-readable, for logs


@dataclass
class EvalOutcome:
    """Ground-truth outcome used for Table 6-style prescriptive evaluation."""
    label: str
    name: str
    rhs: float
    gt_fn: Any
    is_survival: bool = False


@dataclass
class ProblemInstance:
    """Complete problem instance for constraint learning with multiple constraints and prescriptive eval."""
    # Data splitting
    X_test: np.ndarray         # (n_test, d_context) - contextual features for evaluation

    # Problem definition
    cost_vector: np.ndarray     # c in min c'x
    variable_lb: np.ndarray     # global lower bounds on x
    variable_ub: np.ndarray     # global upper bounds on x
    n_features: int

    # Indices defining what the optimizer can change vs what is fixed per patient
    decision_var_indices: List[int]
    context_var_indices: List[int]

    # Constraints
    constraints: List[LearnedConstraint]

    # Ground Truth Models or Callables (trained on full data: Train + Test)
    gt_objective: Any           # Callable or trained ML model for objective
    gt_constraints: List[Any]   # Callables or trained ML models for constraints
    
    constraint_model_configs: Optional[List[dict]] = None # List of dicts specifying {"model_type": str, "model_params": dict} for each constraint model data

    domain_constraints: List[DomainConstraint] = field(default_factory=list)

    X_train: Optional[np.ndarray] = None # (n_train, d_context) - contextual features for training evaluation

    # Chemotherapy / OptiCL: convex hull of training treatment vectors (n_train, n_decision)
    trust_region_points: Optional[np.ndarray] = None

    # GT outcomes and thresholds for paper Table 6 evaluation (independent of active optimizer constraints)
    eval_outcomes: Optional[List[EvalOutcome]] = None

    # Observed trial outcomes on the test split (for diagnostics / replication checks)
    observed_test_outcomes: Optional[Dict[str, np.ndarray]] = None

    # Indices of decision variables that must be binary (e.g. drug _Ind columns)
    binary_var_indices: List[int] = field(default_factory=list)

    # Full-cohort features/outcomes for GT R² diagnostics (461 retained arms)
    gt_eval_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    # Ordered feature names for X_train / X_test (context + treatment columns)
    feature_names: Optional[List[str]] = None

    # Publication year per X_train row (gastric), for temporal CV folds. None for
    # synthetic (no time) -> the robustness-parameter CV falls back to KFold.
    train_pub_years: Optional[np.ndarray] = None

    # Deterministic identities between outcome labels (gastric: DLT). Empty
    # elsewhere. Read by ScenarioBank only when uncertainty.derive_linked_labels
    # is on; see LabelLink.
    label_links: List[LabelLink] = field(default_factory=list)


# Non-standard combined ECOG buckets (Bertsimas A.1: mark unavailable unless a
# standard breakdown is available via ECOG_0..4 or ECOG_01).
_NONSTANDARD_ECOG_COLS = (
    "ECOG_23", "ECOG_12", "ECOG_034", "ECOG_012", "ECOG_234",
)

_INDIVIDUAL_KPS_COLS = tuple(f"KPS_{k}" for k in (100, 90, 80, 70, 60, 50, 40, 30, 20, 10))

# Buccheri et al. (1996) KPS -> ECOG mapping (Bertsimas A.1).
_KPS_TO_ECOG = (
    (100, 0.0), (90, 0.0), (80, 1.0), (70, 1.0), (60, 2.0),
    (50, 2.0), (40, 3.0), (30, 3.0), (20, 4.0), (10, 4.0),
)


def _fit_ecog01_frac0_predictor(df, _float):
    """
    Bertsimas et al. (2016) Appendix A.1: estimate p0/(p0+p1) from combined ECOG_01
    using linear vs quadratic regression on arms with a full ECOG breakdown.
    """
    p01_vals, frac0_vals = [], []
    for _, row in df.iterrows():
        parts = {}
        for g in range(5):
            v = _float(row.get(f"ECOG_{g}"))
            if not np.isnan(v):
                parts[g] = v
        if 0 in parts and 1 in parts:
            p0, p1 = parts[0], parts[1]
            denom = p0 + p1
            if denom > 1e-12:
                p01_vals.append(denom)
                frac0_vals.append(p0 / denom)
    if len(p01_vals) < 5:
        return lambda p01: 0.5

    x = np.asarray(p01_vals, dtype=float)
    y = np.asarray(frac0_vals, dtype=float)
    X_lin = np.column_stack([np.ones_like(x), x])
    beta_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
    ssr_lin = float(np.sum((y - X_lin @ beta_lin) ** 2))

    X_quad = np.column_stack([np.ones_like(x), x, x ** 2])
    beta_quad, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
    ssr_quad = float(np.sum((y - X_quad @ beta_quad) ** 2))

    if ssr_quad < ssr_lin:
        return lambda p01, b=beta_quad: float(np.clip(b[0] + b[1] * p01 + b[2] * p01 ** 2, 0.0, 1.0))
    return lambda p01, b=beta_lin: float(np.clip(b[0] + b[1] * p01, 0.0, 1.0))


def _has_nonstandard_ecog_grouping(row, _float) -> bool:
    return any(
        not np.isnan(_float(row.get(col)))
        for col in _NONSTANDARD_ECOG_COLS
    )


def _mean_ecog_from_kps(row, _float, min_bins: int = 2):
    """
    KPS -> mean ECOG via Buccheri mapping (Bertsimas A.1).

    Only arms reporting proportions on individual KPS bins (KPS_10 ... KPS_100)
    are converted; grouped KPS columns alone are treated as unavailable.
    """
    weighted = 0.0
    total = 0.0
    n_bins = 0
    for kps, ecog in _KPS_TO_ECOG:
        v = _float(row.get(f"KPS_{kps}"))
        if not np.isnan(v) and v > 0:
            weighted += ecog * v
            total += v
            n_bins += 1
    if n_bins >= min_bins and total > 0:
        return weighted / total
    return np.nan


def _mean_ecog_row(row, _float, ecog01_frac0):
    """
    Mean ECOG from raw performance-status columns (Bertsimas A.1).

    Maragno Appendix D.1 then imputes any remaining missing context (including
    mean ECOG) via multiple imputation from the other cohort characteristics.

    The shortcut ``0.5*ECOG_01 + 2*ECOG_2 + 3*ECOG_3`` is incorrect because it
    (i) fixes the ECOG 0/1 split at 50/50 instead of estimating p0/(p0+p1)
    from other trials, (ii) omits the denominator (total patient fraction),
    and (iii) ignores ECOG_4.
    """
    parts = {}
    for g in range(5):
        v = _float(row.get(f"ECOG_{g}"))
        if not np.isnan(v):
            parts[g] = v

    if len(parts) >= 2:
        total = sum(parts.values())
        if total > 0:
            return sum(g * p for g, p in parts.items()) / total

    e01 = _float(row.get("ECOG_01"))
    if not np.isnan(e01) and e01 > 0:
        frac0 = ecog01_frac0(e01)
        p0 = frac0 * e01
        p1 = (1.0 - frac0) * e01
        e2 = _float(row.get("ECOG_2")); e2 = 0.0 if np.isnan(e2) else e2
        e3 = _float(row.get("ECOG_3")); e3 = 0.0 if np.isnan(e3) else e3
        e4 = _float(row.get("ECOG_4")); e4 = 0.0 if np.isnan(e4) else e4
        total = p0 + p1 + e2 + e3 + e4
        if total > 0:
            return (p1 + 2.0 * e2 + 3.0 * e3 + 4.0 * e4) / total

    kps_mean = _mean_ecog_from_kps(row, _float)
    if not np.isnan(kps_mean):
        return kps_mean

    if _has_nonstandard_ecog_grouping(row, _float):
        return np.nan

    return np.nan


def compute_gt_r2_table(instance: ProblemInstance):
    """
    In-sample R² for GT ensemble members (trained on full 461 cohort) vs Table EC.11.
    GT models are separate from the embedded optimizer models (Table EC.10).
    """
    import pandas as pd
    from sklearn.metrics import r2_score

    try:
        from ..models.train import train_model
    except ImportError:
        from src.models.train import train_model

    try:
        from .gastric_model_specs import GT_ENSEMBLE_SPECS, GT_ENSEMBLE_PAPER_R2
    except ImportError:
        from src.data.gastric_model_specs import GT_ENSEMBLE_SPECS, GT_ENSEMBLE_PAPER_R2

    paper_r2 = GT_ENSEMBLE_PAPER_R2
    if instance.gt_eval_data is None:
        raise ValueError("instance.gt_eval_data is required for GT R² diagnostics")

    rows = []
    for outcome_name, data in instance.gt_eval_data.items():
        X = data["X"]
        y = data["y_pct"] if outcome_name != "os" else data["y"]
        for spec in GT_ENSEMBLE_SPECS[outcome_name]:
            mtype = spec["model_type"]
            model = train_model(X, y, mtype, spec.get("params", {}))
            pred = model.predict(X)
            r2 = float(r2_score(y, pred))
            rows.append({
                "outcome": outcome_name,
                "model_type": mtype,
                "r2": r2,
                "paper_r2": paper_r2[outcome_name][mtype],
                "delta": r2 - paper_r2[outcome_name][mtype],
            })
    return pd.DataFrame(rows)


def filter_constraints(instance: ProblemInstance, names: List[str]) -> ProblemInstance:
    """Return a copy of instance keeping only constraints whose names are in names.

    Every field not being filtered must be forwarded. ``train_pub_years`` was the
    field this hand-written constructor missed: ``cv_score_knob`` calls this right
    after ``_fold_instance``, so on contextual gastric the fold instance reached the
    solvers with ``train_pub_years=None``. ``uncertainty.instance_folds`` then read
    None and fell back to random KFold -- silently, and despite the fold's rows
    carrying their years. D's radius was still estimated from the fold's own rows
    (so nothing held out leaked in), but the temporal forward-chaining this repo
    claims on gastric was not in force inside any CV or rho-sweep fold. Measured:
    2-5% difference in ``scale(y_c)`` per fold between the two schemes.
    """
    name_set = set(names)
    indices = [i for i, c in enumerate(instance.constraints) if c.name in name_set]
    configs = instance.constraint_model_configs
    if configs is not None:
        configs = [configs[i] for i in indices]
    # A label link needs its target AND every source still modeled: under
    # `dlt_only` the four components are gone, so the identity has nothing to
    # derive from and DLT falls back to a free draw (ScenarioBank says so).
    kept = {instance.constraints[i].name for i in indices}
    links = [ln for ln in (instance.label_links or [])
             if ln.target in kept and kept.issuperset(ln.sources)]
    return ProblemInstance(
        X_test=instance.X_test,
        cost_vector=instance.cost_vector,
        variable_lb=instance.variable_lb,
        variable_ub=instance.variable_ub,
        n_features=instance.n_features,
        decision_var_indices=instance.decision_var_indices,
        context_var_indices=instance.context_var_indices,
        constraints=[instance.constraints[i] for i in indices],
        gt_objective=instance.gt_objective,
        gt_constraints=[instance.gt_constraints[i] for i in indices],
        constraint_model_configs=configs,
        domain_constraints=instance.domain_constraints,
        X_train=instance.X_train,
        trust_region_points=instance.trust_region_points,
        eval_outcomes=instance.eval_outcomes,
        observed_test_outcomes=instance.observed_test_outcomes,
        gt_eval_data=instance.gt_eval_data,
        binary_var_indices=list(instance.binary_var_indices or []),
        feature_names=instance.feature_names,
        train_pub_years=instance.train_pub_years,
        label_links=links,
    )

def _synthetic_f_true(x):
    """x can be (d,) or (n, d)."""
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1) + 0.5 * np.prod(x, axis=1)


def synthetic_nonlinear(n_train: int = 200,
                        n_test: int = 100,
                        n_features: int = 2,
                        noise_std: float = 0.1,
                        seed: int = 42,
                        fixed_constraint_config: dict = None) -> ProblemInstance:
    """
    Synthetic problem with known ground truth.

    f_true(x) = sum_j x_j^2 + 0.5 * prod_j x_j
    Nonlinear but smooth; easy to visualize in 2D.

    Optimization problem:
        min  -sum(x)          (want x large)
        s.t. f_true(x) <= b   (constraint limits x)
             0 <= x <= 1

    ``fixed_constraint_config`` -- ``{"model_type": str, "model_params": dict}`` for
    the single learned constraint, as ``run_cv.py --problem synthetic`` writes to
    ``results/cv/synthetic_selected_configs.json``. It lands in
    ``constraint_model_configs``, which ``nominal.resolve_constraint_config`` reads,
    so ONE assignment reaches every method and the ``ScenarioBank``'s per-draw
    refits alike -- the embedded model is then a CV result rather than
    ``config.yaml``'s hard-coded ``rf`` (2026-08-19 deck, next step 2). ``None``
    leaves it unset and each caller falls back to ``config.yaml``'s ``model``
    block, reproducing pre-CV runs.
    """
    rng = np.random.RandomState(seed)

    # Ground truth function
    f_true = _synthetic_f_true

    # Generate training data spread over [0, 1]^d
    X_train = rng.uniform(0, 1, size=(n_train, n_features))
    y_true = f_true(X_train)
    y_train = y_true + rng.normal(0, noise_std, size=n_train)

    # For synthetic, we have no "patient contexts", so X_test is just dummy rows
    # to iterate over (they won't constrain anything). Let's make it shape (1, 0)
    # so we just solve the global problem once.
    X_test = np.empty((1, 0))

    # Cost vector: minimize -sum(x) (i.e., maximize sum)
    cost_vector = -np.ones(n_features)

    # Variable bounds
    variable_lb = np.zeros(n_features)
    variable_ub = np.ones(n_features)

    # Set b so that the ML constraint is binding at the optimum,
    # not the box constraints. f_true at x=(1,...,1) equals
    # d + 0.5, so choosing b = 0.5*d < d + 0.5 ensures the
    # constraint boundary lies inside [0,1]^d.
    constraint_rhs = 0.5 * n_features

    constraint1_model_data = MLModelData(
        X_train=X_train,
        y_train=y_train,
        y_true=y_true,
        weight=1.0
    )

    constraint1 = LearnedConstraint(
        name="synthetic_constraint",
        models_data=[constraint1_model_data],
        rhs=constraint_rhs,
        f_true=f_true
    )

    def gt_objective(x):
        return np.dot(x, cost_vector)

    return ProblemInstance(
        X_test=X_test,
        cost_vector=cost_vector,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        n_features=n_features,
        decision_var_indices=list(range(n_features)),
        context_var_indices=[],
        constraints=[constraint1],
        gt_objective=gt_objective,
        gt_constraints=[f_true],
        constraint_model_configs=(
            [dict(fixed_constraint_config)] if fixed_constraint_config else None),
    )


def _reactor_dataset(n_train: int, noise_std: float, seed: int, cache_dir: str):
    """``(U, y_clean, y_noisy)`` for the reactor, integrating the ODEs once and caching.

    Each oracle call is a stiff ODE solve at ``rtol=atol=1e-10`` (~160 ms), so a
    1,000-point design is ~3 minutes. That is cheap once and intolerable per run,
    hence the cache -- keyed by ``(n_train, seed)`` because both change the rows.
    ``noise_std`` is deliberately NOT in the key: the clean flows are what cost
    minutes, and the noise is a cheap draw on top, so changing it re-uses the cache.
    """
    import os
    from src.data.dma_mr import benzene_flow_batch, sample_designs

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"reactor_dataset_n{n_train}_s{seed}.npz")
    legacy = os.path.join(cache_dir, "reactor_dataset.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        U, y_clean = z["U"], z["y"]
    elif os.path.exists(legacy) and n_train == 1000 and seed == 42:
        z = np.load(legacy)                      # the first build, before keying
        U, y_clean = z["U"], z["y"]
        np.savez(cache, U=U, y=y_clean)
    else:
        U = sample_designs(n_train, seed=seed)
        y_clean = benzene_flow_batch(U)
        np.savez(cache, U=U, y=y_clean)
    U, y_clean = U[:n_train], y_clean[:n_train]
    ok = np.isfinite(y_clean)
    if not ok.all():
        # An integration failure is a missing oracle value, not a zero. Drop the
        # row rather than train on a fabricated one.
        U, y_clean = U[ok], y_clean[ok]
    rng = np.random.RandomState(seed + 1)
    y_noisy = y_clean + rng.normal(0.0, noise_std, size=len(y_clean))
    return U, y_clean, y_noisy


def reactor_micl(n_train: int = 1000,
                 noise_std: float = 2.0,
                 seed: int = 42,
                 fixed_constraint_config: dict = None,
                 cost_vector=None,
                 cache_dir: str = "results/reactor") -> ProblemInstance:
    """The C-MICL regression case study: a DMA-MR membrane reactor design.

    Five design variables ``(v0, v_He, T, dt, L)``, minimize a linear operating
    cost, subject to a LEARNED constraint that the outlet benzene flow reach 50
    (Ovalle et al. 2025, App. D.1; reactor of Carrasco & Lima 2017).

    WHY IT IS HERE. It is the only instance whose oracle is MECHANISTIC. Gastric's
    judge is a fitted GT ensemble and synthetic's is a fitted proxy, so both carry
    an error of their own exactly where a constrained optimum lives -- on the
    constraint boundary. Measured on synthetic (2026-08-21) the proxy judge
    disagreed with the analytic truth on 5 of 5 nominal decisions. Here ground-truth
    feasibility is INTEGRATED (:func:`src.data.dma_mr.benzene_flow`), so that
    failure mode is absent, and it is published ground for the comparison: C-MICL's
    Figure 1 reports W-MICL -- our wrapper -- reaching only 0.45-0.85 ground-truth
    feasibility here, never the 0.9 target.

    SIGN CONVENTION. Every constraint in this repo is ``sum(w_i f_i(x)) <= rhs``,
    and the requirement is a LOWER bound ``F_C6H6 >= 50``. So the single model
    carries ``weight = -1`` against ``rhs = -50``, and ``gt_constraints`` returns
    ``-F_C6H6`` to match -- ``metrics.py`` computes ``max(0, gt_fn(x) - rhs)`` and
    would silently invert the feasibility column otherwise. D still perturbs the
    labels on their natural scale (benzene flow); the weight only applies at
    embedding time.

    ``cost_vector`` is FIXED, defaulting to ones. C-MICL redraws it per instance,
    but a new ``c`` is a different optimization problem, not a new sample of this
    one, so redrawing would confound the rho axis with problem-to-problem variation.
    Note the raw units are not commensurate -- ``T ~ 1e3`` and ``v0 ~ 1e3`` dominate
    ``dt ~ 1`` in ``sum(x)`` -- so with ones the objective is effectively about
    ``v0``, ``v_He`` and ``T``. That is a real property of the stated formulation,
    not a bug, but pass an explicit ``cost_vector`` to weight the variables evenly.

    ``noise_std = 2.0`` because the paper says only that "Gaussian noise was added"
    without a level; 2.0 reproduces their reported ReLU-NN performance (our 5-fold
    CV R^2 0.958 vs their 0.954, read off Table 2 as ``1 - MSE/1.2871``). Calibrated
    to their MODEL FIT, never to any feasibility result.

    THE DESIGN IS UNIFORM OVER THE BOX; THE OPTIMIZER IS NOT. C-MICL sample all five
    variables uniformly and independently, but the optimizer is confined to the box
    INTERSECTED with the D.1b-e ratio constraints. Measured on the n=1000, seed=42
    design: only **216 of 1000 rows (21.6%)** satisfy those ratios, and only **30
    (3.0%)** satisfy them AND reach ``F_C6H6 >= 50``. The binding ones are
    ``v0/L >= 20`` (43.4% of rows), ``v0/v_He >= 0.75`` (69.8%) and ``v0 <= 1.1 T``
    (81.0%); the other four are satisfied by >=97%.

    So the learned constraint is fit largely OUTSIDE the region where it is
    evaluated, and the binding boundary is supported by ~30 rows. This is kept, not
    corrected: uniform sampling is what the paper specifies, and reproducing their
    benchmark is the point. It is also why the instance is worth having -- model
    uncertainty at the decision boundary is genuine here rather than manufactured,
    which is what the wrapper's 0.45-0.85 ground-truth feasibility in their Figure 1
    reflects, and it is the concrete form of the caveat C-MICL raise in their own
    Discussion about feasible regions biased away from the data.
    """
    from src.data.dma_mr import (
        benzene_flow, DECISION_NAMES, DECISION_RANGES, PRODUCT_FLOOR,
    )

    U, y_clean, y_noisy = _reactor_dataset(n_train, noise_std, seed, cache_dir)
    d = len(DECISION_NAMES)

    variable_lb = np.array([DECISION_RANGES[k][0] for k in DECISION_NAMES])
    variable_ub = np.array([DECISION_RANGES[k][1] for k in DECISION_NAMES])
    cost_vector = (np.ones(d) if cost_vector is None
                   else np.asarray(cost_vector, dtype=float))

    # C-MICL D.1b-e, cross-multiplied into the sum(coeffs*x) <= rhs form. Ratios
    # are safe to clear because every variable's box is strictly positive.
    i_v0, i_vHe, i_T, i_dt, i_L = range(d)

    def _dc(pairs, rhs):
        c = np.zeros(d)
        for j, v in pairs:
            c[j] += v
        return DomainConstraint(coeffs=c, rhs=float(rhs))

    domain_constraints = [
        _dc([(i_dt, 10.0), (i_L, -1.0)], 0.0),      # L/dt  >= 10
        _dc([(i_L, 1.0), (i_dt, -150.0)], 0.0),     # L/dt  <= 150
        _dc([(i_vHe, 0.75), (i_v0, -1.0)], 0.0),    # v0/v_He >= 0.75
        _dc([(i_v0, 1.0), (i_vHe, -3.0)], 0.0),     # v0/v_He <= 3.0
        _dc([(i_L, 20.0), (i_v0, -1.0)], 0.0),      # v0/L  >= 20
        _dc([(i_v0, 1.0), (i_L, -120.0)], 0.0),     # v0/L  <= 120
        _dc([(i_v0, 1.0), (i_T, -1.1)], 0.0),       # v0 <= 1.1 T
    ]

    def neg_benzene(x):
        """``-F_C6H6(x)``, the ground truth in the repo's ``<= rhs`` convention."""
        return -benzene_flow(np.asarray(x, dtype=float).ravel())

    model_data = MLModelData(
        X_train=U,
        y_train=y_noisy,
        y_true=y_clean,
        weight=-1.0,
        # No label_bounds: benzene flow is bounded below by 0, but at rho=1 the
        # per-row shift is one unexplained sd (~2.4) against labels of 11-70, so a
        # bound at 0 would never bind and would only add a clip nothing uses.
        label_bounds=None,
    )
    constraint = LearnedConstraint(
        name="benzene_constraint",
        models_data=[model_data],
        rhs=-float(PRODUCT_FLOOR),
        f_true=neg_benzene,
    )

    return ProblemInstance(
        X_test=np.empty((1, 0)),          # single-decision, as synthetic
        cost_vector=cost_vector,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        n_features=d,
        decision_var_indices=list(range(d)),
        context_var_indices=[],
        constraints=[constraint],
        gt_objective=lambda x: float(np.dot(cost_vector, np.asarray(x, float).ravel())),
        gt_constraints=[neg_benzene],
        domain_constraints=domain_constraints,
        feature_names=list(DECISION_NAMES),
        constraint_model_configs=(
            [dict(fixed_constraint_config)] if fixed_constraint_config else None),
    )


def gastric_cancer(seed: int = 42,
                   cv_tune_gt: bool = False,
                   constraint_cv: bool = False,
                   fixed_constraint_configs: dict = None,
                   fixed_gt_ensemble_configs: dict = None,
                   train_subsample_frac: float = None,
                   subsample_seed: int = None,
                   tox_ub: float = None) -> ProblemInstance:
    """
    Chemotherapy regimen design for advanced gastric cancer.

    Data processing follows constraint-learning v11 (Data Processing v10.ipynb):
    context-first features, weekly average dose, GI_34 outcomes, and
    BLOOD_4 = min(BLOOD_34, max(G4)). Constraint/GT models use train-set
    percentile toxicity targets with UB 0.6 (Section 5.5).
    """
    import os
    import pandas as pd

    try:
        from .gastric_v11 import (
            GASTRIC_CTX_COLS,
            GASTRIC_TOX_UB,
            build_gastric_cohort,
            cohort_to_arrays,
            percentile_inverse,
            split_gastric_v11,
            train_percentile_scores,
        )
        from .gastric_model_specs import GASTRIC_EMBED_CONFIGS, GT_ENSEMBLE_SPECS
    except ImportError:
        from src.data.gastric_v11 import (
            GASTRIC_CTX_COLS,
            GASTRIC_TOX_UB,
            build_gastric_cohort,
            cohort_to_arrays,
            percentile_inverse,
            split_gastric_v11,
            train_percentile_scores,
        )
        from src.data.gastric_model_specs import GASTRIC_EMBED_CONFIGS, GT_ENSEMBLE_SPECS

    # ------------------------------------------------------------------
    # 1.  Load raw data
    # ------------------------------------------------------------------
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "data", "Gastric_Cancer_Spreadsheet.csv")
    df = pd.read_csv(csv_path, encoding="latin-1")
    print(f"Step 1: Loaded raw data. Observations: {len(df)}")


    df_cohort = build_gastric_cohort(df)
    print(f'Step 2: v11 cohort after inner joins. Observations: {len(df_cohort)}')

    df_train, df_test, t_cols = split_gastric_v11(df_cohort)
    # Publication year per training row (aligned to X_train row order), for temporal CV folds.
    train_years_full = df_train["Pub_Year"].to_numpy(dtype=float)
    print(
        f'Step 3: Train/Test split complete. '
        f'Train observations: {len(df_train)}, Test observations: {len(df_test)}'
    )

    X_train, X_test, arrays = cohort_to_arrays(df_train, df_test, t_cols)
    n_ctx = len(GASTRIC_CTX_COLS)
    n_feat = X_train.shape[1]
    n_drug_features = len(t_cols)

    dlt_train = arrays['train']['dlt']
    os_train = arrays['train']['os']
    blood_train = arrays['train']['blood']
    const_train = arrays['train']['constitutional']
    gi_train = arrays['train']['gi']
    inf_train = arrays['train']['infection']

    observed_test = arrays['test']
    X_valid = arrays['full_X']
    full_targets = arrays['full']

    dlt_valid = full_targets['dlt']
    blood_valid = full_targets['blood']
    const_valid = full_targets['constitutional']
    gi_valid = full_targets['gi']
    inf_valid = full_targets['infection']
    os_valid = full_targets['os']

    decision_var_indices = list(range(n_ctx, n_ctx + n_drug_features))
    context_var_indices = list(range(n_ctx))

    # ------------------------------------------------------------------
    # 9.5  Optional training subsample (robustness probe).
    # ------------------------------------------------------------------
    # Resample WITHOUT replacement only the rows used to FIT the embedded/robust
    # constraint models (and the trust region / anchors that live on the observed
    # data). The GT ensemble's percentile reference (`raw_train_targets`, used at
    # the GT block below) stays on the FULL training targets, so the oracle is
    # identical across realizations. `train_subsample_frac=None` -> full training
    # set (default path byte-for-byte unchanged).
    if train_subsample_frac is not None:
        n_tr = X_train.shape[0]
        m = max(1, int(round(train_subsample_frac * n_tr)))
        sub_rng = np.random.RandomState(subsample_seed)
        sub_idx = np.sort(sub_rng.choice(n_tr, size=m, replace=False))
        X_fit = X_train[sub_idx]
        print(
            f"Step 9.5: Training subsample {m}/{n_tr} rows "
            f"(frac={train_subsample_frac}, seed={subsample_seed})."
        )
    else:
        sub_idx = None
        X_fit = X_train

    def _sub(arr):
        """Index a full-length training array down to the fit subsample."""
        return arr if sub_idx is None else arr[sub_idx]

    # ------------------------------------------------------------------
    # 9.6  Variable bounds (built AFTER the subsample, so they see only fit rows).
    # ------------------------------------------------------------------
    # DECISION (treatment) columns: the optimizer CHOOSES these, so their box must
    # come from the fit rows alone -- X_valid includes the test arms, which let the
    # held-out cohort widen the action space. X_fit also matches the trust region
    # (built from X_fit below), which already confines decisions to the hull of the
    # same rows, so this tightens the leaf-pruning in embed._embed_leaves to agree
    # with the hull rather than changing the feasible set.
    variable_lb = np.zeros(n_feat)
    variable_ub = np.zeros(n_feat)

    for j, col in enumerate(t_cols):
        offset = n_ctx + j
        variable_lb[offset] = X_fit[:, offset].min()
        variable_ub[offset] = X_fit[:, offset].max()
        if col.endswith('_Ind') and variable_ub[offset] == 0:
            variable_ub[offset] = 1.0

    # CONTEXT columns: train+test ON PURPOSE, and not the same kind of dependence.
    # Contexts are never chosen -- evaluation overwrites them with lb=ub=the given
    # test row (chemo_metrics.solve_for_context) -- so the box only has to CONTAIN
    # the contexts we will be asked to prescribe for. Narrowing it to X_fit would
    # break every test solve: embed._embed_leaves drops leaves whose region lies
    # outside [variable_lb, variable_ub], and the split is temporal, so every test
    # row has Pub_Year strictly above the training max by construction -- the
    # surviving leaves would cover no test context and `sum(z) == 1` would be
    # infeasible.
    X_full = np.vstack([X_train, X_test]) if len(X_test) > 0 else X_train
    for j in range(n_ctx):
        variable_lb[j] = X_full[:, j].min()
        variable_ub[j] = X_full[:, j].max()

    # ------------------------------------------------------------------
    # 10.  Constraint models (percentile targets, fixed UB=0.6; v11 / Sec. 5.5)
    # ------------------------------------------------------------------
    constraints = []

    # Toxicity upper bound: default to the paper's 0.6, or override for an RHS sweep.
    # The same value is used for the embedded constraint and the GT eval threshold.
    tox_ub = GASTRIC_TOX_UB if tox_ub is None else float(tox_ub)

    raw_train_targets = {
        "dlt": dlt_train,
        "blood": blood_train,
        "constitutional": const_train,
        "infection": inf_train,
        "gi": gi_train,
    }

    for name, y_raw in raw_train_targets.items():
        y_fit = _sub(y_raw)
        # Percentile reference is the FULL clean training target (not the subsample),
        # shared with the GT ensemble's reference below and fixed across subsamples,
        # so "<= tox_ub" means the same raw toxicity for the optimizer and the judge.
        y_pct = train_percentile_scores(y_raw, y_fit)
        model_data = MLModelData(
            X_train=X_fit,
            y_train=y_pct,
            y_true=y_fit,
            weight=1.0,
            obj_weight=0.0,
            # Percentile ranks: a label perturbation may not push a rank outside
            # [0, 1] (src/methods/uncertainty.py clips against this).
            label_bounds=(0.0, 1.0),
        )
        constraints.append(LearnedConstraint(
            name=f"{name}_constraint",
            models_data=[model_data],
            rhs=tox_ub,
            f_true=None,
        ))

    # ------------------------------------------------------------------
    # 10.1  DLT is not free: DLT_PROP = 1 - prod(1 - tox) over the four modeled
    #       toxicities, exactly. Give the bank the identity so a scenario is a
    #       relabeling of a *trial* rather than five unrelated label shifts.
    # ------------------------------------------------------------------
    # The identity lives on the raw proportion scale, the labels on the percentile
    # scale, so the round trip is: perturbed percentile -> raw (percentile_inverse,
    # against the same full-training reference the forward transform used) ->
    # 1 - prod(1 - .) -> percentile again against DLT's own reference. Every map is
    # the one the nominal labels were built with, so at a zero shift this returns
    # the nominal DLT labels (up to the identity's 2e-16; `baseline` absorbs that).
    DLT_COMPONENTS = ["blood", "constitutional", "infection", "gi"]
    _pct_inv = {name: percentile_inverse(raw_train_targets[name])
                for name in DLT_COMPONENTS}
    _dlt_ref = raw_train_targets["dlt"]

    def _dlt_from_components(pct_by_constraint):
        survive = 1.0
        for name in DLT_COMPONENTS:
            tox_raw = _pct_inv[name](pct_by_constraint[f"{name}_constraint"])
            survive = survive * (1.0 - tox_raw)
        return train_percentile_scores(_dlt_ref, 1.0 - survive)

    _dlt_baseline = _dlt_from_components({
        f"{name}_constraint": train_percentile_scores(raw_train_targets[name],
                                                      _sub(raw_train_targets[name]))
        for name in DLT_COMPONENTS
    })
    label_links = [LabelLink(
        target="dlt_constraint",
        sources=[f"{name}_constraint" for name in DLT_COMPONENTS],
        derive=_dlt_from_components,
        baseline=_dlt_baseline,
        identity="DLT = 1 - prod(1 - tox) over " + ", ".join(DLT_COMPONENTS),
    )]

    # Inject OS model as an unconstrained bounding system directly applied to the objective
    os_fit = _sub(os_train)
    os_model_data = MLModelData(
        X_train=X_fit,
        y_train=os_fit,
        y_true=os_fit,
        weight=1.0,
        obj_weight=-1.0 # Maximize OS
    )
    constraints.append(LearnedConstraint(
        name="os_constraint",
        models_data=[os_model_data],
        rhs=np.max(os_fit),
        f_true=None
    ))

    # ------------------------------------------------------------------
    # 10.5 Constraint Models parameter selection
    # ------------------------------------------------------------------
    try:
        from ..models.train import train_best_model_cv, train_fixed_ensemble
    except ImportError:
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from src.models.train import train_best_model_cv, train_fixed_ensemble
    
    cv_param_grids = {
        "linear": {"alpha": [0.1, 1, 10, 100, 1000], "l1_ratio": list(np.arange(0.1, 1.0, 0.2))},
        "svm": {"C": [0.1, 1, 10, 100]},
        "cart": {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10], "min_samples_leaf": [0.02, 0.04, 0.06], "max_features": [0.4, 0.6, 0.8, 1.0]},
        "rf": {"n_estimators": [10, 25], "max_features": [1.0], "max_depth": [2, 3, 4]},
        "gbm": {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2], "max_depth": [2, 3, 4, 5], "n_estimators": [20]},
        "xgb": {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2], "max_depth": [2, 3, 4, 5], "n_estimators": [20]},
        "mlp": {"hidden_layer_sizes": [(10,), (20,), (50,), (100,)]},
    }

    constraint_model_configs = []
    
    if constraint_cv:
        print("Running CV for constraint models...")
        for c in constraints:
            for md in c.models_data:
                _, best_type, best_params = train_best_model_cv(
                    md.X_train, md.y_train, cv_param_grids,
                    random_state=seed, return_params=True, scoring="r2",
                )
                constraint_model_configs.append({"model_type": best_type, "model_params": best_params})
    elif fixed_constraint_configs:
        if isinstance(fixed_constraint_configs, dict) and any(
            isinstance(v, dict) for v in fixed_constraint_configs.values()
        ):
            # Dict keyed by constraint name -> per-constraint config
            for c in constraints:
                cfg = fixed_constraint_configs.get(c.name)
                if cfg is None:
                    raise ValueError(
                        f"fixed_constraint_configs missing key '{c.name}'. "
                        f"Available keys: {list(fixed_constraint_configs.keys())}"
                    )
                constraint_model_configs.append(cfg)
        else:
            # Legacy: single config applied to all constraints
            for c in constraints:
                for md in c.models_data:
                    constraint_model_configs.append(fixed_constraint_configs)
    else:
        constraint_model_configs = list(GASTRIC_EMBED_CONFIGS)

    # ------------------------------------------------------------------
    # 11.  Ground Truth Models (fit on all 461 retained arms; Appendix D.3).
    # ------------------------------------------------------------------
    gt_target_arrays = {
        "dlt": dlt_valid,
        "blood": blood_valid,
        "constitutional": const_valid,
        "infection": inf_valid,
        "gi": gi_valid,
        "os": os_valid,
    }
    gt_train_pct = {
        name: train_percentile_scores(raw_train_targets[name], raw_train_targets[name])
        for name in raw_train_targets
    }
    gt_fit_targets = {
        **{name: train_percentile_scores(raw_train_targets[name], gt_target_arrays[name])
           for name in raw_train_targets},
        "os": os_valid,
    }

    # Use CV-selected GT ensemble configs when provided, otherwise fall back to
    # fixed paper specs (Table EC.12 / GT_ENSEMBLE_SPECS).
    if fixed_gt_ensemble_configs:
        _gt_specs = fixed_gt_ensemble_configs
        print("Using CV-selected GT ensemble configs (fixed_gt_ensemble_configs).")
    else:
        _gt_specs = GT_ENSEMBLE_SPECS

    gt_models = {}
    print("Training Ground Truth ensemble models...")
    for t_name, y_t in gt_fit_targets.items():
        gt_models[t_name] = train_fixed_ensemble(X_valid, y_t, _gt_specs[t_name])

    # Objective is to maximize OS (so c = -1 for OS). We create a callable that returns the predicted OS.
    # We want to minimize -OS -> maximize OS
    def gt_objective(x):
        return -gt_models["os"].predict(np.atleast_2d(x))

    gt_constraints = []
    # Append constraint functions in the same order as `constraints`
    for c in constraints:
        target_name = c.name.replace("_constraint", "")
        # capture the model in a default argument to avoid late-binding loop closures
        def gt_fn(x, m=gt_models[target_name]):
            return m.predict(np.atleast_2d(x))
        gt_constraints.append(gt_fn)

    # ------------------------------------------------------------------
    # 12.  Assemble ProblemInstance
    # ------------------------------------------------------------------
    cost_vector = np.zeros(n_feat)

    max_drugs_coeffs = np.zeros(n_feat)
    for j, col in enumerate(t_cols):
        if col.endswith("_Ind"):
            max_drugs_coeffs[n_ctx + j] = 1.0
    domain_constraints = [DomainConstraint(coeffs=max_drugs_coeffs, rhs=3.0)]

    eval_outcomes = []
    for outcome_name, label in [
        ("dlt", "Any_DLT"),
        ("blood", "Blood"),
        ("constitutional", "Constitutional"),
        ("infection", "Infection"),
        ("gi", "Gastrointestinal"),
    ]:
        eval_outcomes.append(EvalOutcome(
            label=label,
            name=outcome_name,
            rhs=tox_ub,
            gt_fn=gt_models[outcome_name],
            is_survival=False,
        ))
    eval_outcomes.append(EvalOutcome(
        label="Overall_Survival",
        name="os",
        rhs=np.nan,
        gt_fn=gt_models["os"],
        is_survival=True,
    ))

    trust_region_points = X_fit[:, decision_var_indices].copy()
    binary_var_indices = [
        n_ctx + j for j, col in enumerate(t_cols) if col.endswith("_Ind")
    ]

    gt_eval_data = {
        name: {
            "X": X_valid,
            "y": gt_target_arrays[name],
            "y_pct": gt_fit_targets[name],
            "y_train_pct_ref": gt_train_pct.get(name),
        }
        for name in gt_target_arrays
    }

    feature_names = GASTRIC_CTX_COLS + t_cols

    return ProblemInstance(
        X_test=X_test,
        X_train=X_fit,
        cost_vector=cost_vector,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        n_features=n_feat,
        decision_var_indices=decision_var_indices,
        context_var_indices=context_var_indices,
        constraints=constraints,
        gt_objective=gt_objective,
        gt_constraints=gt_constraints,
        constraint_model_configs=constraint_model_configs,
        domain_constraints=domain_constraints,
        trust_region_points=trust_region_points,
        eval_outcomes=eval_outcomes,
        observed_test_outcomes=observed_test,
        gt_eval_data=gt_eval_data,
        binary_var_indices=binary_var_indices,
        feature_names=feature_names,
        train_pub_years=_sub(train_years_full),
        label_links=label_links,
    )


if __name__ == "__main__":
    import os
    import pandas as pd

    gastric_cancer_instance = gastric_cancer()

    n_train = gastric_cancer_instance.X_train.shape[0] if gastric_cancer_instance.X_train is not None else 0
    n_test = gastric_cancer_instance.X_test.shape[0]
    n_total = n_train + n_test
    n_drugs = len([
        j for j in gastric_cancer_instance.decision_var_indices
        if gastric_cancer_instance.variable_ub[j] == 1.0
        and gastric_cancer_instance.variable_lb[j] == 0.0
    ])

    os.makedirs("results/diagnostics", exist_ok=True)

    summary_rows = [
        {"field": "n_total_rows", "value": n_total},
        {"field": "n_train_rows", "value": n_train},
        {"field": "n_test_rows", "value": n_test},
        {"field": "n_drugs", "value": n_drugs},
        {"field": "n_features", "value": gastric_cancer_instance.n_features},
        {"field": "n_constraints", "value": len(gastric_cancer_instance.constraints)},
        {"field": "n_decision_vars", "value": len(gastric_cancer_instance.decision_var_indices)},
        {"field": "n_context_vars", "value": len(gastric_cancer_instance.context_var_indices)},
        {"field": "cost_vector_norm", "value": float(np.linalg.norm(gastric_cancer_instance.cost_vector))},
    ]
    summary_df = pd.DataFrame(summary_rows)

    constraint_rows = []
    for constraint in gastric_cancer_instance.constraints:
        constraint_rows.append({
            "name": constraint.name,
            "rhs": constraint.rhs,
            "n_models": len(constraint.models_data),
            "model_weights": ";".join(str(md.weight) for md in constraint.models_data),
            "obj_weights": ";".join(str(md.obj_weight) for md in constraint.models_data),
        })
    constraint_df = pd.DataFrame(constraint_rows)

    print("\n" + "=" * 60)
    print("GASTRIC CANCER DEBUG SUMMARY")
    print("=" * 60)
    print(f"  total={n_total}  train={n_train}  test={n_test}  drugs={n_drugs}")
    if gastric_cancer_instance.observed_test_outcomes and gastric_cancer_instance.eval_outcomes:
        print("\nTest-set satisfaction (GT ensemble, tox_limit=0.6):")
        for outcome in gastric_cancer_instance.eval_outcomes:
            if outcome.is_survival:
                gt_mean = float(outcome.gt_fn.predict(gastric_cancer_instance.X_test).mean())
                print(f"  {outcome.label:18s} GT mean months={gt_mean:.3f}")
                continue
            gt_pred = outcome.gt_fn.predict(gastric_cancer_instance.X_test)
            gt_sat = float(np.mean(gt_pred <= outcome.rhs))
            print(f"  {outcome.label:18s} GT sat={gt_sat:.3f}  (rhs={outcome.rhs:.1f})")

    print("\nGT ensemble R² vs Table EC.11 (trained on full 461 cohort):")
    gt_r2_df = compute_gt_r2_table(gastric_cancer_instance)
    print(gt_r2_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    gt_r2_path = "results/diagnostics/gt_r2_ec11.csv"
    gt_r2_df.to_csv(gt_r2_path, index=False)
    print(f"Saved GT R² diagnostics to {gt_r2_path}")

    from src.models.embed import verify_embedded_predictions
    from src.data.gastric_model_specs import GASTRIC_EMBED_CONFIGS as _EMBED_CONFIGS
    print("\nEmbedding exactness check (embedded Gurobi vs sklearn predict):")
    embed_report = verify_embedded_predictions(
        gastric_cancer_instance,
        configs=_EMBED_CONFIGS,
        n_points=5,
    )
    for line in embed_report:
        print(f"  {line}")
    print(summary_df.to_string(index=False))
    print("\nConstraint details:")
    print(constraint_df.to_string(index=False))

    summary_path = "results/diagnostics/gastric_cancer_debug_summary.csv"
    constraints_path = "results/diagnostics/gastric_cancer_debug_constraints.csv"
    summary_df.to_csv(summary_path, index=False)
    constraint_df.to_csv(constraints_path, index=False)

    print(f"\nSaved debug summary to {summary_path}")
    print(f"Saved constraint details to {constraints_path}")
    
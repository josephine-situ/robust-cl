"""
Data generation for constraint learning experiments.

We need:
1. Training data (X, y) where y = f_true(X) + noise
2. A known ground truth f_true for synthetic experiments
3. A downstream optimization problem: min c'x s.t. f(x) <= b
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, List, Any


@dataclass
class MLModelData:
    """Data for a single ML model used inside a constraint."""
    X_train: np.ndarray        # (n, d_features_for_this_model)
    y_train: np.ndarray        # (n,)
    y_true: Optional[np.ndarray] # (n,) - True noiseless values if available
    weight: float = 1.0        # Coefficient for this model in the constraint (w_i in sum(w_i * f_i(x)) <= b)

@dataclass
class LearnedConstraint:
    """A single constraint modeled via one or more ML models that are linearly combined: sum(w_i * f_i(x)) <= rhs."""
    name: str
    models_data: List[MLModelData] # List of datasets/weights, one for each ML model in this constraint
    rhs: float                 # b in sum_i(w_i * f_i(x)) <= b
    f_true: Optional[Callable] = None # Or list of callables, if known


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

def _synthetic_f_true(x):
    """x can be (d,) or (n, d)."""
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1) + 0.5 * np.prod(x, axis=1)


def synthetic_nonlinear(n_train: int = 200,
                        n_test: int = 100,
                        n_features: int = 2,
                        noise_std: float = 0.1,
                        seed: int = 42) -> ProblemInstance:
    """
    Synthetic problem with known ground truth.

    f_true(x) = sum_j x_j^2 + 0.5 * prod_j x_j
    Nonlinear but smooth; easy to visualize in 2D.

    Optimization problem:
        min  -sum(x)          (want x large)
        s.t. f_true(x) <= b   (constraint limits x)
             0 <= x <= 1
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
    )


def gastric_cancer(test_frac: float = 0.2, seed: int = 42) -> ProblemInstance:
    """
    Chemotherapy regimen design for advanced gastric cancer.

    Based on Bertsimas et al. (Management Science, 2016) and
    Maragno et al. (Operations Research, 2025).

    Each trial arm is encoded with three variables per drug
    (binary indicator, instantaneous dose mg/m², average weekly
    dose mg/m²/week) plus nine contextual covariates that are
    fixed at their training-set means during optimisation.

    Learned constraint : DLT proportion  ≤  0.5
    Linear objective   : proxy for maximising overall survival
                         (negative ridge-regression coefficients
                          of OS on features).
    """
    import pandas as pd
    import os
    from collections import Counter

    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1.  Load raw data
    # ------------------------------------------------------------------
    csv_path = os.path.join(os.path.dirname(__file__),
                            "Gastric_Cancer_Spreadsheet.csv")
    df = pd.read_csv(csv_path, encoding="latin-1")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _float(v):
        """Coerce to float; return NaN on failure."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    # ------------------------------------------------------------------
    # 2.  Identify the set of "common" drugs (appear in ≥ 3 arms)
    #     This mirrors Bertsimas et al. who keep drugs seen ≥ 1 time
    #     but we need a threshold for the feature matrix to be useful.
    # ------------------------------------------------------------------
    drug_records: list[dict] = []          # one entry per (arm, drug-slot)
    for row_i, (_, row) in enumerate(df.iterrows()):
        for slot in range(1, 6):
            name = row.get(f"D{slot}_Name")
            if not (pd.notna(name) and str(name).strip()):
                continue
            dose  = _float(row.get(f"D{slot}_Dose"))
            ndose = _float(row.get(f"D{slot}_NDose"))
            cycle = _float(row.get(f"D{slot}_Cycle"))
            if np.isnan(dose):
                continue
            drug_records.append(dict(
                row_i=row_i,
                drug=str(name).strip(),
                dose=dose,
                ndose=ndose if not np.isnan(ndose) else 1.0,
                cycle=cycle  if (not np.isnan(cycle) and cycle > 0) else 21.0,
            ))

    MIN_DRUG_COUNT = 3
    drug_counts = Counter(r["drug"] for r in drug_records)
    common_drugs = sorted(d for d, c in drug_counts.items()
                          if c >= MIN_DRUG_COUNT)
    n_drugs = len(common_drugs)
    drug_to_idx = {d: i for i, d in enumerate(common_drugs)}

    # ------------------------------------------------------------------
    # 3.  Build drug-feature matrix  (n_rows × 3·n_drugs)
    #     Per drug:  [binary, instantaneous_dose, avg_weekly_dose]
    #
    #     instantaneous_dose = dose per administration  (mg/m²)
    #     avg_weekly_dose    = dose × n_doses / (cycle_days / 7)
    #     following Bertsimas et al. §3.1
    # ------------------------------------------------------------------
    n_rows = len(df)
    drug_feat = np.zeros((n_rows, 3 * n_drugs))

    for rec in drug_records:
        if rec["drug"] not in drug_to_idx:
            continue
        d  = drug_to_idx[rec["drug"]]
        ri = rec["row_i"]
        drug_feat[ri, 3 * d]     = 1.0                            # binary
        drug_feat[ri, 3 * d + 1] = rec["dose"]                    # inst. dose
        cycle_wk = rec["cycle"] / 7.0
        drug_feat[ri, 3 * d + 2] = rec["dose"] * rec["ndose"] / cycle_wk  # avg wk

    n_drug_features = 3 * n_drugs

    # ------------------------------------------------------------------
    # 4.  Contextual features  (9 covariates from Bertsimas Table 3)
    # ------------------------------------------------------------------
    CTX_NAMES = [
        "frac_male", "age_med", "mean_ecog",
        "primary_stomach", "primary_gej",
        "prior_palliative_chemo", "asia",
        "n_patient", "pub_year",
    ]

    def _mean_ecog(row):
        """Weighted ECOG from the various reporting formats."""
        parts = {}
        for g in range(5):
            v = _float(row.get(f"ECOG_{g}"))
            if not np.isnan(v):
                parts[g] = v
        if len(parts) >= 2:
            total = sum(parts.values())
            if total > 0:
                return sum(g * p for g, p in parts.items()) / total

        # ECOG 0–1 combined
        e01 = _float(row.get("ECOG_01"))
        if not np.isnan(e01):
            e2 = _float(row.get("ECOG_2")); e2 = 0. if np.isnan(e2) else e2
            e3 = _float(row.get("ECOG_3")); e3 = 0. if np.isnan(e3) else e3
            return 0.5 * e01 + 2.0 * e2 + 3.0 * e3

        # KPS → ECOG rough map  (Buccheri et al. 1996, used in Bertsimas A.1)
        for hi, lo, ecog_val in [
            ("KPS_100_90", None, 0.0), ("KPS_80_70", None, 1.0),
            ("KPS_60_50", None, 2.0),
        ]:
            v = _float(row.get(hi))
            if not np.isnan(v):
                # can't fully reconstruct; return rough midpoint
                return ecog_val + 0.5
        return np.nan

    ctx_data = np.full((n_rows, 9), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        ctx_data[i, 0] = _float(row.get("FRAC_MALE"))
        ctx_data[i, 1] = _float(row.get("AGE_MED"))
        ctx_data[i, 2] = _mean_ecog(row)
        ctx_data[i, 3] = _float(row.get("Primary_Stomach"))
        ctx_data[i, 4] = _float(row.get("Primary_GEJ"))
        ctx_data[i, 5] = _float(row.get("Prior_Palliative_Chemo"))
        ctx_data[i, 6] = _float(row.get("Asia"))
        ctx_data[i, 7] = _float(row.get("N_Patient"))
        ctx_data[i, 8] = _float(row.get("Pub_Year"))

    X_all = np.hstack([drug_feat, ctx_data])        # (n_rows, n_feat)

    # ------------------------------------------------------------------
    # 5.  Outcome:  DLT proportion  (learned constraint)
    #
    #     "Grouped Independent" approach (Bertsimas §2.2 / App. A.3):
    #       • Each toxicity *group* score = max rate in that group
    #       • DLT = 1 − Π_g (1 − group_score_g)
    #
    #     Groups:
    #       – Grade 4 blood  (Neutro4, Thrombo4, Leuko4, Anemia4)
    #       – Grade 3/4 nonblood groups  (excl. alopecia, nausea, vomiting)
    #         one group per CTCAE category
    # ------------------------------------------------------------------
    NONBLOOD_GROUP_COLS = {
        "constitutional": "CONSTITUTIONAL_34",
        "epidermal":      "EPIDERMAL_34",
        "gi":             "GINONV_34",      # GI excl. nausea/vomiting
        "infection":      "INFECTION_34",
        "neurological":   "NEUROLOGICAL_34",
        "pain":           "PAIN_34",
        "pulmonary":      "PULMONARY_34",
        "renal":          "RENAL_34",
        "vascular":       "VASCULAR_34",
        "cardiac":        "CARDIO_34",
        "metabolic":      "METABOLIC_34",
        "hemorrhage":     "HEMORRHAGE_34",
        "allergy":        "ALLERGY_34",
    }
    BLOOD_G4_COLS = ["Neutro4", "Thrombo4", "Leuko4", "Anemia4"]

    def _compute_dlt(row):
        group_scores: list[float] = []

        # Grade-4 blood (single group, score = max over subtypes)
        blood_vals = [_float(row.get(c)) for c in BLOOD_G4_COLS]
        blood_vals = [v for v in blood_vals if not np.isnan(v)]
        if blood_vals:
            group_scores.append(max(blood_vals))

        # Nonblood groups
        for _, col in NONBLOOD_GROUP_COLS.items():
            v = _float(row.get(col))
            if not np.isnan(v) and v > 0:
                group_scores.append(v)

        if not group_scores:
            return np.nan

        prob_no_dlt = 1.0
        for gs in group_scores:
            prob_no_dlt *= (1.0 - gs)
        return 1.0 - prob_no_dlt

    y_dlt = np.array([_compute_dlt(row) for _, row in df.iterrows()])

    # Also extract OS (for building the cost-vector proxy)
    y_os = np.array([_float(row.get("OS")) for _, row in df.iterrows()])

    # ------------------------------------------------------------------
    # 6.  Filter to usable rows
    #     – non-NaN DLT  (learned constraint target)
    #     – non-NaN OS   (needed for cost proxy)
    #     – at least one common drug present
    # ------------------------------------------------------------------
    has_common_drug = drug_feat[:, ::3].sum(axis=1) > 0
    valid = (~np.isnan(y_dlt)
             & ~np.isnan(y_os)
             & has_common_drug)

    X_valid   = X_all[valid].copy()
    dlt_valid = y_dlt[valid].copy()
    os_valid  = y_os[valid].copy()

    # Impute remaining NaN features with column means
    col_mean = np.nanmean(X_valid, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(X_valid)
    X_valid[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    n_samples, n_feat = X_valid.shape

    # ------------------------------------------------------------------
    # 7. Train/Test Split
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    indices = np.arange(n_samples)
    idx_train, idx_test = train_test_split(indices, test_size=test_frac, random_state=seed)

    X_train = X_valid[idx_train]
    X_test  = X_valid[idx_test]
    dlt_train = dlt_valid[idx_train]
    os_train  = os_valid[idx_train]

    # ------------------------------------------------------------------
    # 8.  Cost vector  (linear proxy for maximising OS on train set)
    #
    #     Fit a weighted ridge regression  OS ~ X  on training data
    #     (weights ∝ √n_patients, §3.2 of Bertsimas et al.)
    #     Then cost = −β̂  so  min cost'x  ≈  max predicted OS.
    # ------------------------------------------------------------------
    # Standardise columns for numerically stable regression
    x_mu  = X_train.mean(axis=0)
    x_sig = X_train.std(axis=0)
    x_sig[x_sig == 0] = 1.0
    Xs = (X_train - x_mu) / x_sig

    # Sample weights proportional to sqrt(n_patients)
    n_pat_col = n_drug_features + CTX_NAMES.index("n_patient")
    weights = np.sqrt(np.maximum(X_train[:, n_pat_col], 1.0))
    W = np.diag(weights / weights.mean())

    lam = 1.0
    XtWX = Xs.T @ W @ Xs + lam * np.eye(n_feat)
    XtWy = Xs.T @ W @ os_train
    beta_std = np.linalg.solve(XtWX, XtWy)

    # Transform back to original scale; negate to get a cost to minimise
    cost_vector = -(beta_std / x_sig)

    # ------------------------------------------------------------------
    # 9.  Variable bounds & variable indices
    #     Drug features : [observed min, observed max] (decision)
    #     Contextual features : unbounded as they'll be fixed per test row (context)
    # ------------------------------------------------------------------
    variable_lb = np.zeros(n_feat)
    variable_ub = np.zeros(n_feat)

    # Drug features (Decision variables): use overall observed ranges
    for j in range(n_drug_features):
        variable_lb[j] = X_valid[:, j].min()
        variable_ub[j] = X_valid[:, j].max()
        # Ensure non-degenerate bounds for binary indicators
        if variable_ub[j] == 0:
            variable_ub[j] = 1.0

    # Context variables limits initially just large bounds, will be fixed during prescriptive eval
    variable_lb[n_drug_features:] = -np.inf
    variable_ub[n_drug_features:] = np.inf

    decision_var_indices = list(range(n_drug_features))
    context_var_indices = list(range(n_drug_features, n_feat))

    # ------------------------------------------------------------------
    # 10.  Constraint RHS
    #      DLT proportion ≤ 0.5  (median phase-I threshold,
    #      Bertsimas et al. §2.2)
    # ------------------------------------------------------------------
    constraint_rhs = 0.5

    constraint1_model_data = MLModelData(
        X_train=X_train,
        y_train=dlt_train,
        y_true=dlt_train,
        weight=1.0
    )

    constraint1 = LearnedConstraint(
        name="dlt_constraint",
        models_data=[constraint1_model_data],
        rhs=constraint_rhs,
        f_true=None
    )

    # ------------------------------------------------------------------
    # 11.  Ground Truth Models (Fit on all data: train + test)
    # ------------------------------------------------------------------
    # Use random forest models as the complex "ground truth"
    gt_objective_model = RandomForestRegressor(n_estimators=100, random_state=seed)
    gt_objective_model.fit(X_valid, os_valid)
    
    # Since cost vector minimizes but we want to maximize OS, our objective in the evaluation
    # should be the negated OS if we want lower to be better to match optimization, 
    # or just raw OS. Let's evaluate as negated OS to keep 'lower is better'.
    def gt_objective(x):
        # Model returns OS, we return -OS because optimization is minimization of cost_vector
        return -gt_objective_model.predict(np.atleast_2d(x))

    gt_dlt_model = RandomForestRegressor(n_estimators=100, random_state=seed)
    gt_dlt_model.fit(X_valid, dlt_valid)
    
    def gt_dlt_constraint(x):
        return gt_dlt_model.predict(np.atleast_2d(x))

    # ------------------------------------------------------------------
    # 12.  Assemble ProblemInstance
    # ------------------------------------------------------------------
    return ProblemInstance(
        X_test=X_test[:, context_var_indices], # Keep only context fields for X_test iteration
        cost_vector=cost_vector,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        n_features=n_feat,
        decision_var_indices=decision_var_indices,
        context_var_indices=context_var_indices,
        constraints=[constraint1],
        gt_objective=gt_objective,
        gt_constraints=[gt_dlt_constraint],
    )
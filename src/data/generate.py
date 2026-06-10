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
    obj_weight: float = 0.0    # Coefficient for this model in the objective

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
    
    constraint_model_configs: Optional[List[dict]] = None # List of dicts specifying {"model_type": str, "model_params": dict} for each constraint model data

    X_train: Optional[np.ndarray] = None # (n_train, d_context) - contextual features for training evaluation

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


def gastric_cancer(seed: int = 42, 
                   cv_tune_gt: bool = False,
                   constraint_cv: bool = False,
                   fixed_constraint_configs: dict = None) -> ProblemInstance:
    """
    Chemotherapy regimen design for advanced gastric cancer.

    Based on Bertsimas et al. (Management Science, 2016) and
    Maragno et al. (Operations Research, 2025).

    Each trial arm is encoded with three variables per drug
    (binary indicator, instantaneous dose mg/m², average weekly
    dose mg/m²/week) plus nine contextual covariates that are
    fixed at their training-set means during optimization.

    Learned constraints: Overall survival
                         Grade 3/4 constitutional
                         Grade 3/4 gastrointestinal
                         Grade 3/4 infection
                         Grade 4 blood (max of neutro/thrombo/leuko/lympho/anemia)
                         Any DLT: DLT = 1 − Π_g (1 − group_score_g))
                          
    Linear objective: maximize overall survival

    """
    import pandas as pd
    import os
    from collections import Counter

    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1.  Load raw data
    # ------------------------------------------------------------------
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "data", "Gastric_Cancer_Spreadsheet.csv")
    df = pd.read_csv(csv_path, encoding="latin-1")
    print(f"Step 1: Loaded raw data. Observations: {len(df)}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _float(v):
        """Coerce to float; return NaN on failure."""
        # if isinstance(v, str):
        #     v = v.strip()
        #     if v.upper() == "NC":
        #         return 0.0  # Test if the authors assumed Not Collected = 0%
        #     # Also watch out for strings like "<0.01" or "12%" which float() will choke on
        #     v = v.replace('<', '').replace('>', '').replace('%', '')
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    # ------------------------------------------------------------------
    # 2.  Identify the set of "common" drugs (appear in ≥ 3 arms)
    #     This results in 28 drugs (same as CL paper) and 84 drug-related features.
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
    #     Per drug:  [binary, instantaneous_dose, avg_daily_dose]
    #
    #     instantaneous_dose = dose per administration  (mg/m²)
    #     avg_daily_dose    = dose × n_doses / cycle_days
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
        drug_feat[ri, 3 * d + 2] = rec["dose"] * rec["ndose"] / rec["cycle"]  # avg daily dose

    n_drug_features = 3 * n_drugs

    # ------------------------------------------------------------------
    # 4.  Contextual features  (9 covariates from CL appendix)
    # ------------------------------------------------------------------
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
    # 5. Extract Raw Outcomes & Features
    # ------------------------------------------------------------------
    # Extract OS
    y_os_raw = np.array([_float(row.get("OS")) for _, row in df.iterrows()])

    # Extract the 5 Grade 4 blood toxicities
    BLOOD_G4_COLS = ["Neutro4", "Thrombo4", "Leuko4", "Anemia4", "Lympho4"]
    blood_data = np.zeros((n_rows, len(BLOOD_G4_COLS)))
    for i, col in enumerate(BLOOD_G4_COLS):
        blood_data[:, i] = np.array([_float(row.get(col)) for _, row in df.iterrows()])

    # Extract the 4 specific Grade 3/4 non-blood toxicities used for DLTs
    NONBLOOD_DLT_COLS = {
        "constitutional": "CONSTITUTIONAL_34",
        "gi":             "GINONV_34",
        "infection":      "INFECTION_34",
        "neurological":   "NEUROLOGICAL_34"
    }
    nonblood_data = np.zeros((n_rows, len(NONBLOOD_DLT_COLS)))
    for i, col in enumerate(NONBLOOD_DLT_COLS.values()):
        nonblood_data[:, i] = np.array([_float(row.get(col)) for _, row in df.iterrows()])

    # ------------------------------------------------------------------
    # 6. Filter to usable rows (TARGET: 461)
    # ------------------------------------------------------------------

    # 1. Exclude rows with missing OS
    valid_os = ~np.isnan(y_os_raw)

    # Exclude rows with no drugs
    valid_drug = df['D1_Name'].notna()

    print(f"Step 6: Filtered has OS and drugs. Observations: {np.sum(valid_os & valid_drug)}")

    # 2. Exclude missing ALL blood toxicities (Check ALL blood columns, not just G4)
    blood_cols_all = [c for c in df.columns if any(k in c for k in ["Neutro", "Thrombo", "Leuko", "Anemia", "Lympho"])]
    valid_blood = np.array([
        any(not pd.isna(row.get(c)) and str(row.get(c)).strip() != "" for c in blood_cols_all)
        for _, row in df.iterrows()
    ])

    # 3. Exclude missing ALL Grade 3/4 toxicities (Check ALL G3/4 columns)
    g34_cols_all = [c for c in df.columns if "34" in c or "4" in c]
    valid_nonblood = np.array([
        any(not pd.isna(row.get(c)) and str(row.get(c)).strip() != "" for c in g34_cols_all)
        for _, row in df.iterrows()
    ])

    # Require all three conditions to be met
    valid_mask = valid_os & valid_drug & valid_blood & valid_nonblood

    print(f"Step 6: Filtered to usable rows. Observations: {np.sum(valid_mask)}")

    # Apply the mask to your arrays
    X_valid = X_all[valid_mask].copy()
    os_valid = y_os_raw[valid_mask].copy()
    blood_valid_raw = blood_data[valid_mask].copy()
    nonblood_valid_raw = nonblood_data[valid_mask].copy()

    # ------------------------------------------------------------------
    # 6b. Multiple Imputation for Partial Missingness
    # ------------------------------------------------------------------
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # Stack contextual features and all raw toxicities so they can borrow information from each other
    combined_for_imputation = np.hstack([X_valid, blood_valid_raw, nonblood_valid_raw])
    
    imputer = IterativeImputer(random_state=seed, max_iter=10)
    combined_imputed = imputer.fit_transform(combined_for_imputation)

    # Extract the imputed feature and toxicity arrays back out
    idx_blood_start = X_valid.shape[1]
    idx_nonblood_start = idx_blood_start + 5

    X_valid = combined_imputed[:, :idx_blood_start]
    blood_imputed = combined_imputed[:, idx_blood_start:idx_nonblood_start]
    nonblood_imputed = combined_imputed[:, idx_nonblood_start:]

    # Toxicities represent proportions/probabilities, so clip bounds to [0, 1]
    blood_imputed = np.clip(blood_imputed, 0, 1)
    nonblood_imputed = np.clip(nonblood_imputed, 0, 1)

    # ------------------------------------------------------------------
    # 6c. Compute final derived outcomes from the IMPUTED data
    # ------------------------------------------------------------------
    # Grade 4 blood toxicity is the MAX of the 5 individual blood toxicities
    blood_valid = np.max(blood_imputed, axis=1)

    # Calculate overall DLT = 1 - product(1 - t_i) for the 5 toxicity groups
    dlt_valid = np.zeros(np.sum(valid_mask))
    for i in range(np.sum(valid_mask)):
        prob_no_dlt = (1.0 - blood_valid[i])
        for j in range(4): # constitutional, gi, infection, neurological
            prob_no_dlt *= (1.0 - nonblood_imputed[i, j])
        dlt_valid[i] = 1.0 - prob_no_dlt

    # Extract the individual constraint targets needed for Step 10
    const_valid = nonblood_imputed[:, 0]
    gi_valid    = nonblood_imputed[:, 1]
    inf_valid   = nonblood_imputed[:, 2]

    print(f"Step 6b: Imputed partial missingness and computed outcomes. Observations: {X_valid.shape[0]}")

    # ------------------------------------------------------------------
    # 7. Train/Test Split.
    #    Split temporally (training through 2008, testing 2009-2012).
    #    Remove observations with drugs only seen once in training.
    #    Exclude trials from test if they have new drugs not in training.
    # ------------------------------------------------------------------
    # Define our dimensions to fix NameErrors
    n_samples, n_feat = X_valid.shape
    
    pub_year_idx = n_drug_features + 8
    pub_years = X_valid[:, pub_year_idx]

    train_mask = pub_years <= 2008
    test_mask = (pub_years >= 2009) & (pub_years <= 2012)
    
    # Extract just the binary drug indicators for all rows: shape (n_samples, n_drugs)
    drug_indicators = X_valid[:, :n_drug_features:3] > 0
    
    # Count how many times each drug appears in the training set
    train_drug_counts = drug_indicators[train_mask].sum(axis=0)
    
    # 1. Exclude test trials with new drugs not seen in training
    # (A drug is "new" if train_drug_counts == 0)
    unseen_in_train = train_drug_counts == 0
    has_unseen_drug = (drug_indicators & unseen_in_train).any(axis=1)
    test_mask = test_mask & ~has_unseen_drug
    
    # 2. Identify "sparse" treatments (drugs seen EXACTLY once in training)
    # and remove ALL observations (train or test) that use them.
    sparse_in_train = train_drug_counts == 1
    has_sparse_drug = (drug_indicators & sparse_in_train).any(axis=1)
    
    train_mask = train_mask & ~has_sparse_drug
    test_mask = test_mask & ~has_sparse_drug
    
    # 3. Ensure an arm has at least one valid common drug remaining
    has_any_drug = drug_indicators.any(axis=1)
    train_mask = train_mask & has_any_drug
    test_mask = test_mask & has_any_drug

    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    print(f"Step 7: Train/Test split complete. Train observations: {len(idx_train)}, Test observations: {len(idx_test)}")

    X_train = X_valid[idx_train]
    X_test  = X_valid[idx_test]
    dlt_train = dlt_valid[idx_train]
    os_train  = os_valid[idx_train]
    blood_train = blood_valid[idx_train]
    const_train = const_valid[idx_train]
    gi_train = gi_valid[idx_train]
    inf_train = inf_valid[idx_train]

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
    # Making these finite based on X_full correctly bounds the Big-M in Gurobi models
    X_full = np.concatenate([X_valid, X_test], axis=0) if len(X_test) > 0 else X_valid
    variable_lb[n_drug_features:] = X_full[:, n_drug_features:].min(axis=0)
    variable_ub[n_drug_features:] = X_full[:, n_drug_features:].max(axis=0)

    decision_var_indices = list(range(n_drug_features))
    context_var_indices = list(range(n_drug_features, n_feat))

    # ------------------------------------------------------------------
    # 10.  Constraint RHS
    # ------------------------------------------------------------------
    constraints = []
    
    constraint_targets = {
        "dlt": dlt_train,
        "blood": blood_train,
        "constitutional": const_train,
        "infection": inf_train,
        "gi": gi_train
    }
    
    for name, y_target in constraint_targets.items():
        rhs_val = np.quantile(y_target, 0.6)
        model_data = MLModelData(
            X_train=X_train,
            y_train=y_target,
            y_true=y_target,
            weight=1.0,
            obj_weight=0.0
        )
        constraints.append(LearnedConstraint(
            name=f"{name}_constraint",
            models_data=[model_data],
            rhs=rhs_val,
            f_true=None
        ))
        
    # Inject OS model as an unconstrained bounding system directly applied to the objective
    os_model_data = MLModelData(
        X_train=X_train,
        y_train=os_train,
        y_true=os_train,
        weight=1.0,
        obj_weight=-1.0 # Maximize OS
    )
    constraints.append(LearnedConstraint(
        name="os_constraint",
        models_data=[os_model_data],
        rhs=np.max(os_train),
        f_true=None
    ))

    # ------------------------------------------------------------------
    # 10.5 Constraint Models parameter selection
    # ------------------------------------------------------------------
    try:
        from ..models.train import train_best_model_cv, train_model, train_ensemble_model_cv
    except ImportError:
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from src.models.train import train_best_model_cv, train_model, train_ensemble_model_cv
    
    cv_param_grids = {
        "linear": {"alpha": [0.1, 1, 10, 100, 1000], "l1_ratio": np.arange(0.1, 1.0, 0.2)},
        "svm": {"C": [0.1, 1, 10, 100]},
        "cart": {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10], "min_samples_leaf": [0.02, 0.04, 0.06], "max_features": [0.4, 0.6, 0.8, 1.0]},
        "rf": {"n_estimators": [10, 25], "max_features": ["auto"], "max_depth": [2, 3, 4]},
        "gbm": {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2], "max_depth": [2, 3, 4, 5], "n_estimators": [20]},
        "mlp": {"hidden_layer_sizes": [(10,), (20,), (50,), (100,)]}
    }

    constraint_model_configs = []
    
    if constraint_cv:
        print("Running CV for constraint models...")
        for c in constraints:
            for md in c.models_data:
                _, best_type, best_params = train_best_model_cv(md.X_train, md.y_train, cv_param_grids, random_state=seed, return_params=True)
                constraint_model_configs.append({"model_type": best_type, "model_params": best_params})
    elif fixed_constraint_configs:
        # User provides exact config
        for c in constraints:
            for md in c.models_data:
                # E.g. {"model_type": "rf", "model_params": {...}}
                constraint_model_configs.append(fixed_constraint_configs)
    else:
        # Default config that nominal.py will fallback to
        for c in constraints:
            for md in c.models_data:
                constraint_model_configs.append({"model_type": "rf", "model_params": {"n_estimators": 50, "max_depth": 5, "random_state": 42}})

    # ------------------------------------------------------------------
    # 11.  Ground Truth Models (Fit on all data: train + test). 
    # ------------------------------------------------------------------

    exact_hyperparams = {
        "dlt": {"model_type": "rf", "params": {"n_estimators": 500, "max_depth": 6, "max_features": 1.0}},
        "blood": {"model_type": "rf", "params": {"n_estimators": 500, "max_depth": 8, "max_features": 1.0}},
        "constitutional": {"model_type": "rf", "params": {"n_estimators": 500, "max_depth": 6, "max_features": 1.0}},
        "infection": {"model_type": "rf", "params": {"n_estimators": 250, "max_depth": 6, "max_features": 1.0}},
        "gi": {"model_type": "rf", "params": {"n_estimators": 250, "max_depth": 6, "max_features": 1.0}},
        "os": {"model_type": "rf", "params": {"n_estimators": 250, "max_depth": 8, "max_features": 1.0}},
    }

    gt_cv_param_grids = {
        "rf": {"n_estimators": [20], "max_depth": [4]}
    }

    # gt_cv_param_grids = {
    #     "linear": {"alpha": [0.1, 1, 10, 100], "l1_ratio": np.arange(0.1, 1.0, 0.1)},
    #     "svm": {"C": [0.1, 1, 10, 100]},
    #     "cart": {"max_depth": [3, 4, 5, 6, 7], "min_samples_leaf": [0.02, 0.04, 0.06], "max_features": [0.4, 0.6, 0.8, 1.0]},
    #     "rf": {"n_estimators": [10, 25, 125, 250, 500], "max_features": ["auto", 1.0], "max_depth": [2, 4, 6, 8]},
    #     "gbm": {"learning_rate": [0.01, 0.025, 0.05], "max_depth": [2, 3, 4, 5, 6, "auto"], "n_estimators": [10, 25, 125, 250, 500]},
    #     "mlp": {"hidden_layer_sizes": [(10,), (20,), (50,), (100,)]}
    # }

    full_targets = {
        "dlt": dlt_valid,
        "blood": blood_valid,
        "constitutional": const_valid,
        "infection": inf_valid,
        "gi": gi_valid,
        "os": os_valid
    }

    gt_models = {}
    print("Training Ground Truth models...")
    for t_name, y_t in full_targets.items():
        if cv_tune_gt:
            # GT Ensemble over best of each class
            gt_models[t_name] = train_ensemble_model_cv(X_valid, y_t, gt_cv_param_grids, random_state=seed)
        else:
            conf = exact_hyperparams[t_name]
            gt_models[t_name] = train_model(X_valid, y_t, model_type=conf["model_type"], params=conf["params"])

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
    
    return ProblemInstance(
        X_test=X_test,
        cost_vector=cost_vector, # cost is just -1 * predicted OS, which is handled via obj_weight in the MLModelData for the OS constraint
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        n_features=n_feat,
        decision_var_indices=decision_var_indices,
        context_var_indices=context_var_indices,
        constraints=constraints,
        gt_objective=gt_objective,
        gt_constraints=gt_constraints,
        constraint_model_configs=constraint_model_configs,
    )


if __name__ == "__main__":
    import os
    import pandas as pd

    gastric_cancer_instance = gastric_cancer(cv_tune_gt=False)

    os.makedirs("results", exist_ok=True)

    summary_rows = [
        {"field": "n_features", "value": gastric_cancer_instance.n_features},
        {"field": "n_test_rows", "value": gastric_cancer_instance.X_test.shape[0]},
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
    print(summary_df.to_string(index=False))
    print("\nConstraint details:")
    print(constraint_df.to_string(index=False))

    summary_path = "results/gastric_cancer_debug_summary.csv"
    constraints_path = "results/gastric_cancer_debug_constraints.csv"
    summary_df.to_csv(summary_path, index=False)
    constraint_df.to_csv(constraints_path, index=False)

    print(f"\nSaved debug summary to {summary_path}")
    print(f"Saved constraint details to {constraints_path}")
    
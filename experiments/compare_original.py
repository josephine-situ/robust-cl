"""Compare robust-cl gastric pipeline vs constraint-learning v11 processed data."""
import pickle
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.generate import gastric_cancer

ORIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "constraint-learning",
    "Gastric Cancer", "processed-data",
)


def main():
    inst = gastric_cancer()
    print("=== OUR PIPELINE ===")
    print(
        f"train={inst.X_train.shape[0]}, test={inst.X_test.shape[0]}, "
        f"features={inst.n_features}, drugs={len(inst.decision_var_indices) // 3}"
    )
    for o in inst.eval_outcomes:
        if o.is_survival:
            continue
        gt = o.gt_fn.predict(inst.X_test)
        sat = float((gt <= o.rhs).mean())
        print(f"  {o.label}: rhs={o.rhs:.4f}, GT sat on full test={sat:.3f}")

    v11_train = pd.read_csv(os.path.join(ORIG, "v11_gastric_train2008.csv"))
    v11_test = pd.read_csv(os.path.join(ORIG, "v11_gastric_test2008.csv"))
    with open(os.path.join(ORIG, "v11_gastric_columns.pkl"), "rb") as f:
        cols = pickle.load(f)

    print("\n=== ORIGINAL v11 ===")
    print(f"train={len(v11_train)}, test={len(v11_test)}")
    print(f"n_X={len(cols['X'])}, n_T={len(cols['T'])}, drugs={len(cols['T']) // 3}")
    print("X:", cols["X"])
    print("outcomes:", cols["outcomes"])

    feat_cols = cols["X"] + cols["T"]
    X_ours = np.vstack([inst.X_train, inst.X_test])
    orig = pd.concat([v11_train, v11_test], ignore_index=True)

    ctx_ours = X_ours[:, inst.context_var_indices]
    ctx_orig = orig[cols["X"]].values
    drug_ours = X_ours[:, inst.decision_var_indices]
    drug_orig = orig[cols["T"]].values

    print("\n=== COLUMN ORDER CHECK ===")
    print(f"ours: ctx {inst.context_var_indices}, drugs {inst.decision_var_indices[0]}-{inst.decision_var_indices[-1]}")
    print(f"v11:  ctx 0-{len(cols['X'])-1}, drugs {len(cols['X'])}-{len(feat_cols)-1}")

    ctx_diffs = []
    for i, name in enumerate(cols["X"]):
        diff = np.abs(ctx_ours[:, i] - ctx_orig[:, i])
        ctx_diffs.append((name, float(diff.max()), float(diff.mean())))
    print("\nContext max abs diff (ours vs v11):")
    for name, mx, mn in ctx_diffs:
        print(f"  {name}: max={mx:.6f}, mean={mn:.6f}")

    drug_diff = np.abs(drug_ours - drug_orig)
    print(f"\nDrug features: max abs diff={drug_diff.max():.6f}, mean={drug_diff.mean():.6f}")
    print(f"  rows with any drug diff > 1e-4: {int((drug_diff.max(axis=1) > 1e-4).sum())}/{len(drug_diff)}")

    outcome_map = {
        "dlt": "DLT_PROP", "blood": "BLOOD_4", "constitutional": "CONSTITUTIONAL_34",
        "infection": "INFECTION_34", "gi": "GI_34", "os": "OS",
    }
    print("\n=== OUTCOME DIFFS (test set) ===")
    obs = inst.observed_test_outcomes
    for k, col in outcome_map.items():
        ours_v = obs[k]
        orig_v = v11_test[col].values
        diff = np.abs(ours_v - orig_v)
        print(f"  {col}: max={diff.max():.4f}, mean={diff.mean():.4f}, ours_mean={ours_v.mean():.4f}, v11_mean={orig_v.mean():.4f}")


if __name__ == "__main__":
    main()

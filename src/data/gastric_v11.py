"""
Gastric cancer data processing aligned with constraint-learning v11 notebook
(Data Processing v10.ipynb) and Maragno et al. Appendix D.1.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Decimals kept on the context block after imputation -- see build_gastric_cohort.
CTX_DECIMALS = 6

# utils_gastric.py (constraint-learning v11)
GASTRIC_CTX_COLS = [
    "Pub_Year", "Asia", "N_Patient", "FRAC_MALE", "AGE_MED",
    "Prior_Palliative_Chemo", "Primary_Stomach", "Primary_GEJ", "ECOG_MEAN",
]
GASTRIC_T_NAMES = [
    "9-Aminocamptothecin", "Actinomycin", "BBR 3438", "BMS-182248-01", "BOF-A2",
    "Bevacizumab", "Bortezomib", "Bryostatin-1", "Caelyx", "Capecitabine",
    "Carboplatin", "Carmustine", "Cetuximab", "Cisplatin", "Cyclophosphamide",
    "Cytarabine", "DHA-paclitaxel", "Diaziquone", "Docetaxel", "Doxifluridine",
    "Doxorubicin", "Epirubicin", "Erlotinib", "Esorubicin", "Etoposide",
    "Everolimus", "Flavopiridol", "Fluorouracil", "Fotemustine", "Gefitinib",
    "Gemcitabine", "Heptaplatin", "IFN", "Iproplatin", "Irinotecan", "Irofulven",
    "Ixabepilone", "Lapatinib", "Leucovorin", "Levoleucovorin", "Lovastatin",
    "Matuzumab", "Merbarone", "Methotrexate", "Mitomycin", "Mitoxantrone",
    "NK105", "OSI-7904L", "Oxaliplatin", "PALA", "PN401", "Paclitaxel",
    "Pegamotecan", "Pemetrexed", "Pirarubicin", "Piroxantrone", "Pravastatin",
    "Raltitrexed", "S-1", "Saracatinib", "Sorafenib", "Sunitinib", "Thioguanine",
    "Topotecan", "Trastuzumab", "Triazinate", "Trimetrexate", "UFT", "Vincristine",
    "Vindesine", "Vinorelbine", "methyl-CCNU",
]
GASTRIC_REF_T_COLS = [
    name + suffix
    for suffix, name in itertools.product(["_Ind", "_Avg", "_Inst"], GASTRIC_T_NAMES)
]
BLOOD_COLS_ALL = [
    "Neutro12", "Neutro34", "Neutro4", "Thrombo12", "Thrombo34", "Thrombo4",
    "Leuko12", "Leuko34", "Leuko4", "Anemia12", "Anemia34", "Anemia4",
    "Lympho34", "Lympho4", "BLOOD_34",
]
BLOOD_G4_COLS = ["Neutro4", "Thrombo4", "Leuko4", "Anemia4", "Lympho4"]
TOX_NB_COLS = ["GI_34", "INFECTION_34", "CONSTITUTIONAL_34"]

# Paper / constraint-learning v11: toxicity UB and Table 6 satisfaction threshold
GASTRIC_TOX_UB = 0.6


def train_percentile_scores(y_train_ref: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Percentile rank within train reference, scaled to [0, 1] (v11 train_models.py)."""
    from scipy import stats

    ref = np.asarray(y_train_ref, dtype=float)
    return np.array([stats.percentileofscore(ref, x) / 100.0 for x in y], dtype=float)


def _float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def _ecog_fill_v11(df: pd.DataFrame) -> pd.Series:
    """ECOG_MEAN per Data Processing v10.ipynb (then imputed with other context)."""
    ecog_cols = [c for c in df.columns if "ECOG" in c]
    df_ecog = df[ecog_cols].copy()

    x = df_ecog[["ECOG_0", "ECOG_1"]].dropna(how="any").mean()
    ecog_prop0_01 = x["ECOG_0"] / x.sum()
    df_ecog_fill = df_ecog.copy()
    missing0 = df_ecog["ECOG_0"].isna()
    df_ecog_fill.loc[missing0, "ECOG_0"] = (
        ecog_prop0_01 * df_ecog.loc[missing0, "ECOG_01"]
    )
    df_ecog_fill.loc[missing0, "ECOG_1"] = (
        (1 - ecog_prop0_01) * df_ecog.loc[missing0, "ECOG_01"]
    )

    x = df_ecog[["ECOG_2", "ECOG_3"]].dropna(how="any").mean()
    ecog_prop2_23 = x["ECOG_2"] / x.sum()
    missing2 = df_ecog["ECOG_2"].isna()
    df_ecog_fill.loc[missing2, "ECOG_2"] = (
        ecog_prop2_23 * df_ecog.loc[missing2, "ECOG_23"]
    )
    df_ecog_fill.loc[missing2, "ECOG_3"] = (
        (1 - ecog_prop2_23) * df_ecog.loc[missing2, "ECOG_23"]
    )

    missing01 = df_ecog["ECOG_01"].isna()
    df_ecog_fill.loc[missing01, "ECOG_01"] = df_ecog_fill.loc[
        missing01, ["ECOG_0", "ECOG_1"]
    ].sum(axis=1)
    missing23 = df_ecog["ECOG_23"].isna()
    df_ecog_fill.loc[missing23, "ECOG_23"] = df_ecog_fill.loc[
        missing23, ["ECOG_2", "ECOG_3"]
    ].sum(axis=1)

    def _find_mean(row):
        if pd.isnull(row["ECOG_0"]):
            return np.nan
        return float(
            (row[["ECOG_0", "ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]] * [0, 1, 2, 3, 4]).sum()
        )

    return df_ecog_fill.apply(_find_mean, axis=1)


def _build_tx_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Treatment indicators, Inst dose, and weekly Avg dose (v11 formula)."""
    drug_slots = ["D1_", "D2_", "D3_", "D4_", "D5_"]
    records = []
    for idx, row in df.iterrows():
        pub_year = _float(row.get("Pub_Year"))
        for slot in drug_slots:
            name = row.get(f"{slot}Name")
            if not (pd.notna(name) and str(name).strip()):
                continue
            dose = _float(row.get(f"{slot}Dose"))
            if np.isnan(dose):
                continue
            ndose = _float(row.get(f"{slot}NDose"))
            ndose = 1.0 if np.isnan(ndose) else ndose
            cycle = _float(row.get(f"{slot}Cycle"))
            cycle = 21.0 if (np.isnan(cycle) or cycle <= 0) else cycle
            drug = str(name).strip()
            dose_avg = dose * ndose / (cycle / 7.0)
            records.append(
                dict(row_i=idx, Drug=drug, Pub_Year=pub_year,
                     Dose_Inst=dose, Dose_Avg=dose_avg)
            )

    if not records:
        return pd.DataFrame(index=df.index)

    tx_long = pd.DataFrame(records)
    drug_first_year = tx_long.groupby("Drug")["Pub_Year"].min().to_dict()

    wide_ind = tx_long.pivot_table(
        index="row_i", columns="Drug", values="Dose_Inst", aggfunc="first", fill_value=0
    )
    wide_ind = (wide_ind > 0).astype(float)
    wide_ind.columns = [f"{c}_Ind" for c in wide_ind.columns]

    wide_inst = tx_long.pivot_table(
        index="row_i", columns="Drug", values="Dose_Inst", aggfunc="first", fill_value=0
    )
    wide_inst.columns = [f"{c}_Inst" for c in wide_inst.columns]

    wide_avg = tx_long.pivot_table(
        index="row_i", columns="Drug", values="Dose_Avg", aggfunc="first", fill_value=0
    )
    wide_avg.columns = [f"{c}_Avg" for c in wide_avg.columns]

    df_tx = wide_ind.join(wide_avg, how="outer").join(wide_inst, how="outer")
    # Keep only rows with at least one drug (v11 inner-merges df_tx; no zero-drug rows).

    ind_cols = [c for c in df_tx.columns if c.endswith("_Ind")]

    def _eval_year(row):
        drugs = [c.replace("_Ind", "") for c in ind_cols if row[c] > 0.5]
        if not drugs:
            return np.nan
        return max(drug_first_year.get(d, 0) for d in drugs)

    df_tx["Eval_Year"] = df_tx.apply(_eval_year, axis=1)
    return df_tx


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[col] = df[col].map(_float) if col in df.columns else np.nan
    return out


def _build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Blood, non-blood tox, DLT_PROP (v11 definitions)."""
    blood_avail = [c for c in BLOOD_COLS_ALL if c in df.columns]
    df_blood = _numeric_frame(df, blood_avail)
    df_blood = df_blood.dropna(
        how="all", subset=[c for c in BLOOD_G4_COLS if c in df_blood.columns]
    )

    imp_blood = IterativeImputer(random_state=0, max_iter=500, min_value=0)
    blood_complete = pd.DataFrame(
        imp_blood.fit_transform(df_blood),
        columns=df_blood.columns,
        index=df_blood.index,
    )
    g4 = [c for c in BLOOD_G4_COLS if c in blood_complete.columns]
    blood4 = blood_complete[g4].max(axis=1)
    if "BLOOD_34" in blood_complete.columns:
        blood4 = np.minimum(blood_complete["BLOOD_34"], blood4)
    blood4.name = "BLOOD_4"

    tox_nb = [c for c in TOX_NB_COLS if c in df.columns]
    df_tox = _numeric_frame(df, tox_nb)
    df_tox = df_tox.join(blood4)
    df_tox = df_tox.dropna(how="all", axis=1)

    imp_tox = IterativeImputer(random_state=0, max_iter=500, min_value=0)
    tox_complete = pd.DataFrame(
        imp_tox.fit_transform(df_tox),
        columns=df_tox.columns,
        index=df_tox.index,
    )
    tox_complete["DLT_PROP"] = 1.0 - (1.0 - tox_complete).prod(axis=1)
    return tox_complete


def build_gastric_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full v11 feature/outcome matrix (461 rows after inner joins).
    Columns: context + treatment + outcomes + Eval_Year.
    """
    df_x = _numeric_frame(
        df,
        ["Pub_Year", "Asia", "N_Patient", "FRAC_MALE", "AGE_MED",
         "Prior_Palliative_Chemo", "Primary_Stomach", "Primary_GEJ"],
    )
    df_x["ECOG_MEAN"] = _ecog_fill_v11(df)

    imp_x = IterativeImputer(random_state=0, max_iter=100, min_value=0)
    df_x = pd.DataFrame(
        imp_x.fit_transform(df_x),
        columns=df_x.columns,
        index=df.index,
    )
    # Collapse float32 round-trip duplicates. FRAC_MALE arrives with both 0.69
    # and float32(0.69) = 0.689999998 present as distinct float64 values, and
    # trees then split between two copies of the same number -- a split that
    # fits nothing and that no leaf-box separation can be made safe against
    # (see the SPLIT_EPS note in src/models/embed.py). These are proportions and
    # counts over at most a few hundred patients, so their real resolution is
    # ~1/N_Patient; six decimals is orders finer than the data supports.
    df_x = df_x.round(CTX_DECIMALS)

    df_tx = _build_tx_wide(df)
    outcomes = _build_outcomes(df)
    os_df = _numeric_frame(df, ["OS"]).dropna(how="any")
    df_outcomes = os_df.join(outcomes, how="inner")

    df_full = df_x.join(df_tx, how="inner").join(
        df_outcomes.drop(columns=["OS"], errors="ignore"), how="inner"
    )
    df_full["OS"] = df_outcomes["OS"]
    return df_full


def _reference_t_cols(columns: pd.Index) -> List[str]:
    return [c for c in GASTRIC_REF_T_COLS if c in columns]


def split_gastric_v11(
    df: pd.DataFrame,
    year: int = 2008,
    window: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Temporal train/test split with Eval_Year filter and sparse-treatment removal.
    Returns train, test, and treatment column order (v11 T_cols_sub).
    """
    t_ref = _reference_t_cols(df.columns)
    t_present = [c for c in t_ref if c in df.columns]

    train = df.loc[df["Pub_Year"] <= year].copy()
    test = df.loc[
        (df["Pub_Year"] > year) & (df["Pub_Year"] <= year + window)
    ].copy()
    if "Eval_Year" in test.columns:
        test = test.loc[test["Eval_Year"] <= year]

    freq = train[t_present].gt(0).sum(axis=0)
    cols_remove = list(freq[freq <= 1].index)

    train = train.loc[train[cols_remove].sum(axis=1) == 0]
    test = test.loc[test[cols_remove].sum(axis=1) == 0]
    train = train.drop(columns=cols_remove, errors="ignore")
    test = test.drop(columns=cols_remove, errors="ignore")

    t_cols_sub = [c for c in GASTRIC_REF_T_COLS if c in train.columns]
    return train, test, t_cols_sub


def cohort_to_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    t_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Feature matrix [context | treatment] and outcome dicts for train/test."""
    feature_cols = GASTRIC_CTX_COLS + t_cols
    x_train = train[feature_cols].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)

    outcome_keys = {
        "dlt": "DLT_PROP",
        "blood": "BLOOD_4",
        "constitutional": "CONSTITUTIONAL_34",
        "infection": "INFECTION_34",
        "gi": "GI_34",
        "os": "OS",
    }
    train_out = {k: train[col].to_numpy(dtype=float) for k, col in outcome_keys.items()}
    test_out = {k: test[col].to_numpy(dtype=float) for k, col in outcome_keys.items()}
    full_out = {
        k: np.concatenate([train_out[k], test_out[k]])
        for k in outcome_keys
    }
    x_full = np.vstack([x_train, x_test])
    return x_train, x_test, {
        "train": train_out,
        "test": test_out,
        "full": full_out,
        "full_X": x_full,
        "feature_cols": feature_cols,
        "t_cols": t_cols,
    }

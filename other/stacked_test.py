# stack_rsf_cox_gbsa.py
# -----------------------------------------------------------
# Reproducible RSF + Cox + GBSA stacking on myeloid-leukemia
# -----------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT")

import other.simple_utils as u   # <-- the helper file you uploaded

SEED = 0
N_FOLDS = 5
TAU = 7          # years for IPCW concordance

# %%

# ------------------------------------------------------------------
# 1. Load raw CSVs
# ------------------------------------------------------------------
status      = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\target_train.csv")
clinical    = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\X_train\\clinical_train.csv")
molecular   = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\X_train\\molecular_train.csv")

# -- fill blank outcomes with 0-year, censored
missing_mask = status["OS_YEARS"].isna() | status["OS_STATUS"].isna()
status.loc[missing_mask, "OS_YEARS"]  = 0.0   # censored at t = 0
status.loc[missing_mask, "OS_STATUS"] = 0.0

ids_master = status.ID            # 3 323 rows, master order

# %%

# ------------------------------------------------------------------
# 2. Align clinical & molecular rows to that master ID order
# ------------------------------------------------------------------
clinical   = clinical.set_index("ID").loc[ids_master].reset_index()
molecular  = molecular[molecular.ID.isin(ids_master)]

# ---------- clinical transforms ----------
clinical, _ = u.transform_columns(
    clinical, clinical,
    ["BM_BLAST","WBC","ANC","MONOCYTES","PLT"]
)
clinical, _ = u.fill_nan(
    clinical, clinical,
    ["BM_BLAST","WBC","ANC","MONOCYTES","HB","PLT"],
    method="mean"
)
cyto_flag,_ = u.cyto_patient_risk(clinical.CYTOGENETICS, clinical.CYTOGENETICS)
sex_flag,_  = u.patient_gender   (clinical.CYTOGENETICS, clinical.CYTOGENETICS)

clinical = pd.concat([clinical, cyto_flag, sex_flag], axis=1)\
           .drop(columns=["CENTER","CYTOGENETICS","ADVERSE_CYTO"])

# ---------- molecular aggregation ----------
mol_feat,_ = u.molecular_transform(
    molecular, molecular,
    list(ids_master), list(ids_master)      # keep same order
)

# ---------- merge & reduce ----------
X = pd.concat(
        [clinical.drop(columns=["ID"]),
         mol_feat.drop(columns=["ID"])],
        axis=1)

X, _ = u.reduce_df(X, X, num=100)          # ~40 columns
X = X.set_index(ids_master)

# ---------- structured outcome array ----------
y = np.array([(bool(ev), float(t)) for ev,t in
              zip(status.OS_STATUS, status.OS_YEARS)],
             dtype=[("status", bool), ("time", float)])

print(f"Final design matrix : {X.shape}")
print(f"Events              : {y['status'].sum()} / {len(y)}")

# %%

# ------------------------------------------------------------------
# 3. 5-fold stratified cross-validation
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_rsf  = np.zeros(len(y))
oof_cox  = np.zeros(len(y))
oof_gbsa = np.zeros(len(y))

for tr, val in cv.split(X, y["status"]):
    X_tr, X_val = X.iloc[tr], X.iloc[val]
    y_tr, y_val = y[tr], y[val]

    rsf = RandomSurvivalForest(
            n_estimators      = 300,
            max_depth         = 12,
            min_samples_split = 10,
            min_samples_leaf  = 8,
            max_features      = "sqrt",
            n_jobs            = -1,
            random_state      = SEED
    ).fit(X_tr, y_tr)

    cox = CoxPHSurvivalAnalysis(alpha=100).fit(X_tr, y_tr)

    gbsa = GradientBoostingSurvivalAnalysis(
            learning_rate = 0.05,
            n_estimators  = 200,
            max_depth     = 3,
            random_state  = SEED
    ).fit(X_tr, y_tr)

    oof_rsf [val] = rsf .predict(X_val)
    oof_cox [val] = cox .predict(X_val)
    oof_gbsa[val] = gbsa.predict(X_val)

# %%

# ------------------------------------------------------------------
# 4. Evaluate base learners
# ------------------------------------------------------------------
def c_ipcw(pred):
    return concordance_index_ipcw(y, y, pred, tau=TAU)[0]

print("\n=== 5-fold pooled IPCW C-indices (τ = 7 yr) ===")
print(f"RSF  : {c_ipcw(oof_rsf):.4f}")
print(f"Cox  : {c_ipcw(oof_cox):.4f}")
print(f"GBSA : {c_ipcw(oof_gbsa):.4f}")

# %%

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_gbsa = np.zeros(len(y))

for tr, val in cv.split(X, y["status"]):
    X_tr, X_val = X.iloc[tr], X.iloc[val]
    y_tr, y_val = y[tr], y[val]

    gbsa = GradientBoostingSurvivalAnalysis(
            learning_rate = 0.05,
            n_estimators  = 200,
            max_depth     = 3,
            random_state  = SEED
    ).fit(X_tr, y_tr)

    oof_gbsa[val] = gbsa.predict(X_val)
    
print(f"GBSA : {c_ipcw(oof_gbsa):.4f}")

# %%

# ------------------------------------------------------------------
# 5. Stack with a meta-Cox on the three OOF columns
# ------------------------------------------------------------------
meta_X = pd.DataFrame({
    "rsf" : oof_rsf,
    "cox" : oof_cox,
    "gbsa": oof_gbsa
})
meta_cox = CoxPHSurvivalAnalysis().fit(meta_X, y)
oof_stack = meta_cox.predict(meta_X)

print("----------------------------------------------")
print(f"STACK (Cox on 3 base preds) : {c_ipcw(oof_stack):.4f}")

# %%

# ------------------------------------------------------------------
# 6. Train full-data models ready for test prediction
# ------------------------------------------------------------------
rsf_full  = rsf .fit(X, y)       # reuse last hyper-params
cox_full  = cox .fit(X, y)
gbsa_full = gbsa.fit(X, y)

# Meta-Cox re-fit on full-data base predictions
full_meta = CoxPHSurvivalAnalysis().fit(
    pd.DataFrame({
        "rsf" : rsf_full .predict(X),
        "cox" : cox_full .predict(X),
        "gbsa": gbsa_full.predict(X)
    }),
    y
)

print("Stacked model ready.  Use `full_meta.predict(base_pred_df)` on new patients.")

# %%

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof_rsf  = np.zeros(len(y))
oof_cox  = np.zeros(len(y))
oof_gbsa = np.zeros(len(y))

for tr,val in cv.split(X, y["status"]):
    X_tr, y_tr = X.iloc[tr], y[tr]
    X_val      = X.iloc[val]

    rsf  = RandomSurvivalForest(n_estimators=300, max_depth=20,
                                min_samples_leaf=10, max_features="sqrt",
                                n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    cox  = CoxPHSurvivalAnalysis(alpha=100).fit(X_tr, y_tr)
    gbsa = GradientBoostingSurvivalAnalysis(learning_rate=0.05,
                                            n_estimators=200,
                                            max_depth=3,
                                            random_state=0).fit(X_tr, y_tr)

    oof_rsf [val] = rsf .predict(X_val)
    oof_cox [val] = cox .predict(X_val)
    oof_gbsa[val] = gbsa.predict(X_val)

# --- normalise columns ---
def mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
meta_X = pd.DataFrame({"rsf": mm(oof_rsf), "cox": mm(oof_cox), "gbsa": mm(oof_gbsa)})

meta_cox  = CoxPHSurvivalAnalysis(alpha=0.1).fit(meta_X, y)
oof_stack = meta_cox.predict(meta_X)

def c(pred): return concordance_index_ipcw(y, y, pred, tau=7)[0]
print("RSF   :", c(oof_rsf))
print("Cox   :", c(oof_cox))
print("GBSA  :", c(oof_gbsa))
print("STACK :", c(oof_stack))

# %%

import hashlib, numpy as np

# md5 of the ID column
ids_hash = hashlib.md5(",".join(X.index.astype(str)).encode()).hexdigest()
print("ID-order hash:", ids_hash)
print("first 10 IDs :", list(X.index[:10]))

# %%
























































































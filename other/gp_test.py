import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT")

import other.simple_utils as u   # the helper file you supplied
# --------------------------------------------------------------
# 1. Load and prepare data  — identical to simple_model.py
# --------------------------------------------------------------
status      = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\target_train.csv")
clinical    = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\X_train\\clinical_train.csv")
molecular   = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\X_train\\molecular_train.csv")

nan_ids = [val for val in list(status['ID']) if not val in list(status.dropna(subset=['OS_YEARS', 'OS_STATUS'])['ID'])]

status = status.fillna(0)
# --- keep only patients that have OS information
status      = status.dropna(subset=["OS_YEARS", "OS_STATUS"])
ids         = list(status.ID)

clinical    = clinical[clinical.ID.isin(ids)].reset_index(drop=True)
molecular   = molecular[molecular.ID.isin(ids)].reset_index(drop=True)

# ---- basic transforms (identical columns & functions as in your script)
clinical, _ = u.transform_columns(
    clinical, clinical,               # test df is irrelevant here
    ['BM_BLAST','WBC','ANC','MONOCYTES','PLT']
)
clinical, _ = u.fill_nan(
    clinical, clinical,
    ['BM_BLAST','WBC','ANC','MONOCYTES','HB','PLT'],
    method='mean'
)
cyto_flag,_ = u.cyto_patient_risk(clinical.CYTOGENETICS, clinical.CYTOGENETICS)
sex_flag, _ = u.patient_gender(clinical.CYTOGENETICS, clinical.CYTOGENETICS)

clinical    = pd.concat([clinical, cyto_flag, sex_flag], axis=1)\
                 .drop(columns=['CENTER','CYTOGENETICS','ADVERSE_CYTO'])

mol_features,_ = u.molecular_transform(
    molecular, molecular,             # test df is irrelevant here
    list(clinical.ID), list(clinical.ID)
)

mis_order_clinical = (clinical.ID.values != status.ID.values).sum()
mis_order_molecular = (mol_features.ID.values != status.ID.values).sum()
print("Positions where clinical.ID ≠ status.ID :", mis_order_clinical) # prints 0
print("Positions where mol_features.ID ≠ status.ID :", mis_order_molecular) # prints 0

# ---- combine and keep columns that appear ≥100× in training
X           = pd.concat([clinical.drop(columns=['ID']), mol_features.drop(columns=['ID'])], axis=1)
X, _        = u.reduce_df(X, X, num=100)   # leaves ~40 columns

nan_ids_feature = [1 if val in nan_ids else 0 for val in ids]
X.insert(40, 'MISSING_OUTCOME', nan_ids_feature)

# ---- structured outcome array
y = np.array([(bool(ev),  float(t)) for ev,t in zip(status.OS_STATUS, status.OS_YEARS)],
             dtype=[('status', bool), ('time', float)])

# %%

# --------------------------------------------------------------
# 2. 5-fold stratified CV, identical split for everybody
# --------------------------------------------------------------
SEED = 0
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(y))
for tr, val in cv.split(X, y["status"]):
    rsf = RandomSurvivalForest(
        n_estimators      = 300,
        max_depth         = 20,
        min_samples_split = 6,
        min_samples_leaf  = 3,
        max_features      = 0.15,
        n_jobs            = -1,
        random_state      = 0
    ).fit(X.iloc[tr], y[tr])
    oof_pred[val] = rsf.predict(X.iloc[val])

c_ipcw = concordance_index_ipcw(y, y, oof_pred, tau=7)[0]
print(f"Baseline RSF 5-fold IPCW C-index = {c_ipcw:.4f}")

# %%

SEED = 0
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(y))
for tr, val in cv.split(X, y["status"]):
    rsf = RandomSurvivalForest(
       n_estimators      = 200,
        max_depth         = None,
        min_samples_split = 10,
        min_samples_leaf  = 10,
        max_features      = "sqrt",
        n_jobs            = -1,
        random_state      = 0
    ).fit(X.iloc[tr], y[tr])
    oof_pred[val] = rsf.predict(X.iloc[val])

c_ipcw = concordance_index_ipcw(y, y, oof_pred, tau=7)[0]
print(f"Tuned RSF 5-fold IPCW C-index = {c_ipcw:.4f}")

# %%




















































































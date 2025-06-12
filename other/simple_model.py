import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Path to directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

# Paths to files used for the training
status_file = data_dir+'\\target_train.csv'  # Contains survival information about patients, used as training target
clinical_file = data_dir+'\\X_train\\clinical_train.csv'  # Clinical information of patients used for training
molecular_file = data_dir+'\\X_train\\molecular_train.csv'  # Molecular information of patients used for training

# Paths to the test files used for submissions
clinical_file_test = data_dir+'\\X_test\\clinical_test.csv'  # Clinical test data for model submission
molecular_file_test = data_dir+'\\X_test\\molecular_test.csv'  # Molecular test data for model submission

# Features from the clinical data to include in the model
clinical_features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']

import os
os.chdir(data_dir)

# Import utility functions from model_rsf_utils
#import other.model_rsf_utils2 as u
import other.simple_utils as u

# Set random seed for reproducibility
random_seed = 1
u.set_random_seed(random_seed)

# %%

status_df = pd.read_csv(status_file) # shape (3323, 3)
clinical_df = pd.read_csv(clinical_file) # shape (3323, 9)
molecular_df = pd.read_csv(molecular_file) # shape (10935, 11)

status_df = status_df.dropna(subset=["OS_YEARS", "OS_STATUS"])
status_df = u.remove_low_censoring(status_df, 0.0)
patient_ids = list(status_df['ID'])
clinical_df = clinical_df[[True if val in patient_ids else False for val in clinical_df['ID']]].reset_index(drop=True)
molecular_df = molecular_df[[True if val in patient_ids else False for val in molecular_df['ID']]].reset_index(drop=True)

clinical_df_test = pd.read_csv(clinical_file_test)
molecular_df_test = pd.read_csv(molecular_file_test)

all_train_ids = list(clinical_df['ID'])
all_test_ids = list(clinical_df_test['ID'])

y = np.array([(bool(val), float(bal)) for val,bal in zip(list(status_df['OS_STATUS']), list(status_df['OS_YEARS']))], dtype=[('status', bool), ('time', float)])

# %%

clinical_df, clinical_df_test = u.transform_columns(clinical_df, clinical_df_test, ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'PLT'])
clinical_df, clinical_df_test = u.fill_nan(clinical_df, clinical_df_test, ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT'], method='mean')

cyto_train, cyto_test = u.cyto_patient_risk(clinical_df['CYTOGENETICS'], clinical_df_test['CYTOGENETICS'])
clinical_df = pd.concat([clinical_df, cyto_train], axis=1)
clinical_df_test = pd.concat([clinical_df_test, cyto_test], axis=1)

gender_train, gender_test = u.patient_gender(clinical_df['CYTOGENETICS'], clinical_df_test['CYTOGENETICS'])
clinical_df = pd.concat([clinical_df, gender_train], axis=1)
clinical_df_test = pd.concat([clinical_df_test, gender_test], axis=1)

clinical_df = clinical_df.drop(columns=['CENTER', 'CYTOGENETICS'])
clinical_df_test = clinical_df_test.drop(columns=['CENTER', 'CYTOGENETICS'])

molecular_df, molecular_df_test = u.molecular_transform(molecular_df, molecular_df_test, all_train_ids, all_test_ids)

# %%

data_df = pd.concat([clinical_df.drop(columns=['ADVERSE_CYTO']), molecular_df], axis=1)
test_df = pd.concat([clinical_df_test.drop(columns=['ADVERSE_CYTO']), molecular_df_test], axis=1)

#data_df = pd.concat([clinical_df, molecular_df], axis=1)
#test_df = pd.concat([clinical_df_test, molecular_df_test], axis=1)

data_df = data_df.drop(columns=['ID'])
test_df = test_df.drop(columns=['ID'])

data_df, test_df = u.reduce_df(data_df, test_df, num=100)

# %%

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
oof_rsf = np.zeros(len(y),)
oof_cox = np.zeros(len(y),)

it = 0

verbose = 1

for train_idx, val_idx in skf.split(data_df, y["status"]):
    X_tr, y_tr = data_df.iloc[train_idx], y[train_idx]
    X_val, y_val = data_df.iloc[val_idx], y[val_idx]
    
    rsf = RandomSurvivalForest(n_estimators=300, max_depth=20, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    pred_rsf = rsf.predict(X_val)
    oof_rsf[val_idx] = pred_rsf
    
    cox = CoxPHSurvivalAnalysis(alpha=100).fit(X_tr, y_tr)
    # CoxPHSurvivalAnalysis returns risk scores where higher → higher hazard
    pred_cox = cox.predict(X_val)
    oof_cox[val_idx] = pred_cox
    
    if verbose==2:
        print(f'Iteration {it}:')
        
        ind_rsf = concordance_index_censored(y_val['status'], y_val['time'], pred_rsf)[0]
        indp_rsf = concordance_index_ipcw(y_tr, y_val, pred_rsf, tau=7)[0]
        print(f'Cox CI:   {ind_rsf:0.5f}')
        print(f'Cox IPCW: {indp_rsf:0.5f}')
        
        ind_cox = concordance_index_censored(y_val['status'], y_val['time'], pred_cox)[0]
        indp_cox = concordance_index_ipcw(y_tr, y_val, pred_cox, tau=7)[0]
        print(f'RSF CI:   {ind_cox:0.5f}')
        print(f'RSF IPCW: {indp_cox:0.5f}')
        
        print()
    
    it += 1

if verbose>=1:
    print('Whole data:')
    
    ind_rsf = concordance_index_censored(y['status'], y['time'], oof_rsf)[0]
    indp_rsf = concordance_index_ipcw(y, y, oof_rsf, tau=7)[0]
    print(f'RSF CI:   {ind_rsf:0.5f}')
    print(f'RSF IPCW: {indp_rsf:0.5f}')
    
    ind_cox = concordance_index_censored(y['status'], y['time'], oof_cox)[0]
    indp_cox = concordance_index_ipcw(y, y, oof_cox, tau=7)[0]
    print(f'Cox CI:   {ind_cox:0.5f}')
    print(f'Cox IPCW: {indp_cox:0.5f}')
    
# %%

#oof_rsf = np.array([(val-min(oof_rsf))/(max(oof_rsf)-min(oof_rsf)) for val in oof_rsf])
#oof_cox = np.array([(val-min(oof_cox))/(max(oof_cox)-min(oof_cox)) for val in oof_cox])
    
meta_X = pd.DataFrame({
    "cox_score": oof_cox,
    "rsf_score": oof_rsf
})

meta_cox = CoxPHSurvivalAnalysis().fit(meta_X, y)

if verbose>=1:
    pred_meta = meta_cox.predict(meta_X)
    
    print('Stacked prediction:')
    
    ind_stack = concordance_index_censored(y['status'], y['time'], pred_meta)[0]
    indp_stack = concordance_index_ipcw(y, y, pred_meta, tau=7)[0]
    print(f'Stacked CI:   {ind_stack:0.5f}')
    print(f'Stacked IPCW: {indp_stack:0.5f}')
    print()
    print(f'Cox coefficient: {meta_cox.coef_[0]:0.5f}')
    print(f'RSF coefficient: {meta_cox.coef_[1]:0.5f}')
    print(f'Scaled Cox coefficient: {meta_cox.coef_[0]*(max(oof_cox)-min(oof_cox)):0.3f}')
    print(f'Scaled RSF coefficient: {meta_cox.coef_[1]*(max(oof_rsf)-min(oof_rsf)):0.3f}')

# %%

rsf_full = RandomSurvivalForest(n_estimators=300, max_depth=21, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(data_df, y)
cox_full = CoxPHSurvivalAnalysis(alpha=100).fit(data_df, y)

test_rsf = rsf_full.predict(test_df)
test_cox = cox_full.predict(test_df)

meta_X_test = pd.DataFrame({
    "rsf_score": test_rsf,
    "cox_score": test_cox
})

final_risk = meta_cox.predict(meta_X_test)

submission_df = pd.DataFrame({'ID': all_test_ids, 'risk_score': final_risk})
#submission_df.to_csv(data_dir + "\\submission_files\\stack0.csv", index=False)

# %%

train_df, val_df, train_y, val_y = train_test_split(data_df, y, test_size=0.3, random_state=10)

model = CoxPHSurvivalAnalysis(alpha=10, verbose=1)
model.fit(train_df, train_y)

pred = model.predict(val_df)
pred_cox = np.array([(val-min(pred))/(max(pred)-min(pred)) for val in pred])

ind = concordance_index_censored(val_y['status'], val_y['time'], pred)[0]
indp = concordance_index_ipcw(train_y, val_y, pred, tau=7)[0]

print(f'{ind:0.5f}')
print(f'{indp:0.5f}')

# %%

#train_df, val_df, train_y, val_y = train_test_split(data_df, y, test_size=0.3, random_state=2)

model = RandomSurvivalForest(n_estimators=300, max_depth=21, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0)
model.fit(train_df, train_y)

pred = model.predict(val_df)
pred_rsf = np.array([(val-min(pred))/(max(pred)-min(pred)) for val in pred])

pred_train = model.predict(train_df)

ind = concordance_index_censored(val_y['status'], val_y['time'], pred)[0]
indp = concordance_index_ipcw(train_y, val_y, pred, tau=7)[0]

ind_train = concordance_index_censored(train_y['status'], train_y['time'], pred_train)[0]
indp_train = concordance_index_ipcw(train_y, train_y, pred_train, tau=7)[0]

print(f'{ind:0.5f}')
print(f'{indp:0.5f}')
print()
print(f'{ind_train:0.5f}')
print(f'{indp_train:0.5f}')

# %%

pred_comb = (pred_cox + pred_rsf) / 2

ind_comb = concordance_index_censored(val_y['status'], val_y['time'], pred_comb)[0]
indp_comb = concordance_index_ipcw(train_y, val_y, pred_comb, tau=7)[0]

print('Averaged cox and rsf pred:')
print(f'{ind_comb:0.5f}')
print(f'{indp_comb:0.5f}')

# %%

pred_sel = np.array([val if kal<=-1.7 else bal for val,bal,kal in zip(pred_cox, pred_rsf, list(val_df['WBC']))])

ind_sel = concordance_index_censored(val_y['status'], val_y['time'], pred_sel)[0]
indp_sel = concordance_index_ipcw(train_y, val_y, pred_sel, tau=7)[0]

print('WBC:')
print(f'{ind_sel:0.5f}')
print(f'{indp_sel:0.5f}')

pred_sel = np.array([val if kal<=-1.6 else bal for val,bal,kal in zip(pred_cox, pred_rsf, list(val_df['ANC']))])
ind_sel = concordance_index_censored(val_y['status'], val_y['time'], pred_sel)[0]
indp_sel = concordance_index_ipcw(train_y, val_y, pred_sel, tau=7)[0]

print('\nANC:')
print(f'{ind_sel:0.5f}')
print(f'{indp_sel:0.5f}')

pred_sel = np.array([val if kal<=-1.7 or nal<=-1.6 else bal for val,bal,kal,nal in zip(pred_cox, pred_rsf, list(val_df['WBC']), list(val_df['ANC']))])

ind_sel = concordance_index_censored(val_y['status'], val_y['time'], pred_sel)[0]
indp_sel = concordance_index_ipcw(train_y, val_y, pred_sel, tau=7)[0]

print('\nWBC and ANC:')
print(f'{ind_sel:0.5f}')
print(f'{indp_sel:0.5f}')

# %%

SEED = 0
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def oof_score(model_ctor, X, y, **fit_kwargs):
    """
    Fit the model on each CV fold, collect out-of-fold predictions,
    then compute the IPCW C-index at tau=7 years on those predictions.
    """
    oof_pred = np.zeros(len(y))
    for tr, val in cv.split(X, y["status"]):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y[tr], y[val]
        
        if str(model_ctor)=='<class \'sksurv.linear_model.coxph.CoxPHSurvivalAnalysis\'>':
            model = model_ctor(**fit_kwargs).fit(X_tr, y_tr)
        else:
            model = model_ctor(random_state=SEED, **fit_kwargs).fit(X_tr, y_tr)
        oof_pred[val] = model.predict(X_val)

    # Concordance of the pooled out-of-fold predictions
    c_ipcw = concordance_index_ipcw(y, y, oof_pred, tau=7)[0]
    return c_ipcw, oof_pred

# %%

# Baseline RSF
rsf_c, rsf_pred = oof_score(
    RandomSurvivalForest,
    X=data_df, y=y,
    n_estimators=300,
    max_depth=20,
    min_samples_split=6,
    min_samples_leaf=3,
    n_jobs=-1
)

cox_c, cox_pred = oof_score(CoxPHSurvivalAnalysis, X=data_df, y=y, alpha=100, ties='breslow')

# %%

print(rsf_c, cox_c)

# %%















































































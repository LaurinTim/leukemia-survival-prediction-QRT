import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

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
patient_ids = list(status_df['ID'])
clinical_df = clinical_df[[True if val in patient_ids else False for val in clinical_df['ID']]].reset_index(drop=True)
molecular_df = molecular_df[[True if val in patient_ids else False for val in molecular_df['ID']]].reset_index(drop=True)

clinical_df_test = pd.read_csv(clinical_file_test)
molecular_df_test = pd.read_csv(molecular_file_test)

all_train_ids = list(clinical_df['ID'])
all_test_ids = list(clinical_df_test['ID'])

y = np.array([(bool(val), float(bal)) for val,bal in zip(list(status_df['OS_STATUS']), list(status_df['OS_YEARS']))], dtype=[('status', bool), ('time', float)])

# %%



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

train_df, val_df, train_y, val_y = train_test_split(data_df, y, test_size=0.3, random_state=2)

model = XGBRegressor(objective='survival:cox', eval_metric='cox-nloglik', tree_method='hist')
model.fit(train_df, train_y['time'], sample_weight=train_y['status'], verbose=False)

pred = model.predict(val_df)

ind = concordance_index_censored(val_y['status'], val_y['time'], pred)[0]
indp = concordance_index_ipcw(train_y, val_y, pred, tau=7)[0]

print(f'{ind:0.5f}')
print(f'{indp:0.5f}')

# %%
train_df, val_df, train_y, val_y = train_test_split(data_df, y, test_size=0.2, random_state=6)

train_times = train_y['time']
train_status = train_y['status'].astype(int)
train_y_lower = train_times.copy()
train_y_upper = np.where(train_status==1, train_times, np.inf)
dtrain = xgb.DMatrix(train_df, label_lower_bound=train_y_lower, label_upper_bound=train_y_upper)

val_times = val_y['time']
val_status = val_y['status'].astype(int)
val_y_lower = val_times.copy()
val_y_upper = np.where(val_status==1, val_times, np.inf)
dval = xgb.DMatrix(val_df, label_lower_bound=val_y_lower, label_upper_bound=val_y_upper)

params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': "logistic",
    'aft_loss_distribution_scale': 0.5,
    'device': 'gpu',
    'learning_rate': 0.4,
    "max_depth": 5,
    "max_leaves": 3,
    "max_bin": 9,
    "gamma": 2,
}

model = xgb.train(params, dtrain, num_boost_round=10000, 
                  evals=[(dval, 'validation')], early_stopping_rounds=100, 
                  verbose_eval=0)

val_pred_log = 1/model.predict(dval)
val_pred = np.exp(val_pred_log)

train_pred_log = 1/model.predict(dtrain)
train_pred = np.exp(train_pred_log)

val_ind = concordance_index_censored(val_y['status'], val_y['time'], val_pred)[0]
val_indp = concordance_index_ipcw(train_y, val_y, val_pred, tau=7)[0]

train_ind = concordance_index_censored(train_y['status'], train_y['time'], train_pred)[0]
train_indp = concordance_index_ipcw(train_y, train_y, train_pred, tau=7)[0]

print(f'Validation C-Index:    {val_ind:0.5f}')
print(f'Validation IPCW Index: {val_indp:0.5f}')
print()
print(f'Training C-Index:      {train_ind:0.5f}')
print(f'Training IPCW Index:   {train_indp:0.5f}')

# %%

scan = [0]

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)

for value in scan:
    train_ind_vals = np.zeros(K)
    train_indp_vals = np.zeros(K)
    val_ind_vals = np.zeros(K)
    val_indp_vals = np.zeros(K)
    it = 0
    
    for train_idx, val_idx in skf.split(data_df, y['status']):
        #train_df, val_df, train_y, val_y = train_test_split(data_df, y, test_size=0.3, random_state=2)
        train_df, train_y = data_df.iloc[train_idx], y[train_idx]
        val_df, val_y = data_df.iloc[val_idx], y[val_idx]
        
        train_times = train_y['time']
        train_status = train_y['status'].astype(int)
        train_y_lower = train_times.copy()
        train_y_upper = np.where(train_status==1, train_times, np.inf)
        dtrain = xgb.DMatrix(train_df, label_lower_bound=train_y_lower, label_upper_bound=train_y_upper)
        
        val_times = val_y['time']
        val_status = val_y['status'].astype(int)
        val_y_lower = val_times.copy()
        val_y_upper = np.where(val_status==1, val_times, np.inf)
        dval = xgb.DMatrix(val_df, label_lower_bound=val_y_lower, label_upper_bound=val_y_upper)
        
        params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': "logistic",
            'aft_loss_distribution_scale': 2,
            'device': 'gpu',
            'learning_rate': 0.4,
            "max_depth": 5,
            "max_leaves": 3,
            "max_bin": 9,
            "gamma": 2,
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10000, 
                          evals=[(dval, 'validation')], early_stopping_rounds=100, 
                          verbose_eval=0)
        
        val_pred_log = 1/model.predict(dval)
        val_pred = np.exp(val_pred_log)
        
        train_pred_log = 1/model.predict(dtrain)
        train_pred = np.exp(train_pred_log)
        
        val_ind = concordance_index_censored(val_y['status'], val_y['time'], val_pred)[0]
        val_indp = concordance_index_ipcw(train_y, val_y, val_pred, tau=7)[0]
        
        train_ind = concordance_index_censored(train_y['status'], train_y['time'], train_pred)[0]
        train_indp = concordance_index_ipcw(train_y, train_y, train_pred, tau=7)[0]
        
        val_ind_vals[it] = val_ind
        val_indp_vals[it] = val_indp
        train_ind_vals[it] = train_ind
        train_indp_vals[it] = train_indp
        
        it += 1
        
    print(f'Values at {value:0.1f}:')
        
    #print(f'Validation C-Index:    {np.average(val_ind):0.5f}')
    print(f'Validation IPCW Index: {np.average(val_indp):0.4f}')
    #print()
    #print(f'Training C-Index:      {np.average(train_ind):0.5f}')
    #print(f'Training IPCW Index:   {np.average(train_indp):0.5f}')
        
    print()

# %%

from sksurv.linear_model import CoxPHSurvivalAnalysis

train_df, val_df, train_y, val_y = train_test_split(data_df, y, test_size=0.3, random_state=2)

model = CoxPHSurvivalAnalysis(alpha=10, verbose=1)
model.fit(train_df, train_y)

pred = model.predict(val_df)
pred_cox = np.array([(val-min(pred))/(max(pred)-min(pred)) for val in pred])

ind = concordance_index_censored(val_y['status'], val_y['time'], pred)[0]
indp = concordance_index_ipcw(train_y, val_y, pred, tau=7)[0]

print(f'{ind:0.5f}')
print(f'{indp:0.5f}')








































































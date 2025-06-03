# Updating model_rsf.py

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
from operator import itemgetter
from sksurv.ensemble import RandomSurvivalForest

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
from lifelines import CoxPHFitter

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
import other.model_rsf_utils2 as u
#import model_rsf_utils as u

# Set random seed for reproducibility
random_seed = 1
u.set_random_seed(random_seed)

# %%

# Load datasets
status_df_original = pd.read_csv(status_file)
clinical_df_original = pd.read_csv(clinical_file)
molecular_df_original = pd.read_csv(molecular_file)

# Map effects of mutations to survival data
effects_map = u.effect_to_survival_map()

# Prepare dataset
d = u.DatasetPrep(status_df_original, clinical_df_original, molecular_df_original, ["CHR", "GENE"], effects_map)

# Extract processed datasets
status_df = d.status_df
clinical_df = d.clinical_df
molecular_df = d.molecular_df

clinical_df_sub, molecular_df_sub = d.submission_data_prep()

# %%

clis, mols = d.submission_data_prep(clinical_df_sub=clinical_df_original, molecular_df_sub=molecular_df_original)

# %%

# Instantiate dataset class
a = u.Dataset(status_df, clinical_df, molecular_df, clinical_df_sub, molecular_df_sub, min_occurences=30)

# Convert dataset into feature matrices
X_data_df = a.X
X_data_df.columns = [val[0] for val in list(X_data_df.columns)]
y = a.y

# Convert y into structured array
y = np.array([(bool(val[0]), float(val[1])) for val in y], dtype=[('status', bool), ('time', float)])

# Add duration and event columns to X_data_df
#X_data_df.insert(0, 'event', y['status'])
#X_data_df.insert(0, 'duration', y['time'])

# Get data for the submission set
clinical_df_sub, molecular_df_sub = d.submission_data_prep()
X_sub_df, patient_ids_sub = a.X_sub, a.patient_ids_sub
#X_sub_df.columns = [val[0] for val in list(X_sub_df.columns)]
X_sub = np.array(X_sub_df)

# %%

#X_train, X_val, y_train, y_val = train_test_split(X_data_df, y, test_size=0.3, random_state=1)
Xt, idst = a.submission_data(clis, mols, test=True)
Xt.columns = [val[0] for val in list(Xt.columns)]

# %%

# Transform selected features to resemble a normal distribution
'''
for df in [X_data_df, X_sub_df]:
    df['BM_BLAST'] = np.log1p(np.array(df['BM_BLAST']))
    df['PLT'] = np.array(df['PLT'])**0.3
    df['WBC'] = np.log(np.array(df['WBC']) + 0.05)
    df['ANC'] = (np.array(df['ANC']) + 0.1)**0.1
    df['MONOCYTES'] = np.log((np.array(df['MONOCYTES']) + 0.1)**0.5)
'''

yeo = PowerTransformer(method='yeo-johnson')

# Transform BM_BLAST
X_data_df['BM_BLAST'] = yeo.fit_transform(np.array(X_data_df['BM_BLAST']).reshape(-1,1)).flatten()
X_sub_df['BM_BLAST'] = yeo.transform(np.array(X_sub_df['BM_BLAST']).reshape(-1,1)).flatten()

# Transform PLT
X_data_df['PLT'] = yeo.fit_transform(np.array(X_data_df['PLT']).reshape(-1,1)).flatten()
X_sub_df['PLT'] = yeo.transform(np.array(X_sub_df['PLT']).reshape(-1,1)).flatten()

# Transform WBC
X_data_df['WBC'] = yeo.fit_transform(np.array(X_data_df['WBC']).reshape(-1,1)).flatten()
X_sub_df['WBC'] = yeo.transform(np.array(X_sub_df['WBC']).reshape(-1,1)).flatten()

# Transform ANC
X_data_df['ANC'] = yeo.fit_transform(np.array(X_data_df['ANC']).reshape(-1,1)).flatten()
X_sub_df['ANC'] = yeo.transform(np.array(X_sub_df['ANC']).reshape(-1,1)).flatten()

# Transform MONOCYTES
X_data_df['MONOCYTES'] = yeo.fit_transform(np.array(X_data_df['MONOCYTES']).reshape(-1,1)).flatten()
X_sub_df['MONOCYTES'] = yeo.transform(np.array(X_sub_df['MONOCYTES']).reshape(-1,1)).flatten()

# %% Function to get the weights for each sample depending on its distribution in selected features in the test and submission set

def get_weights(compare_columns):
    # Scale the train and submission data with a standard scaler
    X_combined_df = pd.concat([X_data_df[compare_columns], X_sub_df[compare_columns]])
    scaler = StandardScaler().fit(X_combined_df)
    X_data_df_scaled = scaler.transform(X_data_df[compare_columns])
    X_sub_df_scaled = scaler.transform(X_sub_df[compare_columns])
    
    # Fit KDEs (Kernel Density Estimation)
    kde_data = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(X_data_df_scaled)
    kde_sub = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(X_sub_df_scaled)
    
    # Get the logarithmic probability scores for the train and submission set
    log_p_data = kde_data.score_samples(X_data_df_scaled) # shape: (3173,)
    log_p_sub = kde_sub.score_samples(X_data_df_scaled) # shape: (3173,)
    
    # Get the weights
    importance_weights = np.exp(log_p_sub - log_p_data)
    importance_weights = importance_weights/np.max(importance_weights)
    #importance_weights = np.array([np.sum(importance_weights <= val) for val in importance_weights])/len(importance_weights)
    
    return importance_weights
    
# %%

# Columns with which the weights should be calculated
#compare_columns = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']
compare_columns = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC']
importance_weights = get_weights(compare_columns)

'''
w0 = get_weights(compare_columns)
w1 = get_weights([val for val in compare_columns if not val=='BM_BLAST'])
w2 = get_weights([val for val in compare_columns if not val=='HB'])
w3 = get_weights([val for val in compare_columns if not val=='PLT'])
w4 = get_weights([val for val in compare_columns if not val=='WBC'])
w5 = get_weights([val for val in compare_columns if not val=='ANC'])

importance_weights = pd.DataFrame([w0, w1, w2, w3, w4, w5], index=['All', 'No BM_BLAST', 'No HB', 'No PLT', 'No WBC', 'No ANC']).T
'''

# Add weight column to X_data_df
X_data_df.insert(0, 'weight', importance_weights)

# %%

def sets(X, y, validation_file='Validation_IDs.csv', complete_train=False):
    val_ids = pd.read_csv(data_dir + '\\val_ids\\' + validation_file)
    
    if complete_train:
        X_train = X
        y_train = y
        X_val = X[[True if val in list(val_ids['ID']) else False for val in a.patient_ids]]
        y_val = y[[True if val in list(val_ids['ID']) else False for val in a.patient_ids]]
    
    else:
        X_train = X[[False if val in list(val_ids['ID']) else True for val in a.patient_ids]]
        y_train = y[[False if val in list(val_ids['ID']) else True for val in a.patient_ids]]
        X_val = X[[True if val in list(val_ids['ID']) else False for val in a.patient_ids]]
        y_val = y[[True if val in list(val_ids['ID']) else False for val in a.patient_ids]]
        
    return X_train, X_val, y_train, y_val

# Split dataset into training and validation sets
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

# %%

model = CoxPHFitter(penalizer=0.01)

# %%

X_train0, X_val0, y_train0, y_val0 = train_test_split(X_data_df, y, test_size=0.3, random_state=1)
scan0, features_elim0 = u.scan_features(X_train0, X_val0, y_train0, y_val0, model=model)

# %%

X_train1, X_val1, y_train1, y_val1 = sets(X_data_df, y, validation_file='Validation_IDs_90.csv', complete_train=False)
scan1, features_elim1 = u.scan_features(X_train1, X_val1, y_train1, y_val1, model=model)

# %%

use_cols0 = features_elim0[54:]
use_cols1 = features_elim1[40:]

# %%

#use_cols00 = use_cols00
#use_cols11 = use_cols11

# %%

def test_cox_split(X_train1, X_val1, y_train1, y_val1):
    # Train Cox Proportional Hazard model
    cox = CoxPHFitter(penalizer=0.01)
    X_train1 = X_train1.drop(columns=['weight'])
    cox.fit(X_train1, duration_col='duration', event_col='event')#, weights_col='weight')
    #cox.fit(X_data_df1, duration_col='duration', event_col='event', weights_col='weight')

    # Evaluate Cox model
    preds1 = cox.predict_partial_hazard(X_val1.drop(columns=['duration', 'event', 'weight']))
    ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], preds1)[0]
    indp1 = concordance_index_ipcw(y_train1, y_val1, preds1)[0]
    return ind1, indp1
    

def test_cox(use_cols, random_state=1):
    # Prepare dataset with selected features
    X_data_df1 = X_data_df[use_cols + ['duration', 'event', 'weight']]
    
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_data_df1, y, test_size=0.3, random_state=random_state)
    ind10, indp10 = test_cox_split(X_train1, X_val1, y_train1, y_val1)
    
    X_train1, X_val1, y_train1, y_val1 = sets(X_data_df1, y, validation_file='Validation_IDs_90.csv', complete_train=False)
    ind11, indp11 = test_cox_split(X_train1, X_val1, y_train1, y_val1)

    print(ind10, indp10)
    print(ind11, indp11)

# %%

X_train_df, X_val_df, y_train, y_val = train_test_split(X_data_df, y, test_size=0.3, random_state=1)

# Compute feature importance scores
scores = u.fit_and_score_features_cox(X_train_df, drop_weights=True)

# Rank features based on their importance
vals0 = pd.DataFrame(scores, index=[val for val in X_data_df.columns if not val in ['duration', 'event', 'weight']], columns=["C-Index", "IPCW C-Index"])

# %%

X_train_df, X_val_df, y_train, y_val = sets(X_data_df, y, validation_file='Validation_IDs_90.csv', complete_train=False)

# Compute feature importance scores
scores1 = u.fit_and_score_features_cox(X_train_df, drop_weights=True)

# Rank features based on their importance
vals1 = pd.DataFrame(scores1, index=[val for val in X_data_df.columns if not val in ['duration', 'event', 'weight']], columns=["C-Index", "IPCW C-Index"])

# %%

# Select features based on a threshold
threshold = 0.525
use_cols0 = [i for i in vals0.index if vals0.loc[i].iloc[1] >= threshold]

# %%

# Select features based on a threshold
#threshold = 0.515
use_cols1 = [i for i in vals1.index if vals1.loc[i].iloc[1] >= threshold]
    
# %%

test_cox(use_cols0)
print()
test_cox(use_cols1)
    
# %%

use_cols2 = [val for val in use_cols0 if val in use_cols1]

print('With common features of use_cols and use_cols1:')
test_cox(use_cols2)
print()

for i in [val for val in use_cols0 if not val in use_cols1]:
    curr_cols = use_cols2 + [i]
    print(f'Added feature: {i}')
    test_cox(curr_cols)
    print()
    
# %%

use_cols2 = [val for val in use_cols0 if val in use_cols1]

print('With common features of use_cols and use_cols1:')
test_cox(use_cols2)
print()

for i in [val for val in use_cols1 if not val in use_cols0]:
    curr_cols = use_cols2 + [i]
    print(f'Added feature: {i}')
    test_cox(curr_cols)
    print()

# %%

# Prepare dataset with selected features
#X_data_df1 = X_data_df[use_cols1 + ['duration', 'event', 'weight']]
X_data_df1 = X_data_df[use_cols1 + ['duration', 'event']]

# Train-test split with selected features
X_train1, X_val1, y_train1, y_val1 = train_test_split(X_data_df1, y, test_size=0.3, random_state=1)

# Train Cox Proportional Hazard model
cox = CoxPHFitter(penalizer=0.0)
cox.fit(X_train1, duration_col='duration', event_col='event')#, weights_col='weight')
#cox.fit(X_data_df1, duration_col='duration', event_col='event', weights_col='weight')

# Evaluate Cox model
preds1 = cox.predict_partial_hazard(X_val1.drop(columns=['duration', 'event']))#, 'weight']))
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], preds1)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, preds1)[0]
print(ind1, indp1)

# %%

# Prepare dataset with selected features
X_data_df1 = X_data_df[use_cols1 + ['duration', 'event']]#, 'weight']]

# Train-test split with selected features
X_train1, X_val1, y_train1, y_val1 = sets(X_data_df1, y, validation_file='Validation_IDs_90.csv', complete_train=False)

# Train Cox Proportional Hazard model
#cox = CoxPHFitter(penalizer=0.23)
cox = CoxPHFitter(penalizer=0.02)
cox.fit(X_train1, duration_col='duration', event_col='event')#, weights_col='weight')
#cox.fit(X_train1.drop(columns=['weight']), duration_col='duration', event_col='event')
#cox.fit(X_data_df1, duration_col='duration', event_col='event', weights_col='weight')

# Evaluate Cox model
preds1 = cox.predict_partial_hazard(X_val1.drop(columns=['duration', 'event']))#, 'weight']))
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], preds1)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, preds1)[0]
print(ind1, indp1)

# %%

# Prepare dataset with selected features
X_data_df1 = X_data_df[use_cols1 + ['duration', 'event']]#, 'weight']]

# Train-test split with selected features
X_train1, X_val1, y_train1, y_val1 = sets(X_data_df1, y, validation_file='Validation_IDs_90.csv', complete_train=True)

# Train Cox Proportional Hazard model
#cox = CoxPHFitter(penalizer=0.23)
cox = CoxPHFitter(penalizer=0.05)
#cox.fit(X_train1, duration_col='duration', event_col='event', weights_col='weight')
cox.fit(X_train1, duration_col='duration', event_col='event')

# Evaluate Cox model
preds1 = cox.predict_partial_hazard(X_val1.drop(columns=['duration', 'event']))#, 'weight']))
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], preds1)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, preds1)[0]
print(ind1, indp1)

# %%

# Generate predictions for submission
pt_sub = cox.predict_partial_hazard(X_sub_df[use_cols1])
submission_df = pd.DataFrame([patient_ids_sub, pt_sub], index=["ID", "risk_score"]).T
submission_df.to_csv(data_dir + "\\submission_files\\cox_weights_no2.csv", index=False)

# %%

# Prepare dataset with selected features
X_data_df1 = X_data_df[use_cols0]
X1 = np.array(X_data_df1)

X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.3, random_state=1)

# Train Random Survival Forest model
clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
clf.fit(X_train1, y_train1)
#clf.fit(X1, y)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], pt)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {clf.score(X_train1, y_train1):0.4f}, {concordance_index_ipcw(y_train1, y_train1, clf.predict(X_train1))[0]:0.4f}")
print(f"Validation C-Index and IPCW C-Index: {ind1:0.4f}, {indp1:0.4f}")

# %%

# Prepare dataset with selected features
X_data_df1 = X_data_df[use_cols0]
X1 = np.array(X_data_df1)

X_train1, X_val1, y_train1, y_val1 = sets(X1, y, validation_file='Validation_IDs_90.csv', complete_train=False)

# Train Random Survival Forest model
clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf.fit(X_train1, y_train1)
clf.fit(X1, y)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], pt)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {clf.score(X_train1, y_train1):0.4f}, {concordance_index_ipcw(y_train1, y_train1, clf.predict(X_train1))[0]:0.4f}")
print(f"Validation C-Index and IPCW C-Index: {ind1:0.4f}, {indp1:0.4f}")

# %%

# Prepare test submission data
clinical_df_sub, molecular_df_sub = d.submission_data_prep()
X_sub_df, patient_ids_sub = a.submission_data(clinical_df_sub, molecular_df_sub)
X_sub_df1 = X_sub_df[use_cols0]
X_sub = np.array(X_sub_df1)

# %%

# Generate predictions for submission
pt_sub = clf.predict(X_sub)
submission_df = pd.DataFrame([patient_ids_sub, pt_sub], index=["ID", "risk_score"]).T
#submission_df.to_csv(data_dir + "\\submission_files\\rsff2.csv", index=False)

















































































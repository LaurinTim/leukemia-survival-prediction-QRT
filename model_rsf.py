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
import model_rsf_utils as u

# %%

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

#get the prepared submission molecular and clical data
clinical_df_sub, molecular_df_sub = d.submission_data_prep()

# %%

# Instantiate dataset class
a = u.Dataset(status_df, clinical_df, molecular_df, clinical_df_sub, molecular_df_sub, min_occurences=30)

# Convert dataset into feature matrices
X_df = a.X
X = np.array(X_df)
y = a.y
X_sub_df = a.X_sub
X_sub = np.array(X_sub_df)

# Convert y into structured array
y = np.array([(bool(val[0]), float(val[1])) for val in y], dtype=[('status', bool), ('time', float)])

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

# %%

# Compute feature importance scores
scores, s = u.fit_and_score_features(X_df, y)

# Rank features based on their importance
vals = pd.DataFrame(scores, index=X_df.columns, columns=["C-Index", "IPCW C-Index"])

# %%

# Select features based on a threshold
threshold = 0.52
use_cols = [i for i in vals.index if vals.loc[i[0]].iloc[0,1] >= threshold]

# %%

# Prepare dataset with selected features
X_df1 = X_df[use_cols]
X1 = np.array(X_df1)

# Train-test split with selected features
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.3, random_state=1)

# Train Cox Proportional Hazard model
cox = CoxPHSurvivalAnalysis()
cox.fit(X_train1, y_train1)

# Evaluate Cox model
preds1 = cox.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], preds1)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, preds1)[0]
print(ind1, indp1)

# %%

# Train Random Survival Forest model
clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
clf.fit(X_train1, y_train1)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], pt)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {clf.score(X_train1, y_train1):0.4f}, {concordance_index_ipcw(y_train1, y_train1, clf.predict(X_train1))[0]:0.4f}")
print(f"Validation C-Index and IPCW C-Index: {ind1:0.4f}, {indp1:0.4f}")

# %%

# Train Random Survival Forest model
clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
clf.fit(X1, y)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], pt)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {clf.score(X1, y):0.4f}, {concordance_index_ipcw(y, y, clf.predict(X1))[0]:0.4f}")
print(f"Validation C-Index and IPCW C-Index: {ind1:0.4f}, {indp1:0.4f}")

# %%

# Prepare test submission data
#clinical_df_sub, molecular_df_sub = d.submission_data_prep()
#X_sub_df, patient_ids_sub = a.submission_data(clinical_df_sub, molecular_df_sub)
patient_ids_sub = a.patient_ids_sub
X_sub_df1 = X_sub_df[use_cols]
X_sub = np.array(X_sub_df1)

# %%

# Generate predictions for submission
pt_sub = clf.predict(X_sub)
submission_df1 = pd.DataFrame([patient_ids_sub, pt_sub], index=["ID", "risk_score"]).T
submission_df1.to_csv(data_dir + "\\submission_files\\rsf2.csv", index=False)

























































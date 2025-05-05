# Updating model_rsf.py

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
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
import matplotlib.pyplot as plt

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
import utils as u

# %%

# Set random seed for reproducibility
random_seed = 1
u.set_random_seed(random_seed)

# %%

# Load datasets
status_df_original = pd.read_csv(status_file) # shape (3323, 3)
clinical_df_original = pd.read_csv(clinical_file) # shape (3323, 9)
molecular_df_original = pd.read_csv(molecular_file) # shape (10935, 11)

# Map effects of mutations to survival data
effects_map = u.effect_to_survival_map()

# Prepare dataset
d = u.DatasetPrep(status_df_original, clinical_df_original, molecular_df_original, ["CHR", "GENE"], effects_map)

# Extract processed datasets
status_df = d.status_df # shape (3173, 9)
clinical_df = d.clinical_df # shape (3173, 9)
molecular_df = d.molecular_df # shape (10545, 154)

#get the prepared submission molecular and clical data
clinical_df_sub, molecular_df_sub = d.submission_data_prep() # shapes (1193, 9) for clinical_df_sub, (3089, 154) for molecular_df_sub

# %%

# Instantiate dataset class
a = u.Dataset(status_df, clinical_df, molecular_df, clinical_df_sub, molecular_df_sub, min_occurences=30)

# Convert dataset into feature matrices
X_df = a.X # shape (3173, 78)
X = np.array(X_df)
y = a.y # shape (3173, 2)
X_sub_df = a.X_sub # shape (1193, 78)
X_sub = np.array(X_sub_df)

# Convert y into structured array
y = np.array([(bool(val[0]), float(val[1])) for val in y], dtype=[('status', bool), ('time', float)])

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

# %%

#rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, min_samples_split=80, min_samples_leaf=10, n_jobs=-1, random_state=1)
#vals, fe = u.test_features(X_df, y, rsf)
rsf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-5)
vals_cox, fe_cox = u.test_features(X_df, y, rsf)

# %%

#rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, min_samples_split=80, min_samples_leaf=10, n_jobs=-1, random_state=1)
#vals2, fe2, best_score2, best_cols2 = u.test_features2(X_df, y, rsf)
rsf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-5)
vals2_cox, fe2_cox, best_score2_cox, best_cols2_cox = u.test_features2(X_df, y, rsf)

# %%

#vals_cox.to_csv(data_dir + "cox_remove_features.csv")
#vals2_cox.to_csv(data_dir + "cox_all_combinations.csv")

# %%

use_cols = [val for val in X_df.columns if val in fe_cox[26:]]

# %%

# Train-test split with selected features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

X_train = pd.DataFrame(X_train, columns=X_df.columns)
X_val = pd.DataFrame(X_val, columns=X_df.columns)

X_train = np.array(X_train[use_cols])
X_val = np.array(X_val[use_cols])

# Train Random Survival Forest model
#clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest(n_estimators=300, max_depth=10, min_samples_split=20, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest(n_estimators=300, max_depth=None, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest()
clf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-9)
clf.fit(X_train, y_train)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X_val)
ind = concordance_index_censored(y_val['status'], y_val['time'], pt)[0]
indp = concordance_index_ipcw(y_train, y_val, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {clf.score(X_train, y_train):0.4f}, {concordance_index_ipcw(y_train, y_train, clf.predict(X_train))[0]:0.4f}")
print(f"Validation C-Index and IPCW C-Index: {ind:0.4f}, {indp:0.4f}")

# %%

X1 = np.array(X_df[use_cols])

# Train Random Survival Forest model
#clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest(n_estimators=300, max_depth=None, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest()
clf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-9)
clf.fit(X1, y)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X1)
ind = concordance_index_censored(y['status'], y['time'], pt)[0]
indp = concordance_index_ipcw(y, y, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {ind:0.4f}, {indp:0.4f}")

# %%

# Prepare test submission data
patient_ids_sub = a.patient_ids_sub
X_sub_df1 = X_sub_df[use_cols]
X_sub = np.array(X_sub_df1)

# %%

# Generate predictions for the test set
pt_sub = clf.predict(X_sub)
submission_df1 = pd.DataFrame([patient_ids_sub, pt_sub], index=["ID", "risk_score"]).T
submission_df1.to_csv(data_dir + "\\submission_files\\rsf6.csv", index=False)













































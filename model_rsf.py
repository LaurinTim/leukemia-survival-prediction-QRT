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
import model_rsf_utils as u

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
rsf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-9)
vals_cox, fe_cox = u.test_features(X_df, y, rsf)

# %%

from time import time

rsf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-9)

#xx = X_df[[val for val in X_df.columns if not val in ["CHR_19", "GENE_PHF6"]]]
#yy = y

xx, yy = u.test_features(X_df, y, rsf)

# %%

rsf = CoxPHSurvivalAnalysis(n_iter=20, tol=1e-9)

xxx = xx[:, 0:]

st = time()
rsf.fit(xxx, yy)
pred = rsf.predict(xxx)
ind = concordance_index_censored(yy['status'], yy['time'], pred)[0]
indp = concordance_index_ipcw(yy, yy, pred)[0]
print(time()-st)
print(ind, indp)

# %%

#rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, min_samples_split=80, min_samples_leaf=10, n_jobs=-1, random_state=1)
#vals2, fe2, best_score2, best_cols2 = u.test_features2(X_df, y, rsf)
rsf = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-9)
vals2_cox, fe2_cox, best_score2_cox, best_cols2_cox = u.test_features2(X_df, y, rsf)

# %%

vals_cox.to_csv(data_dir + "cox_remove_features.csv")
vals2_cox.to_csv(data_dir + "cox_all_combinations.csv")

# %%

use_cols = [val for val in X_df.columns if val in fe_cox[30:]]

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
clf = RandomSurvivalForest(n_estimators=300, max_depth=None, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest()
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
clf = RandomSurvivalForest(n_estimators=300, max_depth=None, min_samples_split=6, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest()
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
submission_df1.to_csv(data_dir + "\\submission_files\\rsf5.csv", index=False)

# %%

# Compute feature importance scores
scores = u.fit_and_score_features(X_df, y)

# Rank features based on their importance
# The first two rows are the Concordance Index and IPCW Concordance Index obtained by training a model with RandomSurvivalForest and only using the
# feature in the index, the next two are the same but lifelines.CoxPHFitter is used for the model. Similarly, for the 5th and 6th column 
# sksurv.linear_model.CoxPHSurvivalAnalysis is used and for the 7th and 8th column sksurv.linear_model.CoxnetSurvivalAnalysis is used. In the 9th 
# column is the p score obtained from the summary of the lifelines.CoxPHFitter method.
vals = pd.DataFrame(scores, index=X_df.columns, columns=["C-Index", "IPCW C-Index", "Cox Reg C-Index", "Cox Reg IPCW C-Index", "Skl C-Index", "Skl IPCW C-Index", 
                                                         "Lasso C-Index", "Lasso IPCW C-Index", "p score"]) # shape (78, 9)
# %%

scores = u.fit_and_score_features2(X_df, y)
vals1 = pd.DataFrame(scores, index=X_df.columns, columns=["C-Index", "IPCW C-Index", "Cox Reg C-Index", "Cox Reg IPCW C-Index", "Skl C-Index", "Skl IPCW C-Index", 
                                                         "Lasso C-Index", "Lasso IPCW C-Index", "p score"]) # shape (78, 9)

# %%


# Select features based on a threshold
threshold = 0.57
threshold_p = 1e-0
use_cols = [i for i in vals.index if vals1.loc[i].iloc[1] >= threshold and vals1.loc[i].iloc[-1] <= threshold_p] # shape (32)

# %%

scores = u.fit_and_score_features2(X_df[use_cols], y)
vals2 = pd.DataFrame(scores[0:1], index=["use_cols features"], columns=["C-Index", "IPCW C-Index", "Cox Reg C-Index", "Cox Reg IPCW C-Index", "Skl C-Index", "Skl IPCW C-Index", 
                                                         "Lasso C-Index", "Lasso IPCW C-Index", "p score"]) # shape (78, 9)

# %%

X_df2 = X_df.copy()
vals_ord = vals.sort_values("IPCW C-Index", axis=0)
feature_order = list(vals_ord.index)
scores = np.zeros((len(feature_order), 2))

for i, curr_feature in tqdm(enumerate(feature_order), total=len(feature_order)):
    clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
    clf.fit(np.array(X_df2), y)
    pred = clf.predict(np.array(X_df2))
    scores[i,0] = concordance_index_censored(y["status"], y["time"], pred)[0]
    scores[i,1] = concordance_index_ipcw(y, y, pred)[0]
    X_df2 = X_df2.drop(curr_feature, axis=1)
    
# This is the performance of the model on the training set using RandomSurvivalForest and eliminating one after another the features
# with the lowest IPCW Concorcance Index taken from vals. This means that the number of features decreases by 1 each iteration. The 
# Index is the feature that gets eliminated from the next row onwards.
vals1 = pd.DataFrame(scores, index=feature_order, columns=["C-Index", "IPCW C-Index"]) # shape (78, 2)

# %%

vals_ord = vals.sort_values("IPCW C-Index", axis=0)
feature_order = list(vals_ord.index)
scores = np.zeros((len(feature_order), 2))

for i, curr_feature in tqdm(enumerate(feature_order), total=len(feature_order)):
    X_df2 = X_df.drop(curr_feature, axis=1)
    clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
    clf.fit(np.array(X_df2), y)
    pred = clf.predict(np.array(X_df2))
    scores[i,0] = concordance_index_censored(y["status"], y["time"], pred)[0]
    scores[i,1] = concordance_index_ipcw(y, y, pred)[0]
    
# This is the performance of the model on the training set using RandomSurvivalForest and eliminating features starting with the one 
# with the lowest IPCW Concorcance Index taken from vals. So the model is always trained with 1 less feature than is in X_df. The Index 
# is the feature that gets eliminated for the current row.
vals2 = pd.DataFrame(scores, index=feature_order, columns=["C-Index", "IPCW C-Index"]) # shape (78, 2)

# %%

vals_ord = vals.sort_values("IPCW C-Index", axis=0)
feature_order = list(vals_ord.index)
scores = np.zeros((len(feature_order), 2))

for i, curr_feature in tqdm(enumerate(feature_order), total=len(feature_order)):
    X_df2 = X_df.copy()
    X_df2[curr_feature] = np.random.permutation(X_df[curr_feature].values)
    clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
    clf.fit(np.array(X_df2), y)
    pred = clf.predict(np.array(X_df2))
    scores[i,0] = concordance_index_censored(y["status"], y["time"], pred)[0]
    scores[i,1] = concordance_index_ipcw(y, y, pred)[0]
    
# This is the performance of the model on the training set using RandomSurvivalForest and permuting the values for one feature.
vals3 = pd.DataFrame(scores, index=feature_order, columns=["C-Index", "IPCW C-Index"]) # shape (78, 2)

# %%

X_sum = np.sum(np.array(X_df).astype(bool), axis=0) / len(X_df)
X_sum_sub = np.sum(np.array(X_sub_df).astype(bool), axis=0) / len(X_sub_df)

# This is a dataframe containing the fraction of each feature in the train and test data that are not 0 (so we can see how often a 
# feature is present). The index indicates the feature and in the first column are the values for the training data, in the second for 
# the test data.
X_nf = pd.DataFrame(list([[val,bal] for val,bal in zip(X_sum, X_sum_sub)]), index=list(X_df.columns), columns=["train data", "test data"]) # shape (78, 2)

# %%

'''
vals.to_csv("C:\\Users\\main\\Desktop\\test files\\Features_Score.csv", index=True)
vals1.to_csv("C:\\Users\\main\\Desktop\\test files\\model_performance_eliminating_features.csv", index=True)
vals2.to_csv("C:\\Users\\main\\Desktop\\test files\\model_performance_eliminating_single_features.csv", index=True)
vals3.to_csv("C:\\Users\\main\\Desktop\\test files\\model_performance_feature_permutation.csv", index=True)
X_nf.to_csv("C:\\Users\\main\\Desktop\\test files\\non_zero_fraction.csv", index=True)
'''

# %%

cox = RandomSurvivalForest()

cox.fit(np.array(X_df["DEPTH_SUM"]).reshape(-1,1), y)
pred = cox.predict(np.array(X_df["DEPTH_SUM"]).reshape(-1,1))
print(concordance_index_censored(y["status"], y["time"], pred)[0])
print(concordance_index_ipcw(y, y, pred)[0])

# %%

# Select features based on a threshold
threshold = 0.62
threshold_p = 1e-0
use_cols = [i for i in vals.index if vals.loc[i].iloc[1] >= threshold and vals.loc[i].iloc[-1] <= threshold_p] # shape (32)

# %%

scores = np.zeros((len(use_cols), 2))

for i, curr_feature in tqdm(enumerate(use_cols), total=len(use_cols)):
    X_df2 = X_df.copy()
    X_df2 = X_df2[use_cols]
    X_df2[curr_feature] = np.random.permutation(X_df[curr_feature].values)
    clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
    clf.fit(np.array(X_df2), y)
    pred = clf.predict(np.array(X_df2))
    scores[i,0] = concordance_index_censored(y["status"], y["time"], pred)[0]
    scores[i,1] = concordance_index_ipcw(y, y, pred)[0]


# This is the performance of the model on the training set using RandomSurvivalForest and permuting the values for one feature.
vals4 = pd.DataFrame(scores, index=use_cols, columns=["C-Index", "IPCW C-Index"]) # shape (78, 2)

# %%

threshold = 0.83
use_cols = [i for i in vals3.index if vals3.loc[i].iloc[1] <= threshold] # shape (32)

# %%

# Prepare dataset with selected features
X_df1 = X_df[use_cols] # shape (3173, len(use_cols))
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

clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
clf.fit(X_train1, y_train1)
scores = np.zeros((len(use_cols), 2))

pred = clf.predict(X_val1)
ci = concordance_index_censored(y_val1["status"], y_val1["time"], pred)[0]
cip = concordance_index_ipcw(y_train1, y_val1, pred)[0]

print(f"Validation C-Index:      {ci:0.4f}")
print(f"Validation IPCW C-Index: {cip:0.4f}")

for i, curr_feature in tqdm(enumerate(use_cols), total=len(use_cols)):
    X_valp = np.copy(X_val1)
    X_valp[:,i] = np.random.permutation(X_val1[:,i])
    pred = clf.predict(X_valp)
    scores[i,0] = concordance_index_censored(y_val1["status"], y_val1["time"], pred)[0]
    scores[i,1] = concordance_index_ipcw(y_train1, y_val1, pred)[0]


# This is the performance of the model on the training set using RandomSurvivalForest and permuting the values for one feature.
vals5 = pd.DataFrame(scores, index=use_cols, columns=["C-Index", "IPCW C-Index"]) # shape (78, 2)

# %%

threshold = 0.7075
use_cols = [i for i in vals5.index if vals5.loc[i].iloc[1] <= threshold] # shape (32)
threshold = 0.7076
use_cols1 = [i for i in vals5.index if vals5.loc[i].iloc[1] <= threshold] # shape (32)
print([val for val in use_cols1 if not val in use_cols])

# %%

# Prepare dataset with selected features
X_df1 = X_df[use_cols] # shape (3173, len(use_cols))
X1 = np.array(X_df1)

# Train-test split with selected features
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.3, random_state=1)

# Train Random Survival Forest model
clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest()
clf.fit(X_train1, y_train1)
#threshold = 0.5

# Evaluate Random Survival Forest model
pt = clf.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], pt)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, pt)[0]
print(f"Training C-Index and IPCW C-Index:   {clf.score(X_train1, y_train1):0.4f}, {concordance_index_ipcw(y_train1, y_train1, clf.predict(X_train1))[0]:0.4f}")
print(f"Validation C-Index and IPCW C-Index: {ind1:0.4f}, {indp1:0.4f}")

# %%

# Train Random Survival Forest model, this is the actual model used for the test set
clf = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
#clf = RandomSurvivalForest()
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
patient_ids_sub = a.patient_ids_sub
X_sub_df1 = X_sub_df[use_cols]
X_sub = np.array(X_sub_df1)

# %%

# Generate predictions for the test set
pt_sub = clf.predict(X_sub)
submission_df1 = pd.DataFrame([patient_ids_sub, pt_sub], index=["ID", "risk_score"]).T
submission_df1.to_csv(data_dir + "\\submission_files\\rsf3.csv", index=False)

























































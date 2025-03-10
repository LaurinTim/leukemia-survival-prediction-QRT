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

#path to directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
#paths to files used for the training
status_file = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
clinical_file = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
molecular_file = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training
#path to the test files used for submissions
clinical_file_test = data_dir+'\\X_test\\clinical_test.csv' #contains clinical information of patients used for the submission
molecular_file_test = data_dir+'\\X_test\\molecular_test.csv' #contains molecular information of patients used for the submission

#features from the clinical data we want to include in the model
clinical_features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']

import os
os.chdir(data_dir)

#import commands from utils
import model_skl_utils as u

# %%

random_seed = 1

u.set_random_seed(random_seed)

# %%

status_df_original = pd.read_csv(status_file)
clinical_df_original = pd.read_csv(clinical_file)
molecular_df_original = pd.read_csv(molecular_file)
effects_map = u.effect_to_survival_map()

# %%

#molecular_dummies_columns = ["CHR", "GENE", "EFFECT"]
molecular_dummies_columns = ["CHR", "GENE"]

d = u.DatasetPrep(status_df_original, clinical_df_original, molecular_df_original, molecular_dummies_columns, effects_map)

# %%

status_df = d.status_df
clinical_df = d.clinical_df
molecular_df = d.molecular_df

# %%

a = u.Dataset(status_df, clinical_df, molecular_df, min_occurences=30)

# %%

X_df = a.X
X = np.array(X_df)
y = a.y

y = np.array([(bool(val[0]), float(val[1])) for val in y], dtype = [('status', bool), ('time', float)])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

# %%

scores = u.fit_and_score_features(X_train, y_train)

# %%

vals = pd.Series(scores, index=X_df.columns).sort_values(ascending=False)

# %%

threshold = 0.52

use_cols = [i for i in vals.index if vals[i]>=threshold]

# %%

X_df1 = X_df[use_cols]
X1 = np.array(X_df1)

X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.3, random_state=1)

# %%

cox = CoxPHSurvivalAnalysis()
cox.fit(X_train1, y_train1)

preds1 = cox.predict(X_val1)
ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], preds1)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, preds1)[0]
print(ind1, indp1)

# %%

threshold = 0.52

use_cols = [i for i in vals.index if vals[i]>=threshold]

# %%

X_df1 = X_df[use_cols]
X1 = np.array(X_df1)

X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.3, random_state=1)

# %%

clf = RandomSurvivalForest(n_estimators=200, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
clf.fit(X_train1, y_train1)

# %%

pt = clf.predict(X_val1)

ind1 = concordance_index_censored(y_val1['status'], y_val1['time'], pt)[0]
indp1 = concordance_index_ipcw(y_train1, y_val1, pt)[0]
print(ind1, indp1)

# %%

cdt, mdt = d.submission_data_prep()
Xtd, idt = a.submission_data(cdt, mdt)
Xtd = Xtd[use_cols]
Xt = np.array(Xtd)[:1000]
ptt = clf.predict(Xt)
ptc = clf.predict(X1[:1000])

# %%

pttt = list([[float(val), float(bal), float(val-bal)] for val,bal in zip(ptt,ptc)])

# %%

clinical_df_sub, molecular_df_sub = d.submission_data_prep()

# %%

X_sub_df, patient_ids_sub = a.submission_data(clinical_df_sub, molecular_df_sub)

# %%

X_sub_df1 = X_sub_df[use_cols]
X_sub = np.array(X_sub_df1)

pt_sub = clf.predict(X_sub)

submission_df = pd.DataFrame([patient_ids_sub, pt_sub], index = ["ID", "risk_score"]).T

submission_df.to_csv(data_dir + "\\submission_files\\rsf0.csv", index = False)

# %%

cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

# %%


preds = cox.predict(X_val)
ind = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
indp = concordance_index_ipcw(y_train, y_val, preds)[0]
print(ind, indp)

# %%

ct, mt = d.submission_data_prep()


























































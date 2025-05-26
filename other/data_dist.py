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
import other.utils as u

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

def hist(col, func=lambda x: x, bins=100, zero=True, density=True):
    fig, ax = plt.subplots(figsize=(10,5))
    
    xh = np.array(X_df[column])
    
    xh_sub = np.array(X_sub_df[column])
    
    if not zero:
        xh = xh[xh!=0]
        xh_sub = xh_sub[xh_sub!=0]
        
    xh = func(xh)
    xh_sub = func(xh_sub)
        
    min_data = min([min(xh), min(xh_sub)])
    max_data = max([max(xh), max(xh_sub)])
    bin_width = (max_data - min_data) / bins
    bins_edges = [min_data + val * bin_width for val in range(bins+1)]
    
    ax.hist(xh, bins=bins_edges, density=density, label='train')
    ax.hist(xh_sub, bins=bins_edges, histtype="step", linewidth=2, density=density, label='test')
    
    #ax.hist(xh, bins=bins, density=density, label='train')
    #ax.hist(xh_sub, bins=bins, histtype="step", linewidth=2, density=density, label='test')
    
    if density:
        ax.set_ylabel(ylabel='Distribution of patient data', fontsize=12)
    else:
        ax.set_ylabel(ylabel='Patients in each bin', fontsize=12)
    ax.set_xlabel(xlabel=column, fontsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.set_title(column + " hist", fontsize=14)
    #ax.set_xlim(0, 10)
        
    return

# %%

column = 'PLT'

# For BM_BLAST (not very good):
#f = lambda x: np.log1p(x)
# For WBC:
#f = lambda x: np.log(x+0.05)
# For ANC:
#f = lambda x: (x+0.1)**0.1
# For PLT:
f = lambda x: x**0.3
# For MONOCYTES:
#f = lambda x: np.log((x+0.1)**0.5)
#f = lambda x: x

hist(column, func=f, bins=50, zero=True, density=True)

# %%

column = "DEPTH_SUM"

fig, ax = plt.subplots(figsize=(10,5))

xx = y["time"][y["status"]==True]
yy = X_df[column][y["status"]==True]

ax.scatter(xx, yy, s=5)
#ax.set_yscale("log")
ax.set_xlabel(xlabel="Survival time")
ax.set_ylabel(ylabel=column)
ax.set_title(column + "/time scatter plot")

plt.show()

# %%

column = "BM_BLAST"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column])
xha = np.array(X_df[column])

xh_sub = np.array(X_sub_df[column])
xha_sub = np.array(X_sub_df[column])

#xh[xh == 0] = sorted(set(xh))[1]
#xh_sub[xh_sub == 0] = sorted(set(xh_sub))[1]
#xh = xh[xh!=0]
#xh_sub = xh_sub[xh_sub!=0]

ax.hist(np.log1p(xh+1e-9), bins=20, density=True)
ax.hist(np.log1p(xh_sub+1e-9), bins=20, histtype="step", linewidth=3, density=True)
#ax.hist(xh, bins=100)
ax.set_title(column + " hist")

plt.show()

# %%

column = "HB"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column])
xha = np.array(X_df[column])

xh_sub = np.array(X_sub_df[column])
xha_sub = np.array(X_sub_df[column])

#xh[xh == 0] = sorted(set(xh))[1]
#xh_sub[xh_sub == 0] = sorted(set(xh_sub))[1]
#xh = xh[xh!=0]
#xh_sub = xh_sub[xh_sub!=0]

ax.hist(xh, bins=15, density=True)
ax.hist(xh_sub, bins=15, histtype="step", linewidth=3, density=True)
#ax.hist(xh, bins=100)
ax.set_title(column + " hist")

plt.show()

# %%

column = "HB"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

#xh = xh[xh!=0]

ax.hist(np.log(xh+1e-9), bins=50)
#ax.hist(xh, bins=50)
ax.set_title(column + " hist")

plt.show()

# %%

column = "PLT"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

#xh = xh[xh!=0]

ax.hist(np.log(xh+1e-9), bins=50)
#ax.hist(xh, bins=50)
ax.set_title(column + " hist")

plt.show()

# %%

column = "WBC"

fig, ax = plt.subplots(figsize=(10,5))

xha = np.array(X_df[column][y["status"]==True])
xh = np.array(X_df[column])

#xh = xh[xh!=0]

#ax.hist(np.log((xh-0.15)+1e-9), bins=60)
ax.hist(xh, bins=200)
ax.set_title(column + " hist")

plt.show()

# %%

column = "ANC"

fig, ax = plt.subplots(figsize=(10,5))

xha = np.array(X_df[column][y["status"]==True])
xh = np.array(X_df[column])

#xh = xh[xh!=0]

#ax.hist(np.log((xh+1)*1e-9), bins=200)
ax.hist(xh, bins=200)
ax.set_title(column + " hist")

plt.show()

# %%

column = "MONOCYTES"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

#xh = xh[xh!=0]

ax.hist(np.log(xh+1e-9), bins=50)
#ax.hist(xh, bins=50)
ax.set_title(column + " hist")

plt.show()

# %%

column = "VAF_SUM"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

xh = xh[xh!=0]

#ax.hist(xh**0.5, bins=30)
ax.hist(xh, bins=50)
ax.set_title(column + " hist")

plt.show()

# %%

column = "VAF_MEDIAN"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

xh = xh[xh!=0]

ax.hist(xh**0.5, bins=30)
ax.set_title(column + " hist")

plt.show()

# %%

column = "DEPTH_SUM"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

xh = xh[xh!=0]

ax.hist(xh**0.2, bins=50)
#ax.hist(xh, bins=100)
ax.set_title(column + " hist")

plt.show()

# %%

column = "DEPTH_MEDIAN"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

xh = xh[xh!=0]

ax.hist(xh**0.2, bins=50)
ax.set_title(column + " hist")

plt.show()

# %%

column = "DEPTH_MEDIAN"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

xh = xh[xh!=0]

ax.hist(xh**0.2, bins=50)
ax.set_title(column + " hist")

plt.show()

# %%

column = "EFFECT_MEDIAN_SURVIVAL"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

#xh = xh[xh!=0]

ax.hist(xh, bins=200)
ax.set_title(column + " hist")

plt.show()

# %%

column = "MUTATIONS_NUMBER"

fig, ax = plt.subplots(figsize=(10,5))

xh = np.array(X_df[column][y["status"]==True])
xha = np.array(X_df[column])

#xh = xh[xh!=0]

ax.hist(xh, bins=18)
ax.set_title(column + " hist")

plt.show()

# %%

ytest = y[X_df["EFFECT_MEDIAN_SURVIVAL"]==0]
ytest1 = y[X_df["EFFECT_MEDIAN_SURVIVAL"]>0]
xt = np.array([val for val,bal in zip(X_df["EFFECT_MEDIAN_SURVIVAL"], y["status"]) if val>0 and not bal])

























































































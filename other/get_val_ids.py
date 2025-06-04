import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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
X_train = a.X # shape (3173, 78)
X_test = a.X_sub # shape (1193, 78)

# Align test and train features
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# %%

# Standardize and apply PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#X_train_pca = X_train_scaled
#X_test_pca = X_test_scaled

# Adversarial validation setup
X_combined = np.vstack([X_train_pca, X_test_pca])
y_combined = np.hstack([np.zeros(X_train_pca.shape[0]), np.ones(X_test_pca.shape[0])])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_combined, y_combined)
probs = clf.predict_proba(X_combined[:X_train_pca.shape[0]])[:, 1]  # Probability of being test-like

# Score and select top 25% most test-like training samples
threshold = np.percentile(probs, 90)
selected_indices = np.where(probs >= threshold)[0]

# Output selected validation set IDs
selected_ids = pd.DataFrame(a.patient_ids[[selected_indices]].reshape(-1,1), columns=['ID'])
selected_ids.to_csv(data_dir + '\\val_ids\\Validation_IDs_90.csv', index=False)

# %%

prob_df = pd.DataFrame(np.zeros(X_train.shape), columns=X_train.columns)
pca = PCA(n_components=1, random_state=42)
scaler = StandardScaler()
clf = RandomForestClassifier(n_estimators=100, random_state=42)

for col in tqdm(X_train.columns, total=X_train.shape[1]):
    curr_X_train = X_train[col].to_frame()
    curr_X_test = X_test[col].to_frame()
    
    curr_train_scaled = scaler.fit_transform(curr_X_train)
    curr_test_scaled = scaler.transform(curr_X_test)
    
    curr_train_pca = pca.fit_transform(curr_train_scaled)
    curr_test_pca = pca.transform(curr_test_scaled)
    
    curr_X_combined = np.vstack([curr_train_pca, curr_test_pca])
    curr_y_combined = np.hstack([np.zeros(curr_train_pca.shape[0]), np.ones(curr_test_pca.shape[0])])
    
    clf.fit(curr_X_combined, curr_y_combined)
    curr_probs = clf.predict_proba(curr_X_combined[:curr_train_pca.shape[0]])[:, 1]
    
    prob_df[col] = curr_probs


# %%

percent_threshold = 95

threshold = np.array([np.percentile(val, percent_threshold) for val in np.array(prob_df).T])

threshold = pd.DataFrame(threshold, index=X_train.columns)















































































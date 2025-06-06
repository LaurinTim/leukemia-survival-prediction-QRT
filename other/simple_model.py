import re
import pandas as pd
import numpy as np

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
clinical_df = clinical_df[[True if val in patient_ids else False for val in clinical_df['ID']]]
molecular_df = molecular_df[[True if val in patient_ids else False for val in molecular_df['ID']]]

clinical_df_test = pd.read_csv(clinical_file_test)
molecular_df_test = pd.read_csv(molecular_file_test)

# %%

clinical_df, clinical_df_test = u.transform_columns(clinical_df, clinical_df_test, ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'PLT'])
clinical_df, clinical_df_test = u.fill_nan(clinical_df, clinical_df_test, ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT'], method='mean')

cyto_train, cyto_test = u.cyto_patient_risk(clinical_df['CYTOGENETICS'], clinical_df_test['CYTOGENETICS'])
clinical_df = pd.concat([clinical_df, cyto_train], axis=1)
clinical_df_test = pd.concat([clinical_df_test, cyto_test], axis=1)

gender_train, gender_test = u.patient_gender(clinical_df['CYTOGENETICS'], clinical_df_test['CYTOGENETICS'])
clinical_df = pd.concat([clinical_df, gender_train], axis=1)
clinical_df_test = pd.concat([clinical_df_test, gender_test], axis=1)

# %%

























































































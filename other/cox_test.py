import warnings
warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm
import random
from operator import itemgetter
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

from scipy.stats import logrank

from time import time

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

# %%

clin_train = pd.read_csv(clinical_file)
clin_test = pd.read_csv(clinical_file_test)
mol_train = pd.read_csv(molecular_file)
mol_test  = pd.read_csv(molecular_file_test)
target   = pd.read_csv(status_file)

target = target.dropna(subset=["OS_YEARS", "OS_STATUS"])
clin_train = clin_train[[True if val in list(target["ID"]) else False for val in list(clin_train["ID"])]]
mol_train = mol_train[[True if val in list(target["ID"]) else False for val in list(mol_train["ID"])]]

medians = clin_train.median(numeric_only=True)
for df in [clin_train, clin_test]:
    for col in ['BM_BLAST','WBC','ANC','HB','PLT']:
        df[col] = df[col].fillna(medians[col])
    # Drop columns with too many missing or complex data
    df.drop(['MONOCYTES','CYTOGENETICS'], axis=1, inplace=True)

for df in [clin_train, clin_test]:
    for col in ['BM_BLAST','WBC','ANC','PLT']:
        df[f'log_{col}'] = np.log1p(df[col])
        df.drop(col, axis=1, inplace=True)

# %%

# 4. Aggregate molecular features for training set
mol_agg = mol_train.groupby('ID').agg(
    mut_count=('GENE','count'),
    mean_vaf=('VAF','mean'),
    max_vaf=('VAF','max'),
    median_vaf=('VAF','median')
).reset_index()

mol_test_agg = mol_test.groupby('ID').agg(
    mut_count=('GENE','count'),
    mean_vaf=('VAF','mean'),
    max_vaf=('VAF','max'),
    median_vaf=('VAF','median')
).reset_index()

# 5. Create binary flags for top mutated genes
top_genes = mol_train['GENE'].value_counts().head(110).index.tolist()
genes = mol_train[mol_train['GENE'].isin(top_genes)]
gene_flags = genes.assign(val=1).pivot_table(index='ID', columns='GENE', values='val', 
                                            aggfunc='sum', fill_value=0)
gene_flags = (gene_flags > 0).astype(int).reset_index()

# 6. Merge clinical and molecular features for training
train_df = pd.merge(clin_train, target, on='ID', how='inner')
train_df = pd.merge(train_df, mol_agg, on='ID', how='left')
train_df = pd.merge(train_df, gene_flags, on='ID', how='left')

# 7. Fill missing molecular features for patients without mutations
for col in ['mut_count','mean_vaf','max_vaf','median_vaf']:
    train_df[col] = train_df[col].fillna(0)
for gene in top_genes:
    if gene not in train_df: 
        train_df[gene] = 0
    train_df[gene] = train_df[gene].fillna(0)

# %%

# 9. Prepare test data with same transformations
test_df = pd.merge(clin_test, mol_test_agg, on='ID', how='left')
test_df = pd.merge(test_df, gene_flags, on='ID', how='left')  # gene_flags reused from training

for col in ['mut_count','mean_vaf','max_vaf','median_vaf']:
    test_df[col] = test_df[col].fillna(0)
for gene in top_genes:
    if gene not in test_df: 
        test_df[gene] = 0
    test_df[gene] = test_df[gene].fillna(0)

# 10. Align dummy variables: add missing CENTER columns to test as zeros
#center_cols = [c for c in train_df.columns if c.startswith('CENTER_')]
#for c in center_cols:
#    test_df[c] = 0
train_df.drop('CENTER', axis=1, inplace=True, errors='ignore')
test_df.drop('CENTER', axis=1, inplace=True, errors='ignore')

train_model = train_df.drop(columns=['ID'])

#train_model['OS_YEARS'] = train_model['OS_YEARS']+1e-5

# %%

# 11. Fit Cox model on training data
cph = CoxPHFitter()
cph.fit(train_model, duration_col='OS_YEARS', event_col='OS_STATUS')
summary = cph.summary  # optional: view coefficients and stats

'''
use_cols = [val for val,bal in zip(summary.index, summary['p']) if bal<=0.98]

tst = train_model[use_cols + ['OS_YEARS', 'OS_STATUS']]

cph.fit(tst, duration_col='OS_YEARS', event_col='OS_STATUS')
summary_tst = cph.summary  # optional: view coefficients and stats

tst = tst.drop(columns=['OS_YEARS', 'OS_STATUS'])
'''
tst = train_model.drop(columns=['OS_YEARS', 'OS_STATUS'])

pred = cph.predict_partial_hazard(tst)
c = concordance_index_censored([bool(val) for val in train_model['OS_STATUS']], train_model['OS_YEARS'], pred)[0]
yy = np.array([(bool(val), float(bal)) for val,bal in zip(list(train_model['OS_STATUS']), list(train_model['OS_YEARS']))], dtype=[('status', bool), ('years', float)])
ci = concordance_index_ipcw(yy, yy, pred)[0]
print(c, ci)

# %%



cft = CoxPHSurvivalAnalysis(verbose=2, alpha=99)
cft.fit(np.array(tst), yy)

pred = cft.predict(np.array(tst))
c = concordance_index_censored([bool(val) for val in train_model['OS_STATUS']], train_model['OS_YEARS'], pred)[0]
ci = concordance_index_ipcw(yy, yy, pred)[0]
print(c, ci)

# %%



# %%

# 12. Predict risk scores for test set
test_model = test_df.drop(columns=['ID'])
risk_scores = cph.predict_partial_hazard(test_model).values.flatten()




























































































import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES', 'NSM']

#set working directory to data_dir
import os
os.chdir(data_dir)

#import method to create test_results file
from create_test_results_file import test_results

# %%

random_seed = 1

np.random.seed(random_seed)
random.seed(random_seed)

# %% Transformers for DatasetGen
    
class TransStatus(object):
    def __call__(self, sample):
        #res = np.array([(sample[0], sample[1])], dtype = [('s', bool), ('y', float)])
        res = np.array(sample)
        #res = np.array([(sample[0], sample[1])], dtype = [('status', bool), ('years', float)], copy=True)
        #res = np.array(sample)
        return torch.tensor(res)#torch.tensor(res, dtype=torch.float32)
        
class TransClinical(object):
    def __call__(self, sample):
        res = np.array(sample.loc[sample.ID!='n', features[:-1]])
        return torch.tensor(res[0]).float()

class TransMolecular(object):
    def __call__(self, sample):
        return torch.tensor([len(sample)])

# %% Generate a custom dataset

class DatasetGen(Dataset):
    def __init__(self, annotations_file, clinical_file, molecular_file, clinical_transform=None, molecular_transform=None, status_transform=None):
        self.patient_status = pd.read_csv(annotations_file).dropna(subset=['OS_YEARS', 'OS_STATUS'])
        
        self.patient_clinical = pd.read_csv(clinical_file)
        self.patient_clinical = self.patient_clinical.loc[self.patient_clinical['ID'].isin(self.patient_status['ID'])]
        self.patient_clinical = self.patient_clinical.fillna(0)
        
        self.patient_molecular = pd.read_csv(molecular_file)
        self.patient_molecular = self.patient_molecular.loc[self.patient_molecular['ID'].isin(self.patient_status['ID'])]
        
        self.clinical_transform = clinical_transform
        self.molecular_transform = molecular_transform
        self.status_transform = status_transform
        
    def __len__(self):
        return len(self.patient_status)
    
    def __getitem__(self, idx):
        patient_id = self.patient_status.iloc[idx, 0]
        os_years = self.patient_status.iloc[idx, 1]
        os_status = self.patient_status.iloc[idx, 2]
        status = np.array([os_status, os_years])
        info_clinical = self.patient_clinical.loc[self.patient_clinical.ID == patient_id]
        info_molecular = self.patient_molecular[self.patient_molecular.ID == patient_id]
        
        if self.clinical_transform and self.molecular_transform:
            info_clinical = self.clinical_transform(info_clinical)
            info_molecular = self.molecular_transform(info_molecular)
            info = torch.cat((info_clinical, info_molecular))
            
        if self.status_transform:
            status = self.status_transform(status)
            
        return info, (bool(status[0]), status[1])

# %% Get dataset from the training data, split it into training and test, create dataloaders for both

data = DatasetGen(data_dir+'\\target_train.csv', data_dir+'\\X_train\\clinical_train.csv', data_dir+'\\X_train\\molecular_train.csv', 
                  clinical_transform=TransClinical(), molecular_transform=TransMolecular(), status_transform=TransStatus())

X = np.array([val[0].cpu().numpy() for val in data])
y = np.array([(bool(val[1][0]), float(val[1][1])) for val in data], dtype = [('status', bool), ('time', float)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=94)

bad_seed = bool(np.max(y_train['time']) < np.max(y_test['time']))
if bad_seed:
    print('-'*100)
    print('ERROR: THIS IS A BAD SEED, THE MAX TIME IN TRAIN IS SMALLER THAN IN VAL OR TEST, PLEASE SPLIT DATA AGAIN')
    print('-'*100)

X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=94)

# %%

cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

preds = cox.predict(X_val)
ind = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
indp = concordance_index_ipcw(y_train, y_val, preds)[0]
print(ind, indp)

# %%

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores


scores = fit_and_score_features(X_train, y_train)
print(pd.Series(scores, index=features).sort_values(ascending=False))

# %%

pipe = Pipeline(
    [
        ("select", SelectKBest(fit_and_score_features, k=4)),
        ("model", CoxPHSurvivalAnalysis()),
    ]
)

pipe.fit(X_train, y_train)
transformer, final_estimator = (s[1] for s in pipe.steps)
pd.Series(final_estimator.coef_, index=['BM_BLAST', 'HB', 'PLT', 'NSM'])





















































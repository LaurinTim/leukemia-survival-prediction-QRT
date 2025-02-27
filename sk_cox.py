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

#path to directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
#paths to files used for the training
file_status = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
file_clinical = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
file_molecular = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training
#path to the test files used for submissions
file_clinical_test = data_dir+'\\X_test\\clinical_test.csv' #contains clinical information of patients used for the submission
file_molecular_test = data_dir+'\\X_test\\molecular_test.csv' #contains molecular information of patients used for the submission

#features from the clinical data we want to include in the model
features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']

#features from the molecular data we want to include in the model
features_mol = ['CHR', 'GENE', 'EFFECT']

#set working directory to data_dir
import os
os.chdir(data_dir)

#import commands from utils
from utils import get_gene_embeddings, get_gene_map, get_gene_embedding, effect_to_survival_map, fit_and_score_features

# %% set random seed of numpy, random and pytorch

random_seed = 1

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

# %% get patiend status, clinical and molecular data

########################
#patient status
########################

#dataframe containing the status of patients, only keep patients with a valid status
data_st = pd.read_csv(file_status).dropna(subset=['OS_YEARS', 'OS_STATUS'])
#array containing all the patient IDs we include in the analysis
idp = np.array(data_st.ID)
#drop 'ID' columns from data_st
data_st = data_st[['OS_YEARS', 'OS_STATUS']] #shape=(3173, 2)

########################
#patient clinical data
########################

#dataframe containing clinical data of all patients
data_cl = pd.read_csv(file_clinical)
#only keep patients with valid status
data_cl = data_cl.loc[data_cl['ID'].isin(idp)]
#array containing 'ID' of all valid patients in data_cl
id_cl = np.array(data_cl.ID)
#only keep the columns of data_cl defined in features
data_cl = data_cl[features]
#fill cells that are not defined in features with 0 (this can be improved, e.g. take mean or median of the whole column)
data_cl = data_cl.fillna(0) #shape=(3173, 6)

########################
#patient molecular data
########################

#dataframe containing molecular data of all patients
data_mol = pd.read_csv(file_molecular)
#only keep patients with valid status
data_mol = data_mol.loc[data_mol['ID'].isin(idp)]
#array containing 'ID' of all valid patients in data_mol
id_mol = np.array(data_mol.ID)
#array containing 'VAF' of all valid patients in data_mol
vaf = np.nan_to_num(np.array(data_mol.VAF), nan=0.0).reshape(-1, 1)
#take the log of vaf and scale it using StandardScaler
scaler = StandardScaler()
scaled_vaf = scaler.fit_transform(np.log(vaf+1e-4))
#only keep columns defined in features_mol in data_mol
data_mol = data_mol[['ID'] + features_mol] #shape=(10545, 3)

# %% use Ordinal Encoding to encode the "CHR" column in data_mol containing information about the chromosome affected by the somatic mutation

#use OrdinalEncoder from sklearn.preprocessing
enc = OrdinalEncoder()
#ordinal encoding of the chromosomes
chroms = enc.fit_transform(data_mol[['CHR']])
#if no chromosome is specified in a row of data_mol set the corresponding element in chroms to -1
chroms = np.nan_to_num(chroms, nan=-1)
#drop the 'CHR' column from data_mol
data_mol = data_mol.drop(columns=['CHR'])

# %% Use embeddings to represent the different genes in data_mol["GENE"]

#dimension of embeddings
embedding_dim = 50

#dictionary that maps the genes in data_mol['GENE'] to the integers from 0 to the number of unique genes where 0 is assigned is no gene is specified in data_mol['GENE']
gene_to_idx = get_gene_map(file_molecular, file_status)

#get embeddings for the genes using pytorch
gene_embeddings = get_gene_embeddings(embedding_dim, file_molecular, file_status) #shape=(number of unique genes + 1, embedding_dim)

#array filled with zeros in which to put get_gene_embedding of each patient, each row corresponds to one patient
mol_gene = np.zeros((len(idp), embedding_dim))

#get the genes for each patient and set mol_gene for the current patient to get_gene_embedding of the current genes
for i in tqdm(range(len(idp))):
    #ID of current patient
    curr_patient_id = idp[i]
    #genes in data_mol['GENE'] corresponding to the current patient
    curr_genes = np.array(data_mol.loc[id_mol==curr_patient_id]['GENE'])
    #get the averaged embedding of the genes in curr_gene and save it in mol_gene[i]
    mol_gene[i] = get_gene_embedding(curr_genes, gene_embeddings, gene_to_idx, embedding_dim)

#drop the 'GENE' column from data_mol
data_mol = data_mol.drop(columns=['GENE'])

# %% Use target encoding to represent the different effects in data_mol["EFFECT"]

#dictionary mapping effects to survival times
effects_survival_map = effect_to_survival_map(file_molecular, file_status)

#array filled with zeros in which to put the averaged survival time from all effects that a patient has
mol_effect = np.zeros((len(idp), 2))

#if a patient has no somatic mutations set the corresponding element in mol_effect to the median survival time of all patients
global_median_survival = np.median(data_st["OS_YEARS"]) #np.median(data_st.loc[data_st["OS_STATUS"]== 0]["OS_YEARS"])

#go through all patients to compute mol_effect
for i in tqdm(range(len(idp))):
    #ID of current patient
    curr_patient_id = idp[i]
    #all effects of the current patient
    curr_effects = np.array(data_mol.loc[id_mol==curr_patient_id]['EFFECT'])
    #check if the current patient has at least 1 effect
    if len(curr_effects)>0:
        #get tuple (length=len(curr_effects)) with the survival times associated with the elements of curr_effects
        curr_survival = np.array(itemgetter(*curr_effects)(effects_survival_map)).reshape(-1, 1)
        #get the values of vaf for the current patient and use these as weights for the corresponding effects
        curr_vaf = vaf[id_mol==curr_patient_id]
        #normalize the weights
        curr_vaf = (curr_vaf/np.sum(curr_vaf)) if np.sum(curr_vaf>0) else 1
        #set mol_effect[i] to the average of curr_survival
        mol_effect[i] = [np.average(curr_survival*curr_vaf), len(curr_effects)]
    else:
        #if the current patient has no effects set mol_effect[i] to the median survival time of all patients
        mol_effect[i] = [global_median_survival, 0]

# %% get the input for CoxPHSurvivalAnalysis

#training data for the model, concatenate data_cl.values, mol_effect and mol_gene
X = np.concatenate((data_cl.values, mol_effect, mol_gene), axis=1)

#target data for the model
y = np.array([(bool(val[1]), float(val[0])) for val in np.array(data_st)], dtype = [('status', bool), ('time', float)])

#split data in training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

#check if the max time in the training set is larger than in the validation set, if not another random seed has to be used
bad_seed = bool(np.max(y_train['time']) < np.max(y_val['time']))
if bad_seed:
    print('-'*100)
    print('ERROR: THIS IS A BAD SEED, THE MAX TIME IN TRAIN IS SMALLER THAN IN VAL OR TEST, PLEASE SPLIT DATA AGAIN')
    print('-'*100)

# %%

cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

preds = cox.predict(X_val)
ind = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
indp = concordance_index_ipcw(y_train, y_val, preds)[0]
print(ind, indp)

# %%

scores = fit_and_score_features(X_train, y_train)

# %%

vals = pd.Series(scores, index=features+["Effect_survival", "number_of_mutations"]+["GENE::"+str(i) for i in range(embedding_dim)]).sort_values(ascending=False)

# %% Use one hot encoding for the different effects

data_mol = pd.get_dummies(data_mol, columns=['EFFECT'])

min_occurences = 5
data_mol = data_mol.drop(columns=['ID'])
data_mol_sum = np.sum(data_mol, axis=0)
sparse_features = data_mol.columns[(data_mol_sum < min_occurences)]
data_mol = data_mol.drop(columns=sparse_features)

mol_effect = np.zeros((idp.shape[0], data_mol.shape[1]))

for i in tqdm(range(len(idp))):
    #ID of current patient
    curr_patient_id = idp[i]
    #all effects of the current patient
    curr_mol = data_mol.iloc[id_mol == curr_patient_id]
    #sum over the effects and set mol_effect[i] to this
    mol_effect[i] = np.array(np.sum(curr_mol, axis=0))


















































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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']
#features_mol = ['CHR', 'GENE', 'EFFECT']
features_mol = ['CHR', 'GENE', 'EFFECT']

#set working directory to data_dir
import os
os.chdir(data_dir)

#import method to create test_results file
from create_test_results_file import test_results

from utils import fit_and_score_features

# %%

random_seed = 1

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

# %%

data_st = pd.read_csv(data_dir+'\\target_train.csv').dropna(subset=['OS_YEARS', 'OS_STATUS'])
idp = np.array(data_st.ID)
data_st = data_st[['OS_YEARS', 'OS_STATUS']]

data_cl = pd.read_csv(data_dir+'\\X_train\\clinical_train.csv')
data_cl = data_cl.loc[data_cl['ID'].isin(idp)]
id_cl = np.array(data_cl.ID)
data_cl = data_cl[features]
data_cl = data_cl.fillna(0) #shape = (3173, 6)

# %%

data_mol = pd.read_csv(data_dir+'\\X_train\\molecular_train.csv')
data_mol = data_mol.loc[data_mol['ID'].isin(idp)]

scaler = MinMaxScaler()
vaf_mol = np.nan_to_num(np.array(data_mol.VAF))
vaf_mol = scaler.fit_transform(vaf_mol.reshape(-1,1)).reshape(1,-1)[0]*5

id_mol = np.array(data_mol.ID)
data_mol = data_mol[features_mol]

enc = OrdinalEncoder()
chroms = enc.fit_transform(data_mol[['CHR']])
chroms = np.nan_to_num(chroms, nan=-1)
data_mol = data_mol.drop(columns=['CHR'])

# %%

unique_genes = sorted(data_mol['GENE'].unique())
gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes, start=1)}
gene_to_idx['UNKNOWN'] = 0
#data_mol['gene_index'] = data_mol['GENE'].map(lambda g: gene_to_idx.get(g, 0))

class GeneEmbeddingModel(nn.Module):
    def __init__(self, num_genes, embedding_dim):
        super(GeneEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_genes, embedding_dim)
        
    def forward(self, gene_idx):
        return self.embedding(gene_idx)
    
embedding_dim = 50
num_genes = len(unique_genes) + 1
gene_model = GeneEmbeddingModel(num_genes, embedding_dim)
with torch.no_grad():
    gene_embeddings = gene_model.embedding.weight.cpu().numpy()
    
def get_gene_embedding(patient_genes, vector_size=50):
    """Averages embeddings of all mutated genes for a patient."""
    indices = [gene_to_idx.get(g, 0) for g in patient_genes]
    vectors = [gene_embeddings[idx] for idx in indices]
    
    if len(vectors)>0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)
    
mol_gene = pd.DataFrame(embedding_dim, index = np.arange(len(idp)), columns=['GENE:' + str(i) for i in range(embedding_dim)])

for i in tqdm(range(len(idp))):
    curr_mol = data_mol[id_mol == idp[i]]
    curr_genes = np.array(curr_mol['GENE'])
    mol_gene.iloc[i] = get_gene_embedding(curr_genes, vector_size=embedding_dim)

#data_mol = data_mol.drop(columns=['GENE'])

# %%

data_mol = pd.get_dummies(data_mol) #shape = (10545, 137)
data_mol_sum = data_mol.sum(axis=0)
min_occurences = 5
sparse_features = data_mol.columns[(data_mol_sum < min_occurences)]
data_mol = data_mol.drop(columns=sparse_features)
data_mol.insert(0, 'CHR', chroms)

# %%

data = pd.DataFrame(0, index=np.arange(len(idp)), columns=list(data_cl.columns)+list(data_mol.columns))#+['VAF']), shape = (3173, 166)

for i in tqdm(range(len(idp))):
    curr_cl = data_cl[id_cl == idp[i]].reset_index(drop=True)
    curr_mol = data_mol[id_mol == idp[i]]
    curr_mol_chrom = np.array(curr_mol['CHR'])
    #curr_vaf_mol = np.array(np.sum(vaf_mol[id_mol == idp[i]]))
    curr_mol = curr_mol.sum(0).to_frame().transpose().reset_index(drop=True)
    #curr_mol = (curr_mol.T * vaf_mol[id_mol == idp[i]]).T.sum(0).to_frame().transpose()
    #if i % 100 == 0: print(np.sum(np.array(curr_mol)))
    #data.iloc[i] = np.append(np.array(curr_cl), np.append(np.array(curr_mol), curr_vaf_mol))
    data.iloc[i] = np.append(np.array(curr_cl), np.array(curr_mol))

# %% Get dataset from the training data, split it into training and test, create dataloaders for both

X = np.array(data)
y = np.array([(bool(val[1]), float(val[0])) for val in np.array(data_st)], dtype = [('status', bool), ('time', float)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=94)

bad_seed = bool(np.max(y_train['time']) < np.max(y_test['time']))
if bad_seed:
    print('-'*100)
    print('ERROR: THIS IS A BAD SEED, THE MAX TIME IN TRAIN IS SMALLER THAN IN VAL OR TEST, PLEASE SPLIT DATA AGAIN')
    print('-'*100)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.0001, random_state=random_seed)

# %%

cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

preds = cox.predict(X_val)
ind = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
indp = concordance_index_ipcw(y_train, y_val, preds)[0]
print(min_occurences, ind, indp)

# %%

scores0 = fit_and_score_features(X_train, y_train)

# %%

vals0 = pd.Series(scores0, index=data.columns).sort_values(ascending=False)

# %%

preds_test = cox.predict(X_test)
ind_test = concordance_index_censored(y_test['status'], y_test['time'], preds_test)[0]
indp_test = concordance_index_ipcw(y_train, y_test, preds_test)[0]
print(min_occurences, ind_test, indp_test)

# %%

final_cl_df = pd.read_csv(data_dir + "\\X_test\\clinical_test.csv").fillna(value=0)
final_idp = np.array(final_cl_df.ID)
final_cl_df = final_cl_df[features]
final_mol_df = pd.read_csv(data_dir + "\\X_test\\molecular_test.csv").fillna(value=0)
final_mol_id = np.array(final_mol_df.ID)
final_mol_df = final_mol_df[features_mol]
final_mol_df = pd.get_dummies(final_mol_df)
final_sparse_features = [val for val in sparse_features if val in final_mol_df.columns]
final_mol_df = final_mol_df.drop(columns=final_sparse_features)
final_mol_df = final_mol_df.reindex(columns=data_mol.columns, fill_value=0)

final_df = pd.DataFrame(0, index=np.arange(len(final_idp)), columns=list(final_cl_df.columns)+list(final_mol_df.columns))

# %%

for i in tqdm(range(len(final_idp))):
    curr_cl = final_cl_df[final_idp == final_idp[i]].reset_index(drop=True)
    curr_mol = final_mol_df[final_mol_id == final_idp[i]]
    curr_mol = curr_mol.sum(0).to_frame().transpose().reset_index(drop=True)
    final_df.iloc[i] = np.append(np.array(curr_cl), np.array(curr_mol))

# %%

final_pred = cox.predict(final_df)
sub_df = pd.DataFrame([final_idp, final_pred], index = ['ID', 'risk_score']).transpose()

# %%

sub_df.to_csv(data_dir + '\\submission_files\\sk1.csv', index = False)

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





















































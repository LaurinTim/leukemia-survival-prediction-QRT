#file containing different commands used in the analysis

import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter
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

from time import time

#Directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

# %%

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

class EmbeddingModel(torch.nn.Module):
    '''
    
    Embedding module for the gene and chromosome embeddings
    
    '''
    def __init__(self, num, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(num, embedding_dim)
        
    def forward(self, idx):
        return self.embedding(idx)
    
def fit_and_score_features(X, y):
    '''

    Parameters
    ----------
    X : numpy.ndarray
        Array containing the data used to train the model.
    y : numpy.ndarray
        Structured array where each element is a tuple of length 2 and type 
        [(bool), (float)] containing the target for the training.

    Returns
    -------
    scores : numpy.ndarray
        Array (length=X.shape[1]) containing the concordance indices for 
        each feature in X.

    '''
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in tqdm(range(n_features)):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

# %%

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

file_status = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
file_clinical = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
file_molecular = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training

# %%

class DataPrep():
    def __init__(self, status_file, clinical_file, molecular_file):
        self.status_file = status_file
        self.clinical_file = clinical_file
        self.molecular_file = molecular_file
        
        self.status_df = pd.read_csv(status_file).dropna(subset=["OS_YEARS", "OS_STATUS"])
        self.status_df = self.status_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.patient_ids = np.array(self.status_df.loc[:,"ID"])
        self.num_patients = self.patient_ids.shape[0]
        self.status_columns = np.array(self.status_df.columns)
        self.status_arr = self.status_df.to_numpy(copy=True)
        
        self.clinical_df = pd.read_csv(clinical_file)
        self.clinical_df = self.__valid_patients_df(self.clinical_df)
        self.clinical_df = self.__fillna_df(self.clinical_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
        self.clinical_df = self.clinical_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.clinical_columns = np.array(self.clinical_df.columns)
        self.clinical_arr = self.clinical_df.to_numpy(copy=True)
        
        self.molecular_df = pd.read_csv(molecular_file)
        self.molecular_df = self.__valid_patients_df(self.molecular_df)
        self.__molecular_id_fill()
        self.molecular_df = self.__fillna_df(self.molecular_df, ["START", "END", "VAF", "DEPTH"])
        self.molecular_df = self.molecular_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.molecular_columns = np.array(self.molecular_df.columns)
        self.molecular_arr = self.molecular_df.to_numpy(copy=True)
        self.__molecular_id_sort()
        self.__get_effects_to_survival_map()
        
    def __valid_patients_df(self, df):
        return_df = df[df.loc[:,"ID"].isin(self.patient_ids)]
        return return_df.reset_index(drop=True)
    
    def __fillna_df(self, df, columns):
        #return_df = df.fillna({col: df[col].median() for col in df.select_dtypes(include=['float']).columns})
        return_df = df.fillna({col: 0 for col in df.select_dtypes(include=['float']).columns if col not in ["CHR"]})
        return return_df
    
    def __molecular_id_fill(self) -> None:
        curr_len = self.molecular_df.shape[0]+1
        for i in range(self.num_patients):
            curr_id = self.patient_ids[i]
            if not curr_id in list(self.molecular_df.loc[:,"ID"]):
                curr_len = self.molecular_df.shape[0]+1
                curr_arr = np.array([np.nan]*(self.molecular_df.shape[1]))
                self.molecular_df.loc[curr_len] = curr_arr
                self.molecular_df.iloc[self.molecular_df.shape[0]-1,0] = curr_id
                
    def __molecular_id_sort(self) -> None:
        self.molecular_split = np.split(self.molecular_arr, np.unique(self.molecular_arr[:,0], return_index=True)[1][1:])
        
    def __get_effects_to_survival_map(self) -> None:
        comb_df = self.molecular_df.merge(self.status_df, on='ID', how='left')
        
        #use lifelines.KaplanMeierFitter to account for the censoring when evaluating the map
        kmf = KaplanMeierFitter()
        self.effects_survival_map = {}
        
        #use groupby to create a tuple for each unique effect containing (effect name, data_molecular[data_molecular['EFFECT] == efect name]) which get passed on to loop as effect and subset
        for effect, subset in comb_df.groupby("EFFECT"):
            #use Kaplan-Meier estimate using subset["OS_YEARS"] as durations and subset["OS_STATUS"] for the censoring
            kmf.fit(durations=list(subset["OS_YEARS"]), event_observed=list(subset["OS_STATUS"]))
            #check if kmf can calculate the median (it cannot if subset has to few values from patients with status=1, this is the case for the the effect "inframe_variant")
            if kmf.median_survival_time_ == np.inf: #kmf cannot compute the median
                #set the value in the map for the current effect to the max survival time in subset
                self.effects_survival_map[effect] = np.float64(np.max(subset["OS_YEARS"]))
            else:
                #set the value in the map for the current effect to the median survival time obtained using the Kaplan Meier estimate
                self.effects_survival_map[effect] = kmf.median_survival_time_

# %%

dat = DataPrep(file_status, file_clinical, file_molecular)

# %%

###################################################################################################
#Commands to get X_train, y_train from files
###################################################################################################
#maybe some columns should not be set to the median value if nan, eg. VAF, check this later
#in Dataset.get_encoded_chromosomes nan values are set to -1, check if it makes a difference to set this to 0

class Dataset():
    def __init__(self, status_arr, clinical_arr, molecular_arr, 
                 status_columns=np.array(['ID', 'OS_YEARS', 'OS_STATUS']), 
                 clinical_columns=np.array(['ID', 'CENTER', 'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'CYTOGENETICS']), 
                 molecular_columns=np.array(['ID', 'CHR', 'START', 'END', 'REF', 'ALT', 'GENE', 'PROTEIN_CHANGE', 'EFFECT', 'VAF', 'DEPTH']),
                 clinical_features=np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']), 
                 effects_survival_map=None,
                 gene_embedding_dim=50, chromosome_embedding_dim=10, chromosomes_min_occurences=5):
        
        self.clinical_features = clinical_features
        self.gene_embedding_dim = gene_embedding_dim
        self.chromosome_embedding_dim = chromosome_embedding_dim
        self.chromosomes_min_occurences = chromosomes_min_occurences
        
        self.status_arr = status_arr
        self.status_columns = status_columns
        self.patient_ids = self.status_arr[:,0]
        self.patient_num = self.patient_ids.shape[0]
        
        self.clinical_arr = clinical_arr
        self.clinical_columns = clinical_columns
        self.clinical_id = self.clinical_arr[:,0]
        
        self.molecular_arr = molecular_arr
        self.molecular_columns = molecular_columns
        self.molecular_id = self.molecular_arr[:,0]
        self.molecular_split = np.split(self.molecular_arr, np.unique(self.molecular_arr[:,0], return_index=True)[1][1:])
                            
        self.__get_unique_chromosomes()
        self.__get_chromosome_model()
        self.__get_chromosome_embeddings()
        self.__get_chromosome_map()
        self.__get_unique_genes()
        self.__get_gene_model()
        self.__get_gene_embeddings()
        self.__get_gene_map()
        
        if effects_survival_map == None:
            self.__get_effects_to_survival_map()
        else:
            self.effects_survival_map = effects_survival_map
        
                    
                
    def __call__(self):
        print("Dataset containing training data.")
        
    def cyto_patient_risk(self, cyto):
        cyto=cyto.strip().upper()
        
        favorable_markers = ["T(8;21)", "INV(16)", "T(15;17)"]
        adverse_markers = ["MONOSOMY 7", "-7", "COMPLEX", "MONOSOMY 5", "-5", "DEL(5Q)", "DEL(7Q)"]
        
        if cyto in ["46,XX", "46,XY"]:
            return 0
        
        for marker in favorable_markers:
            if marker in cyto:
                return 0
            
        for marker in adverse_markers:
            if marker in cyto:
                return 2
            
        return 1
    
    def clinical_transformer(self):
        clinical_transformed = np.zeros((self.patient_num, len(self.clinical_features)+3))
        arr_pos = 0
        cyto_pos = np.where(self.clinical_columns == "CYTOGENETICS")[0][0]
        
        for feature in self.clinical_features:
            curr_column = int(np.where(self.clinical_columns==feature)[0][0])
            curr_feature_arr = self.clinical_arr[:, curr_column]
            #curr_feature_median = curr_feature_df.median()
            #curr_feature_df = curr_feature_df.fillna(value = curr_feature_median)
            clinical_transformed[:, arr_pos] = curr_feature_arr
            arr_pos += 1
        
        for i in range(self.patient_num):
            curr_cyto = self.clinical_arr[i, cyto_pos]
            if type(curr_cyto) != str: 
                continue
            curr_risk = self.cyto_patient_risk(curr_cyto)
            if curr_risk == 0:
                clinical_transformed[i, -3] = 1
            if curr_risk == 1:
                clinical_transformed[i, -2] = 1
            if curr_risk == 2:
                clinical_transformed[i, -1] = 1
            
        clinical_transformed = pd.DataFrame(clinical_transformed, index=np.arange(self.patient_ids.shape[0]), columns=np.append(self.clinical_features, ["CYTO_LOW","CYTO_MED","CYTO_HIGH"]))
        
        return clinical_transformed
    
    def start_end_transformer(self):
        start_end_transformed = np.zeros((self.patient_num, 3))
        
        start_pos = np.where(self.molecular_columns == "START")[0][0]
        end_pos = np.where(self.molecular_columns == "END")[0][0]
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_split[i]
            if curr_molecular[0][0] != curr_patient_id or len(set(curr_molecular[:,0]))>1:
                print("-"*50, "ERROR", "-"*50)
            curr_start = curr_molecular[:, start_pos]
            curr_end = curr_molecular[:, end_pos]
            curr_len = curr_end-curr_start
            #if len(curr_len)==0 or (len(curr_len)==1 and np.isnan(curr_len[0])):
            #    continue
            start_end_transformed[i][0] = np.mean(curr_len)
            start_end_transformed[i][1] = np.std(curr_len)
            start_end_transformed[i][2] = np.max(curr_len)
        
        start_end_transformed = pd.DataFrame(start_end_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["LEN_MEAN", "LEN_STD", "LEN_MAX"])
        
        return start_end_transformed
    
    def __get_unique_chromosomes(self) -> None:
        #self.unique_chromosomes = list(set(self.molecular_arr[:,np.where(self.molecular_columns=="CHR")[0][0]]))
        self.unique_chromosomes = ['11', '5', '3', '4', '2', '22', '17', 'X', '12', '9', '7', '1', '8', '16', '20', '21', '19', '15', '13', '6', '18', '14', '10', np.nan]
        
    def __get_chromosome_model(self) -> None:
        #number of different genes, the +1 comes from cases where the gene is not known
        num_chromosomes = len(self.unique_chromosomes) + 1
        
        #get model
        self.chromosome_model = EmbeddingModel(num_chromosomes, self.chromosome_embedding_dim)
        
    def __get_chromosome_embeddings(self) -> None:        
        #get the arguments of the model for the gene embeddings
        with torch.no_grad():
            self.chromosome_embeddings = self.chromosome_model.embedding.weight.cpu().numpy()
    
    def __get_chromosome_map(self) -> None:        
        #create map that maps the unique genes to the integers 1 to len(unique_genes)
        self.chromosome_to_idx = {chromosome: idx for idx, chromosome in enumerate(self.unique_chromosomes, start=1)}
        
        #if no gene is specified in data_molecular it is mapped to 0
        self.chromosome_to_idx["UNKNOWN"] = 0
        
    def get_chromosome_embedding(self, patient_chromosomes):
        #if patient_chromosomes is empty return array containing zeros, else return the mean of the embeddings
        if len(patient_chromosomes)==0:
            return np.zeros(self.chromosome_embedding_dim)
            
        #get the indices corresponding to the chromosomes
        indices = [self.chromosome_to_idx.get(g, 0) for g in patient_chromosomes]
        
        #get the embeddings of the indices
        vectors = [self.chromosome_embeddings[idx] for idx in indices]
        
        return np.mean(vectors, axis=0)
    
    def chromosomes_transformer(self):
        chromosomes_transformed = np.zeros((self.patient_num, self.chromosome_embedding_dim))
        
        chromosomes_pos = np.where(self.molecular_columns == "CHR")[0][0]
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_split[i]
            if curr_molecular[0][0] != curr_patient_id or len(set(curr_molecular[:,0]))>1:
                print("-"*50, "ERROR", "-"*50)
            curr_chromosomes = curr_molecular[:, chromosomes_pos]
            #if len(curr_chromosomes)==1 and type(curr_chromosomes[0])!=str: continue
            chromosomes_transformed[i] = self.get_chromosome_embedding(curr_chromosomes)
                        
        chromosomes_transformed = pd.DataFrame(chromosomes_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["CHR:"+str(i) for i in range(self.chromosome_embedding_dim)])
        
        return chromosomes_transformed
        
    def __get_unique_genes(self) -> None:
        temp_df = pd.DataFrame(self.molecular_arr[:,np.where(self.molecular_columns=="GENE")[0][0]], index=np.arange(self.molecular_arr.shape[0]), columns=["GENE"]).dropna()
        self.unique_genes = sorted(temp_df["GENE"].unique())
        
    def __get_gene_model(self) -> None:
        #number of different genes, the +1 comes from cases where the gene is not known
        num_genes = len(self.unique_genes) + 1
        
        #get model
        self.gene_model = EmbeddingModel(num_genes, self.gene_embedding_dim)
        
    def __get_gene_embeddings(self) -> None:        
        #get the arguments of the model for the gene embeddings
        with torch.no_grad():
            self.gene_embeddings = self.gene_model.embedding.weight.cpu().numpy()
            
    def __get_gene_map(self) -> None:        
        #create map that maps the unique genes to the integers 1 to len(unique_genes)
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.unique_genes, start=1)}
        
        #if no gene is specified in data_molecular it is mapped to 0
        self.gene_to_idx["UNKNOWN"] = 0
        
    def get_gene_embedding(self, patient_genes):
        #if patient_genes is empty return array containing zeros, else return the mean of the embeddings
        if len(patient_genes)==0:
            return np.zeros(self.gene_embedding_dim)
            
        #get the indices corresponding to the genes
        indices = [self.gene_to_idx.get(g, 0) for g in patient_genes]
        
        #get the embeddings of the indices
        vectors = [self.gene_embeddings[idx] for idx in indices]
        
        return np.mean(vectors, axis=0)
    
    def genes_transformer(self):
        genes_transformed = np.zeros((self.patient_num, self.gene_embedding_dim))
        
        genes_pos = np.where(self.molecular_columns == "GENE")[0][0]
        
        for i in range(self.patient_num):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_split[i]
            if curr_molecular[0][0] != curr_patient_id or len(set(curr_molecular[:,0]))>1:
                print("-"*50, "ERROR", "-"*50)
            curr_genes = curr_molecular[:, genes_pos]
            #if len(curr_genes)==1 and type(curr_genes[0])!=str: continue
            genes_transformed[i] = self.get_gene_embedding(curr_genes)
            
        genes_transformed = pd.DataFrame(genes_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["GENE:"+str(i) for i in range(self.gene_embedding_dim)])
        
        return genes_transformed
    
    def __get_effects_to_survival_map(self) -> None:
        molecular_df = pd.DataFrame(self.molecular_arr, index=np.arange(self.molecular_arr.shape[0]), columns=self.molecular_columns)
        status_df = pd.DataFrame(self.status_arr, index=np.arange(self.patient_num), columns=self.status_columns)
        comb_df = molecular_df.merge(status_df, on='ID', how='left')
        
        #use lifelines.KaplanMeierFitter to account for the censoring when evaluating the map
        kmf = KaplanMeierFitter()
        self.effects_survival_map = {}
        
        #use groupby to create a tuple for each unique effect containing (effect name, data_molecular[data_molecular['EFFECT] == efect name]) which get passed on to loop as effect and subset
        for effect, subset in comb_df.groupby("EFFECT"):
            #use Kaplan-Meier estimate using subset["OS_YEARS"] as durations and subset["OS_STATUS"] for the censoring
            kmf.fit(durations=list(subset["OS_YEARS"]), event_observed=list(subset["OS_STATUS"]))
            #check if kmf can calculate the median (it cannot if subset has to few values from patients with status=1, this is the case for the the effect "inframe_variant")
            if kmf.median_survival_time_ == np.inf: #kmf cannot compute the median
                #set the value in the map for the current effect to the max survival time in subset
                self.effects_survival_map[effect] = np.float64(np.max(subset["OS_YEARS"]))
            else:
                #set the value in the map for the current effect to the median survival time obtained using the Kaplan Meier estimate
                self.effects_survival_map[effect] = kmf.median_survival_time_
    
    def effects_transformer(self):
        effects_transformed = np.zeros((self.patient_num, 2))
        
        effects_pos = np.where(self.molecular_columns == "EFFECT")[0][0]
        vaf_pos = np.where(self.molecular_columns == "VAF")[0][0]
        
        #if a patient has no somatic mutations set the corresponding element in mol_effect to the median survival time of all patients
        global_median_survival = np.median(self.status_arr[:, np.where(self.status_columns=="OS_YEARS")[0][0]]) #np.median(data_st.loc[data_st["OS_STATUS"]== 0]["OS_YEARS"])
        
        for i in range(self.patient_num):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_split[i]
            curr_effects = curr_molecular[:,effects_pos]
            
            if curr_molecular[0][0] != curr_patient_id or len(set(curr_molecular[:,0]))>1:
                print("-"*50, "ERROR", "-"*50)
                
            if len(curr_effects)>0 and not (len(curr_effects)==1 and str(curr_effects[0])=='nan'):
                #get tuple (length=len(curr_effects)) with the survival times associated with the elements of curr_effects
                curr_survival = np.array(itemgetter(*curr_effects)(self.effects_survival_map))
                #get the values of vaf for the current patient and use these as weights for the corresponding effects
                curr_vaf = curr_molecular[:, vaf_pos]
                #normalize the weights
                curr_vaf = (curr_vaf/np.sum(curr_vaf)) if np.sum(curr_vaf>0) else np.array([1])
                #set mol_effect[i] to the average of curr_survival
                effects_transformed[i] = [np.average(curr_survival*curr_vaf), len(curr_effects)]
            else:
                #if the current patient has no effects set mol_effect[i] to the median survival time of all patients
                effects_transformed[i] = [global_median_survival, 0]
                
        effects_transformed = pd.DataFrame(effects_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["EFFECT_TRANSFORMED", "NUMBER_OF_MUTATIONS"])
        
        return effects_transformed
    
    def __classify_mutation(self, ref, alt):
        if len(ref) == len(alt):
            return 0
        elif len(ref) > len(alt):
            return 1
        else:
            return 2
    
    def mutation_transformer(self):
        mutation_transformed = np.zeros((self.patient_num, 3))
        
        ref_pos = np.where(self.molecular_columns == "REF")[0][0]
        alt_pos = np.where(self.molecular_columns == "ALT")[0][0]
        
        for i in range(self.patient_num):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_split[i]
            if curr_molecular[0][0] != curr_patient_id or len(set(curr_molecular[:,0]))>1:
                print("-"*50, "ERROR", "-"*50)
                
            curr_ref = curr_molecular[:, ref_pos]
            curr_alt = curr_molecular[:, alt_pos]
            
            curr_mut = np.zeros(3)
            for k in range(curr_ref.shape[0]):
                if str(curr_ref[k]) == 'nan' or str(curr_alt[k]) == 'nan':
                    continue
                curr_mut_class = self.__classify_mutation(curr_ref[k], curr_alt[k])
                curr_mut[curr_mut_class] += 1
                
            mutation_transformed[i] = curr_mut
            
        mutation_transformed = pd.DataFrame(mutation_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["MUT_SUB", "MUT_DEL", "MUT_INS"])
        
        return mutation_transformed
        
    def molecular_transformer(self):
        chromosomes_transformed = self.chromosomes_transformer().reset_index(drop=True)
        effects_transformed = self.effects_transformer().reset_index(drop=True)
        genes_transformed = self.genes_transformer().reset_index(drop=True)
        start_end_transformed = self.start_end_transformer()
        #mutation_transformed = self.mutation_transformer()
        
        #molecular_transformed = pd.concat([effects_transformed, chromosomes_transformed, genes_transformed, start_end_transformed, mutation_transformed], axis=1)
        molecular_transformed = pd.concat([effects_transformed, chromosomes_transformed, genes_transformed, start_end_transformed], axis=1)
        #molecular_transformed = pd.concat([effects_transformed, chromosomes_transformed, genes_transformed], axis=1)
        #molecular_transformed = pd.concat([effects_transformed, genes_transformed], axis=1)
        
        return molecular_transformed
    
    def train_data_transformed(self):
        clinical_transformed = self.clinical_transformer().reset_index(drop=True)
        #clinical_transformed = clinical_transformed.drop(["CYTO_LOW","CYTO_MED","CYTO_HIGH"], axis=1)
        molecular_transformed = self.molecular_transformer().reset_index(drop=True)
        molecular_transformed = molecular_transformed.drop(["LEN_MEAN", "LEN_STD", "LEN_MAX"], axis=1)
        #molecular_transformed = molecular_transformed.drop(["MUT_SUB", "MUT_DEL", "MUT_INS"], axis=1)
        #molecular_transformed = molecular_transformed.drop(["GENE:"+str(i) for i in range(50)], axis=1)
        #molecular_transformed = molecular_transformed.drop(["CHR:"+str(i) for i in range(10)], axis=1)
        
        
        X = pd.concat([clinical_transformed, molecular_transformed], axis=1)
        
        y = np.array([(bool(val[1]), float(val[0])) for val in self.status_arr[:,1:]], dtype = [('status', bool), ('time', float)])
                
        return X, y

# %%

set_random_seed(1)

d = Dataset(dat.status_arr, dat.clinical_arr, dat.molecular_arr, effects_survival_map=dat.effects_survival_map)

X, y = d.train_data_transformed()
X = X.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

#alpha seems to be best below 3000, ties probably better with "breslow", changing tol and n_iter currently does not matter as the loss converges very early (before 20 iterations)

cox = CoxPHSurvivalAnalysis(alpha=0)
cox.fit(X_train, y_train)

preds = cox.predict(X_val)
ind = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
indp = concordance_index_ipcw(y_train, y_val, preds)[0]
print(ind, indp)

# %%

scores = fit_and_score_features(np.array(X_train), np.array(y_train))

vals = pd.Series(scores, index=X.columns).sort_values(ascending=False)

# %% with LEN, MUT info

inda = np.zeros(1000)
indpa = np.zeros(1000)

for i in tqdm(range(1000)):
    set_random_seed(i)
    
    d = Dataset(dat.status_arr, dat.clinical_arr, dat.molecular_arr, effects_survival_map=dat.effects_survival_map)
    
    X, y = d.train_data_transformed()
    X = X.fillna(0)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    
    #alpha seems to be best below 3000, ties probably better with "breslow", changing tol and n_iter currently does not matter as the loss converges very early (before 20 iterations)
    
    cox = CoxPHSurvivalAnalysis()
    cox.fit(X_train, y_train)
    
    preds = cox.predict(X_val)
    try:
        inda[i] = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
        indpa[i] = concordance_index_ipcw(y_train, y_val, preds)[0]
    except:
        inda[i]=0.71
        indpa[i]=0.67
    
# %% without LEN, MUT info

indb = np.zeros(1000)
indpb = np.zeros(1000)

for i in tqdm(range(1000)):
    set_random_seed(i)
    
    d = Dataset(dat.status_arr, dat.clinical_arr, dat.molecular_arr, effects_survival_map=dat.effects_survival_map)
    
    X, y = d.train_data_transformed()
    X = X.fillna(0)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    
    #alpha seems to be best below 3000, ties probably better with "breslow", changing tol and n_iter currently does not matter as the loss converges very early (before 20 iterations)
    
    cox = CoxPHSurvivalAnalysis()
    cox.fit(X_train, y_train)
    
    preds = cox.predict(X_val)
    try:
        indb[i] = concordance_index_censored(y_val['status'], y_val['time'], preds)[0]
        indpb[i] = concordance_index_ipcw(y_train, y_val, preds)[0]
    except:
        indb[i]=0.71
        indpb[i]=0.67

# %%

x_axis = np.arange(1000)

plt.figure()

plt.scatter(x_axis, inda, c='blue')
plt.scatter(x_axis, indb, c='red')

plt.figure()

plt.scatter(x_axis, indpa, c='blue')
plt.scatter(x_axis, indpb, s=30, c='red')

print(np.sum(indb>inda), np.sum(indpb>indpa))

# %%

set_random_seed(1)  

d = Dataset(dat.status_arr, dat.clinical_arr, dat.molecular_arr)
e = d.effects_transformer()

# %%

a = d.chromosomes_transformer().reset_index(drop=True)

# %%

t = d.molecular_arr[:,6]

# %%

set_random_seed(1)    

tst = Dataset(file_status, file_clinical, file_molecular, chromosome_embedding_dim=5)

Xc, yc = tst.train_data_transformed()
Xc = Xc.fillna(0)

X_trainc, X_valc, y_trainc, y_valc = train_test_split(Xc, yc, test_size=0.3, random_state=1)

#alpha seems to be best below 3000, ties probably better with "breslow", changing tol and n_iter currently does not matter as the loss converges very early (before 20 iterations)

coxc = CoxPHSurvivalAnalysis(alpha=0, tol=1e-9, n_iter=100, verbose=1)
coxc.fit(X_trainc, y_trainc)

predsc = coxc.predict(X_valc)
indc = concordance_index_censored(y_valc['status'], y_valc['time'], predsc)[0]
indpc = concordance_index_ipcw(y_trainc, y_valc, predsc)[0]
print(indc, indpc)























































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
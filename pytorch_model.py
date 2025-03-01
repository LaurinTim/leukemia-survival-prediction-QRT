#file containing different commands used in the analysis

import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm
import random
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import torch
import copy
from torch.utils.data import DataLoader, Dataset

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from time import time as ttime

#Directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
file_status = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
file_clinical = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
file_molecular = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training

# %% Check if CUDA cores are available for training, if yes set the batch size to 128, otherwise 32

# Constant parameters accross models
# Detect available accelerator; Downgrade batch size if only CPU available
if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
    print("CUDA-enabled GPU/TPU is available.")
    BATCH_SIZE = 256  # batch size for training
    torch.set_default_device('cuda')
    device = 'cuda'
    #torch.set_default_device('cpu')
    #device = 'cpu'
else:
    print("No CUDA-enabled GPU found, using CPU.")
    BATCH_SIZE = 32  # batch size for training
    device = torch.device('cpu')
    
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

def plot_losses(train_losses, test_losses, title: str = "Cox", norm = True, ran = None) -> None:
    if ran == None:
        x = np.linspace(1, len(train_losses), len(train_losses))
    
    else:
        train_losses = train_losses[ran[0]:ran[1]]
        test_losses = test_losses[ran[0]:ran[1]]
        x = np.linspace(max(ran[0],0), min(ran[1], len(train_losses)+max(ran[0],0)), len(train_losses))
    
    if norm == True:
        train_losses = torch.stack(train_losses) / train_losses[0]
        test_losses = torch.stack(test_losses) / test_losses[0]

    plt.scatter(x, train_losses.cpu(), label="training", color = 'C0')
    plt.scatter(x, test_losses.cpu(), label="test", color = 'C1', s = 20)
    plt.xlabel("Epochs")
    if norm == True: plt.ylabel("Normalized loss")
    else: plt.ylabel("Loss")
    plt.title(title)
    plt.yscale("log")
    plt.legend()
    plt.show()

def compare_models(model1, model2):
    '''

    Parameters
    ----------
    model1 : pytorch model
        First model.
    model2 : pytorch model
        Second model, parameters need to have the same shape as for first model.

    Returns
    -------
    bool
        Returns True if the parameters of both models are identical, otherwise False.

    '''
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def adjust_learning_rate(optimizer, last_losses, epoch, initial_lr=1e-4, decay_factor=0.5, epoch_interval=10):
    """Reduce LR every decay_epoch epochs by decay_factor."""
    if initial_lr <= 1e-5:
        return
    else:
        if epoch % epoch_interval == 0 and epoch != 0:
            lr = initial_lr * decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(1e-5, lr)
        #if len(last_losses)==5 and last_losses[1]>last_losses[0] and last_losses[2]>last_losses[1] and last_losses[3]>last_losses[2] and last_losses[4]>last_losses[3]:
        #    lr = optimizer.param_groups[0]['lr']*0.5
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = max(1e-5, lr)

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
        self.molecular_void_ids = []
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
                self.molecular_void_ids += [curr_id]
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

clinical_columns=np.array(['ID', 'CENTER', 'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'CYTOGENETICS'])
clinical_features=np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES'])
clinical_indices = np.array([2, 6, 7, 3, 4, 5])

# %% Transformers for DatasetGen
    
class TransStatus(object):
    def __call__(self, sample):
        res = torch.tensor(np.array([sample[2], sample[1]]))
        return res
        
class TransClinical(object):
    def __call__(self, sample):
        clinical_features=np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES'])
        
        res = torch.zeros(clinical_features.shape[0]+3)
        
        for i in range(clinical_features.shape[0]):
            res[i] = float(sample[clinical_indices[i]])
            
        cyto = sample[8]
        
        if str(cyto) != 'nan':
            cyto_risk = self.cyto_patient_risk(cyto)
            res[-3+cyto_risk] = 1
            
        return res.float()
    
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
    
# %%

ct = np.array(['P116577', 'KI', 1.0, 2.0, 0.5, 0.1, 9.9, 31.0, '46,xy,+8[10]/46,xy[10]'])
tct = TransClinical().__call__(ct)

# %%

class TransMolecular(object):
    def __call__(self, sample, 
                 global_median_survival, effects_survival_map, 
                 chromosomes_map, chromosome_embeddings, 
                 genes_map, gene_embeddings, 
                 gene_embedding_dim=50, chromosome_embedding_dim=10):
        self.gene_embedding_dim = gene_embedding_dim
        self.chromosome_embedding_dim = chromosome_embedding_dim
        
        if effects_survival_map == None:
            self.__get_effects_to_survival_map()
        else:
            self.effects_survival_map = effects_survival_map
        
        res = torch.zeros(2+self.chromosome_embedding_dim+self.gene_embedding_dim+3)
        
        effects = sample[:,8]
        vaf = np.array([float(val) for val in sample[:,9]])
        weights = (vaf/np.sum(vaf)) if np.sum(vaf)>0 else np.array([1])
        survival = np.array(itemgetter(*effects)(effects_survival_map))
        effects_transformed = np.array([np.average(survival*weights), len(effects)])
        
        chroms = sample[:,1]
        if len(chroms)==0:
            chromosomes_transformed = np.zeros(chromosome_embedding_dim)
        else:
            indices = np.array([chromosomes_map.get(g, 0) for g in chroms])
            chromosomes_transformed = np.mean(np.array([chromosome_embeddings[i] for i in indices]), axis=0)
            
        genes = sample[:,6]
        if len(genes)==0:
            genes_transformed = np.zeros(gene_embedding_dim)
        else:
            indices = np.array([genes_map.get(g, 0) for g in genes])
            genes_transformed = np.mean(np.array([gene_embeddings[i] for i in indices]), axis=0)
            
        start = np.array([float(val) for val in sample[:,2]])
        end = np.array([float(val) for val in sample[:,3]])
        length = end-start
        length_mean = np.mean(length)
        length_std = np.std(length)
        length_max = np.max(length)
        start_end_transformed = np.array([length_mean, length_std, length_max])
        
        res = torch.tensor(np.concatenate((effects_transformed, chromosomes_transformed, genes_transformed, start_end_transformed))).float()
        
        return res

# %%
    
mt = np.array([['P116360', '21', 44514777.0, 44514777.0, 'T', 'G', 'U2AF1', 'p.Q157P', 'non_synonymous_codon', 0.097, 2160.0], 
               ['P116360', '4', 106190900.0, 106190900.0, 'C', 'T', 'TET2', 'p.T1393I', 'non_synonymous_codon', 0.021, 837.0], 
               ['P116360', '20', 31022837.0, 31022838.0, 'AT', 'A', 'ASXL1', 'p.L775fs*1', 'frameshift_variant', 0.2194, 1603.0], 
               ['P116360', '9', 5073770.0, 5073770.0, 'G', 'T', 'JAK2', 'p.V617F', 'non_synonymous_codon', 0.04, 1430.0], 
               ['P116360', '4', 106158509.0, 106158510.0, 'G', 'GT', 'TET2', 'p.K1138fs*1', 'frameshift_variant', 0.1051, 1858.0]])

tmt = TransMolecular().__call__(mt, 
                                a.global_median_survival, a.effects_survival_map, 
                                a.chromosome_to_idx, a.chromosome_embeddings, 
                                a.gene_to_idx, a.gene_embeddings, 
                                a.gene_embedding_dim, a.chromosome_embedding_dim)

# %% Generate a custom dataset

class DatasetGen(Dataset):
    def __init__(self, status_arr, clinical_arr, molecular_arr, effects_survival_map, molecular_void_ids,
                 status_columns=np.array(['ID', 'OS_YEARS', 'OS_STATUS']), 
                 clinical_columns=np.array(['ID', 'CENTER', 'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'CYTOGENETICS']), 
                 molecular_columns=np.array(['ID', 'CHR', 'START', 'END', 'REF', 'ALT', 'GENE', 'PROTEIN_CHANGE', 'EFFECT', 'VAF', 'DEPTH']),
                 clinical_features=np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']), 
                 gene_embedding_dim=50, chromosome_embedding_dim=10, chromosomes_min_occurences=5,
                 status_transformer=None, clinical_transformer=None, molecular_transformer=None):
        
        self.clinical_features = clinical_features
        self.gene_embedding_dim = gene_embedding_dim
        self.chromosome_embedding_dim = chromosome_embedding_dim
        self.chromosomes_min_occurences = chromosomes_min_occurences
        self.effects_survival_map = effects_survival_map
        self.molecular_void_ids = molecular_void_ids
        
        self.status_transform = status_transformer
        self.clinical_transform = clinical_transformer
        self.molecular_transform = molecular_transformer
        
        self.status_arr = status_arr
        self.status_columns = status_columns
        self.patient_ids = self.status_arr[:,0]
        self.patient_num = self.patient_ids.shape[0]
        self.global_median_survival = np.median(self.status_arr[:,1])
        
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
        
        self.X = np.zeros((self.patient_num, len(clinical_features)+3+2+self.chromosome_embedding_dim+self.gene_embedding_dim+3))
        self.y = np.zeros((self.patient_num, 2))
        
    def __len__(self):
        return self.patient_num
    
    def __getitem__(self, idx):
        status_item = self.status_arr[idx]
        patient_id = status_item[0]
        clinical_item = self.clinical_arr[idx]
        molecular_item = self.molecular_split[idx]
        
        if self.status_transform:
            status_item = self.status_transform(status_item)
        
        if self.clinical_transform and self.molecular_transform:
            clinical_item = self.clinical_transform(clinical_item)
            if patient_id in self.molecular_void_ids:
                molecular_item = torch.zeros(2+self.chromosome_embedding_dim+self.gene_embedding_dim+3)
            else:
                molecular_item = self.molecular_transform(molecular_item, 
                             self.global_median_survival, self.effects_survival_map, 
                             self.chromosome_to_idx, self.chromosome_embeddings, 
                             self.gene_to_idx, self.gene_embeddings, 
                             self.gene_embedding_dim, self.chromosome_embedding_dim)
                                        
            info = torch.cat((clinical_item, molecular_item))
            
        self.X[idx] = info.cpu().numpy()
        self.y[idx] = status_item.cpu().numpy()
            
        return info, (bool(status_item[0]), status_item[1])
    
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

# %%

set_random_seed(1)

a = DatasetGen(dat.status_arr, dat.clinical_arr, dat.molecular_arr, dat.effects_survival_map, dat.molecular_void_ids, 
               status_transformer = TransStatus(), clinical_transformer = TransClinical(), molecular_transformer = TransMolecular())

train_data, val_data, test_data = torch.utils.data.random_split(a, [0.6, 0.2, 0.2], generator=torch.Generator(device=device))

dataloader_train = DataLoader(train_data, batch_size = BATCH_SIZE)
dataloader_val = DataLoader(val_data, batch_size = BATCH_SIZE)
dataloader_test = DataLoader(test_data, batch_size = BATCH_SIZE)

train_time = torch.tensor([val[1][1] for val in train_data]).float()
val_time = torch.tensor([val[1][1] for val in val_data])
test_time = torch.tensor([val[1][1] for val in test_data])

bad_seed = bool(train_time.max() < val_time.max() or train_time.max() < test_time.max())
if bad_seed:
    print('-'*100)
    print('ERROR: THIS IS A BAD SEED, THE MAX TIME IN TRAIN IS SMALLER THAN IN VAL OR TEST, PLEASE SPLIT DATA AGAIN')
    print(train_time.max(), val_time.max(), test_time.max())
    print('-'*100)

# %%

train_x = torch.tensor([val[0].cpu().numpy() for val in train_data])
train_event = torch.tensor([val[1][0] for val in train_data]).bool()

val_x = torch.tensor([val[0].cpu().numpy() for val in val_data])
val_event = torch.tensor([val[1][0] for val in val_data])

test_x = torch.tensor([val[0].cpu().numpy() for val in test_data])
test_event = torch.tensor([val[1][0] for val in test_data])

# %% Sanity check

x, (event, time) = next(iter(dataloader_train))
num_features = x.size(1)

print(f"x (shape)    = {x.shape}")
print(f"num_features = {num_features}")
print(f"event        = {event.shape}")
print(f"time         = {time.shape}")
print(f"batch size   = {BATCH_SIZE}")

# %%

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(74, 150),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(150, 150),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(150, 150),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(150, 150),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(150, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        return logit

cox_model = NeuralNetwork()

# %% Define learning rate, epoch and optimizer

LEARNING_RATE = 1e-3
EPOCHS = 50
optimizer = torch.optim.AdamW(cox_model.parameters(), lr=LEARNING_RATE, weight_decay=0.7)
con = ConcordanceIndex()

# %%

def train_loop(dataloader, model, optimizer):
    model.train()
    
    curr_loss = torch.tensor(0.0)
    weight = 0
    for i, batch in enumerate(dataloader):
        x, (event, time) = batch
        optimizer.zero_grad()
        log_hz = model(x)
                
        loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        loss.backward()
        optimizer.step()
        curr_loss += loss.detach() * len(x)/BATCH_SIZE
            
        weight += len(x)/BATCH_SIZE
        
    optimizer.zero_grad()
    
    curr_loss /= weight
    return curr_loss

def val_loop(model, epoch):    
    optimizer.zero_grad()
    model.eval()
    
    curr_con_ind = torch.tensor(0.0)
    curr_con_ind_ipcw = torch.tensor(0.0)
    curr_loss = torch.tensor(0.0)
    
    x, event, time = val_x, val_event, val_time
    
    with torch.no_grad():
        pred = model(x)
        
        loss = neg_partial_log_likelihood(pred, event, time, reduction="mean")
        curr_loss += loss.detach()
        
        con_ind = con(pred, event, time)
        curr_con_ind = con_ind
        
        try:
            weight_ipcw = get_ipcw(train_event, train_time, time)
        except:
            curr_con_ind_ipcw = 0
            print('ERROR FOR IPCW WEIGHTS IN TEST LOOP')
        else:
            con_ind_ipcw = con(pred.float(), event, time.float(), weight = weight_ipcw)
            curr_con_ind_ipcw = con_ind_ipcw

    return curr_loss, curr_con_ind, curr_con_ind_ipcw

# %% Iterate through Train and Test loops

set_random_seed(1)

train_losses = []
val_losses = []

val_con_inds = []
val_con_ind_ipcws = []

for t in tqdm(range(EPOCHS)):
    curr_train_loss = train_loop(dataloader_train, cox_model, optimizer)
    curr_val_loss, curr_val_con_ind, curr_val_con_ind_ipcw = val_loop(cox_model, t)
    
    train_losses.append(curr_train_loss)
    val_losses.append(curr_val_loss)
    
    adjust_learning_rate(optimizer, val_losses[-5:], t, initial_lr=optimizer.param_groups[0]['lr'], decay_factor=0.5, epoch_interval=10)
    
    val_con_inds.append(curr_val_con_ind)
    val_con_ind_ipcws.append(curr_val_con_ind_ipcw)
    
    
    if t % (EPOCHS // 4) == 0:
        print(f"\nEpoch {t+1}\n-------------------------------")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:0.3e}")
        print(f"Training loss: {curr_train_loss:0.6f}, Validation loss: {curr_val_loss:0.6f}")
        print(f"Concordance Index validation:  {curr_val_con_ind:0.3f}, IPCW Concordance Index validation:  {curr_val_con_ind_ipcw:0.3f}")
print('\n' + '-'*50)
print("Done!")
print('-'*50)

# %% Plot the training and test losses

title = "COX"
ns = 0
ne = 3000

plt.figure()
plot_losses(train_losses, val_losses, title, norm = True, ran = [ns, ne])

plt.figure()
plot_losses(val_con_ind_ipcws, val_con_ind_ipcws, title, norm = True, ran = [ns, ne])





































































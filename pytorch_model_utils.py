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


clinical_columns=np.array(['ID', 'CENTER', 'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'CYTOGENETICS'])
clinical_features=np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES'])
clinical_indices = np.array([2, 6, 7, 3, 4, 5])

def get_device():
    '''

    Returns
    -------
    BATCH_SIZE : int
        Batch size used for the data.
    device : str
        Either "cuda" or "cpu", what device pytorch should sue by default.

    '''
    #Check if CUDA cores are available for training, if yes set the batch size to 128, otherwise 32
    
    # Constant parameters accross models
    # Detect available accelerator; Downgrade batch size if only CPU available
    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available. Batch size set to 256.")
        BATCH_SIZE = 256  # batch size for training
        device = 'cuda'
        #torch.set_default_device('cpu')
        #device = 'cpu'
    else:
        print("No CUDA-enabled GPU found, using CPU. Batch size set to 32.")
        BATCH_SIZE = 32  # batch size for training
        device = 'cpu'
        
    return BATCH_SIZE, device

def set_random_seed(random_seed) -> None:
    '''

    Parameters
    ----------
    random_seed : int
        Set the random seeds of NumPy, pytorch to random_seed.

    '''
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

class EmbeddingModel(torch.nn.Module):
    '''
    Embedding module for the gene and chromosome embeddings
    '''
    def __init__(self, num, embedding_dim):
        '''

        Parameters
        ----------
        num : int
            Number of elements that should be embedded.
        embedding_dim : int
            Length of the embeddings.

        '''
        super(EmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(num, embedding_dim)
        
    def forward(self, idx):
        return self.embedding(idx)

def plot_losses(train_losses, test_losses, title: str = "Cox", norm = True, ran = None) -> None:
    '''

    Parameters
    ----------
    train_losses : torch.tensor
        Tensor containing the losses of each epoch during training.
    test_losses : np.array/torch.tensor
        Tensor containing the losses of each epoch during training.
    title : str, optional
        Title of the generated plot. The default is "Cox".
    norm : bool, optional
        Whether or not train_losses and test_losses should be normalized. If 
        set to true, each elements of the two tensors gets divided by its 
        first element. The default is True.
    ran : list of length 2, optional
        Interval of epochs for which the plot is generated. By default all 
        epochs are used. If ran is not None, only the the elements in 
        train_losses and test_losses between ran[0] and ran[1] get plotted. 
        ran[0] must be smaller than the number of epochs but ran[1] can also 
        be larger, in which case the plot is generated until the last epoch.
        The default is None.

    '''
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

def adjust_learning_rate(optimizer, last_losses, epoch, initial_lr=1e-4, decay_factor=0.5, epoch_interval=10, min_lr=1e-5):
    """Reduce LR every decay_epoch epochs by decay_factor."""
    if initial_lr <= min_lr:
        return
    else:
        if epoch % epoch_interval == 0 and epoch != 0:
            lr = initial_lr * decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(min_lr, lr)
                
class DataPrep():
    def __init__(self, status_file, clinical_file, molecular_file):
        '''

        Parameters
        ----------
        status_file : str
            Path to the file with the target information.
        clinical_file : str
            Path to the file with the clinical information.
        molecular_file : str
            Path to the file with the molecular information.

        '''
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
        '''

        Parameters
        ----------
        df : DataFrame
            Dataframe containing either molecular or clinical information.

        Returns
        -------
        DataFrame
            Input dataframe but only with the rows that correspond to a 
            patient with valid status (no element of status is nan).

        '''
        return_df = df[df.loc[:,"ID"].isin(self.patient_ids)]
        return return_df.reset_index(drop=True)
    
    def __fillna_df(self, df, columns):
        '''

        Parameters
        ----------
        df : DataFrame
            Dataframe containing either molecular or clinical information.
        columns : list of strings
            List containing the columns names in which any nan values should be filled.

        Returns
        -------
        return_df : DataFrame
            Input dataframe where the nan values of the columns in columns are
            set to 0.

        '''
        #return_df = df.fillna({col: df[col].median() for col in df.select_dtypes(include=['float']).columns})
        return_df = df.fillna({col: 0 for col in df.select_dtypes(include=['float']).columns if col not in ["CHR"]})
        return return_df
    
    def __molecular_id_fill(self) -> None:
        '''

        Adds rows in self.molecular_df for each patient with a valid status
        that is not yet in the dataframe. The row is filled with the 
        patient ID at the first position and nan otherwise.

        '''
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
        '''
        
        Create self.molecular_split, which is a list of arrays. The i-th 
        element of this list contains the information in self.molecular_arr 
        corresponding to the patient with i-th ID in self.patient_ids and has 
        shape (# of somatic mutations, 11).

        '''
        self.molecular_split = np.split(self.molecular_arr, np.unique(self.molecular_arr[:,0], return_index=True)[1][1:])
        
    def __get_effects_to_survival_map(self) -> None:
        '''

        Get the expected lifetime associated with each effect of the somatic 
        mutations. This is evaluated using a Kaplan-Meier estimate to get the 
        median survival team of each effect. These values are put in a 
        dictionary mapping each effect to the associated expected median 
        survival time.

        '''
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
                
class TransStatus(object):
    def __call__(self, sample):
        '''

        Parameters
        ----------
        sample : ndarray
            Array containing the status of one patient.

        Returns
        -------
        res : torch.tensor
            Tensor of length 2 containing the event indicator for the patient 
            at position 0 and the patients survival time at position 1.

        '''
        res = torch.tensor(np.array([sample[2], sample[1]]))
        return res
        
class TransClinical(object):
    def __call__(self, sample):
        '''

        Parameters
        ----------
        sample : ndarray
            Array containing the clinical information of one patient.

        Returns
        -------
        torch.tensor
            Transformed sample. 
            The information in clinical_features 
            of the sample is kept ant put at the beginning of the transformed 
            array.
            Following this are 3 elements where at most one of them 
            is set to 1 and the other two to 0. These contain information 
            about the cytogenetics of the patient, the element set to 1 shows 
            the associated risk with the cytogenetics where the first element 
            corresponds to low rist, the second to medium risk and the third 
            to high risk. If all 3 are 0 then there is no information about 
            the cytogenetics of the patient available.

        '''
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
        '''

        Parameters
        ----------
        cyto : str
            Cytogenetics of the current patient.

        Returns
        -------
        int
            Risk associated with the Cytogenetics.

        '''
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

class TransMolecular(object):
    def __call__(self, sample, 
                 global_median_survival, effects_survival_map, 
                 chromosomes_map, chromosome_embeddings, 
                 genes_map, gene_embeddings, 
                 chromosome_embedding_dim=10, gene_embedding_dim=50):
        '''
        
        Parameters
        ----------
        sample : list
            List with the information of the different somatic mutations of 
            the patient as elements.
        global_median_survival : float
            Median survival time of all patients.
        effects_survival_map : dict
            Dictionary mapping the effects to a expected survival time.
        chromosomes_map : dict
            Dictionary mapping the chromosomes to integers.
        chromosome_embeddings : ndarray
            Array containing the embeddings of each chromosome type.
        genes_map : dict
            Dictionary mapping the genes to integers.
        gene_embeddings : ndarray
            Array containing the embeddings of each gene type.
        chromosome_embedding_dim : int, optional
            Length of each chromosome embedding. The default is 10.
        gene_embedding_dim : int, optional
            Length of gene embedding. The default is 50.
            
         Returns
         -------
         torch.tensor
             Transformed sample.
             In the first 2 elements the expected median survival time we get 
             from the effects and the number of somatic mutations are put.
             In the next chromosome_embedding_dim elements the mean of the 
             sum of the embeddings of the chromosomes associated with the 
             somatic mutations are put.
             In the next gene_embedding_dim elements the mean of the sum of 
             the embeddings of the genes associated with the somatic mutations
             are put.
             To get the last 3 elements the length of the change on a 
             chromosome for each somatic mutation is calculated. The average, 
             maximum and standard devation of these lengths is then put into
             the last 3 elements.

        '''
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

class TorchStandardScaler:
    def fit(self, tns):
        '''

        Parameters
        ----------
        tns : torch.tensor
            Tensor to scale.

        Returns
        -------
        Sets self.mean and self.std to the mean and standard deviation of tns.

        '''
        self.mean = tns.mean(0, keepdim=True)
        self.std = tns.std(0, unbiased=False, keepdim=True)
    def transform(self, tns):
        '''

        Parameters
        ----------
        tns : torch.tensor
            Tensor to scale.

        Returns
        -------
        tns : torch.tensor
            Scaled tensor.

        '''
        tns -= self.mean
        tns /= (self.std + 1e-7)
        return tns
    def fit_transform(self, tns):
        '''

        Parameters
        ----------
        tns : tns
            Tensor to scale.

        Returns
        -------
        torch.tensor
            Scaled tensor.

        '''
        self.fit(tns)
        return self.transform(tns)

class DatasetGen(Dataset):
    def __init__(self, status_arr, clinical_arr, molecular_arr, effects_survival_map, molecular_void_ids,
                 status_columns=np.array(['ID', 'OS_YEARS', 'OS_STATUS']), 
                 clinical_columns=np.array(['ID', 'CENTER', 'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'CYTOGENETICS']), 
                 molecular_columns=np.array(['ID', 'CHR', 'START', 'END', 'REF', 'ALT', 'GENE', 'PROTEIN_CHANGE', 'EFFECT', 'VAF', 'DEPTH']),
                 clinical_features=np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']), 
                 chromosome_embedding_dim=10, gene_embedding_dim=50, chromosomes_min_occurences=5,
                 status_transformer=None, clinical_transformer=None, molecular_transformer=None):
        '''

        Parameters
        ----------
        status_arr : ndarray
            Array containing the status information.
        clinical_arr : ndarray
            Array containing the clinical information.
        molecular_arr : ndarray
            Array containing the molecular information.
        effects_survival_map : dict
            Dictionary mapping the effects to their median survival time.
        molecular_void_ids : ndarray
            IDs of patients with no somatic mutations.
        status_columns : list, optional
            List containing the names of the columns of status_arr. 
            The default is np.array(['ID', 'OS_YEARS', 'OS_STATUS']).
        clinical_columns : TYPE, optional
            List containing the names of the columns of clinical_arr. 
            The default is np.array(['ID', 'CENTER', 'BM_BLAST', 'WBC', 
                                     'ANC', 'MONOCYTES', 'HB', 'PLT', 
                                     'CYTOGENETICS']).
        molecular_columns : TYPE, optional
            List containing the names of the columns of molecular_arr. 
            The default is np.array(['ID', 'CHR', 'START', 'END', 'REF', 
                                     'ALT', 'GENE', 'PROTEIN_CHANGE', 
                                     'EFFECT', 'VAF', 'DEPTH']).
        clinical_features : list, optional
            List containing the columns of clinical_arr that we want to 
            include in the training of the model as they are. 
            The default is np.array(['BM_BLAST', 'HB', 'PLT', 'WBC', 
                                     'ANC', 'MONOCYTES']).
        chromosome_embedding_dim : TYPE, optional
            Dimension of the chromosome embeddings. The default is 10.
        chromosomes_min_occurences : int, optional
            CURRENTLY NOT USED. The default is 5.
        gene_embedding_dim : int, optional
            Dimension of the gene embeddings. The default is 50.
        status_transformer : class, optional
            Class with the transformer for the elements of the status 
            information. If set to None, no transformation is applied, 
            otherwise the given class is used. The default is None.
        clinical_transformer : class, optional
            Class with the transformer for the elements of the clinical 
            information. If set to None, no transformation is applied, 
            otherwise the given class is used. The default is None.
        molecular_transformer : class, optional
            Class with the transformer for the elements of the molecular 
            information. If set to None, no transformation is applied, 
            otherwise the given class is used. The default is None.

        '''
        
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
        
        self.X = torch.zeros((self.patient_num, len(clinical_features)+3+2+self.chromosome_embedding_dim+self.gene_embedding_dim+3))
        self.y = torch.zeros((self.patient_num, 2))
        
        self.__getData__()
        
    def __len__(self):
        '''

        Returns
        -------
        int
            Number of patients.

        '''
        return self.patient_num
    
    def __getData__(self):
        '''

        Returns
        -------
        Fill self.X with the transformed clinical and molecular info.
        Fill self.y with the transformed status.

        '''
        for idx in range(self.patient_num):
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
                                 self.chromosome_embedding_dim, self.gene_embedding_dim)
                                            
                info = torch.cat((clinical_item, molecular_item))
                
            self.X[idx] = info
            self.y[idx] = status_item
            
        scaler = TorchStandardScaler()
        self.X = scaler.fit_transform(self.X)
         
    def __getitem__(self, idx):
        '''

        Parameters
        ----------
        idx : int
            Get transformed information of the patient associated with the ID
            self.patient_ids[idx].

        Returns
        -------
        torch.tensor, (bool, float)
            Returns a tensor containing the transformed clinical and molecular 
            info for the patient at position 0. At position 1 a tuple 
            containing the event indicator (bool) and the survival time (float) 
            of the patient.

        '''
        
        data_item = self.X[idx]
        status_item = (bool(self.y[idx,0]), self.y[idx,1])
        
        return data_item, status_item
        
        '''
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
                             self.chromosome_embedding_dim, self.gene_embedding_dim)
                                        
            info = torch.cat((clinical_item, molecular_item))
            
        self.X[idx] = info.cpu().numpy()
        self.y[idx] = status_item.cpu().numpy()
            
        return info, (bool(status_item[0]), status_item[1])'''
    
    def __get_unique_chromosomes(self) -> None:
        '''

        Set self.unique_chromosomes to a list containing the different 
        chromosomes in the molecular information.

        '''
        #self.unique_chromosomes = list(set(self.molecular_arr[:,np.where(self.molecular_columns=="CHR")[0][0]]))
        self.unique_chromosomes = ['11', '5', '3', '4', '2', '22', '17', 'X', '12', '9', '7', '1', '8', '16', '20', '21', '19', '15', '13', '6', '18', '14', '10', np.nan]
        
    def __get_chromosome_model(self) -> None:
        '''
        
        Set self.chromosome_model to the model used to get the chromosome
        embeddings.

        '''
        #number of different genes, the +1 comes from cases where the gene is not known
        num_chromosomes = len(self.unique_chromosomes) + 1
        
        #get model
        self.chromosome_model = EmbeddingModel(num_chromosomes, self.chromosome_embedding_dim)
        
    def __get_chromosome_embeddings(self) -> None:
        '''
        
        Set self.chromosome_embeddings to the parameters of 
        self.chromosome_model. This gives a list with the same length as 
        self.unique_chromosomes and each element the embedding associated with 
        one chromosome type.

        '''
        #get the arguments of the model for the gene embeddings
        with torch.no_grad():
            self.chromosome_embeddings = self.chromosome_model.embedding.weight.cpu().numpy()
    
    def __get_chromosome_map(self) -> None:
        '''
        
        Set self.chromosome_to_idx to a dictionary mapping the different 
        chromosome types in self.unique_chromosomes to integers.

        '''
        #create map that maps the unique genes to the integers 1 to len(unique_genes)
        self.chromosome_to_idx = {chromosome: idx for idx, chromosome in enumerate(self.unique_chromosomes, start=1)}
        
        #if no gene is specified in data_molecular it is mapped to 0
        self.chromosome_to_idx["UNKNOWN"] = 0
        
    def __get_unique_genes(self) -> None:
        '''

        Set self.unique_genes to a list containing the different genes in the 
        molecular information.

        '''
        temp_df = pd.DataFrame(self.molecular_arr[:,np.where(self.molecular_columns=="GENE")[0][0]], index=np.arange(self.molecular_arr.shape[0]), columns=["GENE"]).dropna()
        self.unique_genes = sorted(temp_df["GENE"].unique())
        
    def __get_gene_model(self) -> None:
        '''
        
        Set self.gene_model to the model used to get the gene embeddings.

        '''
        #number of different genes, the +1 comes from cases where the gene is not known
        num_genes = len(self.unique_genes) + 1
        
        #get model
        self.gene_model = EmbeddingModel(num_genes, self.gene_embedding_dim)
        
    def __get_gene_embeddings(self) -> None:   
        '''
        
        Set self.gene_embeddings to the parameters of self.gene_model. This 
        gives a list with the same length as self.unique_genes and each 
        element the embedding associated with one gene type.

        '''
        #get the arguments of the model for the gene embeddings
        with torch.no_grad():
            self.gene_embeddings = self.gene_model.embedding.weight.cpu().numpy()
            
    def __get_gene_map(self) -> None:
        '''
        
        Set self.gene_to_idx to a dictionary mapping the different gene types 
        in self.unique_chromosomes to integers.

        '''
        #create map that maps the unique genes to the integers 1 to len(unique_genes)
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.unique_genes, start=1)}
        
        #if no gene is specified in data_molecular it is mapped to 0
        self.gene_to_idx["UNKNOWN"] = 0

def get_x_and_event(tns):
    x = torch.tensor([val[0].cpu().numpy() for val in tns])
    event = train_event = torch.tensor([val[1][0] for val in tns]).bool()
    return x, event


                
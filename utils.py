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

#Directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

# %%

def test_results(model, parameters_file, data, features, model_name, return_df=False):
    '''
    
    A csv file for the submission is created at C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\submission_files\\{model_name}.csv.
    The first row is the header with the column names "ID" and "risk_score". "ID" is the patient ID and "risk_score" the output of the model.

    Parameters
    ----------
    model : torch.nn.Sequential
        Pytorch model that has to match the info in parameter_file.
    parameters_file : str
        Path to file with the parameters for model.
    data : pandas.DataFrame
        Pandas DataFrame containing patient ID in column 'ID' and the features 
        of the model, taken from the test files.
    features: list of strings
        Names of the columns containing the features of the model in the order 
        that the model expects.
    model_name: str
        Name of the model, this is also the name the the created csv file will 
        have.
    return_df: bool, optional
        If set to True (default) then the DataFrame that is created gets 
        returned but not saved.

    Returns
    -------
    None.

    '''
    data = data.fillna(value = 0)
    
    model.load_state_dict(torch.load(parameters_file))
    model.eval()
    
    ID = np.array(data.ID)
    
    model_input = torch.tensor(np.array(data[features])).float()
    
    pred = model(model_input)
    pred = [float(val[0]) for val in pred]
    
    df = pd.DataFrame([ID, pred], index = ["ID", "risk_score"]).transpose()
    
    if return_df == True:
        return df
    
    else:
        data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
        df.to_csv(data_dir + "\\submission_files\\" + model_name + ".csv", index = False)

#Code from https://github.com/Novartis/torchsurv/blob/main/docs/notebooks/helpers_introduction.py, creates a scatter plot with normalized losses for the training and test data

def plot_losses(train_losses, test_losses, title: str="Cox", norm=True, ran=None) -> None:
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

    plt.scatter(x, train_losses, label="training", color = 'C0')
    plt.scatter(x, test_losses, label="test", color = 'C1', s = 20)
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
        Second model, parameters need to have the same shape as for model1.

    Returns
    -------
    bool
        Returns True if the parameters of model1 and model2 are identical, 
        otherwise False.

    '''
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def status_to_StructuredArray(data):
    '''

    Parameters
    ----------
    data : pytorch.tensor
        Tensor where each element has at posision 1 information about the 
        patient status (0 (alive) or 1 (dead)) in type bool and at position 2 
        the number of years the patient has lived since the diagnosis if 
        status==0 and number of years the patient died after the diagnosis if
        status == 1.

    Returns
    -------
    arr : structured numpy.ndarray [('status', '?'), ('years', '<f8')]
        Structured array with the status in bool at position 1 of each element 
        and a number of years in float at position 2.

    '''
    arr = np.array([(bool(val[0]), float(val[1])) for val in data], dtype = [('status', bool), ('years', float)])
    
    return arr

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
    
def __get_unique_genes(data_file_molecular=data_dir+'\\X_train\\molecular_train.csv', data_file_status=data_dir+'\\target_train.csv'):
    '''

    Parameters
    ----------
    data_file_molecular : Path, optional
        Path to the file containing the molecular data of the patients. The 
        default is data_dir+'\\X_train\\molecular_train.csv'.
    data_file_status : Path, optional
        Path to the file containing the status data of the patients. The 
        default is data_dir+'\\target_train.csv'.

    Returns
    -------
    unique_genes : list
        List containing the unique genes in molecular data with valid status.

    '''
    #dataframe containing all genes in the training molecular data
    data_molecular = pd.read_csv(data_file_molecular)[['ID', 'GENE']]
    
    #dataframe containing all patient IDs we want to consider in the analysis since we cannot use patients where the status or survival time is undefined
    data_status = pd.read_csv(data_file_status).dropna(subset=['OS_YEARS', 'OS_STATUS'])['ID']
    
    #only keep genes from patients that have a valid status and survival time
    data_molecular = data_molecular.loc[data_molecular['ID'].isin(data_status)]
    
    #list containing the unique genes in data_molecular
    unique_genes = sorted(data_molecular['GENE'].unique())
    
    return unique_genes

def __get_gene_model(embedding_dim=50, data_file_molecular=data_dir+'\\X_train\\molecular_train.csv', data_file_status=data_dir+'\\target_train.csv'):
    '''

    Parameters
    ----------
    embedding_dim : int, optional
        Dimension of the embeddings. The default is 50.
    data_file_molecular : Path, optional
        Path to the file containing the molecular data of the patients. The 
        default is data_dir+'\\X_train\\molecular_train.csv'.
    data_file_status : Path, optional
        Path to the file containing the status data of the patients. The 
        default is data_dir+'\\target_train.csv'.

    Returns
    -------
    gene_model : torch.nn.Model
        Pytorch model containing the embeddings of the genes.

    '''
    #get the unique genes in data_file_molecular with valid patient status
    unique_genes = __get_unique_genes(data_file_molecular, data_file_status)
    
    #number of different genes, the +1 comes from cases where the gene is not known
    num_genes = len(unique_genes) + 1
    
    #get model
    gene_model = EmbeddingModel(num_genes, embedding_dim)
    
    return gene_model

def get_gene_embeddings(embedding_dim=50, data_file_molecular=data_dir+'\\X_train\\molecular_train.csv', data_file_status=data_dir+'\\target_train.csv'):
    '''

    Parameters
    ----------
    embedding_dim : Path, optional
        DESCRIPTION. The default is 50.
    data_file_molecular : Path, optional
        Path to the file containing the molecular data of the patients. The 
        default is data_dir+'\\X_train\\molecular_train.csv'.
    data_file_status : Path, optional
        Path to the file containing the status data of the patients. The 
        default is data_dir+'\\target_train.csv'.

    Returns
    -------
    gene_embeddings : numpy.ndarray
        Array of shape (number of unique genes + 1, embedding_dim) containing the 
        embeddings of the genes.

    '''
    #get the model for the gene embeddings
    gene_model = __get_gene_model(embedding_dim, data_file_molecular, data_file_status)
    
    #get the arguments of the model for the gene embeddings
    with torch.no_grad():
        gene_embeddings = gene_model.embedding.weight.cpu().numpy()
    
    return gene_embeddings
    
def get_gene_map(data_file_molecular=data_dir+'\\X_train\\molecular_train.csv', data_file_status=data_dir+'\\target_train.csv'):
    '''

    Parameters
    ----------
    data_file_molecular : Path, optional
        Path to the file containing the molecular data of the patients. The 
        default is data_dir+'\\X_train\\molecular_train.csv'.
    data_file_status : Path, optional
        Path to the file containing the status data of the patients. The 
        default is data_dir+'\\target_train.csv'.

    Returns
    -------
    gene_to_idx : dict
        Dictionary mapping the genes to integers from 0 to the number of 
        unique genes.

    '''
    #get the unique genes in data_file_molecular with valid patient status
    unique_genes = __get_unique_genes(data_file_molecular, data_file_status)
    
    #create map that maps the unique genes to the integers 1 to len(unique_genes)
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes, start=1)}
    
    #if no gene is specified in data_molecular it is mapped to 0
    gene_to_idx['UNKNOWN'] = 0
    
    return gene_to_idx

def get_gene_embedding(patient_genes, gene_embeddings, gene_to_idx, embedding_dim=50):
    '''

    Parameters
    ----------
    patient_genes : numpy.ndarray or list
        List containing the genes corresponding to one patient.
    gene_embeddings : numpy.ndarray
        Array with the gene embeddings.
    gene_to_ids: dict
        Dictionary mapping the genes to integers from 0 to the number of 
        unique genes.
    embedding_dim : int, optional
        Dimension of the embeddings. The default is 50.

    Returns
    -------
    numpy.ndarray
        If patient_genes is not empty return the mean of the embeddings 
        corresponding to the genes in patient_genes, else return an array 
        containing zeros.

    '''
    #get the indices corresponding to the genes
    indices = [gene_to_idx.get(g, 0) for g in patient_genes]
    
    #get the embeddings of the indices
    vectors = [gene_embeddings[idx] for idx in indices]
    
    #if patient_genes is not empty return the mean of the embeddings, else an array containing zeros
    if len(vectors)>0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)
    
###################################################################################################
#use target encoding for the somatic mutation effects using a Kaplan-Meier estimate
###################################################################################################

def effect_to_survival_map(data_file_molecular=data_dir+'\\X_train\\molecular_train.csv', data_file_status=data_dir+'\\target_train.csv'):
    '''

    Parameters
    ----------
    data_file_molecular : Path, optional
        Path to the file containing the molecular data of the patients. The 
        default is data_dir+'\\X_train\\molecular_train.csv'.
    data_file_status : Path, optional
        Path to the file containing the status data of the patients. The 
        default is data_dir+'\\target_train.csv'.

    Returns
    -------
    effect_survival_map : dict
        Dictionary mapping the effects of the somatic mutations to the median 
        lifetime of patients with the corresponding effect evaluated using
        the Kaplan-Meier estimate.

    '''
    #dataframe containing all genes in the training molecular data
    data_molecular = pd.read_csv(data_file_molecular)[['ID', 'EFFECT']]
    
    #dataframe containing all patient IDs we want to consider in the analysis since we cannot use patients where the status or survival time is undefined
    data_status = pd.read_csv(data_file_status).dropna(subset=['OS_YEARS', 'OS_STATUS'])
    
    #only keep genes from patients that have a valid status and survival time
    data_molecular = data_molecular.loc[data_molecular['ID'].isin(data_status['ID'])]
    
    #merge data_molecular and data_status on 'ID'
    data_molecular = data_molecular.merge(data_status, on='ID', how='left')
    
    #use lifelines.KaplanMeierFitter to account for the censoring when evaluating the map
    kmf = KaplanMeierFitter()
    effect_survival_map = {}
    
    #use groupby to create a tuple for each unique effect containing (effect name, data_molecular[data_molecular['EFFECT] == efect name]) which get passed on to loop as effect and subset
    for effect, subset in data_molecular.groupby("EFFECT"):
        #use Kaplan-Meier estimate using subset["OS_YEARS"] as durations and subset["OS_STATUS"] for the censoring
        kmf.fit(durations=subset["OS_YEARS"], event_observed=subset["OS_STATUS"])
        #check if kmf can calculate the median (it cannot if subset has to few values from patients with status=1, this is the case for the the effect "inframe_variant")
        if kmf.median_survival_time_ == np.inf: #kmf cannot compute the median
            #set the value in the map for the current effect to the max survival time in subset
            effect_survival_map[effect] = np.float64(np.max(subset["OS_YEARS"]))
        else:
            #set the value in the map for the current effect to the median survival time obtained using the Kaplan Meier estimate
            effect_survival_map[effect] = kmf.median_survival_time_
    
    return effect_survival_map

# %%

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
###################################################################################################
#Create a model for the gene embeddings and get a function mapping genes to their embeddings
###################################################################################################
#it would be faster to pass unique_genes directly to get_gene_model and get_gene_map since then it would not need to be calculated twice but for the amount of data we have it does not matter and creates less clutter this way

class EmbeddingModel(torch.nn.Module):
    '''
    
    Embedding module for the gene and chromosome embeddings
    
    '''
    def __init__(self, num, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(num, embedding_dim)
        
    def forward(self, idx):
        return self.embedding(idx)

# %%

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

file_status = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
file_clinical = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
file_molecular = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training

# %%

###################################################################################################
#Commands to get X_train, y_train from files
###################################################################################################
#maybe some columns should not be set to the median value if nan, eg. VAF, check this later
#in Dataset.get_encoded_chromosomes nan values are set to -1, check if it makes a difference to set this to 0

class Dataset():
    def __init__(self, status_file, clinical_file, molecular_file, clinical_file_test=None, molecular_file_test=None, 
                 clinical_features=['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES'], gene_embedding_dim=50, chromosomes_min_occurences=5, chromosome_embedding_dim=10):
        self.status_file = status_file
        self.clinical_file = clinical_file
        self.molecular_file = molecular_file
        
        self.clinical_features = clinical_features
        self.gene_embedding_dim = gene_embedding_dim
        self.chromosome_embedding_dim = chromosome_embedding_dim
        self.chromosomes_min_occurences = chromosomes_min_occurences
        
        self.status_df = pd.read_csv(status_file).dropna(subset=["OS_YEARS", "OS_STATUS"]).reset_index(drop=True)
        self.patient_ids = np.array(self.status_df.loc[:,"ID"])
        
        self.clinical_df = pd.read_csv(clinical_file)
        self.clinical_df = self.__valid_patients_df(self.clinical_df)
        self.clinical_id = np.array(self.clinical_df.loc[:,"ID"])
        self.clinical_df = self.__fillna_df(self.clinical_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
        
        self.molecular_df = pd.read_csv(molecular_file)
        self.molecular_df = self.__valid_patients_df(self.molecular_df)
        self.molecular_id = np.array(self.molecular_df.loc[:,"ID"])
        self.molecular_df = self.__fillna_df(self.molecular_df, ["START", "END", "VAF", "DEPTH"])
        self.vaf = np.array(self.molecular_df.loc[:,"VAF"])
        
        self.clinical_file_test = clinical_file_test
        self.molecular_file_test = molecular_file_test
        
        #self.__get_chromosome_encoder()
        self.__get_unique_chromosomes()
        self.__get_chromosome_model()
        self.__get_chromosome_embeddings()
        self.__get_chromosome_map()
        self.__get_unique_genes()
        self.__get_gene_model()
        self.__get_gene_embeddings()
        self.__get_gene_map()
        self.__get_effects_to_survival_map()
        
                
    def __call__(self):
        if self.clinical_file_test==None:
            print("Dataset containing training data.")
        else:
            print("Dataset containing training and test data.")
            
    def __valid_patients_df(self, df):
        return_df = df[df.loc[:,"ID"].isin(self.patient_ids)].reset_index(drop=True)
        return return_df
    
    def __fillna_df(self, df, columns):
        #return_df = df.fillna({col: df[col].median() for col in df.select_dtypes(include=['float']).columns})
        return_df = df.fillna({col: 0 for col in df.select_dtypes(include=['float']).columns if col not in ["CHR"]})
        return return_df
    
    def clinical_transformer(self):
        clinical_transformed = pd.DataFrame(0, index=np.arange(self.patient_ids.shape[0]), columns=self.clinical_features)
        
        for feature in self.clinical_features:
            curr_feature_df = self.clinical_df.loc[:,feature]
            #curr_feature_median = curr_feature_df.median()
            #curr_feature_df = curr_feature_df.fillna(value = curr_feature_median)
            clinical_transformed.loc[:,feature] = curr_feature_df
            
        #clinical_transformed.index = self.patient_ids
        
        return clinical_transformed
    
    def length_start_end_transformer(self):
        start_end_transformed = pd.DataFrame(0, index=np.arange(self.patient_ids.shape[0]))
        
        return start_end_transformed
    
    def __get_unique_chromosomes(self) -> None:
        #self.unique_chromosomes = list(self.molecular_df["CHR"].unique())
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
        chromosomes_transformed = np.zeros((self.patient_ids.shape[0], self.chromosome_embedding_dim))
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_df.loc[self.molecular_id==curr_patient_id]
            curr_chromosomes = np.array(curr_molecular.loc[:,"CHR"])
            chromosomes_transformed[i] = self.get_chromosome_embedding(curr_chromosomes)
            
        chromosomes_transformed = pd.DataFrame(chromosomes_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["CHR:"+str(i) for i in range(self.chromosome_embedding_dim)])
        #chromosomes_transformed = pd.DataFrame(chromosomes_transformed, index=self.patient_ids, columns=["CHR:"+str(i) for i in range(self.chromosome_embedding_dim)])
        
        return chromosomes_transformed
    
    def chromosomes_transformer_onehot(self):
        chrom_onehot = pd.get_dummies(self.molecular_df["CHR"], prefix="CHR")
        chromosomes_transformed = pd.DataFrame(0, index = np.arange(self.patient_ids.shape[0]), columns=["CHR:"+str(i) for i in np.arange(23)])
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_onehot = chrom_onehot.loc[self.molecular_id==curr_patient_id]
            curr_onehot_sum = np.array(np.sum(curr_onehot, axis=0))
            chromosomes_transformed.iloc[i] = curr_onehot_sum
            
        chromosomes_transformed_sum = np.sum(chromosomes_transformed, axis=0)
        sparse_features = chromosomes_transformed.columns[(chromosomes_transformed_sum < self.chromosomes_min_occurences)]
        chromosomes_transformed = chromosomes_transformed.drop(columns=sparse_features)
        
        return chromosomes_transformed
    
    def __get_unique_genes(self) -> None:
        self.unique_genes = sorted(self.molecular_df['GENE'].unique())
        
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
        genes_transformed = np.zeros((self.patient_ids.shape[0], self.gene_embedding_dim))
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_df.loc[self.molecular_id==curr_patient_id]
            curr_genes = np.array(curr_molecular.loc[:,"GENE"])
            genes_transformed[i] = self.get_gene_embedding(curr_genes)
            
        genes_transformed = pd.DataFrame(genes_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["GENE:"+str(i) for i in range(self.gene_embedding_dim)])
        #genes_transformed = pd.DataFrame(genes_transformed, index=self.patient_ids, columns=["GENE:"+str(i) for i in range(self.gene_embedding_dim)]).sort_index()
        
        return genes_transformed
    
    def get_gene_embedding1(self, patient_genes, weights):
        #if patient_genes is empty return array containing zeros, else return the mean of the embeddings
        if len(patient_genes)==0:
            return np.zeros(self.gene_embedding_dim)
            
        #get the indices corresponding to the genes
        indices = [self.gene_to_idx.get(g, 0) for g in patient_genes]
                
        #get the embeddings of the indices
        vectors = [self.gene_embeddings[idx]*weights[i] for idx,i in zip(indices,range(len(weights)))]
        
        return np.mean(vectors, axis=0)
    
    def genes_transformer1(self):
        genes_transformed = np.zeros((self.patient_ids.shape[0], self.gene_embedding_dim))
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_df.loc[self.molecular_id==curr_patient_id]
            curr_genes = np.array(curr_molecular.loc[:,"GENE"])
            #get the values of vaf for the current patient and use these as weights for the corresponding effects
            curr_vaf = self.vaf[self.molecular_id==curr_patient_id]
            #normalize the weights
            curr_vaf = (curr_vaf/np.sum(curr_vaf)) if np.sum(curr_vaf>0) else np.array([1])
            genes_transformed[i] = self.get_gene_embedding(curr_genes, curr_vaf)
            
        genes_transformed = pd.DataFrame(genes_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["GENE:"+str(i) for i in range(self.gene_embedding_dim)])
        
        return genes_transformed
    
    def __get_effects_to_survival_map(self) -> None:
        #merge data_molecular and data_status on 'ID'
        comb_df = self.molecular_df.merge(self.status_df, on='ID', how='left')
        
        #use lifelines.KaplanMeierFitter to account for the censoring when evaluating the map
        kmf = KaplanMeierFitter()
        self.effects_survival_map = {}
        
        #use groupby to create a tuple for each unique effect containing (effect name, data_molecular[data_molecular['EFFECT] == efect name]) which get passed on to loop as effect and subset
        for effect, subset in comb_df.groupby("EFFECT"):
            #use Kaplan-Meier estimate using subset["OS_YEARS"] as durations and subset["OS_STATUS"] for the censoring
            kmf.fit(durations=subset["OS_YEARS"], event_observed=subset["OS_STATUS"])
            #check if kmf can calculate the median (it cannot if subset has to few values from patients with status=1, this is the case for the the effect "inframe_variant")
            if kmf.median_survival_time_ == np.inf: #kmf cannot compute the median
                #set the value in the map for the current effect to the max survival time in subset
                self.effects_survival_map[effect] = np.float64(np.max(subset["OS_YEARS"]))
            else:
                #set the value in the map for the current effect to the median survival time obtained using the Kaplan Meier estimate
                self.effects_survival_map[effect] = kmf.median_survival_time_    
    
    def effects_transformer(self):
        effects_transformed = np.zeros((self.patient_ids.shape[0], 2))
        
        #if a patient has no somatic mutations set the corresponding element in mol_effect to the median survival time of all patients
        global_median_survival = np.median(self.status_df["OS_YEARS"]) #np.median(data_st.loc[data_st["OS_STATUS"]== 0]["OS_YEARS"])
        
        for i in range(self.patient_ids.shape[0]):
            curr_patient_id = self.patient_ids[i]
            curr_molecular = self.molecular_df.loc[self.molecular_id==curr_patient_id]
            curr_effects = np.array(curr_molecular.loc[:,"EFFECT"])
            if len(curr_effects)>0:
                #get tuple (length=len(curr_effects)) with the survival times associated with the elements of curr_effects
                curr_survival = np.array(itemgetter(*curr_effects)(self.effects_survival_map))
                #get the values of vaf for the current patient and use these as weights for the corresponding effects
                curr_vaf = self.vaf[self.molecular_id==curr_patient_id]
                #normalize the weights
                curr_vaf = (curr_vaf/np.sum(curr_vaf)) if np.sum(curr_vaf>0) else np.array([1])
                #set mol_effect[i] to the average of curr_survival
                effects_transformed[i] = [np.average(curr_survival*curr_vaf), len(curr_effects)]
            else:
                #if the current patient has no effects set mol_effect[i] to the median survival time of all patients
                effects_transformed[i] = [global_median_survival, 0]
                
        effects_transformed = pd.DataFrame(effects_transformed, index=np.arange(self.patient_ids.shape[0]), columns=["EFFECT_TRANSFORMED", "NUMBER_OF_MUTATIONS"])
        #effects_transformed = pd.DataFrame(effects_transformed, index=self.patient_ids, columns=["EFFECT_TRANSFORMED", "NUMBER_OF_MUTATIONS"]).sort_index()
        
        return effects_transformed
    
    def molecular_transformer(self):
        chromosomes_transformed = self.chromosomes_transformer().reset_index(drop=True)
        effects_transformed = self.effects_transformer().reset_index(drop=True)
        genes_transformed = self.genes_transformer().reset_index(drop=True)
        
        molecular_transformed = pd.concat([effects_transformed, chromosomes_transformed, genes_transformed], axis=1)
        #molecular_transformed = pd.concat([effects_transformed, genes_transformed], axis=1)
        
        return molecular_transformed
    
    def train_data_transformed(self):
        clinical_transformed = self.clinical_transformer().reset_index(drop=True)
        molecular_transformed = self.molecular_transformer().reset_index(drop=True)
        
        X = pd.concat([clinical_transformed, molecular_transformed], axis=1)
        
        y = np.array([(bool(val[1]), float(val[0])) for val in np.array(self.status_df[["OS_YEARS","OS_STATUS"]])], dtype = [('status', bool), ('time', float)])
                
        return X, y

# %%

set_random_seed(1)    

tstc = Dataset(file_status, file_clinical, file_molecular)

Xc, yc = tstc.train_data_transformed()
Xc = Xc.fillna(0)

X_trainc, X_valc, y_trainc, y_valc = train_test_split(Xc, yc, test_size=0.3, random_state=1)

#alpha seems to be best below 3000, ties probably better with "breslow", changing tol and n_iter currently does not matter as the loss converges very early (before 20 iterations)

coxc = CoxPHSurvivalAnalysis()
coxc.fit(X_trainc, y_trainc)

predsc = coxc.predict(X_valc)
indc = concordance_index_censored(y_valc['status'], y_valc['time'], predsc)[0]
indpc = concordance_index_ipcw(y_trainc, y_valc, predsc)[0]
print(indc, indpc)

# %%

set_random_seed(1)  

dd = Dataset(file_status, file_clinical, file_molecular)
ee = dd.effects_transformer().reset_index(drop=True)

# %%

aa = dd.chromosomes_transformer()
aa.index = dd.patient_ids
aa = aa.sort_index().reset_index(drop=True)

# %%

t = np.array(['11', '5', '3', '4', '2', '5'])
emm = dd.chromosome_embeddings
   
# %%

scores1 = fit_and_score_features(np.array(X_trainc), np.array(y_trainc))
vals1 = pd.Series(scores1, index=X_trainc.columns).sort_values(ascending=False)

# %%
from time import time

tst_df = tst.molecular_df
tst_id = tst.patient_ids
tst_idm = tst.molecular_id

# %%
start_time=time()

tst1 = np.zeros((tst_id.shape[0], 3))

for i in tqdm(range(tst_id.shape[0])):
#for i in range(1):
    curr_id = tst_id[i]
    curr_df = tst_df[tst_idm==curr_id]
    curr_start = curr_df.loc[:,"START"]
    curr_end = curr_df.loc[:,"END"]
    curr_len = curr_end-curr_start
    curr_start = np.average(curr_start)
    curr_end = np.average(curr_end)
    curr_len = np.average(curr_len)
    tst1[i] = np.array([curr_start, curr_end, curr_len])
    
tst1 = pd.DataFrame(tst1, index=np.arange(tst1.shape[0]), columns=["AV_START", "AV_END", "AV_LEN"])
    
end_time = time()
print(f"\nRuntime: {end_time-start_time:0.4f}")

# %%
start_time=time()

tst1 = pd.DataFrame(0, index=np.arange(tst_id.shape[0]), columns=["AV_START", "AV_END", "AV_LEN"])

for i in tqdm(range(tst_id.shape[0])):
#for i in range(1):
    curr_id = tst_id[i]
    curr_df = tst_df[tst_idm==curr_id]
    curr_start = curr_df.loc[:,"START"]
    curr_end = curr_df.loc[:,"END"]
    curr_len = curr_end-curr_start
    curr_start = np.average(curr_start)
    curr_end = np.average(curr_end)
    curr_len = np.average(curr_len)
    tst1.iloc[i] = np.array([curr_start, curr_end, curr_len])
        
end_time = time()
print(f"\nRuntime: {end_time-start_time:0.4f}")

# %%
tst_col = np.array(tst_df.columns)
tst_df = np.array(tst_df)
tst_id = np.array(tst_id)
tst_idm = np.array(tst_idm)

# %%

start_time=time()

tst1 = np.zeros((tst_id.shape[0], 3))

for i in tqdm(range(tst_id.shape[0])):
#for i in range(1):
    curr_id = tst_id[i]
    curr_df = tst_df[tst_idm==curr_id]
    curr_start = curr_df[:,2]
    curr_end = curr_df[:,3]
    curr_len = curr_end-curr_start
    curr_start = np.average(curr_start)
    curr_end = np.average(curr_end)
    curr_len = np.average(curr_len)
    tst1[i] = np.array([curr_start, curr_end, curr_len])
    
tst1 = pd.DataFrame(tst1, index=np.arange(tst1.shape[0]), columns=["AV_START", "AV_END", "AV_LEN"])
    
end_time = time()
print(f"\nRuntime: {end_time-start_time:0.4f}")

# %%

start_time=time()

start_col = np.array(tst_df[:,tst_col=="START"]).reshape(-1,1)
end_col = np.array(tst_df[:,tst_col=="END"]).reshape(-1,1)
tst_dat = np.concatenate((start_col, end_col), axis=1)

tst11 = np.zeros((tst_id.shape[0], 3))

for i in tqdm(range(tst_id.shape[0])):
#for i in range(1):
    curr_id = tst_id[i]
    curr_dat = tst_dat[tst_idm==curr_id]
    curr_len = curr_dat[:,1]-curr_dat[:,0]
    curr_start = np.average(curr_dat[:,0])
    curr_end = np.average(curr_dat[:,1])
    curr_len = np.average(curr_len)
    tst11[i] = np.array([curr_start, curr_end, curr_len])
    
tst11 = pd.DataFrame(tst1, index=np.arange(tst1.shape[0]), columns=["AV_START", "AV_END", "AV_LEN"])
    
end_time = time()
print(f"\nRuntime: {end_time-start_time:0.4f}")

# %%

tst_df = tst.molecular_df
tst_id = tst.patient_ids
tst_idm = tst.molecular_id

tst_col = np.array(tst_df.columns)
tst_df = np.array(tst_df)
tst_id = np.array(tst_id)
tst_idm = np.array(tst_idm)

tst_df = tst_df[tst_idm.argsort()]
tst_idm = tst_idm[tst_idm.argsort()]
tst_df = np.split(tst_df, np.unique(tst_idm, return_index=True)[1])
tst_umod = np.unique(tst_idm)

# %%

start_time=time()

tst1 = np.zeros((tst_id.shape[0], 3))

for i in tqdm(range(tst_id.shape[0])):
    curr_id = tst_id[i]
    if not curr_id in tst_umod:
        tst1[i] = np.zeros(3)
        continue
    
    curr_loc = int(np.where(tst_umod==curr_id)[0])
    curr_dat = tst_df[curr_loc]
    curr_len = curr_dat[:,3]-curr_dat[:,2]
    curr_start = np.average(curr_dat[:,2])
    curr_end = np.average(curr_dat[:,3])
    curr_len = np.average(curr_len)
    tst1[i] = np.array([curr_start, curr_end, curr_len])
    
tst1 = pd.DataFrame(tst1, index=np.arange(tst1.shape[0]), columns=["AV_START", "AV_END", "AV_LEN"])
    
end_time = time()
print(f"\nRuntime: {end_time-start_time:0.4f}")

# %%
occurences = np.arange(30)

res1 = np.zeros(len(occurences))
res2 = np.zeros(len(occurences))

#res11 = np.zeros(5)
#res22 = np.zeros(5)

for i in tqdm(range(len(occurences))):
#for i in tqdm(range(5)):
    set_random_seed(2)
    tst = Dataset(file_status, file_clinical, file_molecular, chromosome_embedding_dim=int(occurences[i]))
    Xc, yc = tst.train_data_transformed()
    X_trainc, X_valc, y_trainc, y_valc = train_test_split(Xc, yc, test_size=0.3, random_state=1)
    coxc = CoxPHSurvivalAnalysis()
    coxc.fit(X_trainc, y_trainc)

    predsc = coxc.predict(X_valc)
    indc = concordance_index_censored(y_valc['status'], y_valc['time'], predsc)[0]
    indpc = concordance_index_ipcw(y_trainc, y_valc, predsc)[0]
    res1[i] = indc
    res2[i] = indpc
    
# %%

x_axis = np.arange(len(occurences))
#x_axis = np.arange(5)
plt.scatter(x_axis, res1)
#plt.scatter(x_axis, res22)

# %%
print(qq)

occurences = np.arange(200)

res1 = np.zeros(len(occurences))
res2 = np.zeros(len(occurences))

#res11 = np.zeros(5)
#res22 = np.zeros(5)

for i in tqdm(range(len(occurences))):
#for i in tqdm(range(5)):
    set_random_seed(1)
    tst = Dataset(file_status, file_clinical, file_molecular, chromosomes_min_occurences=int(occurences[i]))
    Xc, yc = tst.train_data_transformed()
    X_trainc, X_valc, y_trainc, y_valc = train_test_split(Xc, yc, test_size=0.3, random_state=1)
    coxc = CoxPHSurvivalAnalysis()
    coxc.fit(X_trainc, y_trainc)

    predsc = coxc.predict(X_valc)
    indc = concordance_index_censored(y_valc['status'], y_valc['time'], predsc)[0]
    indpc = concordance_index_ipcw(y_trainc, y_valc, predsc)[0]
    res1[i] = indc
    res2[i] = indpc

# %%

x_axis = np.arange(len(occurences))
#x_axis = np.arange(5)
plt.scatter(x_axis, res1)
#plt.scatter(x_axis, res22)

# %%

set_random_seed(1)

tst = Dataset(file_status, file_clinical, file_molecular, chromosomes_min_occurences=5)

Xc, yc = tst.train_data_transformed()
Xc = Xc.fillna(0)

X_trainc, X_valc, y_trainc, y_valc = train_test_split(Xc, yc, test_size=0.3, random_state=1)

coxc = CoxPHSurvivalAnalysis()
coxc.fit(X_trainc, y_trainc)

predsc = coxc.predict(X_valc)
indc = concordance_index_censored(y_valc['status'], y_valc['time'], predsc)[0]
indpc = concordance_index_ipcw(y_trainc, y_valc, predsc)[0]
print(indc, indpc)

# %%

for i in range(200):
    if not np.array_equal(X[i], np.array(Xc)[i]):
        print(i)
    
# %%

print(np.array_equal(X[129],np.array(Xc)[129]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
###################################################################################################
#Create a model for the gene embeddings and get a function mapping genes to their embeddings
###################################################################################################
#it would be faster to pass unique_genes directly to get_gene_model and get_gene_map since then it would not need to be calculated twice but for the amount of data we have it does not matter and creates less clutter this way

class GeneEmbeddingModel(torch.nn.Module):
    '''
    
    Embedding module for the gene embeddings
    
    '''
    def __init__(self, num_genes, embedding_dim):
        super(GeneEmbeddingModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_genes, embedding_dim)
        
    def forward(self, gene_idx):
        return self.embedding(gene_idx)
    
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
    gene_model = GeneEmbeddingModel(num_genes, embedding_dim)
    
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
    def __init__(self, status_file, clinical_file, molecular_file, clinical_file_test=None, molecular_file_test=None, clinical_features=['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES'], gene_embedding_dim=50):
        self.status_file = status_file
        self.clinical_file = clinical_file
        self.molecular_file = molecular_file
        
        self.clinical_features = clinical_features
        self.gene_embedding_dim = gene_embedding_dim
        
        self.status_df = pd.read_csv(status_file).dropna(subset=["OS_YEARS", "OS_STATUS"])
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
        
        self.__get_chromosome_encoder()
        self.__get_unique_genes()
        self.__get_gene_model(self.gene_embedding_dim)
        
                
    def __call__(self):
        if self.clinical_file_test==None:
            print("Dataset containing training data.")
        else:
            print("Dataset containing training and test data.")
            
    def __valid_patients_df(self, df):
        return_df = df[df.loc[:,"ID"].isin(self.patient_ids)]
        return return_df
    
    def __fillna_df(self, df, columns):
        return_df = df.fillna({col: df[col].median() for col in df.select_dtypes(include=['float']).columns})
        return return_df
    
    def __get_chromosome_encoder(self):
        encoder = OrdinalEncoder()
        self.chromosome_encoder = encoder.fit(np.array(self.molecular_df.loc[:,"CHR"]).reshape(-1,1))
    
    def get_encoded_chromosomes(self, chromosomes):
        encoded_chromosomes = self.chromosome_encoder.fit(chromosomes)
        encoded_chromosomes = np.nan_to_num(encoded_chromosomes, nan=-1)
        return encoded_chromosomes
    
    def __get_unique_genes(self) -> None:
        self.unique_genes = sorted(self.molecular_df['GENE'].unique())
        
    def __get_gene_model(self, embedding_dim) -> None:
        #number of different genes, the +1 comes from cases where the gene is not known
        num_genes = len(self.unique_genes) + 1
        
        #get model
        self.gene_model = GeneEmbeddingModel(num_genes, embedding_dim)
                    
# %%

tst = Dataset(file_status, file_clinical, file_molecular)
        
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
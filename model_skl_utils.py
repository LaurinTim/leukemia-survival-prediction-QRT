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

#path to directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
#paths to files used for the training
status_file = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
clinical_file = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
molecular_file = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training
#path to the test files used for submissions
file_clinical_test = data_dir+'\\X_test\\clinical_test.csv' #contains clinical information of patients used for the submission
file_molecular_test = data_dir+'\\X_test\\molecular_test.csv' #contains molecular information of patients used for the submission

#features from the clinical data we want to include in the model
clinical_columns = ["ID", "CENTER", "BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTOGENETICS"]
clinical_features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']
clinical_indices = np.array([2, 6, 7, 3, 4, 5])

molecular_columns = ["ID", "CHR", "START", "END", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT", "VAF", "DEPTH"]

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

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
# %%

class DatasetPrep():
    def __init__(self, status_df, clinical_df, molecular_df, molecular_dummies_columns, effects_map):
        self.status_df = status_df.dropna(subset=["OS_YEARS", "OS_STATUS"])
        self.clinical_df = clinical_df
        self.molecular_df = molecular_df
        self.molecular_dummies_columns = molecular_dummies_columns
        
        self.status_df = self.status_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.patient_ids = np.array(self.status_df.loc[:,"ID"])
        self.patient_num = self.patient_ids.shape[0]
        self.status_columns = np.array(self.status_df.columns)
        self.status_arr = np.array(self.status_df)
        
        self.clinical_df = self.__valid_patients_df(self.clinical_df)
        self.clinical_df = self.__fillna_df(self.clinical_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
        self.clinical_df = self.clinical_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.clinical_columns = np.array(self.clinical_df.columns)
        self.clinical_arr = self.clinical_df.to_numpy(copy=True)
        self.clinical_columns = np.array(self.clinical_df.columns)
        self.clinical_arr = np.array(self.clinical_df)
        
        self.molecular_df = self.__valid_patients_df(self.molecular_df)
        self.molecular_df = self.__fillna_df(self.molecular_df, ["START", "END", "VAF", "DEPTH"])
        self.molecular_df = pd.get_dummies(molecular_df, columns = self.molecular_dummies_columns)
        
        self.molecular_df.insert(7, "EFFECT_MEDIAN_SURVIVAL", np.array(itemgetter(*np.array(self.molecular_df["EFFECT"]))(effects_map))) #* np.array(self.molecular_df["VAF"]))
        
        self.molecular_df = self.molecular_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.molecular_columns = np.array(self.molecular_df.columns)
        self.molecular_arr = self.molecular_df.to_numpy(copy=True)
        
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
                
    def __molecular_id_split(self) -> None:
        '''
        
        Create self.molecular_split, which is a list of arrays. The i-th 
        element of this list contains the information in self.molecular_arr 
        corresponding to the patient with i-th ID in self.patient_ids and has 
        shape (# of somatic mutations, 11).

        '''
        self.molecular_split = np.split(self.molecular_arr, np.unique(self.molecular_arr[:,0], return_index=True)[1][1:])
        
class Dataset():
    def __init__(self, status_df, clinical_df, molecular_df, min_occurences=5):
        self.status_df = status_df
        self.patient_ids = np.array(self.status_df.loc[:,"ID"])
        self.patient_num = self.patient_ids.shape[0]
        
        self.clinical_df = clinical_df
        self.clinical_ids = np.array(clinical_df.loc[:,"ID"])
        
        self.molecular_df = molecular_df
        self.molecular_ids = np.array(molecular_df.loc[:,"ID"])
        self.molecular_arr = np.array(self.molecular_df)
        #self.molecular_split = np.split(self.molecular_arr, np.unique(self.molecular_arr[:,0], return_index=True)[1][1:])
        
        self.X = np.zeros((self.patient_num, len(clinical_features)+3+self.molecular_df.shape[1]-5))
        self.y = np.zeros((self.patient_num, 2))
        
        self.__getData()
        
        X_sum = np.sum(self.X, axis=0)
        self.X = pd.DataFrame(self.X, index=np.arange(self.patient_num), columns=[clinical_features + ["CRYO_LOW", "CRYO_MEDIUM", "CRYO_HIGH"] + 
                                                                                  ["MUTATIONS_NUMBER", "MUTATION_LENGTH", "EFFECT_MEDIAN_SURVIVAL"] + list(self.molecular_df.columns)[8:]])
        
        sparse_features = self.X.columns[11:]
        sparse_features = sparse_features[X_sum[11:] < min_occurences]
        self.X = self.X.drop(columns=sparse_features)
        
    def __getData(self) -> None:
        for idx in range(self.patient_num):
            curr_X_item, curr_y_item = self.__getItem(idx)
            self.X[idx] = curr_X_item
            self.y[idx] = curr_y_item
            
        self.X = np.nan_to_num(self.X, nan=0)
        
    def __getItem(self, idx):
        curr_patient_id = self.patient_ids[idx]
        
        curr_status = np.array(self.status_df.iloc[idx])
        curr_clinical = np.array(self.clinical_df.loc[idx])
        
        if curr_status[0] != curr_patient_id:
            print("STATUS ID ERROR")
            
        if curr_clinical[0] != curr_patient_id:
            print("CLINICAL ID ERROR")
            
        status_item = (bool(curr_status[2]), curr_status[1])
        
        clinical_item = np.zeros(len(clinical_features)+3)
        clinical_item[0:len(clinical_features)] = curr_clinical[clinical_indices]
        curr_cyto_risk = self.cyto_patient_risk(curr_clinical[8])
        clinical_item[len(clinical_features)+curr_cyto_risk] = 1
        
        curr_molecular = np.array(self.molecular_df[self.molecular_df["ID"]==curr_patient_id])
        
        if len(curr_molecular)==0:
            molecular_item = np.zeros(self.molecular_df.shape[1]-5)
        
        else:
            molecular_item = np.zeros((len(curr_molecular), len(curr_molecular[0])-5))
            molecular_item[:,1] = curr_molecular[:,2]-curr_molecular[:,1]
            molecular_item[:,3:] = curr_molecular[:,8:]
            molecular_item = np.sum(molecular_item, axis=0)
            molecular_item[0] = len(curr_molecular)
            molecular_item[2] = np.sum(curr_molecular[:,7])
            #molecular_item[2:] = np.array([min(1,val) for val in molecular_item[2:]])
        
        item = np.append(clinical_item, molecular_item)
        
        return item, status_item
        
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
        if str(cyto)=="nan":
            return 1
            
        cyto=cyto.strip().upper()
        
        favorable_markers = ["T(8;21)", "INV(16)", "T(15;17)", "T(16;16)", "T((8;21)"]
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
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
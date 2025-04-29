#file containing different commands used in the analysis

import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from tqdm import tqdm
import random
from operator import itemgetter
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

from scipy.stats import logrank

#path to directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
#paths to files used for the training
status_file = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
clinical_file = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
molecular_file = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training
#path to the test files used for submissions
clinical_file_sub = data_dir+'\\X_test\\clinical_test.csv' #contains clinical information of patients used for the submission
molecular_file_sub = data_dir+'\\X_test\\molecular_test.csv' #contains molecular information of patients used for the submission

#features from the clinical data we want to include in the model
clinical_columns = ["ID", "CENTER", "BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTOGENETICS"]
clinical_features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']
clinical_indices = np.array([2, 6, 7, 3, 4, 5])

molecular_columns = ["ID", "CHR", "START", "END", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT", "VAF", "DEPTH"]

cyto_markers = ["46,XX", "46,XY", "T(8;21)", "INV(16)", "T(15;17)", "T(16;16)", "T((8;21)", "MONOSOMY 7", "-7", "COMPLEX", "MONOSOMY 5", "-5", "DEL(5Q)", "DEL(7Q)"]

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

def cox_score(x, y, model):
    model.fit(x, y)
    return model.score(x, y), concordance_index_ipcw(y, y, model.predict(x))[0]
    
def regression_score(df, y, model):    
    model.fit(df, duration_col="duration", event_col="status")
    pred = model.predict_partial_hazard(df)
    ci = concordance_index_censored(y["status"], y["time"], pred)[0]
    ci_ipcw = concordance_index_ipcw(y, y, pred)[0]
    
    return ci, ci_ipcw, model.summary.loc[list(df.columns)[2], "p"]
    #return model.summary.loc[list(df.columns)[2], "p"]
    
def skl_score(X, y, model):
    model.fit(X, y)
    pred = model.predict(X)
    ci = concordance_index_censored(y["status"], y["time"], pred)[0]
    ci_ipcw = concordance_index_ipcw(y, y, pred)[0]
    
    return ci, ci_ipcw

def lasso_score(X, y, model):
    model.fit(X, y)
    pred = model.predict(X)
    ci = concordance_index_censored(y["status"], y["time"], pred)[0]
    ci_ipcw = concordance_index_ipcw(y, y, pred)[0]
    
    return ci, ci_ipcw

def fit_and_score_features(X_df, y):
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
    X = np.array(X_df)
    n_features = X.shape[1]
    features = list(X_df.columns)
    scores = np.zeros((n_features, 9))
    
    df = X_df.copy()
    
    df.insert(0, "duration", list(y["time"]))
    df.insert(0, "status", list(y["status"]))
        
    rsf_model = RandomSurvivalForest(n_estimators=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=1)
    #rsf_model = RandomSurvivalForest(n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=3, n_jobs=-1, random_state=0)
    #rsf_model = RandomSurvivalForest()
    PH_model = CoxPHFitter(penalizer=0.0)
    skl_model = CoxPHSurvivalAnalysis(n_iter=100, tol=1e-9)
    lasso_model = CoxnetSurvivalAnalysis()
    
    for j in tqdm(range(n_features)):
        Xj = X[:, j : j + 1]
        scores[j,0], scores[j,1] = cox_score(Xj, y, rsf_model)
        scores[j,2], scores[j,3], scores[j,-1] = regression_score(df[["duration", "status", features[j]]], y, PH_model)
        scores[j,4], scores[j,5] = skl_score(Xj, y, skl_model)
        scores[j,6], scores[j,7] = skl_score(Xj, y, lasso_model)
        
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
        '''

        Parameters
        ----------
        status_df : DataFrame
            Dataframe containing the data with the target data, so patient 
            status (1 for dead and 0 for alive) and the survival time.
        clinical_df : DataFrame
            Dataframe containing the clinical training data.
        molecular_df : DataFrame
            Dataframe containing the molecular training data.
        molecular_dummies_columns : list
            Columns in molecular_df for which one hot encoding should be used.
        effects_map : dict
            Dictionary mapping the effects of the somatic mutations to the 
            median lifetime of patients with the corresponding effect.

        '''
        self.status_df = status_df.dropna(subset=["OS_YEARS", "OS_STATUS"])
        self.clinical_df = clinical_df
        self.molecular_df = molecular_df
        self.molecular_dummies_columns = molecular_dummies_columns
        self.effects_map = effects_map
        
        self.status_df = self.status_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.patient_ids = np.array(self.status_df.loc[:,"ID"])
        self.patient_num = self.patient_ids.shape[0]
        self.status_columns = np.array(self.status_df.columns)
        self.status_arr = np.array(self.status_df)
        
        self.clinical_df = self.__valid_patients_df(self.clinical_df)
        self.clinical_df_nan = self.clinical_df
        self.clinical_df = self.__fillna_df(self.clinical_df, self.clinical_df_nan, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
        self.clinical_df = self.clinical_df.sort_values(by=["ID"]).reset_index(drop=True)
        self.clinical_columns = np.array(self.clinical_df.columns)
        self.clinical_arr = self.clinical_df.to_numpy(copy=True)
        self.clinical_columns = np.array(self.clinical_df.columns)
        self.clinical_arr = np.array(self.clinical_df)
        
        self.molecular_df = self.__valid_patients_df(self.molecular_df)
        self.molecular_df_nan = self.molecular_df
        self.molecular_df = self.__fillna_df(self.molecular_df, self.molecular_df_nan, ["START", "END", "VAF", "DEPTH"])
        self.molecular_df = pd.get_dummies(self.molecular_df, columns = self.molecular_dummies_columns)
        
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
    
    def __fillna_df(self, df, df_nan, columns):
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
        return_df = df.fillna({col: df_nan[col].mean() for col in df.select_dtypes(include=['float']).columns if col in columns})
        #return_df = df.fillna({col: df[col].median() for col in df.select_dtypes(include=['float']).columns})
        #return_df = df.fillna({col: 0 for col in df.select_dtypes(include=['float']).columns if col not in ["CHR"]})
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
        
    def submission_data_prep(self):
        '''
        
        Prepares the data for the submission in clinical_file_sub for the 
        clinical data and molecular_file_sub for the molecular data the same 
        as the training data.

        Returns
        -------
        clinical_df_sub : DataFrame
            Dataframe containing the prepared clinical data for a submission.
        molecular_df_sub : DataFrame
            Dataframe containing the prepared molecular data for a submission.

        '''
        #get submission clinical data
        clinical_df_sub = pd.read_csv(clinical_file_sub)
        #clinical_df_sub = pd.read_csv(clinical_file)
        #fill the nan values in selected columns with the mean value from the same columns in self.clinical_df_nan
        clinical_df_sub = self.__fillna_df(clinical_df_sub, self.clinical_df_nan, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
        
        #sort the clinical data according to the patient ids
        clinical_sub_sort_index = [float(val[3:]) for val in list(clinical_df_sub.loc[:,"ID"])]
        clinical_df_sub.insert(9, "sort_index", clinical_sub_sort_index)
        clinical_df_sub = clinical_df_sub.sort_values(by=["sort_index"]).reset_index(drop=True)
        clinical_df_sub = clinical_df_sub.drop(columns=["sort_index"])
        
        #get all patient ids and the number of patients in the submission data
        patient_ids_sub = np.array(clinical_df_sub.loc[:,"ID"])
        patient_num_sub = patient_ids_sub.shape[0]
        
        #get the submission molecular data
        molecular_df_sub = pd.read_csv(molecular_file_sub)
        #molecular_df_sub = pd.read_csv(molecular_file)
        #use one hot encoding for the columns in self.molecular_dummies_columns
        molecular_df_sub = pd.get_dummies(molecular_df_sub, columns = self.molecular_dummies_columns)
        #fill the nan values in selected columns with the mean value from the same columns in self.molecular_df_nan
        molecular_df_sub = self.__fillna_df(molecular_df_sub, self.molecular_df_nan, ["START", "END", "VAF", "DEPTH"])
             
        #sort the molecular data according to the patient ids           
        molecular_sub_sort_index = [float(val[3:]) for val in list(molecular_df_sub.loc[:,"ID"])]
        molecular_df_sub.insert(11, "sort_index", molecular_sub_sort_index)
        molecular_df_sub = molecular_df_sub.sort_values(by=["sort_index"]).reset_index(drop=True)
        molecular_df_sub = molecular_df_sub.drop(columns=["sort_index"])
        
        #get the median survival of all patient in the training data
        global_median_survival = np.median(self.status_arr[:,1])
        #add a column to the molecular data with the expected median survival time obtained from self.effects_map. If the effect is not in self.effects_map use global_median_survival instead.
        molecular_df_sub.insert(7, "EFFECT_MEDIAN_SURVIVAL", np.array([self.effects_map.get(val) if val in self.effects_map.keys() else global_median_survival for val in molecular_df_sub["EFFECT"]]))
        
        #the submission molecular dataframe needs to have the same columns as the training molecular dataframe
        #add all columns from the training molecular df that are not already in the submission molecular df
        for curr_col in list(self.molecular_df.columns):
            if not curr_col in molecular_df_sub.columns:
                molecular_df_sub.insert(0, curr_col, np.zeros(len(molecular_df_sub)))
        
        #remove all columns from the submission molecular df that are not in the columns of the training molecular df
        for curr_col in list(molecular_df_sub.columns):
            if not curr_col in self.molecular_df.columns:
                molecular_df_sub = molecular_df_sub.drop(columns = [curr_col])
        
        #reorder the columns of the submission molecular df so they are in the same order as for the training molecular df
        molecular_df_sub = molecular_df_sub[self.molecular_df.columns]
        #molecular_df_sub = self.__fillna_df(molecular_df_sub, self.molecular_df_nan, ["START", "END", "VAF", "DEPTH"])
                
        return clinical_df_sub, molecular_df_sub
        
class Dataset():
    def __init__(self, status_df, clinical_df, molecular_df, clinical_df_sub, molecular_df_sub, min_occurences=5):
        '''

        Parameters
        ----------
        status_df : DataFrame
            Dataframe containing the data with the target data, so patient 
            status (1 for dead and 0 for alive) and the survival time as 
            prepared by the DatasetPrep class.
        clinical_df : DataFrame
            Dataframe containing the clinical training data as prepared by the 
            DatasetPrep class.
        molecular_df : DataFrame
            Dataframe containing the molecular training data as prepared by 
            the DatasetPrep class.
        min_occurences : int, optional
            Minimum nuber of patients that have a feature in the molecular 
            dataframe for the feature to be kept. Other features are discarded. 
            The default is 5.

        '''
        self.status_df = status_df
        self.patient_ids = np.array(self.status_df.loc[:,"ID"])
        self.patient_num = self.patient_ids.shape[0]
        
        self.clinical_df = clinical_df
        self.clinical_ids = np.array(clinical_df.loc[:,"ID"])
        
        self.molecular_df = molecular_df
        self.molecular_ids = np.array(molecular_df.loc[:,"ID"])
        self.molecular_arr = np.array(self.molecular_df)
        
        #use Dataset.X as training data and Dataset.y as training target
        self.X = np.zeros((self.patient_num, len(clinical_features)+2+len(cyto_markers)+self.molecular_df.shape[1]+1))
        self.y = np.zeros((self.patient_num, 2))
        
        self.__getData()
        
        X_sum = np.sum(self.X.astype(bool), axis=0)
        self.X = pd.DataFrame(self.X, index=np.arange(self.patient_num), columns=[clinical_features + ["XX", "XY"] + ["CYTOGENETICS_"+val for val in cyto_markers] + 
                                                                                  ["MUTATIONS_NUMBER", "AVG_MUTATION_LENGTH", "MEDIAN_MUTATION_LENGTH", "EFFECT_MEDIAN_SURVIVAL"] + ["MUTATIONS_SUB", "MUTATIONS_DEL", "MUTATIONS_INS"] + ["VAF_SUM", "VAF_MEDIAN", "DEPTH_SUM", "DEPTH_MEDIAN"] + list(self.molecular_df.columns)[10:]])
        
        #self.X.loc[:, "WBC"] = np.log((self.X["WBC"]-0.15)+1e-9)
        #self.X.loc[:, "ANC"] = np.log((self.X["ANC"]+1)*1e-9)
        
        #remove columns corresponding to features from self.X which are present in less than min_occurences patients
        sparse_features = self.X.columns
        self.sparse_features = sparse_features[X_sum < min_occurences]
        self.X = self.X.drop(columns=self.sparse_features)
        
        #get the submission data and patient ids
        self.X_sub, self.patient_ids_sub = self.submission_data(clinical_df_sub, molecular_df_sub)
        
        #repeat the search for the sparse features in the submission data
        #remove any columns from both the training and submission data that occur less than min_occurences/3 times in the submission data
        X_sum_sub = np.sum(np.array(self.X_sub).astype(bool), axis=0)
        sparse_features_sub = self.X_sub.columns
        self.sparse_features_sub = sparse_features_sub[X_sum_sub < min_occurences/3]
        self.X_sub = self.X_sub.drop(columns=self.sparse_features_sub)
        self.X = self.X.drop(columns=self.sparse_features_sub)
        
        #remove multiindex columns from X and X_sub
        self.X.columns = ['_'.join(col) for col in self.X.columns]
        self.X_sub.columns = ['_'.join(col) for col in self.X_sub.columns]
        
    def __getData(self) -> None:
        '''
        
        Fill self.X and self.y with the transformed data.

        '''
        self.pos=0
        
        for idx in range(self.patient_num):
            curr_X_item, curr_y_item = self.__getItem(idx)
            self.X[idx] = curr_X_item
            self.y[idx] = curr_y_item
            
            self.pos+=1
            
        self.X = np.nan_to_num(self.X, nan=0)
        
        #self.X[:,len(clinical_features) + 2 + len(cyto_markers) + 3][self.X[:,len(clinical_features) + 2 + len(cyto_markers) + 3] == 0] = np.median(self.y[:,1][[True if val == 0 and bal == 1 else False for val,bal in zip(self.X[:,len(clinical_features) + 2 + len(cyto_markers) + 3], self.y[:,0])]])
        
    def __getItem(self, idx):
        '''

        Parameters
        ----------
        idx : int
            Patient with ID at position idx in self.patient_ids for which the 
            transformed clinical and molecular data is determined.

        Returns
        -------
        item : ndarray
            Array containing the transformed clinical and molecular data.
            It is structured in the following way:
                item[0:len(clinical_features)]:
                    Data for the current patient from the clinical df of the 
                    columns specified in clinical_features
                    
                item[len(clinical_features):len(clinical_features)+2]:
                    Whether the patient has XX or XY chromosomes, the first 
                    element is 1 if XX is present and the second element is 1 
                    if XY is present. Some patients have 0 at both elements if 
                    this information could not be found in the cytogenetic data 
                    of the current patient, e.g. if the cytogenetics are "complex".
                    
                item[len(clinical_features)+2:len(clinical_features)+2+len(cyto_markers)]:
                    Each element contains a 1 if the corresponding cytogenetic 
                    marker could be found in the patients cytogenetics.
                    
                item[len(clinical_features)+2+len(cyto_markers):len(clinical_features)+2+len(cyto_markers)+4]:
                    The first element is the number of somatic mutations of the patient. 
                    The second element is the average length of the somatic mutations. 
                    The third element is the median length of the somatic mutations 
                    with length greater than 0. 
                    The fourth element is the sum of the expected median survival times 
                    obtained from the effects of the somatic mutations.
                    
                item[len(clinical_features)+2+len(cyto_markers)+4:len(clinical_features)+2+len(cyto_markers)+4+3]:
                    The number of sumatic mutations resulting from a substitution, 
                    deletion or insertion in the sequence of the gene.
                    
                item[len(clinical_features)+2+len(cyto_markers)+4+3:len(clinical_features)+2+len(cyto_markers)+4+3+4]:
                    In the first two elements is the total sum and median multiplied by the 
                    number of mutations of the variant allele functions of the mutations.
                    In the last two elements is the total sum and median multiplied by the 
                    number of mutations of the depth of the mutations.
                    
                item[len(clinical_features)+2+len(cyto_markers)+4+3+4:len(clinical_features)+2+len(cyto_markers)+4+3+4+len(list(self.molecular_df.colums)[:10])]:
                    One hot encoded features, per default "CHR" and "GENE".
        status_item : tuple
            Tuple with the status of the current patient with type bool at 
            position 0 and the survival time of the current patient with type 
            float at position 1.

        '''
        #id of the current patient
        curr_patient_id = self.patient_ids[idx]
        
        #status and clinical data for the current patient
        curr_status = np.array(self.status_df.iloc[idx])
        curr_clinical = np.array(self.clinical_df.loc[idx])
        
        #check that the id in curr_clinical matches curr_patient_id
        if curr_status[0] != curr_patient_id:
            print("STATUS ID ERROR")
            
        if curr_clinical[0] != curr_patient_id:
            print("CLINICAL ID ERROR")
            
        #status_item obtained from curr_status to return
        status_item = (bool(curr_status[2]), curr_status[1])
        
        #get clinical_item which is part of item which gets returned
        clinical_item = np.zeros(len(clinical_features)+2+len(cyto_markers))
        clinical_item[0:len(clinical_features)] = curr_clinical[clinical_indices]
        curr_cyto = curr_clinical[8]
        #if there are no cytogenetics for the current patient available, the columns for XX or XY chromosomes are left at 0
        if str(curr_cyto)!="nan":
            curr_cyto = curr_cyto.strip().upper()
            if ",XX," == curr_cyto[2:6] or ",XX[" == curr_cyto[2:6] or "46,XX" == curr_cyto:
                clinical_item[len(clinical_features)] = 1
            if ",XY," == curr_cyto[2:6] or ",XY[" == curr_cyto[2:6] or "46,XY" == curr_cyto:
                clinical_item[len(clinical_features)+1] = 1
            if "~" == curr_cyto[2]:
                if ",XX," == curr_cyto[5:9] or ",XX[" == curr_cyto[5:9]:
                    clinical_item[len(clinical_features)] = 1
                if ",XY," == curr_cyto[5:9] or ",XY[" == curr_cyto[5:9]:
                    clinical_item[len(clinical_features)+1] = 1
        #check which cytogenetic markers are in the current patients cytogenetics
        curr_cyto_risk = self.cyto_patient_risk(curr_cyto)
        clinical_item[len(clinical_features)+2:] = curr_cyto_risk
        
        #molecular data for the current patient
        curr_molecular = np.array(self.molecular_df[self.molecular_df["ID"]==curr_patient_id])
        
        #get molecular_item which is part of item which gets returned
        #if the current patient has no recorded somatic mutations, the molecular item is set to an array containing only zeros
        if len(curr_molecular)==0:
            molecular_item = np.zeros(self.molecular_df.shape[1]+1)
            
            #set MEDIAN_EFFECT_SURVIVAL to 2.8
            #molecular_item[3] = np.median(np.array(self.status_df)[:,1][np.array(self.status_df)[:,2]==1])
        
        else:
            molecular_item = np.zeros((len(curr_molecular), len(curr_molecular[0])+1))
            #get the lengths of each somatic mutation
            molecular_item[:,1] = curr_molecular[:,2]-curr_molecular[:,1]
            molecular_lengths = molecular_item[:,1]
            #get the one hot encoded features
            molecular_item[:,11:] = curr_molecular[:,10:]
            #sum over all one hot encoded features
            molecular_item = np.sum(molecular_item, axis=0)
            
            #number of mutations
            molecular_item[0] = len(curr_molecular)
            #average length of the mutations
            #molecular_item[1] = np.sum(molecular_lengths)/len(molecular_lengths)
            #median length of the mutations with length greater than 0
            #molecular_item[2] = np.median([val for val in molecular_lengths if val>0])
            #if only mutations of length 0 are present set molecular_item[2] to 0
            if str(molecular_item[2]) == "nan":
                molecular_item[2] = 0
                
            #take the mean over the expected median survival time of the different effects of the mutations
            molecular_item[3] = np.mean(curr_molecular[:,7])#/len([val for val in curr_molecular[:,7] if val>0])
            
            #get the number of mutations from a substitution, deletion and insertion
            molecular_ref = curr_molecular[:,3]
            molecular_alt = curr_molecular[:,4]
            
            molecular_mutation_type = np.zeros(3)
            molecular_lengths = []
            molecular_lengths_median = []
            for i in range(len(molecular_ref)):
                temp_molecular_mutation_type, temp_mut_len = self.classify_mutation(molecular_ref[i], molecular_alt[i])
                molecular_mutation_type += temp_molecular_mutation_type
                molecular_lengths.append(temp_mut_len)
                if temp_molecular_mutation_type[0]!=1:
                    molecular_lengths_median.append(temp_mut_len)
                
            molecular_item[1] = np.sum(molecular_lengths)/len(molecular_lengths)
            #molecular_item[2] = np.median([val for val in molecular_lengths_median if val>0])
            if len(molecular_lengths_median)==0:
                molecular_item[2] = 0
            else:
                molecular_item[2] = np.median(molecular_lengths_median)
            
            #molecular_mutation_type = np.array([self.classify_mutation(val, bal) for val,bal in zip(molecular_ref, molecular_alt)])
            #molecular_item[4:7] = np.sum(molecular_mutation_type, axis=0)
            molecular_item[4:7] = molecular_mutation_type
            
            #vaf and depth of the mutations
            molecular_vaf = curr_molecular[:,8]
            molecular_depth = curr_molecular[:,9]
            
            #sum of the vafs
            molecular_item[7] = np.sum(molecular_vaf)
            #median of the vafs multiplied by the number of mutations
            molecular_item[8] = np.median(molecular_vaf)*len(molecular_vaf)
            #sum of the depths
            molecular_item[9] = np.sum(molecular_depth)
            #median of the depths multiplied by the number of mutations
            molecular_item[10] = np.median(molecular_depth)*len(molecular_depth)
        
        #combine the clinical and molecular item to get the item for the return
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
            return np.zeros(len(cyto_markers))
        
        cyto=cyto.strip().upper()
        
        res = np.zeros(len(cyto_markers))
        
        for i in range(len(cyto_markers)):
            if cyto_markers[i] in cyto:
                res[i] = 1
        
        return res
    
    def classify_mutation(self, ref, alt):
        """
        Classify a mutation based on the REF and ALT values.
        
        Returns:
            mutation_type (str): One of 'substitution', 'deletion', or 'insertion'
            mutation_descriptor (str): A string describing the mutation (e.g., "A>G", "del_16", "ins_3")
            length_change (int): The difference in length (0 for substitutions)
        """
        
        res = np.zeros(3)
        mut_len = 0
        
        if str(ref)=="nan" or str(alt)=="nan":
            return res, mut_len
        
        if len(ref) == len(alt):
            # Substitution
            if ref=="-":
                res[2] = 1
                mut_len = 1
            elif alt=="-":
                res[1] = 1
                mut_len = 1
            else:
                res[0] = 1
                mut_len = len(ref)
        elif len(ref) > len(alt):
            # Deletion: nucleotides removed
            res[1] = 1
            mut_len = len(ref)
        else:
            # Insertion: extra nucleotides added
            res[2] = 1
            mut_len = len(alt)
            
        if mut_len == 0 and (res[1]!=0 or res[2]!=0): print("AAAAAAA")
        
        return res, mut_len
    
    
    def submission_data(self, clinical_df_sub, molecular_df_sub):
        '''

        Parameters
        ----------
        clinical_df_sub : DataFrame
            Dataframe containinc the clinical data for the submission as prepared 
            by DatasetPrep.submission_data_prep.
        molecular_df_sub : DataFrame
            Dataframe containinc the molecular data for the submission as prepared 
            by DatasetPrep.submission_data_prep.

        Returns
        -------
        X_sub : DataFrame
            Dataframe with the transformed clinical and molecular data for the 
            submission.
        patient_ids_sub : ndarray
            Array with the different patient ids in the submission data.

        '''
        patient_ids_sub = np.array(clinical_df_sub.loc[:,"ID"])
        patient_num_sub = patient_ids_sub.shape[0]
        
        X_sub = np.zeros((patient_num_sub, len(clinical_features)+2+len(cyto_markers)+self.molecular_df.shape[1]+1))
        
        for i in range(patient_num_sub):
            curr_patient_id = patient_ids_sub[i]
            curr_clinical = clinical_df_sub[clinical_df_sub["ID"] == curr_patient_id]
            curr_molecular = molecular_df_sub[molecular_df_sub["ID"] == curr_patient_id]
            X_sub[i] = self.__getItem_sub(np.array(curr_clinical)[0], np.array(curr_molecular))
            
        X_sub = pd.DataFrame(X_sub, index=np.arange(patient_num_sub), columns=[clinical_features + ["XX", "XY"] + ["CYTOGENETICS_"+val for val in cyto_markers] + 
                                                                                  ["MUTATIONS_NUMBER", "AVG_MUTATION_LENGTH", "MEDIAN_MUTATION_LENGTH", "EFFECT_MEDIAN_SURVIVAL"] + ["MUTATIONS_SUB", "MUTATIONS_DEL", "MUTATIONS_INS"] + ["VAF_SUM", "VAF_MEDIAN", "DEPTH_SUM", "DEPTH_MEDIAN"] + list(self.molecular_df.columns)[10:]])
            
        X_sub = X_sub.drop(columns=self.sparse_features)
        
        return X_sub, patient_ids_sub
        
    def __getItem_sub(self, curr_clinical, curr_molecular):
        '''

        Parameters
        ----------
        curr_clinical : ndarray
            Array containing the clinical data of the current patient.
        curr_molecular : ndarray
            Array containing the molecular data of the current patient.

        Returns
        -------
        item : ndarray
            Same as item in Dataset.__getItem.

        '''
        #prepare this the same way as in Dataset.__getItem
        curr_patient_id = curr_clinical[0]
        clinical_item = np.zeros(len(clinical_features)+2+len(cyto_markers))
        clinical_item[0:len(clinical_features)] = curr_clinical[clinical_indices]
        curr_cyto = curr_clinical[8]
        if str(curr_cyto)!="nan":
            curr_cyto = curr_cyto.strip().upper()
            if ",XX," == curr_cyto[2:6] or ",XX[" == curr_cyto[2:6] or "46,XX" == curr_cyto:
                clinical_item[len(clinical_features)] = 1
            if ",XY," == curr_cyto[2:6] or ",XY[" == curr_cyto[2:6] or "46,XY" == curr_cyto:
                clinical_item[len(clinical_features)+1] = 1
            if "~" == curr_cyto[2]:
                if ",XX," == curr_cyto[5:9] or ",XX[" == curr_cyto[5:9]:
                    clinical_item[len(clinical_features)] = 1
                if ",XY," == curr_cyto[5:9] or ",XY[" == curr_cyto[5:9]:
                    clinical_item[len(clinical_features)+1] = 1
        curr_cyto_risk = self.cyto_patient_risk(curr_cyto)
        clinical_item[len(clinical_features)+2:] = curr_cyto_risk
                
        if len(curr_molecular)==0:
            molecular_item = np.zeros(self.molecular_df.shape[1]+1)
        
        else:
            molecular_item = np.zeros((len(curr_molecular), len(curr_molecular[0])+1))
            molecular_item[:,1] = curr_molecular[:,2]-curr_molecular[:,1]
            #molecular_lengths = molecular_item[:,1]
            molecular_item[:,11:] = curr_molecular[:,10:]
            molecular_item = np.sum(molecular_item, axis=0)
            
            molecular_item[0] = len(curr_molecular)
            molecular_item[3] = np.mean(curr_molecular[:,7])
            #molecular_item[1] = np.sum(molecular_lengths)/len(molecular_lengths)
            #molecular_item[2] = np.median([val for val in molecular_lengths if val>0])
            
            if str(molecular_item[2]) == "nan":
                molecular_item[2] = 0
            
            molecular_ref = curr_molecular[:,3]
            molecular_alt = curr_molecular[:,4]
            
            molecular_mutation_type = np.zeros(3)
            molecular_lengths = []
            molecular_lengths_median = []
            for i in range(len(molecular_ref)):
                temp_molecular_mutation_type, temp_mut_len = self.classify_mutation(molecular_ref[i], molecular_alt[i])
                molecular_mutation_type += temp_molecular_mutation_type
                molecular_lengths.append(temp_mut_len)
                if temp_molecular_mutation_type[0]!=1:
                    molecular_lengths_median.append(temp_mut_len)
                
            molecular_item[1] = np.sum(molecular_lengths)/len(molecular_lengths)
            #molecular_item[2] = np.median([val for val in molecular_lengths_median if val>0])
            if len(molecular_lengths_median)==0:
                molecular_item[2] = 0
            else:
                molecular_item[2] = np.median(molecular_lengths_median)
                
            #molecular_mutation_type = np.array([self.classify_mutation(val, bal) for val,bal in zip(molecular_ref, molecular_alt)])
            #molecular_item[4:7] = np.sum(molecular_mutation_type, axis=0)
            molecular_item[4:7] = molecular_mutation_type
            
            molecular_vaf = curr_molecular[:,8]
            molecular_depth = curr_molecular[:,9]
            molecular_item[7] = np.sum(molecular_vaf)
            molecular_item[8] = np.median(molecular_vaf)*len(molecular_vaf)
            molecular_item[9] = np.sum(molecular_depth)
            molecular_item[10] = np.median(molecular_depth)*len(molecular_depth)
        
        item = np.append(clinical_item, molecular_item)
        
        return item


        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
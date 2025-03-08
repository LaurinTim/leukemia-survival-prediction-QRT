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
        clinical_df_sub = pd.read_csv(clinical_file_sub)
        #clinical_df_sub = pd.read_csv(clinical_file)
        clinical_df_sub = self.__fillna_df(clinical_df_sub, self.clinical_df_nan, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
        
        clinical_sub_sort_index = [float(val[3:]) for val in list(clinical_df_sub.loc[:,"ID"])]
        clinical_df_sub.insert(9, "sort_index", clinical_sub_sort_index)
        clinical_df_sub = clinical_df_sub.sort_values(by=["sort_index"]).reset_index(drop=True)
        clinical_df_sub = clinical_df_sub.drop(columns=["sort_index"])
        
        patient_ids_sub = np.array(clinical_df_sub.loc[:,"ID"])
        patient_num_sub = patient_ids_sub.shape[0]
        
        molecular_df_sub = pd.read_csv(molecular_file_sub)
        #molecular_df_sub = pd.read_csv(molecular_file)
        molecular_df_sub = pd.get_dummies(molecular_df_sub, columns = self.molecular_dummies_columns)
        molecular_df_sub = self.__fillna_df(molecular_df_sub, self.molecular_df_nan, ["START", "END", "VAF", "DEPTH"])
                        
        molecular_sub_sort_index = [float(val[3:]) for val in list(molecular_df_sub.loc[:,"ID"])]
        molecular_df_sub.insert(11, "sort_index", molecular_sub_sort_index)
        molecular_df_sub = molecular_df_sub.sort_values(by=["sort_index"]).reset_index(drop=True)
        molecular_df_sub = molecular_df_sub.drop(columns=["sort_index"])
        
        global_median_survival = np.median(self.status_arr[:,1])
        molecular_df_sub.insert(7, "EFFECT_MEDIAN_SURVIVAL", np.array([self.effects_map.get(val) if val in self.effects_map.keys() else global_median_survival for val in molecular_df_sub["EFFECT"]]))
        
        for curr_col in list(self.molecular_df.columns):
            if not curr_col in molecular_df_sub.columns:
                molecular_df_sub.insert(0, curr_col, np.zeros(len(molecular_df_sub)))
                
        for curr_col in list(molecular_df_sub.columns):
            if not curr_col in self.molecular_df.columns:
                molecular_df_sub = molecular_df_sub.drop(columns = [curr_col])
        
        molecular_df_sub = molecular_df_sub[self.molecular_df.columns]
        molecular_df_sub = self.__fillna_df(molecular_df_sub, self.molecular_df_nan, ["START", "END", "VAF", "DEPTH"])
                
        return clinical_df_sub, molecular_df_sub
        
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
        
        self.X = np.zeros((self.patient_num, len(clinical_features)+2+len(cyto_markers)+self.molecular_df.shape[1]+1))
        self.y = np.zeros((self.patient_num, 2))
        
        self.__getData()
        
        X_sum = np.sum(self.X.astype(bool), axis=0)
        self.X = pd.DataFrame(self.X, index=np.arange(self.patient_num), columns=[clinical_features + ["XX", "XY"] + ["CYTOGENETICS_"+val for val in cyto_markers] + 
                                                                                  ["MUTATIONS_NUMBER", "AVG_MUTATION_LENGTH", "MEDIAN_MUTATION_LENGTH", "EFFECT_MEDIAN_SURVIVAL"] + ["MUTATIONS_SUB", "MUTATIONS_DEL", "MUTATIONS_INS"] + ["VAF_SUM", "VAF_MEDIAN", "DEPTH_SUM", "DEPTH_MEDIAN"] + list(self.molecular_df.columns)[10:]])
        
        sparse_features = self.X.columns
        self.sparse_features = sparse_features[X_sum < min_occurences]
        self.X = self.X.drop(columns=self.sparse_features)
        
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
        
        curr_molecular = np.array(self.molecular_df[self.molecular_df["ID"]==curr_patient_id])
        
        if len(curr_molecular)==0:
            molecular_item = np.zeros(self.molecular_df.shape[1]+1)
        
        else:
            molecular_item = np.zeros((len(curr_molecular), len(curr_molecular[0])+1))
            molecular_item[:,1] = curr_molecular[:,2]-curr_molecular[:,1]
            molecular_lengths = molecular_item[:,1]
            molecular_item[:,11:] = curr_molecular[:,10:]
            molecular_item = np.sum(molecular_item, axis=0)
            
            molecular_item[0] = len(curr_molecular)
            molecular_item[3] = np.sum(curr_molecular[:,7])
            molecular_item[1] = np.sum(molecular_lengths)/len(molecular_lengths)
            molecular_item[2] = np.median([val for val in molecular_lengths if val>0])
            if str(molecular_item[2]) == "nan":
                molecular_item[2] = 0
            
            molecular_ref = curr_molecular[:,3]
            molecular_alt = curr_molecular[:,4]
            molecular_mutation_type = np.array([self.classify_mutation(val, bal) for val,bal in zip(molecular_ref, molecular_alt)])
            molecular_item[4:7] = np.sum(molecular_mutation_type, axis=0)
            
            molecular_vaf = curr_molecular[:,8]
            molecular_depth = curr_molecular[:,9]
            
            molecular_item[7] = np.sum(molecular_vaf)
            molecular_item[8] = np.median(molecular_vaf)*len(molecular_vaf)
            molecular_item[9] = np.sum(molecular_depth)
            molecular_item[10] = np.median(molecular_depth)*len(molecular_depth)
        
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
        
        if str(ref)=="nan" or str(alt)=="nan":
            return res
        
        if len(ref) == len(alt):
            # Substitution
            res[0] = 1
            if ref=="-":
                res[2] = 1
            if alt=="-":
                res[1] = 1
        elif len(ref) > len(alt):
            # Deletion: nucleotides removed
            res[1] = 1
        else:
            # Insertion: extra nucleotides added
            res[2] = 1
        
        return res
    
    def submission_data(self, clinical_df_sub, molecular_df_sub):
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
            molecular_lengths = molecular_item[:,1]
            molecular_item[:,11:] = curr_molecular[:,10:]
            molecular_item = np.sum(molecular_item, axis=0)
            
            molecular_item[0] = len(curr_molecular)
            molecular_item[3] = np.sum(curr_molecular[:,7])
            molecular_item[1] = np.sum(molecular_lengths)/len(molecular_lengths)
            molecular_item[2] = np.median([val for val in molecular_lengths if val>0])
            if str(molecular_item[2]) == "nan":
                molecular_item[2] = 0
            
            molecular_ref = curr_molecular[:,3]
            molecular_alt = curr_molecular[:,4]
            molecular_mutation_type = np.array([self.classify_mutation(val, bal) for val,bal in zip(molecular_ref, molecular_alt)])
            molecular_item[4:7] = np.sum(molecular_mutation_type, axis=0)
            
            molecular_vaf = curr_molecular[:,8]
            molecular_depth = curr_molecular[:,9]
            molecular_item[7] = np.sum(molecular_vaf)
            molecular_item[8] = np.median(molecular_vaf)*len(molecular_vaf)
            molecular_item[9] = np.sum(molecular_depth)
            molecular_item[10] = np.median(molecular_depth)*len(molecular_depth)
        
        item = np.append(clinical_item, molecular_item)
        
        return item


        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
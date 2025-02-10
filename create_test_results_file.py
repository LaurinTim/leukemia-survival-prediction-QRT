import warnings
warnings.filterwarnings("ignore")

import lifelines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Our package
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood, log_hazard, survival_function
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator
from torchsurv.stats.ipcw import get_ipcw

# %%

def test_results(model, parameters_file, data, features, model_name, return_df = False):
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
        Pandas DataFrame containing patient ID in column 'ID' and the features of the model, taken from the test files.
    features: list of strings
        Names of the columns containing the features of the model in the order that the model expects.
    model_name: str
        Name of the model, this is also the name the the created csv file will have.
    return_df: bool, optional
        If set to True (default) then the DataFrame that is created gets returned but not saved.

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
        return






















































































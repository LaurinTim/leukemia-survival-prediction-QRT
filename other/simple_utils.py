import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import PowerTransformer

# %%

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def fill_nan(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list, method: str = 'zeros'):
    '''
    Fill nan values of train and test df. For median and mean, the nan values 
    in the test dataframe are filled with the values derived from the training 
    dataframe.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    test_df : pd.DataFrame
        Test dataframe.
    columns : list
        List of columns in which nan values are filled in. All columns have to 
        be numeric.
    method : str, optional
        How the nan values are filled. If method is equal to 'zeros, all nan 
        values are filled with zeros. For 'median', the nan values are filled 
        with the median value of the non nan elements of each column in the 
        training dataframe. Same for 'mean', except that the average instead 
        of the median is used. The default is 'zeros'.

    Returns
    -------
    train_df : pd.DataFrame
        Training dataframe with filled nan values.
    test_df : pd.DataFrame
        Test dataframe with filled nan values.

    '''
    if method=='zeros':
        values = {col: 0 for col in columns}
        
    elif method=='median':
        values = {col: train_df[col].median(skipna=True) for col in columns}
        
    elif method=='mean':
        values = {col: train_df[col].mean(skipna=True) for col in columns}
        
    train_df = train_df.fillna(values)
    test_df = test_df.fillna(values)
    
    return train_df, test_df
    
def transform_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list):
    yeo = PowerTransformer(method='yeo-johnson')
    
    for col in columns:
        train_df[col] = yeo.fit_transform(np.array(train_df[col]).reshape(-1,1)).flatten()
        test_df[col] = yeo.transform(np.array(test_df[col]).reshape(-1,1)).flatten()
    
    return train_df, test_df

def cyto_patient_risk_with_favorable(cyto_train: pd.DataFrame, cyto_test: pd.DataFrame):
    '''
    
    Parameters
    ----------
    cyto_train : pd.DataFrame
        Cytogenetic information for each patient in the training data.
    cyto_test : pd.DataFrame
        Cytogenetic information for each patient in the test data.

    Returns
    -------
    res_train : pd.DataFrame
        Dataframe containing information about the presence of 
        favorable/adverse markers present in each patients cytogenetics. This 
        ataframe has three columns, one for normal cytogenetics, one for 
        favorable and one for adverse. The favorable and adverse columns 
        contain a 1 for each patient where a corresponding marker was found in 
        the cytogenetics, otherwise 0. If neither favorable nor adverse 
        markers were found for a patient, the value in the normal column get 
        set to 1 for the patient, otherwise it is 0.
    res_test : pd.DataFrame
        Same as res_train but for the patients in the test set.

    '''
    cyto_train = cyto_train.fillna('').apply(lambda x: x.strip().lower())
    cyto_test = cyto_test.fillna('').apply(lambda x: x.strip().lower())
            
    favorable_markers = ['t(15;17)', 'inv(16)', 't(8;21)'] # ['t(15;17)(q22;q12)']
    adverse_markers = ["del(5)", "-5", "del(7)", "-7", "t(6;9)", "inv(3)", "t(3;3)", "t(9;22)", "t(4;11)", "del(3)", 'complex']
    
    def _process(df: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(np.zeros((df.shape[0], 3)), columns=['NORMAL_CYTO', 'FAVORABLE_CYTO', 'ADVERSE_CYTO'])
        
        for i in range(df.shape[0]):
            curr_cyto = cyto_train.iloc[i]
            
            has_favorable = np.array([val in curr_cyto for val in favorable_markers]).any()
            if has_favorable:
                res.iloc[i,1] = 1
                continue
            
            has_adverse = np.array([val in curr_cyto for val in adverse_markers]).any()
            if not has_adverse:
                num_cyto = len([kal for val in curr_cyto.split('/') for kal in val.split(',')])
                if num_cyto>=5: has_adverse=True
            if has_adverse:
                res.iloc[i,2] = 1
                
            if not has_favorable and not has_adverse:
                res.iloc[i,0] = 1
            
        return res
    
    res_train = _process(cyto_train)
    res_test = _process(cyto_test)
    
    return res_train, res_test

def cyto_patient_risk(cyto_train: pd.DataFrame, cyto_test: pd.DataFrame):
    '''
    
    Parameters
    ----------
    cyto_train : pd.DataFrame
        Cytogenetic information for each patient in the training data.
    cyto_test : pd.DataFrame
        Cytogenetic information for each patient in the test data.

    Returns
    -------
    res_train : pd.DataFrame
        Dataframe containing information about the presence of 
        adverse markers present in each patients cytogenetics. This 
        ataframe has two columns, one for normal cytogenetics and one for 
        the presence of adverse markers. The adverse column contain a 1 for 
        each patient where a adverse marker was found in  the cytogenetics, 
        otherwise 0. If no adverse marker was found for a patient, the value 
        in the normal column is set to 1 for the patient, otherwise it is 0.
    res_test : pd.DataFrame
        Same as res_train but for the patients in the test set.

    '''
    cyto_train = cyto_train.fillna('').apply(lambda x: x.strip().lower())
    cyto_test = cyto_test.fillna('').apply(lambda x: x.strip().lower())
            
    adverse_markers = ["del(5)", "-5", "del(7)", "-7", "t(6;9)", "inv(3)", "t(3;3)", "t(9;22)", "t(4;11)", "del(3)", 'complex']
    
    def _process(df: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(np.zeros((df.shape[0], 2)), columns=['NORMAL_CYTO', 'ADVERSE_CYTO'])
        
        for i in range(df.shape[0]):
            curr_cyto = df.iloc[i]
                        
            has_adverse = np.array([val in curr_cyto for val in adverse_markers]).any()
            if not has_adverse:
                num_cyto = len([kal for val in curr_cyto.split('/') for kal in val.split(',')])
                if num_cyto>=5: has_adverse=True
                
            if has_adverse:
                res.iloc[i,1] = 1
            else:
                res.iloc[i,0] = 1
            
        return res
    
    res_train = _process(cyto_train)
    print()
    res_test = _process(cyto_test)
    
    return res_train, res_test































































































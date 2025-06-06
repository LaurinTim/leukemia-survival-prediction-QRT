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
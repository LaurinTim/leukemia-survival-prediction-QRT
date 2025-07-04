import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import PowerTransformer

# %%

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def remove_low_censoring(status_df, censoring_time):
    #censored_df = status_df[[True if val==1 else False for val in list(status_df['OS_STATUS'])]]
    #remove_df = censored_df[[True if val<=censoring_time else False for val in list(censored_df['OS_YEARS'])]]
    #remove_ids = list(remove_df)
    return_df = status_df[[False if val==1 and bal<=censoring_time else True for val,bal in zip(status_df['OS_STATUS'], status_df['OS_YEARS'])]]
    
    return return_df
    
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
    res_test = _process(cyto_test)
    
    return res_train, res_test

def patient_gender(cyto_train: pd.DataFrame, cyto_test: pd.DataFrame):
    cyto_train = cyto_train.fillna('').apply(lambda x: x.strip().lower())
    cyto_test = cyto_test.fillna('').apply(lambda x: x.strip().lower())
    
    def _process(df: pd.DataFrame) -> pd.DataFrame:        
        xx_mask = list(df.str.contains('xx').astype(int))
        xy_mask = list(df.str.contains('xy').astype(int))
        unknown_mask = [1 if val+bal==0 else 0 for val,bal in zip(xx_mask, xy_mask)]
                
        res = pd.DataFrame([xx_mask, xy_mask, unknown_mask], index=['XX', 'XY', 'UNKNOWN']).T
        
        return res
    
    res_train = _process(cyto_train)
    res_test = _process(cyto_test)
    
    return res_train, res_test

def molecular_transform(train_df, test_df, all_train_ids, all_test_ids):
    gene_counts_train = train_df.groupby(["ID", "GENE"]).size().unstack(fill_value=0)
    gene_counts_test  = test_df.groupby(["ID", "GENE"]).size().unstack(fill_value=0)
    all_genes = sorted(set(gene_counts_train.columns) | set(gene_counts_test.columns))
    gene_counts_train = gene_counts_train.reindex(columns=all_genes, fill_value=0)
    gene_counts_test  = gene_counts_test.reindex(columns=all_genes, fill_value=0)
    
    effect_counts_train = train_df.groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
    effect_counts_test  = test_df.groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
    
    all_effects = sorted(set(effect_counts_train.columns) | set(effect_counts_test.columns))
    effect_counts_train = effect_counts_train.reindex(columns=all_effects, fill_value=0)
    effect_counts_test  = effect_counts_test.reindex(columns=all_effects, fill_value=0)
    
    agg_stats_train = train_df.groupby("ID").agg(
        total_mutations = ("ID", "size"),
        unique_genes    = ("GENE", "nunique"),
        mean_VAF        = ("VAF", "mean"),
        max_VAF         = ("VAF", "max")
    )
    agg_stats_test = test_df.groupby("ID").agg(
        total_mutations = ("ID", "size"),
        unique_genes    = ("GENE", "nunique"),
        mean_VAF        = ("VAF", "mean"),
        max_VAF         = ("VAF", "max")
    )
        
    train_features = agg_stats_train.join(gene_counts_train, how="outer").join(effect_counts_train, how="outer")    
    test_features  = agg_stats_test.join(gene_counts_test,  how="outer").join(effect_counts_test,  how="outer")
    
    train_features = train_features.reindex(all_train_ids, fill_value=0)
    test_features  = test_features.reindex(all_test_ids, fill_value=0)
        
    train_features = train_features.reset_index().rename(columns={"index": "ID"})
    test_features  = test_features.reset_index().rename(columns={"index": "ID"})
        
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)
    
    return train_features, test_features

def reduce_df(train_df, test_df, num=30):
    nonzero_counts = (train_df != 0).sum(axis=0)
    cols_to_keep = nonzero_counts[nonzero_counts >= num].index
    
    train_df_reduced = train_df.loc[:, cols_to_keep]
    test_df_reduced = test_df.loc[:, cols_to_keep]
    
    return train_df_reduced, test_df_reduced






























































































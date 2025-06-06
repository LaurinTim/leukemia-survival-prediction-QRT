import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import PowerTransformer

# %%

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def fill_nan():
    return
    
def transform_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list):
    yeo = PowerTransformer(method='yeo-johnson')
    
    for col in columns:
        train_df[col] = yeo.fit_transform(np.array(train_df[col]).reshape(-1,1)).flatten()
        test_df[col] = yeo.transform(np.array(test_df[col]).reshape(-1,1)).flatten()
    
    return train_df, test_df
import pandas as pd
import numpy as np
from sksurv.preprocessing import OneHotEncoder

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES']

# %%

target = pd.read_csv(data_dir+"\\target_train.csv")
cl = pd.read_csv(data_dir+"\\X_train\\clinical_train.csv")
mo = pd.read_csv(data_dir+"\\X_train\\molecular_train.csv")
idp = list(cl.ID)

# %%

target.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
cl = cl.loc[cl['ID'].isin(target['ID'])]
mo = mo.loc[mo['ID'].isin(target['ID'])]

# %%

#tst = mo[['CHR', 'REF', 'ALT', 'GENE', 'PROTEIN_CHANGE', 'EFFECT']]
tst = mo[['CHR', 'GENE', 'EFFECT']]
a = pd.get_dummies(tst)
b = a[mo.ID == idp[0]]

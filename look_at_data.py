import pandas as pd
import numpy as np

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\Challenge Data QRT"

features = ['BM_BLAST', 'HB', 'PLT']

# %%

target = pd.read_csv(data_dir+"\\target_train.csv")
cl = pd.read_csv(data_dir+"\\X_train\\clinical_train.csv")
mo = pd.read_csv(data_dir+"\\X_train\\molecular_train.csv")

# %%

target.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
cl = cl.loc[cl['ID'].isin(target['ID'])]
mo = mo.loc[mo['ID'].isin(target['ID'])]

# %%

temp_id = str(target.ID.iloc[0])
temp_mo = mo[mo.ID == temp_id]

# %%

target = target.iloc[:10]

# %%

t = np.array([(str(val),bal,kal) for val,bal,kal in zip(target.ID, target.OS_YEARS, target.OS_STATUS)], dtype = [('ID', (np.str_,10)), ('y', float), ('s', bool)])

# %%

cl = pd.read_csv(data_dir+"\\X_train\\clinical_train.csv")

# %%

cl = cl.loc[cl.ID!='n', features]
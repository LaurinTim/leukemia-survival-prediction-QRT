from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\Challenge Data QRT\\target_train.csv")
df = df[[True if val in [0,1] else False for val in df.OS_STATUS]]
df.OS_STATUS = [bool(val) for val in df.OS_STATUS]

# %%

print(concordance_index_censored(df.OS_STATUS, df.OS_YEARS, df.OS_YEARS))

# %%

print(concordance_index_censored([False,True,True], [2.4,0.8,2.1], [0.5, 1.2, 0.6]))
print(concordance_index_ipcw([False,True,True], [2.4,0.8,2.1], [0.5, 1.2, 0.6]))

# %%

print(concordance_index_censored([False,True], [3.1,2.1], [0.5, 0.6]))

# %%

print(concordance_index_ipcw(np.dtype([(False, 0.5), (True, 0.6)]), np.dtype([(False, 3.1), (True, 2.1)]), [0.5, 0.6]))

# %%

print(concordance_index_ipcw(np.array([(False, 1.5), (True, 2.1)], dtype = [('status', bool), ('years', float)]), np.array([(False, 3.1), (True, 2.1)], dtype = [('status', bool), ('years', float)]), [0.5, 0.6]))

# %%

tst = np.array([(False, 0.5), (True, 0.6)], dtype = [('status', bool), ('years', float)])
#Same as cox_model.py but now use 4 features, as an additional feature the number of somatic mutations, call this NSM

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch

# Our package
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

#IPCW Concordance Index from sksurv
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

# %%

#_ = torch.manual_seed(42)
n = 64
time_train = torch.randint(low=5, high=250, size=(2*n,)).float()
event_train = torch.randint(low=0, high=2, size=(2*n,)).bool()
time_test = torch.randint(low=5, high=250, size=(n,)).float()
event_test = torch.randint(low=0, high=2, size=(n,)).bool()
estimate = torch.randn((n,))

# %%

cindex = ConcordanceIndex()
ct = cindex(estimate, event_test, time_test) # Harrell's c-index
ipcw = get_ipcw(event_train, time_train, time_test) # ipcw at subject time
#ipcw = 1/ipcw
pt = cindex(estimate, event_test, time_test, weight=ipcw) # Uno's c-index

print(f"Indices with torchsurv:\nCInd: {ct:0.3f}, IPCW CInd: {pt:0.3f}\n")

# %%

status_train = np.array([(bool(val), float(bal)) for val,bal in zip(event_train,time_train)], dtype = [('event', bool), ('time', float)])
status_test = np.array([(bool(val), float(bal)) for val,bal in zip(event_test,time_test)], dtype = [('event', bool), ('time', float)])

cs = concordance_index_censored([val[0] for val in status_test], [val[1] for val in status_test], estimate.numpy())[0]
ps = concordance_index_ipcw(status_train, status_test, estimate.numpy())[0]

print(f"Indices with sksurv:\nCInd: {cs:0.3f}, IPCW CInd: {ps:0.3f}\n")
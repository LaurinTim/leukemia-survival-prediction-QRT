#file containing different commands used in the analysis

import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm
import random
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import torch
import copy
from torch.utils.data import DataLoader, Dataset

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from time import time as ttime

#Directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"
file_status = data_dir+'\\target_train.csv' #containts information about the status of patients, used as training target
file_clinical = data_dir+'\\X_train\\clinical_train.csv' #contains clinical information of patients used for training
file_molecular = data_dir+'\\X_train\\molecular_train.csv' #contains molecular information of patients used for training

#set working directory
import os
os.chdir(data_dir)

#from pytorch_model_utils import * #get_device, set_random_seed, EmbeddingModel, plot_losses, compare_models, adjust_learning_rate, DataPrep, 
import pytorch_model_utils as u

# %% Check if CUDA cores are available for training, if yes set the batch size to 128, otherwise 32

BATCH_SIZE, device = u.get_device()
torch.set_default_device(device)
       
# %%

dat = u.DataPrep(file_status, file_clinical, file_molecular)

# %%

u.set_random_seed(1)

a = u.DatasetGen(dat.status_arr, dat.clinical_arr, dat.molecular_arr, dat.effects_survival_map, dat.molecular_void_ids, 
               status_transformer = u.TransStatus(), clinical_transformer = u.TransClinical(), molecular_transformer = u.TransMolecular())

train_data, val_data, test_data = torch.utils.data.random_split(a, [0.6, 0.2, 0.2], generator=torch.Generator(device=device))

dataloader_train = DataLoader(train_data, batch_size = BATCH_SIZE)
dataloader_val = DataLoader(val_data, batch_size = BATCH_SIZE)
dataloader_test = DataLoader(test_data, batch_size = BATCH_SIZE)

train_time = torch.tensor([val[1][1] for val in train_data]).float()
val_time = torch.tensor([val[1][1] for val in val_data])
test_time = torch.tensor([val[1][1] for val in test_data])

bad_seed = bool(train_time.max() < val_time.max() or train_time.max() < test_time.max())
if bad_seed:
    print('-'*100)
    print('ERROR: THIS IS A BAD SEED, THE MAX TIME IN TRAIN IS SMALLER THAN IN VAL OR TEST, PLEASE SPLIT DATA AGAIN')
    print(train_time.max(), val_time.max(), test_time.max())
    print('-'*100)

# %%

train_x, train_event = u.get_x_and_event(train_data)
val_x, val_event = u.get_x_and_event(val_data)
test_x, test_event = u.get_x_and_event(test_data)

# %% Sanity check

x, (event, time) = next(iter(dataloader_train))
num_features = x.size(1)

print(f"x (shape)    = {x.shape}")
print(f"num_features = {num_features}")
print(f"event        = {event.shape}")
print(f"time         = {time.shape}")
print(f"batch size   = {BATCH_SIZE}")

# %%

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(74, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(250, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        return logit

cox_model = NeuralNetwork()

# %%

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features),
            torch.nn.Linear(74, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 250),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(250, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        return logit

cox_model = NeuralNetwork()

# %%

u.set_random_seed(1)

train_losses = []
val_losses = []

val_con_inds = []
val_con_ind_ipcws = []
train_con_ind_ipcws = []


# %% Define learning rate, epoch and optimizer

LEARNING_RATE = 5e-5
EPOCHS = 500
optimizer = torch.optim.AdamW(cox_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
con = ConcordanceIndex()

# %%

best_ind = 0
best_ipcw_ind = 0
best_epoch = 0
best_loss = -1
best_model = NeuralNetwork()
train_times = []

def train_loop(dataloader, model, optimizer):
    global train_times
    
    model.train()
    
    curr_loss = torch.tensor(0.0)
    weight = 0
    start_time = ttime()
    for i, batch in enumerate(dataloader):
        x, (event, time) = batch
        optimizer.zero_grad()
        log_hz = model(x)
                
        loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        loss.backward()
        optimizer.step()
        curr_loss += loss.detach() * len(x)/BATCH_SIZE
            
        weight += len(x)/BATCH_SIZE
        
    optimizer.zero_grad()
    train_times.append(ttime()-start_time)
    
    curr_loss /= weight
    
    model.eval()
    with torch.no_grad():
        pred = model(train_x)
        weight_ipcw = get_ipcw(train_event, train_time, train_time)
        con_ind_ipcw = con(pred.float(), train_event, train_time.float(), weight = weight_ipcw)
    
    return curr_loss, con_ind_ipcw

def val_loop(model, epoch):
    global best_ind, best_ipcw_ind, best_epoch, best_loss, best_model
    
    optimizer.zero_grad()
    model.eval()
    
    curr_con_ind = torch.tensor(0.0)
    curr_con_ind_ipcw = torch.tensor(0.0)
    curr_loss = torch.tensor(0.0)
    
    x, event, time = val_x, val_event, val_time
    
    with torch.no_grad():
        pred = model(x)
        
        loss = neg_partial_log_likelihood(pred, event, time, reduction="mean")
        curr_loss += loss.detach()
        
        con_ind = con(pred, event, time)
        curr_con_ind = con_ind
        
        weight_ipcw = get_ipcw(train_event, train_time, time)
        con_ind_ipcw = con(pred.float(), event, time.float(), weight = weight_ipcw)
        curr_con_ind_ipcw = con_ind_ipcw
            
        if curr_con_ind_ipcw > best_ipcw_ind:
            best_ind = curr_con_ind
            best_ipcw_ind = curr_con_ind_ipcw
            best_loss = loss
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
            best_epoch = epoch+1

    return curr_loss, curr_con_ind, curr_con_ind_ipcw

# %%Iterate through Train and Test loops

for t in tqdm(range(EPOCHS)):
    curr_train_loss, curr_train_con_ind_ipcw = train_loop(dataloader_train, cox_model, optimizer)
    curr_val_loss, curr_val_con_ind, curr_val_con_ind_ipcw = val_loop(cox_model, t)
    
    train_losses.append(curr_train_loss)
    val_losses.append(curr_val_loss)
    
    u.adjust_learning_rate(optimizer, val_losses[-5:], t, initial_lr=optimizer.param_groups[0]['lr'], decay_factor=0.5, epoch_interval=20, min_lr=1e-6)
    
    val_con_inds.append(curr_val_con_ind)
    val_con_ind_ipcws.append(curr_val_con_ind_ipcw)
    train_con_ind_ipcws.append(curr_train_con_ind_ipcw)
    
    
    if t==EPOCHS-1 or t % (EPOCHS // (EPOCHS // 50)) == 0:
        print(f"\nEpoch {t+1}\n-------------------------------")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:0.3e}")
        print(f"Training loss: {curr_train_loss:0.6f}, Validation loss: {curr_val_loss:0.6f}")
        print(f"Concordance Index validation:  {curr_val_con_ind:0.4f}, IPCW Concordance Index validation:  {curr_val_con_ind_ipcw:0.4f}")
print('\n' + '-'*50)
print("Done!")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_loss:0.6f}")
print(f"Best Concordance Index: {best_ind:0.4f}")
print(f"Best IPCW Concordance Index: {best_ipcw_ind:0.4f}")
print('-'*50)

# %% Plot the training and test losses

title = "Dropout probability 50%"
ns = 0
ne = 3000

plt.figure()
u.plot_losses(train_losses, val_losses, title, norm = True, ran = [ns, ne])

plt.figure()
#plt.xlim((0,100))
x_axis = np.linspace(1, len(val_con_ind_ipcws), len(val_con_ind_ipcws))
plt.scatter(x_axis[ns:ne], [val.cpu() for val in train_con_ind_ipcws][ns:ne], label="train", color = 'C0', s = 20)
plt.scatter(x_axis[ns:ne], [val.cpu() for val in val_con_ind_ipcws][ns:ne], label="test", color = 'C1', s = 20)
plt.xlabel("Epochs")
plt.ylabel("IPCW Index")
plt.title(title)
plt.yscale("log")
plt.legend()
plt.show()

# %%

'''
Dropout probablity 70%:
    --------------------------------------------------
    Best Epoch: 100
    Best Validation Loss: 5.879705
    Best Concordance Index: 0.7204
    Best IPCW Concordance Index: 0.6942
    --------------------------------------------------
    
    Comment: IPCW index still going up at the end
    
Dropout probablity 60%:
    --------------------------------------------------
    Best Epoch: 89
    Best Validation Loss: 5.864575
    Best Concordance Index: 0.7238
    Best IPCW Concordance Index: 0.6948
    --------------------------------------------------
    
Dropout probablity 50%:
    --------------------------------------------------
    Best Epoch: 80
    Best Validation Loss: 5.864045
    Best Concordance Index: 0.7205
    Best IPCW Concordance Index: 0.6917
    --------------------------------------------------
    
Dropout probablity 40%:
    --------------------------------------------------
    Best Epoch: 84
    Best Validation Loss: 5.861908
    Best Concordance Index: 0.7222
    Best IPCW Concordance Index: 0.6963
    --------------------------------------------------
    
Dropout probablity 30%:
    --------------------------------------------------
    Best Epoch: 96
    Best Validation Loss: 5.863719
    Best Concordance Index: 0.7203
    Best IPCW Concordance Index: 0.6966
    --------------------------------------------------
    
Dropout probablity 20%:
    --------------------------------------------------
    Best Epoch: 16
    Best Validation Loss: 5.852377
    Best Concordance Index: 0.7258
    Best IPCW Concordance Index: 0.6962
    --------------------------------------------------
    
Dropout probablity 10%:
    --------------------------------------------------
    Best Epoch: 14
    Best Validation Loss: 5.851727
    Best Concordance Index: 0.7253
    Best IPCW Concordance Index: 0.6971
    --------------------------------------------------
    
Dropout probablity 0%:
    --------------------------------------------------
    Best Epoch: 11
    Best Validation Loss: 5.850536
    Best Concordance Index: 0.7259
    Best IPCW Concordance Index: 0.6974
    --------------------------------------------------
    
'''

# %%

'''
IPCW Concordance Index over 0.69 reached for:
    random seed = 1
    lr = 1e-4
    weight_decay = 0.1
    100 epochs
    
    torch.nn.BatchNorm1d(num_features),
    torch.nn.Linear(74, 250),
    torch.nn.Tanh(),
    torch.nn.Dropout(p=0.7),
    torch.nn.Linear(250, 250),
    torch.nn.Tanh(),
    torch.nn.Dropout(p=0.7),
    torch.nn.Linear(250, 250),
    torch.nn.Tanh(),
    torch.nn.Dropout(p=0.7),
    torch.nn.Linear(250, 250),
    torch.nn.Tanh(),
    torch.nn.Dropout(p=0.7),
    torch.nn.Linear(250, 1),
    
'''



































































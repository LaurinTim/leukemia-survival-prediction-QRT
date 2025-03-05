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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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
import model_2nn2_utils as u

# %% Check if CUDA cores are available for training, if yes set the batch size to 128, otherwise 32

u.set_random_seed(1)

BATCH_SIZE, device = u.get_device()
torch.set_default_device(device)
       
# %%

dat = u.DataPrep(file_status, file_clinical, file_molecular)

# %%

a = u.DatasetGen(dat.status_arr, dat.clinical_arr, dat.molecular_arr, dat.effects_survival_map, dat.molecular_void_ids, 
               status_transformer = u.TransStatus(), clinical_transformer = u.TransClinical(), molecular_transformer = u.TransMolecular())

#The patient with the most somatic mutations has 17
max_mutations = 17

# %%

#u.set_random_seed(101)

def custom_collate(batch):
    # Assume each item in the batch is a tuple: (features, label)
    # And assume features is a tuple (clinical_tensor, molecular_tensor)
    clinical_list = []
    molecular_list = []
    labels = []
    
    for item in batch:
        # For example, if your __getitem__ returns (clinical_features, molecular_features) as one tensor,
        # and label as second element, adjust accordingly.
        features, label = item
        # Suppose you have split clinical and molecular features beforehand.
        # Here, let's assume clinical features are fixed and molecular features are variable.
        clinical, molecular = features[0], features[1]
        clinical_list.append(clinical)
        molecular_list.append(molecular)
        while len(molecular_list[0])<max_mutations:
            molecular_list[0] = torch.cat((molecular_list[0], torch.zeros((1,len(molecular_list[0][0])))), 0)
        labels.append(label)
    
    # Stack clinical features since they are fixed-size:
    clinical_batch = torch.stack(clinical_list)
    # Pad molecular features (each is a tensor of shape [num_mutations, feature_dim]):
    padded_molecular = pad_sequence(molecular_list, batch_first=True, padding_value=0)
    labels = torch.stack([torch.tensor(l) for l in labels])
    
    return (clinical_batch, padded_molecular), labels

# %%

train_data, val_data, test_data = torch.utils.data.random_split(a, [0.6, 0.2, 0.2], generator=torch.Generator(device=device).manual_seed(17))

dataloader_train = DataLoader(train_data, batch_size = BATCH_SIZE, collate_fn=custom_collate)
dataloader_val = DataLoader(val_data, batch_size = BATCH_SIZE)
dataloader_test = DataLoader(test_data, batch_size = BATCH_SIZE)

train_time = torch.tensor([val[1][1] for val in train_data])
val_time = torch.tensor([val[1][1] for val in val_data])
test_time = torch.tensor([val[1][1] for val in test_data])

bad_seed = bool(train_time.max() < val_time.max() or train_time.max() < test_time.max())
if bad_seed:
    print('-'*100)
    print('ERROR: THIS IS A BAD SEED, THE MAX TIME IN TRAIN IS SMALLER THAN IN VAL OR TEST, PLEASE SPLIT DATA AGAIN')
    print(train_time.max(), val_time.max(), test_time.max())
    print('-'*100)

# %%

train_d = next(iter(DataLoader(train_data, batch_size=len(train_data), collate_fn=custom_collate)))
train_xc = train_d[0][0]
train_xm = train_d[0][1]
train_event = train_d[1][:,0].bool()
train_time = train_d[1][:,1]

val_d = next(iter(DataLoader(val_data, batch_size=len(val_data), collate_fn=custom_collate)))
val_xc = val_d[0][0]
val_xm = val_d[0][1]
val_event = val_d[1][:,0].bool()
val_time = val_d[1][:,1]

test_d = next(iter(DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate)))
test_xc = test_d[0][0]
test_xm = test_d[0][1]
test_event = test_d[1][:,0].bool()
test_time = test_d[1][:,1]

# %%

mut_num_train = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in train_xm])
mut_num_val = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in val_xm])
mut_num_test = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in test_xm])

# %%

dataloader_all = DataLoader(a, batch_size=BATCH_SIZE, collate_fn=custom_collate)
all_d = next(iter(DataLoader(a, batch_size=len(a), collate_fn=custom_collate)))
all_xc = all_d[0][0]
all_xm = all_d[0][1]
all_event = all_d[1][:,0].bool()
all_time = all_d[1][:,1]
mut_num_all = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in all_xm])

mut_num_all_batches = []
for i, batch in enumerate(dataloader_all):
    curr_data, _ = batch
    curr_mol = curr_data[1]
    curr_scaling = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in curr_mol])
    #scaling_train[i] = curr_scaling
    mut_num_all_batches.append(curr_scaling)

# %%

mut_num_train_batches = []
for i, batch in enumerate(dataloader_train):
    curr_data, _ = batch
    curr_mol = curr_data[1]
    curr_scaling = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in curr_mol])
    #scaling_train[i] = curr_scaling
    mut_num_train_batches.append(curr_scaling)

# %%

def masked_mean_pooling(embeddings, lengths):
    """
    embeddings: Tensor of shape [batch_size, max_mutations, feature_dim]
    lengths: Tensor of shape [batch_size] with the actual number of mutations per patient
    """
    # Create a mask of shape [batch_size, max_mutations]
    max_len = embeddings.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).float()  # shape: [batch_size, max_mutations, 1]
    
    # Sum the embeddings, taking only valid (non-padded) entries into account
    sum_embeddings = (embeddings * mask).sum(dim=1)
    
    # Divide by the actual lengths (unsqueezed to match dimensions)
    mean_embeddings = sum_embeddings / lengths.unsqueeze(-1).float()
    return mean_embeddings

# %%

dropout_prob = 0.1

# Branch to process clinical data (one row per patient)
class ClinicalBranch(torch.nn.Module):
    def __init__(self):
        super(ClinicalBranch, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(train_xc.size(1)),
            torch.nn.Linear(train_xc.size(1), 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(100, 100)
        )
        
    def forward(self, x):
        # x shape: (batch_size, clinical_input_dim)
        return self.fc(x)

# Branch to process molecular data (multiple mutations per patient)
class MolecularBranch(torch.nn.Module):
    def __init__(self):
        super(MolecularBranch, self).__init__()
        self.mutation_encoder = torch.nn.Sequential(
            torch.nn.Linear(train_xm.size(2), 160),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(160, 160)
        )
        
    def forward(self, x, scaling):
        """
        x: a tensor of shape (batch_size, num_mutations, mutation_input_dim)
           For patients with no mutations, x can be a tensor of shape (batch_size, 0, mutation_input_dim)
        """        
        batch_size = x.size(0)
        if x.size(1) == 0:
            # If no mutations, return a zero vector for each patient.
            # Alternatively, you could return a learned "no mutation" embedding.
            return torch.zeros(batch_size, self.mutation_encoder[-1].out_features)
        else:
            # Process each mutation with the same encoder.
            # The encoder is applied to the last dimension.
            mutation_features = self.mutation_encoder(x)
            # Use a permutation-invariant pooling (e.g., average pooling) over the mutations.
            pooled = mutation_features.mean(dim=1) #* scaling
            #pooled = masked_mean_pooling(mutation_features, scaling)
            #pooled = torch.nan_to_num(pooled, nan=0.0)
            #print(pooled)
            #pooled.numpy()
            return pooled

# Final model that merges clinical and molecular representations
class CombinedSurvivalModel(torch.nn.Module):
    def __init__(self):
        super(CombinedSurvivalModel, self).__init__()
        self.clinical_branch = ClinicalBranch()
        self.molecular_branch = MolecularBranch()
        
        # Combine the two branches and create a final risk score (log hazard)
        self.final_layers = torch.nn.Sequential(
            torch.nn.BatchNorm1d(260),
            torch.nn.Linear(260, 350),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_prob),
            #torch.nn.Linear(340, 400),
            #torch.nn.Tanh(),
            #torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(350, 1)  # Cox models output an unconstrained risk score
        )
        
    def forward(self, clinical_data, molecular_data, scaling):
        # clinical_data: (batch_size, clinical_input_dim)
        # molecular_data: (batch_size, num_mutations, mutation_input_dim)
        clinical_out = self.clinical_branch(clinical_data)
        molecular_out = self.molecular_branch(molecular_data, scaling)
        combined = torch.cat([clinical_out, molecular_out], dim=1)
        risk_score = self.final_layers(combined)
        return risk_score
    
cox_model = CombinedSurvivalModel()

# %%

u.set_random_seed(1)

train_losses = []
val_losses = []

val_con_inds = []
val_con_ind_ipcws = []
train_con_ind_ipcws = []


# %% Define learning rate, epoch and optimizer

LEARNING_RATE = 1e-5
EPOCHS = 100
optimizer = torch.optim.AdamW(cox_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
con = ConcordanceIndex()

# %%

best_ind = 0
best_ipcw_ind = 0
best_epoch = 0
best_loss = -1
best_model = CombinedSurvivalModel()
train_times = []

def train_loop(dataloader, model, optimizer):
    global train_times
    
    model.train()
    
    curr_loss = torch.tensor(0.0)
    weight = 0
    start_time = ttime()
    for i, batch in enumerate(dataloader):
        x, status = batch
        status = status.T
        event = status[0]
        time = status[1]
        optimizer.zero_grad()
        log_hz = model(x[0], x[1], mut_num_train_batches[i])
                
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
        pred = model(train_xc, train_xm, mut_num_train)
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
    
    xc, xm, event, time = val_xc, val_xm, val_event, val_time
    
    with torch.no_grad():
        pred = model(xc, xm, mut_num_val)
        
        loss = neg_partial_log_likelihood(pred, event, time, reduction="mean")
        curr_loss += loss.detach()
        
        con_ind = con(pred, event, time)
        curr_con_ind = con_ind
        
        weight_ipcw = get_ipcw(train_event, train_time, time)
        con_ind_ipcw = con(pred.float(), event, time.float(), weight = weight_ipcw)
        curr_con_ind_ipcw = con_ind_ipcw
            
        if curr_con_ind_ipcw >= best_ipcw_ind:
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
    
    u.adjust_learning_rate(optimizer, val_losses[-5:], t, initial_lr=optimizer.param_groups[0]['lr'], decay_factor=0.5, epoch_interval=10, min_lr=1e-5)
    
    val_con_inds.append(curr_val_con_ind)
    val_con_ind_ipcws.append(curr_val_con_ind_ipcw)
    train_con_ind_ipcws.append(curr_train_con_ind_ipcw)
    
    
    if t==EPOCHS-1 or t % (EPOCHS // (EPOCHS // 10)) == 0:
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

title = "Dropout probability 10%"
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

cox_model.eval()
with torch.no_grad():
    pred_test = cox_model(test_xc, test_xm, mut_num_test)
    #pred_test = best_model(test_xc, test_xm, mut_num_test)
    weight_test = get_ipcw(train_event, train_time, test_time)
    ind_test = float(con(pred_test, test_event, test_time))
    ipcw_test = float(con(pred_test.float(), test_event, test_time.float(), weight = weight_test))
    
print(ind_test, ipcw_test)

# %%

cox_model.eval()
with torch.no_grad():
    pred_val = cox_model(val_xc, val_xm, mut_num_val)
    #pred_test = best_model(test_xc, test_xm, mut_num_test)
    weight_val = get_ipcw(train_event, train_time, val_time)
    ind_val = float(con(pred_val, val_event, val_time))
    ipcw_val = float(con(pred_val, val_event, val_time, weight = weight_val))
    
print(ind_val, ipcw_val)

# %%

cox_model.eval()
with torch.no_grad():
    pred_train = cox_model(train_xc, train_xm, mut_num_train)
    #pred_test = best_model(test_xc, test_xm, mut_num_test)
    weight_train = get_ipcw(train_event, train_time, train_time)
    ind_train = float(con(pred_train, train_event, train_time))
    ipcw_train = float(con(pred_train, train_event, train_time, weight = weight_train))
    
print(ind_train, ipcw_train)

# %%

def train_all_loop(dataloader, model, optimizer):
    global train_times
    
    model.train()
    
    curr_loss = torch.tensor(0.0)
    weight = 0

    for i, batch in enumerate(dataloader):
        x, status = batch
        status = status.T
        event = status[0]
        time = status[1]
        optimizer.zero_grad()
        log_hz = model(x[0], x[1], mut_num_all_batches[i])
                
        loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        loss.backward()
        optimizer.step()
        curr_loss += loss.detach() * len(x)/BATCH_SIZE
            
        weight += len(x)/BATCH_SIZE
        
    optimizer.zero_grad()
    
    curr_loss /= weight
    
    model.eval()
    with torch.no_grad():
        pred = model(all_xc, all_xm, mut_num_all)
        weight_ipcw = get_ipcw(all_event, all_time, all_time)
        con_ind_ipcw = con(pred.float(), all_event, all_time.float(), weight = weight_ipcw)
    
    return curr_loss, con_ind_ipcw

# %%

all_losses = []
all_con_ind_ipcws = []

for t in tqdm(range(EPOCHS)):
    curr_all_loss, curr_all_con_ind_ipcw = train_all_loop(dataloader_train, cox_model, optimizer)
    
    all_losses.append(curr_all_loss)
    
    u.adjust_learning_rate(optimizer, val_losses[-5:], t, initial_lr=optimizer.param_groups[0]['lr'], decay_factor=0.5, epoch_interval=10, min_lr=1e-5)
    
    all_con_ind_ipcws.append(curr_all_con_ind_ipcw)
    
    
    if t==EPOCHS-1 or t % (EPOCHS // (EPOCHS // 10)) == 0:
        print(f"\nEpoch {t+1}\n-------------------------------")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:0.3e}")
        print(f"Training loss: {curr_all_loss:0.6f}, Training IPCW Index: {curr_all_con_ind_ipcw:0.6f}")
print('\n' + '-'*50)
print("Done!")
print(f"Final Loss: {all_losses[-1]:0.6f}")
print(f"Final IPCW Concordance Index: {all_con_ind_ipcws[-1]:0.4f}")
print('-'*50)

# %%

title = "Training with all data"
ns = 0
ne = 3000
x_axis = np.linspace(1, len(val_con_ind_ipcws), len(val_con_ind_ipcws))

plt.figure()
plt.scatter(x_axis[ns:ne], [val.cpu() for val in all_losses][ns:ne], color="C0")
plt.xlabel("Epochs")
plt.ylabel("Loss [log]")
plt.title(title)
plt.yscale("log")
plt.legend()
plt.show()

plt.figure()
#plt.xlim((0,100))
plt.scatter(x_axis[ns:ne], [val.cpu() for val in train_con_ind_ipcws][ns:ne], label="train", color = 'C0', s = 20)
plt.xlabel("Epochs")
plt.ylabel("IPCW Index")
plt.title(title)
plt.yscale("log")
plt.legend()
plt.show()

# %%

clinical_arr_sub, molecular_arr_sub, molecular_void_ids_sub, patient_ids_sub, patient_num_sub = dat.submission_data_prep()
X_clinical_sub, X_molecular_sub = a.get_submission_data(clinical_arr_sub, molecular_arr_sub, molecular_void_ids_sub, patient_ids_sub, patient_num_sub)
sub_dataset = [[(val,bal), kal] for val,bal,kal in zip(X_clinical_sub, X_molecular_sub, np.zeros(patient_num_sub))]
sub_dataset = custom_collate(sub_dataset)

sub_xc = sub_dataset[0][0]
sub_xm = sub_dataset[0][1]

mut_num_sub = torch.tensor([max(1,len([val for val in bal if val.sum()!=0])) for bal in sub_xm])

# %%

cox_model.eval()
with torch.no_grad():
    pred_sub = cox_model(sub_xc, sub_xm, mut_num_sub)
pred_sub = pred_sub.cpu().numpy().reshape(1,-1)[0]
sub_sort_index = [float(val[3:]) for val in patient_ids_sub]
sub_df = pd.DataFrame([patient_ids_sub, pred_sub, sub_sort_index], index=["ID","risk_score","sort_index"], columns=np.arange(patient_num_sub)).transpose()
sub_df = sub_df.sort_values(by=["sort_index"]).reset_index(drop=True)
sub_df = sub_df.drop(columns=["sort_index"])
sub_df.to_csv(data_dir + "\\submission_files\\pytorch_2nn2_v0.csv", index=False)

# %%

'''
With all clinical features and chromosome embeddings:
    --------------------------------------------------
    Best Epoch: 50
    Best Validation Loss: 5.876966
    Best Concordance Index: 0.7130
    Best IPCW Concordance Index: 0.7022
    --------------------------------------------------

Without WBC, AND, MONOCYTES:
    --------------------------------------------------
    Best Epoch: 48
    Best Validation Loss: 5.851595
    Best Concordance Index: 0.7235
    Best IPCW Concordance Index: 0.7052
    --------------------------------------------------
    
Without chromosome embeddings:
    --------------------------------------------------
    Best Epoch: 44
    Best Validation Loss: 5.875919
    Best Concordance Index: 0.7125
    Best IPCW Concordance Index: 0.7027
    --------------------------------------------------
    
Without chromosome embeddings, WBC, AND, MONOCYTES:
    --------------------------------------------------
    Best Epoch: 29
    Best Validation Loss: 5.870313
    Best Concordance Index: 0.7199
    Best IPCW Concordance Index: 0.6922
    --------------------------------------------------
    
Multiplying the median life expactancy from the effect with VAF and no chromosome embeddings:
    --------------------------------------------------
    Best Epoch: 34
    Best Validation Loss: 5.853043
    Best Concordance Index: 0.7229
    Best IPCW Concordance Index: 0.7057
    --------------------------------------------------
    
With the number of somatic mutations added to the training data:
    --------------------------------------------------
    Best Epoch: 6
    Best Validation Loss: 5.894910
    Best Concordance Index: 0.7248
    Best IPCW Concordance Index: 0.7088
    --------------------------------------------------
    
With the number of somatic mutations added to the training data and multiplying the median life expactancy from the effect with VAF:
    --------------------------------------------------
    Best Epoch: 5
    Best Validation Loss: 5.936570
    Best Concordance Index: 0.7252
    Best IPCW Concordance Index: 0.7088
    --------------------------------------------------
    
'''




































































import warnings
warnings.filterwarnings("ignore")

import lifelines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Our package
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood, log_hazard, survival_function
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator
from torchsurv.stats.ipcw import get_ipcw

# PyTorch boilerplate - see https://github.com/Novartis/torchsurv/blob/main/docs/notebooks/helpers_introduction.py
#from helpers_introduction import Custom_dataset, plot_losses

# %%

# Constant parameters accross models
# Detect available accelerator; Downgrade batch size if only CPU available
if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
    print("CUDA-enabled GPU/TPU is available.")
    BATCH_SIZE = 128  # batch size for training
else:
    print("No CUDA-enabled GPU found, using CPU.")
    BATCH_SIZE = 32  # batch size for training

EPOCHS = 100
LEARNING_RATE = 1e-2

# %%

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\Challenge Data QRT"

features = ['BM_BLAST', 'HB', 'PLT']

# %%
    
class TransStatus(object):
    def __call__(self, sample):
        #res = np.array([(sample[0], sample[1])], dtype = [('s', bool), ('y', float)])
        res = np.array(sample)
        #res = np.array([(sample[0], sample[1])], dtype = [('status', bool), ('years', float)], copy=True)
        #res = np.array(sample)
        return res#torch.tensor(res, dtype=torch.float32)
        
class TransClinical(object):
    def __call__(self, sample):
        res = np.array(sample.loc[sample.ID!='n', features])
        return torch.tensor(res[0]).float()

# %%

class DatasetGen(Dataset):
    def __init__(self, annotations_file, clinical_file, molecular_file, clinical_transform=None, molecular_transform=None, status_transform=None):
        self.patient_status = pd.read_csv(annotations_file).dropna(subset=['OS_YEARS', 'OS_STATUS'])
        self.patient_clinical = pd.read_csv(clinical_file)
        self.patient_molecular = pd.read_csv(molecular_file)
        self.patient_clinical = self.patient_clinical.loc[self.patient_clinical['ID'].isin(self.patient_status['ID'])]
        self.patient_clinical = self.patient_clinical.fillna(0)
        self.patient_molecular = self.patient_molecular.loc[self.patient_molecular['ID'].isin(self.patient_status['ID'])]
        self.clinical_transform = clinical_transform
        self.molecular_transform = molecular_transform
        self.status_transform = status_transform
        
    def __len__(self):
        return len(self.patient_status)
    
    def __getitem__(self, idx):
        patient_id = self.patient_status.iloc[idx, 0]
        os_years = self.patient_status.iloc[idx, 1]
        os_status = self.patient_status.iloc[idx, 2]
        status = np.array([os_status, os_years])
        info_clinical = self.patient_clinical.loc[self.patient_clinical.ID == patient_id]
        #info_molecular = self.patient_molecular.iloc[idx]
        #info = pd.concat([info_clinical, info_molecular])
        if self.clinical_transform:
            info_clinical = self.clinical_transform(info_clinical)
            #info_molecular = self.molecular_transform(info_molecular)
        if self.status_transform:
            status = self.status_transform(status)
        return info_clinical, (bool(status[0]), status[1])
    
    def get_data(self):
        temp_list = []
        
        for i in range(len(self.patient_status)):
            temp_val = self.__getitem__(i)
            temp_list += [temp_val]
            
        return temp_list

# %%

complete_data = DatasetGen(data_dir+'\\target_train.csv', data_dir+'\\X_train\\clinical_train.csv', data_dir+'\\X_train\\molecular_train.csv', 
                  clinical_transform=TransClinical(), status_transform=TransStatus())

train_data, val_data, test_data = torch.utils.data.random_split(complete_data, [0.7, 0.0, 0.3])

dataloader_train = DataLoader(train_data, batch_size = BATCH_SIZE)
dataloader_val = DataLoader(val_data, batch_size = BATCH_SIZE)
dataloader_test = DataLoader(test_data, batch_size = 317)

train_data_val = train_data.dataset.get_data()

# %%

# Sanity check
x, (event, time) = next(iter(dataloader_train))
num_features = x.size(1)

print(f"x (shape)    = {x.shape}")
print(f"num_features = {num_features}")
print(f"event        = {event.shape}")
print(f"time         = {time.shape}")

# %%

cox_model = torch.nn.Sequential(
    torch.nn.BatchNorm1d(num_features),  # Batch normalization
    torch.nn.Linear(num_features, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(64, 1),  # Estimating log hazards for Cox models
)

# %%

LEARNING_RATE = 1e-2
EPOCHS = 50
optimizer = torch.optim.Adam(cox_model.parameters(), lr=LEARNING_RATE)
con = ConcordanceIndex()

# %%

train_event = torch.tensor([val[1][0] for val in train_data_val]).bool()
train_time = torch.tensor([val[1][1] for val in train_data_val]).float()

#weight_ipcw = get_ipcw(train_event, train_time)

# %%

def train_loop(dataloader, model, optimizer):
    model.train()

    curr_loss = torch.tensor(0.0)
    for i, batch in enumerate(dataloader):
        x, (event, time) = batch
        optimizer.zero_grad()
        log_hz = model(x)
        
        loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        loss.backward()
        optimizer.step()
        curr_loss += loss.detach()

    return curr_loss
            
def test_loop(dataloader, model):
    batch_num = len(dataloader)
    test_con_ind = 0
    test_con_ind_ipcw = 0
    model.eval()

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            x, (event, time) = batch
            pred = cox_model(x)
            con_ind = con(pred, event, time)
            test_con_ind += con_ind
            weight_ipcw = get_ipcw(train_event, train_time, torch.tensor([val[0] for val in pred]).float())
            con_ind_ipcw = con(pred.float(), event, time.float(), weight = weight_ipcw)
            test_con_ind_ipcw += con_ind_ipcw

    test_con_ind /= batch_num
    test_con_ind_ipcw /= batch_num
    return test_con_ind, test_con_ind_ipcw

# %%

for t in range(EPOCHS):
    curr_train_loss = train_loop(dataloader_train, cox_model, optimizer)
    curr_con_ind, curr_con_ind_ipcw = test_loop(dataloader_test, cox_model)
    if t % (EPOCHS // 10) == 0:
        print(f"Epoch {t+1}\n-------------------------------")
        print(f"Training loss: {curr_train_loss:0.2f}, Concordance Index: {curr_con_ind:0.3f}, IPCW Concordance Index: {curr_con_ind_ipcw:0.3f}\n")
print("Done!")

# %%

#torch.manual_seed(42)

# Init optimizer for Cox
optimizer = torch.optim.Adam(cox_model.parameters(), lr=LEARNING_RATE)

# Initiate empty list to store the loss on the train and validation sets
train_losses = []
val_losses = []

# training loop
for epoch in range(EPOCHS):
    epoch_loss = torch.tensor(0.0)
    for i, batch in enumerate(dataloader_train):
        x, (event, time) = batch
        optimizer.zero_grad()
        log_hz = cox_model(x)  # shape = (16, 1)
        a,b,c = log_hz, event, time
        r = x
        #break
        loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach()

    if epoch % (EPOCHS // 10) == 0:
        print(f"Epoch: {epoch:03}, Training loss: {epoch_loss:0.2f}")

    # Reccord loss on train and test sets
    epoch_loss /= i + 1
    train_losses.append(epoch_loss)
    with torch.no_grad():
        x, (event, time) = next(iter(dataloader_val))
        val_losses.append(
            neg_partial_log_likelihood(cox_model(x), event, time, reduction="mean")
        )

# %%

con = ConcordanceIndex()

batch_num = len(dataloader_test)
test_con_ind = 0

for i, batch in enumerate(dataloader_test):
    with torch.no_grad():
        x, (event, time) = batch
        pred = cox_model(x)
        con_ind = con(pred, event, time)
        test_con_ind += con_ind

test_con_ind /= batch_num    
print(f'Concordance index of test data: {test_con_ind:0.3f}')

# %%

optimizer = torch.optim.Adam(cox_model.parameters(), lr=LEARNING_RATE)
con = ConcordanceIndex()
torch.manual_seed(42)

for i in range(EPOCHS):
    size = len(complete_data.patient_status)*0.5
    epoch_loss = torch.tensor(0.0)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    #cox_model.train()
    for k, batch in enumerate(dataloader_train):
        X, (temp_event, temp_time) = batch
        optimizer.zero_grad()
        pred = cox_model(X)
        x,y,z = pred, temp_event, temp_time
        t = X
        
        loss = neg_partial_log_likelihood(pred, temp_event, temp_time, reduction="mean")
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach()
        #optimizer.zero_grad()
    
    if i % 20 == 0:
        loss, current = loss, k*len(X)+len(X)
        conc = con(pred, temp_event, temp_time)
        print(f"Epoch: {i:03}, Concordance Index: {conc:>5f}, Epoch loss: {epoch_loss:0.2f}")
    
















































































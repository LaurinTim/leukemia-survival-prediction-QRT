import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torchvision.models as models
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw
from torchsurv.loss import cox
import warnings
warnings.filterwarnings('ignore')

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

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
        return torch.tensor(res[0], dtype=torch.float32)

# %%

class DatasetGen(Dataset):
    def __init__(self, annotations_file, clinical_file, molecular_file, clinical_transform=None, molecular_transform=None, status_transform=None):
        self.patient_status = pd.read_csv(annotations_file).dropna(subset=['OS_YEARS', 'OS_STATUS'])
        self.patient_clinical = pd.read_csv(clinical_file)
        self.patient_molecular = pd.read_csv(molecular_file)
        self.patient_clinical = self.patient_clinical.loc[self.patient_clinical['ID'].isin(self.patient_status['ID'])]
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
        return info_clinical, status

# %%

complete_data = DatasetGen(data_dir+'\\target_train.csv', data_dir+'\\X_train\\clinical_train.csv', data_dir+'\\X_train\\molecular_train.csv', 
                  clinical_transform=TransClinical(), status_transform=TransStatus())

train_data, test_data = torch.utils.data.random_split(complete_data, [0.7, 0.3])

train_status = [val[1] for val in train_data]
train_event = torch.tensor([val[0] for val in train_status]).bool()
train_time = torch.tensor([val[1] for val in train_status]).float()

test_status = [val[1] for val in test_data]
test_event = torch.tensor([val[0] for val in test_status]).bool()
test_time = torch.tensor([val[1] for val in test_status]).float()

# %%

train_data, test_data = torch.utils.data.random_split(complete_data, [0.7, 0.3])

train_status = [val[1].numpy() for val in train_data]
train_status = np.array([np.array([(val[0],max(val[1],0.001))], dtype = [('s',bool),('y',float)]) for val in train_status])
#train_status = [val[0] for val in train_status]

test_status = [val[1].numpy() for val in test_data]
test_status = np.array([np.array([(val[0],max(val[1],0.001))], dtype = [('s',bool),('y',float)]) for val in test_status])
test_data_status = [val[0] for val in test_status]

# %%

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            torch.nn.BatchNorm1d(3),
            nn.Linear(3, 1)
        )
        ''',
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )'''

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# %%

class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_train, target, pred):
        #print(max(input_train['y']), max(target['y']))
        print(target)
        loss = concordance_index_ipcw(input_train, target, pred)[0]
        return loss
    
my_loss = custom_loss()

# %%

con = ConcordanceIndex()

class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, event, time, pred):
        w = get_ipcw(event, time)
        loss = con(pred, event, time, weight = w)
        return loss
    
my_loss = custom_loss()

# %%

con = ConcordanceIndex()

class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, event, time, pred):
        loss = cox.neg_partial_log_likelihood(pred, event, time, reduction = 'mean')
        return loss
    
my_loss = custom_loss()

# %%

model = torch.nn.Sequential(
    torch.nn.BatchNorm1d(3),  # Batch normalization
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(64, 1),  # Estimating log hazards for Cox models
)

# %%

def train_loop(dataloader, model, my_loss, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = torch.nan_to_num(model(X))
        #pred = torch.tensor([val[0] for val in pred]).float()
        #return pred
        temp_event = torch.tensor([val[0] for val in y]).bool()
        temp_time = torch.tensor([val[1] for val in y]).float()
        #return temp_time
        loss = my_loss(temp_event, temp_time, pred)
        #loss = cox.neg_partial_log_likelihood(pred, temp_event, temp_time)

        # Backpropagation
        loss.backward()
        optimizer.step()
        #optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss, batch * batch_size + len(X)
            conc = con(pred, temp_event, temp_time)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], Concordance Index: {conc:>5f}")
            
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = torch.nan_to_num(model(X))
            #pred = torch.tensor([val[0] for val in pred]).float()
            
            temp_event = torch.tensor([val[0] for val in y]).bool()
            temp_time = torch.tensor([val[1] for val in y]).float()
            
            test_loss += my_loss(temp_event, temp_time, pred)

    test_loss /= num_batches
    conc = con(pred, temp_event, temp_time)
    print(f"Test Error: Avg loss: {test_loss:>8f}, Concordance Index: {conc:>5f} \n")

# %%

batch_size = 2222
learning_rate = 1e-3
epochs = 3

# %%

train_dataloader = DataLoader(train_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, my_loss, optimizer)
    test_loop(test_dataloader, model, my_loss)
print("Done!")

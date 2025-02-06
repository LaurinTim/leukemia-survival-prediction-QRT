#Same as cox_model.py but now use 4 features, as an additional feature the number of somatic mutations, call this NSM

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.utils.data import DataLoader, Dataset

# Our package
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

#IPCW Concordance Index from sksurv
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\Challenge Data QRT"

features = ['BM_BLAST', 'HB', 'PLT', 'NSM']

#set working directory to data_dir
import os
os.chdir(data_dir)

#import method to create test_results file
from create_test_results_file import test_results

# %% Check if CUDA cores are available for training, if yes set the batch size to 128, otherwise 32

# Constant parameters accross models
# Detect available accelerator; Downgrade batch size if only CPU available
if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
    print("CUDA-enabled GPU/TPU is available.")
    BATCH_SIZE = 128  # batch size for training
    #torch.set_default_device('cuda')
else:
    print("No CUDA-enabled GPU found, using CPU.")
    BATCH_SIZE = 32  # batch size for training
    #device = torch.device('cpu')

# %% Code from https://github.com/Novartis/torchsurv/blob/main/docs/notebooks/helpers_introduction.py, creates a scatter plot with normalized losses for the training and test data

def plot_losses(train_losses, test_losses, title: str = "Cox") -> None:
    x = np.linspace(1, len(train_losses), len(train_losses))
    
    train_losses = torch.stack(train_losses) / train_losses[0]
    test_losses = torch.stack(test_losses) / test_losses[0]

    plt.scatter(x, train_losses, label="training", color = 'C0')
    plt.scatter(x, test_losses, label="test", color = 'C1', s = 20)
    plt.xlabel("Epochs")
    plt.ylabel("Normalized loss")
    plt.title(title)
    plt.yscale("log")
    plt.show()

# %%

def compare_models(model1, model2):
    '''

    Parameters
    ----------
    model1 : pytorch model
        First model.
    model2 : pytorch model
        Second model, parameters need to have the same shape as for first model.

    Returns
    -------
    bool
        Returns True if the parameters of both models are identical, otherwise False.

    '''
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

# %%

def status_to_StructuredArray(data):
    '''

    Parameters
    ----------
    data : pytorch.tensor
        Some tensor where each element has at posision 1 information about the status (0 or 1/bool) and at position 2 a number of years (float).

    Returns
    -------
    arr : structured array [('status', '?'), ('years', '<f8')]
        Structured array with the status in bool at position 1 of each element and a number of years in float at position 2.

    '''
    arr = np.array([(bool(val[0]), float(val[1])) for val in data], dtype = [('status', bool), ('years', float)])
    
    return arr

# %% Transformers for DatasetGen
    
class TransStatus(object):
    def __call__(self, sample):
        #res = np.array([(sample[0], sample[1])], dtype = [('s', bool), ('y', float)])
        res = np.array(sample)
        #res = np.array([(sample[0], sample[1])], dtype = [('status', bool), ('years', float)], copy=True)
        #res = np.array(sample)
        return torch.tensor(res)#torch.tensor(res, dtype=torch.float32)
        
class TransClinical(object):
    def __call__(self, sample):
        res = np.array(sample.loc[sample.ID!='n', features[:-1]])
        return torch.tensor(res[0]).float()

class TransMolecular(object):
    def __call__(self, sample):
        return torch.tensor([len(sample)])

# %% Generate a custom dataset

class DatasetGen(Dataset):
    def __init__(self, annotations_file, clinical_file, molecular_file, clinical_transform=None, molecular_transform=None, status_transform=None):
        self.patient_status = pd.read_csv(annotations_file).dropna(subset=['OS_YEARS', 'OS_STATUS'])
        
        self.patient_clinical = pd.read_csv(clinical_file)
        self.patient_clinical = self.patient_clinical.loc[self.patient_clinical['ID'].isin(self.patient_status['ID'])]
        self.patient_clinical = self.patient_clinical.fillna(0)
        
        self.patient_molecular = pd.read_csv(molecular_file)
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
        info_molecular = self.patient_molecular[self.patient_molecular.ID == patient_id]
        
        if self.clinical_transform and self.molecular_transform:
            info_clinical = self.clinical_transform(info_clinical)
            info_molecular = self.molecular_transform(info_molecular)
            info = torch.cat((info_clinical, info_molecular))
            
        if self.status_transform:
            status = self.status_transform(status)
            
        return info, (bool(status[0]), status[1])

# %% Get dataset from the training data, split it into training and test, create dataloaders for both

complete_data = DatasetGen(data_dir+'\\target_train.csv', data_dir+'\\X_train\\clinical_train.csv', data_dir+'\\X_train\\molecular_train.csv', 
                  clinical_transform=TransClinical(), molecular_transform=TransMolecular(), status_transform=TransStatus())

train_data, val_data, test_data = torch.utils.data.random_split(complete_data, [0.7, 0.0, 0.3])#, generator=torch.Generator(device='cuda'))

dataloader_train = DataLoader(train_data, batch_size = BATCH_SIZE)
dataloader_val = DataLoader(val_data, batch_size = BATCH_SIZE)
dataloader_test = DataLoader(test_data, batch_size = BATCH_SIZE)

train_event = torch.tensor([val[1][0] for val in train_data]).bool()
train_time = torch.tensor([val[1][1] for val in train_data]).float()
train_status_arr = status_to_StructuredArray([val[1] for val in train_data])

test_x = torch.tensor([val[0].cpu().numpy() for val in test_data])
test_event = torch.tensor([val[1][0] for val in test_data])
test_time = torch.tensor([val[1][1] for val in test_data])
test_status_arr = status_to_StructuredArray([val[1] for val in test_data])

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
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        return logit
    '''
    def state_dict(self):
        state_dict_temp = super().state_dict()
        return state_dict_temp
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        return'''

cox_model = NeuralNetwork()

# %% Define learning rate, epoch and optimizer

LEARNING_RATE = 1e-2
EPOCHS = 50
optimizer = torch.optim.Adam(cox_model.parameters(), lr=LEARNING_RATE)

# %% Train and Test loops

#global best_ind, best_model
best_ind = 0
best_ipcw_ind = 0
best_ind_sk = 0
best_ipcw_ind_sk = 0
best_epoch = 0
best_model = NeuralNetwork()
#best_state = OrderedDict()

def train_loop(dataloader, model, optimizer):
    model.train()
    
    batch_num = len(dataloader)
    curr_con_ind = torch.tensor(0.0)
    curr_con_ind_ipcw = torch.tensor(0.0)
    curr_loss = torch.tensor(0.0)
    
    for i, batch in enumerate(dataloader):
        x, (event, time) = batch
        optimizer.zero_grad()
        log_hz = model(x)
        
        #print(log_hz.device, event.device, time.device)
        
        loss = neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        loss.backward()
        optimizer.step()
        curr_loss += loss.detach()
        
        con = ConcordanceIndex()
        con_ind = con(log_hz, event, time)
        curr_con_ind += con_ind
        
        con = ConcordanceIndex()
        try:
            weight_ipcw = get_ipcw(train_event, train_time, time)
        except:
            curr_con_ind_ipcw += 0
            print('ERROR FOR IPCW WEIGHTS IN TRAIN LOOP')
        else:
            con_ind_ipcw = con(log_hz.float(), event, time.float(), weight = weight_ipcw)
            curr_con_ind_ipcw += con_ind_ipcw
                        
    curr_loss /= batch_num
    curr_con_ind /= batch_num
    curr_con_ind_ipcw /= batch_num
    return curr_loss, curr_con_ind, curr_con_ind_ipcw

def test_loop(model, epoch):
    global best_ind, best_ipcw_ind, best_ind_sk, best_ipcw_ind_sk, best_epoch, best_model
    
    model.eval()
    
    curr_con_ind = torch.tensor(0.0)
    curr_con_ind_ipcw = torch.tensor(0.0)
    curr_loss = torch.tensor(0.0)
    
    #Indices calculated with sksurv metric
    curr_con_ind_sk = 0
    curr_con_ind_ipcw_sk = 0
    
    x, event, time = test_x, test_event, test_time
    
    with torch.no_grad():
        pred = model(x)
        
        loss = neg_partial_log_likelihood(pred, event, time, reduction="mean")
        curr_loss += loss.detach()
        
        con = ConcordanceIndex()
        con_ind = con(pred, event, time)
        curr_con_ind = con_ind
        
        con = ConcordanceIndex()
        try:
            weight_ipcw = get_ipcw(train_event, train_time, time)
        except:
            curr_con_ind_ipcw = 0
            print('ERROR FOR IPCW WEIGHTS IN TEST LOOP')
        else:
            con_ind_ipcw = con(pred.float(), event, time.float(), weight = weight_ipcw)
            curr_con_ind_ipcw = con_ind_ipcw
            
        curr_con_ind_sk = concordance_index_censored([val[0] for val in test_status_arr], [val[1] for val in test_status_arr], [val[0] for val in pred.detach().numpy()])[0]
        
        curr_con_ind_ipcw_sk = concordance_index_ipcw(train_status_arr, test_status_arr, [val[0] for val in pred.detach().numpy()])[0]
    
        if curr_con_ind_ipcw > best_ipcw_ind:
        #if curr_con_ind_ipcw_sk > best_ind_sk:
            #print(f"\nNew best model found, old Index: {best_ind:0.3f}, new Index: {curr_con_ind_ipcw:0.3f}")
            best_ind = curr_con_ind
            best_ipcw_ind = curr_con_ind_ipcw
            best_ind_sk = curr_con_ind_sk
            best_ipcw_ind_sk = curr_con_ind_ipcw_sk
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
            best_epoch = epoch+1
            
            if abs(best_ipcw_ind - best_ipcw_ind_sk) > 0.02:
                print("-"*100 + "ERROR" + "-"*100)
                print(f"Best IPCW Index with torchsurv: {best_ipcw_ind:0.3f}")
                print(f"Best IPCW Index with sksurv:    {best_ipcw_ind_sk:0.3f}")
        
        #else:
            #model.load_state_dict(copy.deepcopy(best_model.state_dict()))
            #print(f"\nNo new best model found, index to beat: {best_ind:0.3f}")
        
    
    #curr_loss /= num_el
    #curr_con_ind /= batch_num
    #curr_con_ind_ipcw /= batch_num
    return curr_loss, curr_con_ind, curr_con_ind_ipcw, curr_con_ind_sk, curr_con_ind_ipcw_sk

# %% Iterate through Train and Test loops

train_losses = []
test_losses = []

train_con_inds = []
test_con_inds = []
test_con_inds_sk = []

train_con_ind_ipcws = []
test_con_ind_ipcws = []
test_con_ind_ipcws_sk = []

for t in range(EPOCHS):
    curr_train_loss, curr_train_con_ind, curr_train_con_ind_ipcw = train_loop(dataloader_train, cox_model, optimizer)
    curr_test_loss, curr_test_con_ind, curr_test_con_ind_ipcw, curr_test_con_ind_sk, curr_test_con_ind_ipcw_sk = test_loop(cox_model, t)
    
    train_losses.append(curr_train_loss)
    test_losses.append(curr_test_loss)
    
    train_con_inds.append(curr_train_con_ind)
    test_con_inds.append(curr_test_con_ind)
    test_con_inds_sk.append(curr_test_con_ind_sk)
    
    train_con_ind_ipcws.append(curr_train_con_ind_ipcw)
    test_con_ind_ipcws.append(curr_test_con_ind_ipcw)
    test_con_ind_ipcws_sk.append(curr_test_con_ind_ipcw_sk)
    
    if t % (EPOCHS // 5) == 0:
        print(f"\nEpoch {t+1}, Index to beat: {best_ipcw_ind:0.3f} ({best_ipcw_ind_sk:0.3f}), Best Epoch: {best_epoch}\n-------------------------------")
        #print(f"Model is best model: {compare_models(cox_model, best_model)}")
        #print(f"torchsurv Index: {curr_test_con_ind:0.3f}, sksurv Index: {curr_test_con_ind_sk:0.3f}")
        print(f"Training loss: {curr_train_loss:0.3f}, Test loss: {curr_test_loss:0.3f}")
        print(f"Concordance Index train: {curr_train_con_ind:0.3f},         IPCW Concordance Index train: {curr_train_con_ind_ipcw:0.3f}")
        print(f"Concordance Index test:  {curr_test_con_ind:0.3f} ({curr_test_con_ind_sk:0.3f}), IPCW Concordance Index test:  {curr_test_con_ind_ipcw:0.3f} ({curr_test_con_ind_ipcw_sk:0.3f})")
        #print(f"Test IPCW Concordance Index with sksurv.metrics: {curr_test_con_ind_ipcw_sk:0.3f}")
print('\n' + '-'*50)
print(f"Done! The best epoch was {best_epoch}.")
print(f"The Concordance Index of the test data is: {best_ind:0.3f}, IPCW Concordance Index of the test data is: {best_ipcw_ind:0.3f}, Index using sksurv: {best_ipcw_ind_sk:0.3f}")
print('-'*50)

# %% Plot the training and test losses

plot_losses(train_losses, test_losses, "Cox")

# %% Lists with elements [CI, IPCW CI, IPCW CI SK] to see if it is better to copy state dict from best model or not and if yes if the metric from sksurv is better
#repeat both methods 5 times with 50 iterations

#set cox_model state dict to the one of best model each epoch, check for best model with torchsurv metric
a = [[0.731, 0.754, 0.692], [0.730, 0.751, 0.692], [0.728, 0.747, 0.691], [0.732, 0.754, 0.692], [0.728, 0.749, 0.695]]

#do not set cox_model state dict to the one of best model each epoch, check for best model with torchsurv metric
b = [[0.730, 0.745, 0.690], [0.726, 0.752, 0.692], [0.729, 0.746, 0.692], [0.727, 0.749, 0.694], [0.730, 0.740, 0.696]]

#do not set cox_model state dict to the one of best model each epoch, check for best model with sksurv metric
c = [[0.728, 0.747, 0.697], [0.728, 0.740, 0.695], [0.729, 0.744, 0.698], [0.729, 0.740, 0.691], [0.724, 0.743, 0.693]]

#set cox_model state dict to the one of best model each epoch, check for best model with sksurv metric
c = [[0.727, 0.743, 0.695], [0.727, 0.740, 0.695], [0.728, 0.740, 0.697], [0.728, 0.750, 0.695], [0.731, 0.748, 0.697]]

# %% !!!ONLY RUN IF THE MODEL SHOULD GET SAVED!!!

cox_model.load_state_dict(copy.deepcopy(best_model.state_dict()))
torch.save(cox_model.state_dict(), data_dir + '\\saved_models\\model32.pth')

# %% Check if the Conconcordance and IPCW indices obtained from the test_results method match the ones calculated during the training

test_model = NeuralNetwork()

df_test = pd.DataFrame(test_x, columns = features)
df_test.insert(0, "ID", [str(int(val)) for val in np.linspace(1, len(df_test), len(df_test))])

a = test_results(test_model, data_dir + "\\saved_models\\model32.pth", df_test, features, "model_temp", return_df = True)
pred_test = torch.tensor(a.risk_score).float()

con = ConcordanceIndex()
weight_test = get_ipcw(train_event, train_time, test_time)
ind_test = con(pred_test, test_event, test_time)
ind_ipcw_test = con(pred_test, test_event, test_time.float(), weight = weight_test)

print(f"Concordance Index and IPCW Concordance Index obtrained from data in test_data:\nConcordance Index: {ind_test:0.3f}\nIPCW Concordance Index: {ind_ipcw_test:0.3f}")

# %% Run the rest_results method and create a csv file for the submission

final_df = pd.read_csv(data_dir + "\\X_test\\clinical_test.csv")
final_df = final_df[['ID'] + features[:-1]]
final_mol_df = pd.read_csv(data_dir + "\\X_test\\molecular_test.csv")
final_df.insert(4, 'NSM', [len(final_mol_df[final_mol_df.ID == val]) for val in list(final_df.ID)])

test_model = NeuralNetwork()

test_results(test_model, data_dir + "\\saved_models\\model32.pth", final_df, features, "model32")


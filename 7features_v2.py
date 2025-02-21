#Same as 4features.py but now use 7 features, the additional features are WBC, ANC and MONOCYTES

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

from tqdm import tqdm

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

features = ['BM_BLAST', 'HB', 'PLT', 'WBC', 'ANC', 'MONOCYTES', 'NSM']

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
    BATCH_SIZE = 256  # batch size for training
    torch.set_default_device('cuda')
    device = 'cuda'
    #torch.set_default_device('cpu')
    #device = 'cpu'
else:
    print("No CUDA-enabled GPU found, using CPU.")
    BATCH_SIZE = 32  # batch size for training
    device = torch.device('cpu')

# %% Code from https://github.com/Novartis/torchsurv/blob/main/docs/notebooks/helpers_introduction.py, creates a scatter plot with normalized losses for the training and test data

def plot_losses(train_losses, test_losses, title: str = "Cox", norm = True, ran = None) -> None:
    if ran == None:
        x = np.linspace(1, len(train_losses), len(train_losses))
    
    else:
        train_losses = train_losses[ran[0]:ran[1]]
        test_losses = test_losses[ran[0]:ran[1]]
        x = np.linspace(max(ran[0],0), min(ran[1], len(train_losses)+max(ran[0],0)), len(train_losses))
    
    if norm == True:
        train_losses = torch.stack(train_losses) / train_losses[0]
        test_losses = torch.stack(test_losses) / test_losses[0]

    plt.scatter(x, train_losses.cpu(), label="training", color = 'C0')
    plt.scatter(x, test_losses.cpu(), label="test", color = 'C1', s = 20)
    plt.xlabel("Epochs")
    if norm == True: plt.ylabel("Normalized loss")
    else: plt.ylabel("Loss")
    plt.title(title)
    plt.yscale("log")
    plt.legend()
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

def adjust_learning_rate(optimizer, last_losses, epoch, initial_lr=1e-4, decay_factor=0.5, epoch_interval=10):
    """Reduce LR every decay_epoch epochs by decay_factor."""
    if initial_lr <= 1e-5:
        return
    else:
        if epoch % epoch_interval == 0 and epoch != 0:
            lr = initial_lr * decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(1e-5, lr)
        if len(last_losses)==5 and last_losses[1]>last_losses[0] and last_losses[2]>last_losses[1] and last_losses[3]>last_losses[2] and last_losses[4]>last_losses[3]:
            lr = optimizer.param_groups[0]['lr']*0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(1e-5, lr)

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

val_data, test_data, train_data = torch.utils.data.random_split(complete_data, [0.2, 0.6, 0.2], generator=torch.Generator(device=device))
#train_data, val_data, test_data = torch.utils.data.random_split(complete_data, [0.6, 0.2, 0.2], generator=torch.Generator(device=device))

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

train_x = torch.tensor([val[0].cpu().numpy() for val in train_data])
train_event = torch.tensor([val[1][0] for val in train_data]).bool()

val_x = torch.tensor([val[0].cpu().numpy() for val in val_data])
val_event = torch.tensor([val[1][0] for val in val_data])

test_x = torch.tensor([val[0].cpu().numpy() for val in test_data])
test_event = torch.tensor([val[1][0] for val in test_data])
    
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
            torch.nn.Linear(num_features, 28),
            torch.nn.Tanh(),
            torch.nn.Dropout(),
            torch.nn.Linear(28, 56),
            torch.nn.Tanh(),
            torch.nn.Dropout(),
            torch.nn.Linear(56, 112),
            torch.nn.Tanh(),
            torch.nn.Dropout(),
            torch.nn.Linear(112, 1),
            torch.nn.Sigmoid()
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

LEARNING_RATE = 1e-4
EPOCHS = 100
optimizer = torch.optim.AdamW(cox_model.parameters(), lr=LEARNING_RATE, weight_decay=0.5)
con = ConcordanceIndex()

# %% Train and Test loops

#global best_ind, best_model
best_ind = 0
best_ipcw_ind = 0
best_epoch = 0
best_loss = -1
best_model = NeuralNetwork()
#best_state = OrderedDict()

def train_loop(dataloader, model, optimizer):
    model.train()
    
    curr_loss = torch.tensor(0.0)
    weight = 0
    
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
    
    curr_loss /= weight
    return curr_loss

def test_loop(model, epoch):
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
        
        try:
            weight_ipcw = get_ipcw(train_event, train_time, time)
        except:
            curr_con_ind_ipcw = 0
            print('ERROR FOR IPCW WEIGHTS IN TEST LOOP')
        else:
            con_ind_ipcw = con(pred.float(), event, time.float(), weight = weight_ipcw)
            curr_con_ind_ipcw = con_ind_ipcw
                
        if loss < best_loss or best_loss < 0:
            best_ind = curr_con_ind
            best_ipcw_ind = curr_con_ind_ipcw
            best_loss = loss
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
            best_epoch = epoch+1
        
        #else:
            #model.load_state_dict(copy.deepcopy(best_model.state_dict()))
            #print(f"\nNo new best model found, index to beat: {best_ind:0.3f}")

    return curr_loss, curr_con_ind, curr_con_ind_ipcw

# %% Iterate through Train and Test loops

train_losses = []
val_losses = []

# %%

val_con_inds = []
val_con_ind_ipcws = []

for t in tqdm(range(EPOCHS)):
    curr_train_loss = train_loop(dataloader_train, cox_model, optimizer)
    curr_val_loss, curr_val_con_ind, curr_val_con_ind_ipcw = test_loop(cox_model, t)
    
    train_losses.append(curr_train_loss)
    val_losses.append(curr_val_loss)
    
    adjust_learning_rate(optimizer, val_losses[-5:], t, initial_lr=optimizer.param_groups[0]['lr'], decay_factor=0.9, epoch_interval=10)
    
    val_con_inds.append(curr_val_con_ind)
    val_con_ind_ipcws.append(curr_val_con_ind_ipcw)
    
    
    if t % (EPOCHS // 20) == 0:
        print(f"\nEpoch {t+1}, Loss to beat: {best_loss:0.3f}, Best Epoch: {best_epoch}\n-------------------------------")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:0.3e}")
        print(f"Training loss: {curr_train_loss:0.6f}, Validation loss: {curr_val_loss:0.6f}, Best loss: {best_loss:0.6f}")
        print(f"Concordance Index validation:  {curr_val_con_ind:0.3f}, IPCW Concordance Index validation:  {curr_val_con_ind_ipcw:0.3f}")
print('\n' + '-'*50)
print(f"Done! The best epoch was {best_epoch}.")
print(f"The Concordance Index of the validation data is: {best_ind:0.3f}, IPCW Concordance Index of the validation data is: {best_ipcw_ind:0.3f}")
print('-'*50)

# %% Plot the training and test losses

title = "learning rate 1e-4"
ns = 900
ne = 3000

plot_losses(val_losses, val_losses, title, norm = True, ran = [ns, ne])

# %%

df_test = pd.DataFrame(test_x.cpu().numpy(), columns = features)
df_test.insert(0, "ID", [str(int(val)) for val in np.linspace(1, len(df_test), len(df_test))])
model_input_test = torch.tensor(np.array(df_test[features])).float()
pred_test = best_model(model_input_test)
loss_test = neg_partial_log_likelihood(pred_test, test_event, test_time, reduction="mean")
weight_test = get_ipcw(train_event, train_time, test_time)
ind_test = con(pred_test.float(), test_event, test_time.float(), weight = weight_test)

print(f"Test loss: {loss_test:0.6f}")
print(f"Test Index: {ind_test:0.3f}")

# %% !!!ONLY RUN IF THE MODEL SHOULD GET SAVED!!!

cox_model.load_state_dict(copy.deepcopy(best_model.state_dict()))
torch.save(cox_model.state_dict(), data_dir + '\\saved_models\\model90.pth')

# %% Check if the Conconcordance and IPCW indices obtained from the test_results method match the ones calculated during the training

test_model = NeuralNetwork()

df_val = pd.DataFrame(val_x.cpu().numpy(), columns = features)
df_val.insert(0, "ID", [str(int(val)) for val in np.linspace(1, len(df_val), len(df_val))])

a = test_results(test_model, data_dir + "\\saved_models\\model90.pth", df_val, features, "model_temp", return_df = True)
pred_val = torch.tensor(a.risk_score).float()

con = ConcordanceIndex()
weight_val = get_ipcw(train_event, train_time, val_time)
ind_val = con(pred_val, val_event, val_time)
ind_ipcw_val = con(pred_val, val_event, val_time.float(), weight = weight_val)

print(f"Concordance Index and IPCW Concordance Index obtrained from data in test_data:\nConcordance Index: {ind_val:0.3f}\nIPCW Concordance Index: {ind_ipcw_val:0.3f}")

# %% Run the rest_results method and create a csv file for the submission

final_df = pd.read_csv(data_dir + "\\X_test\\clinical_test.csv")
final_df = final_df[['ID'] + features[:-1]]
final_mol_df = pd.read_csv(data_dir + "\\X_test\\molecular_test.csv")
final_df.insert(4, 'NSM', [len(final_mol_df[final_mol_df.ID == val]) for val in list(final_df.ID)])

test_model = NeuralNetwork()

test_results(test_model, data_dir + "\\saved_models\\model90.pth", final_df, features, "model90")

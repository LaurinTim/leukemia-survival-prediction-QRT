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

# Load GBSG2 dataset
df = lifelines.datasets.load_gbsg2()
df.head(5)

# %%

df_onehot = pd.get_dummies(df, columns=["horTh", "menostat", "tgrade"]).astype("float")
df_onehot.drop(
    ["horTh_no", "menostat_Post", "tgrade_I"],
    axis=1,
    inplace=True,
)
df_onehot.head(5)

# %%

df_train, df_test = train_test_split(df_onehot, test_size=0.3)
df_train, df_val = train_test_split(df_train, test_size=0.3)
print(
    f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}"
)

# %%

class Custom_dataset(Dataset):
    """ "Custom dataset for the GSBG2 brain cancer dataset"""

    # defining values in the constructor
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # Getting data size/length
    def __len__(self):
        return len(self.df)

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # Targets
        event = torch.tensor(sample["cens"]).bool()
        time = torch.tensor(sample["time"]).float()
        # Predictors
        x = torch.tensor(sample.drop(["cens", "time"]).values).float()
        return x, (event, time)

# %%

# Dataloader
dataloader_train = DataLoader(
    Custom_dataset(df_train), batch_size=BATCH_SIZE, shuffle=True
)
dataloader_val = DataLoader(
    Custom_dataset(df_val), batch_size=len(df_val), shuffle=False
)
dataloader_test = DataLoader(
    Custom_dataset(df_test), batch_size=len(df_test), shuffle=False
)

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

torch.manual_seed(42)

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
    #with torch.no_grad():
    #    x, (event, time) = next(iter(dataloader_val))
    #    val_losses.append(
    #        neg_partial_log_likelihood(cox_model(x), event, time, reduction="mean")
    #    )
    
# %%

optimizer = torch.optim.Adam(cox_model.parameters(), lr=LEARNING_RATE)
con = ConcordanceIndex()
torch.manual_seed(42)

for i in range(EPOCHS):
    size = len(df_train)
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
    
    if i % 10 == 0:
        loss, current = loss, k*len(X)+len(X)
        conc = con(pred, temp_event, temp_time)
        print(f"Epoch: {i:03}, Loss: {loss:>0.3f}  [{current:>5d}/{size:>5d}], Concordance Index: {conc:>5f}, Epoch loss: {epoch_loss:0.2f}")
    
















































































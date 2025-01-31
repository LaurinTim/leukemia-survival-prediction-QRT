import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torchvision.models as models

data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\Challenge Data QRT"

# %%

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# %%

class DatasetGen(Dataset):
    def __init__(self, annotations_file, clinical_file, molecular_file, clinical_transform=None, molecular_transform=None, status_transform=None):
        self.patient_status = pd.read_csv(annotations_file)
        self.patient_clinical = pd.read_csv(clinical_file)
        self.patient_molecular = pd.read_csv(molecular_file)
        self.clinical_transform = clinical_transform
        self.molecular_transform = molecular_transform
        self.status_transform = status_transform
        
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        patient_id = self.patient_status.iloc[idx, 0]
        os_years = self.patient_status.iloc[idx, 1]
        os_status = self.patient_status.iloc[idx, 2]
        status = torch.tensor([os_years, os_status])
        info_clinical = self.patient_clinical.iloc[idx]
        info_molecular = self.patient_molecular.iloc[idx]
        info = pd.concat([info_clinical, info_molecular])
        if self.transform:
            info_clinical = self.clinical_transform(info_clinical)
            info_molecular = self.molecular_transform(info_molecular)
        if self.target_transform:
            status = self.status_transform(status)
        return info_clinical, info_molecular, status

# %%

training_data = pd.read_csv(data_dir+"")
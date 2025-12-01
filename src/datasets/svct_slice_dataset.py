import torch
import torch.nn.functional as F
import numpy as np
import glob
import lightning.pytorch as pl
from src.physics.pbct import PBCT

class SVCTSliceDataset(torch.utils.data.Dataset):
    def __init__(self, filenames):
        self.files = sorted(filenames)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)

class SVCTSliceDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=1, num_workers=4, pin_memory=True, train_fraction=0.8):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_fraction = train_fraction

    def setup(self, stage=None):
        filenames = sorted(glob.glob(self.data_path + '*.pt'))
        # Split dataset by volumes (patients)
        num_train = int(len(filenames) * self.train_fraction)
        num_val = int(len(filenames) * 0.2)
        train_files = filenames[num_val:num_train + num_val]
        val_files = filenames[:num_val]

        self.train_data = SVCTSliceDataset(train_files)
        self.val_data = SVCTSliceDataset(val_files)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
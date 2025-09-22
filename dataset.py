import numpy as np
import torch
from torch.utils.data import Dataset

class SQGDataset(Dataset):
    def __init__(self, data_path, mean, std):
        """
        Args:
            data_path (str): Path to data file.
            mean, std: normalization stats.
        """
        self.mean = mean
        self.std = std
        self.data = np.load(data_path+'.npy')
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = (self.data[idx] - self.mean) / self.std
        return x

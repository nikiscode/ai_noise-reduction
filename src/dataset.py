import torch
from torch.utils.data import Dataset
import numpy as np
import os

class OpticalSignalDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.npy')])
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_signal = np.load(self.clean_files[idx])
        noisy_signal = np.load(self.noisy_files[idx])

        # Convert to tensors
        clean_signal = torch.tensor(clean_signal, dtype=torch.float32)
        noisy_signal = torch.tensor(noisy_signal, dtype=torch.float32)

        return noisy_signal.unsqueeze(0), clean_signal.unsqueeze(0)  # Add channel dim
# Example usage:
# dataset = OpticalSignalDataset(clean_dir='data/clean', noisy_dir='data/noisy  ')
# noisy_signal, clean_signal = dataset[0] 

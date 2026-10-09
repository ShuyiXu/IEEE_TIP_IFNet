# data_loader.py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split

class HVCD(Dataset):
    def __init__(self, t1_dir, t2_dir, label_dir, indices):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.label_dir = label_dir
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        t1_path = os.path.join(self.t1_dir, f'block_{index}.mat')
        t2_path = os.path.join(self.t2_dir, f'block_{index}.mat')
        label_path = os.path.join(self.label_dir, f'block_{index}.mat')

        t1 = sio.loadmat(t1_path)['T1']
        t2 = sio.loadmat(t2_path)['T2']
        label = sio.loadmat(label_path)['Binary']

        # 最大最小值归一化
        t1 = (t1 - np.min(t1)) / (np.max(t1) - np.min(t1))
        t2 = (t2 - np.min(t2)) / (np.max(t2) - np.min(t2))

        # 转换为 PyTorch 张量
        t1 = torch.from_numpy(t1).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        t2 = torch.from_numpy(t2).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        label = torch.from_numpy(label).long()

        return t1, t2, label

def get_train_test_indices(num_samples, train_size=0.7):
    indices = np.arange(1, num_samples + 1)
    train_indices, test_indices = train_test_split(indices, train_size=train_size, shuffle=True, random_state=22)
    return train_indices, test_indices






# stgcn_project/stgcn_lib/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset

class STGCN_Dataset(Dataset):
    """
    用于加载预处理后的数据集的PyTorch Dataset类。
    """
    def __init__(self, processed_data_path, split='train'):
        """
        Args:
            processed_data_path (str): 处理后的 .npz 文件的路径
            split (str): 'train', 'val', 或 'test'
        """
        data = np.load(processed_data_path)

        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be 'train', 'val', or 'test'.")

        x_key, y_key = f'X_{split}', f'Y_{split}'

        self.X = torch.from_numpy(data[x_key]).float()
        self.Y = torch.from_numpy(data[y_key]).float()
        self.adj = torch.from_numpy(data['adj']).float()

        # 保存标准化参数，以备后续反标准化使用
        self.mean = data['mean']
        self.std = data['std']

        print(f"Loaded {split} data: X shape={self.X.shape}, Y shape={self.Y.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_adj(self):
        return self.adj
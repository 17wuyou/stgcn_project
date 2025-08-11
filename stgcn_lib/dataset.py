# stgcn_project/stgcn_lib/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset

class STGCN_Dataset(Dataset):
    """
    用于加载预生成数据集的PyTorch Dataset类。
    """
    def __init__(self, data_path):
        """
        Args:
            data_path (str): .npz文件的路径
        """
        data = np.load(data_path)
        
        # 将数据转换为float32类型的Tensor
        # 注意：这里不直接移动到GPU，让DataLoader来处理
        self.X = torch.from_numpy(data['X']).float()
        self.Y = torch.from_numpy(data['Y']).float()
        self.adj = torch.from_numpy(data['adj']).float()
        
        print(f"Data loaded from {data_path}")
        print(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}, Adj shape: {self.adj.shape}")

    def __len__(self):
        """返回数据集中的样本总数"""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """根据索引获取一个样本"""
        return self.X[idx], self.Y[idx]
        
    def get_adj(self):
        """获取邻接矩阵"""
        return self.adj
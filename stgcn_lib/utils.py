import yaml
import numpy as np
import torch

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_normalized_laplacian(adj):
    """
    计算归一化的拉普拉斯矩阵，对应论文公式(5)中的传播矩阵。
    L_norm = D_tilde^(-0.5) * A_tilde * D_tilde^(-0.5)
    A_tilde = A + I_n
    D_tilde是A_tilde的度矩阵

    Args:
        adj (np.ndarray): 邻接矩阵, shape (N, N)

    Returns:
        torch.Tensor: 归一化的拉普拉斯矩阵
    """
    n = adj.shape[0]
    adj_tilde = adj + np.eye(n)
    d_tilde = np.sum(adj_tilde, axis=1)
    
    # 计算D_tilde^(-0.5)
    d_inv_sqrt = np.power(d_tilde, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # 计算L_norm
    normalized_laplacian = d_mat_inv_sqrt @ adj_tilde @ d_mat_inv_sqrt
    
    return torch.from_numpy(normalized_laplacian).float()

def generate_dummy_data(config):
    """生成模拟数据以供演示"""
    N = config['data']['num_nodes']
    T_in = config['data']['num_timesteps_input']
    T_out = config['data']['num_timesteps_output']
    C_in = config['data']['num_features']
    B = config['training']['batch_size'] * 5 # 创建几个批次的数据

    # 模拟邻接矩阵 (随机稀疏图)
    adj = np.random.rand(N, N)
    adj[adj > 0.1] = 1
    adj[adj <= 0.1] = 0
    adj = np.maximum(adj, adj.T) # 确保对称
    np.fill_diagonal(adj, 0)

    # 模拟交通数据
    X = torch.randn(B, T_in, N, C_in)
    Y = torch.randn(B, T_out, N, 1) # 假设预测目标只有1个特征

    return X, Y, adj
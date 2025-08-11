# stgcn_project/scripts/preprocess_metr_la.py

import os
import sys
import numpy as np
import pandas as pd
import pickle
import argparse

# 将项目根目录添加到Python路径中，确保可以找到 stgcn_lib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from stgcn_lib.utils import load_config

def load_metr_la_data(config):
    """加载原始METR-LA数据并返回 DataFrame 和邻接矩阵"""
    adj_path = config['data']['metr_la']['adj_path']
    data_path = config['data']['metr_la']['raw_path']

    if not os.path.exists(adj_path) or not os.path.exists(data_path):
        print(f"Raw data files not found in {os.path.dirname(adj_path)}. Please run 'scripts/download_data.py' first.")
        sys.exit(1)

    with open(adj_path, 'rb') as f:
        # 兼容不同格式的.pkl文件
        try:
            _, _, adj_mx = pickle.load(f, encoding='latin1')
        except (IndexError, ValueError):
            f.seek(0)
            adj_mx = pickle.load(f, encoding='latin1')

    df = pd.read_hdf(data_path)
    return df, adj_mx

def z_score_normalize(data, mean, std):
    """Z-Score 标准化"""
    return (data - mean) / std

def create_sliding_window(data, T_in, T_out):
    """使用滑动窗口创建样本"""
    X, Y = [], []
    total_len = len(data)
    i = 0
    while i + T_in + T_out <= total_len:
        X.append(data[i : i + T_in])
        Y.append(data[i + T_in : i + T_in + T_out])
        i += 1
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess METR-LA dataset.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)

    print("--- Starting Data Preprocessing ---")
    print("Loading raw METR-LA data...")
    df, adj = load_metr_la_data(config)

    traffic_data = df.values.astype(np.float32)

    num_samples = len(traffic_data)
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    train_end = int(num_samples * train_split)

    train_data = traffic_data[:train_end]

    mean = np.mean(train_data)
    std = np.std(train_data)
    if std == 0: std = 1 # 避免除以零
    print(f"Calculated stats from training data: mean={mean:.4f}, std={std:.4f}")

    traffic_data_norm = z_score_normalize(traffic_data, mean, std)

    T_in = config['data']['num_timesteps_input']
    T_out = config['data']['num_timesteps_output']

    X_all, Y_all = create_sliding_window(traffic_data_norm, T_in, T_out)

    num_all_samples = len(X_all)
    train_all_end = int(num_all_samples * train_split)
    val_all_end = int(num_all_samples * (train_split + val_split))

    X_train, Y_train = X_all[:train_all_end], Y_all[:train_all_end]
    X_val, Y_val = X_all[train_all_end:val_all_end], Y_all[train_all_end:val_all_end]
    X_test, Y_test = X_all[val_all_end:], Y_all[val_all_end:]

    X_train, Y_train = X_train[..., np.newaxis], Y_train[..., np.newaxis]
    X_val, Y_val = X_val[..., np.newaxis], Y_val[..., np.newaxis]
    X_test, Y_test = X_test[..., np.newaxis], Y_test[..., np.newaxis]

    print(f"Train shapes: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation shapes: X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, Y={Y_test.shape}")

    save_path = config['data']['metr_la']['processed_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        adj=adj,
        mean=mean,
        std=std
    )
    print(f"Preprocessed data saved to {save_path}")
    print("--- Data Preprocessing Finished ---")
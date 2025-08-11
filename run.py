# stgcn_project/run.py

import torch
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # <--- 这是核心的修改，导入numpy库

# 使用 __init__.py 简化导入
from stgcn_lib import (
    load_config,
    calculate_normalized_laplacian,
    generate_dummy_data,
    STGCN_Dataset,
    STGCN,
    Trainer
)

def main(args):
    # 1. 加载配置
    config = load_config(args.config)
    print("Configuration loaded.")
    
    if args.retrain:
        config['checkpoint']['load_checkpoint'] = False
        print("Retrain flag is set. Checkpoints will be ignored.")
    
    data_mode = config['data']['data_mode']
    print(f"Data mode: {data_mode}")

    # 2. 根据配置准备数据和邻接矩阵
    if data_mode == 'dummy':
        # --- 模式一：动态生成模拟数据 ---
        X, Y, adj = generate_dummy_data(config)
        train_dataset = TensorDataset(X, Y)
        val_dataset = TensorDataset(X[:10], Y[:10]) # 创建一个小的验证集

    elif data_mode == 'semi_realistic':
        # --- 模式二：加载半真实数据集 ---
        path = config['data']['semi_realistic_path']
        if not os.path.exists(path):
            print(f"Dataset file not found. Please run 'python generate_dataset.py' first.")
            return
        data = np.load(path)
        X, Y, adj = data['X'], data['Y'], data['adj']
        # 简单划分一下训练和验证
        split_idx = int(len(X) * 0.8)
        train_dataset = TensorDataset(torch.from_numpy(X[:split_idx]).float(), torch.from_numpy(Y[:split_idx]).float())
        val_dataset = TensorDataset(torch.from_numpy(X[split_idx:]).float(), torch.from_numpy(Y[split_idx:]).float())
        
    elif data_mode == 'metr_la':
        # --- 模式三：加载真实数据集 ---
        path = config['data']['metr_la']['processed_path']
        if not os.path.exists(path):
            print(f"Dataset file not found. Please run 'scripts/download_data.py' and 'scripts/preprocess_metr_la.py' first.")
            return
        train_dataset = STGCN_Dataset(path, split='train')
        val_dataset = STGCN_Dataset(path, split='val')
        adj = train_dataset.get_adj().numpy()

    else:
        raise ValueError("Invalid data_mode in config.yaml. Choose from 'dummy', 'semi_realistic', 'metr_la'.")

    # 3. 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 4. 计算归一化的拉普拉斯矩阵
    l_norm = calculate_normalized_laplacian(adj)
    print("Normalized laplacian calculated.")

    # 5. 初始化模型
    model = STGCN(config, l_norm)
    print("STGCN model initialized.")
    
    # 6. 初始化并启动训练器
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run STGCN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--retrain', action='store_true', help='Force retraining from scratch, ignoring checkpoints.')
    args = parser.parse_args()
    main(args)
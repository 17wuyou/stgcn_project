# stgcn_project/run.py (修改后)

import torch
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset

# 使用 __init__.py 简化导入
from stgcn_lib import (
    load_config,
    calculate_normalized_laplacian,
    generate_dummy_data, # 导入动态生成函数
    STGCN_Dataset,       # 导入文件加载类
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
    
    # 2. 根据配置准备数据和邻接矩阵
    if config['data']['use_file_dataset']:
        # --- 模式一：从文件加载数据 ---
        print("Mode: Loading data from file.")
        data_path = config['data']['data_path']
        if not os.path.exists(data_path):
            print(f"Dataset file not found at {data_path}.")
            print("Please run 'python generate_dataset.py' first.")
            return
            
        dataset = STGCN_Dataset(data_path)
        adj = dataset.get_adj().numpy() # 计算时使用numpy
    
    else:
        # --- 模式二：动态生成模拟数据 ---
        print("Mode: Generating dummy data on the fly.")
        X, Y, adj = generate_dummy_data(config)
        dataset = TensorDataset(X, Y)

    # 3. 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 4. 计算归一化的拉普拉斯矩阵
    l_norm = calculate_normalized_laplacian(adj)
    print("Normalized laplacian calculated.")

    # 5. 初始化模型 (将l_norm传递给模型)
    model = STGCN(config, l_norm)
    print("STGCN model initialized.")
    
    # 6. 初始化并启动训练器
    trainer = Trainer(model, config, dataloader)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run STGCN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--retrain', action='store_true', help='Force retraining from scratch, ignoring checkpoints.')
    args = parser.parse_args()
    main(args)
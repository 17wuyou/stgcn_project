import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader

from stgcn_lib.utils import load_config, generate_dummy_data, calculate_normalized_laplacian
from stgcn_lib.model import STGCN
from stgcn_lib.trainer import Trainer

def main(args):
    # 1. 加载配置
    config = load_config(args.config)
    print("Configuration loaded.")
    
    # 2. 准备数据和邻接矩阵
    # 在真实项目中，这里会加载真实数据
    X, Y, adj = generate_dummy_data(config)
    l_norm = calculate_normalized_laplacian(adj)
    print("Dummy data and normalized laplacian generated.")

    # 3. 创建DataLoader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 4. 初始化模型
    model = STGCN(config, l_norm)
    print("STGCN model initialized.")
    
    # 5. 初始化并启动训练器
    trainer = Trainer(model, config, dataloader, l_norm)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run STGCN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    main(args)
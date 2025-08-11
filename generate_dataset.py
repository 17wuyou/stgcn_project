# stgcn_project/generate_dataset.py

import numpy as np
import argparse
import os
from stgcn_lib.utils import load_config # 从我们自己的库导入

def generate_semi_realistic_data(num_samples, config):
    """
    生成具有时空相关性的半真实数据集。
    - 时间相关性: 每个节点的时间序列是带有噪声的正弦波。
    - 空间相关性: 相邻节点的正弦波相位略有不同。
    """
    N = config['data']['num_nodes']
    T_in = config['data']['num_timesteps_input']
    T_out = config['data']['num_timesteps_output']
    
    # 1. 生成邻接矩阵 (与之前类似，但现在是固定的)
    print("Generating adjacency matrix...")
    adj = np.random.rand(N, N)
    adj[adj > 0.1] = 1
    adj[adj <= 0.1] = 0
    adj = np.maximum(adj, adj.T) # 确保对称
    np.fill_diagonal(adj, 0)
    print("Adjacency matrix generated.")

    # 2. 生成长时序数据
    print("Generating long time-series with spatio-temporal properties...")
    total_timesteps = num_samples + T_in + T_out # 需要的总时间点数
    time_axis = np.arange(total_timesteps)
    long_series = np.zeros((total_timesteps, N, 1))

    for i in range(N):
        # 每个节点的相位略有不同，以模拟空间相关性
        phase_shift = (i / N) * 2 * np.pi
        # 每个节点有不同的频率和振幅
        frequency = np.random.uniform(0.05, 0.2)
        amplitude = np.random.uniform(0.8, 1.2)
        
        series = amplitude * np.sin(frequency * time_axis + phase_shift)
        noise = np.random.normal(0, 0.1, total_timesteps)
        long_series[:, i, 0] = series + noise
    
    # 3. 使用滑动窗口创建样本
    print("Creating samples using sliding window...")
    X, Y = [], []
    for i in range(num_samples):
        start_x = i
        end_x = start_x + T_in
        start_y = end_x
        end_y = start_y + T_out
        
        X.append(long_series[start_x:end_x, :, :])
        Y.append(long_series[start_y:end_y, :, :])

    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Dataset created with shapes: X={X.shape}, Y={Y.shape}")
    
    return X, Y, adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate semi-realistic dataset for STGCN.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate.')
    args = parser.parse_args()
    
    # 从config.yaml加载配置以获取节点数等信息
    config = load_config(args.config)
    
    output_path = config['data']['data_path']
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成并保存数据
    X, Y, adj = generate_semi_realistic_data(args.num_samples, config)
    np.savez_compressed(output_path, X=X, Y=Y, adj=adj)
    
    print(f"Dataset successfully saved to {output_path}")
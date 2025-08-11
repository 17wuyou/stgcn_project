# STGCN: 时空图卷积网络 PyTorch 实现

本项目是论文 **"Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"** (Yu, Yin, and Zhu) 的一个清晰、模块化的PyTorch实现。

该框架旨在处理时空图数据，尤其适用于交通流量预测。本项目从零开始搭建，包含了数据处理、模型构建、训练、验证和检查点管理等完整流程，并支持多种数据模式，方便测试和验证。

## 🚀 主要功能

- **模块化设计**: 代码结构清晰，分为数据处理、模型定义、训练逻辑等模块，易于理解和扩展。
- **灵活的配置**: 所有超参数和路径均由 `config.yaml` 文件统一管理，无需修改代码即可调整实验设置。
- **多种数据模式**:
    1.  **模拟数据模式**: 动态生成纯随机数据，用于快速验证代码逻辑和模型能否跑通。
    2.  **半真实数据集模式**: 生成具有时空特性的模拟数据，用于初步的模型效果评估。
    3.  **真实数据集模式 (METR-LA)**: 支持标准的METR-LA交通数据集，提供从下载、预处理到训练的完整工作流。
- **断点续传**: 自动保存和加载模型检查点，可以从上次中断的地方继续训练。
- **强制重训练**: 支持 `--retrain` 命令行参数，可以忽略检查点，从头开始训练。

## 📂 项目结构

```
stgcn_project/
├── checkpoints/              # 存放模型检查点文件
├── config.yaml               # 集中管理所有超参数和配置
├── data/
│   ├── raw/                  # 存放下载的原始数据
│   └── processed/            # 存放预处理后的数据
├── stgcn_lib/                # 核心库，存放所有模型和工具代码
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── scripts/                  # 存放一次性数据处理脚本
│   ├── download_data.py
│   └── preprocess_metr_la.py
├── generate_dataset.py       # 用于生成半真实数据集
├── run.py                    # 项目的主入口，用于启动训练
└── README.md                 # 项目说明文件
└── requirements.txt          # 项目依赖的Python包
```

## 🔧 环境安装

1.  **克隆项目**
    ```bash
    git clone https://github.com/17wuyou/stgcn_project.git
    cd stgcn_project
    ```

2.  **创建并激活Python虚拟环境** (推荐)
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ 运行指南 (Usage)

本项目通过修改 `config.yaml` 文件中的 `data_mode` 参数来切换不同的数据源。

---

### 模式一：使用动态模拟数据 (Dummy Data Mode)

**工作原理**: 此模式用于快速测试。它会在内存中动态生成纯随机的张量作为输入数据，不涉及任何文件读写。非常适合用于验证代码流程是否通顺。

**操作步骤**:

1.  **配置 `config.yaml`**:
    将 `data_mode` 设置为 `'dummy'`。
    ```yaml
    data:
      data_mode: 'dummy'
      # ... 其他参数在此模式下会被忽略
    ```

2.  **运行主程序**:
    ```bash
    python run.py
    ```
    程序将打印 "Data mode: dummy" 并开始使用随机数据进行训练。

---

### 模式二：使用半真实数据集 (Semi-realistic Mode)

**工作原理**: 此模式首先会运行一个脚本来生成一个具有时序特性（带噪声的正弦波）和空间结构的 `.npz` 数据文件，然后模型会加载这个文件进行训练。

**操作步骤**:

1.  **配置 `config.yaml`**:
    将 `data_mode` 设置为 `'semi_realistic'`。
    ```yaml
    data:
      data_mode: 'semi_realistic'
      semi_realistic_path: "./data/semi_realistic_dataset.npz"
      # ...
    ```

2.  **生成数据集 (只需运行一次)**:
    运行 `generate_dataset.py` 脚本来创建数据文件。
    ```bash
    python generate_dataset.py
    ```
    执行后，会在 `./data/` 目录下生成 `semi_realistic_dataset.npz` 文件。

3.  **运行主程序**:
    ```bash
    python run.py
    ```
    程序将打印 "Data mode: semi_realistic"，并加载刚刚生成的数据文件开始训练。

---

### 模式三：使用真实数据集 (METR-LA)

**工作原理**: 这是最完整的模式，模拟了真实世界的研究流程。它分三步：下载原始数据 -> 预处理数据 -> 加载处理好的数据进行训练。

**操作步骤**:

1.  **配置 `config.yaml`**:
    将 `data_mode` 设置为 `'metr_la'`。
    ```yaml
    data:
      data_mode: 'metr_la'
      metr_la:
        raw_path: "./data/raw/metr-la.h5"
        adj_path: "./data/raw/adj_mx.pkl"
        processed_path: "./data/processed/metr_la.npz"
      # ...
    ```

2.  **下载原始数据 (只需运行一次)**:
    运行 `scripts/download_data.py` 脚本。
    ```bash
    python scripts/download_data.py
    ```
    脚本会自动下载 `metr-la.h5` 和 `adj_mx.pkl` 文件到 `./data/raw/` 目录下。

3.  **预处理数据 (只需运行一次)**:
    运行 `scripts/preprocess_metr_la.py` 脚本。
    ```bash
    python scripts/preprocess_metr_la.py
    ```
    该脚本会读取原始数据，进行Z-Score标准化、滑动窗口切分，并将最终的训练/验证/测试集保存到 `./data/processed/metr_la.npz` 文件中。

4.  **运行主程序**:
    ```bash
    python run.py
    ```
    程序将打印 "Data mode: metr_la"，加载处理好的真实数据集开始训练和验证。

## 🧠 其他功能

### 断点续传与强制重训练

- **自动续训**: 默认情况下 (`load_checkpoint: true`)，每次运行 `run.py` 都会自动在 `checkpoints` 目录下寻找最新的模型文件并加载，从上次的进度继续训练。
- **强制重训**: 如果你想忽略所有检查点，从零开始一次全新的训练，请使用 `--retrain` 参数：
  ```bash
  python run.py --retrain
  ```

## 致谢

本项目的设计和思想源于原作者的杰出工作。
- **论文**: Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. *IJCAI*.
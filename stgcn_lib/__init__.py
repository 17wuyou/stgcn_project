# stgcn_lib/__init__.py

# 从各个模块中导入核心类，以便可以直接从 stgcn_lib 导入它们
from .model import STGCN, STConvBlock, TemporalConvLayer, SpatialGraphConvLayer
from .trainer import Trainer
from .utils import load_config, calculate_normalized_laplacian, generate_dummy_data

# 使用 __all__ 定义当 'from stgcn_lib import *' 时应该导入的内容
# 这是一个良好的编程实践
__all__ = [
    'STGCN',
    'STConvBlock',
    'TemporalConvLayer',
    'SpatialGraphConvLayer',
    'Trainer',
    'load_config',
    'calculate_normalized_laplacian',
    'generate_dummy_data'
]
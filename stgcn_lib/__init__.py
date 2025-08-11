# stgcn_project/stgcn_lib/__init__.py

from .model import STGCN
from .trainer import Trainer
from .utils import load_config, calculate_normalized_laplacian, generate_dummy_data # 确保 generate_dummy_data 在这里
from .dataset import STGCN_Dataset

__all__ = [
    'STGCN',
    'Trainer',
    'load_config',
    'calculate_normalized_laplacian',
    'generate_dummy_data', # 确保 generate_dummy_data 在这里
    'STGCN_Dataset'
]
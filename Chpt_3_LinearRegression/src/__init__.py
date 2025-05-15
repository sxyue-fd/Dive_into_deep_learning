"""
线性回归项目
使用PyTorch API实现的线性回归
"""

from .data import generate_data
from .model import create_model, LinearRegression
from .trainer import Trainer
from .utils import load_config, set_seed, get_true_params

__all__ = [
    'generate_data',
    'create_model',
    'LinearRegression',
    'Trainer',
    'load_config',
    'set_seed',
    'get_true_params'
]

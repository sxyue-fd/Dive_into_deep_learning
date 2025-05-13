"""
线性回归项目
包含从零实现和使用PyTorch API两种实现方式
"""

from .data import generate_data
from .model import create_model, LinearRegressionFromZero, LinearRegressionSimplified
from .trainer import Trainer
from .utils import load_config, set_seed, get_true_params

__all__ = [
    'generate_data',
    'create_model',
    'LinearRegressionFromZero',
    'LinearRegressionSimplified',
    'Trainer',
    'load_config',
    'set_seed',
    'get_true_params'
]

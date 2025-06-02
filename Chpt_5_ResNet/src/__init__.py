"""
ResNet18 CIFAR-10 项目
使用自定义实现的ResNet18模型进行CIFAR-10图像分类
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Project"
__description__ = "ResNet18 CIFAR-10 Image Classification"

from .model import create_resnet18, ResNet18, BasicBlock
from .data import create_data_loaders, CIFAR10DataLoader
from .trainer import ResNetTrainer
from .utils import (
    setup_logging, set_random_seed, calculate_accuracy,
    AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint
)
from .visualization import ResNetVisualizer

__all__ = [
    'create_resnet18', 'ResNet18', 'BasicBlock',
    'create_data_loaders', 'CIFAR10DataLoader',
    'ResNetTrainer',
    'setup_logging', 'set_random_seed', 'calculate_accuracy',
    'AverageMeter', 'EarlyStopping', 'save_checkpoint', 'load_checkpoint',
    'ResNetVisualizer'
]

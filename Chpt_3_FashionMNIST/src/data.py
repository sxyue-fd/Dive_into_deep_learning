"""
FashionMNIST数据集加载模块
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_fashion_mnist(config):
    """
    加载FashionMNIST数据集
    
    Args:
        config: 配置对象，包含数据集相关的配置信息
        
    Returns:
        train_loader: 训练数据加载器（全部60000张图像）
        test_loader: 测试数据加载器（10000张图像）
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 确保使用项目根目录作为基准
    root_dir = config['data'].get('root_dir', 'data')
    data_root = project_root / root_dir.lstrip('./')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.FashionMNIST(
        root=data_root, 
        train=True,
        download=True,
        transform=transform
    )
      # 加载测试集
    test_dataset = datasets.FashionMNIST(
        root=data_root, 
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,  # 直接使用完整训练集
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, test_loader

def get_fashion_mnist_labels():
    """
    获取FashionMNIST数据集的标签名称
    
    Returns:
        list: 标签名称列表
    """
    return ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

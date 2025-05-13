"""
FashionMNIST数据集加载模块
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_fashion_mnist(config):
    """
    加载FashionMNIST数据集
    
    Args:
        config: 配置对象，包含数据集相关的配置信息
        
    Returns:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    root = config['data']['root_dir']
    train_dataset = datasets.FashionMNIST(
        root=root, 
        train=True,
        download=True,
        transform=transform
    )
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )
    
    # 加载测试集
    test_dataset = datasets.FashionMNIST(
        root=root, 
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, valid_loader, test_loader

def get_fashion_mnist_labels():
    """
    获取FashionMNIST数据集的标签名称
    
    Returns:
        list: 标签名称列表
    """
    return ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

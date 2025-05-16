"""
Copyright (c) 2025
此项目基于 MIT License 进行许可
有关详细信息，请参阅项目根目录中的 LICENSE 文件
"""
"""
线性回归数据生成和加载模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SyntheticDataset(Dataset):
    """合成数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def save_data(X, y, data_dir, split='train'):
    """保存数据到指定目录
    
    Args:
        X: 特征数据
        y: 标签数据
        data_dir: 数据保存的目录
        split: 数据集划分('train' 或 'test')
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    np.save(str(Path(data_dir) / f'X_{split}.npy'), X)
    np.save(str(Path(data_dir) / f'y_{split}.npy'), y)

def load_data(data_dir, split='train'):
    """从指定目录加载数据
    
    Args:
        data_dir: 数据保存的目录
        split: 数据集划分('train' 或 'test')
    
    Returns:
        dataset: 加载的数据集
    """
    X = np.load(str(Path(data_dir) / f'X_{split}.npy'))
    y = np.load(str(Path(data_dir) / f'y_{split}.npy'))
    return SyntheticDataset(X, y)

def generate_and_save_data(config):
    """生成数据并保存到磁盘
    
    Args:
        config: 配置对象，包含数据生成相关的配置信息
    """
    # 设置随机种子
    np.random.seed(config['data']['seed'])
    
    # 生成特征
    num_examples = config['data']['num_examples']
    true_w = np.array(config['data']['true_w'])
    true_b = config['data']['true_b']
    
    # 生成随机特征
    X = np.random.normal(
        config['data']['features_mean'],
        config['data']['features_std'],
        (num_examples, len(true_w))
    )
    
    # 生成带噪声的标签
    y = np.dot(X, true_w) + true_b
    y += np.random.normal(0, config['data']['noise_sigma'], y.shape)
    
    # 计算划分点
    train_size = int(config['data']['train_ratio'] * len(X))
    
    # 保存训练集和测试集
    data_dir = Path(config['data']['data_dir'])
    save_data(X[:train_size], y[:train_size], data_dir, 'train')
    save_data(X[train_size:], y[train_size:], data_dir, 'test')

def get_data_loaders(config):
    """获取数据加载器
    
    Args:
        config: 配置对象，包含数据加载相关的配置信息
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    data_dir = Path(config['data']['data_dir'])
    
    # 如果数据文件不存在，先生成并保存数据
    if not (data_dir / 'X_train.npy').exists():
        generate_and_save_data(config)
    
    # 加载数据集
    train_dataset = load_data(data_dir, 'train')
    test_dataset = load_data(data_dir, 'test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False
    )
    
    return train_loader, test_loader

"""
线性回归数据生成模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class SyntheticDataset(Dataset):
    """合成数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_data(config):
    """
    生成带有噪声的线性回归数据
    
    Args:
        config: 配置对象，包含数据生成相关的配置信息
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
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
    
    # 创建数据集
    dataset = SyntheticDataset(X, y)
    
    # 分割数据集
    train_size = int(config['data']['train_ratio'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
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

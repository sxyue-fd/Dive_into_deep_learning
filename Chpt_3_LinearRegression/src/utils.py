"""
工具函数模块
"""
import yaml
import torch
import random
import numpy as np

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_true_params(config):
    """
    获取真实的模型参数
    
    Args:
        config: 配置对象
    
    Returns:
        tuple: (w, b) 真实的权重和偏差
    """
    return np.array(config['data']['true_w']), config['data']['true_b']

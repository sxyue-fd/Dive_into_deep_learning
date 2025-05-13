"""
工具函数模块
"""
import matplotlib.pyplot as plt
from .data import get_fashion_mnist_labels

def show_fashion_mnist(images, labels):
    """
    显示Fashion-MNIST图像和标签
    
    Args:
        images: 图像数据，形状为(n, 1, 28, 28)
        labels: 标签数据，形状为(n,)
    """
    labels = labels.numpy()
    class_names = get_fashion_mnist_labels()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(class_names[lbl])
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    import torch
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

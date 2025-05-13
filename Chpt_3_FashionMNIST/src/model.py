"""
FashionMNIST分类模型模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRegression(nn.Module):
    """Softmax回归模型"""
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 784)  # 展平输入
        x = self.linear(x)
        return x

class MLP(nn.Module):
    """多层感知机模型"""
    def __init__(self, hidden_units, dropout_rate=0.5, input_dim=784, num_classes=10):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, 784)  # 展平输入
        return self.model(x)

def create_model(config):
    """
    根据配置创建模型
    
    Args:
        config: 配置对象，包含模型相关的配置信息
        
    Returns:
        nn.Module: 创建的模型
    """
    model_type = config['model']['type']
    
    if model_type == 'softmax':
        model = SoftmaxRegression()
    elif model_type == 'mlp':
        model = MLP(
            hidden_units=config['model']['hidden_units'],
            dropout_rate=config['model']['dropout_rate']
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    return model

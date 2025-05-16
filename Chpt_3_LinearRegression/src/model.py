"""
Copyright (c) 2025
此项目基于 MIT License 进行许可
有关详细信息，请参阅项目根目录中的 LICENSE 文件
"""
"""
线性回归模型模块
"""
import torch.nn as nn

class LinearRegression(nn.Module):
    """使用PyTorch API实现的线性回归"""
    def __init__(self, input_dim=2, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def create_model(config):
    """根据配置创建模型"""
    return LinearRegression(config['model']['input_dim'], config['model']['output_dim'])

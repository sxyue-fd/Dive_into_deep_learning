"""
线性回归模型模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegressionFromZero:
    """从零开始实现的线性回归"""
    def __init__(self, input_dim):
        # 初始化模型参数
        self.w = torch.randn(input_dim, 1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    
    def forward(self, x):
        """前向传播"""
        return torch.matmul(x, self.w) + self.b
    
    def parameters(self):
        """返回模型参数"""
        return [self.w, self.b]
    
    def to(self, device):
        """将模型移动到指定设备"""
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        return self
    
    def train(self):
        """设置为训练模式"""
        self.w.requires_grad_(True)
        self.b.requires_grad_(True)
    
    def eval(self):
        """设置为评估模式"""
        self.w.requires_grad_(False)
        self.b.requires_grad_(False)

class LinearRegressionSimplified(nn.Module):
    """使用PyTorch API实现的线性回归"""
    def __init__(self, input_dim=2, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def create_model(config):
    """
    根据配置创建模型
    
    Args:
        config: 配置对象，包含模型相关的配置信息
        
    Returns:
        model: 创建的模型
    """
    model_type = config['model']['type']
    input_dim = config['model']['input_dim']
    output_dim = config['model']['output_dim']
    
    if model_type == 'from_zero':
        model = LinearRegressionFromZero(input_dim)
    elif model_type == 'simplified':
        model = LinearRegressionSimplified(input_dim, output_dim)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    return model

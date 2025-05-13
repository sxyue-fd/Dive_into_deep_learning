import torch
import torch.nn as nn

class HousePriceMLP(nn.Module):
    """
    房价预测的多层感知机模型
    
    特点:
    - 使用单隐藏层结构（input_size -> hidden_size -> 1）
    - 采用ReLU激活函数
    - 使用Dropout防止过拟合
    """
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.3):
        """
        初始化模型
        
        Args:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层神经元数量
            dropout_rate (float): Dropout比率
        """
        super().__init__()
        
        # 隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层
        self.output = nn.Linear(hidden_size, 1)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入数据
            
        Returns:
            torch.Tensor: 预测结果
        """
        x = self.hidden(x)
        return self.output(x)

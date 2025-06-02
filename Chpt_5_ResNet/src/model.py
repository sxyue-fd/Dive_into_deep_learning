"""
ResNet18 模型实现
自定义实现ResNet18架构，适配CIFAR-10数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock(nn.Module):
    """ResNet基础块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        downsample: 下采样层
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """ResNet18网络架构
    
    专门为CIFAR-10数据集设计，输入尺寸为32x32
    
    Args:
        num_classes: 分类数量
        layers: 每个stage的block数量
    """

    def __init__(self, num_classes: int = 10, layers: List[int] = [2, 2, 2, 2]):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        # 初始卷积层，适配CIFAR-10的32x32输入
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        
        # 全局平均池化和分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int, 
                    stride: int = 1) -> nn.Sequential:
        """构建残差层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """获取中间层特征图，用于可视化"""
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        return features


def create_resnet18(num_classes: int = 10) -> ResNet18:
    """创建ResNet18模型
    
    Args:
        num_classes: 分类数量
        
    Returns:
        ResNet18模型实例
    """
    return ResNet18(num_classes=num_classes)


# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = create_resnet18(num_classes=10)
    
    # 测试输入
    x = torch.randn(4, 3, 32, 32)  # CIFAR-10输入尺寸
    
    # 前向传播测试
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 特征图测试
    features = model.get_feature_maps(x)
    for name, feature in features.items():
        print(f"{name}: {feature.shape}")

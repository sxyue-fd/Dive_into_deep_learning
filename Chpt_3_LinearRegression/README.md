# 线性回归项目

本项目提供了线性回归的两种实现方式：
- 从零实现：手动实现梯度计算和参数更新
- 简化实现：使用PyTorch的自动微分功能

## 项目结构

```
Chpt_3_LinearRegression/
├── configs/          # 配置文件目录
│   └── config.yaml   # 模型和训练配置
├── data/            # 数据目录
│   ├── raw/        # 原始数据
│   └── processed/  # 处理后的数据
├── outputs/         # 输出目录
│   ├── models/     # 保存的模型
│   ├── logs/       # 训练日志
│   └── visualizations/ # 可视化结果
└── src/            # 源代码
    ├── data.py     # 数据生成
    ├── model.py    # 模型定义
    ├── trainer.py  # 训练逻辑
    ├── utils.py    # 工具函数
    └── main.py     # 主程序
```

## 快速开始

1. 确保已安装依赖：
```bash
pip install -r ../requirements.txt
```

2. 训练模型：
```bash
cd src
python main.py --config ../configs/config.yaml
```

## 配置说明

在`configs/config.yaml`中可以配置：
- 数据生成参数：样本数、真实参数、噪声等
- 模型类型：'from_zero'或'simplified'
- 训练参数：学习率、epoch数等
- 日志和可视化设置

## 实现说明

1. 从零实现
   - 手动定义模型参数和梯度
   - 实现梯度下降更新
   - 更好地理解原理

2. 简化实现
   - 使用PyTorch的nn.Linear
   - 自动处理反向传播
   - 代码更简洁高效

## 数据生成

- 使用已知参数生成带噪声的线性数据
- 可配置特征数量和分布
- 支持训练集和测试集划分

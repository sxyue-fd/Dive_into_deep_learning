# Fashion-MNIST图像分类项目

本项目实现了一个完整的Fashion-MNIST服装图像分类系统，支持使用Softmax回归和多层感知机(MLP)两种方法进行训练和预测。项目采用模块化设计，具有良好的可配置性和可扩展性。

## 项目介绍

### 数据集说明

Fashion-MNIST是一个替代MNIST手写数字数据集的图像数据集，包含10类不同的服装图像：

- T-shirt（T恤）
- Trouser（裤子）
- Pullover（套衫）
- Dress（连衣裙）
- Coat（外套）
- Sandal（凉鞋）
- Shirt（衬衫）
- Sneaker（运动鞋）
- Bag（包）
- Ankle boot（短靴）

每张图片为28x28的灰度图像，训练集60,000张，测试集10,000张。

### 模型实现

项目提供两种分类模型：

1. Softmax回归
   - 简单的线性分类器
   - 直接将784维输入映射到10个类别
   - 适合快速实验和基准测试
   - 参数量少，训练速度快

2. 多层感知机(MLP)
   - 可配置的多层神经网络
   - 支持灵活配置隐藏层数量和维度
   - 使用ReLU激活函数提升非线性表达能力
   - 采用Dropout进行正则化防止过拟合
   - 通常能获得更好的分类性能

## 项目结构

```plaintext
Chpt_3_FashionMNIST/
├── configs/          # 配置文件目录
│   └── config.yaml   # 模型和训练配置
├── data/            # 数据目录
│   └── FashionMNIST/ # 数据集文件
├── outputs/         # 输出目录
│   ├── models/      # 保存的模型权重
│   ├── logs/        # 训练日志文件
│   └── visualizations/ # 可视化结果
└── src/             # 源代码
    ├── data.py      # 数据加载和预处理
    ├── model.py     # 模型定义
    ├── trainer.py   # 训练和评估逻辑
    ├── utils.py     # 工具函数
    └── main.py      # 主程序入口
```

## 快速开始

### 1. 环境配置

确保Python 3.8+环境，安装依赖：

```bash
pip install -r ../requirements.txt
```

### 2. 模型训练

在项目目录下执行以下命令：

```bash
cd src
python main.py --config ../configs/config.yaml
```

### 3. 自定义配置

在`configs/config.yaml`中可以调整以下参数：

```yaml
data:
  batch_size: 256      # 批次大小
  num_workers: 4       # 数据加载线程数

model:
  type: "mlp"         # 模型类型：'softmax'或'mlp'
  hidden_units: [256, 128]  # MLP隐藏层维度
  dropout_rate: 0.5   # Dropout比率

training:
  learning_rate: 0.1  # 学习率
  weight_decay: 0.001 # L2正则化系数
  num_epochs: 5       # 训练轮次
  device: "cuda"      # 训练设备：'cuda'或'cpu'
```

## 功能特点

1. 完整的训练流程
   - 自动下载和处理数据集
   - 支持CPU和GPU训练
   - 实时损失曲线可视化
   - 自动保存最佳模型
   - 训练过程日志记录

2. 丰富的可视化功能
   - 数据集类别分布统计
   - 训练过程损失和准确率曲线
   - 混淆矩阵分析
   - 预测结果样本可视化

3. 良好的扩展性
   - 模块化的代码结构
   - 基于配置文件的参数管理
   - 易于添加新的模型架构
   - 灵活的训练策略配置

## 使用建议

1. 模型选择
   - 快速实验：使用Softmax回归
   - 追求性能：使用MLP并调整隐藏层
   - 防止过拟合：适当调整dropout_rate

2. 性能优化
   - 增加训练轮次(num_epochs)提升精度
   - 调整batch_size平衡速度和内存
   - 使用GPU加速训练
   - 根据收敛情况调整learning_rate

## 结果示例

在默认配置下（5轮训练，学习率0.1，权重衰减1e-4），典型性能如下：

- MLP配置 [256, 128]：
  - 训练准确率：~84%
  - 测试准确率：~85%
  
注：实际性能可能因训练轮数、学习率等超参数设置而有所不同。

## 作者

基于李沐《动手学深度学习》第三章 Fashion-MNIST 分类任务，由学习者重构完成。

## 许可证

MIT License

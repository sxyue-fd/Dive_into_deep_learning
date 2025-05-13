# Fashion-MNIST分类项目

本项目实现了两种方法对Fashion-MNIST数据集进行分类：
- Softmax回归：直接的线性分类器
- 多层感知机(MLP)：带有隐藏层的神经网络

## 项目结构

```
Chpt_3_FashionMNIST/
├── configs/          # 配置文件目录
│   └── config.yaml   # 模型和训练配置
├── data/            # 数据目录
├── outputs/         # 输出目录
│   ├── models/      # 保存的模型
│   ├── logs/        # 训练日志
│   └── visualizations/ # 可视化结果
└── src/             # 源代码
    ├── data.py      # 数据加载
    ├── model.py     # 模型定义
    ├── trainer.py   # 训练逻辑
    ├── utils.py     # 工具函数
    └── main.py      # 主程序
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
- 数据集参数：批量大小、工作进程数等
- 模型类型：'softmax'或'mlp'
- 训练参数：学习率、epoch数、优化器等
- 日志和可视化设置

## 模型说明

1. Softmax回归
   - 直接将输入映射到类别概率
   - 适合线性可分的数据

2. 多层感知机
   - 包含多个隐藏层
   - 可以学习更复杂的特征
   - 使用ReLU激活和Dropout正则化

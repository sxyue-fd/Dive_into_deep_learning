# 房价预测项目

基于PyTorch实现的房价预测模型，使用Kaggle房价预测竞赛数据集（House Prices - Advanced Regression Techniques）。

## 数据集说明

项目使用Kaggle房价预测竞赛的数据集，包含：
- 训练集（train.csv）：1460个样本，81个特征
- 测试集（test.csv）：1459个样本，80个特征（不含目标变量）
- 特征说明（data_description.txt）：详细的特征含义说明
- 提交样例（sample_submission.csv）：预测结果提交格式

数据集特点：
- 包含数值和分类特征
- 存在缺失值
- 特征之间可能有关联性
- 目标变量为房屋售价（SalePrice）

## 项目结构

```
Chpt_3_Kaggle_house_price/
├── configs/            # 配置文件
│   └── config.yaml    # 主配置文件
├── data/              # 数据文件
│   ├── raw/          # 原始数据
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/    # 处理后的数据
├── outputs/          # 输出文件
│   ├── models/      # 保存的模型
│   ├── logs/        # 训练日志
│   └── visualizations/# 可视化结果
└── src/             # 源代码
    ├── data.py      # 数据处理
    ├── model.py     # 模型定义
    ├── trainer.py   # 训练器
    ├── utils.py     # 工具函数
    └── main.py      # 主程序
```

## 功能特点

- 完整的数据预处理流程
  - 数值特征标准化
  - 分类特征独热编码
  - 缺失值处理
- K折交叉验证训练
- 实时训练可视化
- 过拟合检测和预警
- 模型检查点保存
- 详细的训练日志
- 可配置的超参数

## 环境要求

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
pyyaml>=5.4.0
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据：
- 将train.csv和test.csv放入data/raw/目录

3. 修改配置：
- 根据需要修改configs/config.yaml中的参数

4. 开始训练：
```bash
python src/main.py
```

## 配置说明

config.yaml中的主要配置项：

```yaml
model:
  hidden_size: 32      # 隐藏层大小
  dropout_rate: 0.3    # Dropout比率

training:
  batch_size: 64       # 批量大小
  num_epochs: 50       # 训练轮数
  learning_rate: 0.001 # 学习率
  weight_decay: 0.05   # L2正则化系数
  k_folds: 5          # 交叉验证折数
```

## 结果查看

- 训练日志：outputs/logs/training_results.txt
- 训练过程图：outputs/visualizations/
- 模型文件：outputs/models/

## 注意事项

- 确保数据文件格式正确
- GPU训练自动启用（如果可用）
- 可通过修改config.yaml调整参数
- 训练过程中会自动保存最佳模型

## 维护者

[Your Name]

## 许可证

MIT License

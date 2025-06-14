# ResNet18 CIFAR-10 图像分类项目

## 项目概述

本项目使用自定义实现的 ResNet18 深度残差网络对 CIFAR-10 数据集进行图像分类任务。项目严格遵循深度学习最佳实践，包含完整的训练、验证、测试和可视化流程，并配备了专门的日志管理系统。

## 项目特点

- **自定义 ResNet18 实现**：完全手动实现 ResNet18 架构，包含 BasicBlock 残差块
- **CIFAR-10 优化**：针对 32×32 图像尺寸进行架构调整
- **完整训练流程**：包含数据增强、学习率调度、Dropout正则化、早停机制
- **专业日志系统**：独立的ResNet日志管理器，训练/评估模式分离，同步txt结果文件
- **丰富可视化**：训练曲线、混淆矩阵、特征图、卷积核可视化、激活热力图等
- **性能监控**：GPU/内存使用率监控，训练时间统计
- **数据加载优化**：支持预取和持久化工作进程，提升训练效率

## 性能基准

- **训练准确率**：≥ 90%
- **验证准确率**：≥ 85%
- **训练速度**：单 epoch ≤ 5 分钟（NVIDIA A800 80GB PCIe）

## 项目结构

```text
Chpt_5_ResNet/
├── src/                       # 源代码
│   ├── data.py               # 数据加载和预处理（支持数据加载优化）
│   ├── model.py              # ResNet18 模型实现（支持Dropout）
│   ├── trainer.py            # 训练器（集成ResNet日志管理器）
│   ├── utils.py              # 工具函数
│   ├── main.py               # 主程序入口
│   ├── config_parser.py      # 配置解析器
│   ├── resnet_log_manager.py # 专用日志管理系统
│   └── visualization.py      # 可视化功能（含卷积核可视化）
├── configs/
│   └── config_performance.yaml # 性能优化配置文件
├── data/
│   └── raw/                  # CIFAR-10 原始数据
├── outputs/
│   ├── logs/                 # 训练/评估日志
│   │   ├── train_yyyymmdd_hhmmss.log     # 训练日志
│   │   ├── train_results_yyyymmdd_hhmmss.txt # 训练结果
│   │   └── eval_yyyymmdd_hhmmss.log      # 评估日志
│   ├── models/               # 保存的模型
│   └── visualizations/       # 可视化结果
├── test/
│   ├── test_modules.py       # 核心模块测试
│   └── test_compatibility.py # 兼容性测试
├── requirements.txt          # 依赖包
└── README.md                # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行训练

```bash
# 使用默认配置训练
python src/main.py

# 使用自定义配置和命令行参数
python src/main.py --config configs/config_performance.yaml --epochs 50 --lr 0.01

# 指定性能预设
python src/main.py --performance_preset baseline

# 评估模式
python src/main.py --run_mode evaluation --model_path outputs/models/best_model.pth
```

### 3. 模块测试

```bash
# 运行所有测试
python test/test_modules.py

# 运行兼容性测试
python test/test_compatibility.py
```

## 配置系统

### 配置文件结构

项目使用 `config_performance.yaml` 进行配置管理，支持以下功能：

#### 运行模式配置
```yaml
# 运行模式：training（训练）或 evaluation（评估）
run_mode: "training"
```

#### 性能预设
```yaml
# 性能预设：quick_test（快速测试）或 baseline（基准测试）
performance_preset: "baseline"

presets:
  quick_test:
    epochs: 5
    batch_size: 64
    dropout_rate: 0.2
  baseline:
    epochs: 100
    batch_size: 128
    dropout_rate: 0.5
```

#### 数据加载优化
```yaml
data:
  num_workers: 4
  prefetch_factor: 2        # 预取因子，提升数据加载效率
  persistent_workers: true  # 持久化工作进程
```

#### 日志控制
```yaml
logging:
  log_frequency: 20    # 训练进度日志频率（每N个batch）
  save_frequency: 5    # 详细摘要保存频率（每N个epoch）
```

## 模型架构

### ResNet18 结构

- **输入层**：3×32×32 CIFAR-10 图像
- **卷积层**：3×3 卷积 + BatchNorm + ReLU
- **残差块**：4 个残差层，每层包含 2 个 BasicBlock，每个 BasicBlock 包含两个 3×3 卷积层和恒等映射
- **下采样**：每个残差层的第一个 BasicBlock 使用 1×1 卷积调整维度
- **全局平均池化**：替代全连接层
- **分类器**：单层全连接，输出 10 类概率

### BasicBlock 设计

```text
输入 → Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → (+) → ReLU → 输出
  ↓                                                        ↑
  → 恒等映射或1×1卷积调整维度 ─────────────────────────────────┘
```

## 数据处理

### CIFAR-10 数据集

- **类别**：10 类（飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车）
- **训练集**：50,000 张图像（训练 40,000 + 验证 10,000）
- **测试集**：10,000 张图像
- **图像尺寸**：32×32×3

### 数据增强策略

- **训练时增强**：
  - RandomHorizontalFlip (p=0.5)
  - RandomCrop (32×32, padding=4)
- **标准化**：ImageNet 统计值
  - mean=[0.485, 0.456, 0.406]
  - std=[0.229, 0.224, 0.225]

## 训练策略

### 优化设置

- **优化器**：Adam (lr=0.001, betas=[0.9, 0.999])
- **学习率调度**：CosineAnnealingLR (T_max=100)
- **权重衰减**：1e-4
- **批次大小**：128

### 训练技巧

- **Dropout正则化**：可配置的dropout率（quick_test: 0.2, baseline: 0.5）
- **早停机制**：验证损失连续 10 轮不下降
- **梯度裁剪**：防止梯度爆炸
- **混合精度**：加速训练，节省显存
- **检查点保存**：定期保存最佳模型

## 日志系统

项目采用专用的 `ResNetLogManager` 日志管理系统：

### 文件命名规范

- **训练模式**：
  - `train_yyyymmdd_hhmmss.log` - 详细训练日志
  - `train_results_yyyymmdd_hhmmss.txt` - 结构化结果文件
- **评估模式**：
  - `eval_yyyymmdd_hhmmss.log` - 评估日志

### 日志频率控制

- **训练进度日志**：每 `log_frequency` 个batch记录一次（默认20）
- **详细摘要**：每 `save_frequency` 个epoch保存一次（默认5）
- **同步时间戳**：训练日志和结果文件使用相同时间戳

## 评估指标

### 分类性能

- **Top-1 准确率**：最高概率类别的准确率
- **Top-5 准确率**：前5个最高概率类别的准确率
- **混淆矩阵**：10×10 类别混淆矩阵
- **分类报告**：精确率、召回率、F1分数

### 训练监控

- **损失曲线**：训练/验证损失变化
- **准确率曲线**：训练/验证准确率变化
- **学习率曲线**：学习率调度变化
- **性能指标**：GPU使用率、内存占用、训练时间

## 可视化功能

### 训练可视化

- **训练曲线**：损失和准确率变化图
- **学习率调度**：学习率变化曲线
- **性能监控**：资源使用率图表

### 模型可视化

- **混淆矩阵**：热力图形式的分类结果
- **预测示例**：正确和错误预测的样本
- **特征图**：不同层的特征提取结果
- **卷积核可视化**：第一层卷积核权重可视化
- **激活热力图**：残差块的激活模式

### 数据可视化

- **数据分布**：各类别样本数量统计
- **数据增强**：增强前后的对比图
- **样本展示**：随机展示各类别样本

## 文件说明

### 核心模块

- **`src/data.py`**：CIFAR-10 数据加载器，支持数据增强、分割和加载优化
- **`src/model.py`**：ResNet18 模型定义，包含 BasicBlock 实现和Dropout支持
- **`src/trainer.py`**：训练器类，封装训练/验证逻辑，集成日志管理器
- **`src/utils.py`**：工具函数，包含检查点、指标计算等功能
- **`src/main.py`**：主程序，命令行接口，使用ResNet日志管理器
- **`src/config_parser.py`**：配置解析器，支持预设和参数映射
- **`src/resnet_log_manager.py`**：专用日志管理系统，支持训练/评估模式分离
- **`src/visualization.py`**：可视化功能模块，包含卷积核可视化

### 配置和测试

- **`configs/config_performance.yaml`**：性能优化配置，包含预设和优化参数
- **`test/test_modules.py`**：核心模块单元测试脚本
- **`test/test_compatibility.py`**：兼容性测试脚本

## 实验结果

### 预期性能

根据项目基准要求，期望达到以下性能：

- **训练集准确率**：90%+
- **验证集准确率**：85%+
- **训练时间**：每个 epoch < 5 分钟

### 可复现性

项目使用固定随机种子（42），确保结果可复现：

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

## 注意事项

1. **GPU 要求**：建议使用 CUDA 兼容的 GPU，显存 ≥ 4GB
2. **内存需求**：系统内存 ≥ 8GB
3. **存储空间**：项目完整运行需要约 2GB 存储空间
4. **Python 版本**：支持 Python 3.8+

## 故障排除

### 常见问题

1. **CUDA 内存不足**：减小 batch_size 或使用梯度累积
2. **训练过慢**：检查 num_workers 设置，使用混合精度训练
3. **精度不收敛**：调整学习率，检查数据增强策略
4. **可视化错误**：确保输出目录存在，检查文件权限

### 调试技巧

```bash
# 使用快速测试预设
python src/main.py --performance_preset quick_test

# 评估已训练模型
python src/main.py --run_mode evaluation --model_path outputs/models/best_model.pth

# 调整日志频率
python src/main.py --log_frequency 10 --save_frequency 3
```

## 性能优化特性

### 数据加载优化
- **预取机制**：通过 `prefetch_factor` 参数预加载数据
- **持久化工作进程**：避免重复创建进程，提升效率
- **自适应配置**：单线程模式下自动禁用优化参数

### Dropout正则化
- **可配置dropout率**：支持不同训练场景的正则化需求
- **自适应应用**：在训练时启用，评估时自动禁用

### 日志系统优化
- **频率控制**：可调节的日志记录频率，减少I/O开销
- **分离式设计**：训练和评估使用不同的日志策略
- **结构化输出**：txt文件包含纯净的结构化结果数据

## 最新功能特性

### v1.2 更新内容

- **专业日志系统**：全新的 `ResNetLogManager`，支持训练/评估模式分离
- **配置系统重构**：简化为快速测试和基准两种预设，支持命令行覆盖
- **性能优化**：数据加载预取、持久化工作进程、可配置Dropout
- **可视化增强**：新增卷积核权重可视化功能
- **日志频率控制**：可调节的训练进度和详细摘要保存频率

## 许可证

本项目基于 MIT 许可证开源。

## 贡献指南

欢迎提交 Issue 和 Pull Request！请确保：

1. 代码符合 PEP 8 规范
2. 添加必要的测试
3. 更新相关文档
4. 提交信息清晰明确

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues：在本仓库提交 Issue
- 项目维护者：[您的联系方式]

---

**祝您使用愉快！Happy Deep Learning! 🚀**

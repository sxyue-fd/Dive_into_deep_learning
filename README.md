# 深度学习实践项目

本项目是《动手学深度学习》(Dive into Deep Learning) 教材的实践项目集合。

## 项目结构

```
Dive_into_deep_learning/
├── Chpt_3_FashionMNIST/      # Fashion-MNIST分类项目
│   ├── 使用Softmax回归和MLP的实现
├── Chpt_3_LinearRegression/   # 线性回归项目
│   ├── 从零实现和简化实现
├── Chpt_3_Kaggle_house_price/ # Kaggle房价预测项目
│   └── 完整的机器学习工作流程
```

## 项目特点

- 模块化设计：每个子项目都采用标准的项目结构
- 配置驱动：使用yaml文件统一管理参数
- 完整记录：包含训练日志和可视化结果
- 版本控制：适当的.gitignore配置
- 文档完善：详细的README和注释

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

1. 克隆项目：
```bash
git clone <repository-url>
cd Dive_into_deep_learning
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行子项目：
- 每个子项目都有自己的README.md，包含详细说明

## 子项目说明

### 1. Fashion-MNIST分类

实现了两种方法对Fashion-MNIST数据集进行分类：
- Softmax回归：简单直接的分类器
- 多层感知机(MLP)：具有隐藏层的神经网络

### 2. 线性回归

提供了线性回归的两种实现方式：
- 从零实现：手动实现梯度计算和参数更新
- 简化实现：使用PyTorch的自动微分

### 3. Kaggle房价预测

完整的机器学习项目实践：
- 数据预处理
- 特征工程
- 模型训练
- 交叉验证
- 结果分析

## 参考资料

1. Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola. Dive into Deep Learning. 2023.
   - 项目基于此教材的实践内容
   - [教材链接](https://d2l.ai/)

2. 《动手学深度学习》PyTorch实现
   - [GitHub实现](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
   - [在线版本](https://tangshusen.me/Dive-into-DL-PyTorch/#/)

3. PyTorch官方文档
   - [pytorch.org](https://pytorch.org/docs/stable/index.html)

4. Fashion-MNIST数据集
   - [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

5. Kaggle房价预测竞赛
   - [kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 许可证

本项目采用 MIT License 进行许可。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 版权声明

Copyright (c) 2025

本项目中的所有代码均基于 MIT License 发布。详细信息请参阅各子项目目录中的 LICENSE 文件。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

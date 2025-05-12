import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from sklearn.preprocessing import StandardScaler

# 设置数据路径
data_dir = './data/Kaggle_house'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv')) # shape: (1460, 81)
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv')) # shape: (1459, 80)

# 数据预处理
# 1. 提取目标变量和合并特征
train_labels = train_data['SalePrice'].values.reshape(-1, 1)
all_features = pd.concat([train_data.drop(['SalePrice', 'Id'], axis=1), # 去除训练数据的房价（因为它是Label而不是特征）和id（因为编号不是有效的特征）
                          test_data.drop(['Id'], axis=1)]) # 去除测试集的id

# 2. 分别处理数值和分类特征
numeric_features = all_features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = all_features.select_dtypes(include=['object']).columns

print(f"数值特征数量: {len(numeric_features)}")
print(f"分类特征数量: {len(categorical_features)}")

# 3. 处理数值特征：填充缺失值（使用中位数）并标准化
for feature in numeric_features:
    # 计算中位数
    median_val = all_features[feature].median()
    # 填充缺失值
    all_features[feature] = all_features[feature].fillna(median_val)
    # 标准化
    mean_val = all_features[feature].mean()
    std_val = all_features[feature].std()
    all_features[feature] = (all_features[feature] - mean_val) / std_val

# 4. 处理分类特征：填充缺失值（使用最频繁值）
for feature in categorical_features:
    # 找出最常见的值
    most_frequent = all_features[feature].mode()[0]
    # 填充缺失值
    all_features[feature] = all_features[feature].fillna(most_frequent)

# 5. 独热编码（保留缺失值列）
all_features = pd.get_dummies(all_features, columns=categorical_features, 
                             dummy_na=False, drop_first=False)

print(f"处理后的特征维度: {all_features.shape}")

# 6. 分离回训练集和测试集
n_train = train_data.shape[0]
train_features = all_features[:n_train].values.astype(np.float32)
test_features = all_features[n_train:].values.astype(np.float32)

# 7. 标准化标签
label_scaler = StandardScaler()
train_labels_scaled = label_scaler.fit_transform(train_labels)

# 8. 转换为PyTorch张量
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_scaled, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

print(f"训练特征张量形状: {train_features_tensor.shape}")
print(f"训练标签张量形状: {train_labels_tensor.shape}")
print(f"测试特征张量形状: {test_features_tensor.shape}")

# 创建数据集
train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)

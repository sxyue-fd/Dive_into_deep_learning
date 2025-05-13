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
from sklearn.model_selection import KFold

# ====================== 数据加载与预处理 ======================
# 设置数据路径并加载数据集
data_dir = './data/Kaggle_house'
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # 训练数据: 1460样本，81特征
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))    # 测试数据: 1459样本，80特征

# 特征工程第一步：提取目标变量和特征
# - 从训练集分离出房价（目标变量）
# - 合并训练集和测试集的特征以保证一致的预处理
train_labels = train_data['SalePrice'].values.reshape(-1, 1)
all_features = pd.concat([
    train_data.drop(['SalePrice', 'Id'], axis=1),  # 移除训练集的房价和ID
    test_data.drop(['Id'], axis=1)                 # 移除测试集的ID
])

# 特征工程第二步：区分数值特征和分类特征
# - 根据数据类型自动识别特征类型
# - 数值特征：整数和浮点数
# - 分类特征：对象类型（字符串）
numeric_features = all_features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = all_features.select_dtypes(include=['object']).columns

print(f"数值特征数量: {len(numeric_features)}")
print(f"分类特征数量: {len(categorical_features)}")

# 特征工程第三步：处理数值特征
# - 使用中位数填充缺失值（避免异常值影响）
# - 进行标准化处理（均值为0，标准差为1）
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

# ====================== 模型定义 ======================
class HousePriceMLP(nn.Module):
    """
    房价预测的多层感知机模型
    - 使用单隐藏层结构（input_size -> hidden_size -> 1）
    - 采用ReLU激活函数
    - 使用Dropout(0.3)防止过拟合
    """
    def __init__(self, input_size, hidden_size=32):
        super(HousePriceMLP, self).__init__()
        # 隐藏层配置
        # - hidden_size个神经元
        # - ReLU激活提供非线性
        # - Dropout随机丢弃30%的神经元
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # 输出层：单个神经元（预测房价）
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

# 定义评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(data_loader.dataset)

# ====================== 训练函数定义 ======================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, fold):
    """
    模型训练函数
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数（MSE）
        optimizer: 优化器（Adam）
        device: 计算设备（CPU/GPU）
        num_epochs: 训练轮数
        fold: 当前折序号（用于显示）
    """
    # 初始化实时可视化
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # 初始化训练记录
    train_losses = []     # 训练损失历史
    train_epochs = []     # 训练步数记录
    val_losses = []       # 验证损失历史
    best_val_loss = float('inf')  # 最佳验证损失
    step_count = 0        # 训练步数计数器
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            output = model(X)
            loss = criterion(output, y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)
            
            # 每10个batch更新一次图像
            if i % 10 == 0:
                step_count += 1
                current_train_loss = train_loss / ((i + 1) * X.size(0))
                train_losses.append(current_train_loss)
                train_epochs.append(epoch + i/len(train_loader))
                  # 更新损失曲线（使用对数坐标）
                ax.clear()
                ax.semilogy(train_epochs, train_losses, label='Train Loss', alpha=0.6)
                if val_losses:  # 如果有验证损失则也画出来
                    ax.semilogy(range(1, epoch + 1), val_losses, label='Val Loss', alpha=0.6)
                    # 检测过拟合（如果验证损失比训练损失大50%以上）
                    if val_losses[-1] > current_train_loss * 1.5:
                        ax.text(0.98, 0.98, 'Warning: Overfitted', 
                               transform=ax.transAxes,
                               horizontalalignment='right',
                               verticalalignment='top',
                               color='red',
                               bbox=dict(facecolor='white', alpha=0.8))
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss (log scale)')
                ax.set_title(f'Training Progress (Fold {fold+1})')
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                plt.pause(0.1)
          # 计算epoch的平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    plt.ioff()
    return train_losses, val_losses

# ====================== 训练配置 ======================
# 模型超参数
input_size = train_features.shape[1]    # 输入特征维度
hidden_size = 32                        # 隐藏层神经元数量
dropout_rate = 0.3                      # Dropout比率
batch_size = 64                         # 批量大小
num_epochs = 50                         # 训练轮数
learning_rate = 0.001                   # 学习率
weight_decay = 0.05                     # L2正则化系数
k_folds = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== K折交叉验证设置 ======================
# 初始化K折交叉验证
kf = KFold(n_splits=k_folds,            # 5折交叉验证
           shuffle=True,                 # 随机打乱数据
           random_state=42)             # 固定随机种子以复现结果

# 初始化性能记录器
fold_train_losses = []                  # 记录每折的训练损失
fold_val_losses = []                    # 记录每折的验证损失
best_fold = 0                           # 记录最佳性能的折
best_val_loss = float('inf')            # 记录最佳验证损失

# K折交叉验证训练循环
for fold, (train_idx, val_idx) in enumerate(kf.split(train_features_tensor)):
    print(f'\nFold {fold + 1}/{k_folds}')
    
    # 准备数据加载器
    train_data = torch.utils.data.TensorDataset(
        train_features_tensor[train_idx], 
        train_labels_tensor[train_idx]
    )
    val_data = torch.utils.data.TensorDataset(
        train_features_tensor[val_idx], 
        train_labels_tensor[val_idx]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size
    )    # 创建模型、损失函数和优化器
    model = HousePriceMLP(input_size, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 添加L2正则化
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, device, num_epochs, fold
    )
      # 记录最终训练和验证损失
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    fold_train_losses.append(final_train_loss)
    fold_val_losses.append(final_val_loss)    # 更新最佳验证损失（仅用于记录）
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_fold = fold

# 打印最终结果
print('\n交叉验证结果:')
print(f'{"折数":^6} {"训练MSE":^12} {"验证MSE":^12}')
print('-' * 32)
for fold in range(k_folds):
    print(f'{fold + 1:^6d} {fold_train_losses[fold]:^12.4f} {fold_val_losses[fold]:^12.4f}')
print('-' * 32)
print(f'平均MSE: 训练={sum(fold_train_losses) / k_folds:.4f}, '
      f'验证={sum(fold_val_losses) / k_folds:.4f}')

# ====================== 记录训练结果 ======================
# 获取当前时间作为记录标识
current_time = time.strftime('%Y%m%d_%H%M%S')
result_file = 'training_results.txt'

# 准备记录内容
avg_train_mse = sum(fold_train_losses) / k_folds
avg_val_mse = sum(fold_val_losses) / k_folds

result_content = f"""
{'='*50}
训练时间: {current_time}
{'='*50}
超参数配置:
- 隐藏层大小 (hidden_size): {hidden_size}
- Dropout率 (dropout_rate): {dropout_rate}
- 批量大小 (batch_size): {batch_size}
- 训练轮数 (num_epochs): {num_epochs}
- 学习率 (learning_rate): {learning_rate}
- L2正则化系数 (weight_decay): {weight_decay}
- 交叉验证折数 (k_folds): {k_folds}

训练结果:
- 平均训练MSE: {avg_train_mse:.4f}
- 平均验证MSE: {avg_val_mse:.4f}
- 最佳验证MSE (Fold {best_fold + 1}): {best_val_loss:.4f}

各折详细结果:
{'折数':^6} {'训练MSE':^12} {'验证MSE':^12}
{'-'*32}
"""

# 添加每折的详细结果
for fold in range(k_folds):
    result_content += f"{fold + 1:^6d} {fold_train_losses[fold]:^12.4f} {fold_val_losses[fold]:^12.4f}\n"

result_content += f"{'-'*32}\n\n"

# 写入结果文件
with open(result_file, 'a', encoding='utf-8') as f:
    f.write(result_content)
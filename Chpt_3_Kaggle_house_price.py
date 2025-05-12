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

# 定义MLP模型
class HousePriceMLP(nn.Module):
    def __init__(self, input_size):
        super(HousePriceMLP, self).__init__()
        # 第一层: input_size -> 256
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # 第二层: 256 -> 128
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # 输出层: 128 -> 1
        self.output = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
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

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, fold):
    # 创建图形
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    train_losses = []
    train_epochs = []
    val_losses = []
    best_val_loss = float('inf')
    step_count = 0
    
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')  # 保存最佳模型（使用Python文件名作为前缀）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = f'Chpt_3_Kaggle_house_price_model_fold{fold+1}.pth'
            torch.save(model.state_dict(), model_filename)
    
    plt.ioff()
    return train_losses, val_losses

# 设置超参数和训练配置
input_size = train_features.shape[1]
batch_size = 64
num_epochs = 50
learning_rate = 0.001
k_folds = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备K折交叉验证
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 记录每折的性能
fold_train_losses = []
fold_val_losses = []
best_fold = 0
best_val_loss = float('inf')

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
    )
    
    # 创建模型、损失函数和优化器
    model = HousePriceMLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, device, num_epochs, fold
    )
      # 记录最终训练和验证损失
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    fold_train_losses.append(final_train_loss)
    fold_val_losses.append(final_val_loss)
    
    # 如果这一折的验证损失是目前最好的，保存模型
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_fold = fold
        # 保存新的最佳模型，并删除之前的模型
        model_filename = f'Chpt_3_Kaggle_house_price_best_model.pth'
        torch.save(model.state_dict(), model_filename)
        
# 打印最终结果
print('\n交叉验证结果:')
print(f'{"折数":^6} {"训练MSE":^12} {"验证MSE":^12}')
print('-' * 32)
for fold in range(k_folds):
    print(f'{fold + 1:^6d} {fold_train_losses[fold]:^12.4f} {fold_val_losses[fold]:^12.4f}')
print('-' * 32)
print(f'平均MSE: 训练={sum(fold_train_losses) / k_folds:.4f}, '
      f'验证={sum(fold_val_losses) / k_folds:.4f}')
print(f'\n最佳模型来自第 {best_fold + 1} 折，验证MSE = {best_val_loss:.4f}')

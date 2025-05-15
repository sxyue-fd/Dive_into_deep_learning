import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class HousePriceDataset:
    """房价预测数据集处理类"""
    
    def __init__(self, config):
        """
        初始化数据集处理器
        
        Args:
            config (dict): 配置字典，包含数据路径等信息
        """
        self.config = config
        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()
    
    def load_and_preprocess(self):
        """加载并预处理数据"""
        # 获取项目根目录
        base_dir = Path(__file__).resolve().parent.parent
        
        # 加载原始数据
        train_file = base_dir / self.config['data']['train_file']
        test_file = base_dir / self.config['data']['test_file']
        
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        
        # 提取目标变量和特征
        train_labels = train_data['SalePrice'].values.reshape(-1, 1)
        features = self._process_features(train_data, test_data)
        
        # 标准化标签
        train_labels_scaled = self.label_scaler.fit_transform(train_labels)
        
        # 特征预处理
        features_array = features.values.astype(np.float32)
        
        # 转换为张量
        train_features = torch.tensor(features_array[:len(train_data)], dtype=torch.float32)
        test_features = torch.tensor(features_array[len(train_data):], dtype=torch.float32)
        train_labels = torch.tensor(train_labels_scaled, dtype=torch.float32)
        
        return train_features, train_labels, test_features
    
    def _process_features(self, train_data, test_data):
        """特征处理"""
        # 合并训练集和测试集以进行一致的特征处理
        all_features = pd.concat([
            train_data.drop(['SalePrice', 'Id'], axis=1),
            test_data.drop(['Id'], axis=1)
        ])
        
        # 区分数值特征和分类特征
        numeric_features = all_features.select_dtypes(
            include=['int64', 'float64']).columns
        categorical_features = all_features.select_dtypes(
            include=['object']).columns
            
        # 处理数值特征
        for feature in numeric_features:
            # 使用中位数填充缺失值
            median_val = all_features[feature].median()
            all_features[feature] = all_features[feature].fillna(median_val)
            # 标准化
            mean_val = all_features[feature].mean()
            std_val = all_features[feature].std()
            all_features[feature] = (all_features[feature] - mean_val) / std_val
            
        # 处理分类特征
        for feature in categorical_features:
            most_frequent = all_features[feature].mode()[0]
            all_features[feature] = all_features[feature].fillna(most_frequent)
            
        # 独热编码
        all_features = pd.get_dummies(
            all_features, 
            columns=categorical_features,
            dummy_na=False,
            drop_first=False
        )
        
        return all_features
        
    def get_loaders(self, train_features, train_labels, batch_size, k_fold_indices=None):
        """创建数据加载器"""
        if k_fold_indices is not None:
            train_idx, val_idx = k_fold_indices
            # 创建训练集加载器
            train_data = TensorDataset(
                train_features[train_idx],
                train_labels[train_idx]
            )
            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True
            )
            
            # 创建验证集加载器
            val_data = TensorDataset(
                train_features[val_idx],
                train_labels[val_idx]
            )
            val_loader = DataLoader(
                val_data,
                batch_size=batch_size
            )
            return train_loader, val_loader
        else:
            # 创建完整训练集的加载器
            dataset = TensorDataset(train_features, train_labels)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
    def inverse_transform_labels(self, scaled_labels):
        """将标准化的标签转换回原始范围"""
        return self.label_scaler.inverse_transform(scaled_labels)
            
    def inverse_transform_labels(self, scaled_labels):
        """将标准化的标签转换回原始范围"""
        return self.label_scaler.inverse_transform(scaled_labels)

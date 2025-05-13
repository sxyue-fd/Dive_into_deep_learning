import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import time
from sklearn.model_selection import KFold

from data import HousePriceDataset
from model import HousePriceMLP
from trainer import ModelTrainer
from utils import setup_seed, load_config, get_device, save_training_results

def main():
    """主函数"""
    # 加载配置
    config = load_config('configs/config.yaml')
    config['timestamp'] = time.strftime('%Y%m%d_%H%M%S')
    
    # 设置随机种子
    setup_seed(42)
    
    # 获取设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 准备数据
    dataset = HousePriceDataset(config)
    train_features, train_labels, test_features = dataset.load_and_preprocess()
    
    # 更新输入维度
    config['model']['input_size'] = train_features.shape[1]
    
    # 初始化K折交叉验证
    kf = KFold(
        n_splits=config['training']['k_folds'],
        shuffle=True,
        random_state=42
    )
    
    # 记录每折的结果
    fold_results = []
    
    # K折交叉验证训练循环
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        print(f'\nTraining Fold {fold + 1}/{config["training"]["k_folds"]}')
        
        # 准备数据加载器
        train_loader, val_loader = dataset.get_loaders(
            train_features,
            train_labels,
            config['training']['batch_size'],
            (train_idx, val_idx)
        )
        
        # 创建模型
        model = HousePriceMLP(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            dropout_rate=config['model']['dropout_rate']
        ).to(device)
        
        # 设置损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 创建训练器
        trainer = ModelTrainer(model, criterion, optimizer, device, config)
        
        # 训练模型
        train_losses, val_losses = trainer.train_fold(
            train_loader,
            val_loader,
            fold
        )
        
        # 记录结果
        fold_results.append({
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1]
        })
    
    # 保存训练结果
    save_training_results(
        config,
        fold_results,
        Path(config['paths']['log_dir']) / 'training_results.txt'
    )

if __name__ == "__main__":
    main()

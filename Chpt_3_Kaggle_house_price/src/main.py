import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import time
import logging
from sklearn.model_selection import KFold

from data import HousePriceDataset
from model import HousePriceMLP
from trainer import ModelTrainer
from utils import (
    setup_seed, 
    load_config, 
    get_device, 
    save_training_results,
    setup_logging,
    get_session_id
)

def main():
    """主函数"""
    # 生成会话ID
    session_id = get_session_id()
    
    # 加载配置
    config = load_config('configs/config.yaml')
    
    # 设置日志
    log_file = setup_logging(config, session_id)
    logging.info(f"Starting new training session: {session_id}")
    
    # 设置随机种子
    setup_seed(42)
    
    # 获取设备
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # 准备数据
    dataset = HousePriceDataset(config)
    train_features, train_labels, test_features = dataset.load_and_preprocess()
    
    # 收集数据集信息
    dataset_info = {
        'total_samples': len(train_features) + len(test_features),
        'train_samples': len(train_features),
        'val_samples': len(train_features) // config['training']['k_folds'],  # 每折的验证集大小
        'original_features': dataset.original_feature_dim,  # 需要在HousePriceDataset中添加这个属性
        'processed_features': train_features.shape[1],
        'numerical_features': len(dataset.numerical_features),  # 需要在HousePriceDataset中添加这个属性
        'categorical_features': len(dataset.categorical_features)  # 需要在HousePriceDataset中添加这个属性
    }
    logging.info(f"数据集信息：总样本数 {dataset_info['total_samples']}, "
                f"特征维度 {dataset_info['processed_features']}")
    
    # 更新输入维度
    config['model']['input_size'] = train_features.shape[1]
    logging.info(f"Model input size set to: {config['model']['input_size']}")
    
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
        logging.info(f'\nTraining Fold {fold + 1}/{config["training"]["k_folds"]}')
        
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
        trainer = ModelTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            dataset=dataset,
            session_id=session_id
        )
        
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
        
        logging.info(f"Fold {fold + 1} completed - Train MSE: {train_losses[-1]:.4f}, Val MSE: {val_losses[-1]:.4f}")
    
    # 保存训练结果
    result_file = save_training_results(
        config, 
        fold_results, 
        session_id,
        dataset_info
    )
    logging.info(f"Training completed. Results saved to: {result_file}")

if __name__ == "__main__":
    main()

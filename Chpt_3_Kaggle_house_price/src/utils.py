import logging
import yaml
from pathlib import Path
import torch
import random
import numpy as np

def setup_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device():
    """获取计算设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_training_results(config, fold_results, output_file):
    """
    保存训练结果
    
    Args:
        config: 训练配置
        fold_results: 每个fold的训练结果
        output_file: 输出文件路径
    """
    avg_train_mse = np.mean([res['train_loss'] for res in fold_results])
    avg_val_mse = np.mean([res['val_loss'] for res in fold_results])
    best_fold_idx = np.argmin([res['val_loss'] for res in fold_results])
    best_val_loss = fold_results[best_fold_idx]['val_loss']
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"训练时间: {config['timestamp']}\n")
        f.write(f"{'='*50}\n")
        
        # 写入超参数配置
        f.write("超参数配置:\n")
        for key, value in config['model'].items():
            f.write(f"- {key}: {value}\n")
        for key, value in config['training'].items():
            f.write(f"- {key}: {value}\n")
        
        # 写入训练结果
        f.write("\n训练结果:\n")
        f.write(f"- 平均训练MSE: {avg_train_mse:.4f}\n")
        f.write(f"- 平均验证MSE: {avg_val_mse:.4f}\n")
        f.write(f"- 最佳验证MSE (Fold {best_fold_idx + 1}): {best_val_loss:.4f}\n")
        
        # 写入每折的详细结果
        f.write("\n各折详细结果:\n")
        f.write(f"{'折数':^6} {'训练MSE':^12} {'验证MSE':^12}\n")
        f.write("-" * 32 + "\n")
        
        for i, res in enumerate(fold_results):
            f.write(f"{i+1:^6d} {res['train_loss']:^12.4f} {res['val_loss']:^12.4f}\n")
        
        f.write("-" * 32 + "\n\n")

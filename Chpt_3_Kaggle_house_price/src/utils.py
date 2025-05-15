import logging
import yaml
from pathlib import Path
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def setup_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent

def load_config(config_path):
    """加载配置文件"""
    base_dir = get_project_root()
    # 将配置路径解析为相对于项目根目录的完整路径
    full_config_path = base_dir / config_path
    
    with open(full_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device():
    """获取计算设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_absolute_path(relative_path):
    """将相对路径转换为绝对路径"""
    return get_project_root() / relative_path

def save_training_results(config, fold_results, output_file):
    """
    保存训练结果
    
    Args:
        config: 训练配置
        fold_results: 每个fold的训练结果
        output_file: 输出文件路径（相对于项目根目录）
    """
    avg_train_mse = np.mean([res['train_loss'] for res in fold_results])
    avg_val_mse = np.mean([res['val_loss'] for res in fold_results])
    best_fold_idx = np.argmin([res['val_loss'] for res in fold_results])
    best_val_loss = fold_results[best_fold_idx]['val_loss']
      # 确保输出目录存在
    output_path = get_absolute_path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'a', encoding='utf-8') as f:
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

def plot_training_progress(train_epochs, train_losses, val_losses, current_train_loss, fold, output_path):
    """
    更新和保存训练过程可视化
    
    Args:
        train_epochs: 训练轮次
        train_losses: 训练损失
        val_losses: 验证损失
        current_train_loss: 当前训练损失
        fold: 当前折数
        output_path: 输出图像路径（相对于项目根目录）
    """
    # 确保输出目录存在
    output_path = get_absolute_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_epochs, train_losses, label='Train Loss', alpha=0.7, color='#1f77b4')
    
    if val_losses:  # 如果有验证损失则画出来
        plt.semilogy(range(1, len(val_losses) + 1), val_losses, label='Val Loss', alpha=0.7, color='#ff7f0e')
        # 检测过拟合
        if val_losses[-1] > current_train_loss * 1.5:
            plt.text(0.98, 0.98, 'Warning: Overfitting', 
                   transform=plt.gca().transAxes,
                   horizontalalignment='right',
                   verticalalignment='top',
                   color='red',
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Epochs/Steps')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Training Progress (Fold {fold+1})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
      # 保存图像
    plt.savefig(output_path)
    plt.close()

def plot_predictions_vs_actual(y_pred, y_true, output_path, title="预测值 vs 实际值"):
    """
    绘制预测值与实际值的对比散点图
    
    Args:
        y_pred: 模型预测值
        y_true: 实际值
        output_path: 输出图像路径（相对于项目根目录）
        title: 图表标题
    """
    # 确保输出目录存在
    output_path = get_absolute_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 添加对角线（理想预测线）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """
    绘制特征重要性（适用于线性模型）
    
    Args:
        model: 训练好的线性模型
        feature_names: 特征名称列表
        output_path: 输出图像路径（相对于项目根目录）
        top_n: 展示前n个最重要的特征
    """
    # 确保输出目录存在
    output_path = get_absolute_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 获取模型权重
    weights = model.linear.weight.data.cpu().numpy().flatten()
    
    # 创建特征重要性的DataFrame
    import pandas as pd
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(weights)
    })
    
    # 按重要性排序并选择前top_n个
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('特征重要性（权重绝对值）')
    plt.ylabel('特征名称')
    plt.title(f'Top {top_n} 特征重要性')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

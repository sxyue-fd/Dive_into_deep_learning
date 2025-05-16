"""
Copyright (c) 2025
此项目基于 MIT License 进行许可
有关详细信息，请参阅项目根目录中的 LICENSE 文件
"""

"""
工具函数模块
"""
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_true_params(config):
    """
    获取真实的模型参数
    
    Args:
        config: 配置对象
    
    Returns:
        tuple: (w, b) 真实的权重和偏差
    """
    return np.array(config['data']['true_w']), config['data']['true_b']

def plot_training_curves(train_losses, test_losses, output_path):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        output_path: 输出图像的路径
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label="Train Loss", color='#1f77b4', alpha=0.7)
    plt.semilogy(test_losses, label="Test Loss", color='#ff7f0e', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.close()

def plot_training_progress(epoch, train_losses, test_losses, output_path):
    """
    实时更新和保存训练进度可视化
    
    Args:
        epoch: 当前轮次
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        output_path: 输出图像路径
    """
    # 清除当前轴而不是创建新图形，避免闪烁
    plt.clf()
    
    plt.semilogy(train_losses, label="Train Loss", color='#1f77b4', alpha=0.7)
    plt.semilogy(test_losses, label="Test Loss", color='#ff7f0e', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Training Progress (Epoch {epoch+1})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 检测过拟合
    if len(test_losses) > 5 and test_losses[-1] > train_losses[-1] * 1.5:
        plt.text(0.98, 0.98, 'Warning: Overfitting', 
                transform=plt.gca().transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                color='red',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存图像并刷新显示
    plt.savefig(output_path)
    plt.draw()
    plt.pause(0.1)  # 短暂暂停以允许绘图更新，时间更短减少闪烁

def visualize_predictions(X, y_true, y_pred, output_path, true_w=None, true_b=None, learned_w=None, learned_b=None):
    """
    可视化预测结果
    
    Args:
        X: 特征数据
        y_true: 真实标签
        y_pred: 预测标签 (不使用，预测直接使用学习到的参数计算)
        output_path: 输出图像的路径
        true_w: 真实权重，默认为None
        true_b: 真实偏置，默认为None
        learned_w: 学习到的权重，默认为None
        learned_b: 学习到的偏置，默认为None
    """
    # 绘制预测结果
    plt.figure(figsize=(12, 5))
    
    # 根据学习到的参数计算模型预测值 (不带噪声)
    y_pred_pure = np.dot(X, learned_w) + learned_b
      # 特征1的预测
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y_true, c="#1f77b4", label="True Data (with noise)", alpha=0.6)
    
    # 添加拟合线 - 特征1
    if learned_w is not None and learned_b is not None:
        x_sorted = np.sort(X[:, 0])
        # 计算一维特征对应的预测值，考虑所有特征的平均影响
        if X.shape[1] > 1:  # 如果有多个特征
            mean_other_features = np.mean(X[:, 1:], axis=0)
            y_fit = learned_w[0] * x_sorted
            for i in range(1, len(learned_w)):
                y_fit += learned_w[i] * mean_other_features[i-1]
            y_fit += learned_b
        else:  # 只有一个特征
            y_fit = learned_w[0] * x_sorted + learned_b
        
        plt.plot(x_sorted, y_fit, '-r', linewidth=2, label="Learned Regression Line")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.title("Predictions by Feature 1")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
      # 特征2的预测
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 1], y_true, c="#1f77b4", label="True Data (with noise)", alpha=0.6)
    
    # 添加拟合线 - 特征2
    if learned_w is not None and learned_b is not None and X.shape[1] > 1:
        x_sorted = np.sort(X[:, 1])
        # 计算第二个特征对应的预测值，考虑其他特征的平均影响
        mean_other_features = np.mean(X[:, [0] + list(range(2, X.shape[1]))], axis=0)
        y_fit = learned_w[1] * x_sorted
        y_fit += learned_w[0] * mean_other_features[0]
        if X.shape[1] > 2:  # 如果有超过2个特征
            for i in range(2, len(learned_w)):
                y_fit += learned_w[i] * mean_other_features[i-1]
        y_fit += learned_b
        
        plt.plot(x_sorted, y_fit, '-r', linewidth=2, label="Learned Regression Line")
    
    plt.xlabel("Feature 2")
    plt.ylabel("Target")
    plt.title("Predictions by Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
      # 如果提供了权重和偏置且数据集有两个特征，绘制3D图
    if true_w is not None and true_b is not None and learned_w is not None and learned_b is not None and X.shape[1] > 1:
        # 随机采样100个点用于散点图显示，避免图像过于拥挤
        idx = np.random.choice(len(X), size=100, replace=False)
        sampled_X = X[idx]
        sampled_y = y_true[idx]
        
        # 根据学习到的参数计算采样点的模型预测值
        sampled_y_pred = np.dot(sampled_X, learned_w) + learned_b
        
        fig = plt.figure(figsize=(12, 5))
        
        # 创建数据点的3D散点图 - 真实平面
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(sampled_X[:, 0], sampled_X[:, 1], sampled_y, 
                    c='b', marker='o', s=30, alpha=0.6, label='Original Data (with noise)')
        
        # 创建网格点
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 50),
                            np.linspace(x2_min, x2_max, 50))
        
        # 计算真实平面
        z_true = true_w[0] * x1 + true_w[1] * x2 + true_b
        ax1.plot_surface(x1, x2, z_true, alpha=0.3, color='r')
        
        # 计算真实平面上的点
        true_pred = np.dot(sampled_X, true_w) + true_b
        
        # 显示真实平面上的点
        ax1.scatter(sampled_X[:, 0], sampled_X[:, 1], true_pred, 
                    c='r', marker='^', s=30, alpha=0.8, label='True Points (on plane)')
        
        ax1.set_title('True Regression Plane')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('Y')
        ax1.legend()
        
        # 绘制学习到的平面
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(sampled_X[:, 0], sampled_X[:, 1], sampled_y, 
                    c='b', marker='o', s=30, alpha=0.6, label='Original Data (with noise)')
        
        # 计算学习到的平面
        z_pred = learned_w[0] * x1 + learned_w[1] * x2 + learned_b
        ax2.plot_surface(x1, x2, z_pred, alpha=0.3, color='g')
        
        # 显示学习到的平面上的点
        ax2.scatter(sampled_X[:, 0], sampled_X[:, 1], sampled_y_pred, 
                   c='g', marker='^', s=30, alpha=0.8, label='Predicted Points (on plane)')
        
        ax2.set_title('Learned Regression Plane')
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('Y')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(str(output_path).replace('.png', '_3d.png'))
        plt.close()

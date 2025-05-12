import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_progress(steps, losses, true_params, learned_params):
    """绘制训练进度，包括损失曲线和参数对比
    Args:
        steps: 训练步数列表
        losses: 损失值列表
        true_params: 真实参数(w1, w2, b)的元组
        learned_params: 学习到的参数(w1, w2, b)的元组
    """
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})

    # 绘制损失曲线（对数坐标）
    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, losses, 'r-', linewidth=2)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.title('Training Loss vs. Steps (Log Scale)', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # 在图中添加参数对比信息
    true_w1, true_w2, true_b = true_params
    learned_w1, learned_w2, learned_b = learned_params
    
    info_text = f'True parameters:\nw1={true_w1:.1f}, w2={true_w2:.1f}, b={true_b:.1f}\n\nLearned parameters:\nw1={learned_w1:.1f}, w2={learned_w2:.1f}, b={learned_b:.1f}'
    plt.text(0.5, 0.95, info_text,
             transform=plt.gca().transAxes,
             horizontalalignment='center',
             verticalalignment='top',
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, pad=10))

    plt.show()

def plot_regression_result(features, labels, true_w, true_b, learned_w, learned_b):
    """绘制回归结果的3D可视化，通过采样减少散点数量
    """
    # 随机采样1000个点中的100个点用于散点图显示
    idx = np.random.choice(len(features), size=100, replace=False)
    sampled_features = features[idx]
    sampled_labels = labels[idx]
    
    fig = plt.figure(figsize=(12, 5))
    
    # 创建数据点的3D散点图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(sampled_features[:, 0], sampled_features[:, 1], sampled_labels, 
                c='b', marker='o', s=30, alpha=0.6, label='Original Data')
    
    # 创建网格点
    x1_min, x1_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    x2_min, x2_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 50),
                         np.linspace(x2_min, x2_max, 50))
    
    # 计算真实平面
    z_true = true_w[0] * x1 + true_w[1] * x2 + true_b
    ax1.plot_surface(x1, x2, z_true, alpha=0.3, color='r')
    ax1.set_title('True Regression Plane')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')
    
    # 绘制学习到的平面
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(sampled_features[:, 0], sampled_features[:, 1], sampled_labels, 
                c='b', marker='o', s=30, alpha=0.6, label='Original Data')
    z_pred = learned_w[0] * x1 + learned_w[1] * x2 + learned_b
    ax2.plot_surface(x1, x2, z_pred, alpha=0.3, color='g')
    ax2.set_title('Learned Regression Plane')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Y')
    
    plt.tight_layout()
    plt.show()
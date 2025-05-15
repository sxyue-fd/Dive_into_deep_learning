"""
工具函数模块
"""
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
try:
    from .data import get_fashion_mnist_labels
except ImportError:
    # 当直接运行此模块时使用
    from data import get_fashion_mnist_labels

def show_fashion_mnist(images, labels):
    """
    显示Fashion-MNIST图像和标签
    
    Args:
        images: 图像数据，形状为(n, 1, 28, 28)
        labels: 标签数据，形状为(n,)
    """
    labels = labels.numpy()
    class_names = get_fashion_mnist_labels()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(class_names[lbl])
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_config(config_path):
    """
    加载YAML配置文件并验证关键参数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在时
        ValueError: 配置文件格式错误或缺少必要参数时
        Exception: 其他错误
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证关键配置项是否存在
        required_keys = {
            'data': ['dataset', 'batch_size', 'num_classes', 'input_size'],
            'model': ['type', 'hidden_units', 'dropout_rate'],
            'training': ['learning_rate', 'weight_decay', 'num_epochs', 'device'],
            'logging': ['log_interval']
        }
        
        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"配置文件中缺少 '{section}' 部分")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"配置文件中缺少 '{section}.{key}' 参数")
        
        # 验证数值参数的合理性
        if float(config['training']['learning_rate']) <= 0:
            raise ValueError("学习率必须大于0")
        if float(config['training']['weight_decay']) < 0:
            raise ValueError("权重衰减必须大于等于0")
        if int(config['training']['num_epochs']) <= 0:
            raise ValueError("训练轮数必须大于0")
        if int(config['data']['batch_size']) <= 0:
            raise ValueError("批次大小必须大于0")
        if not (0 <= float(config['model']['dropout_rate']) < 1):
            raise ValueError("Dropout率必须在[0, 1)范围内")
            
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {str(e)}")
    except Exception as e:
        raise Exception(f"加载配置文件时出错: {str(e)}")

def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    import torch
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def plot_training_curves(train_losses, valid_losses, train_accs, valid_accs, steps_per_point, output_path):
    """
    绘制FashionMNIST训练曲线
    
    Args:
        train_losses: 训练损失列表
        valid_losses: 未使用，保留参数以保持兼容性
        train_accs: 训练准确率列表
        valid_accs: 未使用，保留参数以保持兼容性
        steps_per_point: 每个数据点之间的batch数
        output_path: 输出图像的路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 生成步数坐标
    steps = np.arange(len(train_losses)) * steps_per_point
    
    # 损失曲线 (使用对数坐标)
    ax1.semilogy(steps, train_losses, label='Loss', color='#1f77b4', alpha=0.7)
    ax1.set_xlabel('Steps (batches)')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 准确率曲线
    ax2.plot(steps, train_accs, label='Accuracy', color='#2ca02c', alpha=0.7)
    ax2.set_xlabel('Steps (batches)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, output_path):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵（numpy数组）
        class_names: 类别名称列表
        output_path: 输出图像路径
    """
    plt.figure(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 调整布局以防止标签被切断
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_sample_predictions(images, true_labels, pred_labels, output_path, num_samples=10):
    """
    可视化样本预测结果
    
    Args:
        images: 图像数据 [N, 1, 28, 28]
        true_labels: 真实标签
        pred_labels: 预测标签
        output_path: 输出图像路径
        num_samples: 要显示的样本数量
    """
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f'True: {get_fashion_mnist_labels()[true_labels[i]]}\nPred: {get_fashion_mnist_labels()[pred_labels[i]]}',
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_class_distribution(labels, output_path):
    """
    绘制数据集类别分布
    
    Args:
        labels: 标签数据
        output_path: 输出图像路径
    """
    class_names = get_fashion_mnist_labels()
    class_counts = np.bincount(labels)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_counts)), class_counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    
    # 在柱状图上添加具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_sample_predictions(images, true_labels, pred_labels, output_path):
    """
    可视化样本预测结果
    
    Args:
        images: 图像数据，形状为(n, 1, 28, 28)
        true_labels: 真实标签
        pred_labels: 预测标签
        output_path: 输出图像的路径
    """
    class_names = get_fashion_mnist_labels()
    n = min(len(images), 10)  # 最多显示10张图片
    
    plt.figure(figsize=(12, 2*n))
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        title = f"True: {class_names[true_labels[i]]}, Pred: {class_names[pred_labels[i]]}"
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_training_progress(epoch, train_losses, train_accs, valid_losses=None, valid_accs=None, eval_interval=1, output_path=None):
    """
    实时更新和保存训练进度可视化
    
    Args:
        epoch: 当前轮次
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        valid_losses: 验证损失列表，可选
        valid_accs: 验证准确率列表，可选
        eval_interval: 评估间隔
        output_path: 输出图像路径
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 创建网格布局
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 损失曲线（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(train_losses, label='Train Loss', color='#1f77b4', alpha=0.7,
                linewidth=2, marker='o', markersize=4)
    
    if valid_losses:
        valid_epochs = list(range(eval_interval-1, len(train_losses), eval_interval))
        if len(valid_epochs) > len(valid_losses):
            valid_epochs = valid_epochs[:len(valid_losses)]
        ax1.semilogy(valid_epochs, valid_losses, label='Valid Loss', color='#ff7f0e',
                    alpha=0.7, linewidth=2, marker='s', markersize=4)
        
        # 检测过拟合
        if len(valid_losses) > 5 and valid_losses[-1] > train_losses[-1] * 1.2:
            ax1.text(0.98, 0.98, 'Warning: Potential Overfitting',
                    transform=ax1.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 准确率曲线（右上）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train_accs, label='Train Acc', color='#2ca02c',
             alpha=0.7, linewidth=2, marker='o', markersize=4)
    
    if valid_accs:
        ax2.plot(valid_epochs, valid_accs, label='Valid Acc', color='#d62728',
                alpha=0.7, linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 训练进度条（左下）
    ax3 = fig.add_subplot(gs[1, 0])
    progress = (epoch + 1) / eval_interval
    ax3.barh(['Progress'], [progress], color='#1f77b4', alpha=0.7)
    ax3.set_title('Training Progress')
    ax3.set_xlim(0, 1)
    for i, v in enumerate([progress]):
        ax3.text(v, i, f'{v*100:.1f}%', va='center')
    
    # 性能指标表格（右下）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    cell_text = [
        ['Current Train Loss', f'{train_losses[-1]:.4f}'],
        ['Current Train Acc', f'{train_accs[-1]*100:.2f}%']
    ]
    if valid_losses and valid_accs:
        cell_text.extend([
            ['Current Valid Loss', f'{valid_losses[-1]:.4f}'],
            ['Current Valid Acc', f'{valid_accs[-1]*100:.2f}%'],
            ['Best Valid Acc', f'{max(valid_accs)*100:.2f}%']
        ])
    
    table = ax4.table(cellText=cell_text,
                     loc='center',
                     cellLoc='left',
                     edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.pause(0.1)
    plt.close()

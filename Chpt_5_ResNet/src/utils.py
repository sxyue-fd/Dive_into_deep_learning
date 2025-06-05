"""
工具函数模块
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import random
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = float(min_delta)  # 确保 min_delta 是浮点数
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            # 使用 self.min_delta 而不是闭包中的 min_delta
            self.is_better = lambda score, best: score < best - self.min_delta
        else:
            # 使用 self.min_delta 而不是闭包中的 min_delta
            self.is_better = lambda score, best: score > best + self.min_delta
    
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def setup_logging(config: Dict, mode: str = "train"):
    """设置日志系统（支持训练和评估模式分离）
    
    Args:
        config: 配置字典
        mode: 运行模式，"train" 或 "evaluate"
        
    Returns:
        str: 训练会话的时间戳
    """
    os.makedirs(config['paths']['logs'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 根据模式选择不同的日志文件前缀
    if mode == "train":
        log_file = os.path.join(config['paths']['logs'], f'training_{timestamp}.log')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [TRAIN] %(message)s'
    elif mode == "evaluate":
        log_file = os.path.join(config['paths']['logs'], f'evaluation_{timestamp}.log')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [EVAL] %(message)s'
    else:
        log_file = os.path.join(config['paths']['logs'], f'session_{timestamp}.log')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [{}] %(message)s'.format(mode.upper())
    
    # 清除现有的处理器，避免重复日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"{'='*60}")
    logger.info(f"日志系统已配置 - 模式: {mode.upper()}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"{'='*60}")
    
    return timestamp


def set_random_seed(seed: int):
    """设置随机种子
    
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
        torch.backends.cudnn.benchmark = False


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, 
                      topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """计算Top-K准确率
    
    Args:
        output: 模型输出 (batch_size, num_classes)
        target: 真实标签 (batch_size,)
        topk: 计算Top-K准确率的K值
        
    Returns:
        Top-K准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def save_checkpoint(state: Dict, is_best: bool, checkpoint_dir: str, 
                   filename: str = 'checkpoint.pth'):
    """保存模型检查点
    
    Args:
        state: 要保存的状态字典
        is_best: 是否为最佳模型
        checkpoint_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存检查点
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    # 记录保存信息
    if is_best:
        print(f"✓ 最佳模型已保存: {filename}")
    else:
        print(f"✓ 检查点已保存: {filename}")


def clean_old_checkpoints(checkpoint_dir: str, max_checkpoints: int = 5):
    """清理旧的检查点文件，保留最新的几个
    
    注意：此函数已废弃，不再使用检查点数量限制
    
    Args:
        checkpoint_dir: 检查点目录
        max_checkpoints: 保留的最大检查点数量
    """
    # 此函数已废弃，保留只是为了向后兼容
    pass


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: Optional[torch.device] = None) -> Dict:
    """加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 调度器（可选）
        device: 设备（可选）
        
    Returns:
        检查点字典
    """
    # 根据设备加载检查点
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 确保模型在正确的设备上
    model.to(device)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def plot_training_curves(history: Dict, save_path: str):
    """绘制训练曲线
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(epochs, history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-5准确率
    axes[1, 0].plot(epochs, history['val_top5_acc'], label='Validation Top-5 Accuracy', color='green')
    axes[1, 0].set_title('Top-5 Accuracy Curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 学习率曲线
    axes[1, 1].plot(epochs, history['lr'], label='Learning Rate', color='orange')
    axes[1, 1].set_title('Learning Rate Curve')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str):
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def denormalize_image(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """反标准化图像
    
    Args:
        tensor: 标准化后的图像张量
        mean: 均值
        std: 标准差
        
    Returns:
        反标准化后的图像张量
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def plot_sample_predictions(images: torch.Tensor, true_labels: torch.Tensor, 
                           pred_labels: torch.Tensor, class_names: List[str],
                           save_path: str, mean: List[float], std: List[float]):
    """绘制预测示例
    
    Args:
        images: 图像张量
        true_labels: 真实标签
        pred_labels: 预测标签
        class_names: 类别名称
        save_path: 保存路径
        mean: 标准化均值
        std: 标准化标准差
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    num_samples = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 反标准化
        img = denormalize_image(images[i], mean, std)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> str:
    """获取模型摘要信息
    
    Args:
        model: 模型
        input_size: 输入尺寸
        
    Returns:
        模型摘要字符串
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
模型摘要:
总参数数量: {total_params:,}
可训练参数: {trainable_params:,}
非可训练参数: {total_params - trainable_params:,}
输入尺寸: {input_size}
"""
    return summary


# 测试代码
if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i, 1)
    print(f"AverageMeter测试: {meter.avg}")
    
    # 测试早停
    early_stopping = EarlyStopping(patience=3)
    losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]
    for loss in losses:
        early_stopping(loss)
        if early_stopping.early_stop:
            print(f"早停触发于损失: {loss}")
            break
    
    # 测试准确率计算
    output = torch.randn(4, 10)
    target = torch.randint(0, 10, (4,))
    acc1, acc5 = calculate_accuracy(output, target, (1, 5))
    print(f"Top-1准确率: {acc1.item():.2f}%, Top-5准确率: {acc5.item():.2f}%")
    
    print("工具函数测试完成！")

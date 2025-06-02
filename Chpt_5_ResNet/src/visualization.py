"""
可视化模块
专门处理ResNet18项目的各种可视化需求
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from datetime import datetime
import cv2
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from utils import denormalize_image


class ResNetVisualizer:
    """ResNet可视化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config['visualization']
        self.data_config = config['data']
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # 创建可视化目录
        os.makedirs(config['paths']['visualizations'], exist_ok=True)
        
    def plot_data_augmentation_examples(self, train_loader, save_path: str = None):
        """绘制数据增强前后对比
        
        Args:
            train_loader: 训练数据加载器
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['visualizations'], 
                'data_augmentation_examples.png'
            )
        
        # 获取原始数据（无增强）
        import torchvision.transforms as transforms
        from torchvision.datasets import CIFAR10
        
        original_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.data_config['transforms']['normalize']['mean'],
                self.data_config['transforms']['normalize']['std']
            )
        ])
        
        original_dataset = CIFAR10(
            root=self.data_config['data_root'],
            train=True,
            download=False,
            transform=original_transform
        )
        
        # 获取样本
        indices = np.random.choice(len(original_dataset), 8, replace=False)
        
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i, idx in enumerate(indices):
            # 原始图像
            orig_img, label = original_dataset[idx]
            orig_img_denorm = denormalize_image(
                orig_img, 
                self.data_config['transforms']['normalize']['mean'],
                self.data_config['transforms']['normalize']['std']
            )
            orig_img_denorm = torch.clamp(orig_img_denorm, 0, 1)
            
            # 增强后图像
            aug_img, _ = train_loader.dataset[idx]
            aug_img_denorm = denormalize_image(
                aug_img,
                self.data_config['transforms']['normalize']['mean'],
                self.data_config['transforms']['normalize']['std']
            )
            aug_img_denorm = torch.clamp(aug_img_denorm, 0, 1)
            
            # 显示图像
            axes[0, i].imshow(orig_img_denorm.permute(1, 2, 0))
            axes[0, i].set_title(f'原始: {self.class_names[label]}', fontsize=10)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(aug_img_denorm.permute(1, 2, 0))
            axes[1, i].set_title(f'增强: {self.class_names[label]}', fontsize=10)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('原始图像', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('数据增强', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_feature_maps(self, model, images: torch.Tensor, 
                              layer_names: List[str], save_path: str = None):
        """可视化特征图
        
        Args:
            model: 训练好的模型
            images: 输入图像
            layer_names: 要可视化的层名称
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['visualizations'],
                'feature_maps.png'
            )
        
        model.eval()
        device = next(model.parameters()).device
        images = images.to(device)
        
        # 获取特征图
        features = model.get_feature_maps(images[:1])  # 只取第一张图像
        
        # 绘制特征图
        num_layers = len(layer_names)
        fig, axes = plt.subplots(num_layers, 8, figsize=(16, 2 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name in features:
                feature_map = features[layer_name][0]  # 取第一个样本
                
                # 选择前8个通道
                num_channels = min(8, feature_map.size(0))
                
                for ch in range(num_channels):
                    feature = feature_map[ch].cpu().numpy()
                    
                    # 标准化到0-1
                    feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
                    
                    axes[layer_idx, ch].imshow(feature, cmap='viridis')
                    axes[layer_idx, ch].set_title(f'{layer_name}\nCh {ch}', fontsize=8)
                    axes[layer_idx, ch].axis('off')
                
                # 清空多余的子图
                for ch in range(num_channels, 8):
                    axes[layer_idx, ch].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_activation_heatmaps(self, model, images: torch.Tensor, 
                                    target_layer: str = 'layer4', save_path: str = None):
        """可视化激活热力图（类似Grad-CAM）
        
        Args:
            model: 训练好的模型
            images: 输入图像
            target_layer: 目标层名称
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['visualizations'],
                'activation_heatmaps.png'
            )
        
        model.eval()
        device = next(model.parameters()).device
        images = images.to(device)
        
        # 获取特征图
        features = model.get_feature_maps(images)
        
        if target_layer not in features:
            print(f"警告: 层 {target_layer} 不存在")
            return
        
        feature_maps = features[target_layer]
        
        # 计算每个通道的平均激活
        heatmaps = torch.mean(feature_maps, dim=1)  # (batch, H, W)
        
        # 上采样到原图尺寸
        heatmaps = F.interpolate(
            heatmaps.unsqueeze(1), 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        # 绘制热力图
        num_samples = min(8, len(images))
        fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
        
        for i in range(num_samples):
            # 原图
            img = denormalize_image(
                images[i].cpu(),
                self.data_config['transforms']['normalize']['mean'],
                self.data_config['transforms']['normalize']['std']
            )
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'原图 {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # 热力图
            heatmap = heatmaps[i].cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # 叠加热力图
            overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) / 255.0
            
            combined = 0.6 * img + 0.4 * overlay
            combined = np.clip(combined, 0, 1)
            
            axes[1, i].imshow(combined)
            axes[1, i].set_title(f'激活热力图 {i+1}', fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_class_distribution(self, data_loader, save_path: str = None):
        """绘制类别分布
        
        Args:
            data_loader: 数据加载器
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['visualizations'],
                'class_distribution.png'
            )
        
        # 统计类别分布
        class_counts = [0] * 10
        for _, labels in data_loader:
            for label in labels:
                class_counts[label.item()] += 1
        
        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_counts, color='skyblue', alpha=0.7)
        
        # 添加数值标签
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.title('CIFAR-10 类别分布', fontsize=14, fontweight='bold')
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_performance_summary(self, history: Dict, best_metrics: Dict, 
                                     save_path: str = None):
        """绘制模型性能总结
        
        Args:
            history: 训练历史
            best_metrics: 最佳指标
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['visualizations'],
                'performance_summary.png'
            )
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = history['epoch']
        
        # 损失曲线
        axes[0, 0].plot(epochs, history['train_loss'], label='训练', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='验证', linewidth=2)
        axes[0, 0].set_title('损失曲线', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-1准确率
        axes[0, 1].plot(epochs, history['train_acc'], label='训练', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], label='验证', linewidth=2)
        axes[0, 1].set_title('Top-1 准确率', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5准确率
        axes[0, 2].plot(epochs, history['val_top5_acc'], label='验证Top-5', linewidth=2, color='green')
        axes[0, 2].set_title('Top-5 准确率', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Top-5 Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 学习率
        axes[1, 0].plot(epochs, history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('学习率变化', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 性能指标表格
        axes[1, 1].axis('off')
        metrics_text = f"""
        最佳性能指标
        
        验证准确率: {best_metrics.get('val_acc', 0):.2f}%
        验证Top-5: {best_metrics.get('val_top5_acc', 0):.2f}%
        最佳Epoch: {best_metrics.get('best_epoch', 0)}
        总训练时间: {best_metrics.get('total_time', 0):.1f}s
        
        性能基准达成:
        训练准确率 ≥ 90%: {'✓' if history['train_acc'][-1] >= 90 else '✗'}
        验证准确率 ≥ 85%: {'✓' if best_metrics.get('val_acc', 0) >= 85 else '✗'}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                        verticalalignment='center', fontfamily='monospace')
        
        # 训练稳定性分析
        val_acc_std = np.std(history['val_acc'][-10:])  # 最后10个epoch的标准差
        val_loss_trend = np.polyfit(range(len(history['val_loss'])), history['val_loss'], 1)[0]
        
        stability_text = f"""
        训练稳定性分析
        
        验证准确率稳定性: {val_acc_std:.2f}%
        验证损失趋势: {'下降' if val_loss_trend < 0 else '上升'}
        过拟合程度: {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%
        
        收敛情况:
        {'收敛良好' if val_acc_std < 2.0 else '存在震荡'}
        """
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.5, stability_text, fontsize=11,
                        verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# 测试代码
if __name__ == "__main__":
    import yaml
    
    # 加载配置测试
    with open('../configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    visualizer = ResNetVisualizer(config)
    print("可视化器创建成功！")
    print(f"可视化保存路径: {config['paths']['visualizations']}")
    print(f"支持的可视化功能: {list(config['visualization'].keys())}")

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
        """绘制数据增强前后对比 - 显示同一张图片的原始版本、水平翻转版本和随机裁剪版本
        
        Args:
            train_loader: 训练数据加载器
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['visualizations'], 
                'data_augmentation_examples.png'
            )
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 获取原始数据（无增强）
        import torchvision.transforms as transforms
        from torchvision.datasets import CIFAR10
        from PIL import Image
        
        # 确保数据根目录路径正确
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_root = os.path.abspath(os.path.join(project_root, self.data_config['data_root']))
        
        # 创建原始数据集（仅用于获取原始PIL图像）
        raw_dataset = CIFAR10(root=data_root, train=True, download=False, transform=None)
        
        # 定义不同的变换
        normalize = transforms.Normalize(
            self.data_config['transforms']['normalize']['mean'],
            self.data_config['transforms']['normalize']['std']
        )
        
        # 原始变换（仅标准化）
        original_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        # 水平翻转变换
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  # 确保100%翻转
            transforms.ToTensor(),
            normalize
        ])
        
        # 随机裁剪变换
        crop_transform = transforms.Compose([
            transforms.RandomCrop(
                size=self.data_config['transforms']['train']['random_crop']['size'],
                padding=self.data_config['transforms']['train']['random_crop']['padding']
            ),
            transforms.ToTensor(),
            normalize
        ])
        
        # 随机选择3张图片的索引
        num_samples = 3
        np.random.seed(42)  # 设置随机种子以确保可重现性
        indices = np.random.choice(len(raw_dataset), num_samples, replace=False)
        
        # 创建3x3的子图布局
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        for i, idx in enumerate(indices):
            try:
                # 获取原始PIL图像和标签
                pil_img, label = raw_dataset[idx]
                
                # 应用不同的变换到同一张图片
                orig_img_tensor = original_transform(pil_img)
                flip_img_tensor = flip_transform(pil_img)
                crop_img_tensor = crop_transform(pil_img)
                
                # 反标准化用于显示
                orig_img_denorm = denormalize_image(
                    orig_img_tensor, 
                    self.data_config['transforms']['normalize']['mean'],
                    self.data_config['transforms']['normalize']['std']
                )
                flip_img_denorm = denormalize_image(
                    flip_img_tensor,
                    self.data_config['transforms']['normalize']['mean'],
                    self.data_config['transforms']['normalize']['std']
                )
                crop_img_denorm = denormalize_image(
                    crop_img_tensor,
                    self.data_config['transforms']['normalize']['mean'],
                    self.data_config['transforms']['normalize']['std']
                )
                
                # 限制像素值范围到[0,1]
                orig_img_denorm = torch.clamp(orig_img_denorm, 0, 1)
                flip_img_denorm = torch.clamp(flip_img_denorm, 0, 1)
                crop_img_denorm = torch.clamp(crop_img_denorm, 0, 1)
                
                # 显示原始图像（第一行）
                axes[0, i].imshow(orig_img_denorm.permute(1, 2, 0).numpy())
                axes[0, i].set_title(f'Original\n{self.class_names[label]}', fontsize=12, fontweight='bold')
                axes[0, i].axis('off')
                
                # 显示水平翻转图像（第二行）
                axes[1, i].imshow(flip_img_denorm.permute(1, 2, 0).numpy())
                axes[1, i].set_title(f'Horizontal Flip\n{self.class_names[label]}', fontsize=12)
                axes[1, i].axis('off')
                  # 显示随机裁剪图像（第三行）
                axes[2, i].imshow(crop_img_denorm.permute(1, 2, 0).numpy())
                axes[2, i].set_title(f'Random Crop\n{self.class_names[label]}', fontsize=12)
                axes[2, i].axis('off')
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                # 清空出错的子图
                for row in range(3):
                    axes[row, i].axis('off')
                    axes[row, i].set_title(f'Error loading image {idx}', fontsize=10, color='red')
                continue
        
        # 添加行标签（在第一列的左侧）
        fig.text(0.02, 0.83, 'Original Images', fontsize=14, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.50, 'Horizontal Flip', fontsize=14, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.17, 'Random Crop', fontsize=14, fontweight='bold', rotation=90, va='center')
        
        plt.suptitle('Data Augmentation Examples: Same Images with Different Transforms', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, top=0.9)  # 为行标签和标题留出空间        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Data augmentation examples saved to: {save_path}")
        
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
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
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
                    feature = feature_map[ch].detach().cpu().numpy()
                    
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
            print(f"Warning: Layer {target_layer} does not exist")
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
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).detach().numpy()
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Original {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # 热力图
            heatmap = heatmaps[i].detach().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # 叠加热力图
            overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) / 255.0
            
            combined = 0.6 * img + 0.4 * overlay
            combined = np.clip(combined, 0, 1)
            
            axes[1, i].imshow(combined)
            axes[1, i].set_title(f'Activation Heatmap {i+1}', fontsize=10)
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
        plt.title('CIFAR-10 Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
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
        axes[0, 0].plot(epochs, history['train_loss'], label='Training', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Curve', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-1准确率
        axes[0, 1].plot(epochs, history['train_acc'], label='Training', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Top-1 Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5准确率
        axes[0, 2].plot(epochs, history['val_top5_acc'], label='Validation Top-5', linewidth=2, color='green')
        axes[0, 2].set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Top-5 Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 学习率
        axes[1, 0].plot(epochs, history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 性能指标表格
        axes[1, 1].axis('off')
        metrics_text = f"""
        Best Performance Metrics
        
        Validation Accuracy: {best_metrics.get('val_acc', 0):.2f}%
        Validation Top-5: {best_metrics.get('val_top5_acc', 0):.2f}%
        Best Epoch: {best_metrics.get('best_epoch', 0)}
        Total Training Time: {best_metrics.get('total_time', 0):.1f}s
        
        Performance Benchmarks:
        Training Acc >= 90%: {'✓' if history['train_acc'][-1] >= 90 else '✗'}
        Validation Acc >= 85%: {'✓' if best_metrics.get('val_acc', 0) >= 85 else '✗'}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                        verticalalignment='center', fontfamily='monospace')
        
        # 训练稳定性分析
        val_acc_std = np.std(history['val_acc'][-10:])  # 最后10个epoch的标准差
        val_loss_trend = np.polyfit(range(len(history['val_loss'])), history['val_loss'], 1)[0]
        
        stability_text = f"""
        Training Stability Analysis
        
        Val Accuracy Stability: {val_acc_std:.2f}%
        Val Loss Trend: {'Decreasing' if val_loss_trend < 0 else 'Increasing'}
        Overfitting Degree: {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%
        
        Convergence Status:
        {'Well Converged' if val_acc_std < 2.0 else 'Oscillating'}
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
    from pathlib import Path # 添加导入
    
    # 构建正确的配置文件路径
    current_file_dir = Path(__file__).parent
    config_path = current_file_dir.parent / 'configs' / 'config.yaml'

    # 加载配置测试
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    visualizer = ResNetVisualizer(config)
    print("可视化器创建成功！")
    print(f"可视化保存路径: {config['paths']['visualizations']}")
    print(f"支持的可视化功能: {list(config['visualization'].keys())}")

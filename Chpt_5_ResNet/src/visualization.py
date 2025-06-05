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
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not found. Some visualization features may not work properly.")
    cv2 = None
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from utils import denormalize_image


class ResNetVisualizer:
    """ResNet可视化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        # 安全地获取可视化配置，提供默认值
        self.viz_config = config.get('visualization', {
            'figure_size': (12, 8),
            'dpi': 100,
            'save_format': 'png'
        })
        self.data_config = config.get('data', {})
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        # 创建可视化目录
        viz_path = config.get('paths', {}).get('visualizations', 'outputs/visualizations')
        os.makedirs(viz_path, exist_ok=True)
        
    def plot_data_augmentation_examples(self, train_loader, save_path: str = None):
        """绘制数据增强前后对比 - 显示同一张图片的原始版本、水平翻转版本和随机裁剪版本
        
        Args:
            train_loader: 训练数据加载器
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'data_augmentation_examples.png')
        
        # 获取一批数据
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        # 选择前8张图片
        num_images = min(8, len(images))
        
        fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        # 获取归一化参数
        normalize_config = self.data_config.get('transforms', {}).get('normalize', {})
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        std = normalize_config.get('std', [0.229, 0.224, 0.225])
        
        for i in range(num_images):
            # 原图（已增强）
            img = denormalize_image(images[i], mean, std)
            img = torch.clamp(img, 0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'{self.class_names[labels[i]]}')
            axes[0, i].axis('off')
            
            # 水平翻转版本
            flipped_img = torch.flip(img, dims=[2])
            axes[1, i].imshow(flipped_img.permute(1, 2, 0).numpy())
            axes[1, i].set_title('Flipped')
            axes[1, i].axis('off')
        
        plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 数据增强示例已保存至: {save_path}")

    def plot_training_history(self, history: Dict, save_path: str = None):
        """绘制训练历史曲线
        
        Args:
            history: 训练历史记录，包含 train_loss, val_loss, train_acc, val_acc
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(viz_path, f'training_history_{timestamp}.png')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 学习率变化（如果有记录）
        if 'learning_rate' in history:
            ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Learning Rate History\nNot Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # 训练验证差距
        acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
        ax4.plot(epochs, acc_gap, 'purple', linewidth=2)
        ax4.set_title('Training-Validation Accuracy Gap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy Gap (%)')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 训练历史曲线已保存至: {save_path}")

    def plot_confusion_matrix(self, model, test_loader, device, save_path: str = None):
        """绘制混淆矩阵
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            device: 设备
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'confusion_matrix.png')
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 混淆矩阵已保存至: {save_path}")

    def plot_feature_maps(self, model, input_tensor, layer_name: str = None, save_path: str = None):
        """可视化特征图
        
        Args:
            model: 模型
            input_tensor: 输入张量
            layer_name: 层名称
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'feature_maps.png')
        
        model.eval()
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 注册钩子到第一个卷积层
        if hasattr(model, 'conv1'):
            model.conv1.register_forward_hook(get_activation('conv1'))
        
        # 前向传播
        with torch.no_grad():
            _ = model(input_tensor.unsqueeze(0))
        
        # 可视化特征图
        if 'conv1' in activation:
            features = activation['conv1'].squeeze(0)  # 移除batch维度
            num_features = min(16, features.size(0))  # 最多显示16个特征图
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i in range(num_features):
                row, col = i // 4, i % 4
                axes[row, col].imshow(features[i].cpu().numpy(), cmap='viridis')
                axes[row, col].set_title(f'Feature {i+1}')
                axes[row, col].axis('off')
            
            # 隐藏多余的子图
            for i in range(num_features, 16):
                row, col = i // 4, i % 4
                axes[row, col].axis('off')
            
            plt.suptitle('Feature Maps from First Convolutional Layer', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 特征图已保存至: {save_path}")

    def plot_model_architecture(self, model, save_path: str = None):
        """可视化模型架构
        
        Args:
            model: 模型
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'model_architecture.png')
        
        # 这里可以实现模型架构图的绘制
        # 由于复杂性，这里只是一个简单的文本表示
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, str(model), ha='center', va='center', 
                fontsize=8, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('ResNet18 Architecture', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 模型架构图已保存至: {save_path}")    
    def visualize_feature_maps(self, model, input_images, layer_names: List[str], save_path: str = None):
        """可视化指定层的特征图，包括原始图片和初始卷积层
        
        Args:
            model: 模型
            input_images: 输入图像张量 (batch_size, C, H, W)
            layer_names: 要可视化的层名称列表
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'feature_maps.png')
        
        model.eval()
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # 注册钩子到指定层和初始卷积层
        handles = []
        
        # 添加初始卷积层
        if hasattr(model, 'conv1'):
            handle = model.conv1.register_forward_hook(get_activation('conv1'))
            handles.append(handle)
        
        # 添加指定的层
        for layer_name in layer_names:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                if hasattr(layer, 'register_forward_hook'):
                    handle = layer.register_forward_hook(get_activation(layer_name))
                    handles.append(handle)
        
        # 前向传播
        device = next(model.parameters()).device
        if len(input_images) > 1:
            input_images = input_images[:1]  # 只使用第一张图片
        
        input_tensor = input_images.to(device)
        
        with torch.no_grad():
            _ = model(input_tensor)
          # 准备可视化 - 水平布局：4行 x (2 + len(layer_names))列
        # 第1列：原始图片，第2列：conv1，后续列：各层特征图
        total_cols = 2 + len(layer_names)
        fig, axes = plt.subplots(4, total_cols, figsize=(4*total_cols, 12))
        
        # 获取归一化参数用于显示原始图片
        normalize_config = self.data_config.get('transforms', {}).get('normalize', {})
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        std = normalize_config.get('std', [0.229, 0.224, 0.225])
        
        # 第一列：显示原始图片
        original_img = denormalize_image(input_tensor[0].cpu(), mean, std)
        original_img = torch.clamp(original_img, 0, 1)
        
        # 第一列第一行：显示RGB原图
        axes[0, 0].imshow(original_img.permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Original\nImage', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 第一列其余行：显示RGB三个通道
        channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
        for i in range(3):
            axes[i+1, 0].imshow(original_img[i].numpy(), cmap='gray')
            axes[i+1, 0].set_title(channel_names[i], fontsize=10)
            axes[i+1, 0].axis('off')
        
        # 第二列：显示conv1特征图（前4个通道）
        if 'conv1' in activations:
            conv1_features = activations['conv1'][0]  # 取第一个样本
            for i in range(4):
                if i < conv1_features.size(0):
                    feature_map = conv1_features[i].cpu().numpy()
                    axes[i, 1].imshow(feature_map, cmap='viridis')
                    axes[i, 1].set_title(f'Conv1\nCh{i+1}', fontsize=10)
                else:
                    axes[i, 1].axis('off')
                axes[i, 1].set_xticks([])
                axes[i, 1].set_yticks([])
        else:
            for i in range(4):
                axes[i, 1].axis('off')
                axes[i, 1].text(0.5, 0.5, 'Conv1\nNot Available', ha='center', va='center', 
                               transform=axes[i, 1].transAxes)
        
        # 后续列：显示其他层的特征图（每层显示前4个通道）
        col_idx = 2
        for layer_name in layer_names:
            if layer_name in activations:
                features = activations[layer_name][0]  # 取第一个样本的特征
                for i in range(4):
                    if i < features.size(0):
                        feature_map = features[i].cpu().numpy()
                        axes[i, col_idx].imshow(feature_map, cmap='viridis')
                        axes[i, col_idx].set_title(f'{layer_name}\nCh{i+1}', fontsize=10)
                    else:
                        axes[i, col_idx].axis('off')
                    axes[i, col_idx].set_xticks([])
                    axes[i, col_idx].set_yticks([])
            else:
                for i in range(4):
                    axes[i, col_idx].axis('off')
                    axes[i, col_idx].text(0.5, 0.5, f'{layer_name}\nNot Available', ha='center', va='center',
                                         transform=axes[i, col_idx].transAxes)
            col_idx += 1
        
        plt.suptitle('Feature Maps Visualization (Original → Conv1 → Layer1 → Layer2 → Layer3 → Layer4)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 特征图已保存至: {save_path}")
        
        # 移除钩子
        for handle in handles:
            handle.remove()

    def visualize_activation_heatmaps(self, model, input_images, save_path: str = None):
        """可视化激活热力图（使用Grad-CAM技术）
        
        Args:
            model: 模型
            input_images: 输入图像张量
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'activation_heatmaps.png')
        
        model.eval()
        device = next(model.parameters()).device
        
        # 选择前4张图片进行可视化
        num_images = min(4, len(input_images))
        images = input_images[:num_images].to(device)
        
        # 获取归一化参数用于反标准化显示
        normalize_config = self.data_config.get('transforms', {}).get('normalize', {})
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        std = normalize_config.get('std', [0.229, 0.224, 0.225])
        
        fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_images):
            # 原始图像
            img = denormalize_image(images[i].cpu(), mean, std)
            img = torch.clamp(img, 0, 1)
            axes[0, i].imshow(img.permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
            
            # 简单的激活可视化（使用最后一层特征的平均值）
            with torch.no_grad():
                # 获取模型的最后卷积层输出
                if hasattr(model, 'layer4'):
                    x = images[i:i+1]                    # 前向传播到最后一个卷积层（适配我们的ResNet18结构）
                    x = model.conv1(x)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)# 对特征图求平均得到热力图
                    heatmap = torch.mean(x[0], dim=0).cpu().numpy()
                    # 调整到原图大小
                    if cv2 is not None:
                        heatmap = cv2.resize(heatmap, (32, 32))
                    else:
                        # 如果没有cv2，使用PyTorch的插值
                        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
                        heatmap_tensor = F.interpolate(heatmap_tensor, size=(32, 32), mode='bilinear', align_corners=False)
                        heatmap = heatmap_tensor.squeeze().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                else:
                    # 如果没有layer4，创建一个随机热力图作为占位符
                    heatmap = np.random.rand(32, 32)
            
            # 显示热力图
            im = axes[1, i].imshow(heatmap, cmap='jet', alpha=0.7)
            axes[1, i].imshow(img.permute(1, 2, 0).numpy(), alpha=0.3)
            axes[1, i].set_title(f'Activation Heatmap {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle('Activation Heatmaps Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 激活热力图已保存至: {save_path}")

    def plot_class_distribution(self, data_loader, save_path: str = None):
        """绘制类别分布图
        
        Args:
            data_loader: 数据加载器
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'class_distribution.png')
        
        # 统计每个类别的样本数量
        class_counts = np.zeros(len(self.class_names))
        
        for _, labels in data_loader:
            for label in labels:
                class_counts[label.item()] += 1
        
        # 绘制柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        bars = ax1.bar(range(len(self.class_names)), class_counts, 
                      color=plt.cm.tab10(np.linspace(0, 1, len(self.class_names))))
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Class Distribution - Bar Chart')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # 在柱子上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 饼图
        ax2.pie(class_counts, labels=self.class_names, autopct='%1.1f%%', startangle=90,
               colors=plt.cm.tab10(np.linspace(0, 1, len(self.class_names))))
        ax2.set_title('Class Distribution - Pie Chart')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 类别分布图已保存至: {save_path}")

    def denormalize_image(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
        """反标准化图像张量用于可视化
        
        Args:
            tensor: 标准化后的图像张量 (C, H, W)
            mean: 标准化均值
            std: 标准化标准差
            
        Returns:
            反标准化后的图像张量
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def visualize_conv_kernels(self, model, save_path: str = None):
        """可视化第一层卷积核权重
        
        Args:
            model: 模型
            save_path: 保存路径
        """
        if save_path is None:
            viz_path = self.config.get('paths', {}).get('visualizations', 'outputs/visualizations')
            save_path = os.path.join(viz_path, 'conv_kernels.png')
          # 获取第一层卷积层的权重
        if hasattr(model, 'conv1'):
            conv1_weights = model.conv1.weight.data.cpu()  # shape: (out_channels, in_channels, h, w)
            num_kernels = min(4, conv1_weights.size(0))  # 只显示前4个卷积核
            in_channels = conv1_weights.size(1)  # 输入通道数 (通常是3，对应RGB)
            
            # 创建布局：每个输入通道一行，4个卷积核一行排列
            fig, axes = plt.subplots(in_channels, 4, figsize=(10, 2.5 * in_channels))
            if in_channels == 1:
                axes = [axes]
            
            channel_names = ['Red', 'Green', 'Blue'] if in_channels == 3 else [f'Channel {i}' for i in range(in_channels)]
            
            for ch in range(in_channels):
                for i in range(4):
                    if i < num_kernels:
                        # 获取当前卷积核的权重
                        kernel = conv1_weights[i, ch, :, :].numpy()
                        
                        # 标准化到 [0, 1] 范围便于显示
                        kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
                        
                        # 显示卷积核
                        axes[ch][i].imshow(kernel_norm, cmap='viridis')
                        axes[ch][i].set_title(f'Kernel {i+1}', fontsize=10)
                        axes[ch][i].axis('off')
                    else:
                        # 隐藏多余的子图
                        axes[ch][i].axis('off')
                
                # 为每行添加通道标签
                axes[ch][0].text(-0.15, 0.5, f'{channel_names[ch]}\nChannel', 
                               rotation=90, ha='center', va='center', 
                               transform=axes[ch][0].transAxes, fontsize=11, fontweight='bold')
            
            plt.suptitle(f'First 4 Convolutional Kernels Visualization\n(Conv1 Layer: {num_kernels} kernels, {in_channels} input channels)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 卷积核可视化已保存至: {save_path}")
        else:
            print("❌ 模型中未找到conv1层，无法进行卷积核可视化")

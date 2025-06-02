"""
ResNet18 CIFAR-10 主程序
实现完整的训练、验证和可视化流程
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
import os
import time
from datetime import datetime
import logging
from typing import Dict

from model import create_resnet18
from data import create_data_loaders, CIFAR10DataLoader
from trainer import ResNetTrainer
from utils import (
    setup_logging, set_random_seed, plot_training_curves, 
    plot_confusion_matrix, plot_sample_predictions, get_model_summary,
    calculate_accuracy, load_checkpoint
)
from visualization import ResNetVisualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ResNet18 CIFAR-10 训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅进行评估，不训练')
    parser.add_argument('--model-path', type=str, default=None,
                       help='用于评估的模型路径')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model: nn.Module, test_loader, device: torch.device, 
                  class_names: list, config: Dict) -> Dict:
    """评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        class_names: 类别名称
        config: 配置字典
        
    Returns:
        评估结果字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_images = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    logger = logging.getLogger(__name__)
    logger.info("开始模型评估...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())
            total_loss += loss.item()
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 准确率
    accuracy = (all_preds == all_labels).mean() * 100
    
    # 各类别准确率
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean() * 100
            class_accuracies[class_name] = class_acc
    
    avg_loss = total_loss / len(test_loader)
    
    # 记录结果
    logger.info(f"测试损失: {avg_loss:.4f}")
    logger.info(f"测试准确率: {accuracy:.2f}%")
    
    logger.info("各类别准确率:")
    for class_name, acc in class_accuracies.items():
        logger.info(f"  {class_name}: {acc:.2f}%")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'class_accuracies': class_accuracies,
        'predictions': all_preds,
        'labels': all_labels,
        'sample_images': all_images[:16]  # 保存前16张图片用于可视化
    }


def create_visualizations(model: nn.Module, data_loader_obj: CIFAR10DataLoader,
                         train_loader, test_loader, history: Dict, 
                         eval_results: Dict, config: Dict):
    """创建所有可视化图表
    
    Args:
        model: 训练好的模型
        data_loader_obj: 数据加载器对象
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        history: 训练历史
        eval_results: 评估结果
        config: 配置字典
    """
    logger = logging.getLogger(__name__)
    logger.info("开始生成可视化图表...")
    
    visualizer = ResNetVisualizer(config)
    viz_dir = config['paths']['visualizations']
    
    # 1. 训练曲线
    if history:
        plot_training_curves(
            history, 
            os.path.join(viz_dir, 'training_curves.png')
        )
        logger.info("✓ 训练曲线已保存")
    
    # 2. 混淆矩阵
    if eval_results:
        plot_confusion_matrix(
            eval_results['labels'],
            eval_results['predictions'],
            data_loader_obj.class_names,
            os.path.join(viz_dir, 'confusion_matrix.png')
        )
        logger.info("✓ 混淆矩阵已保存")
    
    # 3. 预测示例
    if eval_results and len(eval_results['sample_images']) > 0:
        plot_sample_predictions(
            torch.stack(eval_results['sample_images']),
            torch.tensor(eval_results['labels'][:16]),
            torch.tensor(eval_results['predictions'][:16]),
            data_loader_obj.class_names,
            os.path.join(viz_dir, 'sample_predictions.png'),
            config['data']['transforms']['normalize']['mean'],
            config['data']['transforms']['normalize']['std']
        )
        logger.info("✓ 预测示例已保存")
    
    # 4. 数据增强示例
    try:
        visualizer.plot_data_augmentation_examples(train_loader)
        logger.info("✓ 数据增强示例已保存")
    except Exception as e:
        logger.warning(f"数据增强示例生成失败: {e}")
    
    # 5. 特征图可视化
    try:
        sample_images, _ = data_loader_obj.get_sample_data(4)
        visualizer.visualize_feature_maps(
            model, sample_images, 
            ['layer1', 'layer2', 'layer3', 'layer4']
        )
        logger.info("✓ 特征图已保存")
    except Exception as e:
        logger.warning(f"特征图生成失败: {e}")
    
    # 6. 激活热力图
    try:
        sample_images, _ = data_loader_obj.get_sample_data(8)
        visualizer.visualize_activation_heatmaps(model, sample_images)
        logger.info("✓ 激活热力图已保存")
    except Exception as e:
        logger.warning(f"激活热力图生成失败: {e}")
    
    # 7. 类别分布
    try:
        visualizer.plot_class_distribution(test_loader)
        logger.info("✓ 类别分布已保存")
    except Exception as e:
        logger.warning(f"类别分布生成失败: {e}")
    
    # 8. 性能总结
    if history and eval_results:
        try:
            best_metrics = {
                'val_acc': max(history['val_acc']),
                'val_top5_acc': max(history['val_top5_acc']),
                'best_epoch': history['val_acc'].index(max(history['val_acc'])) + 1,
                'total_time': 0  # 这里可以添加总训练时间
            }
            visualizer.plot_model_performance_summary(history, best_metrics)
            logger.info("✓ 性能总结已保存")
        except Exception as e:
            logger.warning(f"性能总结生成失败: {e}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志和随机种子
    setup_logging(config)
    set_random_seed(config['random_seed'])
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("ResNet18 CIFAR-10 项目启动")
    logger.info("=" * 50)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logger.info("加载数据...")
    data_loader_obj = CIFAR10DataLoader(config)
    train_loader, val_loader, test_loader = data_loader_obj.load_data()
    
    # 创建模型
    logger.info("创建模型...")
    model = create_resnet18(config['model']['num_classes'])
    logger.info(get_model_summary(model, (3, 32, 32)))
    
    history = None
    eval_results = None
    
    if not args.eval_only:
        # 训练模式
        logger.info("开始训练...")
        
        # 创建训练器
        trainer = ResNetTrainer(model, config)
        
        # 恢复训练（如果指定）
        if args.resume:
            logger.info(f"从检查点恢复训练: {args.resume}")
            checkpoint = load_checkpoint(args.resume, model, trainer.optimizer, trainer.scheduler)
            start_epoch = checkpoint['epoch']
            trainer.best_val_acc = checkpoint['best_val_acc']
            trainer.train_history = checkpoint['train_history']
            logger.info(f"从epoch {start_epoch}恢复训练")
        
        # 执行训练
        start_time = time.time()
        history = trainer.train(train_loader, val_loader)
        total_time = time.time() - start_time
        
        logger.info(f"训练完成！总耗时: {total_time:.2f}秒")
        
        # 加载最佳模型进行评估
        best_model_path = os.path.join(config['paths']['models'], 'best.pth')
        if os.path.exists(best_model_path):
            checkpoint = load_checkpoint(best_model_path, model)
            logger.info("已加载最佳模型进行评估")
    
    else:
        # 评估模式
        if args.model_path:
            logger.info(f"加载模型: {args.model_path}")
            checkpoint = load_checkpoint(args.model_path, model)
        else:
            logger.error("评估模式需要指定模型路径 --model-path")
            return
    
    # 模型评估
    logger.info("开始评估...")
    eval_results = evaluate_model(model, test_loader, device, 
                                 data_loader_obj.class_names, config)
    
    # 生成可视化
    create_visualizations(
        model, data_loader_obj, train_loader, test_loader,
        history, eval_results, config
    )
    
    # 保存最终结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(config['paths']['logs'], f'final_results_{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("ResNet18 CIFAR-10 最终结果\n")
        f.write("=" * 40 + "\n\n")
        
        if eval_results:
            f.write(f"测试准确率: {eval_results['accuracy']:.2f}%\n")
            f.write(f"测试损失: {eval_results['loss']:.4f}\n\n")
            
            f.write("各类别准确率:\n")
            for class_name, acc in eval_results['class_accuracies'].items():
                f.write(f"  {class_name}: {acc:.2f}%\n")
        
        if history:
            f.write(f"\n最佳验证准确率: {max(history['val_acc']):.2f}%\n")
            f.write(f"最佳验证Top-5准确率: {max(history['val_top5_acc']):.2f}%\n")
            f.write(f"最终训练准确率: {history['train_acc'][-1]:.2f}%\n")
    
    logger.info(f"最终结果已保存至: {result_file}")
    logger.info("项目执行完成！")


if __name__ == "__main__":
    main()

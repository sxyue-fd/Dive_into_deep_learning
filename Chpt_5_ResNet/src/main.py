"""
ResNet18 CIFAR-10 主程序
实现完整的训练、验证和可视化流程
使用新的ResNetLogManager
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import time
from datetime import datetime
from typing import Dict
from pathlib import Path

from model import create_resnet18
from data import create_data_loaders, CIFAR10DataLoader
from trainer import ResNetTrainer
from utils import (
    setup_logging, set_random_seed, plot_training_curves, 
    plot_confusion_matrix, plot_sample_predictions, get_model_summary,
    calculate_accuracy, load_checkpoint
)
from visualization import ResNetVisualizer
from config_parser import get_early_stopping_config, get_random_seed as get_config_random_seed
from resnet_log_manager import ResNetLogManager


def get_config_path():
    """获取配置文件路径"""
    # 优先使用性能调优配置文件，如果不存在则使用原配置文件
    performance_config = 'configs/config_performance.yaml'
    default_config = 'configs/config.yaml'
    
    # 检查性能配置文件是否存在
    project_root = Path(__file__).resolve().parent.parent
    performance_path = project_root / performance_config
    
    if performance_path.exists():
        return performance_config
    else:
        return default_config


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    # 检查是否为性能配置文件
    if 'performance' in config_path:
        # 使用性能配置解析器
        try:
            from config_parser import load_performance_config
            return load_performance_config(config_path)
        except ImportError:
            print("⚠️ 性能配置解析器未找到，使用标准配置加载")
    
    # 标准配置加载
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model: nn.Module, test_loader, device: torch.device, 
                  class_names: list, config: Dict, log_manager: ResNetLogManager) -> Dict:
    """评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        class_names: 类别名称
        config: 配置字典
        log_manager: 日志文件管理器
        
    Returns:
        评估结果字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_images = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    log_manager.log_info("开始模型评估...")
    
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
    log_manager.log_info(f"测试损失: {avg_loss:.4f}")
    log_manager.log_info(f"测试准确率: {accuracy:.2f}%")
    log_manager.log_info("各类别准确率:")
    for class_name, acc in class_accuracies.items():
        log_manager.log_info(f"  {class_name}: {acc:.2f}%")
    
    # 评估模式不生成结果文件，只记录到日志
    log_manager.log_info(f"总样本数: {len(all_labels)}")
    log_manager.log_info(f"平均每类准确率: {np.mean(list(class_accuracies.values())):.2f}%")
    
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
                         eval_results: Dict, config: Dict, log_manager: ResNetLogManager):
    """创建所有可视化图表
    
    Args:
        model: 训练好的模型
        data_loader_obj: 数据加载器对象
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        history: 训练历史
        eval_results: 评估结果
        config: 配置字典
        log_manager: 日志管理器
    """
    log_manager.log_info("开始生成可视化图表...")
    
    visualizer = ResNetVisualizer(config)
    viz_dir = config['paths']['visualizations']
    
    # 1. 训练曲线
    if history:
        plot_training_curves(
            history, 
            os.path.join(viz_dir, 'training_curves.png')
        )
        log_manager.log_info("✓ 训练曲线已保存")
    
    # 2. 混淆矩阵
    if eval_results:
        plot_confusion_matrix(
            eval_results['labels'],
            eval_results['predictions'],
            data_loader_obj.class_names,
            os.path.join(viz_dir, 'confusion_matrix.png')
        )
        log_manager.log_info("✓ 混淆矩阵已保存")
    
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
        log_manager.log_info("✓ 预测示例已保存")
    
    # 4. 其他可视化（错误处理）
    try:
        visualizer.plot_data_augmentation_examples(train_loader)
        log_manager.log_info("✓ 数据增强示例已保存")
    except Exception as e:
        log_manager.log_warning(f"数据增强示例生成失败: {e}")
    
    try:
        sample_images, _ = data_loader_obj.get_sample_data(4)
        visualizer.visualize_feature_maps(
            model, sample_images, 
            ['layer1', 'layer2', 'layer3', 'layer4']
        )
        log_manager.log_info("✓ 特征图已保存")
    except Exception as e:
        log_manager.log_warning(f"特征图生成失败: {e}")


def main():
    """主函数"""
    # 设置项目根目录
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    
    # 获取配置文件路径
    config_file_path = get_config_path()
    config_file_path_obj = Path(config_file_path)
    
    # 如果配置文件路径不是绝对路径，则相对于项目根目录
    if not config_file_path_obj.is_absolute():
        config_file_path_obj = project_root / config_file_path
    
    # 加载配置
    print(f"使用配置文件: {config_file_path_obj}")
    config = load_config(str(config_file_path_obj))
    
    # 将配置中的相对路径转换为绝对路径
    for path_key in ['logs', 'models', 'visualizations']:
        relative_path = config['paths'][path_key]
        config['paths'][path_key] = str(project_root / relative_path)
    
    # 同样处理数据路径
    data_root = config['data']['data_root']
    if not Path(data_root).is_absolute():
        config['data']['data_root'] = str(project_root / data_root)
    
    # 使用上下文管理器创建日志管理器
    with ResNetLogManager(config['paths']['logs']) as log_manager:
        # 根据运行模式开始会话
        run_mode = config['run_mode']['mode']
        if run_mode == "train":
            log_manager.start_session('train')
        elif run_mode == "evaluate": 
            log_manager.start_session('eval')
        else:
            raise ValueError(f"未知的运行模式: {run_mode}，支持的模式: 'train', 'evaluate'")
        
        # 处理随机种子
        random_seed = get_config_random_seed(config)
        set_random_seed(random_seed)
        
        log_manager.log_info("=" * 50)
        log_manager.log_info("ResNet18 CIFAR-10 项目启动")
        log_manager.log_info("=" * 50)
        
        # 设备配置
        device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
        log_manager.log_info(f"使用设备: {device}")
        
        # 创建数据加载器
        log_manager.log_info("加载数据...")
        data_loader_obj = CIFAR10DataLoader(config)
        train_loader, val_loader, test_loader = data_loader_obj.load_data()        # 创建模型
        log_manager.log_info("创建模型...")
        dropout_rate = config['model'].get('dropout_rate', 0.0)
        model = create_resnet18(config['model']['num_classes'], dropout_rate=dropout_rate)
        model_summary = get_model_summary(model, (3, 32, 32))
        log_manager.log_info(model_summary)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_manager.log_info(f"总参数量: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        # 如果是训练模式，记录模型和配置信息到结果文件
        if run_mode == "train" and hasattr(log_manager, 'write_model_info'):
            log_manager.write_model_info(model_summary, total_params)
            log_manager.write_config_info(config)
        
        # 记录关键配置信息
        log_manager.log_info(f"模型配置 - Dropout率: {dropout_rate}")
        log_manager.log_info(f"数据配置 - 批次大小: {config['data']['batch_size']}")
        log_manager.log_info(f"数据配置 - 工作线程数: {config['data']['num_workers']}")
        log_manager.log_info(f"数据配置 - 预取因子: {config['data'].get('prefetch_factor', 2)}")
        log_manager.log_info(f"数据配置 - 持久化工作线程: {config['data'].get('persistent_workers', True)}")
        
        history = None
        eval_results = None
        
        # 根据配置文件决定运行模式
        if run_mode == "train":
            # 训练模式
            log_manager.log_info("开始训练...")
              # 创建训练器
            trainer = ResNetTrainer(model, config, log_manager)
            
            # 恢复训练（如果指定）
            resume_path = config['run_mode'].get('resume_path')
            if resume_path and os.path.exists(resume_path):
                log_manager.log_info(f"从检查点恢复训练: {resume_path}")
                checkpoint = load_checkpoint(resume_path, model, trainer.optimizer, trainer.scheduler, device)
                start_epoch = checkpoint['epoch']
                trainer.best_val_acc = checkpoint['best_val_acc']
                trainer.train_history = checkpoint['train_history']
                log_manager.log_info(f"从epoch {start_epoch}恢复训练")
            
            # 执行训练
            start_time = time.time()
            history = trainer.train(train_loader, val_loader)
            total_time = time.time() - start_time
            log_manager.log_info(f"训练完成！总耗时: {total_time:.2f}秒")
            
            # 写入详细的训练结果到txt文件
            if hasattr(trainer, 'best_epoch') and hasattr(trainer, 'best_val_acc'):
                # 计算训练统计信息
                max_train_acc = max(history['train_acc']) if history and 'train_acc' in history else 0.0
                min_train_loss = min(history['train_loss']) if history and 'train_loss' in history else 0.0
                min_val_loss = min(history['val_loss']) if history and 'val_loss' in history else 0.0
                
                # 写入最终结果
                log_manager.write_final_results(
                    best_epoch=trainer.best_epoch,
                    best_train_acc=max_train_acc,
                    best_val_acc=trainer.best_val_acc,
                    total_time=total_time
                )
                  # 写入额外的训练统计信息（纯文本格式）
                if hasattr(log_manager, 'write_result'):
                    log_manager.write_result(f"\n训练统计信息:")
                    log_manager.write_result(f"最低训练损失: {min_train_loss:.4f}")
                    log_manager.write_result(f"最低验证损失: {min_val_loss:.4f}")
                    log_manager.write_result(f"最高训练准确率: {max_train_acc:.4f}")
                    log_manager.write_result(f"总训练轮次: {len(history['train_acc']) if history else 0}")
                    
                    # 记录停止原因
                    stop_reason = getattr(trainer, 'stop_reason', 'Normal completion')
                    log_manager.write_result(f"训练停止原因: {stop_reason}")
                    
                    # 记录性能指标
                    avg_epoch_time = total_time / len(history['train_acc']) if history and history['train_acc'] else 0
                    log_manager.write_result(f"平均每轮训练时间: {avg_epoch_time:.2f}秒")
                    
                    # 记录学习率变化
                    if history and 'lr' in history:
                        initial_lr = history['lr'][0] if history['lr'] else 0
                        final_lr = history['lr'][-1] if history['lr'] else 0
                        log_manager.write_result(f"初始学习率: {initial_lr:.6f}")
                        log_manager.write_result(f"最终学习率: {final_lr:.6f}")
                
                log_manager.log_info(f"最佳模型信息 - Epoch: {trainer.best_epoch}, 验证准确率: {trainer.best_val_acc:.4f}")
            else:
                log_manager.write_final_results(
                    best_epoch=1,
                    best_train_acc=0.0,
                    best_val_acc=0.0,
                    total_time=total_time
                )
            
            # 加载最佳模型进行评估
            best_model_path = os.path.join(config['paths']['models'], 'best.pth')
            if os.path.exists(best_model_path):
                checkpoint = load_checkpoint(best_model_path, model, device=device)
                log_manager.log_info("已加载最佳模型进行评估")
        
        elif run_mode == "evaluate":
            # 评估模式
            model_path = config['run_mode']['model_path']
            if model_path:
                # 如果模型路径是相对路径，则相对于项目根目录
                if not Path(model_path).is_absolute():
                    model_path = str(project_root / model_path)
                
                if os.path.exists(model_path):
                    log_manager.log_info(f"加载模型: {model_path}")
                    checkpoint = load_checkpoint(model_path, model, device=device)
                else:
                    log_manager.log_error(f"模型文件不存在: {model_path}")
                    return
            else:
                log_manager.log_error("评估模式需要在配置文件中指定模型路径")
                return
        
        # 模型评估
        log_manager.log_info("开始评估...")
        eval_results = evaluate_model(model, test_loader, device, 
                                     data_loader_obj.class_names, config, log_manager)
        
        # 生成可视化
        create_visualizations(
            model, data_loader_obj, train_loader, test_loader,
            history, eval_results, config, log_manager
        )
        
        log_manager.log_info("项目执行完成！")


if __name__ == "__main__":
    main()

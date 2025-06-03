"""
ResNet18 训练器模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import logging
from typing import Dict, Tuple, List, Optional
import numpy as np
from datetime import datetime
import os
from pathlib import Path  # 添加导入

from model import ResNet18
from utils import AverageMeter, EarlyStopping, save_checkpoint, calculate_accuracy
from performance_config import get_early_stopping_config


class ResNetTrainer:
    """ResNet18训练器"""
    
    def __init__(self, model: ResNet18, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 设置混合精度训练
        self.use_amp = config['device']['mixed_precision'] and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # 初始化优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()        # 早停机制 - 使用兼容性函数
        early_stopping_config = get_early_stopping_config(config)
        
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config['patience'],
            min_delta=early_stopping_config['min_delta'],
            mode=early_stopping_config['mode']
        )
        
        # 清理旧的模型文件
        self._clean_previous_models()
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5_acc': [],
            'lr': []
        }
          # 最佳指标
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # 训练统计信息
        self.training_duration = 0.0
        self.stop_reason = "Training not started"
        
    def _setup_optimizer(self):
        """设置优化器"""
        train_config = self.config['training']
        # 获取优化器特定参数，如果不存在则为空字典，并复制以进行本地修改
        optimizer_params = train_config.get('optimizer_params', {}).copy() 
        
        # 确保通用参数是浮点数
        learning_rate = float(train_config['learning_rate'])
        weight_decay = float(train_config['weight_decay'])

        if train_config['optimizer'].lower() == 'adam':
            # 确保Adam特定的参数类型正确
            if 'eps' in optimizer_params:
                optimizer_params['eps'] = float(optimizer_params['eps'])
            if 'betas' in optimizer_params and isinstance(optimizer_params['betas'], list):
                optimizer_params['betas'] = [float(b) for b in optimizer_params['betas']]
            
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_params # 传递处理过的参数
            )
        elif train_config['optimizer'].lower() == 'sgd':
            # 确保SGD的momentum是浮点数，并从train_config获取
            momentum = float(train_config.get('momentum', 0.9)) # 如果配置中没有，提供一个默认值
            
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=optimizer_params.get('nesterov', False) # nesterov通常在optimizer_params中
            )
        else:            raise ValueError(f"不支持的优化器: {train_config['optimizer']}")
            
    def _setup_scheduler(self):
        """设置学习率调度器"""
        train_config = self.config['training']
        scheduler_name = train_config.get('scheduler', '').lower()
        
        if not scheduler_name or scheduler_name == 'none':
            self.scheduler = None
            return
            
        scheduler_params = train_config.get('scheduler_params', {}).copy()

        if scheduler_name == 'cosine':
            # 过滤出CosineAnnealingLR的有效参数
            cosine_params = {}
            if 'T_max' in scheduler_params:
                cosine_params['T_max'] = int(scheduler_params['T_max'])
            if 'eta_min' in scheduler_params:
                cosine_params['eta_min'] = float(scheduler_params['eta_min'])
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **cosine_params
            )
        elif scheduler_name == 'step':
            # 过滤出StepLR的有效参数
            step_params = {}
            if 'step_size' in scheduler_params:
                step_params['step_size'] = int(scheduler_params['step_size'])
            if 'gamma' in scheduler_params:
                step_params['gamma'] = float(scheduler_params['gamma'])

            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                **step_params
            )        
        else:
            self.logger.warning(f"未知的调度器类型: {scheduler_name}，将不使用调度器")
            self.scheduler = None
    
    def _clean_previous_models(self):
        """清理所有之前的模型文件，为新训练做准备"""
        models_dir = Path(self.config['paths']['models'])
        
        if models_dir.exists():
            # 获取所有模型文件
            model_files = list(models_dir.glob('*.pth')) + list(models_dir.glob('*.pt'))
            
            if model_files:
                self.logger.info(f"清理 {len(model_files)} 个旧模型文件...")
                for model_file in model_files:
                    try:
                        model_file.unlink()
                        self.logger.debug(f"已删除: {model_file.name}")
                    except Exception as e:
                        self.logger.warning(f"删除 {model_file.name} 失败: {e}")
                
                self.logger.info("模型文件清理完成")
            else:
                self.logger.info("没有发现需要清理的旧模型文件")
        else:
            # 创建模型目录
            models_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"创建模型目录: {models_dir}")
            
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            平均损失和准确率
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            # 计算准确率
            acc = calculate_accuracy(output, target, topk=(1,))[0]
            
            # 更新指标
            loss_meter.update(loss.item(), data.size(0))
            acc_meter.update(acc.item(), data.size(0))
            
            # 记录日志
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                self.logger.info(
                    f'训练 Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}\tAcc: {acc.item():.2f}%'
                )
        
        epoch_time = time.time() - start_time
        self.logger.info(f'Epoch {epoch} 训练完成，耗时: {epoch_time:.2f}s')
        
        return loss_meter.avg, acc_meter.avg
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均损失、Top-1准确率和Top-5准确率
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # 计算Top-1和Top-5准确率
                acc1, acc5 = calculate_accuracy(output, target, topk=(1, 5))
                
                # 更新指标
                loss_meter.update(loss.item(), data.size(0))
                acc1_meter.update(acc1.item(), data.size(0))
                acc5_meter.update(acc5.item(), data.size(0))
        
        return loss_meter.avg, acc1_meter.avg, acc5_meter.avg
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练过程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        self.logger.info("开始训练...")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"混合精度: {self.use_amp}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc, val_top5_acc = self.validate_epoch(val_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['val_top5_acc'].append(val_top5_acc)
            self.train_history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # 记录epoch结果
            self.logger.info(
                f'Epoch {epoch}/{self.config["training"]["epochs"]} - '
                f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% - '
                f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%, '
                f'Top-5准确率: {val_top5_acc:.2f}% - '
                f'学习率: {current_lr:.6f} - 耗时: {epoch_time:.2f}s'
            )            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                # 保存最佳模型
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_acc': self.best_val_acc,
                        'train_history': self.train_history,
                        'config': self.config
                    },
                    is_best=True,
                    checkpoint_dir=self.config['paths']['models'],
                    filename='best.pth'
                )
            
            # 周期性保存检查点
            save_frequency = self.config.get('checkpoint', {}).get('save_frequency', 5)
            if epoch % save_frequency == 0:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_acc': self.best_val_acc,
                        'train_history': self.train_history,
                        'config': self.config
                    },
                    is_best=False,
                    checkpoint_dir=self.config['paths']['models'],
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )
            
            # 早停检查
            self.early_stopping(val_loss)
            
            # 综合停止判断
            should_stop, stop_reason = self.should_stop_training(epoch)
            if should_stop:
                self.logger.info(f"训练停止 - {stop_reason}")
                self.stop_reason = stop_reason  # 记录停止原因
                
                # 如果是因为达到目标准确率而停止，记录特殊信息
                if "Target accuracy" in stop_reason:
                    self.logger.info("🎉 恭喜！模型已达到目标准确率！")
                
                break
        
        total_time = time.time() - start_time
        self.training_duration = total_time  # 记录训练时长
        
        # 如果正常完成训练（没有提前停止）
        if not hasattr(self, 'stop_reason') or self.stop_reason == "Training not started":
            self.stop_reason = "Normal completion - all epochs finished"
        
        self.logger.info(f"训练完成！总耗时: {total_time:.2f}s")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.train_history

    def should_stop_training(self, epoch: int) -> Tuple[bool, str]:
        """综合判断是否应该停止训练
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            (should_stop, reason): 是否停止和停止原因
        """
        # 1. 达到最大epoch数
        if epoch >= self.config['training']['epochs']:
            return True, "Maximum epochs reached"
          # 2. 验证准确率达到目标（主要停止条件）
        target_acc = float(self.config['training'].get('target_accuracy', 0.95))
        if self.train_history['val_acc'] and len(self.train_history['val_acc']) > 0:
            current_val_acc = self.train_history['val_acc'][-1] / 100.0  # 转换为小数
            if current_val_acc >= target_acc:
                return True, f"🎯 Target accuracy {target_acc:.1%} achieved (current: {current_val_acc:.1%})"
        
        # 3. 早停检查（基于验证损失，防止过拟合的主要机制）
        if self.early_stopping.early_stop:
            return True, "⏹️ Early stopping triggered - validation loss not improving (preventing overfitting)"
          # 4. 学习率过小检查
        current_lr = self.optimizer.param_groups[0]['lr']
        min_lr = float(self.config['training'].get('min_learning_rate', 1e-8))
        if current_lr < min_lr:
            return True, f"📉 Learning rate too small: {current_lr:.2e} < {min_lr:.2e}"
          # 辅助检测（用于提供额外信息，但不直接停止训练）
        warnings = []
        
        # 损失收敛检测
        convergence_config = self.config['training'].get('convergence', {})
        convergence_patience = int(convergence_config.get('patience', 15))  # 增加容忍度
        convergence_threshold = float(convergence_config.get('threshold', 1e-5))  # 更严格的阈值
        
        if len(self.train_history['val_loss']) >= convergence_patience:
            recent_losses = self.train_history['val_loss'][-convergence_patience:]
            loss_std = np.std(recent_losses)
            if loss_std < convergence_threshold:
                warnings.append(f"Validation loss converged (std: {loss_std:.2e})")
        
        # 过拟合程度检测（仅警告，不停止）
        overfitting_config = self.config['training'].get('overfitting', {})
        overfitting_patience = int(overfitting_config.get('patience', 5))
        gap_threshold = float(overfitting_config.get('train_val_gap_threshold', 0.15))  # 更宽松的阈值
        
        if (len(self.train_history['train_acc']) >= overfitting_patience and 
            len(self.train_history['val_acc']) >= overfitting_patience):
            
            recent_train_acc = self.train_history['train_acc'][-overfitting_patience:]
            recent_val_acc = self.train_history['val_acc'][-overfitting_patience:]
            
            # 计算最近几个epoch的平均准确率差距
            avg_gap = np.mean([t - v for t, v in zip(recent_train_acc, recent_val_acc)]) / 100.0
            
            if avg_gap > gap_threshold:
                warnings.append(f"High train-val gap detected: {avg_gap:.1%}")
          # 记录警告信息（但不停止训练）
        if warnings:
            self.logger.warning(f"Training monitoring alerts: {'; '.join(warnings)}")
        
        return False, ""
    
    def check_target_accuracy_reached(self) -> bool:
        """检查是否达到目标准确率"""
        target_acc = float(self.config['training'].get('target_accuracy', 0.95))
        if self.train_history['val_acc'] and len(self.train_history['val_acc']) > 0:
            current_val_acc = self.train_history['val_acc'][-1] / 100.0  # 转换为小数
            return current_val_acc >= target_acc
        return False


# 测试代码
if __name__ == "__main__":
    import yaml
    from pathlib import Path  # 添加导入
    from data import create_data_loaders
    from model import create_resnet18
    from utils import setup_logging, set_random_seed
    
    # 构建正确的配置文件路径
    current_file_dir = Path(__file__).parent
    config_path = current_file_dir.parent / 'configs' / 'config.yaml'

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置日志和随机种子
    # setup_logging(config) # 日志设置可能也需要调整路径，暂时注释
    set_random_seed(config['data']['random_seed']) # 从data下获取random_seed
    
    # 创建模型和数据
    model = create_resnet18(num_classes=config['data'].get('num_classes', 10)) # 确保num_classes存在
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # 创建训练器
    trainer = ResNetTrainer(model, config)
    
    print("训练器创建成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

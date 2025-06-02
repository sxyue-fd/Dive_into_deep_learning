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
        self.criterion = nn.CrossEntropyLoss()
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode=config['training']['early_stopping']['mode']
        )
        
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
        else:
            raise ValueError(f"不支持的优化器: {train_config['optimizer']}")
            
    def _setup_scheduler(self):
        """设置学习率调度器"""
        train_config = self.config['training']
        
        if train_config['scheduler'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **train_config['scheduler_params']
            )
        elif train_config['scheduler'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                **train_config['scheduler_params']
            )
        else:
            self.scheduler = None
            
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
            )
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
            # 保存检查点
            if epoch % self.config['checkpoint']['save_frequency'] == 0 or is_best:
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
                    is_best=is_best,
                    checkpoint_dir=self.config['paths']['models'],
                    filename=f'checkpoint_{epoch}.pth'
                )
            
            # 早停检查
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                self.logger.info(f"早停触发，在epoch {epoch}停止训练")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成！总耗时: {total_time:.2f}s")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.train_history


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

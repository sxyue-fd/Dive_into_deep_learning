import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from pathlib import Path
import time

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, criterion, optimizer, device, config):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备
            config: 配置信息
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # 创建保存路径
        for path in ['model_save_dir', 'log_dir', 'visualization_dir']:
            Path(config['paths'][path]).mkdir(parents=True, exist_ok=True)
            
        # 配置日志记录
        self.setup_logging()
    
    def setup_logging(self):
        """配置日志记录器"""
        log_file = Path(self.config['paths']['log_dir']) / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def train_fold(self, train_loader, val_loader, fold):
        """
        训练单个fold
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            fold: 当前fold编号
        
        Returns:
            tuple: 训练损失历史和验证损失历史
        """
        # 初始化可视化
        plt.ion()
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # 初始化训练记录
        train_losses = []
        train_epochs = []
        val_losses = []
        best_val_loss = float('inf')
        step_count = 0
        
        # 开始训练
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            train_loss = 0
            
            # 训练阶段
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                
                # 前向传播和损失计算
                output = self.model(X)
                loss = self.criterion(output, y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X.size(0)
                
                # 更新训练曲线
                if i % 10 == 0:
                    step_count += 1
                    current_train_loss = train_loss / ((i + 1) * X.size(0))
                    train_losses.append(current_train_loss)
                    train_epochs.append(epoch + i/len(train_loader))
                    
                    self._update_plot(ax, train_epochs, train_losses, 
                                    val_losses, current_train_loss, fold)
            
            # 计算平均损失
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)
            
            # 记录训练信息
            logging.info(
                f'Fold {fold+1}, Epoch [{epoch+1}/{self.config["training"]["num_epochs"]}], '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}'
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(fold, epoch, is_best=True)
        
        plt.ioff()
        return train_losses, val_losses
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item() * X.size(0)
        return total_loss / len(data_loader.dataset)
    
    def save_checkpoint(self, fold, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fold': fold,
            'epoch': epoch
        }
        
        # 保存最新检查点
        checkpoint_path = Path(self.config['paths']['model_save_dir']) / f'fold_{fold+1}_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = Path(self.config['paths']['model_save_dir']) / f'fold_{fold+1}_best.pth'
            torch.save(checkpoint, best_model_path)
    
    def _update_plot(self, ax, train_epochs, train_losses, val_losses, current_train_loss, fold):
        """更新训练过程可视化"""
        ax.clear()
        ax.semilogy(train_epochs, train_losses, label='Train Loss', alpha=0.6)
        
        if val_losses:  # 如果有验证损失则画出来
            ax.semilogy(range(1, len(val_losses) + 1), val_losses, label='Val Loss', alpha=0.6)
            # 检测过拟合
            if val_losses[-1] > current_train_loss * 1.5:
                ax.text(0.98, 0.98, 'Warning: Overfitting', 
                       transform=ax.transAxes,
                       horizontalalignment='right',
                       verticalalignment='top',
                       color='red',
                       bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'Training Progress (Fold {fold+1})')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.pause(0.1)
        
        # 保存图像
        plt.savefig(Path(self.config['paths']['visualization_dir']) / f'training_fold_{fold+1}.png')

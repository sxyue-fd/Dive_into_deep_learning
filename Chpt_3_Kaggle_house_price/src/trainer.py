import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from pathlib import Path
import time
from utils import get_absolute_path

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
        
        # 创建保存路径（使用绝对路径）
        for path_key in ['model_save_dir', 'log_dir', 'visualization_dir']:
            abs_path = get_absolute_path(config['paths'][path_key])
            config['paths'][path_key] = str(abs_path)  # 更新配置中的路径为绝对路径
            abs_path.mkdir(parents=True, exist_ok=True)
              # 配置日志记录
        self.setup_logging()
        
        # 记录训练配置
        self._log_config()

    def setup_logging(self):
        """配置日志记录器"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.config['paths']['log_dir']) / f'training_{timestamp}.log'
        # 确保日志目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 设置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
    def _log_config(self):
        """记录训练配置信息"""
        logging.info("="*50)
        logging.info("训练配置:")
        logging.info("-"*20)
        logging.info(f"模型结构:")
        logging.info(f"- 输入维度: {self.config['model']['input_size']}")
        logging.info(f"- 隐藏层大小: {self.config['model']['hidden_size']}")
        logging.info(f"- Dropout比率: {self.config['model']['dropout_rate']}")
        logging.info("-"*20)
        logging.info(f"训练参数:")
        logging.info(f"- 批量大小: {self.config['training']['batch_size']}")
        logging.info(f"- 训练轮数: {self.config['training']['num_epochs']}")
        logging.info(f"- 学习率: {self.config['training']['learning_rate']}")
        logging.info(f"- L2正则化: {self.config['training']['weight_decay']}")
        logging.info(f"- 交叉验证折数: {self.config['training']['k_folds']}")
        logging.info("-"*20)
        logging.info(f"设备: {self.device}")
        logging.info("="*50)
        
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
            val_losses.append(val_loss)            # 记录训练信息
            epoch_info = (
                f'Fold {fold+1}, Epoch {epoch+1}/{self.config["training"]["num_epochs"]} | '
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}'
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epoch_info += ' *'  # 标记最佳验证结果
            logging.info(epoch_info)
        
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
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = Path(self.config['paths']['model_save_dir']) / f'fold_{fold+1}_best.pth'
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_model_path)
    
    def _update_plot(self, ax, train_epochs, train_losses, val_losses, current_train_loss, fold):
        """更新训练过程可视化"""
        # 清除当前轴
        ax.clear()
        
        # 绘制训练损失
        ax.semilogy(train_epochs, train_losses, label='Train Loss', alpha=0.7, color='#1f77b4')
        
        # 如果有验证损失则画出来
        if val_losses:
            ax.semilogy(range(1, len(val_losses) + 1), val_losses, label='Val Loss', alpha=0.7, color='#ff7f0e')
            # 检测过拟合
            if val_losses[-1] > current_train_loss * 1.5:
                ax.text(0.98, 0.98, 'Warning: Overfitting', 
                       transform=ax.transAxes,
                       horizontalalignment='right',
                       verticalalignment='top',
                       color='red',
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # 设置图形参数
        ax.set_xlabel('Epochs/Steps')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'Training Progress (Fold {fold+1})')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 刷新图形
        plt.draw()
        plt.pause(0.1)
        
        # 保存图像
        output_path = Path(self.config['paths']['visualization_dir']) / f'training_fold_{fold+1}.png'
        plt.savefig(output_path)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from utils import get_absolute_path

class ModelTrainer:
    """模型训练器"""
    def __init__(self, model, criterion, optimizer, device, config, dataset, session_id):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备
            config: 配置信息
            dataset: 数据集实例
            session_id: 训练会话ID
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.dataset = dataset
        self.session_id = session_id
        
        # 创建保存路径（使用绝对路径）
        for path_key in ['model_save_dir', 'log_dir', 'visualization_dir']:
            abs_path = get_absolute_path(config['paths'][path_key])
            config['paths'][path_key] = str(abs_path)  # 更新配置中的路径为绝对路径
            abs_path.mkdir(parents=True, exist_ok=True)
        
    def train_fold(self, train_loader, val_loader, fold):
        """训练一个fold
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            fold: 当前折数
        
        Returns:
            tuple: (训练损失列表, 验证损失列表)
        """
        from visualization import create_training_plots, create_model_evaluation
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_predictions = None
        best_actuals = None
        
        for epoch in range(self.config['training']['num_epochs']):
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # 验证
            val_loss, predictions, actuals = self._validate(val_loader)
            val_losses.append(val_loss)
            
            # 保存最佳结果
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_predictions = predictions
                best_actuals = actuals
            
            logging.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]} - '
                        f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 创建可视化
        create_training_plots(
            self.config,
            self.session_id,
            train_losses,
            val_losses,
            fold
        )
        
        # 为最佳模型创建评估图
        if best_predictions is not None and best_actuals is not None:
            create_model_evaluation(
                self.config,
                self.session_id,
                best_predictions,
                best_actuals,
                self.dataset
            )
        
        return train_losses, val_losses
    
    def _train_epoch(self, train_loader):
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            # 前向传播
            output = self.model(X)
            loss = self.criterion(output, y)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * X.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def _validate(self, val_loader):
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            tuple: (验证损失, 预测值, 实际值)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss, np.array(predictions), np.array(actuals)
    
    def evaluate(self, test_loader):
        """评估模型在测试集上的表现"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader.dataset)
        return avg_loss, np.array(predictions), np.array(actuals)

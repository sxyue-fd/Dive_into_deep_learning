"""
训练和评估模块
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    """模型训练器"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器 (对于from_zero实现，使用自定义SGD)
        if self.config['model']['type'] == 'from_zero':
            self.optimizer = None
            self.lr = config['training']['learning_rate']
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['training']['learning_rate']
            )
        
        # 创建保存目录
        self.save_dir = Path(config['training']['save_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        self.viz_dir = Path(config['training']['viz_dir'])
        
        for dir_path in [self.save_dir, self.log_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 训练记录
        self.train_losses = []
        self.test_losses = []
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            if self.config['model']['type'] == 'from_zero':
                loss.backward()
                # 手动更新参数
                with torch.no_grad():
                    for param in self.model.parameters():
                        param -= self.lr * param.grad
                        param.grad.zero_()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def train(self, train_loader, test_loader):
        """训练模型"""
        num_epochs = self.config['training']['num_epochs']
        log_interval = self.config['logging']['log_interval']
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 测试
            test_loss = self.evaluate(test_loader)
            self.test_losses.append(test_loss)
            
            if (epoch + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]: '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Test Loss: {test_loss:.4f}')
            
            # 保存检查点
            if (epoch + 1) % 20 == 0:
                self.save_model(f'checkpoint_{epoch+1}.pth')
        
        # 保存最终模型
        self.save_model('final.pth')
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename):
        """保存模型"""
        save_path = self.save_dir / filename
        if self.config['model']['type'] == 'from_zero':
            state_dict = {
                'w': self.model.w,
                'b': self.model.b
            }
        else:
            state_dict = self.model.state_dict()
        
        torch.save({
            'model_state_dict': state_dict,
            'config': self.config
        }, save_path)
    
    def load_model(self, filename):
        """加载模型"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path)
        
        if self.config['model']['type'] == 'from_zero':
            self.model.w = checkpoint['model_state_dict']['w']
            self.model.b = checkpoint['model_state_dict']['b']
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.viz_dir / 'training_curves.png')
        plt.close()
    
    def visualize_predictions(self, test_loader):
        """可视化预测结果"""
        self.model.eval()
        all_data = []
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_data.append(data.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_preds.append(output.cpu().numpy())
        
        # 转换为numpy数组
        X = np.concatenate(all_data)
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        
        # 绘制预测结果
        plt.figure(figsize=(12, 5))
        
        # 特征1的预测
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], y_true, c='b', label='True')
        plt.scatter(X[:, 0], y_pred, c='r', label='Predicted')
        plt.xlabel('Feature 1')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        
        # 特征2的预测
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 1], y_true, c='b', label='True')
        plt.scatter(X[:, 1], y_pred, c='r', label='Predicted')
        plt.xlabel('Feature 2')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'predictions.png')
        plt.close()

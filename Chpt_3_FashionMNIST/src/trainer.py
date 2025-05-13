"""
训练和评估模块
"""
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from .data import get_fashion_mnist_labels

class Trainer:
    """模型训练器"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 创建保存目录
        self.save_dir = Path(config['training']['save_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        self.viz_dir = Path(config['training']['viz_dir'])
        
        for dir_path in [self.save_dir, self.log_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 训练记录
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % self.config['logging']['log_interval'] == 0:
                print(f'Train Batch [{batch_idx}/{len(train_loader)}]: '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(loader), correct / total
    
    def train(self, train_loader, valid_loader):
        """训练模型"""
        num_epochs = self.config['training']['num_epochs']
        best_valid_acc = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            if (epoch + 1) % self.config['logging']['eval_interval'] == 0:
                valid_loss, valid_acc = self.evaluate(valid_loader)
                self.valid_losses.append(valid_loss)
                self.valid_accs.append(valid_acc)
                
                print(f'Valid Loss: {valid_loss:.4f}, Valid Acc: {100.*valid_acc:.2f}%')
                
                # 保存最佳模型
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    self.save_model('best.pth')
            
            # 保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_{epoch+1}.pth')
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename):
        """保存模型"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, save_path)
    
    def load_model(self, filename):
        """加载模型"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        if self.valid_losses:
            valid_epochs = list(range(0, len(self.train_losses), 
                                    self.config['logging']['eval_interval']))
            ax1.plot(valid_epochs, self.valid_losses, label='Valid Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='Train Acc')
        if self.valid_accs:
            ax2.plot(valid_epochs, self.valid_accs, label='Valid Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'training_curves.png')
        plt.close()
    
    def predict(self, x):
        """预测单个样本"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            pred = output.argmax(dim=1)
            return get_fashion_mnist_labels()[pred.item()]

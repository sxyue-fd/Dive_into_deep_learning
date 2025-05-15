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
        self.device = torch.device(config["training"]["device"])
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )
        
        # 创建保存目录
        self.save_dir = Path(config["training"]["save_dir"])
        self.log_dir = Path(config["training"]["log_dir"])
        self.viz_dir = Path(config["training"]["viz_dir"])
        
        for dir_path in [self.save_dir, self.log_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        current_time = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{current_time}.log"
        
        # 记录训练配置
        with open(self.log_file, "w") as f:
            f.write(f"=======================================================\n")
            f.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=======================================================\n")
            f.write("配置参数:\n")
            for key, value in config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            f.write("训练日志:\n")
            # 添加表头
            f.write(f"{'-'*60}\n")
            f.write(f"{'轮次':^10}{'训练损失':^20}{'测试损失':^20}{'耗时(秒)':^10}\n")
            f.write(f"{'-'*60}\n")
        
        # 训练记录
        self.train_losses = []
        self.test_losses = []
        self.epoch_start_time = None
    
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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, loader):
        """评估模型"""
        # 保存当前训练状态以备后续恢复
        training = self.model.training
        # 设置为评估模式
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        # 恢复之前的训练状态
        if training:
            self.model.train()
            
        return total_loss / len(loader)
    
    def train(self, train_loader, test_loader):
        """训练模型"""
        num_epochs = self.config["training"]["num_epochs"]
        log_interval = self.config["logging"]["log_interval"]
        
        # 初始化实时可视化
        plt.ion()  # 开启交互模式
        fig = plt.figure(figsize=(10, 6))  # 创建一个固定的图形
        
        try:
            for epoch in range(num_epochs):
                # 记录epoch开始时间
                self.epoch_start_time = time.time()
                
                # 训练
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # 测试
                test_loss = self.evaluate(test_loader)
                self.test_losses.append(test_loss)
                
                # 计算耗时
                epoch_time = time.time() - self.epoch_start_time
                
                if (epoch + 1) % log_interval == 0:
                    # 表格化输出
                    log_message = f"{epoch+1:^10}{train_loss:^20.6f}{test_loss:^20.6f}{epoch_time:^10.2f}"
                    print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}, Time: {epoch_time:.2f}s")
                    
                    # 记录到日志文件
                    with open(self.log_file, "a") as f:
                        f.write(f"{log_message}\n")
        finally:
            plt.ioff()  # 关闭交互模式
            plt.close(fig)  # 确保关闭图形
        
        # 保存最终模型
        final_model_path = self.save_model("final.pth")
        
        # 获取真实参数和学习到的参数
        from utils import get_true_params
        true_w, true_b = get_true_params(self.config)
        
        learned_w = self.model.linear.weight.cpu().detach().numpy().flatten()
        learned_b = self.model.linear.bias.cpu().detach().numpy().item()
        
        # 记录训练完成信息
        with open(self.log_file, "a") as f:
            f.write(f"{'-'*60}\n")
            f.write(f"\n=======================================================\n")
            f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最终模型保存路径: {final_model_path}\n")
            f.write(f"=======================================================\n\n")
            
            # 记录最终损失值
            f.write(f"训练结果汇总:\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{'初始训练损失':^20}{'最终训练损失':^20}{'初始测试损失':^20}{'最终测试损失':^20}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{self.train_losses[0]:^20.6f}{self.train_losses[-1]:^20.6f}{self.test_losses[0]:^20.6f}{self.test_losses[-1]:^20.6f}\n")
            f.write(f"{'-'*60}\n\n")
            
            # 记录参数对比
            f.write(f"参数对比:\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{'参数':^10}{'真实值':^20}{'学习值':^20}{'误差':^10}\n")
            f.write(f"{'-'*60}\n")
            
            # 记录权重参数
            for i, (tw, lw) in enumerate(zip(true_w, learned_w)):
                error = abs(tw - lw) / (abs(tw) + 1e-8) * 100  # 相对误差百分比
                f.write(f"{'w' + str(i):^10}{tw:^20.6f}{lw:^20.6f}{error:^10.2f}%\n")
            
            # 记录偏置参数
            error_b = abs(true_b - learned_b) / (abs(true_b) + 1e-8) * 100
            f.write(f"{'b':^10}{true_b:^20.6f}{learned_b:^20.6f}{error_b:^10.2f}%\n")
            f.write(f"{'-'*60}\n")
        
        # 绘制最终训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename):
        """保存模型"""
        save_path = self.save_dir / filename
        state_dict = self.model.state_dict()
        
        torch.save({
            "model_state_dict": state_dict,
            "config": self.config,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, filename):
        """加载模型"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        from utils import plot_training_curves
        plot_training_curves(
            self.train_losses,
            self.test_losses,
            self.viz_dir / "training_curves.png"
        )
    
    def visualize_predictions(self, test_loader):
        """可视化预测结果"""
        from utils import visualize_predictions
        
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
        
        # 获取真实参数和学习到的参数
        learned_w = self.model.linear.weight.cpu().detach().numpy().T
        learned_b = self.model.linear.bias.cpu().detach().numpy().item()
        
        # 从配置获取真实参数
        from utils import get_true_params
        true_w, true_b = get_true_params(self.config)
        
        # 使用utils中的可视化函数
        visualize_predictions(
            X, 
            y_true, 
            None,  # 不传入带噪声的y_pred
            self.viz_dir / "predictions.png",
            true_w,
            true_b,
            learned_w.reshape(-1),
            learned_b
        )

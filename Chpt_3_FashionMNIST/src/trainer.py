"""
训练和评估模块
"""
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data import get_fashion_mnist_labels

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        # 创建时间戳（用于日志和结果文件）
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        # 确保weight_decay是浮点数
        weight_decay = float(config['training']['weight_decay'])
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=weight_decay
        )
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        
        # 创建保存目录（处理相对路径）
        save_dir = config['training'].get('save_dir', 'outputs/models')
        log_dir = config['training'].get('log_dir', 'outputs/logs')
        viz_dir = config['training'].get('viz_dir', 'outputs/visualizations')
        
        # 移除开头的./以确保正确拼接路径
        self.save_dir = project_root / save_dir.lstrip('./')
        self.log_dir = project_root / log_dir.lstrip('./')
        self.viz_dir = project_root / viz_dir.lstrip('./')
        
        # 创建所有必要的目录
        for dir_path in [self.save_dir, self.log_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 设置日志记录
        self.setup_logging()
            
        # 训练记录
        self.train_losses = []
        self.train_accs = []
        
    def setup_logging(self):
        """配置日志记录器"""
        log_file = self.log_dir / f'training_{self.timestamp}.log'
        
        # 配置日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
          # 记录训练配置
        logging.info("="*50)
        logging.info("数据集信息:")
        logging.info(f"- 数据集: {self.config['data']['dataset']}")
        logging.info(f"- 类别数量: {self.config['data']['num_classes']}")
        logging.info(f"- 输入维度: {self.config['data']['input_size']}")
        logging.info(f"- 训练集大小: {self.config['data']['train_size']}")
        logging.info(f"- 测试集大小: {self.config['data']['test_size']}")
        logging.info(f"- 批次大小: {self.config['data']['batch_size']}")
        
        logging.info("\n模型配置:")
        logging.info(f"- 模型类型: {self.config['model']['type']}")
        logging.info(f"- 隐藏层: {self.config['model']['hidden_units']}")
        logging.info(f"- Dropout率: {self.config['model']['dropout_rate']}")
        
        logging.info("\n训练配置:")
        logging.info(f"- 学习率: {self.config['training']['learning_rate']}")
        logging.info(f"- Weight Decay: {self.config['training']['weight_decay']}")
        logging.info(f"- 训练轮次: {self.config['training']['num_epochs']}")
        logging.info(f"- 设备: {self.device}")
        logging.info("="*50)

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        log_interval = self.config['logging']['log_interval']
        steps_per_save = log_interval  # 使用相同的间隔来记录数据点
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            # 累积统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 每隔steps_per_save个batch记录一次数据点
            if batch_idx % steps_per_save == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = correct / total
                self.train_losses.append(current_loss)
                self.train_accs.append(current_acc)
            
            if batch_idx % log_interval == 0:
                logging.info(f'Batch: {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
        
        # 计算整个epoch的平均损失和准确率
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
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
    
    def save_training_results(self, test_loss, test_acc):
        """保存训练结果摘要"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.log_dir / f'training_result_{timestamp}.txt'
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write("训练结果摘要\n")
            f.write(f"训练时间: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            
            # 写入模型配置
            f.write("模型配置:\n")
            f.write(f"- 类型: {self.config['model']['type']}\n")
            f.write(f"- 隐藏层: {self.config['model']['hidden_units']}\n")
            f.write(f"- Dropout率: {self.config['model']['dropout_rate']}\n\n")
            
            # 写入训练配置
            f.write("训练配置:\n")
            f.write(f"- 训练轮次: {self.config['training']['num_epochs']}\n")
            f.write(f"- 学习率: {self.config['training']['learning_rate']}\n")
            f.write(f"- Weight Decay: {self.config['training']['weight_decay']}\n")
            f.write(f"- 设备: {self.config['training']['device']}\n\n")
            
            # 写入训练过程数据
            f.write("训练过程:\n")
            f.write(f"- 最终训练损失: {self.train_losses[-1]:.4f}\n")
            f.write(f"- 最终训练准确率: {self.train_accs[-1]*100:.2f}%\n\n")            # 写入测试结果
            f.write("\n测试结果:\n")
            # 修复格式化字符串错误：确保值不是None再进行格式化
            if test_loss is not None:
                f.write(f"- 测试损失: {test_loss:.4f}\n")
            else:
                f.write("- 测试损失: N/A\n")
                
            if test_acc is not None:
                f.write(f"- 测试准确率: {test_acc*100:.2f}%\n")
            else:
                f.write("- 测试准确率: N/A\n")
            
            logging.info(f"训练结果已保存至: {result_file}")
    def train(self, train_loader):
        """训练模型"""
        num_epochs = self.config['training']['num_epochs']
        
        # 初始化实时可视化
        plt.ion()  # 开启交互模式
        
        # 创建实时训练曲线图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.show(block=False)
        
        # 清空之前的记录
        self.train_losses = []
        self.train_accs = []
        
        for epoch in range(num_epochs):
            logging.info(f'\n{"="*20} Epoch {epoch+1}/{num_epochs} {"="*20}')
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 更新实时训练曲线
            if len(self.train_losses) > 0:  # 只在有数据时更新
                self._update_training_plot()
            
            # 输出训练指标摘要
            logging.info(f'Train - Loss: {train_loss:.4f}, Acc: {100.*train_acc:.2f}%')
            
            # 保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_{epoch+1}.pth')
                logging.info(f'已保存检查点: checkpoint_{epoch+1}.pth')

            # 最后一个epoch保存为最佳模型
            if epoch == num_epochs - 1:
                self.save_model('best.pth')
                logging.info(f'已保存最终模型为best.pth')
        
        plt.ioff()  # 关闭交互模式
        plt.close(self.fig)  # 关闭实时训练图形
        
        # 保存最终训练曲线
        self.plot_training_curves()
        
        # 加载最佳模型用于保存结果
        self.load_model('best.pth')
        
        # 暂时不保存测试结果，因为测试集应该在完整训练结束后才使用一次
        self.save_training_results(None, None)
        logging.info("训练完成。请使用测试集进行最终评估。")
    
    def save_model(self, filename):
        """保存模型"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, save_path)
        logging.info(f'模型已保存: {save_path}')
    
    def load_model(self, filename):
        """加载模型"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f'模型已加载: {load_path}')

    def plot_training_curves(self):
        """绘制训练曲线"""
        try:
            from utils import plot_training_curves
            steps_per_point = self.config['logging']['log_interval']
            plot_training_curves(
                self.train_losses,
                None,  # 不使用验证损失
                self.train_accs,
                None,  # 不使用验证准确率
                steps_per_point,  # 每个数据点之间的batch数
                self.viz_dir / 'training_curves.png'
            )
        except Exception as e:
            logging.error(f"绘制训练曲线时出错: {str(e)}")
    
    def predict(self, x):
        """预测单个样本"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            pred = output.argmax(dim=1)
            return get_fashion_mnist_labels()[pred.item()]    
    def visualize_predictions(self, test_loader):
        """可视化预测结果"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_images = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                # 保存所有图像、标签和预测结果
                all_images.append(images.cpu())
                all_preds.extend(preds.cpu().numpy())                
                all_labels.extend(labels.cpu().numpy())
        
        # 将所有图像合并成一个张量
        all_images = torch.cat(all_images, 0)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # 绘制混淆矩阵
        from utils import plot_confusion_matrix, get_fashion_mnist_labels
        plot_confusion_matrix(
            cm,
            get_fashion_mnist_labels(),
            self.viz_dir / 'confusion_matrix.png'
        )
        
        # 可视化样本预测（从所有测试集中随机选择）
        from utils import plot_sample_predictions
        plot_sample_predictions(
            all_images,
            torch.tensor(all_labels),
            torch.tensor(all_preds),
            self.viz_dir / 'sample_predictions.png'
        )
        
        logging.info(f"预测可视化已保存到: {self.viz_dir}")
    
    def visualize_dataset(self, train_loader):
        """可视化数据集分布"""
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        from utils import plot_class_distribution
        plot_class_distribution(
            np.array(all_labels),
            self.viz_dir / 'class_distribution.png'
        )
        
        logging.info(f"数据集分布可视化已保存到: {self.viz_dir}")
    def _update_training_plot(self):
        """更新实时训练曲线"""
        # 每10个数据点更新一次图表，减少绘图频率
        if len(self.train_losses) % 10 != 0:
            return
            
        # 清除当前图形内容
        self.ax1.clear()
        self.ax2.clear()
        
        # 计算步数
        steps = np.arange(len(self.train_losses)) * self.config['logging']['log_interval']
        
        # 绘制损失曲线 (使用对数坐标)
        train_line, = self.ax1.semilogy(steps, self.train_losses, 
                                      label='Loss', 
                                      color='#1f77b4', 
                                      alpha=0.7)
        
        # 设置损失曲线属性
        self.ax1.set_xlabel('Steps (batches)')
        self.ax1.set_ylabel('Loss (log scale)')
        self.ax1.set_title('Training Progress')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制准确率曲线
        self.ax2.plot(steps, self.train_accs, 
                     label='Accuracy', 
                     color='#2ca02c', 
                     alpha=0.7)
        
        # 设置准确率曲线属性
        self.ax2.set_xlabel('Steps (batches)')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend(loc='lower right')
        self.ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 更新布局和显示
        if not hasattr(self, 'bg'):
            plt.tight_layout()
            self.fig.canvas.draw()
            # 保存背景
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        else:
            # 恢复背景
            self.fig.canvas.restore_region(self.bg)
            
        # 刷新画布
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

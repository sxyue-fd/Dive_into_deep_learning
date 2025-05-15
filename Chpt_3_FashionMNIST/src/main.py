"""
主程序
"""
import os
import argparse
import sys
from pathlib import Path

# 获取项目根目录
current_dir = Path(__file__).parent
project_root = current_dir.parent

import torch
# 使用相对导入格式，避免使用src前缀
from data import load_fashion_mnist
from model import create_model
from trainer import Trainer
from utils import load_config, set_seed

def main():    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Fashion-MNIST Training')
    default_config_path = os.path.join(project_root, 'configs', 'config.yaml')
    parser.add_argument('--config', default=default_config_path,
                      help='path to config file')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
      # 准备数据
    train_loader, test_loader = load_fashion_mnist(config)
    
    # 创建模型
    model = create_model(config)
    
    # 创建训练器
    trainer = Trainer(model, config)
    
    # 可视化数据集分布
    trainer.visualize_dataset(train_loader)
    
    # 训练模型
    print("Starting training...")
    trainer.train(train_loader)
    
    # 加载最终模型进行测试集评估
    trainer.load_model('best.pth')
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    
    # 保存包含测试结果的完整报告
    trainer.save_training_results(test_loss, test_acc)
    
    # 可视化预测结果
    trainer.visualize_predictions(test_loader)
    
if __name__ == '__main__':
    main()

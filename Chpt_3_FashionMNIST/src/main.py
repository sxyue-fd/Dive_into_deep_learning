"""
主程序
"""
import os
import argparse
from pathlib import Path

import torch
from .data import load_fashion_mnist
from .model import create_model
from .trainer import Trainer
from .utils import load_config, set_seed

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Fashion-MNIST Training')
    parser.add_argument('--config', default='../configs/config.yaml',
                      help='path to config file')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 准备数据
    train_loader, valid_loader, test_loader = load_fashion_mnist(config)
    
    # 创建模型
    model = create_model(config)
    
    # 创建训练器
    trainer = Trainer(model, config)
    
    # 训练模型
    print("Starting training...")
    trainer.train(train_loader, valid_loader)
    
    # 在测试集上评估
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {100.*test_acc:.2f}%")

if __name__ == '__main__':
    main()

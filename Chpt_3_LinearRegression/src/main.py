"""
主程序
"""
import os
import argparse
from pathlib import Path

import torch
from .data import generate_data
from .model import create_model
from .trainer import Trainer
from .utils import load_config, set_seed, get_true_params

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Linear Regression Training')
    parser.add_argument('--config', default='../configs/config.yaml',
                      help='path to config file')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取真实参数
    true_w, true_b = get_true_params(config)
    print(f"True parameters - w: {true_w}, b: {true_b}")
    
    # 生成数据
    train_loader, test_loader = generate_data(config)
    
    # 创建模型
    model = create_model(config)
    
    # 创建训练器
    trainer = Trainer(model, config)
    
    # 训练模型
    print("\nStarting training...")
    trainer.train(train_loader, test_loader)
    
    # 可视化结果
    trainer.visualize_predictions(test_loader)
    
    # 打印学习到的参数
    if config['model']['type'] == 'from_zero':
        learned_w = model.w.detach().numpy()
        learned_b = model.b.detach().numpy()
    else:
        learned_w = model.linear.weight.detach().numpy()
        learned_b = model.linear.bias.detach().numpy()
    
    print("\nLearned parameters:")
    print(f"w: {learned_w.flatten()}")
    print(f"b: {learned_b.item()}")

if __name__ == '__main__':
    main()

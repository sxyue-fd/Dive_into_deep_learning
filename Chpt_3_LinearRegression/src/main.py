"""
Copyright (c) 2025
此项目基于 MIT License 进行许可
有关详细信息，请参阅项目根目录中的 LICENSE 文件
"""

"""
主程序
"""
import os
import argparse
from pathlib import Path
import sys

# 将src目录添加到路径中，以便能够正确导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from data import get_data_loaders
from model import create_model
from trainer import Trainer
from utils import load_config, set_seed, get_true_params

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
    
    # 构建配置文件的绝对路径
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / 'configs' / 'config.yaml'
    if args.config != '../configs/config.yaml':
        config_path = Path(args.config)
    
    # 加载配置
    config = load_config(str(config_path))
    
    # 更新路径为绝对路径
    output_base_dir = current_dir.parent / 'outputs'
    output_base_dir.mkdir(parents=True, exist_ok=True)
    config['training']['save_dir'] = str(output_base_dir / 'models')
    config['training']['log_dir'] = str(output_base_dir / 'logs')
    config['training']['viz_dir'] = str(output_base_dir / 'visualizations')
    config['data']['data_dir'] = str(current_dir.parent / 'data')
    
    # 获取真实参数
    true_w, true_b = get_true_params(config)
    print(f"True parameters - w: {true_w}, b: {true_b}")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(config)
    
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
    learned_w = model.linear.weight.cpu().detach().numpy()
    learned_b = model.linear.bias.cpu().detach().numpy()
    
    print("\nLearned parameters:")
    print(f"w: {learned_w.flatten()}")
    print(f"b: {learned_b.item()}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
配置兼容性测试脚本
测试新旧配置格式的兼容性
"""

import sys
import yaml
from pathlib import Path

# 添加src路径
sys.path.append(str(Path(__file__).parent / 'src'))

from performance_config import (
    load_performance_config, 
    get_early_stopping_config, 
    get_random_seed,
    get_config_value
)

def test_old_config_compatibility():
    """测试原始配置文件的兼容性"""
    print("🧪 测试原始配置文件兼容性...")
    
    # 加载原始配置文件
    old_config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    
    with open(old_config_path, 'r', encoding='utf-8') as f:
        old_config = yaml.safe_load(f)
    
    # 测试早停配置获取
    early_stopping = get_early_stopping_config(old_config)
    print(f"  ✓ 早停配置: {early_stopping}")
    
    # 测试随机种子获取
    random_seed = get_random_seed(old_config)
    print(f"  ✓ 随机种子: {random_seed}")
    
    # 测试配置值获取
    learning_rate = get_config_value(old_config, ['training', 'learning_rate'])
    print(f"  ✓ 学习率: {learning_rate}")
    
    batch_size = get_config_value(old_config, ['data', 'batch_size'])
    print(f"  ✓ 批次大小: {batch_size}")
    
    print("  ✅ 原始配置文件兼容性测试通过\n")

def test_performance_config():
    """测试性能配置文件"""
    print("🚀 测试性能配置文件...")
    
    # 加载性能配置文件
    performance_config_path = Path(__file__).parent / 'configs' / 'config_performance.yaml'
    
    if not performance_config_path.exists():
        print("  ⚠️ 性能配置文件不存在，跳过测试")
        return
    
    try:
        performance_config = load_performance_config(str(performance_config_path))
        print(f"  ✓ 性能配置加载成功")
        
        # 测试预设加载
        preset = performance_config.get('performance_preset', 'quick_test')
        print(f"  ✓ 当前预设: {preset}")
        
        # 测试早停配置
        early_stopping = get_early_stopping_config(performance_config)
        print(f"  ✓ 早停配置: {early_stopping}")
        
        # 测试关键参数
        epochs = get_config_value(performance_config, ['training', 'epochs'])
        learning_rate = get_config_value(performance_config, ['training', 'learning_rate'])
        batch_size = get_config_value(performance_config, ['data', 'batch_size'])
        
        print(f"  ✓ 关键参数 - epochs: {epochs}, lr: {learning_rate}, batch_size: {batch_size}")
        
        print("  ✅ 性能配置文件测试通过\n")
        
    except Exception as e:
        print(f"  ❌ 性能配置文件测试失败: {e}\n")

def test_trainer_initialization():
    """测试训练器初始化"""
    print("🏋️ 测试训练器初始化...")
    
    try:
        from trainer import ResNetTrainer
        from model import create_resnet18
        
        # 使用原始配置测试
        old_config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        with open(old_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建简单模型
        model = create_resnet18(10)
        
        # 创建训练器（应该能够成功初始化）
        trainer = ResNetTrainer(model, config)
        print("  ✓ 训练器初始化成功（原始配置）")
        
        # 如果性能配置存在，也测试它
        performance_config_path = Path(__file__).parent / 'configs' / 'config_performance.yaml'
        if performance_config_path.exists():
            performance_config = load_performance_config(str(performance_config_path))
            trainer_perf = ResNetTrainer(model, performance_config)
            print("  ✓ 训练器初始化成功（性能配置）")
        
        print("  ✅ 训练器初始化测试通过\n")
        
    except Exception as e:
        print(f"  ❌ 训练器初始化测试失败: {e}\n")

def test_data_loader_compatibility():
    """测试数据加载器兼容性"""
    print("📊 测试数据加载器兼容性...")
    
    try:
        from data import CIFAR10DataLoader
        
        # 测试原始配置
        old_config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        with open(old_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建数据加载器
        data_loader = CIFAR10DataLoader(config)
        print("  ✓ 数据加载器创建成功（原始配置）")
        
        # 测试随机种子获取
        random_seed = get_random_seed(config)
        print(f"  ✓ 随机种子获取: {random_seed}")
        
        print("  ✅ 数据加载器兼容性测试通过\n")
        
    except Exception as e:
        print(f"  ❌ 数据加载器兼容性测试失败: {e}\n")

def main():
    """主测试函数"""
    print("🔧 ResNet 配置兼容性测试")
    print("=" * 50)
    
    # 运行所有测试
    test_old_config_compatibility()
    test_performance_config()
    test_trainer_initialization()
    test_data_loader_compatibility()
    
    print("🎉 所有兼容性测试完成!")

if __name__ == "__main__":
    main()

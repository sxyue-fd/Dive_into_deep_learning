"""
ResNet18 CIFAR-10 项目测试模块
测试各个组件的功能性
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
import os
import unittest
from pathlib import Path

# 设置项目根目录和Python导入路径
current_dir = Path(__file__).parent.absolute()
os.environ['PROJECT_ROOT'] = str(current_dir)
sys.path.insert(0, str(current_dir))

# 导入项目模块
try:
    from src.model import create_resnet18, BasicBlock, ResNet18
    from src.data import CIFAR10DataLoader, create_data_loaders
    from src.trainer import ResNetTrainer
    from src.utils import (
        AverageMeter, EarlyStopping, calculate_accuracy, 
        setup_logging, set_random_seed
    )
    from src.visualization import ResNetVisualizer
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"Python路径: {sys.path}")
    raise


class TestResNet18Model(unittest.TestCase):
    """测试ResNet18模型"""
    
    def setUp(self):
        self.model = create_resnet18(num_classes=10)
        self.input_tensor = torch.randn(4, 3, 32, 32)
    
    def test_model_creation(self):
        """测试模型创建"""
        self.assertIsInstance(self.model, ResNet18)
        self.assertEqual(self.model.fc.out_features, 10)
    
    def test_forward_pass(self):
        """测试前向传播"""
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (4, 10))
        self.assertFalse(torch.isnan(output).any())
    
    def test_feature_extraction(self):
        """测试特征提取"""
        features = self.model.get_feature_maps(self.input_tensor)
        self.assertIn('conv1', features)
        self.assertIn('layer1', features)
        self.assertIn('layer4', features)
    
    def test_parameter_count(self):
        """测试参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        # ResNet18大约有11M参数
        self.assertGreater(total_params, 10_000_000)
        self.assertLess(total_params, 15_000_000)


class TestBasicBlock(unittest.TestCase):
    """测试BasicBlock"""
    
    def setUp(self):
        self.block = BasicBlock(64, 64)
        self.input_tensor = torch.randn(2, 64, 32, 32)
    
    def test_basic_block_forward(self):
        """测试BasicBlock前向传播"""
        output = self.block(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)
    
    def test_basic_block_with_downsample(self):
        """测试带下采样的BasicBlock"""
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        block = BasicBlock(64, 128, stride=2, downsample=downsample)
        output = block(self.input_tensor)
        self.assertEqual(output.shape, (2, 128, 16, 16))


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def setUp(self):
        self.config = {
            'data': {
                'dataset': 'cifar10',
                'data_root': './test_data',
                'batch_size': 32,
                'num_workers': 0,
                'pin_memory': False,
                'transforms': {
                    'train': {
                        'random_horizontal_flip': 0.5,
                        'random_crop': {'size': 32, 'padding': 4}
                    },
                    'normalize': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                },
                'validation_split': 0.2,
                'shuffle': True,
                'random_seed': 42
            }
        }
        self.data_loader = CIFAR10DataLoader(self.config)
    
    def test_data_loader_creation(self):
        """测试数据加载器创建"""
        self.assertEqual(len(self.data_loader.class_names), 10)
        self.assertIn('airplane', self.data_loader.class_names)
    
    def test_transforms(self):
        """测试数据变换"""
        train_transform, val_transform = self.data_loader.get_transforms()
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(val_transform)


class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def test_average_meter(self):
        """测试AverageMeter"""
        meter = AverageMeter()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for val in values:
            meter.update(val)
        
        self.assertEqual(meter.avg, 3.0)
        self.assertEqual(meter.count, 5)
    
    def test_early_stopping(self):
        """测试早停机制"""
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # 模拟损失下降然后上升
        losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for loss in losses:
            early_stopping(loss)
            if early_stopping.early_stop:
                break
        
        self.assertTrue(early_stopping.early_stop)
    
    def test_accuracy_calculation(self):
        """测试准确率计算"""
        # 创建测试数据
        output = torch.tensor([
            [3.0, 1.0, 2.0],  # 预测类别0
            [1.0, 3.0, 2.0],  # 预测类别1
            [1.0, 2.0, 3.0],  # 预测类别2
            [3.0, 2.0, 1.0]   # 预测类别0
        ])
        target = torch.tensor([0, 1, 2, 0])  # 真实标签
        
        acc1, = calculate_accuracy(output, target, topk=(1,))
        self.assertEqual(acc1.item(), 100.0)  # 100%准确率


class TestTrainer(unittest.TestCase):
    """测试训练器（基础功能）"""
    
    def setUp(self):
        self.config = self._create_test_config()
        self.model = create_resnet18(10)
        
    def _create_test_config(self):
        """创建测试配置"""
        return {
            'device': {'use_cuda': False, 'mixed_precision': False},
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'momentum': 0.9,
                'optimizer': 'adam',
                'optimizer_params': {'betas': [0.9, 0.999], 'eps': 1e-8},
                'scheduler': 'cosine',
                'scheduler_params': {'T_max': 10, 'eta_min': 1e-6},
                'early_stopping': {
                    'patience': 5,
                    'min_delta': 1e-4,
                    'mode': 'min'
                },
                'epochs': 5
            },
            'logging': {'log_frequency': 10},
            'checkpoint': {'save_frequency': 2},
            'paths': {'models': './test_outputs/models'}
        }
    
    def test_trainer_creation(self):
        """测试训练器创建"""
        trainer = ResNetTrainer(self.model, self.config)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.criterion)


def run_integration_test():
    """运行集成测试"""
    print("\n" + "=" * 60)
    print("运行集成测试...")
    print("=" * 60)
    
    try:
        # 加载配置
        config_path = current_dir / 'configs' / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修改配置用于测试
        config['data']['batch_size'] = 16
        config['training']['epochs'] = 2
        config['data']['num_workers'] = 0
        
        print("✓ 配置加载成功")
        
        # 设置随机种子
        set_random_seed(42)
        print("✓ 随机种子设置成功")
        
        # 创建模型
        model = create_resnet18(10)
        print("✓ 模型创建成功")
        
        # 创建数据加载器
        data_loader_obj = CIFAR10DataLoader(config)
        print("✓ 数据加载器创建成功")
        
        # 测试数据加载（小批量）
        train_transform, val_transform = data_loader_obj.get_transforms()
        print("✓ 数据变换创建成功")
        
        # 创建可视化器
        visualizer = ResNetVisualizer(config)
        print("✓ 可视化器创建成功")
        
        print("\n所有组件集成测试通过！")
        return True
        
    except Exception as e:
        print(f"集成测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("ResNet18 CIFAR-10 项目测试")
    print("=" * 60)
    
    # 运行单元测试
    print("运行单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestResNet18Model,
        TestBasicBlock,
        TestDataLoader,
        TestUtils,
        TestTrainer
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n单元测试结果:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    # 运行集成测试
    integration_success = run_integration_test()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"单元测试: {'通过' if result.wasSuccessful() else '失败'}")
    print(f"集成测试: {'通过' if integration_success else '失败'}")
    
    if result.wasSuccessful() and integration_success:
        print("🎉 所有测试通过！项目准备就绪。")
        return True
    else:
        print("❌ 部分测试失败，请检查代码。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

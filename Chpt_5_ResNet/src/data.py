"""
CIFAR-10 数据加载和预处理模块
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict
import logging
import os


class CIFAR10DataLoader:
    """CIFAR-10数据加载器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.logger = logging.getLogger(__name__)
        
        # CIFAR-10类别名称
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """获取训练和验证的数据变换
        
        Returns:
            训练变换和验证变换
        """
        # 标准化参数
        normalize = transforms.Normalize(
            mean=self.data_config['transforms']['normalize']['mean'],
            std=self.data_config['transforms']['normalize']['std']
        )
        
        # 训练变换（包含数据增强）
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(
                p=self.data_config['transforms']['train']['random_horizontal_flip']
            ),
            transforms.RandomCrop(
                size=self.data_config['transforms']['train']['random_crop']['size'],
                padding=self.data_config['transforms']['train']['random_crop']['padding']
            ),
            transforms.ToTensor(),
            normalize
        ])
        
        # 验证变换（仅标准化）
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        return train_transform, val_transform
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载CIFAR-10数据
        
        Returns:
            训练、验证、测试数据加载器
        """
        train_transform, val_transform = self.get_transforms()
        
        # 确保数据目录是相对于项目根目录的
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_root = os.path.abspath(os.path.join(project_root, self.data_config['data_root']))
        os.makedirs(data_root, exist_ok=True)  # 确保数据目录存在
        
        # 下载并加载训练数据
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform
        )
        
        # 下载并加载测试数据
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=val_transform
        )
        
        # 分割训练集为训练和验证集
        val_size = int(len(train_dataset) * self.data_config['validation_split'])
        train_size = len(train_dataset) - val_size
        
        # 设置随机种子以确保可重现性
        generator = torch.Generator().manual_seed(self.data_config['random_seed'])
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
        
        # 为验证集设置正确的变换
        val_dataset.dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=val_transform
        )
        
        self.logger.info(f"训练集大小: {len(train_dataset)}")
        self.logger.info(f"验证集大小: {len(val_dataset)}")
        self.logger.info(f"测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=self.data_config['shuffle'],
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory']
        )
        
        return train_loader, val_loader, test_loader
    
    def get_sample_data(self, num_samples: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取样本数据用于可视化
        
        Args:
            num_samples: 样本数量
            
        Returns:
            样本图像和标签
        """
        _, val_transform = self.get_transforms()
        
        # 确保数据目录是相对于项目根目录的
        data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.data_config['data_root'])
        
        dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=val_transform
        )
        
        # 随机采样
        indices = torch.randperm(len(dataset))[:num_samples]
        samples = [dataset[i] for i in indices]
        
        images = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples])
        
        return images, labels
    
    def get_class_distribution(self, loader: DataLoader) -> Dict[str, int]:
        """获取类别分布
        
        Args:
            loader: 数据加载器
            
        Returns:
            类别分布字典
        """
        class_counts = {name: 0 for name in self.class_names}
        
        for _, labels in loader:
            for label in labels:
                class_counts[self.class_names[label.item()]] += 1
        
        return class_counts


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        训练、验证、测试数据加载器
    """
    data_loader = CIFAR10DataLoader(config)
    return data_loader.load_data()


# 测试代码
if __name__ == "__main__":
    import yaml
    import os
    
    # 加载配置 - 修正路径问题
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据加载器
    data_loader = CIFAR10DataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    print("数据加载器创建成功！")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个批次
    for images, labels in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签范围: {labels.min()} - {labels.max()}")
        break
    
    # 测试类别分布
    class_dist = data_loader.get_class_distribution(test_loader)
    print("测试集类别分布:")
    for class_name, count in class_dist.items():
        print(f"  {class_name}: {count}")

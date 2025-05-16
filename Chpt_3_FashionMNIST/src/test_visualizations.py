"""
Copyright (c) 2025
此项目基于 MIT License 进行许可
有关详细信息，请参阅项目根目录中的 LICENSE 文件
"""

"""
测试可视化函数脚本
"""
import os
import torch
import numpy as np
from data import get_fashion_mnist_labels, load_fashion_mnist
from utils import (
    set_seed, plot_training_curves, plot_confusion_matrix,
    plot_sample_predictions, plot_class_distribution,
    plot_training_progress
)

def main():
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    output_dir = "../outputs/visualizations/test_results"
    os.makedirs(output_dir, exist_ok=True)    # 加载一小部分测试数据
    config = {
        'data': {
            'root_dir': 'data',
            'batch_size': 100,
            'num_workers': 0
        }
    }
    _, test_data = load_fashion_mnist(config)
    images, labels = next(iter(test_data))
    
    # 1. 测试类别分布图
    print("测试类别分布图...")
    plot_class_distribution(
        labels.numpy(),
        os.path.join(output_dir, "class_distribution.png")
    )
    
    # 2. 测试混淆矩阵
    print("测试混淆矩阵...")
    # 随机生成一些预测结果
    pred_labels = torch.randint(0, 10, labels.shape)
    cm = torch.zeros(10, 10, dtype=torch.int)
    for t, p in zip(labels, pred_labels):
        cm[t, p] += 1
    plot_confusion_matrix(
        cm.numpy(),
        get_fashion_mnist_labels(),
        os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # 3. 测试样本预测可视化
    print("测试样本预测可视化...")
    plot_sample_predictions(
        images.numpy(),
        labels.numpy(),
        pred_labels.numpy(),
        os.path.join(output_dir, "sample_predictions.png")
    )
    
    # 4. 测试训练曲线
    print("测试训练曲线...")
    # 模拟训练过程数据
    num_epochs = 50
    train_losses = np.exp(-np.linspace(0, 2, num_epochs)) + 0.1 * np.random.randn(num_epochs)
    train_accs = 1 - np.exp(-np.linspace(0, 2, num_epochs)) + 0.05 * np.random.randn(num_epochs)
    plot_training_curves(
        train_losses,
        None,
        train_accs,
        None,
        100,  # steps_per_point
        os.path.join(output_dir, "training_curves.png")
    )
    
    # 5. 测试训练进度可视化
    print("测试训练进度可视化...")
    # 模拟包含验证集的训练数据
    valid_losses = train_losses + 0.2 * np.random.randn(num_epochs)
    valid_accs = train_accs - 0.1 * np.random.randn(num_epochs)
    plot_training_progress(
        epoch=25,  # 模拟训练到第25轮
        train_losses=train_losses.tolist(),
        train_accs=train_accs.tolist(),
        valid_losses=valid_losses.tolist(),
        valid_accs=valid_accs.tolist(),
        eval_interval=1,
        output_path=os.path.join(output_dir, "training_progress.png")
    )
    
    print(f"\n所有可视化图表已保存到: {output_dir}")
    print("生成的图表包括：")
    print("1. class_distribution.png - 类别分布图")
    print("2. confusion_matrix.png - 混淆矩阵")
    print("3. sample_predictions.png - 样本预测可视化")
    print("4. training_curves.png - 训练曲线")
    print("5. training_progress.png - 训练进度可视化")

if __name__ == "__main__":
    main()

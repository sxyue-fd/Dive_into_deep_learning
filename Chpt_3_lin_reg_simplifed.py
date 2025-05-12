import torch
import torch.nn as nn
import torch.optim as optim
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
from visualization_utils import plot_training_progress, plot_regression_result

# 此文件是对Chpt_3_lin_reg_from_zero.py的简化版本实现

# 生成数据集，与from_zero一致
# 样本数1000，特征数2，y=wX+b+噪声，噪声服从均值为0，标准差为0.01的正态分布
# Data size: y(1000,), X(1000,2), w(2,), b(1,)
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)# features=X，大小为(1000，2)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b # labels=y=w1*x1+w2*x2+b，大小为(1000,)，最后一个加法用了广播机制
labels += torch.tensor(np.random.normal(0, .1, size=labels.size()), dtype=torch.float32) # 添加噪声

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 使用torch.nn模块定义线性回归模型，这是一个单层神经网络，也是全连接层
class LinearNet(nn.Module): # nn是torch的模块，Module是nn的核心数据结构
    def __init__(self, n_feature):
        super(LinearNet, self).__init__() # 继承父类的构造函数
        self.linear = nn.Linear(n_feature, 1) # nn.Linear是线性回归模型的核心，输入特征数和输出特征数
        # nn.Linear会自动初始化权重和偏置，权重是一个n_feature*1的矩阵，偏置是一个1*1的矩阵     
    def forward(self, x): # forward 定义前向传播
        y = self.linear(x) 
        return y

net = LinearNet(num_inputs)

"""
# 还可以使用nn.Sequential方式定义net
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......
"""
# 定义损失函数
loss = nn.MSELoss()
# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3

# 记录每个step的损失
steps = []
losses = []
step_count = 0

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
        
        # 记录每个step的损失
        step_count += 1
        steps.append(step_count)
        losses.append(l.item())
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 获取学习到的参数
w = net.linear.weight.data.numpy().reshape(-1)  # 转换为一维numpy数组
b = net.linear.bias.data.item()

# 调用可视化函数
plot_training_progress(
    steps, 
    losses,
    (true_w[0], true_w[1], true_b),
    (w[0], w[1], b)
)

# 添加回归结果的3D可视化
plot_regression_result(
    features,
    labels,
    true_w,
    true_b,
    w,
    b
)


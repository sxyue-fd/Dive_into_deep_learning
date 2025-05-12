import torch
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import random
from visualization_utils import plot_training_progress, plot_regression_result

# 生成数据集
# 样本数1000，特征数2，y=wX+b+噪声，噪声服从均值为0，标准差为0.01的正态分布
# Data size: y(1000,), X(1000,2), w(2,), b(1,)
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)# features=X，大小为(1000，2)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b # labels=y=w1*x1+w2*x2+b，大小为(1000,)，最后一个加法用了广播机制
labels += torch.tensor(np.random.normal(0, .1, size=labels.size()), dtype=torch.float32) # 添加噪声

'''查看张量维度
print('Features shape:', features.shape)  # Should be (1000, 2)
print('Labels shape:', labels.shape)      # Should be (1000,)
print('True w shape:', len(true_w))       # Should be 2
print('True b shape:', type(true_b))      # Should be float
''' 

'''数据集可视化
# 两个feature分别符合（0，1）高斯分布，一起组成二维高斯分布
plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), 10)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()
# Label对其中一个feature符合（带噪声的）线性关系
plt.scatter(features[:, 1].numpy(), labels.numpy(), 10)
plt.xlabel('Second feature')
plt.ylabel('Label')
plt.show()
'''

# 读取小批量数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 样本数
    indices = list(range(num_examples))  # 生成样本索引列表
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):  # 把数据集切割成batch，每次取batch_size个样本
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)  # 返回一个生成器，每次生成一组小批量样本和标签
'''
# 测试data_iter函数
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break  # 只打印第一组小批量样本
'''
# 定义线性回归模型  
def linreg(X, w, b):
    return torch.mm(X, w) + b  # mm是矩阵乘法
# 定义损失函数
def squared_loss(y_hat, y): # y_hat是模型预测值，y是真实值
    return (y_hat - y.view(y_hat.size())) ** 2 / 2  # 平方损失函数，Pytorch的MSELoss没有除以2
# 定义优化算法
def sgd(params, lr, batch_size):  # params是模型参数，lr是学习率，batch_size是小批量样本数
    for param in params:
        param.data -= lr * param.grad / batch_size  # 更新参数
        param.grad.data.zero_()  # 清空梯度

# 训练模型
# 超参数设置
lr = 0.03  # 学习率
num_epochs = 3  # 每个epoch遍历一次整个数据集
batch_size = 10 

# 调用前面定义的模型和损失函数
net = linreg
loss = squared_loss

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

# 记录每个step的损失
steps = []
losses = []
step_count = 0

# 开始训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b)
        l = loss(y_hat, y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        
        # 记录每个step的损失
        step_count += 1
        steps.append(step_count)
        losses.append(l.item())

# 获取学习到的参数
learned_w = w.reshape(-1).detach().numpy()
learned_b = b.item()


# 调用可视化函数
plot_training_progress(
    steps, 
    losses,
    (true_w[0], true_w[1], true_b),
    (learned_w[0], learned_w[1], learned_b)
)

# 添加回归结果的3D可视化
plot_regression_result(
    features,
    labels,
    true_w,
    true_b,
    learned_w,
    learned_b
)
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# 下载数据集并转换为张量
mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# Fashion-MNIST中的标签
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""
# 显示一些样本图片
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    X, y = mnist_train[i]
    plt.imshow(X.squeeze(), cmap='viridis')  # squeeze去掉通道维度，用灰度图显示
    plt.title(f'{labels[y]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
"""
# 读取小批量数据
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
# 返回的train_iter和test_iter的形状为[batch_size,1,28,28]，这里第二个维度代表1个灰度通道，如果是RGB数据集则设为3

# 定义Softmax模型
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        # 初始化模型参数
        init.normal_(self.linear.weight, mean=0, std=0.01)
        init.constant_(self.linear.bias, val=0)

    def forward(self, x):
        # 输入x形状: (batch_size, 1, 28, 28)
        # 将输入数据展平成(batch_size, 784)
        x = x.view(-1, num_inputs)
        y = self.linear(x)
        return y

# 模型参数设置
num_inputs = 784   # 28x28=784，将图像展平
num_outputs = 10   # 10个类别

net = LinearNet(num_inputs, num_outputs)

# 定义损失函数（交叉熵损失）
loss = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练和评估函数
def evaluate_accuracy(net, data_iter):
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer):
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 设置实时绘图
    plt.ion()
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        net.train()
        
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
              # 记录每个batch的loss并更新图像
            if i % 10 == 0:  # 每10个batch记录一次
                # 只记录当前batch的loss，不使用累积平均
                current_loss = l.item()
                train_losses.append(current_loss)
                  # 更新loss曲线
                ax1.clear()
                ax1.semilogy(train_losses)
                ax1.set_xlabel('Batch Steps')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                ax1.grid(True, which='both')  # 同时显示主要和次要网格
                
                # 设置对数刻度的刻度线
                ax1.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))  # 主刻度
                ax1.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))  # 次刻度
                ax1.yaxis.set_minor_formatter(plt.NullFormatter())  # 不显示次刻度的数值
                
                # 强制更新图像
                plt.pause(0.1)
        
        train_acc = train_acc_sum / n
        test_acc = evaluate_accuracy(net, test_iter)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 更新准确率曲线
        ax2.clear()
        epochs = range(1, epoch + 2)
        ax2.plot(epochs, train_accs, label='Train Acc')
        ax2.plot(epochs, test_accs, label='Test Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)
        
        print(f'epoch {epoch + 1}, loss {train_l_sum / n:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    
    return train_losses, train_accs, test_accs

# 训练模型
num_epochs = 5
train_losses, train_accs, test_accs = train(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer)

# 关闭交互模式
plt.ioff()
plt.show()

# 从测试集中随机选择5个样本进行可视化
net.eval()
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=5, shuffle=True)
X, y = next(iter(test_iter))
with torch.no_grad():
    pred = net(X).argmax(dim=1)

plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i].squeeze(), cmap='viridis')
    plt.title(f'True: {labels[y[i]]}\nPred: {labels[pred[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()


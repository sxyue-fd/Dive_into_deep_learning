"""
这是一个PyTorch基础操作的演示代码。主要包含以下几个部分:
1. Tensor的创建
- 演示了使用empty(), rand(), zeros(), tensor()等方法创建张量
2. Tensor的基本操作
- 展示了张量的加法、索引、reshape(view)等操作
- 说明了view操作时的内存共享机制
- 演示了使用clone()来创建独立副本
3. Tensor与NumPy的互操作
- 演示了torch.Tensor与numpy.ndarray之间的相互转换
- 说明了内存共享机制及如何避免
4. GPU操作
- 检查CUDA是否可用
- 演示了将tensor转移到GPU/CPU的方法
- 展示了直接在指定设备上创建tensor
5. 自动求导机制
- 展示了requires_grad=True的使用
- 演示了反向传播的计算过程
- 说明了梯度累加机制及如何清零
- 展示了张量对张量求导的方法
6. 张量修改与梯度计算
- 演示了如何在不影响梯度计算的情况下修改tensor值
- 包括使用.data属性和torch.no_grad()上下文管理器
这些示例代码展示了PyTorch中最基本和最常用的操作，适合作为PyTorch入门学习材料。
"""
import torch
import numpy as np
# Tensor创建
x = torch.empty(2,3)
x = torch.rand(2,3)
x = torch.zeros(2,3)
x = torch.tensor([[1,2,3],[4,5,6]])

# Tensor操作
y = torch.ones(2,3)
result = torch.empty(2, 3)
torch.add(x, y, out=result) #等价于result=x+y
y = x[0,:]
y += 1 # 源tensor也被改了

y = x.view(3,2)
z = x.view(-1, 2)  # -1所指的维度可以根据其他维度的值推出来，这里是6/2=3
# x,y,z都是共享内存的，改变一个会影响其他两个。
x_copy = x.clone().view(3,2) # 使用clone()可以避免这种情况

a = torch.ones(5)
b = a.numpy() # Tensor和numpy数组互相转换很方便，但a和b是共享内存的，改变一个另一个也跟着变
c = np.ones(5)
d = torch.from_numpy(c) # 同样，c和d是共享内存的，改变一个另一个也跟着变
e = torch.tensor(c) # 这样创建的tensor和numpy数组不共享内存，改变一个不会影响另一个

if torch.cuda.is_available():
    device = torch.device("cuda") # GPU
    x = x.to(device) # 将tensor转移到GPU上
    x1 = torch.ones_like(x, device=device) # 直接在GPU上创建Tensor
    x2 = torch.ones_like(x, device="cpu") # 在CPU上创建Tensor

x = torch.ones(2,2,requires_grad=True) # 允许梯度操作
y = x+2 # y是由x通过加法操作得到的，所以它的grad_fn是AddBackward0
z = y*y*3 # z是由y通过乘法操作得到的，所以它的grad_fn是MulBackward0
out = z.mean() # out是由z通过平均操作得到的，所以它的grad_fn是MeanBackward0

a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
a.requires_grad_(True) # 允许梯度操作，使用in-place操作
b = (a * a).sum()

out.backward() # 计算反向传播的梯度，因为out是标量，所以不需要指定梯度参数
x.grad # out=1/4*sum(3*(x+2)^2), out是标量，x是(2,2)张量，所以输出(2,2)张量,每个元素都是4.5

# 反向传播是累加的(accumulated)，所以如果多次调用backward()，需要先清空梯度
out2 = x.sum()
out2.backward()
x.grad # out2的梯度是1，所以x.grad是1+4.5=5.5
out3 = x.sum()
x.grad.data.zero_() # 清空梯度
out3.backward() 
x.grad # out3的梯度是1，所以x.grad是1

# 如果张量对张量求导，则需要指定grad_output参数
# 例如，y和w是同形张量，则y.backward(w)的意义是先计算标量torch.sum(y*w)，再用该标量求导
x = torch.tensor([[1,2],[3,4]], dtype=torch.float, requires_grad=True)
y = x**2
y.backward(torch.ones(2,2)) # y不是标量，需要传入一个同形的张量作为参数
x.grad # 结果为[2., 4.],[6., 8.]]

# 如果要修改tensor的数值但不影响梯度计算，可以用tensor.data操作
x = torch.ones(2, 2, requires_grad=True)
print(x.data,x.data.requires_grad) # x.data是一个新的tensor，不需要计算梯度
# 也可以使用with torch.no_grad()上下文管理器来禁止梯度计算
y = torch.sum(x**2)
x.data *= 10 # 只改变数值不影响梯度计算
y.backward()
print(x,x.grad)

from pyexpat import model
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from  pytorch_test7_model import Auto_Encoder
import os

# 定义超参数
learning_rate = 0.0003
batch_size = 64
epochsize = 30
root = 'mnist_data'
sample_dir = "image"

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 图像相关处理操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 训练集下载

# --collate_fn：如何取样本的，我们可以定义自己的函数来准确的实现想要的功能

# --drop_last：告诉如何处理数据集长度除以batch_size余下的数据。True就抛弃，否则保留

# --pin_memory：锁页内存

# MNIST:手写数字识别数据集
mnist_train = datasets.MNIST(root=root, train=True, transform=transform, download=False)

# Dataloader支持两种数据集类型，一种是上面讲述的 Map-style datasets（索引到样本的映射式的数据集），另一种为iterable-style datasets（迭代式的数据集）。
# batch_size: 设置批训练的大小
# --shuffle：设置为True的时候，每个epoch都会打乱数据集

mnist_train = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)


# 测试集下载
#transform就是对图像数据进行处理 传入dataloader的transform类型
# train： 设置是读取train数据集 还是test test.pt
# download (bool, optional): If true, downloads the dataset from the internet and
# puts it in root directory. If dataset is already downloaded, it is not
# downloaded again.
# transform (callable, optional): A function/transform that takes in an PIL image
# and returns a transformed version. E.g, transforms.RandomCrop
# target_transform (callable, optional): A function/transform that takes in the
# target and transforms it.

mnist_test = datasets.MNIST(root=root, train=False, transform=transform, download=False)
mnist_test = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True)

# image,_ = iter(mnist_test).next()
# print("image.shape:",image.shape)   # torch.Size([64, 1, 28, 28])


#从函数接收的参数state_dict中将参数和缓冲拷贝到当前这个模块及其子模块中.
# 如果函数接受的参数strict是True,那么state_dict的关键字必须确切地严格地和
# 该模块的state_dict()函数返回的关键字相匹配.
#作用：
# 使用 state_dict 反序列化模型参数字典。用来加载模型参数。将 state_dict 中的 parameters 和 buffers 复制到此 module 及其子节点中。
# 概况：给模型对象加载训练好的模型参数，即加载模型参数
#torch.load():加载模型
AE = Auto_Encoder()
AE.load_state_dict(torch.load('AE.ckpt'))


# loss(xi,yi)=(xi-yi)^2
# 这里 loss, x, y 的维度是一样的，可以是向量或者矩阵，i 是下标。
# 很多的 loss 函数都有 size_average 和 reduce 两个布尔类型的参数。因为一般损失函数都是直接计算 batch 的数据，因此返回的 loss 结果都是维度为 (batch_size, ) 的向量。
# (1)如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss
# (2)如果 reduce = True，那么 loss 返回的是标量

criteon = nn.MSELoss()
# Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，
# 它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，
# 每一次迭代学习率都有个确定范围，使得参数比较平稳。
optimizer = optim.Adam(AE.parameters(), lr=learning_rate)

print("start train...")
for epoch in range(epochsize):

    # 训练网络
    for batchidx, (realimage, _) in enumerate(mnist_train):

        # 生成假图像
        fakeimage = AE(realimage)

        # 计算损失
        loss = criteon(fakeimage, realimage)

        # 更新参数
        # 将模型的参数梯度初始化为0
        # 源代码
        # def zero_grad(self):
            # r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
            # for group in self.param_groups:
            #     for p in group['params']: # optimizer.zero_grad()函数会遍历模型的所有参数
            #         if p.grad is not None:
            #             p.grad.detach_() # 通过p.grad.detach_()方法截断反向传播的梯度流
            #             p.grad.zero_() # 再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空
        optimizer.zero_grad()
        # 反向传播计算梯度
        # PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
        # 如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
        # 更具体地说，损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。
        # 如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。
        loss.backward()
        
        # 更新所有参数 
        # step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。
        optimizer.step()


        # 根据pytorch中backward（）函数的计算，当网络参量进行反馈时，梯度是累积计算而不是被替换，
        # 但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，因此需要对每个batch调用一遍zero_grad（）将参数梯度置0.
        # 另外，如果不是处理每个batch清除一次梯度，而是两次或多次再清除一次，相当于提高了batch_size，
        # 对硬件要求更高，更适用于需要更高batch_size的情况。

        if batchidx%300 == 0:
            print("epoch:{}/{}, batchidx:{}/{}, loss:{}".format(epoch, epochsize, batchidx, len(mnist_train), loss))

    # 生成图像
    realimage,_ = iter(mnist_test).next()
    fakeimage = AE(realimage)

    # 真假图像合并成一张
    # torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
    image = torch.cat([realimage, fakeimage], dim=0)

    # 保存图像
    save_image(image, os.path.join(sample_dir, 'image-{}.png'.format(epoch + 1)), nrow=8, normalize=True)

    torch.save(AE.state_dict(), 'AE.ckpt')


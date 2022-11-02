import torch
import torchvision
import torch.nn as nn

class Auto_Encoder(nn.Module):

    def __init__(self):

        super(Auto_Encoder, self).__init__()
        # nn.Sequential():一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数。

        # ReLU(): 激活函数；f(x)=max{0, z}

        # 定义编码器结构
        self.Encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # sigmoid(z)=1/(1+e^(-z))
        # 定义解码器结构
        self.Decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    # view():把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），
    # 然后按照参数组合成其他维度的tensor。比如说是不管你原先的数据是[[[1,2,3],[4,5,6]]]还是[1,2,3,4,5,6]，
    # 因为它们排成一维向量都是6个元素，所以只要view后面的参数一致，得到的结果都是一样的。
    # .size(0)：就是相当于返回代表size的整数
    def forward(self, input):

        code = input.view(input.size(0), -1)
        code = self.Encoder(code)

        output = self.Decoder(code)
        output = output.view(input.size(0), 1, 28, 28)

        return output

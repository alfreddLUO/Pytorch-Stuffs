import torch
import torchvision
import torch.nn as nn

class Auto_Encoder(nn.Module):

    def __init__(self):

        super(Auto_Encoder, self).__init__()
        # nn.Sequential():һ�������������������ģ�齫�����ڴ��빹������˳�����α���ӵ�����ͼ��ִ�У�
        # ͬʱ��������ģ��ΪԪ�ص������ֵ�Ҳ������Ϊ���������

        # ReLU(): �������f(x)=max{0, z}

        # ����������ṹ
        self.Encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # sigmoid(z)=1/(1+e^(-z))
        # ����������ṹ
        self.Decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    # view():��ԭ��tensor�е����ݰ��������ȵ�˳���ų�һ��һά�����ݣ�����Ӧ������ΪҪ���ַ�������洢�ģ���
    # Ȼ���ղ�����ϳ�����ά�ȵ�tensor������˵�ǲ�����ԭ�ȵ�������[[[1,2,3],[4,5,6]]]����[1,2,3,4,5,6]��
    # ��Ϊ�����ų�һά��������6��Ԫ�أ�����ֻҪview����Ĳ���һ�£��õ��Ľ������һ���ġ�
    # .size(0)�������൱�ڷ��ش���size������
    def forward(self, input):

        code = input.view(input.size(0), -1)
        code = self.Encoder(code)

        output = self.Decoder(code)
        output = output.view(input.size(0), 1, 28, 28)

        return output

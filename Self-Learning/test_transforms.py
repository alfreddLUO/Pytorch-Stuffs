from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# transforms相当于一个工具箱， 里面包含一些工具。 e.g. ToTensor, resize
# python的用法 =》 tensor 数据类型
# 通过 transforms.ToTensor去看两个问题
# 1. transforms被使用
# 2. 为什么我们需要Tensor这个数据类型

Image_path = 'sample_dataset/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(Image_path)
print(img)
writer = SummaryWriter("logs")

# 创建具体的工具实例
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

print(tensor_img)
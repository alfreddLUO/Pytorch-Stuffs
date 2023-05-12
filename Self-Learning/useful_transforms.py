from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter



Image_path = 'sample_dataset/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(Image_path)
print(img)
writer = SummaryWriter("logs")

# 创建具体的工具实例
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

print(tensor_img)
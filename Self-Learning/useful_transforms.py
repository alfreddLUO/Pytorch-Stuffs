from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter



Image_path = 'sample_dataset/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(Image_path)
print(img)
writer = SummaryWriter("logs")

# ToTensor
# 创建具体的工具实例
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize_img", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
print(img.size)
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
print(img_resize_2)

# RandomCrop随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

# 技巧
# 关注输入和输出
# 多看官方文档


writer.close()
print(tensor_img)
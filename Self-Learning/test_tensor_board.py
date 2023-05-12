from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
Image_path = 'sample_dataset/hymenoptera_data/train/ants/0013035.jpg'
img_PIL = Image.open(Image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats="HWC")
for i in range(100):
# writer.add_image()
    writer.add_scalar("y=2x", 4*i, i)

writer.close()
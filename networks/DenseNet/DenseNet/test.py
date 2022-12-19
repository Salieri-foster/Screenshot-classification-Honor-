import cv2
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import numpy as np

image = cv2.imread('pic/test.jpg')
image=cv2.resize(image,(416,416))
image=np.expand_dims(image, 0)
image=torch.tensor(image)
image = image.permute(0,3,1,2).contiguous()
 



print(image.shape)
# image=torch.tensor(image).unsqueeze(0)
# print(image.shape)

# img1 = F.to_tensor(image)
# print(type(img1),img1.shape)
# print(img1)
# #
from PIL import Image

features=image
out=F.avg_pool2d(features,7, stride=1).view(features.size(0),-1)
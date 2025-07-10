import torch
import time
import cv2 as cv
import torchvision.transforms as transforms

import os
import time
import cv2
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable

np.seterr(divide='ignore', invalid='ignore')
# np.set_printoptions(threshold='nan')
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from cal_iou import evaluate


from models.SFFNetv2 import SFFNet
from thop import profile

from compare_methods.CMTFNet import CMTFNet
from compare_methods.MANet import MANet
from compare_methods.segnet import SegNet
from compare_methods.fcn import FCN8s
from compare_methods.UNet import UNet
from compare_methods.deeplabv3_plus import DeepLab
from compare_methods.A2FPN import A2FPN
from compare_methods.pspnet import PSPNet
from compare_methods.unetformer import UNetFormer
from compare_methods.FuzzyNet import FuzzyNet
from compare_methods.BANet import BANet
from compare_methods.qtt_fuzzy import MSResNet


#net = LinkNet(n_classes=6)
#net = FuzzyNet(n_class=6)
#net = FCN8s(num_classes=6)
#net = SegNet(input_nbr=3,label_nbr=6)
#net = UNet(n_channels=3, n_classes=6)
#net = SFFNet(num_classes=6)
#net = CMTFNet(num_classes=6)
#net = MANet(num_classes=6)
#net = DeepLab(num_classes=6)
#net = A2FPN(class_num=6)
#net = PSPNet(num_classes=6,downsample_factor=8)
#net = UNetFormer(num_classes=6)
#net = MSResNet(num_classes=6)
net = BANet(num_classes=6, weight_path=None)

#start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

path = '/home/ps/data/code/wgy/1Awork_02/Datasets/A_GID/JPEGImages/01_1_1.png'
#path = '/home/ps/data/code/wgy/1Awork_02/Datasets/Potsdam512/train/JPEGImages/top_potsdam_2_10_RGB0.png'
#path = '/home/ps/data/code/wgy/1Awork_02/Datasets/A_Vaihingen_800/test/top_mosaic_09cm_area33__1__0___0.png'


img = cv.imread(path)
transf = transforms.ToTensor()
input_data = transf(img)
input_data = input_data.unsqueeze(0)
print(type(input_data))
print(input_data.shape)
#input_data1 = torch.randn(1, 3, 512, 512)  # 根据你的模型和输入数据进行相应调整
#print(type(input_data1))
input_data = input_data.to('cuda:0')
# 模型推断
net.eval()
start_time = time.time()
with torch.no_grad():
    for i in range(100):
        output = net(input_data)
    end_time = time.time()

# 计算运行时间和FPS
inference_time = end_time - start_time
fps =  1000 / inference_time / 1.56
one_time = inference_time / 1000
print(f"Inference time: {inference_time} seconds")
print(f"FPS: {fps}")
print(f'inference one time: {one_time}')

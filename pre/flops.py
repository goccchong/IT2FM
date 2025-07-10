from __future__ import division
import torch
import re
import time
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
import os
from dataloaders import custom_transforms as tr
from dataloaders import pascal
from torchvision.utils import make_grid
import utils_1 as utils
from tensorboardX import SummaryWriter

#from models.SFFNetv2 import SFFNet

from compare_methods.CMTFNet import CMTFNet
from compare_methods.MANet import MANet
from compare_methods.segnet import SegNet
from compare_methods.fcn import FCN8s
from compare_methods.UNet import UNet
from compare_methods.deeplabv3_plus import DeepLab
from compare_methods.pspnet import PSPNet
from compare_methods.A2FPN import A2FPN
from compare_methods.unetformer import UNetFormer
from compare_methods.FuzzyNet_he import FuzzyNet
from compare_methods.BANet import BANet
from compare_methods.A2FPN import A2FPN
from compare_methods.qtt_fuzzy import MSResNet
from models.SFFNetv2_wave4 import SFFNet
from thop import profile
from compare_methods.MIFNet import MIFNet


if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    x = x.to('cuda:1')
    
    #frame_work = CMTFNet()
    #frame_work = CMTFNet(num_classes=6)
    #frame_work = UNetFormer(num_classes=8)
    #frame_work = SegNet(input_nbr=3,label_nbr=6)
    #frame_work = UNet(n_channels=3, n_classes=6)
    #frame_work = MANet(num_classes=6)
    #frame_work = FCN8s(num_classes=6) # ok
    #frame_work = MANet(num_channels=3, num_classes=6)
    #frame_work = DeepLab(num_classes=6).to('cuda:0')
    #frame_work = PSPNet().to('cuda:0')
    #frame_work = UNetFormer(num_classes=6)
    #frame_work = FuzzyNet().to('cuda:0')
    #frame_work = PoolFormer(num_classes=6)
    #frame_work = BANet(num_classes=6, weight_path=None)
    #frame_work = A2FPN(class_num=6)
    frame_work = SFFNet().to('cuda:1')
    #frame_work = MSResNet().to('cuda:0')
    #frame_work = MIFNet(in_channels=3, num_classes=6).to('cuda:1')
    #frame_work = MIFNet()

    out = frame_work(x)


    print(out.shape)
    flops, params = profile(frame_work, (x,))
    print('flops: ', flops/1024/1024/1024, 'params: ', params/1024/1024)


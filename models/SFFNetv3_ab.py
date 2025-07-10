import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

import timm

from .MDAF import MDAF
from .FMS import FMS

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res, ade):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2] * ade
        x = self.post_conv(x)
        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat

class FuzzyLayer(nn.Module):
    def __init__(self, fuzzynum, channel, sigma=2):
        super(FuzzyLayer, self).__init__()
        self.fuzzynum = fuzzynum
        self.channel = channel
        self.sigma = sigma
        self.conv1 = nn.Conv2d(self.channel, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, self.channel, 3, padding=1)
        # 添加隶属度上下界参数，初始化为高斯型隶属度函数
        #self.center = nn.Parameter(torch.randn(self.channel, self.fuzzynum))
        

        init_mean = 0.2  # 你希望的均值
        self.center = nn.Parameter(torch.full((self.channel, self.fuzzynum), init_mean))  
        
        
        self.lower_bound = nn.Parameter(self.center - self.sigma)
        self.upper_bound = nn.Parameter(self.center + self.sigma)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(self.channel)

    def fuzzify(self, x):
        fuzzified = torch.zeros_like(x)
        for num, channel, w, h in itertools.product(range(x.size(0)), range(x.size(1)), range(x.size(2)),
                                                    range(x.size(3))):
            for f in range(self.fuzzynum):
                # 计算距离
                lower_diff = x[num][channel][w][h] - self.lower_bound[channel][f]
                upper_diff = self.upper_bound[channel][f] - x[num][channel][w][h]
                lower_diff_num = lower_diff.item()
                upper_diff_num = upper_diff.item()
                comfort_range = upper_diff_num - lower_diff_num

                if lower_diff_num > 0 and upper_diff_num > 0:  # within
                    fuzzified[num][channel][w][h] += torch.exp(-0.5 * (lower_diff ** 2)) + torch.exp(
                        -0.5 * (upper_diff ** 2))
                elif lower_diff_num < 0:  # lower
                    if abs(lower_diff_num) > abs(comfort_range):  # too far
                        x[num][channel][w][h] = self.lower_bound[channel][f]
                        fuzzified[num][channel][w][h] += torch.exp(
                            -0.5 * ((x[num][channel][w][h] - self.lower_bound[channel][f]) ** 2)) + torch.exp(
                            -0.5 * ((self.upper_bound[channel][f] - x[num][channel][w][h]) ** 2))
                    else:
                        x[num][channel][w][h] = self.lower_bound[channel][f] - lower_diff
                        fuzzified[num][channel][w][h] += torch.exp(
                            -0.5 * ((x[num][channel][w][h] - self.lower_bound[channel][f]) ** 2)) + torch.exp(
                            -0.5 * ((self.upper_bound[channel][f] - x[num][channel][w][h]) ** 2))
                elif upper_diff_num < 0:  # higher
                    if abs(upper_diff_num) > abs(comfort_range):  # too far
                        x[num][channel][w][h] = self.upper_diff[channel][f]
                        fuzzified[num][channel][w][h] += torch.exp(
                            -0.5 * ((x[num][channel][w][h] - self.lower_bound[channel][f]) ** 2)) + torch.exp(
                            -0.5 * ((self.upper_bound[channel][f] - x[num][channel][w][h]) ** 2))
                    else:
                        x[num][channel][w][h] = self.lower_bound[channel][f] - lower_diff
                        fuzzified[num][channel][w][h] += torch.exp(
                            -0.5 * ((x[num][channel][w][h] - self.lower_bound[channel][f]) ** 2)) + torch.exp(
                            -0.5 * ((self.upper_bound[channel][f] - x[num][channel][w][h]) ** 2))
            
        return fuzzified
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        fuzzified = self.fuzzify(x)
        return self.bn2(self.conv2(fuzzified))


class SFFNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 # backbone_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 use_aux_loss=True
                 ):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained,
                                          output_stride=32, out_indices=(0, 1, 2, 3))

        self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

        self.MDAF_L = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.MDAF_H = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.fuseFeature = FMS(in_ch=3 * decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        self.fuseFeature2 = FMS(in_ch=decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        self.WF1 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.WF2 = WF(in_channels=decode_channels, decode_channels=decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.down = Conv(in_channels=3 * decode_channels, out_channels=decode_channels, kernel_size=1)
        self.down1 = Conv(in_channels=3 * decode_channels, out_channels=8, kernel_size=1)        
        self.up1 = Conv(in_channels=8, out_channels=decode_channels, kernel_size=1)

        # new
        self.fuzzylayer = FuzzyLayer(fuzzynum=1, channel=8)
        self.conv1 = nn.Conv2d(96, 96, kernel_size=3, padding=1)


    def forward(self, x, imagename=None):
        b = x.size()[0]
        h, w = x.size()[-2:]

        res1, res2, res3, res4 = self.backbone(x)
        #print('res4', res4.size())
        
        res1h, res1w = res1.size()[-2:]

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        middleres = torch.cat([res2, res3, res4], dim=1)

        # multilevel wavelet
        # 1
        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)
        fusefeature_L_copy, fusefeature_H_copy = fusefeature_L, fusefeature_H
        # 2
        #fusefeature_L_2, fusefeature_H_2, glb_2, local_2 = self.fuseFeature2(fusefeature_L, imagename)
        #H_h, H_w = fusefeature_H.size()[-2:]
        #fusefeature_H_2 = F.interpolate(fusefeature_H_2, size=(H_h, H_w), mode='bicubic', align_corners=False)
        #fusefeature_L_2 = F.interpolate(fusefeature_L_2, size=(H_h, H_w), mode='bicubic', align_corners=False)
        #fusefeature_H_2 = self.conv1(fusefeature_H_2)
        #fusefeature_L_2 = self.conv1(fusefeature_L_2)
        # 3
        #fusefeature_L_3, fusefeature_H_3, glb_3, local_3 = self.fuseFeature2(fusefeature_L_2, imagename)
        #fusefeature_H_3 = F.interpolate(fusefeature_H_3, size=(H_h, H_w), mode='bicubic', align_corners=False)
        #fusefeature_L_3 = F.interpolate(fusefeature_L_3, size=(H_h, H_w), mode='bicubic', align_corners=False)
        #fusefeature_H_3 = self.conv1(fusefeature_H_3)
        #fusefeature_L_3 = self.conv1(fusefeature_L_3)
        # fusion
        #fusefeature_H = fusefeature_H + fusefeature_H_2 + fusefeature_H_3
        #fusefeature_L = fusefeature_L + fusefeature_L_2 + fusefeature_L_3
        
        #fusefeature_H = torch.cat([fusefeature_H, fusefeature_H_2, fusefeature_H_3], dim=1)
        #fusefeature_H = 0.3 * self.down(fusefeature_H) + fusefeature_H_copy
        #fusefeature_L = torch.cat([fusefeature_L, fusefeature_L_2, fusefeature_L_3], dim=1)
        #fusefeature_L = 0.3 * self.down(fusefeature_L) + fusefeature_L_copy

        glb = self.MDAF_L(fusefeature_L, glb)
        local = self.MDAF_H(fusefeature_H, local)

        res = self.WF1(glb, local)

class IT2F2CM(nn.Module):
    def __init__(self, channels, fuzzynum=3, sigma=0.5):
        super().__init__()
        self.channels = channels
        self.fuzzynum = fuzzynum
        self.sigma = nn.Parameter(torch.tensor(sigma))
        
        self.center = nn.Parameter(torch.rand(channels, fuzzynum))
        self.rho = nn.Parameter(torch.ones(channels, fuzzynum) * 0.5)
        
        self.conv_pre = nn.Conv2d(channels, channels, 1)
        self.conv_post = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    
    def gaussian_membership(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma)**2)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_pre(x)
        
        mu_lower = self.center - self.rho * self.sigma
        mu_upper = self.center + self.rho * self.sigma
        
        membership = torch.zeros(b, c, h, w, self.fuzzynum).to(x.device)
        
        for i in range(self.fuzzynum):
            lower = self.gaussian_membership(x, mu_lower[:, i].view(1, c, 1, 1), self.sigma)
            upper = self.gaussian_membership(x, mu_upper[:, i].view(1, c, 1, 1), self.sigma)
            membership[:, :, :, :, i] = (lower + upper) / 2
        
        membership = membership.mean(dim=-1)
        x = x * membership
        
        return self.conv_post(x)

class WTF2CM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.haar_filters = nn.Parameter(torch.tensor([
            [1, 1], 
            [1, -1]
        ]).float().view(1, 1, 2, 2), requires_grad=False)
        
        self.conv_low = nn.Conv2d(channels//4, channels//4, 3, padding=1)
        self.conv_high = nn.Conv2d(channels*3//4, channels//2, 3, padding=1)
    
    def haar_transform(self, x):
        A = F.conv2d(x, self.haar_filters, stride=2, padding=0)
        A = F.conv2d(A, self.haar_filters.transpose(2,3), stride=1, padding=1)
        
        LL = F.conv2d(A, self.haar_filters, stride=2, padding=0)
        LH = F.conv2d(A, self.haar_filters, stride=2, padding=0)
        HL = F.conv2d(A, self.haar_filters, stride=2, padding=0)
        HH = F.conv2d(A, self.haar_filters, stride=2, padding=0)
        
        return LL, torch.cat([LH, HL, HH], dim=1)
    
    def forward(self, x):
        F_L, F_H = self.haar_transform(x)
        
        F_L = self.conv_low(F_L)
        F_H = self.conv_high(F_H)
        
        return F_L, F_H

class CMFA(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(2, 0, 1)  # [H*W, B, C]
        
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out)
        
        return attn_out.permute(1, 2, 0).view(b, c, h, w)

class S_FCFNet(nn.Module):
    def __init__(self, decode_channels=96, num_classes=6, backbone_name="convnext_tiny"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, features_only=True, pretrained=True, 
            output_stride=32, out_indices=(0, 1, 2, 3)
        )
        
        self.adapters = nn.ModuleList([
            nn.Conv2d(192, decode_channels, 1),
            nn.Conv2d(384, decode_channels, 1),
            nn.Conv2d(768, decode_channels, 1)
        ])
        
        self.it2f2cm = IT2F2CM(decode_channels * 3)
        self.wtf2cm = WTF2CM(decode_channels * 3)
        

        self.cmfa = CMFA(decode_channels)
        self.cmfca = CMFA(decode_channels)
        

        self.decoder = nn.Sequential(
            nn.Conv2d(decode_channels * 3, decode_channels, 3, padding=1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decode_channels, num_classes, 1)
        )
    
    def forward(self, x):

        f1, f2, f3, f4 = self.backbone(x)
        
        # 特征适配
        f2 = F.interpolate(self.adapters[0](f2), size=f1.shape[2:], mode='bilinear')
        f3 = F.interpolate(self.adapters[1](f3), size=f1.shape[2:], mode='bilinear')
        f4 = F.interpolate(self.adapters[2](f4), size=f1.shape[2:], mode='bilinear')
        FI = torch.cat([f2, f3, f4], dim=1)  
        

        FS = self.it2f2cm(FI)
        
        FL, FH = self.wtf2cm(FI)
        
        FS = self.cmfa(FS)
        FL = self.cmfca(FL)
        FH = self.cmfca(FH)
        
        FO = torch.cat([FS, FL, FH], dim=1)
        

        out = self.decoder(FO)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear')

        middleres_fuzzy = middleres # 288,128,128
        middleres = self.down(middleres)  # 96,128,128
        
        #middleres_fuzzy = middleres
        middleres_fuzzy = F.interpolate(middleres_fuzzy, scale_factor=0.125) # 288,16,16
        middleres_fuzzy = self.down1(middleres_fuzzy) # 8,16,16
        middleres_fuzzy = self.fuzzylayer(middleres_fuzzy)
        middleres_fuzzy = F.interpolate(middleres_fuzzy, scale_factor=8) # 8,128,128
        middleres_fuzzy = self.up1(middleres_fuzzy)  # 96,128,128

        res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res = 0.5 * middleres_fuzzy + res
        res = middleres + res
        res = self.WF2(res, res1)
        res = self.segmentation_head(res)

        if self.training:
            if self.use_aux_loss == True:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x
            else:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x
        else:
            x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
            return x

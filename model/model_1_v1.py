#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Project_xf_follow_sfcn-opi_Pannuke 
@File    ：model_1_v1.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/16 16:02

这个model就是在sfcn-opi模型的基础上面加了文泰的细检测分支
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rev_blocks import RevGroupBlock

import albumentations as albu
def ConvBnAct(in_channel, out_channel, kernel_size, stride, pad, act=nn.Identity()): # 128, 2, 1, 1,0,act
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad),
        nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1),
        act
    )


class RevBlockC(nn.Module):  # Module 1 里面的Residual Block    RevBlockC不进行下采样
    def __init__(self, in_channel, out_channel, stride, act, **kwargs):
        super().__init__()
        assert in_channel == out_channel  # 输入通道应该等于输出通道
        assert stride == 1
        temp_channel = in_channel // 1  # 一个中间变量
        self.conv1 = ConvBnAct(in_channel, temp_channel, kernel_size=3, stride=1, pad=1, act=act)
        self.conv2 = ConvBn(temp_channel, out_channel, kernel_size=3, stride=1, pad=1)
        # 上面经过两个卷积层，图片的通道数和大小都不会变化。

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_shortcut = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = x_shortcut + y  # 残差链接
        # y = self.relu(y)
        y_final = F.relu(y,inplace=True)

        return y_final

class ResBlockA(nn.Module):  # 在 Module 2中第一个Residual Block需要进行下采样一倍(降低分辨率)
    def __init__(self, in_channel, out_channel, stride, act):
        super().__init__()

        temp_ch = out_channel // 1  # 这一步的目的就是为了看着好看，输出通道的位置
        self.conv1 = ConvBnAct(in_channel, temp_ch, kernel_size=3, stride=stride, pad=1, act=act)  # 这里使用卷积进行下采样
        self.conv2 = ConvBn(temp_ch, out_channel, kernel_size=3, stride=1, pad=1)

        self.conv3 = ConvBn(in_channel, out_channel, kernel_size=3, stride=stride, pad=1)     #  对x也进行卷积，为了下采样

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x_shortcut = self.conv3(x)
        y_final = y + x_shortcut  # 这里的y就是论文中的Fi，x_shortcut就是Ws
        y_final = F.relu(y_final,inplace=True)
        return y_final


def DeconvBnAct(in_channel, out_channel, kernel_size, stride, padding,output_padding, act=nn.Identity()):
    return nn.Sequential(   # (5, 5, 3, 2, 1, 1,act)
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding,output_padding),
        nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1),
        act
    )
def Deconv(in_channel, out_channel, kernel_size, stride, padding,output_padding, act=nn.Identity()):
    return nn.Sequential(   # (5, 5, 3, 2, 1, 1,act)
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding,output_padding),
    )

def PsConvBnAct(upscale, in_ch, out_ch, ker_sz, stride, pad, group=1, dilation=1):
    return nn.Sequential(
        nn.PixelShuffle(upscale),
        nn.Conv2d(in_ch//(upscale**2), out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))

def ConvBn(in_channel, out_channel, kernel_size, stride, pad, act=nn.Identity()):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad),
        nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1),
    )

class MainNet(nn.Module):
    model_id = 1

    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(0.02, inplace=False)  # 设置一个激活函数 0.02是一个超参数，inplace = True表示原地进行操作
        # act = nn.ReLU(inplace=True)  # 文泰这里是 nn.LeakyReLU(0.02, inplace=True)
        act_soft = nn.Softmax(dim=1)
# ==============================================   第一阶段 粗检测=======================================

        self._bm1 = nn.ModuleList()                                # 第一阶段，共享底层
        self.conv1 = ConvBnAct(3, 32, 3, 1, 1, act)  # 首先经过一个卷积层
        self.rvb1 = RevGroupBlock(32, 32, 1, act, RevBlockC, 9)  # 这里是Module1  RevBlockC不进行下采样

        self.rb2 = ResBlockA(32, 64, 2, act)                       # Module2 的开始部分，进行分辨率的降低，RevBlockA 进行下采样
        self.rvb2 = RevGroupBlock(64, 64, 1, act, RevBlockC, 8)    # Module2 的剩下8个块  RevBlockC不进行下采样

        self.rb3 = ResBlockA(64, 128, 2, act)                       # Module 3 的开始部分，进行分辨率的降低，RevBlockA 进行下采样
        self.rvb3 = RevGroupBlock(128, 128, 1, act, RevBlockC, 8)  # Module3 的剩下8个块  RevBlockC不进行下采样


        self.conv2 = nn.Conv2d(128, 2, 1, 1, 0)                       # 经过一个1x1的卷积，带有两个filter(就是变成两个通道),步长为1

        self.deconv1 = nn.ConvTranspose2d(2, 2, 3, 2, 1,1)            # 这里的参数有待商榷 输入通道2，输出通道2，卷积核2x2,步长为2，填充为0 ，最后得到的结果为32x32x2

        self.conv3 = nn.Conv2d(64, 2, 1, 1, 0)                        # module2 结束后的卷积
        self.deconv2 = nn.ConvTranspose2d(2, 2, 3, 2, 1,1)            # 检测分支的和进行逆卷积,经过softmax   # 这里的参数有待商榷2 2 2 2 0 act

        self._bm1.extend([self.conv1, self.rvb1, self.rb2, self.rvb2, self.rb3, self.rvb3,self.conv2,self.deconv1,self.conv3,self.deconv2])       # 没有共享层的核检测层


# =============================================  第二阶段，细检测 ==========================================================
        self._bm2 = nn.ModuleList()
        self.rvb4 = RevGroupBlock(128, 128, 1, act, RevBlockC, 9)     # Module4 的剩下8个块  RevBlockC不进行下采样
        self.b4_conv1 = nn.Sequential(
            ConvBnAct(128, 128, 1, 1, 0, act),
            PsConvBnAct(4, 128, 128, 5, 1, 2),
            nn.Conv2d(128, 1, 1, 1, 0, bias=True)
        )
        self._bm2.extend([self.rvb4, self.b4_conv1])
# =============================================  第三阶段，分类 ==========================================================
        self._bm3 = nn.ModuleList()
        self.rvb5 = RevGroupBlock(128, 128, 1, act, RevBlockC, 9)
        self.b5_conv1 = nn.Sequential(
            ConvBnAct(128, 128, 1, 1, 0, act),
            PsConvBnAct(4, 128, 128, 5, 1, 2),
            nn.Conv2d(128, 5, 1, 1, 0, bias=True)
        )

        self._bm3.extend([self.rvb5, self.b5_conv1])


        self.enabled_b2_branch = True                   # 细检测阶段
        self.enabled_b3_branch = True                   # 分类阶段
        self.is_freeze_seg1 = False                # 粗检测分支
        self.is_freeze_seg2 = False                # 细检测分支
        self.is_freeze_seg3 = False                # 分类分支

    def set_freeze_seg1(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg1 = b
        self._bm1.train(not b)
        for p in self._bm1.parameters():
            p.requires_grad = not b


    def set_freeze_seg2(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg2 = b
        self._bm2.train(not b)
        for p in self._bm2.parameters():
            p.requires_grad = not b


    def set_freeze_seg3(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg3 = b
        self._bm3.train(not b)
        for p in self._bm3.parameters():
            p.requires_grad = not b


    def seg1_state_dict(self):
        return dict(self._bm1.state_dict())

    def load_seg1_state_dict(self, *args, **kwargs):
        self._bm1.load_state_dict(*args, **kwargs)

    def seg2_state_dict(self):
        return dict(self._bm2.state_dict())

    def load_seg2_state_dict(self, *args, **kwargs):
        self._bm2.load_state_dict(*args, **kwargs)

    def seg3_state_dict(self):
        return dict(self._bm3.state_dict())

    def load_seg3_state_dict(self, *args, **kwargs):
        self._bm3.load_state_dict(*args, **kwargs)



    def forward(self, x):

# ========================= 第一阶段，粗检测层=======================
        y = self.conv1(x)
        y = self.rvb1(y)
        y = self.rb2(y)
        y = self.rvb2(y)        # modul2结束
        y_2 = y                 # y_2用于检测分支

        y = self.rb3(y)
        y = self.rvb3(y)        # module3结束
        y_3 = y                 # y_3用于分类分支

        y_m2 = self.conv3(y_2)
        y = self.conv2(y)
        y = self.deconv1(y)
        y_det_rough = self.deconv2(y + y_m2)


        y_det_fine = None
        y_cla = None
        if self.enabled_b2_branch:              # 开启细检测分支
            y_fine_1 = self.rvb4(y_3)
            y_det_fine = self.b4_conv1(y_fine_1)

        if self.enabled_b3_branch:              # 开启分类分支
            y_cla_1 = self.rvb5(y_3)
            y_cla = self.b5_conv1(y_cla_1)


        return y_det_rough, y_det_fine, y_cla


if __name__ == '__main__':


    a = torch.zeros(8, 3, 64, 64).cuda(0)
    # a = torch.zeros(2, 512, 512, 3).cuda(0).permute(0, 3, 1, 2)
    net = MainNet().cuda(0)
    net.enabled_b2_branch = True
    net.enabled_b3_branch = True
    # model_utils_torch.print_params_size(net)
    y_1,y_2,y_3 = net(a)
    print(y_1.shape)
    print(y_2.shape)
    print(y_3.shape)
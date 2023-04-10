import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class Encoder_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64):
        super(Encoder_Decoder, self).__init__()
        ngf = 64
        inconv_1 = nn.Conv2d(in_channels, ngf, kernel_size=7,
                               stride=1, padding=3, bias=False)
        inconv_2 = nn.Conv2d(ngf, ngf*2, kernel_size=4,
                               stride=2, padding=1, bias=False)
        relu = nn.LeakyReLU(0.2, True)
        outconv1 = nn.ConvTranspose2d(ngf*2, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
        outconv2 = nn.Conv2d(ngf*2, ngf, kernel_size=5,
                                stride=1, padding=2, bias=False)
        outconv3 = nn.Conv2d(ngf, out_channels, kernel_size=7,
                                stride=1, padding=3, bias=False)

        first = [inconv_1, relu]
        second = [inconv_2,relu]

        for i in range(16):
            second +=[base_block(ngf*2)]
        
        second += [outconv1, relu]
        
        third = [outconv2, relu, outconv3, nn.Tanh()]
        
        self.first = nn.Sequential(*first)
        self.second = nn.Sequential(*second)
        self.third = nn.Sequential(*third)

    def forward(self,x):
        x1 = self.first(x)
        x2 = self.second(x1)
        inter = torch.cat((x1,x2),1)
        out = self.third(inter)
        return torch.clamp(out+x, min=-1, max=1)
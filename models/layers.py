import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn as nn
from torch.nn  import functional as F


class DSC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSC, self).__init__()
        self.conv = nn.Sequential(
                                  #dw
                                  nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=in_channel, bias=False),
                                  #pw
                                  nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
                                  )
    def forward(self, x):
        out = self.conv(x)
        return out
    
class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        conv = nn.Conv2d(dim, dim, kernel_size=1)
        sigmoid = nn.Sigmoid()
        attn = [conv, sigmoid]
        self.attn = nn.Sequential(*attn)
        
    def forward(self, x):
        return self.attn(x)


class base_block(nn.Module):
    def __init__(self, in_channel):
        super(base_block, self).__init__()
#######----------- part of conv -----------#######
        self.attn_1 = attention(in_channel)
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(in_channel),
                                    nn.LeakyReLU(0.2, True))

        self.attn_2 = attention(in_channel)
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(in_channel),
                                    nn.LeakyReLU(0.2, True))

        self.attn_3 = attention(in_channel)
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(in_channel),
                                    nn.LeakyReLU(0.2, True))
        
        self.attn_4 = attention(in_channel)
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, padding_mode='reflect'),
                                    nn.InstanceNorm2d(in_channel))



#######----------- channel transformer  -----------#######
        self.c1 = nn.Conv2d(in_channel*2, in_channel, kernel_size=1)
        self.c2 = nn.Conv2d(in_channel*3, in_channel, kernel_size=1)
        self.c3 = nn.Conv2d(in_channel*4, in_channel, kernel_size=1)

#######----------- attention refiner  -----------#######
        self.conv_c1 = DSC(in_channel, in_channel)
        self.conv_c2 = DSC(in_channel, in_channel)
        self.conv_c3 = DSC(in_channel, in_channel)


    def forward(self, x):
#######----------- part 1 -----------#######
        attn_1 = self.attn_1(x)       #shallow attention_1
        x0 = torch.mul(x, attn_1)
        x1 = self.conv_1(x0)           #output of conv_1
        attn_2 = self.attn_2(x1) 
        x_attn_1_c = self.c1(torch.cat((attn_1, attn_2),1)) #final output of first conv
        x_attn_1 = self.conv_c1(x_attn_1_c)
        x_f_1 = torch.mul(x1, x_attn_1)

        x2 = self.conv_2(x_f_1)         # output of conv_2
        attn_3 = self.attn_3(x2)
        x_attn_2_c = self.c2(torch.cat((attn_1, attn_2, attn_3),1)) #final output of second conv
        x_attn_2 = self.conv_c2(x_attn_2_c)
        x_f_2 = torch.mul(x2, x_attn_2)
        
        x3 = self.conv_3(x_f_2)         # output of conv_3
        attn_4 = self.attn_4(x3)
        x_attn_3_c = self.c3(torch.cat((attn_1, attn_2, attn_3, attn_4),1)) #final output of third conv
        x_attn_3 = self.conv_c3(x_attn_3_c)
        x_f_3 = torch.mul(x3, x_attn_3)
        
        out = self.conv_4(x_f_3)      # final output of the fourth conv

        return F.relu(out+x)

## Modifed SAM
class SAM_m(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=True):
        super(SAM_m, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, 3, kernel_size=7, padding=3, bias=bias)
        self.conv3 = nn.Conv2d(3, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = torch.mul(x1, x2)
        x1 = x1 + x
        return x1, img
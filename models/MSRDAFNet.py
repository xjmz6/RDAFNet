import torch
import torch.nn as nn
from .layers import *

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

class Encoder_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64):
        super(Encoder_Decoder, self).__init__()
        #contianer of body block
        self.body_block_1 = nn.ModuleList()
        self.body_block_2 = nn.ModuleList()
        self.body_block_3 = nn.ModuleList()
        # Stage_1 Network
        self.stage_1_enc_1 = nn.Sequential(nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.stage_1_enc_2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, True))

        for i in range(10):
            self.body_block_1.append(base_block(ngf*2))

        self.stage_1_dec_1 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.stage_1_dec_2 = nn.Sequential(nn.Conv2d(ngf*2, ngf, kernel_size=5,stride=1, padding=2, bias=False),
                                           nn.LeakyReLU(0.2, True))

        # Stage_2 Network
        self.stage_2_enc_1 =  nn.Sequential(nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                                       nn.LeakyReLU(0.2, True))
        
        self.stage_2_enc_2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.csff_2_enc_1 = nn.Conv2d(ngf*2, ngf*2, kernel_size=1)
        self.csff_2_dec_1 = nn.Conv2d(ngf*2, ngf*2, kernel_size=1)
        
        for i in range(10):
            self.body_block_2.append(base_block(ngf*2))

        self.stage_2_dec_1 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.stage_2_dec_2 = nn.Sequential(nn.Conv2d(ngf*2, ngf, kernel_size=5,stride=1, padding=2, bias=False),
                                           nn.LeakyReLU(0.2, True))
        
        # Stage_3 Network
        self.stage_3_enc_1 = nn.Sequential(nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.stage_3_enc_2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.csff_3_enc_1 = nn.Conv2d(ngf*2, ngf*2, kernel_size=1)
        self.csff_3_dec_1 = nn.Conv2d(ngf*2, ngf*2, kernel_size=1)
        
        for i in range(12):
            self.body_block_3.append(base_block(ngf*2))

        self.stage_3_dec_1 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.LeakyReLU(0.2, True))
        self.stage_3_dec_2 = nn.Sequential(nn.Conv2d(ngf*2, ngf, kernel_size=5,stride=1, padding=2, bias=False),
                                           nn.LeakyReLU(0.2, True))

        self.sam12 = SAM_m(ngf)
        self.cat12 = nn.Conv2d(ngf*2, ngf, 1, 1, 0)
        self.sam23 = SAM_m(ngf)
        self.cat23 = nn.Conv2d(ngf*2, ngf, 1, 1, 0)
        self.last_conv = nn.Conv2d(ngf, out_channels, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        #container_1 = []
        image = x
        # stage_1
        x1 = self.stage_1_enc_1(image) 
        x2 = self.stage_1_enc_2(x1)
        for i, block_1 in enumerate(self.body_block_1):
            x2 = block_1(x2)
            
            if i == 4:
                x2_enc = x2
            if i == 9:
                x2_dec = x2

        x4 = self.stage_1_dec_1(x2)
        x_cat = torch.cat((x1, x4),1)
        x5 = self.stage_1_dec_2(x_cat)
        sam_feature12, out_1 = self.sam12(x5, image)
        # stage_2
        y2 = self.stage_2_enc_1(image)
        y3 = self.cat12(torch.cat([y2, sam_feature12], dim=1))
        y4 = self.stage_2_enc_2(y3)
        for i, block_2 in enumerate(self.body_block_2):
            
            if i == 5:
                y4 = y4 + self.csff_2_enc_1(x2_enc) + self.csff_2_dec_1(x2_dec)
                y4_enc = y4
            y4 = block_2(y4)
            if i == 9:
                y4_dec = y4
        y6 = self.stage_2_dec_1(y4)
        y_cat = torch.cat((y3, y6),1)
        y7 = self.stage_2_dec_2(y_cat)
        sam_feature23, out_2 = self.sam23(y7, image)
        # stage_3
        z1 = self.stage_3_enc_1(image)
        z0 = self.cat23(torch.cat([z1, sam_feature23], dim =1))
        z2 = self.stage_3_enc_2(z0)
        for i, block_3 in enumerate(self.body_block_3):
            
            if i == 6:
                z2 = z2 + self.csff_3_enc_1(y4_enc) + self.csff_3_dec_1(y4_dec)
            z2 = block_3(z2)
            
        z4 = self.stage_3_dec_1(z2)
        z_cat = torch.cat((z0, z4), 1)
        z5 = self.stage_3_dec_2(z_cat)
        z6 = self.last_conv(z5)
        out_3 = z6 + image
        
        return torch.clamp(out_1, min=-1, max =1), torch.clamp(out_2, min=-1, max =1), torch.clamp(out_3, min=-1, max =1)




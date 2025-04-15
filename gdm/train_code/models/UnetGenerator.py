import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OnlyUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)



class UnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inc11 = DoubleConv(in_channels, 64)
        self.down11 = Down(64, 128)
        self.down12 = Down(128, 256)
        self.down13 = Down(256, 256)
        self.inc12 = DoubleConv(256, 256)
        self.up11 = Up(512, 128)
        self.up12 = Up(256, 64)
        self.up13 = OnlyUp(64, 64)
        
        self.down21 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down23 = Down(256, 256)
        self.inc2 = DoubleConv(256, 256)
        self.up21 = Up(512, 128)
        self.up22 = Up(256, 64)
        self.up23 = OnlyUp(64, 64)
        
        self.down31 = Down(64, 128)
        self.down32 = Down(128, 256)
        self.down33 = Down(256, 256)
        self.inc3 = DoubleConv(256, 256)
        self.up31 = Up(512, 128)
        self.up32 = Up(256, 64)
        self.up33 = OnlyUp(64, 64)
        
        self.down41 = Down(64, 128)
        self.down42 = Down(128, 256)
        self.down43 = Down(256, 256)
        self.inc4 = DoubleConv(256, 256)
        self.up41 = Up(512, 128)
        self.up42 = Up(256, 64)
        self.up43 = OnlyUp(64, 64)
        
        self.down51 = Down(64, 128)
        self.down52 = Down(128, 256)
        self.down53 = Down(256, 256)
        self.inc5 = DoubleConv(256, 256)
        self.up51 = Up(512, 128)
        self.up52 = Up(256, 64)
        self.up53 = OnlyUp(64, 64)
        
        self.down61 = Down(64, 128)
        self.down62 = Down(128, 256)
        self.down63 = Down(256, 256)
        self.inc6 = DoubleConv(256, 256)
        self.up61 = Up(512, 128)
        self.up62 = Up(256, 64)
        self.up63 = OnlyUp(64, out_channels)

    def forward(self, x):
        
        x_inc11 = self.inc11(x)
        x11 = self.down11(x_inc11)
        x12 = self.down12(x11)
        x13 = self.down13(x12)
        x_inc12 = self.inc12(x13)
        x1 = self.up11(x_inc12, x12)
        x1 = self.up12(x1, x11)
        x1 = self.up13(x1)
        
        x21 = self.down21(x1)
        x22 = self.down22(x21)
        x23 = self.down23(x22)
        x_inc2 = self.inc2(x23)
        x2 = self.up21(x_inc2, x22)
        x2 = self.up22(x2, x21)
        x2 = self.up23(x2)
        
        x31 = self.down31(x2)
        x32 = self.down32(x31)
        x33 = self.down33(x32)
        x_inc3 = self.inc3(x33)
        x3 = self.up31(x_inc3, x32)
        x3 = self.up32(x3, x31)
        x3 = self.up33(x3)
        
        x41 = self.down41(x3)
        x42 = self.down42(x41)
        x43 = self.down43(x42)
        x_inc4 = self.inc4(x43)
        x4 = self.up41(x_inc4, x42)
        x4 = self.up42(x4, x41)
        x4 = self.up43(x4)
        
        x51 = self.down51(x4)
        x52 = self.down52(x51)
        x53 = self.down53(x52)
        x_inc5 = self.inc5(x53)
        x5 = self.up51(x_inc5, x52)
        x5 = self.up52(x5, x51)
        x5 = self.up53(x5)
        
        x61 = self.down61(x5)
        x62 = self.down62(x61)
        x63 = self.down63(x62)
        x_inc6 = self.inc6(x63)
        x6 = self.up61(x_inc6, x62)
        x6 = self.up62(x6, x61)
        logits = self.up63(x6)
        
        return logits

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dim=2, norm='instance'):
        super().__init__()

        self.dim = dim
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        if norm == 'batch':
            norm = nn.BatchNorm3d if dim == 3 else nn.BatchNorm2d
        elif norm == 'instance':
            norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv(in_channels, mid_channels, kernel_size=3, padding=1),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            conv(mid_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dim=2, norm='instance'):
        super().__init__()
        pool = nn.MaxPool3d if dim == 3 else nn.MaxPool2d
        self.maxpool_conv = nn.Sequential(
            pool(2),
            DoubleConv(in_channels, out_channels, dim=dim, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dim=2, norm='instance'):
        super().__init__()

        self.dim = dim

        if dim == 3:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dim=dim, norm=norm)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dim=dim, norm=norm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        if self.dim == 3:
            diffZ = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffX = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])
        else:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2):
        super(OutConv, self).__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
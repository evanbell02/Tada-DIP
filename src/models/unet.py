""" Full assembly of the parts to form the complete network """

import torch.nn as nn

from .unet_parts import *

class UNet(nn.Module):
  def __init__(self, n_channels, n_classes, depth=4, base_filters=64, dim=2, norm='instance'):
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.depth = depth
    self.base_filters = base_filters
    self.dim = dim
    self.norm = norm

    self.in_conv = DoubleConv(n_channels, base_filters, dim=dim, norm=norm)

    self.downs = nn.ModuleList()
    for i in range(depth):
      in_ch = base_filters * (2 ** i)
      out_ch = base_filters * (2 ** (i + 1)) if i < depth - 1 else in_ch
      self.downs.append(Down(in_ch, out_ch, dim=dim, norm=norm))

    self.ups = nn.ModuleList()
    for i in range(depth, 0, -1):
      in_ch = base_filters * (2 ** i)
      out_ch = base_filters * (2 ** (i - 2)) if i > 1 else base_filters
      self.ups.append(Up(in_ch, out_ch, dim=dim, norm=norm))

    self.out_conv = OutConv(base_filters, n_classes, dim=dim)

  def forward(self, x):
    x = self.in_conv(x)
    skip_connections = []
    for down in self.downs:
      skip_connections.append(x)
      x = down(x)
    skip_connections = skip_connections[::-1]
    for i in range(self.depth):
      x = self.ups[i](x, skip_connections[i])
    logits = self.out_conv(x)
    return logits
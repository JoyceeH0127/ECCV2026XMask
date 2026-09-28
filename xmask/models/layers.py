# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv_3x3(inc, ouc, bn=True):
    padding = 1
    module = nn.Sequential(
        nn.Conv2d(inc, ouc, kernel_size=3, stride=1, padding=padding, bias=not bn),
        nn.BatchNorm2d(ouc) if bn else nn.Identity(),
        nn.ReLU(inplace=True),
    )
    if not bn:
        init.constant_(module[0].bias, 0.0)
    return module


class UpSampleFuse(nn.Module):
    def __init__(self, incs, ouc, bn=True, relu=True):
        super().__init__()
        self.align_layers = nn.ModuleList(
            [nn.Conv2d(inc, ouc, kernel_size=1, bias=not bn) for inc in incs]
        )
        if not bn:
            for layer in self.align_layers:
                init.constant_(layer.bias, 0.0)
        self.fuse = nn.Sequential(
            nn.Conv2d(ouc, ouc, kernel_size=3, padding=1, bias=not bn),
            nn.BatchNorm2d(ouc) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.fuse_channel = ouc

    def forward(self, xs):
        x0 = self.align_layers[0](xs[0])
        out_shape = x0.shape[-2:]
        for x, layer in zip(xs[1:], self.align_layers[1:]):
            x = F.interpolate(layer(x), out_shape, mode="bilinear", align_corners=False)
            x0 = x0 + x
        return self.fuse(x0)


class SimpleDecoder(nn.Sequential):
    def __init__(self, in_channel=128, fea_channel=64, up_scale=1, out_channel=1):
        super().__init__(
            conv_3x3(in_channel, fea_channel, bn=False),
            conv_3x3(fea_channel, fea_channel, bn=False),
            nn.Conv2d(fea_channel, out_channel * (up_scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(up_scale),
        )
        init.constant_(self[-2].bias, 0.0)


class MaskHead(nn.Module):
    """CoordConv + PixelShuffle embedding head; L2-normalized (dim=16)."""

    def __init__(self, in_channel=128, fea_channel=64, up_scale=2, emb_dim=16):
        super().__init__()
        self.use_coords = True
        curr_in = in_channel + 2 if self.use_coords else in_channel
        self.layer1 = conv_3x3(curr_in, fea_channel, bn=True)
        self.layer2 = conv_3x3(fea_channel, fea_channel, bn=True)
        self.upsample_conv = nn.Conv2d(
            fea_channel, fea_channel * (up_scale ** 2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.upsample_act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(fea_channel, emb_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_coords:
            B, _, H, W = x.shape
            device = x.device
            y_range = torch.linspace(-1, 1, H, device=device)
            x_range = torch.linspace(-1, 1, W, device=device)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
            x = torch.cat(
                [x, grid_x.expand(B, 1, H, W), grid_y.expand(B, 1, H, W)], dim=1
            )
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.upsample_act(self.pixel_shuffle(self.upsample_conv(x)))
        x = self.out_conv(x)
        return F.normalize(x, p=2, dim=1)

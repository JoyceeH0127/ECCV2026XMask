# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .layers import UpSampleFuse, SimpleDecoder, MaskHead


class VGG16BN(nn.Module):
    """VGG16-BN encoder + density decoder + MaskHead embedding."""

    def __init__(self, config=None, pretrained=True):
        super().__init__()
        try:
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = models.vgg16_bn(weights=weights)
        except (AttributeError, TypeError):
            vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        lids = [0, 33, 43]
        self.encoders = nn.ModuleList(
            nn.Sequential(*features[a:b]) for a, b in zip(lids[:-1], lids[1:])
        )
        self.num_channels = [512, 512]
        self.num_stage = len(self.num_channels)
        self.fuse_layer = UpSampleFuse(self.num_channels, ouc=256, bn=False, relu=False)
        self.decoders = SimpleDecoder(
            in_channel=self.fuse_layer.fuse_channel,
            fea_channel=self.fuse_layer.fuse_channel,
            up_scale=2,
            out_channel=2,
        )
        self.maskhead = MaskHead(
            in_channel=self.fuse_layer.fuse_channel,
            fea_channel=self.fuse_layer.fuse_channel,
            up_scale=2,
            emb_dim=16,
        )

    def forward(self, image):
        fea = self.encoding(image)
        return self.decoding(fea)

    def encoding(self, x):
        feas = []
        for module in self.encoders:
            x = module(x)
            feas.append(x)
        feas = feas[-self.num_stage :]
        return self.fuse_layer(feas)

    def decoding(self, fea):
        denmap = self.decoders(fea)
        embedding = self.maskhead(fea)
        if denmap.size(1) > 1:
            den = denmap[:, 1:2] - denmap[:, :1]
        else:
            den = denmap
        return den, denmap, embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.resnet import ResNet50
from model.aspp import ASPP


class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=20):
        """
        DeepLabv3+ model as described in https://arxiv.org/pdf/1802.02611.pdf.
        """
        super().__init__()
        self.num_classes = num_classes

        # encoder
        self.backbone = ResNet50()
        self.aspp = ASPP()

        # decoder
        reduce_channels = 48  # or 32
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=reduce_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
        )

        # refine features
        refine_channels = 256
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256 + reduce_channels,
                out_channels=refine_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(refine_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=refine_channels,
                out_channels=refine_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(refine_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=refine_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        _, _, out_h_dim, out_w_dim = x.shape

        # Pass through encoder
        out_stride4, out_stride16 = self.backbone(x)
        enc_out = self.aspp(out_stride16)

        # upsample encoder output 4x
        _, _, upsamp_h_dim, upsamp_w_dim = out_stride4.shape
        upsampled_enc_out = F.interpolate(
            enc_out,
            size=(upsamp_h_dim, upsamp_w_dim),
            mode="bilinear",
            align_corners=False,
        )

        # reduce channels of low level features
        low_level_features = self.conv1x1(out_stride4)

        # concat
        x = torch.cat((upsampled_enc_out, low_level_features), dim=1)

        # refine features
        x = self.conv3x3(x)

        # upsample 4x
        x = F.interpolate(
            x, size=(out_h_dim, out_w_dim), mode="bilinear", align_corners=False
        )
        return x

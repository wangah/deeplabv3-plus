import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.resnet import ResNet50


class ASPP(nn.Module):
    def __init__(self):
        """
        ASPP module as described in the DeepLabv3 paper:
        https://arxiv.org/pdf/1706.05587.pdf
        """
        super().__init__()
        in_channels = 2048  # output channels of ResNet50.conv5
        out_channels = 256
        self.in_channels = in_channels
        self.out_channels = out_channels
        dilation_per_atrous_conv = (6, 12, 18)

        # start branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.atrous_conv1 = self.make_atrous_conv(dilation_per_atrous_conv[0])
        self.atrous_conv2 = self.make_atrous_conv(dilation_per_atrous_conv[1])
        self.atrous_conv3 = self.make_atrous_conv(dilation_per_atrous_conv[2])
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.UpsamplingBilinear2d() is depricated use F.interpolate()
        )

        # combine branches
        self.concat_conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def make_atrous_conv(self, dilation):
        return nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        _batch_dim, _channel_dim, h_dim, w_dim = x.shape

        # pass through branches
        branch1 = self.conv1x1(x)
        branch2 = self.atrous_conv1(x)
        branch3 = self.atrous_conv2(x)
        branch4 = self.atrous_conv3(x)
        branch5 = F.interpolate(
            self.pooling(x), size=(h_dim, w_dim), mode="bilinear", align_corners=False
        )

        # concat on channels dim
        x = torch.cat((branch1, branch2, branch3, branch4, branch5), dim=1)

        # pass through final 1x1 conv
        x = self.concat_conv1x1(x)
        return x

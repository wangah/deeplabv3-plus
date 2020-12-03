import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO replace strides of 2 with a dilated convolution instead?


class ResidualBlock(nn.Module):
    """
    Standard building block used in ResNet-18/34:
    https://arxiv.org/pdf/1512.03385.pdf

    No bias in conv layers:
    https://arxiv.org/pdf/1502.03167.pdf
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        assert stride in (1, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.main_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.skip_layers = nn.ModuleList()
        if stride == 2:
            self.skip_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                ]
            )

    def forward(self, x):
        main_out = self.main_layers(x)
        skip_out = x
        for f in self.skip_layers:
            skip_out = f(skip_out)
        return F.relu(main_out + skip_out)


class BottleneckBlock(nn.Module):
    """
    Standard 'bottleneck' building block used in ResNet-50/101/152:
    https://arxiv.org/pdf/1512.03385.pdf

    Also does not use biases for the conv layers.
    """

    def __init__(self, in_channels, base_out_channels, stride=1):
        super().__init__()
        assert stride in (1, 2)
        self.in_channels = in_channels
        self.base_out_channels = base_out_channels
        self.stride = stride
        final_out_channels = 4 * base_out_channels

        self.main_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(base_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                base_out_channels,
                base_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                base_out_channels,
                final_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(final_out_channels),
        )

        self.skip_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                final_out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(final_out_channels),
        )

    def forward(self, x):
        main_out = self.main_layers(x)
        skip_out = x
        for f in self.skip_layers:
            skip_out = f(skip_out)
        return F.relu(main_out + skip_out)


class ResNet50(nn.Module):
    """
    ResNet50 model specifically for DeepLabv3+. Returns downsampled feature maps
    at downsampling factors 4 (conv2) and 16 (conv4).
    """
    def __init__(self):
        super().__init__()

        # ignore conv5, avg_pool, fc, softmax layers
        # downsampled by 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = self.create_stack(
            num_blocks=3, in_channels=64, base_out_channels=64, stride=1
        )

        # downsampled by 8
        self.conv3 = self.create_stack(
            num_blocks=4, in_channels=256, base_out_channels=128, stride=2
        )

        # downsampled by 16
        self.conv4 = self.create_stack(
            num_blocks=6, in_channels=512, base_out_channels=256, stride=2
        )

    def create_stack(self, num_blocks, in_channels, base_out_channels, stride):
        block_out_channels = 4 * base_out_channels

        # apply stride at beginning of block
        stack = []
        stack.append(
            BottleneckBlock(
                in_channels=in_channels,
                base_out_channels=base_out_channels,
                stride=stride,
            )
        )

        # further blocks have stride 1,
        for _ in range(1, num_blocks):
            stack.append(
                BottleneckBlock(
                    in_channels=block_out_channels,
                    base_out_channels=base_out_channels,
                    stride=1,
                )
            )
        return nn.Sequential(*stack)

    def forward(self, x):
        x = self.conv1(x)
        out_stride_4 = self.conv2(x)
        x = self.conv3(out_stride_4)
        out_stride_16 = self.conv4(x)
        return out_stride_4, out_stride_16

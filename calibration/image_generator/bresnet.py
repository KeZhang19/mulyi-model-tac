from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class BResNet(nn.Module):
    """ResNet-U-Net style tactile RGB to per-pixel deform/depth regression."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 7, 1, 3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResBlock(c1, c1)
        self.stage2 = ResBlock(c1, c2, stride=2)
        self.stage3 = ResBlock(c2, c3, stride=2)
        self.stage4 = ResBlock(c3, c4, stride=2)
        self.bottleneck = ResBlock(c4, c4)
        self.up1 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up3 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(self.stem(x))   # 224 x 224
        s2 = self.stage2(s1)             # 112 x 112
        s3 = self.stage3(s2)             # 56 x 56
        x = self.stage4(s3)              # 28 x 28
        x = self.bottleneck(x)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        return self.head(x)


if __name__ == "__main__":
    model = BResNet()
    y = model(torch.randn(2, 3, 224, 224))
    print(y.shape)

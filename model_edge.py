import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """精简版 1D Squeeze-and-Excitation：使用 1x1 卷积消除张量 Reshape 开销"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)  # 直接点乘，自动广播，无需 view 和 expand


class ResBlock1D(nn.Module):
    """标准的 1D 残差块封装"""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock1D(out_channels)

        # 维度匹配机制：如果输入输出通道不同，使用 1x1 卷积对齐；相同则直接短接
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x)) + self.shortcut(x)


class EdgeCNN(nn.Module):
    """精简重构版 Wider Res-EdgeCNN"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

        # 模块化堆叠残差块
        self.block1 = ResBlock1D(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = ResBlock1D(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.block2(x)
        return self.classifier(x)
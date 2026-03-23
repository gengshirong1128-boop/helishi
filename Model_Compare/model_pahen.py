import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 1. 注意力模块 (SE Block) =================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ================= 2. 物理感知模块 (PET) =================
class PET_Module(nn.Module):
    """
    物理感知模块：显式估计相位偏移并校正
    """

    def __init__(self):
        super(PET_Module, self).__init__()
        self.estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 估计相位角 (-pi 到 pi)
        phase_est = self.estimator(x) * 3.1415926

        # 构建旋转矩阵
        cos_val = torch.cos(phase_est).unsqueeze(2)
        sin_val = torch.sin(phase_est).unsqueeze(2)

        I = x[:, 0, :].unsqueeze(1)
        Q = x[:, 1, :].unsqueeze(1)

        # 旋转校正
        I_new = I * cos_val - Q * sin_val
        Q_new = I * sin_val + Q * cos_val

        x_corrected = torch.cat([I_new, Q_new], dim=1)
        return x_corrected


# ================= 3. 完整 PA-HEN 架构 (大感受野版) =================
class PA_HEN(nn.Module):
    def __init__(self, num_classes=11):
        super(PA_HEN, self).__init__()

        self.pet = PET_Module()

        self.conv_block = nn.Sequential(
            # 【核心修改】kernel_size=11, padding=5
            # 更大的感受野能更好地捕捉 WBFM 的长时依赖特征
            nn.Conv1d(2, 2, kernel_size=11, padding=5, groups=2),

            nn.Conv1d(2, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool1d(2)
        )

        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # Bi-GRU (64*2)
            nn.ReLU(),
            nn.Dropout(0.3),  # 调低 Dropout 防止欠拟合
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 维度适配
        if x.shape[1] == 128 and x.shape[2] == 2:
            x = x.permute(0, 2, 1)

        x = self.pet(x)
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        _, h_n = self.gru(x)
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.classifier(hidden)
        return out
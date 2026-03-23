import torch
import torch.nn as nn


# ================= 1. 物理感知模块 (PET) =================
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
            nn.Tanh()  # 输出范围 (-1, 1)，对应 (-pi, pi)
        )

    def forward(self, x):
        # x: (Batch, 2, 128)

        # 估计相位偏置 phi
        phi = self.estimator(x) * 3.1415926  # (Batch, 1)

        # 构建旋转矩阵元素
        cos_val = torch.cos(phi).unsqueeze(2)  # (Batch, 1, 1)
        sin_val = torch.sin(phi).unsqueeze(2)

        # 提取 I/Q 路
        I = x[:, 0, :].unsqueeze(1)  # (Batch, 1, 128)
        Q = x[:, 1, :].unsqueeze(1)

        # 执行逆旋转校正 (Anti-Rotation)
        I_new = I * cos_val - Q * sin_val
        Q_new = I * sin_val + Q * cos_val

        # 拼接回 (Batch, 2, 128)
        x_corrected = torch.cat([I_new, Q_new], dim=1)
        return x_corrected


# ================= 2. Transformer 主模型 =================
class TransformerModel(nn.Module):
    def __init__(self, num_classes=11):
        super(TransformerModel, self).__init__()

        # --- 插入 PET 模块 ---
        self.pet = PET_Module()

        # 1. CNN 特征提取
        # Input: (Batch, 2, 128)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # -> (Batch, 32, 64)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.3)
        # Output: (Batch, 64, 64)

        # 2. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. 分类头
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 维度修正 (Batch, 128, 2) -> (Batch, 2, 128)
        if x.shape[1] == 128 and x.shape[2] == 2:
            x = x.permute(0, 2, 1)

        # 【第一步】物理校正
        x = self.pet(x)

        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Transformer (需转置为 Batch, Seq, Dim)
        x = x.permute(0, 2, 1)

        x = self.transformer_encoder(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        # FC
        x = self.fc(x)
        return x
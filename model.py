import torch
import torch.nn as nn

# ==========================================
# 核心教师架构：完全采用你 test.py 中的轻量级 Transformer
# 优势：在 RML2016.10a 数据集上已被证实能突破 90%
# ==========================================

class TransformerModel(nn.Module):
    def __init__(self, num_classes=11):
        super(TransformerModel, self).__init__()

        # 1. CNN 特征提取 (加入 BatchNorm 防止梯度消失)
        # Input: (Batch, 2, 128)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # -> (Batch, 32, 64)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
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

        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Transformer (需转置为 Batch, Seq, Dim)
        x = x.permute(0, 2, 1)  # -> (Batch, 64, 64)
        x = self.transformer_encoder(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        # FC
        x = self.fc(x)
        return x
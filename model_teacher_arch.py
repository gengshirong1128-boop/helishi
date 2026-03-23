import torch
import torch.nn as nn
import math


# ==========================================
# 项目作者：耿世荣 (24201172)
# 模块功能：教师模型架构 (Transformer-CNN 融合)
# ==========================================

class PositionalEncoding(nn.Module):
    """时序信号必备：注入位置信息，打破 10% 盲猜魔咒"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    """
    高精度教师模型：CNN 提取局部物理特征 + Transformer 捕获全局长距离依赖
    """

    def __init__(self, num_classes=10):
        super(TransformerModel, self).__init__()

        # 1. CNN 骨干：降维并提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 序列长度 128 -> 64

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # 序列长度 64 -> 32
        )

        # 2. 位置编码器
        self.pos_encoder = PositionalEncoding(d_model=128, max_len=32)

        # 3. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 2, 128)
        x = self.cnn(x)  # 降维特征提取: (Batch, 128, 32)
        x = x.transpose(1, 2)  # 维度对齐 Transformer: (Batch, 32, 128)

        x = self.pos_encoder(x)  # 注入时间序列位置信息
        x = self.transformer(x)  # 全局注意力计算: (Batch, 32, 128)

        x = x.mean(dim=1)  # 全局平均池化 (GAP): (Batch, 128)
        return self.classifier(x)  # 分类输出
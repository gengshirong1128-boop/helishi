import torch
import torch.nn as nn
import numpy as np

# --- 关键修改：直接导入刚才创建的纯 PyTorch Mamba ---
# 不需要再担心 import error 了
try:
    from mamba_simple import MambaBlock

    HAS_MAMBA = True
    print("[Model] Success: Loaded Pure PyTorch Mamba!")
except ImportError:
    HAS_MAMBA = False
    print("[Model] Warning: mamba_simple.py not found. Using LSTM fallback.")


# --- 1. PET (Physics-Aware Phase Estimation) ---
class PETModule(nn.Module):
    def __init__(self, input_len=128):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        phi_est = self.estimator(x) * np.pi
        cos_phi = torch.cos(phi_est).unsqueeze(-1)
        sin_phi = torch.sin(phi_est).unsqueeze(-1)
        I, Q = x[:, 0:1, :], x[:, 1:2, :]
        return torch.cat([I * cos_phi + Q * sin_phi, -I * sin_phi + Q * cos_phi], dim=1)


# --- 2. RepVGG Block 1D ---
class RepVGGBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.identity = nn.BatchNorm1d(in_channels) if in_channels == out_channels and stride == 1 else None

        if deploy:
            self.rbr_reparam = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2,
                                         bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.act = nn.ReLU()

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'): return self.act(self.rbr_reparam(x))
        id_out = self.identity(x) if self.identity else 0
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'): return
        k3, b3 = self._fuse(self.rbr_dense)
        k1, b1 = self._fuse(self.rbr_1x1)
        k0, b0 = self._fuse_identity(self.identity, k3.shape)
        self.rbr_reparam = nn.Conv1d(self.rbr_dense[0].in_channels, self.rbr_dense[0].out_channels,
                                     self.rbr_dense[0].kernel_size, self.rbr_dense[0].stride,
                                     self.rbr_dense[0].padding, bias=True)
        self.rbr_reparam.weight.data = k3 + torch.nn.functional.pad(k1, [1, 1]) + k0
        self.rbr_reparam.bias.data = b3 + b1 + b0
        del self.rbr_dense, self.rbr_1x1, self.identity

    def _fuse(self, branch):
        k, rm, rv, g, b, eps = branch[0].weight, branch[1].running_mean, branch[1].running_var, branch[1].weight, \
        branch[1].bias, branch[1].eps
        std = (rv + eps).sqrt()
        return k * (g / std).reshape(-1, 1, 1), b - rm * g / std

    def _fuse_identity(self, branch, shape):
        if branch is None: return 0, 0
        rm, rv, g, b, eps = branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        k_val = torch.zeros(shape, device=g.device)
        for i in range(shape[0]): k_val[i, i, 1] = 1
        return k_val * (g / (rv + eps).sqrt()).reshape(-1, 1, 1), b - rm * g / (rv + eps).sqrt()


# --- 3. PMR-Net V3 (True Mamba Version) ---
class PMRNet(nn.Module):
    def __init__(self, num_classes=11, d_model=256, deploy=False):
        super().__init__()

        self.pet = PETModule()

        # 特征提取
        self.stem = nn.Sequential(
            RepVGGBlock1D(2, 64, kernel_size=3, deploy=deploy),
            RepVGGBlock1D(64, d_model, kernel_size=3, stride=2, deploy=deploy),
            nn.Dropout(0.3)
        )

        # 核心：使用 MambaBlock (或降级为 LSTM)
        if HAS_MAMBA:
            # 堆叠两层 Mamba 以获得更强能力
            self.seq_layer = nn.Sequential(
                MambaBlock(d_model=d_model, d_state=16, expand=2),
                MambaBlock(d_model=d_model, d_state=16, expand=2)
            )
        else:
            self.seq_layer = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True,
                                     dropout=0.3)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (B, 2, 128)
        x = self.pet(x)
        x = self.stem(x)

        # (B, D, L) -> (B, L, D) for Mamba/LSTM
        x = x.transpose(1, 2)

        if HAS_MAMBA:
            x = self.seq_layer(x)
        else:
            if isinstance(self.seq_layer, nn.LSTM):
                self.seq_layer.flatten_parameters()
                x, _ = self.seq_layer(x)

        # (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        return self.classifier(x)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock1D):
                m.switch_to_deploy()
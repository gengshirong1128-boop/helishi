import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


# ==========================================
# 项目作者：耿世荣 (24201172)
# 模块功能：数据加载与增强 [cite: 36, 45]
# ==========================================

def load_data_rml2016(filepath, remove_am_ssb=True):
    print(f"正在读取数据: {filepath} ...")
    try:
        with open(filepath, 'rb') as f:
            # 针对 Python 3.13 的特殊处理
            xd = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"读取错误: {e}")
        return None, None, None, None

    all_mods = sorted(list(set([x[0] for x in xd.keys()])))
    snrs = sorted(list(set([x[1] for x in xd.keys()])))

    if remove_am_ssb and 'AM-SSB' in all_mods:
        mods = [m for m in all_mods if m != 'AM-SSB']
    else:
        mods = all_mods

    X, lbl = [], []
    for mod in mods:
        for snr in snrs:
            X.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append(mods.index(mod))

    return np.vstack(X), np.array(lbl), mods, snrs


class RadioSigDataset(Dataset):
    """支持数据增强的 Dataset [cite: 22, 36]"""

    def __init__(self, X, y, transform=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def add_physics_aug(self, signal):
        """数据增强：模拟物理衰落与相移 """
        # 随机相移
        if torch.rand(1) > 0.5:
            phase = torch.rand(1) * 2 * np.pi
            I, Q = signal[0], signal[1]
            I_new = I * torch.cos(phase) - Q * torch.sin(phase)
            Q_new = I * torch.sin(phase) + Q * torch.cos(phase)
            signal = torch.stack([I_new, Q_new], dim=0)
        return signal

    def __getitem__(self, idx):
        signal, label = self.X[idx], self.y[idx]
        if self.transform:
            signal = self.add_physics_aug(signal)
        return signal, label
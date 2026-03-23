import os
import torch
import numpy as np
import warnings

# ==========================================
# 项目作者：耿世荣 (24201172)
# 模块功能：工具类与智能路径识别 [cite: 51, 55]
# ==========================================

def set_random_seed(seed=2025):
    """固定随机种子以保证复现 [cite: 51]"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    # 忽略讨厌的 NumPy 警告
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning) if 'VisibleDeprecationWarning' in globals() else None

def get_smart_path(relative_path):
    """
    智能路径识别：
    如果 code 文件夹内运行，则寻找 ../relative_path
    如果在项目根目录运行，则寻找 ./relative_path
    """
    if os.path.exists(relative_path):
        return relative_path
    parent_path = os.path.join('..', relative_path)
    if os.path.exists(parent_path):
        return parent_path
    return relative_path

def get_next_filename(directory, base_name, ext):
    """生成带序号的文件名 [cite: 51]"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    i = 1
    while True:
        filename = os.path.join(directory, f"{base_name}_{i}{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1
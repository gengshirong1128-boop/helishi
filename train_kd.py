import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# === 导入自定义模块 ===
from model import TransformerModel  # 92.59% 的学霸教师
from model_edge import EdgeCNN  # 我们刚刚升级的带残差轻量级学生

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join('data', 'RML2016.10a_dict.pkl')
MODEL_DIR = 'model_data'
TEACHER_PATH = os.path.join(MODEL_DIR, 'teacher_supreme.pth')
STUDENT_PATH = os.path.join(MODEL_DIR, 'model_student_kd.pth')

BATCH_SIZE = 128
EPOCHS = 80  # 蒸馏需要多一点轮次来消化知识
LEARNING_RATE = 0.001

# 蒸馏超参数
TEMPERATURE = 4.0
ALPHA = 0.5  # 软硬标签权重


def load_data(filepath, remove_am_ssb=True):
    print(f"📦 正在加载数据: {filepath} ...")
    with open(filepath, 'rb') as f:
        xd = pickle.load(f, encoding='latin1')

    snrs = sorted(list(set([x[1] for x in xd.keys()])))
    all_mods = sorted(list(set([x[0] for x in xd.keys()])))
    mods = [m for m in all_mods if m != 'AM-SSB'] if remove_am_ssb and 'AM-SSB' in all_mods else all_mods

    X, lbl = [], []
    for mod in mods:
        for snr in snrs:
            data = xd[(mod, snr)]
            X.append(data)
            for i in range(data.shape[0]):
                lbl.append(mods.index(mod))

    return np.vstack(X), np.array(lbl), mods


def mixup_data(x, y, alpha=0.2):
    '''MixUp 信号混叠：让模型学习分辨叠加信号，大幅提升泛化边界'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    print("=== 🌟 MixUp 高阶知识蒸馏系统启动 ===")

    X, y, mods = load_data(DATA_PATH, remove_am_ssb=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025, stratify=y
    )

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
                             batch_size=1024)

    # 1. 加载 92.59% 的学霸教师
    teacher = TransformerModel(num_classes=len(mods)).to(DEVICE)
    try:
        teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
        print("✅ 成功请出 92.59% 的学霸教师！")
    except Exception as e:
        print(f"❌ 加载教师权重失败，请检查文件是否存在: {TEACHER_PATH}")
        return
    teacher.eval()  # 教师永远不更新，只负责输出答案

    # 2. 初始化学生
    student = EdgeCNN(num_classes=len(mods)).to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 余弦退火：平滑学习率曲线
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f">>> 开始进行高强度蒸馏传承 (Epochs={EPOCHS})...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        student.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 使用 MixUp 魔法：将两个信号叠加
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)

            # 教师给出对“叠加信号”的高级软标签
            with torch.no_grad():
                t_logits = teacher(mixed_inputs)

            optimizer.zero_grad()
            s_logits = student(mixed_inputs)

            # 1. 软标签损失 (KL Divergence)
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(s_logits / TEMPERATURE, dim=1),
                F.softmax(t_logits / TEMPERATURE, dim=1)
            ) * (TEMPERATURE * TEMPERATURE)

            # 2. 硬标签损失 (交叉熵)
            hard_loss = mixup_criterion(F.cross_entropy, s_logits, labels_a, labels_b, lam)

            loss = ALPHA * soft_loss + (1. - ALPHA) * hard_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # 验证阶段 (不使用 MixUp)
        student.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, pred = torch.max(student(inputs), 1)
                val_total += labels.size(0);
                val_correct += (pred == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step()

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), STUDENT_PATH)

    print(f"\n🎉 蒸馏完毕！学生模型最佳全局精度: {best_acc:.2f}% (模型已保存至 {STUDENT_PATH})")
    print("💡 下一步：请运行 main.py 对边缘学生模型进行最终的各 SNR 性能拆解！")


if __name__ == '__main__':
    main()
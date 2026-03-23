import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 导入你 model.py 里的最强教师 (PET + Transformer) ===
try:
    from model import TransformerModel
except ImportError:
    print("❌ 找不到 model.py，请确保你的 PET+Transformer 代码在 model.py 中。")
    exit()

# ================= 核心配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join('data', 'RML2016.10a_dict.pkl')
MODEL_DIR = 'model_data'
RESULT_DIR = 'prediction_result'
MODEL_PATH = os.path.join(MODEL_DIR, 'teacher_supreme.pth')

BATCH_SIZE = 64  # 教师模型调小 Batch，增加梯度更新频率
EPOCHS = 60  # 配合余弦退火，60轮足够
LEARNING_RATE = 0.001

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)

# 全局学术图表设置 (Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def load_data(filepath, remove_am_ssb=True):
    print(f"📦 正在加载并清洗数据: {filepath} ...")
    with open(filepath, 'rb') as f:
        xd = pickle.load(f, encoding='latin1')

    snrs = sorted(list(set([x[1] for x in xd.keys()])))
    all_mods = sorted(list(set([x[0] for x in xd.keys()])))
    mods = [m for m in all_mods if m != 'AM-SSB'] if remove_am_ssb and 'AM-SSB' in all_mods else all_mods

    X, lbl, snr_list = [], [], []
    for mod in mods:
        for snr in snrs:
            data = xd[(mod, snr)]
            X.append(data)
            for i in range(data.shape[0]):
                lbl.append(mods.index(mod))
                snr_list.append(snr)  # 记录每个样本的 SNR 用于后续画图

    return np.vstack(X), np.array(lbl), np.array(snr_list), mods, snrs


def main():
    print("=== 🌟 教师模型终极优化与评测系统启动 ===")

    X, y, snr_labels, mods, snrs = load_data(DATA_PATH, remove_am_ssb=True)

    # 划分数据集 (保留 SNR 标签用于测试阶段统计)
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, _, snr_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=2025, stratify=y
    )

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long(),
                                           torch.from_numpy(snr_test).long()),
                             batch_size=1024)

    model = TransformerModel(num_classes=len(mods)).to(DEVICE)

    # 【核心优化 1】Label Smoothing 解决过拟合与 QAM 混淆
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 【核心优化 2】AdamW 增加权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 【核心优化 3】余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f">>> 开始超强教师模型训练 (Epochs={EPOCHS}) ...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0);
            correct += (pred == labels).sum().item()

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, pred = torch.max(model(inputs), 1)
                val_total += labels.size(0);
                val_correct += (pred == labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        scheduler.step()  # 余弦退火步进

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n✅ 训练结束！最佳全局精度: {best_acc:.2f}%")
    print(">>> 正在生成论文级高质量图表...")

    # ================= 论文级数据评估与绘图 =================
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_preds, all_labels, all_snrs = [], [], []
    with torch.no_grad():
        for inputs, labels, snr_batch in test_loader:
            inputs = inputs.to(DEVICE)
            _, pred = torch.max(model(inputs), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snr_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)

    # 1. 绘制学术级 SNR 曲线图
    acc_by_snr = []
    for snr in snrs:
        idx = (all_snrs == snr)
        acc = np.mean(all_preds[idx] == all_labels[idx]) * 100
        acc_by_snr.append(acc)
        if snr >= 0:
            print(f"SNR {snr:>2}dB: 准确率 = {acc:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(snrs, acc_by_snr, marker='o', markersize=8, color='#d62728', linewidth=2.5,
             label='PET-Transformer (Teacher)')
    plt.axhline(y=90.0, color='#1f77b4', linestyle='--', linewidth=2, label='90% Target')
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.xlabel('Signal-to-Noise Ratio (SNR, dB)', fontweight='bold')
    plt.ylabel('Classification Accuracy (%)', fontweight='bold')
    plt.title('Teacher Model Accuracy across SNRs', fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(loc='lower right')

    snr_curve_path = os.path.join(RESULT_DIR, 'paper_teacher_snr_curve.pdf')
    plt.savefig(snr_curve_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

    # 2. 绘制高信噪比 (>=0dB) 下的学术级混淆矩阵
    high_snr_idx = all_snrs >= 0
    cm = confusion_matrix(all_labels[high_snr_idx], all_preds[high_snr_idx])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=mods, yticklabels=mods,
                annot_kws={"size": 10}, cbar_kws={'label': 'Prediction Probability'})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.ylabel('True Modulation Scheme', fontweight='bold')
    plt.xlabel('Predicted Modulation Scheme', fontweight='bold')
    plt.title('Confusion Matrix at High SNR ($\geq$ 0 dB)', fontweight='bold')

    cm_path = os.path.join(RESULT_DIR, 'paper_teacher_confusion_matrix.pdf')
    plt.savefig(cm_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 评估完成！请前往 prediction_result 文件夹查看：")
    print(f"   1. SNR曲线图: paper_teacher_snr_curve.pdf (矢量图，无限放大)")
    print(f"   2. 混淆矩阵: paper_teacher_confusion_matrix.pdf (矢量图，揭示哪两个类别最容易混淆)")


if __name__ == '__main__':
    main()
import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import TransformerModel

# ================= 核心配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join('data', 'RML2016.10a_dict.pkl')
MODEL_PATH = os.path.join('model_data', 'teacher_supreme.pth')
RESULT_DIR = 'prediction_result'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def load_data(filepath, remove_am_ssb=True):
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
                snr_list.append(snr)

    return np.vstack(X), np.array(lbl), np.array(snr_list), mods, snrs


def main():
    print("=== 🎯 教师模型极速评估系统 (已修复 SNR Bug) ===")

    X, y, snr_labels, mods, snrs = load_data(DATA_PATH, remove_am_ssb=True)

    # 【Bug 已修复】：这里正确地传入了 snr_labels
    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, y, snr_labels, test_size=0.2, random_state=2025, stratify=y
    )

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long(),
                      torch.from_numpy(snr_test).long()),
        batch_size=1024
    )

    print(f"正在加载已训练好的完美权重: {MODEL_PATH}")
    model = TransformerModel(num_classes=len(mods)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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

    acc_by_snr = []
    print("\n>>> 各信噪比 (SNR) 准确率拆解：")
    for snr in snrs:
        idx = (all_snrs == snr)
        if np.sum(idx) > 0:
            acc = np.mean(all_preds[idx] == all_labels[idx]) * 100
        else:
            acc = 0.0
        acc_by_snr.append(acc)
        if snr >= 0:
            mark = "🌟" if acc >= 90.0 else ""
            print(f"    SNR {snr:>2}dB: 准确率 = {acc:>6.2f}% {mark}")

    # ================= 绘制 SNR 曲线 =================
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, acc_by_snr, marker='o', markersize=8, color='#d62728', linewidth=2.5,
             label='PET-Transformer (Teacher)')
    plt.axhline(y=90.0, color='#1f77b4', linestyle='--', linewidth=2, label='90% Target')
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.xlabel('Signal-to-Noise Ratio (SNR, dB)', fontweight='bold')
    plt.ylabel('Classification Accuracy (%)', fontweight='bold')
    plt.title('Teacher Model Accuracy across SNRs', fontweight='bold')
    plt.ylim(0, 105)
    plt.xlim(min(snrs) - 1, max(snrs) + 1)
    plt.legend(loc='lower right')

    snr_curve_path = os.path.join(RESULT_DIR, 'paper_teacher_snr_curve.pdf')
    plt.savefig(snr_curve_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

    # ================= 绘制高信噪比混淆矩阵 =================
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
    plt.title(r'Confusion Matrix at High SNR ($\geq$ 0 dB)', fontweight='bold')

    cm_path = os.path.join(RESULT_DIR, 'paper_teacher_confusion_matrix.pdf')
    plt.savefig(cm_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 评估完成！高质量图表已更新。")


if __name__ == '__main__':
    main()
import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# === 导入自定义模块 ===
from utils import set_random_seed, get_next_filename
from data_loader import RadioSigDataset
from model_edge import EdgeCNN

# ================= 配置 =================
DATA_PATH = os.path.join('data', 'RML2016.10a_dict.pkl')
MODEL_PATH = os.path.join('model_data', 'model_student_kd.pth')
RESULT_DIR = 'prediction_result'
RESULT_FILE = os.path.join(RESULT_DIR, 'test_result.json')
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=== 耿世荣 的边缘端调制识别最终评估系统 (学术版) 启动 ===")
    set_random_seed(2025)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("正在读取数据...")
    try:
        with open(DATA_PATH, 'rb') as f:
            xd = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"❌ 数据读取失败: {e}")
        return

    snrs = sorted(list(set([x[1] for x in xd.keys()])))
    all_mods = sorted(list(set([x[0] for x in xd.keys()])))

    # 【对齐训练集】过滤掉 AM-SSB
    if 'AM-SSB' in all_mods:
        mods = [m for m in all_mods if m != 'AM-SSB']
    else:
        mods = all_mods

    print("正在加载轻量级蒸馏模型 (EdgeCNN)...")
    model = EdgeCNN(num_classes=len(mods)).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 模型权重加载成功！\n")
    except FileNotFoundError:
        print(f"❌ 找不到权重文件 {MODEL_PATH}！")
        return
    model.eval()

    print("=====================================================")
    print(f"| {'信噪比 (SNR)':<12} | {'准确率 (Accuracy)':<18} | {'样本详情 (Correct/Total)':<20} |")
    print("=====================================================")

    results = {
        "overall_accuracy": 0.0,
        "accuracy_by_snr": {},
        "model_info": "EdgeCNN with SE-Attention (Knowledge Distilled)"
    }

    total_correct = 0
    total_samples = 0
    snr_acc_list = []

    for snr in snrs:
        X_snr = []
        lbl_snr = []
        for mod in mods:
            if (mod, snr) in xd:
                data = xd[(mod, snr)]
                X_snr.append(data)
                for i in range(data.shape[0]):
                    lbl_snr.append(mods.index(mod))

        X_snr = np.vstack(X_snr)
        lbl_snr = np.array(lbl_snr)

        test_dataset = RadioSigDataset(X_snr, lbl_snr, transform=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        correct, sub_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                sub_total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = float(correct) / sub_total if sub_total > 0 else 0
        results["accuracy_by_snr"][str(snr)] = acc
        snr_acc_list.append(acc * 100)  # 转换为百分比用于画图

        # 终端输出更加严谨和具体的数值
        target_mark = "🌟 达标" if acc >= 0.9 else ""
        print(f"| {snr:>5} dB      | {acc * 100:>15.2f} %  | {correct:>8} / {sub_total:<10} | {target_mark}")

        total_correct += correct
        total_samples += sub_total

    print("=====================================================")
    overall_acc = total_correct / total_samples
    results["overall_accuracy"] = overall_acc
    print(f"\n全局平均准确率: {overall_acc * 100:.2f}% (含极低信噪比恶劣工况)")

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # ================= 绘制论文级高质量图表 =================
    # 全局设置学术论文常用样式 (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'  # 公式字体风格
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 600

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制主曲线：使用较粗的线条(linewidth=2.5)和清晰的标记点(markersize=8)
    ax.plot(snrs, snr_acc_list, marker='o', markersize=8, linestyle='-',
            color='#1f77b4', linewidth=2.5, label='EdgeCNN (Ours)')

    # 绘制 90% 达标线，使用显眼的深红色
    ax.axhline(y=90.0, color='#d62728', linestyle='--', linewidth=2, label='90% Target')

    # 精细化网格：主网格为实线，副网格为虚线
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

    # 设置坐标轴标签和标题
    ax.set_xlabel('Signal-to-Noise Ratio (SNR, dB)', fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Modulation Recognition Performance across SNRs', fontweight='bold')

    # 锁定坐标轴范围，使图表看起来更丰满
    ax.set_ylim(0, 105)
    ax.set_xlim(min(snrs) - 1, max(snrs) + 1)

    # 图例设置：带边框和轻微背景
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fancybox=False)

    # 自动命名并同时保存为高质量 PNG 和矢量 PDF
    plot_filename_png = get_next_filename(RESULT_DIR, 'paper_snr_curve', '.png')
    plot_filename_pdf = plot_filename_png.replace('.png', '.pdf')

    plt.savefig(plot_filename_png, dpi=600, bbox_inches='tight')
    plt.savefig(plot_filename_pdf, format='pdf', bbox_inches='tight')

    print(f"\n📊 论文级高质量图表已生成！")
    print(f"   -> 插入 Word 请用: {plot_filename_png} (600 DPI高清)")
    print(f"   -> 插入 LaTeX 请用: {plot_filename_pdf} (无损矢量图)")


if __name__ == '__main__':
    main()
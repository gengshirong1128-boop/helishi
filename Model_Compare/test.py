import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# === 【关键】从 model.py 导入模型 ===
# 确保 model.py 和 test.py 在同一个文件夹下
from model import TransformerModel

# ================= 核心配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # 启用一些加速设置
    torch.backends.cudnn.benchmark = True

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
# 请确保路径正确
DATA_PATH = os.path.join('RML2016.10a_dict.pkl', 'RML2016.10a_dict.pkl')

MODEL_DIR = 'model_data'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')


def load_data_rml2016(filename):
    print(f"正在读取数据: {filename} ...")
    try:
        with open(filename, 'rb') as f:
            # 兼容不同 Python 版本的 pickle
            xd = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"读取错误: {e}")
        return None, None, None

    mods, snrs = [sorted(list(set([x[i] for x in xd.keys()]))) for i in [0, 1]]
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append(mods.index(mod))
    X = np.vstack(X)
    lbl = np.array(lbl)
    return X, lbl, mods


# ================= 主训练循环 =================
if __name__ == '__main__':
    if os.path.exists(DATA_PATH):
        X, y, classes = load_data_rml2016(DATA_PATH)
    else:
        print(f"错误：找不到数据集 {DATA_PATH}！")
        exit()

    # 分层抽样，保证分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025, stratify=y
    )

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
                             batch_size=BATCH_SIZE)

    # 初始化模型
    model = TransformerModel(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # 使用 AdamW + Weight Decay (1e-3) 防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # 【修复】删除了 verbose=True，防止报错
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print(f"当前设备: {DEVICE}")
    print(f">>> 开始训练 (Modular PET + Transformer)...")

    best_acc = 0.0

    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            avg_loss = total_loss / len(train_loader)

            # 验证
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_acc = 100 * test_correct / test_total

            # 调度器更新
            scheduler.step(test_acc)
            # 手动获取当前学习率用于打印
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"[Epoch {epoch + 1}/{EPOCHS}] lr: {current_lr:.6f} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"   >>> 模型已保存 (Acc: {best_acc:.2f}%)")

    except KeyboardInterrupt:
        print("\n训练中断...")

    print(f"训练结束，最佳准确率: {best_acc:.2f}%")
    # 确保最后也保存一下（虽然通常保存最佳的就够了）
    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader_pmr import load_data_pmr
from model_pmr import PMRNet
import os
import time

# 参数配置
DATA_FILE = 'RML2016.10a_dict.pkl'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 80
BATCH_SIZE = 128
LR = 0.0005  # <--- 修改点：学习率稍微调小一点点，更稳


def train_pmr():
    train_loader, test_loader, mods, _, _ = load_data_pmr(DATA_FILE, batch_size=BATCH_SIZE)

    print(f"[PMR-Train] Initializing Pure Mamba Model on {DEVICE}...")
    model = PMRNet(num_classes=len(mods), d_model=256, deploy=False).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    print(f"==================================================")
    print(f"[PMR-Train] STARTING! (With Gradient Clipping)")
    print(f"==================================================")

    best_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            # --- 关键修改：梯度裁剪 (Gradient Clipping) ---
            # 这行代码是防止 Loss 变成 6800万 的绝对核心！
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if i == 0 or (i + 1) % 50 == 0:
                print(f"  [Epoch {epoch + 1}] Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # 验证循环
        print(f"  [Epoch {epoch + 1}] Validating...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(f"==> Epoch {epoch + 1} Finished in {epoch_time:.1f}s")
        print(f"    Avg Loss: {avg_loss:.4f} | Val Acc: {acc * 100:.2f}% | Best: {best_acc * 100:.2f}%")

        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'pmr_best_train.pth')
            print("    [Saved Best Model]")
        else:
            patience_counter += 1

        if patience_counter >= 15:
            print("Early stopping triggered.")
            break

    print(f"[PMR-Train] Done. Best Acc: {best_acc * 100:.2f}%")

    # 转换模型
    deploy_model = PMRNet(num_classes=len(mods), d_model=256, deploy=False).to(DEVICE)
    deploy_model.load_state_dict(torch.load('pmr_best_train.pth'))
    deploy_model.eval()
    deploy_model.switch_to_deploy()
    torch.save(deploy_model.state_dict(), 'pmr_deploy.pth')
    print("[PMR-Train] Deploy model saved.")


if __name__ == '__main__':
    if os.path.exists(DATA_FILE):
        train_pmr()
    else:
        print(f"Error: {DATA_FILE} not found.")
import time
import torch
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 加载所有待测模型 =================
models_to_test = {}

# 1. 轻量级 EdgeCNN
try:
    from model_edge import EdgeCNN
    models_to_test['EdgeCNN'] = EdgeCNN(num_classes=11)
except Exception as e:
    print(f"跳过 EdgeCNN: {e}")

# 2. test.py 中的基础 Transformer
try:
    from test import TransformerModel as TestTransformer
    models_to_test['Transformer (test.py)'] = TestTransformer(num_classes=11)
except Exception as e:
    print(f"跳过 test.py Transformer: {e}")

# 3. model.py 中的 PET + Transformer
try:
    from model import TransformerModel as PetTransformer
    models_to_test['PET+Transformer (model.py)'] = PetTransformer(num_classes=11)
except Exception as e:
    print(f"跳过 model.py Transformer: {e}")

# 4. PA_HEN
try:
    from model_pahen import PA_HEN
    models_to_test['PA-HEN'] = PA_HEN(num_classes=11)
except Exception as e:
    print(f"跳过 PA-HEN (请确保文件名为 model_pahen.py): {e}")

# 5. PMRNet (必须调用 switch_to_deploy 以测试部署态性能)
try:
    from model_pmr import PMRNet
    pmr = PMRNet(num_classes=11, d_model=256, deploy=False)
    pmr.switch_to_deploy()
    models_to_test['PMRNet (部署模式)'] = pmr
except Exception as e:
    print(f"跳过 PMRNet: {e}")

# ================= 2. 测量逻辑 =================
def run_benchmark(model, device, dummy_input, iters=300):
    model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    # 预热：防止 CPU 频率初始变化导致数据不准
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # 正式计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / iters) * 1000
    fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = params * 4 / (1024 * 1024)

    return params, size_mb, avg_time_ms, fps

# ================= 3. 执行测试 =================
if __name__ == '__main__':
    print(">>> 启动 CPU 部署性能基准测试 (模拟边缘设备) <<<\n")
    device = torch.device("cpu")
    # 构建与实际数据维度一致的虚拟输入 (Batch=1, Channels=2, Length=128)
    dummy_input = torch.randn(1, 2, 128)

    results = []
    for name, model in models_to_test.items():
        print(f"正在测试: {name} ...")
        params, size_mb, avg_ms, fps = run_benchmark(model, device, dummy_input)
        results.append({
            'name': name,
            'params': params,
            'size_mb': size_mb,
            'avg_ms': avg_ms,
            'fps': fps
        })

    # 按照吞吐量 (FPS) 从高到低排序
    results.sort(key=lambda x: x['fps'], reverse=True)

    # 输出对齐的客观数据表
    print("\n" + "="*85)
    print(f"| {'模型架构':<26} | {'参数量':<10} | {'体积(MB)':<8} | {'单次延迟(ms)':<12} | {'FPS (次/秒)':<12} |")
    print("-" * 85)
    for r in results:
        print(f"| {r['name']:<26} | {r['params']:<10,} | {r['size_mb']:<8.2f} | {r['avg_ms']:<12.2f} | {r['fps']:<12.0f} |")
    print("=" * 85)
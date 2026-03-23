import torch
import os
from model_edge import EdgeCNN  # 导入轻量级模型 (修正自 model_edge)

# ==========================================
# 项目作者：耿世荣
# 学号：24201172
# 模块功能：将 PyTorch 模型导出为边缘端通用的 ONNX 格式
# ==========================================

MODEL_DIR = 'model_data'
# 首选：与训练脚本一致的权重名；兼容旧脚本中可能使用的名字
PREFERRED_MODEL_NAMES = [
    os.path.join(MODEL_DIR, 'model_student_kd.pth'),  # 推荐（由 train_kd.py 生成）
    os.path.join(MODEL_DIR, 'model_student_edge.pth'),  # 兼容旧脚本
]
OUTPUT_ONNX_PATH = os.path.join(MODEL_DIR, 'model_edge_deploy.onnx')


def export_to_onnx():
    print(f"=== 耿世荣 的 ONNX 模型导出工具 ===")

    # 寻找第一个存在的权重文件
    input_model_path = None
    for p in PREFERRED_MODEL_NAMES:
        if os.path.exists(p):
            input_model_path = p
            break

    if input_model_path is None:
        print(f"❌ 错误: 在 {MODEL_DIR} 下未找到以下之一的权重文件:\n  {PREFERRED_MODEL_NAMES}")
        print("请先运行训练脚本生成该文件或将你的权重重命名为其中之一。")
        return

    print(f"使用权重文件: {input_model_path}")

    # 1. 初始化模型并加载权重
    num_classes = 11  # RML2016.10a 有11种调制方式
    model = EdgeCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(input_model_path, map_location='cpu'))
    model.eval()

    # 2. 创建一个虚拟输入张量 (Batch_Size=1, Channels=2, Length=128)
    # 这是 ONNX 追踪模型计算图所必需的
    dummy_input = torch.randn(1, 2, 128)

    # 3. 导出 ONNX
    print(f"正在将模型导出为 ONNX 格式...")
    torch.onnx.export(
        model,  # 运行的模型
        dummy_input,  # 虚拟输入
        OUTPUT_ONNX_PATH,  # 输出路径
        export_params=True,  # 将训练好的权重参数存储在 ONNX 文件内
        opset_version=11,  # ONNX 算子集版本，11 兼容性极佳
        do_constant_folding=True,  # 执行常量折叠优化
        input_names=['input_iq'],  # 指定输入节点的名称
        output_names=['output_prob'],  # 指定输出节点的名称
        dynamic_axes={  # 允许动态的 Batch Size
            'input_iq': {0: 'batch_size'},
            'output_prob': {0: 'batch_size'}
        }
    )

    print(f"✅ 成功！ONNX 模型已保存至: {OUTPUT_ONNX_PATH}")
    print("提示: 你可以使用 Netron (https://netron.app/) 可视化查看这个 ONNX 文件的网络结构。")


if __name__ == '__main__':
    export_to_onnx()
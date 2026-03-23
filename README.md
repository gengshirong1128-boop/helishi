# helishi-Knowledge Distillation

项目目录：D:\helishi1\helishi-Knowledge Distillation

简要说明

这是耿世荣（作者学号 24201172）为 RML2016.10a 数据集实现的知识蒸馏 (Knowledge Distillation) 项目集合。总体思路是：训练一个高精度的“教师”模型（PET + Transformer），然后使用数据增强（MixUp）与软标签蒸馏将知识迁移到一个轻量级的“学生”模型（EdgeCNN / Res-Edge），以便部署到边缘设备。该文件夹包含训练、评估、导出 ONNX、以及绘图/结果保存等完整流程脚本。

先决条件

- Python 3.8+（推荐 3.10+）
- 推荐使用 GPU（支持 CUDA）以加速训练
- 主要依赖（可用项目根目录的 requirements.txt 或自行安装）：
  - torch
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

（仓库根目录中如有 `requirements.txt`，请优先使用；否则按上面列表安装）

目录与文件说明（逐项）

- `data/`
  - 存放 RML2016.10a 数据集（示例文件名：`RML2016.10a_dict.pkl`）。数据以字典形式存储，键为 `(modulation, snr)`。

- `data_loader.py`
  - 用途：读取并解析 RML 数据集，封装为 PyTorch Dataset（`RadioSigDataset`），并实现简单的物理数据增强（相位扰动）。
  - 输入：`data/RML2016.10a_dict.pkl`
  - 输出：NumPy/Tensor 数据、Dataset 实例
  - 关联：被 `main.py`、训练脚本（局部复制）等用于构造 DataLoader。

- `model.py`
  - 用途：提供教师或基准 Transformer-CNN 混合模型（轻/中等规模），在代码中有说明为高性能教师的变体。
  - 关键类：`TransformerModel`（接受 shape (Batch, 2, 128) 或 (Batch,128,2) 输入）
  - 关联：`train_teacher_supreme.py`、`eval_teacher.py` 等脚本加载该模型结构并加载/保存权重。

- `model_teacher_arch.py`
  - 用途：教师模型另一实现（含位置编码、PET+Transformer 风格），用于训练/评估更强的教师模型。
  - 关键类：`PositionalEncoding`, `TransformerModel`（返回分类 logits）
  - 关联：部分脚本在导入时尝试从 `model.py` 或 `model_teacher_arch.py` 获取教师架构。

- `model_edge.py`
  - 用途：学生（轻量级）模型实现，包含 SEAttention（Squeeze-Excitation）与残差块，适合边缘部署。
  - 关键类：`EdgeCNN`、`ResBlock1D`、`SEBlock1D`。
  - 关联：`train_kd.py`（训练学生），`main.py`（加载学生权重评测），`export_onnx.py`（导出 ONNX，注意该脚本导入的是 `model_teacher_arch.EdgeCNN` 并假定权重文件名）。

- `train_teacher_supreme.py`
  - 用途：训练高精度教师模型（Transformer 风格）。包含训练循环、Label Smoothing、AdamW、余弦退火学习率、以及训练后对测试集按 SNR 绘制论文级别曲线图与混淆矩阵并保存 `prediction_result/`。
  - 输入：`data/RML2016.10a_dict.pkl`
  - 输出：教师模型权重（默认 `model_data/teacher_supreme.pth`）和评估图（`prediction_result/paper_teacher_snr_curve.pdf`、`paper_teacher_confusion_matrix.pdf`）。

- `train_kd.py`
  - 用途：知识蒸馏训练流程，将教师输出的软标签与 MixUp 混合数据一起用于训练学生模型。训练过程中保存表现最好的学生权重（默认 `model_data/model_student_kd.pth`）。
  - 关键技术点：MixUp 数据增强、温度缩放 (temperature) 的 KL 散度损失与硬标签损失的加权（alpha）融合、梯度裁剪、余弦退火。
  - 输入：`data/RML2016.10a_dict.pkl`、教师权重（`model_data/teacher_supreme.pth`）
  - 输出：学生权重（`model_data/model_student_kd.pth`）。

- `eval_teacher.py`
  - 用途：加载训练好的教师模型（`model_data/teacher_supreme.pth`）并对测试集进行按 SNR 的性能拆解与绘图（与 `train_teacher_supreme.py` 的评估部分类似，但单独抽离为评估工具）。
  - 输出：`prediction_result/paper_teacher_snr_curve.pdf`、`paper_teacher_confusion_matrix.pdf`。

- `export_onnx.py`
  - 用途：将训练好的学生模型导出为 ONNX 格式以便边缘部署。注意脚本中假定了权重文件名为 `model_data/model_student_edge.pth`，导出为 `model_data/model_edge_deploy.onnx`。
  - 输入：`model_data/model_student_edge.pth`（请根据实际训练产物调整或重命名）
  - 输出：ONNX 文件 `model_edge_deploy.onnx`

- `main.py`
  - 用途：对已训练的学生模型（默认 `model_data/model_student_kd.pth`）进行按 SNR 的最终评测并保存 JSON 格式结果（`prediction_result/test_result.json`），同时生成高质量 PNG/PDF SNR 曲线图。
  - 输入：`data/RML2016.10a_dict.pkl`、`model_data/model_student_kd.pth`
  - 输出：`prediction_result/test_result.json`、论文级曲线图（PNG/PDF）。

- `utils.py`
  - 用途：工具函数集合（固定随机种子、智能路径识别、生成序号文件名等）。被 `main.py` 等调用以提高可复现性与自动化文件命名。

- `model_data/`
  - 用途：默认保存/读取模型权重（示例：`teacher_supreme.pth`, `model_student_kd.pth`, 还有已有的 `model.pth`、`model.onnx` 等）。

- `prediction_result/`
  - 用途：保存评估结果、图表与 JSON。示例：`test_result.json`、`paper_teacher_snr_curve.pdf`。

- `Model_Compare/`
  - 用途：项目中用于保存模型对比或实验比较脚本/结果（视具体文件而定）。

快速开始（示例命令，PowerShell）

1) 安装依赖（示例）

```powershell
python -m pip install -r requirements.txt
# 若没有 requirements.txt，请至少安装：
python -m pip install torch numpy scikit-learn matplotlib seaborn
```

2) 训练教师模型（生成 `model_data/teacher_supreme.pth`）

```powershell
python train_teacher_supreme.py
```

3) 训练知识蒸馏的学生模型（需教师权重存在）

```powershell
python train_kd.py
```

4) 使用训练好的学生模型做最终评估并生成论文图表

```powershell
python main.py
# 结果 JSON: prediction_result/test_result.json
# 图像: prediction_result/paper_snr_curve_*.png / .pdf
```

5) 导出 ONNX（注意：脚本默认寻找 model_student_edge.pth，若训练脚本保存为其它文件，请先重命名或修改脚本）

```powershell
python export_onnx.py
```

文件间关系（文本流程）

数据准备：
  data/RML2016.10a_dict.pkl --> data_loader.py / 内部读取函数

训练教师：
  train_teacher_supreme.py --> 使用 `model.py` 中的 TransformerModel，保存 `model_data/teacher_supreme.pth`。

蒸馏训练学生：
  train_kd.py --> 读取 `teacher_supreme.pth`（作为固定的 teacher），训练 `EdgeCNN` 学生并保存 `model_data/model_student_kd.pth`。

最终评估：
  main.py --> 读取 `model_student_kd.pth` 对每个 SNR 做拆解并生成 JSON 与图表（保存到 `prediction_result/`）。

导出与部署：
  export_onnx.py --> 读取学生权重并导出 ONNX（`model_edge_deploy.onnx`），用于 Netron 查看或推理部署。

常见问题与建议

- 找不到数据文件(`RML2016.10a_dict.pkl`)：请确认 `data/` 下有该 pkl 文件，或者使用 `utils.get_smart_path` 调整相对路径。
- 加载教师/学生权重失败：检查 `model_data/` 下是否存在对应的 .pth 文件；文件名与脚本中定义的变量需一致或修改脚本中常量。
- ONNX 导出失败：确认脚本中 `INPUT_MODEL_PATH` 与实际学生权重文件名一致，并确保模型在 CPU 上可加载；若模型含有仅 GPU 的张量，先 map_location='cpu' 加载。
- 性能不达标：可尝试更长的训练轮数、调大学习率或改变 MixUp 的 alpha/temperature 超参数。

建议的后续改进（可选）

- 将导出权重的路径/文件名统一到配置文件或命令行参数，减少手动修改脚本的需求。
- 增加 `requirements.txt` 或 `environment.yml` 以便环境复现。
- 为关键脚本添加 CLI 参数（argparse），例如数据路径、模型保存路径、batch size、epoch 数等。

作者 & 致谢

作者：耿世荣（学号：24201172）

参考：RML2016.10a 数据集与若干论文中提出的 MixUp、知识蒸馏、Transformer/SE 模块实现思路。


---

如果你希望，我可以：
- 把 README 中的命令改为更完整的 CLI 示例（包含常用参数）；
- 把 export_onnx.py 里默认的权重文件名同步为 `model_student_kd.pth` 并保存脚本小改动；
- 或者为每个脚本补充 argparse 支持和更友好的日志输出。


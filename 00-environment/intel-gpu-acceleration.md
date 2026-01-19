# Intel Ultra 9 185H GPU/NPU 加速指南

你的处理器拥有强大的 AI 加速能力！

## 硬件能力

### Intel Ultra 9 185H 包含：
- **NPU**: AI 专用加速器（支持 INT8/FP16）
- **Intel Arc GPU**: Xe 架构集成显卡
- **CPU**: 16 核心（6P+8E+2LP）

## 加速方案对比

| 方案 | 速度提升 | 配置难度 | 推荐度 | 说明 |
|------|---------|---------|--------|------|
| **OpenVINO** | 2-4x | ⭐⭐ | ⭐⭐⭐⭐⭐ | Intel 官方，支持 NPU+GPU |
| **DirectML** | 1.5-3x | ⭐ | ⭐⭐⭐⭐ | Windows 原生，简单 |
| **Intel Extension for PyTorch** | 2-3x | ⭐⭐⭐ | ⭐⭐⭐ | PyTorch 扩展 |

**推荐**: 先用 **OpenVINO**（最适合 YOLO 推理）

---

## 方案 1: OpenVINO（推荐）

### 优势
- ✅ Intel 官方优化
- ✅ 支持 NPU + GPU + CPU
- ✅ 针对推理优化，速度最快
- ✅ YOLO 原生支持

### 安装步骤

#### 1. 安装 OpenVINO
```bash
# 激活环境
activate.bat

# 安装 OpenVINO
pip install openvino openvino-dev
```

#### 2. 转换 YOLO 模型
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 导出为 OpenVINO 格式
model.export(format='openvino')
# 生成文件: yolov8n_openvino_model/
```

#### 3. 使用 OpenVINO 模型
```python
from ultralytics import YOLO

# 使用 OpenVINO 模型
model = YOLO('yolov8n_openvino_model', task='detect')

# 推理（自动使用 GPU/NPU）
results = model('image.jpg')
```

#### 4. 指定设备
```python
# 默认自动选择最优设备
model = YOLO('yolov8n_openvino_model')

# 或手动指定：
# CPU: device='CPU'
# GPU: device='GPU'
# NPU: device='NPU' (如果支持)
results = model('image.jpg', device='GPU')
```

---

## 方案 2: DirectML（最简单）

### 优势
- ✅ Windows 原生支持
- ✅ 一行命令安装
- ✅ 自动使用所有 GPU

### 安装步骤

#### 1. 安装 PyTorch DirectML
```bash
activate.bat
pip install torch-directml
```

#### 2. 使用 DirectML
```python
import torch
import torch_directml

# 使用 DirectML 设备
dml = torch_directml.device()

# YOLO 使用 DirectML
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# 推理时指定设备
results = model('image.jpg', device=dml)
```

---

## 方案 3: Intel Extension for PyTorch

### 优势
- ✅ 完整的 PyTorch 功能
- ✅ 支持训练和推理
- ✅ Intel GPU 优化

### 安装步骤

#### 1. 安装扩展
```bash
activate.bat
pip install intel-extension-for-pytorch
```

#### 2. 使用 Intel GPU
```python
import torch
import intel_extension_for_pytorch as ipex

# 检查 Intel GPU
if torch.xpu.is_available():
    print(f"Intel GPU 可用: {torch.xpu.get_device_name(0)}")
    device = 'xpu'
else:
    device = 'cpu'

# YOLO 使用 Intel GPU
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('image.jpg', device=device)
```

---

## 性能对比测试脚本

创建 `benchmark.py` 测试不同方案：

```python
import time
from ultralytics import YOLO

def benchmark(model_path, device, runs=10):
    model = YOLO(model_path)

    # 预热
    model('test_image.jpg', device=device, verbose=False)

    # 测试
    start = time.time()
    for _ in range(runs):
        model('test_image.jpg', device=device, verbose=False)
    elapsed = time.time() - start

    fps = runs / elapsed
    print(f"{device:15s}: {fps:.2f} FPS ({elapsed/runs*1000:.1f} ms/frame)")

# 测试不同配置
print("性能测试 (YOLOv8n):")
benchmark('yolov8n.pt', 'cpu')

# 如果安装了 OpenVINO
try:
    benchmark('yolov8n_openvino_model', 'AUTO')  # 自动选择设备
except:
    print("OpenVINO 模型未找到")

# 如果安装了 DirectML
try:
    import torch_directml
    dml = torch_directml.device()
    benchmark('yolov8n.pt', dml)
except:
    print("DirectML 未安装")
```

---

## 快速开始：推荐配置（OpenVINO）

### 完整步骤

```bash
# 1. 激活环境
activate.bat

# 2. 安装 OpenVINO
pip install openvino

# 3. 进入项目目录
cd 02-computer-vision\object-detection

# 4. 转换模型（首次运行）
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"

# 5. 创建优化的检测脚本
```

创建 `run_detection_gpu.py`:
```python
from ultralytics import YOLO

# 使用 OpenVINO 优化模型
model = YOLO('yolov8n_openvino_model')

# 检测（自动使用 GPU/NPU）
results = model('f3e4e77f95542450a9f61163012d9204.png')

# 显示结果
person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
print(f"检测到 {person_count} 个人")

# 保存结果
results[0].save('result_gpu.jpg')
print("结果已保存: result_gpu.jpg")
```

运行：
```bash
python run_detection_gpu.py
```

---

## 预期性能提升

基于 Intel Ultra 9 185H：

| 场景 | CPU (FPS) | OpenVINO GPU (FPS) | 提升 |
|------|-----------|-------------------|------|
| YOLOv8n (640x640) | ~15 FPS | ~40-60 FPS | 3-4x |
| YOLOv8s | ~8 FPS | ~25-35 FPS | 3-4x |
| 实时摄像头 (480p) | ~20 FPS | ~50-70 FPS | 3x |

---

## NPU 支持

Intel Ultra 9 185H 的 NPU 目前：
- ✅ 支持：Windows Studio Effects, AI 照片增强
- ⚠️ PyTorch/YOLO: 需要特定驱动和 SDK
- 📅 未来：OpenVINO 2024+ 将有更好支持

**当前建议**: 使用 **GPU 加速**（Arc iGPU），性能已经很好！

---

## 故障排除

### Q: OpenVINO 找不到 GPU
A: 更新显卡驱动
```bash
# 检查 GPU 是否可用
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

### Q: DirectML 安装失败
A: 确保 Windows 11 或 Windows 10 最新版本

### Q: 想要最简单的方案
A: 先用 CPU 版本学习，等需要实时处理时再配置 GPU

---

## 总结

**立即可用**: CPU 版本（已安装）
**推荐升级**: OpenVINO GPU（3-4x 速度提升）
**最简单的 GPU**: DirectML（1.5-3x 速度提升）

选择建议：
- 🎓 **学习阶段**: CPU 够用
- 🎥 **实时检测**: 配置 OpenVINO GPU
- 🚀 **最佳性能**: OpenVINO + 最新驱动

---

需要我帮你配置 GPU 加速吗？

# GPU 加速快速指南

## 🚀 已配置完成

### 安装的内容
- ✅ OpenVINO (Intel 官方 AI 加速工具包)
- ✅ GPU 优化的 YOLO 模型转换器
- ✅ GPU 加速检测脚本

### 创建的脚本
1. **detect_gpu.py** - GPU 加速图片检测
2. **webcam_gpu.py** - GPU 加速实时摄像头检测
3. **benchmark.py** - 性能对比测试

---

## 📝 使用步骤

### 1. 激活环境
```bash
cd C:\Users\zhaow\Desktop\Machine Learning
activate.bat
cd 02-computer-vision\object-detection
```

### 2. GPU 加速图片检测
```bash
python detect_gpu.py
```

**首次运行会自动**：
- 转换模型为 OpenVINO 格式（需要 1-2 分钟）
- 检测可用的 GPU
- 使用 GPU 进行推理

**输出结果**：
- 检测结果保存为 `result_gpu.jpg`
- 显示 FPS 和推理时间

### 3. GPU 加速实时摄像头
```bash
python webcam_gpu.py
```

**按键说明**：
- `q` - 退出
- `s` - 保存截图
- `空格` - 暂停/继续

### 4. 性能对比测试
```bash
python benchmark.py
```

**测试内容**：
- CPU (PyTorch) 性能
- OpenVINO AUTO 性能
- OpenVINO GPU 性能（如果可用）

---

## 📊 预期性能

基于 Intel Ultra 9 185H + Intel Arc GPU：

| 场景 | CPU | OpenVINO GPU | 提升 |
|------|-----|--------------|------|
| 图片检测 (640x640) | ~23 FPS | ~60-80 FPS | 3-4x |
| 实时摄像头 (480p) | ~20 FPS | ~50-70 FPS | 3x |
| YOLOv8n 推理时间 | ~43 ms | ~12-16 ms | 3x |

---

## 🔧 设备检测

检查 OpenVINO 可用设备：

```python
from openvino.runtime import Core

ie = Core()
devices = ie.available_devices
print("可用设备:", devices)

# 通常会显示: ['CPU', 'GPU']
# GPU 就是你的 Intel Arc 核显
```

---

## 💡 优化建议

### 模型选择
- **YOLOv8n**: 最快，适合实时应用（推荐）
- **YOLOv8s**: 精度更高，速度稍慢
- **YOLOv8m**: 高精度，需要更强 GPU

### 分辨率调整
```python
# 降低分辨率以提高速度
results = model(image, imgsz=320)  # 默认 640
```

### 批处理
```python
# 批量处理多张图片更高效
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model(images)  # 批量推理
```

---

## 🐛 故障排除

### GPU 未检测到
**原因**: 驱动未安装或版本过旧

**解决**:
1. 更新 Intel 显卡驱动
2. 访问: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
3. 下载并安装最新驱动

### OpenVINO 导入错误
```bash
# 重新安装
pip uninstall openvino openvino-dev
pip install openvino openvino-dev
```

### 模型转换失败
```bash
# 手动转换
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"
```

### 性能没有提升
1. 确认使用的是 OpenVINO 模型（`yolov8n_openvino_model`）
2. 检查设备是否设置为 'GPU' 或 'AUTO'
3. 更新显卡驱动到最新版本

---

## 🎯 使用示例

### 快速检测
```bash
# 最简单的方式
python detect_gpu.py
```

### 查看性能
```bash
# 对比 CPU vs GPU
python benchmark.py
```

### 实时检测
```bash
# 开启摄像头实时检测
python webcam_gpu.py
```

---

## 📈 下一步

1. ✅ 完成 GPU 检测
2. 🔜 尝试更大的模型（yolov8s, yolov8m）
3. 🔜 检测视频文件
4. 🔜 多摄像头同时检测
5. 🔜 人体姿态估计

---

## 🆚 CPU vs GPU 对比

运行 `python benchmark.py` 后，你会看到类似：

```
配置                 推理时间         FPS        提升
------------------------------------------------------------
CPU (PyTorch)        43.2 ms        23.1       -
OpenVINO AUTO        15.8 ms        63.3       2.73x
OpenVINO GPU         12.5 ms        80.0       3.46x
------------------------------------------------------------
最快配置: OpenVINO GPU
性能提升: 3.46x
```

---

开始测试吧！🚀

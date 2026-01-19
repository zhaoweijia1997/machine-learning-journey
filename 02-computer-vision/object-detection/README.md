# YOLOv8 实时人形检测 / YOLOv8 Real-time Person Detection

[English](#english) | [中文](#中文)

---

## 中文

基于 YOLOv8 + OpenVINO 的实时人形检测项目，支持 CPU、GPU、NPU 多种推理设备。

## ✨ 功能特性

- 🎯 **多设备支持**: CPU / GPU (Intel Arc) / NPU (Intel AI Boost)
- 🚀 **GPU 加速**: 使用 OpenVINO 优化，GPU 性能达 27+ FPS
- 📸 **实时检测**: 支持摄像头实时人形检测
- 🖼️ **图片检测**: 支持静态图片批量检测
- 📊 **性能测试**: 内置多设备性能对比工具
- ⌨️ **交互控制**: ESC/q 退出、s 截图、空格暂停

## 🎥 性能表现

| 设备 | 图片推理 | 实时摄像头 | 相对速度 |
|------|---------|-----------|---------|
| **GPU** (Intel Arc) | 84.9 FPS | 27 FPS | 2.86x ⭐ 推荐 |
| **NPU** (AI Boost) | 79.3 FPS | 19 FPS | 2.67x |
| **CPU** (Ultra 9) | 29.7 FPS | 16 FPS | 1.00x |

> **注**: 图片推理为纯推理性能，实时摄像头包含完整 pipeline（读取+推理+后处理+显示）

## 📦 环境要求

- Python 3.14+
- Intel Core Ultra 处理器（支持 GPU/NPU 加速）
- Windows 10/11
- 摄像头（用于实时检测）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 激活虚拟环境
venv\Scripts\activate

# 已安装的包：
# - ultralytics 8.4.6
# - openvino 2025.4.1
# - opencv-python 4.13.0.90
# - torch 2.9.1+cpu
```

### 2. 运行检测

#### 图片检测（推荐新手）

```bash
# GPU 加速检测
python detect_gpu.py

# 纯 OpenVINO 推理（性能测试）
python detect_openvino.py
```

#### 实时摄像头检测

```bash
# GPU 版本（推荐，最快）
python webcam_gpu.py

# NPU 版本（体验 Intel AI Boost）
python webcam_npu_direct.py

# CPU 版本
python webcam_cpu.py
```

#### 性能对比测试

```bash
# 测试所有设备性能
python test_npu.py
```

## ⌨️ 控制说明

所有摄像头检测程序支持以下快捷键：

- **ESC 或 q** - 退出程序
- **s** - 保存当前帧截图
- **空格** - 暂停/继续检测
- **+/-** - 调整置信度阈值（部分版本）

## 🐛 常见问题

### 窗口无法关闭？

使用紧急关闭脚本：

```bash
# 强制关闭所有 Python 进程
kill_webcam.bat
```

或手动关闭：

```bash
taskkill /F /IM python.exe
```

### 摄像头无法打开？

1. 检查摄像头是否被其他程序占用
2. 检查 Windows 隐私设置中的摄像头权限
3. 尝试重启系统

### NPU 性能不如预期？

NPU (Intel AI Boost) 主要针对低功耗场景优化（如视频会议背景虚化），对于 YOLO 这类高吞吐量任务，**GPU 是更好的选择**。

## 📁 项目文件

### 核心检测脚本

- `detect_gpu.py` - GPU 加速图片检测
- `detect_openvino.py` - OpenVINO 直接推理
- `webcam_gpu.py` - GPU 实时摄像头检测 ⭐
- `webcam_npu_direct.py` - NPU 实时检测
- `webcam_cpu.py` - CPU 实时检测
- `test_npu.py` - 多设备性能对比

### 工具脚本

- `kill_webcam.bat` - 紧急关闭脚本

### 模型文件

- `yolov8n.pt` - YOLOv8 nano PyTorch 模型
- `yolov8n_openvino_model/` - OpenVINO 优化模型（首次运行自动生成）

## 🔧 技术细节

### 硬件加速

- **GPU**: Intel Arc iGPU，通过 OpenVINO GPU 插件加速
- **NPU**: Intel AI Boost (集成 NPU)，适合低功耗 AI 任务
- **CPU**: Intel Core Ultra 9 185H，16 核心

### 模型优化

使用 OpenVINO 将 YOLOv8 模型转换为 IR 格式：

```python
model = YOLO('yolov8n.pt')
model.export(format='openvino', half=False)
```

OpenVINO 会自动针对 Intel 硬件优化模型结构和推理流程。

## 📊 检测能力

基于 COCO 数据集预训练，可检测 80 类物体，包括：

- **人** (person) - 主要检测目标
- 交通工具：car, bicycle, bus, truck, motorcycle
- 动物：dog, cat, bird, horse
- 日常物品：chair, bottle, laptop, phone

## 🎯 推荐使用方案

| 场景 | 推荐设备 | 脚本 |
|------|---------|------|
| 日常使用 | GPU | `webcam_gpu.py` |
| 性能测试 | ALL | `test_npu.py` |
| 图片批处理 | GPU | `detect_gpu.py` |
| 低功耗场景 | NPU | `webcam_npu_direct.py` |

## 📝 更新日志

### v1.0.0 (2026-01-20)

**首次发布**

- ✅ 实现 CPU/GPU/NPU 多设备支持
- ✅ GPU 加速达到 27 FPS 实时检测
- ✅ 修复 Windows 平台窗口关闭问题
- ✅ 添加性能对比测试工具
- ✅ 完善错误处理和用户提示
- ✅ 隐私保护：个人图片/视频不上传 GitHub

## 📄 许可证

本项目仅供学习使用。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel 推理加速工具包
- [OpenCV](https://opencv.org/) - 计算机视觉库

---

## English

Real-time person detection project based on YOLOv8 + OpenVINO, supporting CPU, GPU, and NPU inference devices.

## ✨ Features

- 🎯 **Multi-device Support**: CPU / GPU (Intel Arc) / NPU (Intel AI Boost)
- 🚀 **GPU Acceleration**: Optimized with OpenVINO, achieving 27+ FPS
- 📸 **Real-time Detection**: Live webcam person detection
- 🖼️ **Image Detection**: Batch image processing support
- 📊 **Performance Benchmarking**: Built-in multi-device comparison tool
- ⌨️ **Interactive Controls**: ESC/q to quit, s for screenshot, space to pause

## 🎥 Performance

| Device | Image Inference | Real-time Webcam | Speedup |
|--------|----------------|------------------|---------|
| **GPU** (Intel Arc) | 84.9 FPS | 27 FPS | 2.86x ⭐ Recommended |
| **NPU** (AI Boost) | 79.3 FPS | 19 FPS | 2.67x |
| **CPU** (Ultra 9) | 29.7 FPS | 16 FPS | 1.00x |

> **Note**: Image inference is pure inference performance, real-time webcam includes full pipeline (capture + inference + post-processing + display)

## 📦 Requirements

- Python 3.14+
- Intel Core Ultra processor (with GPU/NPU support)
- Windows 10/11
- Webcam (for real-time detection)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
venv\Scripts\activate

# Installed packages:
# - ultralytics 8.4.6
# - openvino 2025.4.1
# - opencv-python 4.13.0.90
# - torch 2.9.1+cpu
```

### 2. Run Detection

#### Image Detection (Recommended for beginners)

```bash
# GPU accelerated detection
python detect_gpu.py

# Pure OpenVINO inference (performance testing)
python detect_openvino.py
```

#### Real-time Webcam Detection

```bash
# GPU version (Recommended, fastest)
python webcam_gpu.py

# NPU version (Experience Intel AI Boost)
python webcam_npu_direct.py

# CPU version
python webcam_cpu.py
```

#### Performance Benchmark

```bash
# Test all devices
python test_npu.py
```

## ⌨️ Controls

All webcam detection programs support:

- **ESC or q** - Exit program
- **s** - Save screenshot
- **Space** - Pause/Resume
- **+/-** - Adjust confidence threshold (some versions)

## 🐛 Common Issues

### Window Won't Close?

Use emergency shutdown script:

```bash
# Force kill all Python processes
kill_webcam.bat
```

Or manually:

```bash
taskkill /F /IM python.exe
```

### Camera Won't Open?

1. Check if camera is used by another program
2. Check camera permissions in Windows Privacy Settings
3. Try restarting the system

### NPU Performance Lower Than Expected?

NPU (Intel AI Boost) is optimized for low-power scenarios (e.g., video call background blur). For high-throughput tasks like YOLO, **GPU is the better choice**.

## 📁 Project Files

### Core Detection Scripts

- `detect_gpu.py` - GPU accelerated image detection
- `detect_openvino.py` - OpenVINO direct inference
- `webcam_gpu.py` - GPU real-time webcam detection ⭐
- `webcam_npu_direct.py` - NPU real-time detection
- `webcam_cpu.py` - CPU real-time detection
- `test_npu.py` - Multi-device performance comparison

### Utility Scripts

- `kill_webcam.bat` - Emergency shutdown script

### Model Files

- `yolov8n.pt` - YOLOv8 nano PyTorch model
- `yolov8n_openvino_model/` - OpenVINO optimized model (auto-generated on first run)

## 🔧 Technical Details

### Hardware Acceleration

- **GPU**: Intel Arc iGPU, accelerated via OpenVINO GPU plugin
- **NPU**: Intel AI Boost (integrated NPU), suitable for low-power AI tasks
- **CPU**: Intel Core Ultra 9 185H, 16 cores

### Model Optimization

Converting YOLOv8 model to OpenVINO IR format:

```python
model = YOLO('yolov8n.pt')
model.export(format='openvino', half=False)
```

OpenVINO automatically optimizes model structure and inference pipeline for Intel hardware.

## 📊 Detection Capabilities

Pre-trained on COCO dataset, can detect 80 object classes including:

- **Person** (person) - Primary detection target
- Vehicles: car, bicycle, bus, truck, motorcycle
- Animals: dog, cat, bird, horse
- Daily objects: chair, bottle, laptop, phone

## 🎯 Recommended Usage

| Scenario | Recommended Device | Script |
|----------|-------------------|--------|
| Daily Use | GPU | `webcam_gpu.py` |
| Performance Test | ALL | `test_npu.py` |
| Batch Processing | GPU | `detect_gpu.py` |
| Low-power Scenario | NPU | `webcam_npu_direct.py` |

## 📝 Changelog

### v1.0.0 (2026-01-20)

**Initial Release**

- ✅ CPU/GPU/NPU multi-device support
- ✅ GPU acceleration achieving 27 FPS real-time detection
- ✅ Fixed Windows window closing issue
- ✅ Added performance comparison tool
- ✅ Improved error handling and user prompts
- ✅ Privacy protection: personal images/videos not uploaded to GitHub

## 📄 License

This project is for educational purposes only.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel inference acceleration toolkit
- [OpenCV](https://opencv.org/) - Computer vision library

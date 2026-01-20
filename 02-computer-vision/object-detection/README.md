# 🎯 YOLOv8 目标检测

> 基于 YOLOv8 + OpenVINO 的实时目标检测，支持 GPU/NPU 加速

## ✨ 功能特性

- 🚀 **多设备加速**: CPU / GPU (Intel Arc) / NPU (AI Boost)
- 📸 **实时检测**: 摄像头 + 屏幕捕获
- 🎯 **80类物体**: COCO 数据集预训练

## 📁 目录结构

```
object-detection/
├── apps/                    # 🚀 主要应用程序
│   ├── screen_simple.py     # 屏幕实时检测（推荐）
│   ├── screen_tray_simple.py # 系统托盘版本
│   └── webcam_screen.py     # 摄像头检测
│
├── scripts/                 # 🔧 辅助脚本
│   ├── detect_gpu.py        # GPU 图片检测
│   ├── webcam_gpu.py        # GPU 摄像头
│   ├── webcam_npu_direct.py # NPU 摄像头
│   └── ...
│
├── models/                  # 📦 模型文件
│   ├── yolov8n.pt          # YOLOv8 nano
│   ├── yolov8m.pt          # YOLOv8 medium
│   └── openvino/           # OpenVINO 优化模型
│
├── tests/                   # 🧪 测试文件
│   └── benchmark.py         # 性能测试
│
└── docs/                    # 📖 文档
    ├── QUICK_START.md
    └── GPU_QUICKSTART.md
```

## 🚀 快速开始

```bash
# 激活环境
cd "Machine Learning"
venv\Scripts\activate

# 运行屏幕检测
cd 02-computer-vision/object-detection
python apps/screen_simple.py
```

## 🎥 性能表现

| 设备 | 实时帧率 | 推荐场景 |
|------|---------|---------|
| **GPU** | 27 FPS | ⭐ 日常使用 |
| **NPU** | 19 FPS | 低功耗 |
| **CPU** | 16 FPS | 兼容性 |

## ⌨️ 快捷键

- **ESC / Q** - 退出
- **S** - 截图保存
- **空格** - 暂停/继续

## 📖 更多文档

- [快速入门指南](docs/QUICK_START.md)
- [GPU 加速配置](docs/GPU_QUICKSTART.md)

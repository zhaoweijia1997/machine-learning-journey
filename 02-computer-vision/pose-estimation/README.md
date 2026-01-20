# 🏃 人体姿态估计

> GPU + NPU 加速的实时人体姿态检测，17个关键点追踪

## ✨ 功能特性

- 🦴 **17个关键点**: 头部、四肢、躯干完整骨架
- 🚀 **双加速器**: GPU + NPU 并行处理
- 📺 **多模式**: 摄像头 / 屏幕捕获

## 📁 目录结构

```
pose-estimation/
├── pose_app.pyw        # 🚀 GUI 应用（推荐）
├── 启动姿态估计.bat     # 快捷启动
│
├── scripts/            # 🔧 命令行版本
│   ├── pose_gpu_npu.py # GPU+NPU 并行版
│   ├── pose_gpu.py     # GPU 版本
│   └── pose_webcam_gpu.py
│
└── models/             # 📦 模型文件
    ├── yolov8n-pose.pt
    └── openvino/
```

## 🚀 快速开始

### 方式一：GUI 界面（推荐）

```bash
# 双击运行
pose_app.pyw

# 或命令行
python pose_app.pyw
```

### 方式二：命令行

```bash
# GPU+NPU 并行版本
python scripts/pose_gpu_npu.py --mode screen --monitor 0

# 摄像头模式
python scripts/pose_gpu_npu.py --mode camera
```

## ⌨️ 快捷键

| 按键 | 功能 |
|------|------|
| **C** | 切换到摄像头 |
| **1-9** | 切换到屏幕 1-9 |
| **ESC/Q** | 退出 |
| **S** | 保存截图 |

## 🎥 性能表现

| 指标 | 数值 |
|------|------|
| GPU 推理 | ~14ms (70+ FPS) |
| NPU 推理 | ~22ms (异步) |
| 实际帧率 | 8-30 FPS |

## 🦴 17个关键点

```
头部: 0-鼻子, 1/2-眼睛, 3/4-耳朵
上肢: 5/6-肩膀, 7/8-肘部, 9/10-手腕
躯干: 11/12-臀部
下肢: 13/14-膝盖, 15/16-脚踝
```

## 💡 应用场景

- 健身动作检测
- 体育动作分析
- 人机交互
- 康复训练监控

# 🤖 机器学习学习之旅

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.0-purple.svg)](https://docs.openvino.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 从零开始学习机器学习和计算机视觉，专为 Intel 平台优化

## ✨ 项目亮点

| 功能 | 描述 | 技术栈 |
|------|------|--------|
| 🎯 **目标检测** | 实时检测80+类物体 | YOLOv8 + OpenVINO |
| 🏃 **姿态估计** | 17个人体关键点追踪 | GPU/NPU 并行加速 |
| 👤 **人脸识别** | 人脸注册与实时识别 | OpenVINO 预训练模型 |
| 📚 **深度学习课程** | 从神经元开始的入门教程 | 交互式可视化 |

## 🖥️ 硬件环境

- **处理器**: Intel Ultra 9 185H
- **集成显卡**: Intel Arc Graphics (支持 AI 加速)
- **推荐**: 任何支持 OpenVINO 的 Intel 平台

---

## 🚀 快速开始

### 方式一：一键安装（推荐）

```bash
# 克隆项目
git clone https://github.com/zhaoweijia1997/machine-learning-journey.git
cd machine-learning-journey

# Windows 用户
setup.bat

# Linux/Mac 用户
chmod +x setup.sh && ./setup.sh
```

### 方式二：手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 📁 项目结构

```
Machine Learning/
│
├── 📂 00-environment/          # 🔧 环境配置指南
│   ├── README.md               # 环境概述
│   ├── intel-gpu-acceleration.md  # Intel GPU 加速配置
│   └── github-setup.md         # GitHub 配置
│
├── 📂 01-basics/               # 📖 基础入门
│   └── README.md
│
├── 📂 02-computer-vision/      # 👁️ 计算机视觉项目
│   ├── object-detection/       # 🎯 目标检测 (YOLOv8)
│   ├── pose-estimation/        # 🏃 姿态估计
│   └── face-recognition/       # 👤 人脸识别
│
├── 📂 03-deep-learning/        # 🧠 深度学习从零开始
│   └── 01-neural-network-basics/  # 第1课：神经元
│
└── 📂 utils/                   # 🛠️ 工具脚本
```

---

## �� 功能演示

### 目标检测
```bash
cd 02-computer-vision/object-detection
python screen_simple.py          # 屏幕实时检测
```

### 姿态估计
```bash
cd 02-computer-vision/pose-estimation
python pose_app.pyw              # GUI 应用
```

### 人脸识别
```bash
cd 02-computer-vision/face-recognition
python face_app.pyw              # GUI 应用（支持人脸注册）
```

### 深度学习课程
```bash
# Python 教程
python 03-deep-learning/01-neural-network-basics/lesson1_neuron.py

# 或者打开交互式网页
# 用浏览器打开 03-deep-learning/01-neural-network-basics/lesson1_interactive.html
```

---

## 📚 学习路线

### 🟢 第一阶段：环境搭建
- [x] Python 3.10+ 安装
- [x] OpenVINO 配置
- [x] GPU 驱动安装

### 🟡 第二阶段：计算机视觉实战
- [x] YOLOv8 目标检测
- [x] 人体姿态估计
- [x] 人脸识别系统

### 🔵 第三阶段：深度学习原理
- [x] 第1课：理解神经元
- [ ] 第2课：多层网络
- [ ] 第3课：前向传播
- [ ] 第4课：反向传播

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch, OpenVINO |
| 计算机视觉 | OpenCV, Ultralytics YOLOv8 |
| GUI | Tkinter, PIL |
| 屏幕捕获 | MSS, DXCam |

---

## 📖 详细文档

- [环境配置指南](00-environment/README.md)
- [Intel GPU 加速](00-environment/intel-gpu-acceleration.md)
- [目标检测快速入门](02-computer-vision/object-detection/QUICK_START.md)
- [深度学习课程](03-deep-learning/README.md)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📝 许可证

MIT License - 随意使用，学习愉快！

---

<p align="center">
  <b>⭐ 如果觉得有帮助，请给个 Star！</b>
</p>

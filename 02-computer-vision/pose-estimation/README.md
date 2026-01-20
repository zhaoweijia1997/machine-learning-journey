# YOLOv8 人体姿态估计

GPU + NPU 加速的实时人体姿态检测，支持摄像头和屏幕捕获。

## 功能特点

- 17 个人体关键点检测（头部、四肢、躯干）
- 自动绘制人体骨架
- GPU (Intel Arc) 硬件加速
- NPU (Intel AI Boost) 并行处理
- 支持摄像头和多显示器屏幕捕获

## 文件说明

| 文件 | 说明 | 推荐 |
|------|------|------|
| `pose_app.pyw` | GUI 启动界面版本 | ⭐ 日常使用 |
| `pose_gpu_npu.py` | GPU+NPU 并行版本 | ⭐ 性能测试 |
| `pose_gpu.py` | 命令行版本 | 开发调试 |
| `pose_webcam_gpu.py` | 纯摄像头版本 | 学习入门 |

## 快速开始

### 方式一：GUI 界面（推荐）

双击 `pose_app.pyw` 运行，无需命令行。

### 方式二：命令行

```bash
# GPU+NPU 并行版本（带性能监控）
python pose_gpu_npu.py --mode screen --monitor 0

# 摄像头模式
python pose_gpu_npu.py --mode camera

# 指定参数
python pose_gpu_npu.py --mode screen --monitor 0 --resize 480 --conf 0.5
```

## 按键说明

| 按键 | 功能 |
|------|------|
| C | 切换到摄像头 |
| 1-9 | 切换到屏幕 1-9 |
| ESC/Q | 退出 |
| S | 保存截图 |

## 性能数据

测试环境：Intel Core Ultra 9 185H + Arc iGPU + NPU

| 指标 | 数值 |
|------|------|
| GPU 推理 | ~14ms (70+ fps) |
| NPU 推理 | ~22ms (异步) |
| 实际帧率 | ~8-30 fps |

## 17 个关键点

```
头部: 0-鼻子, 1/2-眼睛, 3/4-耳朵
上肢: 5/6-肩膀, 7/8-肘部, 9/10-手腕
躯干: 11/12-臀部
下肢: 13/14-膝盖, 15/16-脚踝
```

## 依赖

- ultralytics
- opencv-python
- openvino
- mss (屏幕捕获)

## 学习笔记

姿态估计 vs 目标检测：
- 目标检测：检测"人在哪里"（边界框）
- 姿态估计：检测"人在做什么"（关键点+骨架）

应用场景：
- 健身动作检测
- 体育动作分析
- 人机交互
- 康复训练监控

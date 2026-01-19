# 目标检测 - 人形检测项目

## 项目概述

使用 YOLOv8 进行实时人形检测，这是计算机视觉中最常见和实用的应用之一。

## 项目列表

### 1. 基础人形检测
- **文件**: `detect_person_basic.py`
- **功能**: 使用预训练的 YOLOv8 模型检测图片中的人
- **难度**: ⭐ 初级
- **学习重点**:
  - 理解 YOLO 模型的使用
  - 学习图像预处理
  - 可视化检测结果

### 2. 实时视频人形检测
- **文件**: `detect_person_webcam.py`
- **功能**: 使用摄像头进行实时人形检测
- **难度**: ⭐⭐ 中级
- **学习重点**:
  - 视频流处理
  - 实时推理优化
  - FPS 性能监控

### 3. 批量图片人形检测
- **文件**: `detect_person_batch.py`
- **功能**: 批量处理多张图片
- **难度**: ⭐⭐ 中级
- **学习重点**:
  - 批处理优化
  - 结果保存和统计
  - 进度条显示

## 数据集

### COCO 数据集
YOLO 模型预训练在 COCO 数据集上，包含 80 个类别，其中：
- **类别 0**: person（人）

### 自定义数据集（可选）
如果需要训练自己的模型：
- 准备标注数据
- 使用 YOLO 格式标注
- 微调预训练模型

## 快速开始

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # n=nano (最小最快)

# 检测图片
results = model('path/to/image.jpg')

# 显示结果
results[0].show()
```

## 模型选择

YOLOv8 提供多个版本，按速度和精度排序：

| 模型 | 大小 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|----------|
| yolov8n | 6MB | ⚡⚡⚡ | ⭐⭐ | 实时检测，入门学习 |
| yolov8s | 22MB | ⚡⚡ | ⭐⭐⭐ | 平衡速度和精度 |
| yolov8m | 52MB | ⚡ | ⭐⭐⭐⭐ | 更高精度 |
| yolov8l | 87MB | ⚡ | ⭐⭐⭐⭐⭐ | 高精度场景 |
| yolov8x | 136MB | ⚡ | ⭐⭐⭐⭐⭐ | 最高精度 |

**推荐新手**: yolov8n（最快，适合学习和实时应用）

## 性能优化

### Intel Arc GPU 优化
你的 Ultra 9 185H 集成了 Intel Arc GPU，可以使用 OpenVINO 加速：

```python
# 导出为 OpenVINO 格式
model.export(format='openvino')

# 使用 OpenVINO 推理
model = YOLO('yolov8n_openvino_model')
```

### CPU 优化
```python
# 使用更小的模型
model = YOLO('yolov8n.pt')

# 降低输入分辨率
results = model(img, imgsz=320)  # 默认 640
```

## 常见应用场景

1. **安防监控**: 检测特定区域的人员活动
2. **人流统计**: 统计进出人数
3. **社交距离监测**: COVID-19 期间的应用
4. **智能家居**: 人员检测触发设备
5. **体育分析**: 追踪运动员位置

## 下一步学习

完成基础人形检测后，可以进阶到：
- 人体姿态估计（[pose-estimation](../pose-estimation/)）
- 多目标跟踪
- 人脸识别（[face-recognition](../face-recognition/)）
- 人群密度估计

## 参考资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO 数据集](https://cocodataset.org/)
- [OpenVINO 文档](https://docs.openvino.ai/)

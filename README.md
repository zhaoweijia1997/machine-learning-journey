# 机器学习学习之旅

## 硬件环境
- 处理器: Intel Ultra 9 185H
- 集成显卡: Intel Arc Graphics (支持AI加速)

## 学习目标
专注于计算机视觉领域，特别是：
- 人形检测 (Human Detection)
- 目标检测 (Object Detection)
- 人体姿态估计 (Pose Estimation)

## 目录结构

```
Machine Learning/
├── 00-environment/          # 环境配置和依赖管理
├── 01-basics/              # 基础知识和快速入门项目
├── 02-computer-vision/     # 计算机视觉主目录
│   ├── object-detection/   # 目标检测项目
│   ├── pose-estimation/    # 人体姿态估计
│   ├── face-recognition/   # 人脸识别
│   └── segmentation/       # 图像分割
├── datasets/               # 数据集存放目录
├── models/                 # 训练好的模型权重
├── notebooks/              # Jupyter notebooks 实验
├── projects/               # 完整的实战项目
└── utils/                  # 工具脚本和辅助函数
```

## 推荐工具栈

### 深度学习框架
- **PyTorch**: 主流深度学习框架
- **TensorFlow/Keras**: 备选框架
- **OpenVINO**: Intel 优化工具包（充分利用 Intel Arc GPU）

### 计算机视觉库
- **OpenCV**: 计算机视觉基础库
- **YOLO (v8/v9)**: 实时目标检测
- **MediaPipe**: Google 的人体姿态估计方案
- **MMDetection**: 目标检测工具箱

### 开发环境
- Python 3.10+
- Conda/Miniconda（环境管理）
- Jupyter Notebook/Lab

## 快速开始

1. 配置环境（见 [00-environment](00-environment/) 目录）
2. 从基础项目开始（见 [01-basics](01-basics/) 目录）
3. 深入计算机视觉实战（见 [02-computer-vision](02-computer-vision/) 目录）

## 实用工具

项目包含了一些自动化脚本，让 Git 管理更轻松：

### 快速提交（自动清理 + 提交 + 推送）
```bash
# Windows
utils\quick-commit.bat "你的提交信息"

# Linux/Mac
bash utils/quick-commit.sh "你的提交信息"
```

### 清理临时文件
```bash
# Windows
utils\git-clean.bat

# Linux/Mac
bash utils/git-clean.sh
```

详见 [utils/README.md](utils/README.md)

## 学习路线

### 第一阶段：环境搭建
- [ ] 安装 Python 和 Conda
- [ ] 配置 Intel Arc GPU 驱动和 OpenVINO
- [ ] 安装基础库 (NumPy, Pandas, Matplotlib)

### 第二阶段：基础入门
- [ ] Python 机器学习基础
- [ ] OpenCV 图像处理
- [ ] 简单的图像分类项目

### 第三阶段：人形检测实战
- [ ] 使用 YOLO 进行目标检测
- [ ] 人形检测项目实现
- [ ] 实时视频流检测

### 第四阶段：进阶应用
- [ ] 人体姿态估计
- [ ] 多目标跟踪
- [ ] 模型优化和部署

## 资源推荐

### 数据集
- COCO Dataset (人形检测)
- MPII Human Pose Dataset
- CrowdHuman Dataset

### 学习资源
- PyTorch 官方教程
- OpenCV 文档
- Intel OpenVINO 文档

## 进度记录

开始日期: 2026-01-19

---

**注意**: datasets/ 和 models/ 目录中的大文件不会被 Git 追踪，请手动备份重要数据。

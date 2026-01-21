# 🧠 深度学习从零开始

> 从最基础的概念开始，一步步理解深度学习的原理

## 🎯 课程目标

不满足于只会调用 API？想真正理解 AI 的工作原理？这个课程适合你！

我们将从一个最简单的"神经元"开始，逐步构建起完整的知识体系。

---

## 📚 课程目录

### 第1部分：神经网络基础

| 课程 | 内容 | 资源 |
|------|------|------|
| ✅ 第1课 | 理解神经元：权重、偏置、激活函数 | [Python](01-neural-network-basics/lesson1_neuron.py) \| [交互网页](01-neural-network-basics/lesson1_interactive.html) |
| ✅ 第2课 | 多层网络：隐藏层的作用 | [Python](01-neural-network-basics/lesson2_layers.py) \| [交互网页](01-neural-network-basics/lesson2_interactive.html) \| [学习指南](01-neural-network-basics/lesson2_guide.md) |
| ⏳ 第3课 | 前向传播：数据如何流动 | 待续 |
| ⏳ 第4课 | 反向传播与梯度下降：网络如何学习 | 待续 |

### 第2部分：卷积神经网络 (CNN)

| 课程 | 内容 | 资源 |
|------|------|------|
| ⏳ 第5课 | 卷积层原理 | 待续 |
| ⏳ 第6课 | 池化层与特征提取 | 待续 |
| ⏳ 第7课 | 构建图像分类器 | 待续 |

### 第3部分：人脸识别专题

| 课程 | 内容 | 资源 |
|------|------|------|
| ⏳ 第8课 | 人脸检测原理 | 待续 |
| ⏳ 第9课 | 特征向量与嵌入空间 | 待续 |
| ⏳ 第10课 | Triplet Loss 训练 | 待续 |

---

## 🚀 快速开始

### 方式一：交互式网页（推荐新手）

直接用浏览器打开，无需安装任何东西：

```
01-neural-network-basics/lesson1_interactive.html
```

### 方式二：运行 Python 代码

```bash
# 确保已激活虚拟环境
cd 03-deep-learning/01-neural-network-basics
python lesson1_neuron.py
```

---

## 📖 学习方式

每一课都提供两种学习方式：

1. **Python 脚本** - 阅读代码和注释，运行看输出，修改参数做实验
2. **交互网页** - 可视化演示，拖动滑块实时看结果

建议：先看网页理解概念，再读代码加深理解。

---

## 💡 第1课预览：理解神经元

用"要不要带伞"的例子理解神经元：

```
输入:
  - 天气预报说下雨概率 (0-1)
  - 天上有乌云吗 (0 或 1)
  - 地面是湿的吗 (0 或 1)

神经元计算:
  加权求和 → 加偏置 → 激活函数(Sigmoid)

输出:
  - 带伞的概率 (0-1)
```

打开 [交互网页](01-neural-network-basics/lesson1_interactive.html) 亲自体验！

---

## 🛠️ 环境要求

```bash
pip install numpy torch matplotlib
```

或者使用项目根目录的一键安装：

```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh
```

---

## 👥 适合人群

- ✅ 有 Python 基础
- ✅ 想了解 AI/深度学习原理
- ✅ 不满足于只会调用 API
- ✅ 喜欢动手实验

---

<p align="center">
  <b>🎓 学习愉快！</b>
</p>

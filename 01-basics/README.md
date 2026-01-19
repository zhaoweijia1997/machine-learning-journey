# 基础入门

## 学习目标

在正式开始计算机视觉项目前，先掌握这些基础知识。

## 推荐学习路径

### 1. Python 基础回顾（可选）
如果你已经熟悉 Python，可以跳过这一部分。

- 变量和数据类型
- 列表、字典、元组
- 循环和条件语句
- 函数定义
- 类和对象（基础即可）

### 2. NumPy 基础
NumPy 是所有机器学习库的基础。

**必学内容**：
- 数组创建和操作
- 数组索引和切片
- 数组运算
- 广播机制

**快速示例**：
```python
import numpy as np

# 创建数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 基本操作
print(arr.shape)  # (2, 3)
print(arr.mean())  # 3.5

# 切片
print(arr[0, :])  # [1, 2, 3]
```

### 3. OpenCV 图像处理
计算机视觉的核心库。

**必学内容**：
- 读取和显示图像
- 图像格式转换（BGR ↔ RGB）
- 图像缩放和裁剪
- 基本图像操作

**快速示例**：
```python
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 调整大小
resized = cv2.resize(img, (640, 480))

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 4. Matplotlib 数据可视化
用于可视化结果。

**必学内容**：
- 绘制基本图表
- 显示图像
- 子图布局

**快速示例**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘图
plt.plot(x, y)
plt.title('Sin Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

## 实践项目

建议按顺序完成以下小项目来巩固基础：

### 项目 1: 图像读取和显示
- 使用 OpenCV 读取图像
- 显示图像信息（尺寸、通道数等）
- 转换颜色空间
- 保存结果

### 项目 2: 图像基本操作
- 裁剪图像
- 缩放图像
- 旋转图像
- 添加文字和图形

### 项目 3: 图像滤波
- 模糊处理
- 边缘检测
- 阈值处理

### 项目 4: 批量处理
- 批量读取文件夹中的图片
- 统一调整尺寸
- 批量保存

## 学习资源

### 官方文档
- [NumPy 文档](https://numpy.org/doc/)
- [OpenCV Python 教程](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Matplotlib 教程](https://matplotlib.org/stable/tutorials/index.html)

### 推荐教程
- [NumPy 快速入门](https://numpy.org/doc/stable/user/quickstart.html)
- [OpenCV Python 入门](https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html)

## 评估标准

完成基础学习后，你应该能够：

- ✅ 创建和操作 NumPy 数组
- ✅ 读取、显示和保存图像
- ✅ 调整图像大小和裁剪
- ✅ 进行基本的图像处理操作
- ✅ 使用 Matplotlib 可视化结果
- ✅ 理解 BGR vs RGB 颜色空间
- ✅ 批量处理多张图片

## 时间估算

如果你：
- **完全新手**: 1-2 周
- **有 Python 基础**: 3-5 天
- **有编程经验**: 1-2 天

**重要提示**：不需要完全掌握所有细节，了解基本用法即可。在后续项目中边做边学会更高效！

## 下一步

完成基础学习后，直接进入：
- [02-computer-vision/object-detection](../02-computer-vision/object-detection/) - 人形检测项目

我们已经为你准备好了完整的项目代码，可以边做边学！

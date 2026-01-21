# -*- coding: utf-8 -*-
"""
深度学习第3课：损失函数
=========================

前两课我们学了：
- 第1课：单个神经元怎么计算
- 第2课：多个神经元组成网络

这一课要解决一个关键问题：
- 网络的预测"好不好"？差多少？

这就是"损失函数"要回答的问题！
"""

import numpy as np

print("=" * 50)
print("第3课：损失函数 - 衡量预测有多'错'")
print("=" * 50)

# ====================
# 复习：前向传播
# ====================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 沿用第2课的网络参数
W1 = np.array([
    [0.8, -0.5, 0.3, 0.6],
    [0.2, 0.9, -0.7, 0.1],
    [0.5, 0.3, 0.8, 0.4],
])
b1 = np.array([-0.3, -0.4, 0.1, -0.2])
W2 = np.array([0.6, 0.4, 0.5, 0.3])
b2 = -0.5

def forward(x):
    """前向传播"""
    hidden = sigmoid(np.dot(x, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    return output

# ====================
# 新概念：标签（正确答案）
# ====================

print("""
【新概念】标签 (Label)

之前我们只有"输入"，现在加上"正确答案"：

    水果数据 = (特征, 标签)
                 |      |
              输入X   正确答案Y

    例如：
    完美草莓 = ([0.9, 0.3, 0.95], 1)  <- 1表示好吃
    烂苹果   = ([0.5, 0.4, 0.1],  0)  <- 0表示不好吃
""")

# 训练数据：(特征, 标签)
training_data = [
    ("完美草莓", [0.9, 0.3, 0.95], 1),   # 好吃
    ("甜橙子",   [0.85, 0.5, 0.7], 1),   # 好吃
    ("酸柠檬",   [0.2, 0.95, 0.8], 0),   # 不好吃(太酸)
    ("烂苹果",   [0.5, 0.4, 0.1],  0),   # 不好吃(不新鲜)
]

print("-" * 50)
print("【训练数据】")
print("-" * 50)

for name, features, label in training_data:
    label_text = "好吃" if label == 1 else "不好吃"
    print(f"  {name}: 特征={features}, 标签={label}({label_text})")

# ====================
# 预测 vs 真实
# ====================

print("\n" + "-" * 50)
print("【预测 vs 真实】")
print("-" * 50)

predictions = []
labels = []

for name, features, label in training_data:
    pred = forward(np.array(features))
    predictions.append(pred)
    labels.append(label)

    # 判断对错
    pred_label = 1 if pred > 0.5 else 0
    correct = "OK" if pred_label == label else "X"

    print(f"\n  {name}:")
    print(f"    预测: {pred:.1%}")
    print(f"    真实: {label} ({'好吃' if label else '不好吃'})")
    print(f"    判断: [{correct}]")

# ====================
# 损失函数：衡量"错多少"
# ====================

print("\n" + "=" * 50)
print("【核心概念】损失函数 (Loss Function)")
print("=" * 50)

print("""
损失函数回答一个问题：预测值和真实值"差多少"？

最简单的想法：直接算差值
    误差 = 预测 - 真实

问题：
    - 有正有负，会互相抵消
    - 预测0.9，真实1 -> 误差-0.1
    - 预测0.1，真实0 -> 误差+0.1
    - 平均一下变成0？不对！

解决方案：平方！
    误差 = (预测 - 真实)^2

这就是 MSE（均方误差）的由来！
""")

# ====================
# MSE 损失函数
# ====================

print("-" * 50)
print("【MSE 均方误差】")
print("-" * 50)

def mse_loss(predictions, labels):
    """
    均方误差 (Mean Squared Error)

    公式: MSE = (1/n) * sum((pred - label)^2)
    """
    total = 0
    for pred, label in zip(predictions, labels):
        error = (pred - label) ** 2
        total += error
    return total / len(predictions)

# 计算每个样本的误差
print("\n逐个计算误差：")
for i, (name, features, label) in enumerate(training_data):
    pred = predictions[i]
    error = (pred - label) ** 2
    print(f"  {name}: ({pred:.3f} - {label})^2 = {error:.4f}")

# 计算总损失
loss = mse_loss(predictions, labels)
print(f"\n平均损失 (MSE): {loss:.4f}")

# ====================
# 损失的意义
# ====================

print("\n" + "=" * 50)
print("【损失的意义】")
print("=" * 50)

print(f"""
当前网络的损失: {loss:.4f}

损失越小 -> 预测越准确
损失越大 -> 预测越离谱

目标：通过调整权重，让损失变小！

    损失 = 0.25   (现在)
           |
        调整权重
           |
           v
    损失 = 0.10   (更好)
           |
        继续调整
           |
           v
    损失 = 0.01   (很好！)
""")

# ====================
# 对比：好权重 vs 差权重
# ====================

print("=" * 50)
print("【对比实验】好权重 vs 差权重")
print("=" * 50)

# 故意用一组很差的权重
W1_bad = np.random.randn(3, 4) * 0.1
b1_bad = np.zeros(4)
W2_bad = np.random.randn(4) * 0.1
b2_bad = 0

def forward_bad(x):
    hidden = sigmoid(np.dot(x, W1_bad) + b1_bad)
    output = sigmoid(np.dot(hidden, W2_bad) + b2_bad)
    return output

# 用差权重预测
bad_predictions = []
for name, features, label in training_data:
    pred = forward_bad(np.array(features))
    bad_predictions.append(pred)

bad_loss = mse_loss(bad_predictions, labels)

print(f"""
手动设定的权重:
  损失 = {loss:.4f}

随机初始化的权重:
  损失 = {bad_loss:.4f}

{'手动权重更好！' if loss < bad_loss else '随机权重碰巧更好'}

这说明：
- 权重的好坏直接影响损失
- 训练的目标就是找到让损失最小的权重
""")

# ====================
# 交叉熵损失（进阶）
# ====================

print("=" * 50)
print("【进阶】交叉熵损失 (Cross Entropy)")
print("=" * 50)

print("""
MSE 适合回归问题（预测连续值）
交叉熵适合分类问题（预测类别）

交叉熵公式：
  Loss = -[y*log(p) + (1-y)*log(1-p)]

  其中：
  - y 是真实标签 (0或1)
  - p 是预测概率

为什么用交叉熵？
- 当预测错得离谱时，惩罚更大
- 梯度更好，训练更快
""")

def cross_entropy_loss(predictions, labels):
    """交叉熵损失"""
    total = 0
    eps = 1e-15  # 防止log(0)
    for pred, label in zip(predictions, labels):
        pred = np.clip(pred, eps, 1 - eps)
        loss = -(label * np.log(pred) + (1 - label) * np.log(1 - pred))
        total += loss
    return total / len(predictions)

ce_loss = cross_entropy_loss(predictions, labels)
print(f"MSE 损失:    {loss:.4f}")
print(f"交叉熵损失:  {ce_loss:.4f}")

# ====================
# 总结
# ====================

print("\n" + "=" * 50)
print("【第3课总结】")
print("=" * 50)

print("""
[1] 标签 (Label):
   - 每个训练样本的"正确答案"
   - 有了标签才能计算"对不对"

[2] 损失函数 (Loss Function):
   - 衡量预测和真实的差距
   - 损失越小，预测越准

[3] 常见损失函数:
   - MSE (均方误差): 适合回归
   - 交叉熵: 适合分类

[4] 训练的目标:
   - 找到让损失最小的权重
   - 怎么找？下一课揭晓！

[下一课预告] 第4课: 反向传播与梯度下降
   - 怎么知道往哪个方向调整权重？
   - 怎么一步步让损失变小？
""")

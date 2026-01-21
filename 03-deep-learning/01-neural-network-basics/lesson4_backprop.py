# -*- coding: utf-8 -*-
"""
深度学习第4课：反向传播与梯度下降
===================================

这是深度学习最核心的一课！

前面我们学了：
- 第1课：神经元怎么计算
- 第2课：多层网络结构
- 第3课：损失函数（衡量预测有多错）

这一课解决最后的问题：
- 怎么自动调整权重，让损失变小？

答案就是：梯度下降 + 反向传播
"""

import numpy as np

print("=" * 50)
print("第4课：反向传播与梯度下降")
print("=" * 50)

# ====================
# 简化问题：单个神经元
# ====================

print("""
【简化问题】先看最简单的情况

只有1个输入、1个权重、1个输出：

    输入x ──[权重w]──> 输出y = x * w

问题：给定正确答案，怎么调整 w？
""")

# ====================
# 梯度的直觉
# ====================

print("-" * 50)
print("【核心概念1】梯度 = 方向 + 大小")
print("-" * 50)

print("""
假设：
    输入 x = 2
    当前权重 w = 0.5
    预测 y = x * w = 1.0
    真实答案 = 2.0
    损失 = (1.0 - 2.0)^2 = 1.0

问：w 应该变大还是变小？

分析：
    - 预测 1.0，真实 2.0，预测太小了
    - y = x * w，要让 y 变大
    - x 是固定的，所以 w 要变大！

"梯度"就是告诉我们：
    1. 方向：w 该变大还是变小
    2. 大小：该变多少
""")

# 手动计算梯度
x = 2.0
w = 0.5
y_true = 2.0

y_pred = x * w
loss = (y_pred - y_true) ** 2

# 梯度 = d(loss)/d(w)
# loss = (x*w - y_true)^2
# d(loss)/d(w) = 2 * (x*w - y_true) * x
gradient = 2 * (y_pred - y_true) * x

print(f"""
计算：
    预测: y = {x} * {w} = {y_pred}
    损失: ({y_pred} - {y_true})^2 = {loss}
    梯度: 2 * ({y_pred} - {y_true}) * {x} = {gradient}

梯度 = {gradient}（负数）
    - 负号表示：w 要往正方向调整（变大）
    - 绝对值4表示：调整的"力度"
""")

# ====================
# 梯度下降
# ====================

print("-" * 50)
print("【核心概念2】梯度下降")
print("-" * 50)

print("""
梯度下降公式：

    新权重 = 旧权重 - 学习率 * 梯度

为什么是"减"？
    - 梯度指向损失增大的方向
    - 我们要让损失减小，所以反方向走

学习率(Learning Rate)：
    - 控制每一步走多大
    - 太大：可能跳过最优点
    - 太小：收敛太慢
""")

learning_rate = 0.1

print(f"学习率 = {learning_rate}")
print(f"\n开始训练...\n")

# 训练循环
for epoch in range(10):
    # 前向传播
    y_pred = x * w
    loss = (y_pred - y_true) ** 2

    # 计算梯度
    gradient = 2 * (y_pred - y_true) * x

    # 更新权重
    w_old = w
    w = w - learning_rate * gradient

    print(f"轮次{epoch+1}: w={w_old:.4f} -> {w:.4f}, 损失={loss:.4f}")

print(f"""
训练完成！
    最终权重: w = {w:.4f}
    理想权重: w = 1.0 (因为 2*1=2)

w 从 0.5 自动调整到了接近 1.0！
""")

# ====================
# 多层网络的问题
# ====================

print("=" * 50)
print("【核心概念3】反向传播")
print("=" * 50)

print("""
上面是最简单的情况，真实网络有多层：

    输入 -> 隐藏层 -> 输出层 -> 损失

问题：隐藏层的权重怎么算梯度？

答案：链式法则！

    d(损失)/d(隐藏层权重) = d(损失)/d(输出) * d(输出)/d(隐藏层) * d(隐藏层)/d(权重)

"反向传播"就是：
    1. 先算输出层的梯度
    2. 再"反向"传到隐藏层
    3. 一层层往回传
""")

# ====================
# 完整的神经网络训练
# ====================

print("-" * 50)
print("【完整示例】训练一个小网络")
print("-" * 50)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """sigmoid 的导数"""
    s = sigmoid(x)
    return s * (1 - s)

# 简单网络：2输入 -> 2隐藏 -> 1输出
np.random.seed(42)

# 初始化权重
W1 = np.random.randn(2, 2) * 0.5  # 输入层到隐藏层
b1 = np.zeros(2)
W2 = np.random.randn(2) * 0.5     # 隐藏层到输出层
b2 = 0.0

# 训练数据：XOR 问题
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([0, 1, 1, 0])  # XOR 的输出

print("""
训练目标：学习 XOR 运算

    输入    | 期望输出
    --------|--------
    [0, 0]  |   0
    [0, 1]  |   1
    [1, 0]  |   1
    [1, 1]  |   0

这个问题单层网络无法解决，需要多层！
""")

learning_rate = 1.0

print("开始训练...\n")

for epoch in range(1000):
    total_loss = 0

    for i in range(len(X)):
        x = X[i]
        y_true = Y[i]

        # ===== 前向传播 =====
        z1 = np.dot(x, W1) + b1
        a1 = sigmoid(z1)  # 隐藏层输出

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)  # 最终输出

        # 计算损失
        loss = (a2 - y_true) ** 2
        total_loss += loss

        # ===== 反向传播 =====
        # 输出层梯度
        d_loss = 2 * (a2 - y_true)
        d_z2 = d_loss * sigmoid_derivative(z2)

        # 隐藏层梯度（链式法则）
        d_a1 = d_z2 * W2
        d_z1 = d_a1 * sigmoid_derivative(z1)

        # 计算权重梯度
        d_W2 = d_z2 * a1
        d_b2 = d_z2
        d_W1 = np.outer(x, d_z1)
        d_b1 = d_z1

        # ===== 更新权重 =====
        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2
        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1

    # 每100轮打印一次
    if (epoch + 1) % 200 == 0:
        avg_loss = total_loss / len(X)
        print(f"轮次 {epoch+1:4d}: 平均损失 = {avg_loss:.6f}")

# 测试训练结果
print("\n训练完成！测试结果：")
print("-" * 30)

for i in range(len(X)):
    x = X[i]
    y_true = Y[i]

    # 前向传播
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    pred = 1 if a2 > 0.5 else 0
    correct = "OK" if pred == y_true else "X"
    print(f"  输入 {x} -> 预测 {a2:.3f} -> {pred} (期望 {y_true}) [{correct}]")

# ====================
# 总结
# ====================

print("\n" + "=" * 50)
print("【第4课总结】")
print("=" * 50)

print("""
[1] 梯度 (Gradient):
   - 损失函数对权重的偏导数
   - 告诉我们权重该往哪调、调多少

[2] 梯度下降 (Gradient Descent):
   - 新权重 = 旧权重 - 学习率 * 梯度
   - 沿着损失减小的方向更新权重

[3] 反向传播 (Backpropagation):
   - 用链式法则计算每层的梯度
   - 从输出层往输入层"反向"传播

[4] 训练循环:
   前向传播 -> 计算损失 -> 反向传播 -> 更新权重 -> 重复

[5] 学习率 (Learning Rate):
   - 控制每步更新的大小
   - 需要调参找到合适的值

恭喜！你已经理解了深度学习的核心原理！
接下来可以学习 CNN、RNN 等更复杂的网络结构。
""")

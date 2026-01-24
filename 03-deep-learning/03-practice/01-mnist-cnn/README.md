# MNIST 手写数字识别 - CNN 实战

> 第一个深度学习实战项目：从零训练一个 CNN 识别手写数字

---

## 项目简介

**任务**：识别 0-9 的手写数字
**数据集**：MNIST（60,000 训练 + 10,000 测试）
**模型**：卷积神经网络 (CNN)
**预期准确率**：98%+

---

## 运行项目

### 1. 确保已安装依赖

```bash
pip install torch torchvision matplotlib numpy
```

### 2. 运行训练

```bash
cd 03-deep-learning/03-practice/01-mnist-cnn
python mnist_cnn.py
```

### 3. 查看结果

训练完成后会生成：
- `mnist_cnn.pth` - 训练好的模型
- `predictions.png` - 预测结果可视化

---

## 代码结构

```python
mnist_cnn.py
├── load_data()           # 加载 MNIST 数据集
├── SimpleCNN             # CNN 模型定义
│   ├── conv1 + pool1     # 第一层卷积+池化
│   ├── conv2 + pool2     # 第二层卷积+池化
│   └── fc1 + fc2         # 全连接层
├── train_one_epoch()     # 训练一个 epoch
├── test()                # 测试模型
├── visualize_predictions() # 可视化预测
└── main()                # 主流程
```

---

## 模型架构

```
输入: 1×28×28 (灰度图)
    ↓
卷积层1: 1 -> 32 个通道, 3×3 卷积核
    ↓
ReLU 激活
    ↓
池化层1: 28×28 -> 14×14 (2×2 最大池化)
    ↓
卷积层2: 32 -> 64 个通道, 3×3 卷积核
    ↓
ReLU 激活
    ↓
池化层2: 14×14 -> 7×7
    ↓
展平: 64×7×7 = 3136
    ↓
全连接1: 3136 -> 128
    ↓
ReLU 激活
    ↓
全连接2: 128 -> 10 (10个类别)
    ↓
输出: [0-9的分数]
```

**参数量**：约 100,000 个参数

---

## 学习要点

### 1. 数据加载

```python
transform = transforms.Compose([
    transforms.ToTensor(),           # PIL 图像 -> 张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**关键概念**：
- `transforms.ToTensor()` - 把图像从 PIL 格式转成 PyTorch 张量
- `DataLoader` - 自动分批次、打乱数据
- `batch_size=64` - 每次送 64 张图片进网络

---

### 2. 模型定义

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # ...

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        # ...
        return x
```

**关键概念**：
- `nn.Module` - 所有模型的基类
- `__init__` - 定义网络层
- `forward` - 定义数据流动方式

---

### 3. 训练循环

```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 1. 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重
```

**关键概念**：
- `optimizer.zero_grad()` - 必须清空上一次的梯度
- `loss.backward()` - 自动计算所有参数的梯度
- `optimizer.step()` - 根据梯度更新权重

---

### 4. 损失函数

```python
criterion = nn.CrossEntropyLoss()
```

**交叉熵损失**：
- 用于多分类问题
- 输入：模型输出的分数（未归一化）
- 输出：一个数字，表示预测和真实标签的差距
- 越小越好

---

### 5. 优化器

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Adam 优化器**：
- 自动调整学习率
- 比 SGD 更聪明，收敛更快
- 是目前最常用的优化器

---

## 常见问题

### Q1: 为什么要归一化？

```python
transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
```

**原因**：
- 原始像素值是 [0, 255]
- 归一化后变成 [-1, 1] 或 [0, 1]
- 让训练更稳定，收敛更快

---

### Q2: padding=1 是什么意思？

```python
nn.Conv2d(1, 32, kernel_size=3, padding=1)
```

**作用**：
- 3×3 卷积核会让图像变小：28×28 -> 26×26
- padding=1 在边缘填充一圈 0：28×28 -> 28×28
- 保持尺寸不变

---

### Q3: model.train() 和 model.eval() 有什么区别？

```python
model.train()  # 训练模式
model.eval()   # 评估模式
```

**区别**：
- 训练模式：Dropout、BatchNorm 正常工作
- 评估模式：Dropout 关闭、BatchNorm 使用固定参数
- 测试时必须用 `eval()`

---

### Q4: 为什么要用 torch.no_grad()？

```python
with torch.no_grad():
    outputs = model(images)
```

**原因**：
- 测试时不需要计算梯度
- `no_grad()` 关闭梯度计算，节省内存和时间

---

## 实验建议

### 1. 尝试不同的学习率

```python
learning_rate = 0.01   # 太大，可能不收敛
learning_rate = 0.001  # 推荐
learning_rate = 0.0001 # 太小，收敛慢
```

观察训练曲线的变化。

---

### 2. 尝试更深的网络

添加第三层卷积：

```python
self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
self.pool3 = nn.MaxPool2d(2, 2)
```

看准确率是否提升。

---

### 3. 尝试不同的优化器

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)       # 传统 SGD
optimizer = optim.Adam(model.parameters(), lr=0.001)     # Adam
optimizer = optim.RMSprop(model.parameters(), lr=0.001)  # RMSprop
```

对比收敛速度。

---

## 下一步

完成这个项目后，你可以：

1. **改进模型**
   - 添加 Dropout 防止过拟合
   - 使用 Batch Normalization
   - 尝试不同的激活函数

2. **尝试新数据集**
   - CIFAR-10（彩色图像，10个类别）
   - Fashion-MNIST（衣服图片）

3. **学习迁移学习**
   - 使用预训练的 ResNet、VGG
   - 在新数据集上 fine-tune

---

<p align="center">
<b>🎉 恭喜完成第一个深度学习实战项目！</b>
</p>

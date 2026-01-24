"""
MNIST 手写数字识别 - CNN 实战

目标：训练一个卷积神经网络识别 0-9 的手写数字
数据集：MNIST (28x28 灰度图像)
预期准确率：98%+
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 第1步：加载 MNIST 数据集
# ============================================

def load_data(batch_size=64):
    """
    加载 MNIST 数据集

    返回：
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理：转为张量 + 归一化到 [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为张量 (H, W, C) -> (C, H, W)
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])

    # 下载训练集
    train_dataset = datasets.MNIST(
        root='./data',  # 数据存放位置
        train=True,     # 训练集
        download=True,  # 如果没有就下载
        transform=transform
    )

    # 下载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,    # 测试集
        download=True,
        transform=transform
    )

    # 创建数据加载器（自动分批次）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 每批64张图片
        shuffle=True  # 打乱顺序
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  # 测试集不打乱
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_loader, test_loader


# ============================================
# 第2步：定义 CNN 模型
# ============================================

class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型

    架构：
        输入: 1×28×28 (灰度图)
        Conv1: 1×28×28 -> 32×28×28
        Pool1: 32×28×28 -> 32×14×14
        Conv2: 32×14×14 -> 64×14×14
        Pool2: 64×14×14 -> 64×7×7
        Flatten: 64×7×7 -> 3136
        FC1: 3136 -> 128
        FC2: 128 -> 10
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第一层卷积：1个输入通道 -> 32个输出通道
        self.conv1 = nn.Conv2d(
            in_channels=1,    # 输入通道（灰度图）
            out_channels=32,  # 输出通道（32个滤波器）
            kernel_size=3,    # 3×3 卷积核
            padding=1         # 填充，保持尺寸不变
        )

        # 第一层池化：2×2 最大池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积：32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 第二层池化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层1：64×7×7 = 3136 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # 全连接层2：128 -> 10 (10个数字类别)
        self.fc2 = nn.Linear(128, 10)

        # ReLU 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入图像 (batch_size, 1, 28, 28)

        返回：
            输出分数 (batch_size, 10)
        """
        # 第一层：卷积 -> ReLU -> 池化
        x = self.conv1(x)      # (batch, 1, 28, 28) -> (batch, 32, 28, 28)
        x = self.relu(x)
        x = self.pool1(x)      # (batch, 32, 28, 28) -> (batch, 32, 14, 14)

        # 第二层：卷积 -> ReLU -> 池化
        x = self.conv2(x)      # (batch, 32, 14, 14) -> (batch, 64, 14, 14)
        x = self.relu(x)
        x = self.pool2(x)      # (batch, 64, 14, 14) -> (batch, 64, 7, 7)

        # 展平：(batch, 64, 7, 7) -> (batch, 3136)
        x = x.view(x.size(0), -1)

        # 全连接层1
        x = self.fc1(x)        # (batch, 3136) -> (batch, 128)
        x = self.relu(x)

        # 全连接层2（输出层）
        x = self.fc2(x)        # (batch, 128) -> (batch, 10)

        return x


# ============================================
# 第3步：训练函数
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个 epoch

    参数：
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备（CPU 或 GPU）

    返回：
        平均损失
    """
    model.train()  # 设置为训练模式
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 移动数据到设备（CPU/GPU）
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每100个批次打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================
# 第4步：测试函数
# ============================================

def test(model, test_loader, criterion, device):
    """
    测试模型

    返回：
        平均损失, 准确率
    """
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================
# 第5步：可视化预测结果
# ============================================

def visualize_predictions(model, test_loader, device, num_images=10):
    """
    可视化模型预测结果
    """
    model.eval()

    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    # 绘制
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].cpu().numpy().squeeze()  # 移到 CPU 并去掉通道维度
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"真实: {true_label} | 预测: {pred_label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    print("预测结果已保存到 predictions.png")


# ============================================
# 第6步：主函数
# ============================================

def main():
    """
    主函数：完整的训练流程
    """
    print("=" * 60)
    print("MNIST 手写数字识别 - CNN 实战")
    print("=" * 60)

    # 超参数设置
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # 检查是否有 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_loader, test_loader = load_data(batch_size)

    # 2. 创建模型
    print("\n[2/5] 创建模型...")
    model = SimpleCNN().to(device)
    print(model)

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（多分类）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练模型
    print("\n[3/5] 开始训练...")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 测试
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # 5. 可视化预测结果
    print("\n[4/5] 可视化预测结果...")
    visualize_predictions(model, test_loader, device)

    # 6. 保存模型
    print("\n[5/5] 保存模型...")
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("模型已保存到 mnist_cnn.pth")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

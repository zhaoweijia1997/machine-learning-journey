"""
第1步：理解 MNIST 数据

目标：
1. 看看 MNIST 数据长什么样
2. 理解数据的形状和含义
3. 可视化几张图片
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. 加载数据（最简单的方式）
# ============================================

print("=" * 60)
print("第1步：加载 MNIST 数据集")
print("=" * 60)

# 数据转换：PIL 图像 -> PyTorch 张量
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
])

# 下载训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print(f"\n训练集大小: {len(train_dataset)} 张图片")

# ============================================
# 2. 查看单张图片
# ============================================

print("\n" + "=" * 60)
print("第2步：查看单张图片")
print("=" * 60)

# 获取第一张图片
image, label = train_dataset[0]

print(f"\n图片形状: {image.shape}")
print(f"  - 通道数: {image.shape[0]} (1 = 灰度图)")
print(f"  - 高度: {image.shape[1]} 像素")
print(f"  - 宽度: {image.shape[2]} 像素")
print(f"\n标签: {label} (这是数字 {label})")

print(f"\n像素值范围: [{image.min():.3f}, {image.max():.3f}]")
print(f"  - ToTensor() 自动将 [0, 255] 归一化到 [0, 1]")

# ============================================
# 3. 查看像素矩阵
# ============================================

print("\n" + "=" * 60)
print("第3步：查看像素矩阵（左上角 5×5）")
print("=" * 60)

# 提取左上角 5×5 的像素
top_left = image[0, :5, :5]  # [通道, 高度, 宽度]
print("\n左上角像素值：")
print(top_left.numpy())
print("\n0.0 = 黑色，1.0 = 白色")

# ============================================
# 4. 可视化多张图片
# ============================================

print("\n" + "=" * 60)
print("第4步：可视化前 10 张图片")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i in range(10):
    image, label = train_dataset[i]

    # 转为 numpy，去掉通道维度
    img = image.numpy().squeeze()  # (1, 28, 28) -> (28, 28)

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"标签: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png')
print("\n[OK] 图片已保存到 mnist_samples.png")

# ============================================
# 5. 统计数据分布
# ============================================

print("\n" + "=" * 60)
print("第5步：统计每个数字的数量")
print("=" * 60)

# 统计每个数字出现的次数
label_counts = [0] * 10

for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    label_counts[label] += 1

print("\n数字分布：")
for digit in range(10):
    count = label_counts[digit]
    percentage = count / len(train_dataset) * 100
    bar = '█' * int(percentage / 2)
    print(f"数字 {digit}: {count:5d} 张 ({percentage:5.2f}%) {bar}")

print("\n观察：数据分布比较均匀，每个数字约 6000 张")

# ============================================
# 6. 理解 DataLoader
# ============================================

print("\n" + "=" * 60)
print("第6步：理解 DataLoader（批处理）")
print("=" * 60)

from torch.utils.data import DataLoader

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # 每次取 64 张图片
    shuffle=True    # 打乱顺序
)

print(f"\nDataLoader 信息：")
print(f"  - 总共 {len(train_dataset)} 张图片")
print(f"  - 每批 {train_loader.batch_size} 张")
print(f"  - 需要 {len(train_loader)} 批才能遍历一遍")
print(f"    (60000 ÷ 64 = {60000/64:.1f} ≈ {len(train_loader)})")

# 获取一批数据
images_batch, labels_batch = next(iter(train_loader))

print(f"\n一批数据的形状：")
print(f"  - images_batch: {images_batch.shape}")
print(f"    (批次大小, 通道数, 高度, 宽度)")
print(f"  - labels_batch: {labels_batch.shape}")
print(f"    (批次大小,)")

print(f"\n这批数据的标签：")
print(f"  {labels_batch.tolist()[:10]}... (前10个)")

# ============================================
# 总结
# ============================================

print("\n" + "=" * 60)
print("总结：数据理解要点")
print("=" * 60)

print("""
[OK] MNIST 数据集：
   - 60,000 张训练图片
   - 10,000 张测试图片
   - 每张图片 28×28 像素，灰度图（单通道）
   - 10 个类别（数字 0-9）

[OK] 数据形状：
   - 单张图片：(1, 28, 28) = (通道, 高, 宽)
   - 一批图片：(64, 1, 28, 28) = (批次, 通道, 高, 宽)

[OK] 像素值：
   - 原始：0-255 (整数)
   - ToTensor() 后：0.0-1.0 (浮点数)

[OK] DataLoader：
   - 自动分批次
   - 自动打乱顺序
   - 简化训练循环

现在您理解数据了吗？
下一步：理解模型结构
""")

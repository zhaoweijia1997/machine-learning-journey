"""
验证 MNIST 数据已下载并可以读取
"""

from torchvision import datasets, transforms

print("=" * 60)
print("验证 MNIST 数据")
print("=" * 60)

# 加载数据（不下载，只读取本地）
transform = transforms.ToTensor()

try:
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=False,  # 不下载，只读取本地
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

    print("\n[OK] 数据读取成功！")
    print(f"\n训练集：{len(train_dataset):,} 张图片")
    print(f"测试集：{len(test_dataset):,} 张图片")
    print(f"总计：{len(train_dataset) + len(test_dataset):,} 张图片")

    # 读取第一张图片
    image, label = train_dataset[0]
    print(f"\n第一张图片：")
    print(f"  - 形状：{image.shape}")
    print(f"  - 标签：{label}")
    print(f"  - 像素值范围：[{image.min():.3f}, {image.max():.3f}]")

    # 统计文件大小
    import os
    data_dir = './data/MNIST/raw'
    total_size = 0
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"\n{filename}: {size / 1024 / 1024:.2f} MB")

    print(f"\n总大小：{total_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 60)
    print("验证完成！数据已经在您的电脑上了。")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] 数据读取失败：{e}")
    print("\n可能需要重新下载。")

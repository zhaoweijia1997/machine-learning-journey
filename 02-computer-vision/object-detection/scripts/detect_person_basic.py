"""
基础人形检测脚本
使用 YOLOv8 检测图片中的人

作者: zhaoweijia1997
日期: 2026-01-19
"""

from ultralytics import YOLO
import cv2
import os

def detect_person(image_path, save_path='output.jpg', confidence=0.5):
    """
    检测图片中的人

    参数:
        image_path: 输入图片路径
        save_path: 输出图片路径
        confidence: 置信度阈值 (0-1)
    """
    print(f"📷 正在加载图片: {image_path}")

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return

    # 加载预训练的 YOLOv8 模型
    print("🔄 正在加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')  # n = nano，最小最快的模型

    # 进行检测
    print("🔍 正在检测人形...")
    results = model(image_path)

    # 获取检测结果
    result = results[0]

    # 统计检测到的人数
    person_count = 0
    for box in result.boxes:
        # class 0 是 'person'（人）
        if int(box.cls[0]) == 0 and float(box.conf[0]) >= confidence:
            person_count += 1

    print(f"✅ 检测完成！共检测到 {person_count} 个人")

    # 在图片上绘制检测框
    annotated_img = result.plot()

    # 保存结果
    cv2.imwrite(save_path, annotated_img)
    print(f"💾 结果已保存至: {save_path}")

    # 显示详细结果
    print("\n📊 详细检测结果:")
    for i, box in enumerate(result.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls]

        if cls == 0 and conf >= confidence:  # 只显示人
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f"  {i+1}. {class_name} - 置信度: {conf:.2%} - 位置: ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})")

    return annotated_img

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 YOLOv8 人形检测程序")
    print("=" * 60)
    print()

    # 示例：检测图片
    # 替换成你自己的图片路径
    image_path = "test_image.jpg"

    # 如果你没有测试图片，可以从网上下载一张
    if not os.path.exists(image_path):
        print("⚠️  未找到测试图片")
        print("请将一张包含人的图片命名为 'test_image.jpg' 放在当前目录")
        print()
        print("或者修改代码中的 image_path 变量为你的图片路径")
        print()
        print("你也可以使用以下命令下载示例图片:")
        print("  import urllib.request")
        print("  urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', 'test_image.jpg')")
        return

    # 执行检测
    result_img = detect_person(
        image_path=image_path,
        save_path='detected_output.jpg',
        confidence=0.5  # 置信度阈值，可以调整
    )

    print()
    print("=" * 60)
    print("✨ 程序完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

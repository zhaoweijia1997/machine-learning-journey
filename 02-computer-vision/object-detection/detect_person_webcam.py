"""
实时摄像头人形检测
使用 YOLOv8 进行实时人形检测

作者: zhaoweijia1997
日期: 2026-01-19
"""

from ultralytics import YOLO
import cv2
import time

def detect_webcam(camera_id=0, confidence=0.5, show_fps=True):
    """
    使用摄像头进行实时人形检测

    参数:
        camera_id: 摄像头ID (0=默认摄像头)
        confidence: 置信度阈值 (0-1)
        show_fps: 是否显示 FPS
    """
    print("=" * 60)
    print("🚀 实时人形检测程序")
    print("=" * 60)
    print()

    # 加载模型
    print("🔄 正在加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')

    # 打开摄像头
    print(f"📹 正在打开摄像头 {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        print("   请检查:")
        print("   1. 摄像头是否连接")
        print("   2. 摄像头权限是否开启")
        print("   3. 尝试修改 camera_id 参数")
        return

    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"✅ 摄像头已打开")
    print(f"   分辨率: {width}x{height}")
    print(f"   帧率: {fps} FPS")
    print()
    print("⌨️  按键说明:")
    print("   q - 退出程序")
    print("   s - 保存当前帧")
    print("   空格 - 暂停/继续")
    print()

    # 性能统计
    frame_count = 0
    start_time = time.time()
    paused = False

    try:
        while True:
            if not paused:
                # 读取一帧
                ret, frame = cap.read()

                if not ret:
                    print("❌ 无法读取帧")
                    break

                # 执行检测
                results = model(frame, verbose=False)
                result = results[0]

                # 统计检测到的人数
                person_count = 0
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= confidence:
                        person_count += 1

                # 绘制检测结果
                annotated_frame = result.plot()

                # 计算 FPS
                frame_count += 1
                if show_fps and frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time

                    # 在画面上显示信息
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {current_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # 显示人数
                cv2.putText(
                    annotated_frame,
                    f"People: {person_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # 显示画面
                cv2.imshow('YOLOv8 Real-time Person Detection', annotated_frame)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # 退出
                print("\n👋 正在退出...")
                break
            elif key == ord('s'):
                # 保存当前帧
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 已保存截图: {filename}")
            elif key == ord(' '):
                # 暂停/继续
                paused = not paused
                status = "暂停" if paused else "继续"
                print(f"⏯️  {status}")

    except KeyboardInterrupt:
        print("\n⚠️  检测到中断信号，正在退出...")

    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

        # 显示统计信息
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print()
        print("=" * 60)
        print("📊 运行统计:")
        print(f"   总帧数: {frame_count}")
        print(f"   运行时间: {total_time:.2f} 秒")
        print(f"   平均 FPS: {avg_fps:.1f}")
        print("=" * 60)
        print("✨ 程序已退出")

def main():
    """主函数"""
    # 开始检测
    detect_webcam(
        camera_id=0,      # 默认摄像头
        confidence=0.5,   # 置信度阈值
        show_fps=True     # 显示 FPS
    )

if __name__ == "__main__":
    main()

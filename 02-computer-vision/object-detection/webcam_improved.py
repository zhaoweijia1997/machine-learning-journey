# -*- coding: utf-8 -*-
"""
改进的实时摄像头检测
更好的窗口控制和退出机制
"""

from ultralytics import YOLO
import cv2
import time
import os
import sys

def main():
    print("=" * 60)
    print("实时人形检测 (改进版)")
    print("=" * 60)
    print()

    # 检查 OpenVINO 模型
    openvino_model_path = 'yolov8n_openvino_model'

    if not os.path.exists(openvino_model_path):
        print("模型未找到，请先运行: python detect_gpu.py")
        return

    # 加载模型
    print("加载模型...")
    model = YOLO(openvino_model_path, task='detect')
    print("模型加载完成！")
    print()

    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        print("请检查:")
        print("  1. 摄像头是否连接")
        print("  2. 摄像头权限是否开启")
        print("  3. 是否被其他程序占用")
        return

    # 设置较低分辨率以提高性能
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 摄像头已打开: {width}x{height}")
    print()
    print("=" * 60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("  +/-       - 增加/减少置信度阈值")
    print("=" * 60)
    print()
    print("▶ 检测开始！按 ESC 或 q 退出...")
    print()

    # 设置窗口名称
    window_name = 'Real-time Detection [Press ESC or Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False
    confidence = 0.5

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 无法读取帧")
                    break

                # 推理
                results = model(frame, conf=confidence, verbose=False)

                # 统计
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

                # 绘制结果
                annotated_frame = results[0].plot()

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 显示信息
                info_y = 30
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(annotated_frame, f"People: {person_count}",
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(annotated_frame, f"Conf: {confidence:.2f}",
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示画面
                cv2.imshow(window_name, annotated_frame)

            # 按键处理 (等待时间设为1ms)
            key = cv2.waitKey(1) & 0xFF

            # ESC 键 (27) 或 q 键
            if key == 27 or key == ord('q'):
                print("\n✋ 正在退出...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "⏸️ 暂停" if paused else "▶️ 继续"
                print(status)
            elif key == ord('+') or key == ord('='):
                confidence = min(0.9, confidence + 0.05)
                print(f"置信度: {confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                confidence = max(0.1, confidence - 0.05)
                print(f"置信度: {confidence:.2f}")

    except KeyboardInterrupt:
        print("\n⚠️ 检测到 Ctrl+C，正在退出...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        # 清理资源
        print("正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()

        # 强制关闭所有窗口
        for i in range(5):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print()
        print("=" * 60)
        print("运行统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print("=" * 60)
        print("✨ 程序已退出")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序异常: {e}")
        sys.exit(1)

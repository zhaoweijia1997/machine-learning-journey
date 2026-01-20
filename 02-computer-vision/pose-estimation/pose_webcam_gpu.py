# -*- coding: utf-8 -*-
"""
GPU 加速实时姿态估计 - 摄像头版
使用 YOLOv8-Pose 进行实时人体关键点检测
学习目标：理解姿态估计与目标检测的区别
"""

from ultralytics import YOLO
import cv2
import time
import os
import signal
import sys

# 全局变量用于信号处理
running = True

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global running
    print("\n检测到 Ctrl+C，正在退出...")
    running = False

# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)

def main():
    print("=" * 60)
    print("GPU 加速实时姿态估计 - 摄像头版")
    print("=" * 60)
    print()

    print("姿态估计 vs 目标检测:")
    print("  目标检测: 检测人的边界框 (bounding box)")
    print("  姿态估计: 检测人体关键点 (17个关键点)")
    print()
    print("关键点包括:")
    print("  头部: 鼻子、眼睛、耳朵")
    print("  上肢: 肩膀、肘部、手腕")
    print("  下肢: 臀部、膝盖、脚踝")
    print()

    # 检查 OpenVINO 模型
    openvino_model_path = 'yolov8n-pose_openvino_model'

    if not os.path.exists(openvino_model_path):
        print("首次运行，正在下载并转换 YOLOv8-Pose 模型...")
        print("这需要 1-2 分钟")
        print()
        model = YOLO('yolov8n-pose.pt')  # 注意：pose 模型
        model.export(format='openvino', half=False)
        print("模型转换完成！")
        print()

    # 加载模型
    print("加载 YOLOv8-Pose OpenVINO 模型...")
    model = YOLO(openvino_model_path, task='pose')  # 注意：task='pose'

    # 检测设备
    try:
        from openvino.runtime import Core
        ie = Core()
        devices = ie.available_devices
        print(f"可用设备: {', '.join(devices)}")

        if 'GPU' in devices:
            print("使用: Intel Arc GPU 加速")
        else:
            print("使用: AUTO")
    except:
        print("使用: AUTO")

    print()

    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头！")
        print("请检查摄像头连接和权限")
        return

    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width}x{height}")
    print()
    print("按键说明:")
    print("  ESC/q - 退出")
    print("  点击窗口 X 按钮 - 退出")
    print("  Ctrl+C - 强制退出")
    print("  s - 保存截图")
    print("  空格 - 暂停/继续")
    print()

    # 创建窗口
    window_name = 'Pose Estimation [ESC/Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False

    # 性能监控
    inference_times = []
    total_keypoints = 0

    global running

    try:
        while running:
            # 检查窗口是否被关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n窗口已关闭...")
                break

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧")
                    break

                # GPU 推理（计时）
                infer_start = time.time()
                results = model(frame, verbose=False)
                infer_time = (time.time() - infer_start) * 1000  # ms

                # 记录性能数据
                inference_times.append(infer_time)
                if len(inference_times) > 100:
                    inference_times.pop(0)

                # 统计人数和关键点
                person_count = len(results[0].keypoints) if results[0].keypoints is not None else 0
                keypoints_count = sum(len(kp.xy[0]) for kp in results[0].keypoints) if results[0].keypoints is not None else 0
                total_keypoints += keypoints_count

                # 绘制结果（包括骨架线）
                annotated_frame = results[0].plot()

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 计算平均推理时间
                avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0

                # 所有信息显示在右下角
                h_frame, w_frame = annotated_frame.shape[:2]
                line_height = 22
                font_scale = 0.45
                thickness = 1

                texts = [
                    (f"FPS: {current_fps:.1f} | GPU: ON | Pose Estimation", (0, 255, 0)),
                    (f"People: {person_count} | Keypoints: {keypoints_count}", (0, 255, 0)),
                    (f"Inference: {avg_inference:.1f}ms", (0, 255, 255))
                ]

                # 从右下角向上绘制
                for i, (text, color) in enumerate(reversed(texts)):
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = w_frame - text_size[0] - 10
                    text_y = h_frame - 10 - (i * line_height)

                    cv2.putText(
                        annotated_frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness
                    )

                cv2.imshow(window_name, annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC 或 q
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"pose_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}")

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理资源
        print("正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()

        # 强制关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)

        # Windows 下额外的清理
        try:
            cv2.destroyAllWindows()
        except:
            pass

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_inference_final = sum(inference_times) / len(inference_times) if inference_times else 0

        print()
        print("=" * 60)
        print("运行统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  总关键点数: {total_keypoints}")
        print()
        print("性能分析:")
        print(f"  平均推理时间: {avg_inference_final:.2f} ms")
        print(f"  理论最大 FPS: {1000/avg_inference_final:.1f}" if avg_inference_final > 0 else "  理论最大 FPS: N/A")
        print()
        print("学习笔记:")
        print("  YOLOv8-Pose 输出 17 个关键点:")
        print("    0: 鼻子    1-2: 眼睛    3-4: 耳朵")
        print("    5-6: 肩膀   7-8: 肘部    9-10: 手腕")
        print("    11-12: 臀部 13-14: 膝盖  15-16: 脚踝")
        print("=" * 60)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
GPU 加速实时屏幕检测
使用 Intel Arc GPU 进行实时人形检测
"""

from ultralytics import YOLO
import cv2
import time
import os
import numpy as np
import mss
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GPU 加速屏幕检测')
    parser.add_argument('--monitor', type=int, default=1,
                       help='显示器编号 (1, 2, 3..., 默认=1)')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU 加速实时屏幕检测")
    print("=" * 60)
    print()

    # 检查 OpenVINO 模型
    openvino_model_path = 'yolov8n_openvino_model'

    if not os.path.exists(openvino_model_path):
        print("首次运行，正在转换模型...")
        print("这需要 1-2 分钟")
        print()
        model = YOLO('yolov8n.pt')
        model.export(format='openvino', half=False)
        print("模型转换完成！")
        print()

    # 加载模型
    print("加载 OpenVINO 模型...")
    model = YOLO(openvino_model_path, task='detect')

    # 检测设备
    try:
        from openvino.runtime import Core
        ie = Core()
        devices = ie.available_devices
        print(f"可用设备: {', '.join(devices)}")

        if 'GPU' in devices:
            device = 'GPU'
            print("使用: Intel Arc GPU 加速")
        else:
            device = 'AUTO'
            print("使用: AUTO")
    except:
        device = 'AUTO'
        print("使用: AUTO")

    print()

    # 初始化屏幕捕获
    print("正在初始化屏幕捕获...")
    sct = mss.mss()

    # 列出所有显示器
    monitors = sct.monitors[1:]  # 排除第一个（全部显示器）
    print(f"检测到 {len(monitors)} 个显示器:")
    for i, mon in enumerate(monitors, 1):
        print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")

    # 选择显示器
    monitor_idx = min(args.monitor, len(monitors))
    monitor = monitors[monitor_idx - 1]

    print(f"\n使用显示器 {monitor_idx}: {monitor['width']}x{monitor['height']}")
    print()
    print("按键说明:")
    print("  ESC/q - 退出")
    print("  s - 保存截图")
    print("  空格 - 暂停/继续")
    print("  m - 切换显示器")
    print()

    # 创建可调整大小的窗口
    window_name = 'GPU Screen Detection [ESC/Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # 初始大小

    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False

    try:
        while True:
            if not paused:
                # 捕获屏幕
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # GPU 推理（OpenVINO 自动使用最优设备）
                results = model(frame, verbose=False)

                # 统计人数
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

                # 绘制结果
                annotated_frame = results[0].plot()

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:  # 每秒更新一次
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 显示信息
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f} (GPU)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"Monitor {monitor_idx} | People: {person_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # 直接显示，窗口可调整大小
                cv2.imshow(window_name, annotated_frame)

            # 按键处理 - 增加 ESC 键支持
            key = cv2.waitKey(1) & 0xFF

            # ESC 键 (27) 或 q 键
            if key == 27 or key == ord('q'):
                print("\n✋ 正在退出...")
                break
            elif key == ord('s'):
                filename = f"screen_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(f"{status}")
            elif key == ord('m'):
                # 切换显示器
                monitor_idx = (monitor_idx % len(monitors)) + 1
                monitor = monitors[monitor_idx - 1]
                print(f"切换到显示器 {monitor_idx}: {monitor['width']}x{monitor['height']}")

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理资源
        print("正在释放资源...")
        cv2.destroyAllWindows()

        # 强制关闭所有 OpenCV 窗口（Windows 修复）
        for i in range(10):
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

if __name__ == "__main__":
    main()

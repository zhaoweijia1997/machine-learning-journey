# -*- coding: utf-8 -*-
"""
GPU 加速屏幕检测 - 简化版
基于 webcam_gpu.py 改造，使用 DXCam 采集屏幕
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GPU 加速屏幕检测')
    parser.add_argument('--monitor', type=int, default=1,
                       help='显示器编号 (1, 2, 3..., 默认=1)')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU 加速屏幕检测 - 简化版")
    print("=" * 60)
    print()

    # 使用 MSS 捕获整个屏幕（更稳定，不会镜像）
    import mss
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

    # 加载模型
    print("加载 yolov8l OpenVINO 模型...")
    model = YOLO('yolov8l_openvino_model', task='detect')

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
    print("按键说明:")
    print("  ESC/q - 退出")
    print("  s - 保存截图")
    print("  空格 - 暂停/继续")
    print("  m - 切换显示器")
    print()

    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False
    annotated_frame = None

    try:
        while True:
            if not paused:
                # 捕获屏幕
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # GPU 推理
                results = model(frame, verbose=False)

                # 统计人数
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
                cv2.putText(
                    annotated_frame,
                    f"Monitor {monitor_idx} | FPS: {current_fps:.1f} | People: {person_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # 显示帧（移到外面确保每次循环都刷新）
            if annotated_frame is not None:
                cv2.imshow('Screen Detection [ESC/Q to quit]', annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC 或 q
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"screen_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}")
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

        # 强制关闭窗口
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

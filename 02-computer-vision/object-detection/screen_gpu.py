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
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值 (0.0-1.0, 默认=0.5)')
    parser.add_argument('--delay', type=int, default=0,
                       help='每帧延时(毫秒, 默认=0)')
    parser.add_argument('--max-fps', type=int, default=0,
                       help='最大帧率限制 (0=不限制, 默认=0)')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示窗口，仅后台检测')
    parser.add_argument('--resize', type=int, default=0,
                       help='推理输入分辨率 (0=原始, 640/1280等, 默认=0)')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU 加速实时屏幕检测")
    print("=" * 60)
    print()

    # 显示配置
    print("检测配置:")
    print(f"  置信度阈值: {args.conf}")
    print(f"  推理分辨率: {'原始' if args.resize == 0 else f'{args.resize}px'}")
    print(f"  帧延时: {args.delay}ms")
    if args.max_fps > 0:
        print(f"  最大帧率: {args.max_fps} FPS")
    else:
        print(f"  最大帧率: 不限制")
    print(f"  显示窗口: {'否' if args.no_display else '是'}")
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

    # 尝试使用 DXCam 硬件加速
    try:
        import dxcam
        sct = mss.mss()
        monitors = sct.monitors[1:]

        print(f"检测到 {len(monitors)} 个显示器:")
        for i, mon in enumerate(monitors, 1):
            print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")

        monitor_idx = min(args.monitor, len(monitors))

        # 使用 DXCam 硬件加速捕获
        camera = dxcam.create(output_idx=monitor_idx-1, output_color="BGR")
        use_dxcam = True
        print(f"\n✓ DXCam 硬件加速已启用")
        print(f"使用显示器 {monitor_idx}: {monitors[monitor_idx-1]['width']}x{monitors[monitor_idx-1]['height']}")
    except Exception as e:
        # 回退到 MSS
        print(f"DXCam 不可用，使用 MSS: {e}")
        sct = mss.mss()
        monitors = sct.monitors[1:]

        print(f"检测到 {len(monitors)} 个显示器:")
        for i, mon in enumerate(monitors, 1):
            print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")

        monitor_idx = min(args.monitor, len(monitors))
        monitor = monitors[monitor_idx - 1]
        use_dxcam = False
        print(f"\n使用显示器 {monitor_idx}: {monitor['width']}x{monitor['height']}")
    print()
    print("按键说明:")
    print("  ESC/q - 退出")
    print("  s - 保存截图")
    print("  空格 - 暂停/继续")
    print("  m - 切换显示器")
    print()

    # 创建可调整大小的窗口
    if not args.no_display:
        window_name = 'GPU Screen Detection [ESC/Q to quit]'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # 初始大小

    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False

    # 帧率限制
    frame_time_target = 1.0 / args.max_fps if args.max_fps > 0 else 0
    last_frame_time = time.time()

    # 性能监控
    capture_times = []
    inference_times = []
    total_detections = 0

    try:
        while True:
            if not paused:
                # 捕获屏幕（计时）
                capture_start = time.time()
                if use_dxcam:
                    frame = camera.grab()
                    if frame is None:
                        continue
                else:
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                capture_time = (time.time() - capture_start) * 1000  # ms

                # 降低分辨率用于推理（如果指定）
                if args.resize > 0:
                    h, w = frame.shape[:2]
                    scale = args.resize / max(h, w)
                    infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    infer_frame = frame

                # GPU 推理（计时）
                infer_start = time.time()
                results = model(infer_frame, conf=args.conf, verbose=False)
                infer_time = (time.time() - infer_start) * 1000  # ms

                # 记录性能数据
                capture_times.append(capture_time)
                inference_times.append(infer_time)
                if len(capture_times) > 100:
                    capture_times.pop(0)
                    inference_times.pop(0)

                # 统计人数和检测数
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
                detection_count = len(results[0].boxes)
                total_detections += detection_count

                # 绘制结果
                annotated_frame = results[0].plot()

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:  # 每秒更新一次
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 计算平均性能指标
                avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0
                avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
                total_time_per_frame = avg_capture + avg_inference

                # 所有信息显示在右下角
                h_frame, w_frame = annotated_frame.shape[:2]
                line_height = 22
                font_scale = 0.45
                thickness = 1

                # 准备所有文本行
                max_fps_text = f"{args.max_fps}" if args.max_fps > 0 else "Unlimited"
                texts = [
                    (f"FPS: {current_fps:.1f} | DXCam: {'ON' if use_dxcam else 'OFF'} | GPU: ON", (0, 255, 0)),
                    (f"Monitor: {monitor_idx} | People: {person_count} | Objects: {detection_count}", (0, 255, 0)),
                    (f"Capture: {avg_capture:.1f}ms | Inference: {avg_inference:.1f}ms | Total: {total_time_per_frame:.1f}ms", (0, 255, 255)),
                    (f"Conf: {args.conf} | Delay: {args.delay}ms | Max: {max_fps_text} | Resize: {args.resize if args.resize > 0 else 'Ori'}", (255, 255, 0))
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

                # 显示窗口
                if not args.no_display:
                    cv2.imshow(window_name, annotated_frame)

                # 帧率限制
                if frame_time_target > 0:
                    current_time_now = time.time()
                    elapsed = current_time_now - last_frame_time
                    if elapsed < frame_time_target:
                        time.sleep(frame_time_target - elapsed)
                    last_frame_time = time.time()

                # 每帧延时
                if args.delay > 0:
                    time.sleep(args.delay / 1000.0)

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

                if use_dxcam:
                    # 重新初始化 DXCam
                    camera.stop()
                    camera = dxcam.create(output_idx=monitor_idx-1, output_color="BGR")
                    print(f"切换到显示器 {monitor_idx}: {monitors[monitor_idx-1]['width']}x{monitors[monitor_idx-1]['height']}")
                else:
                    monitor = monitors[monitor_idx - 1]
                    print(f"切换到显示器 {monitor_idx}: {monitor['width']}x{monitor['height']}")

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理资源
        print("正在释放资源...")
        if use_dxcam:
            camera.stop()
        cv2.destroyAllWindows()

        # 强制关闭所有 OpenCV 窗口（Windows 修复）
        for i in range(10):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_capture_final = sum(capture_times) / len(capture_times) if capture_times else 0
        avg_inference_final = sum(inference_times) / len(inference_times) if inference_times else 0

        print()
        print("=" * 60)
        print("运行统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  总检测数: {total_detections}")
        print()
        print("性能分析:")
        print(f"  平均捕获时间: {avg_capture_final:.2f} ms")
        print(f"  平均推理时间: {avg_inference_final:.2f} ms")
        print(f"  平均总时间: {avg_capture_final + avg_inference_final:.2f} ms")
        print(f"  理论最大 FPS: {1000/(avg_capture_final + avg_inference_final):.1f}" if (avg_capture_final + avg_inference_final) > 0 else "  理论最大 FPS: N/A")
        print()
        print("瓶颈分析:")
        if avg_capture_final > avg_inference_final:
            print(f"  ⚠️  屏幕捕获是瓶颈 ({avg_capture_final:.1f}ms > {avg_inference_final:.1f}ms)")
            print(f"  优化建议:")
            print(f"    1. 确保 DXCam 硬件加速已启用")
            print(f"    2. 降低捕获分辨率（影响显示质量）")
        else:
            print(f"  ⚠️  GPU 推理是瓶颈 ({avg_inference_final:.1f}ms > {avg_capture_final:.1f}ms)")
            print(f"  优化建议:")
            print(f"    1. 降低输入分辨率（resize 输入图像）- 最有效")
            print(f"    2. 使用更小模型（yolov8n 最快）")
            print(f"    3. 确保 GPU 加速正常工作")
            print(f"  ⚠️  注意: 置信度阈值不影响推理速度！")
        print("=" * 60)

if __name__ == "__main__":
    main()

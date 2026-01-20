# -*- coding: utf-8 -*-
"""
GPU 超高速屏幕检测 - 优化屏幕捕获
使用 Windows Graphics Capture API (dxcam) 实现硬件加速捕获
"""

import cv2
import numpy as np
import time
import sys

try:
    import dxcam
except ImportError:
    print("错误: dxcam 库未安装")
    print("安装命令: pip install dxcam")
    sys.exit(1)

try:
    from openvino import Core
except ImportError:
    print("错误: openvino 库未安装")
    sys.exit(1)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='GPU 超高速屏幕检测 - 硬件加速捕获')
    parser.add_argument('--monitor', type=int, default=2,
                       help='显示器编号 (1, 2, 3..., 默认=2)')
    args = parser.parse_args()

    print("="*60)
    print("GPU 超高速屏幕检测 - 硬件加速捕获")
    print("="*60)
    print()

    # 初始化 OpenVINO
    ie = Core()
    print(f"可用设备: {ie.available_devices}")

    if 'GPU' not in ie.available_devices:
        print("错误: GPU 设备不可用")
        return

    gpu_name = ie.get_property('GPU', "FULL_DEVICE_NAME")
    print(f"GPU 设备: {gpu_name}")
    print()

    # 加载模型 - 使用 yolov8l 模型 (平衡性能与精度)
    model_path = "yolov8l_openvino_model/yolov8l.xml"
    print(f"加载模型: {model_path} (Large - 平衡模式)")
    print("强制设备: GPU")

    model = ie.read_model(model=model_path)

    # GPU 优化配置
    config = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
    }

    compiled_model = ie.compile_model(model=model, device_name="GPU", config=config)
    print("模型已编译到 GPU (延迟优化模式)")
    print()

    # 获取输入输出信息
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    inference_size = 640

    print(f"输入shape: {input_layer.shape}")
    print()

    # 初始化屏幕捕获
    import mss
    sct = mss.mss()
    monitors = sct.monitors[1:]

    print("可用显示器:")
    for i, mon in enumerate(monitors, 1):
        print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")
    print()

    # 当前显示器索引
    current_monitor_idx = min(args.monitor, len(monitors))
    monitor = monitors[current_monitor_idx - 1]

    # 初始化 DXCam（硬件加速捕获）
    print(f"初始化 DXCam 硬件加速捕获 (显示器 {current_monitor_idx})...")
    camera = dxcam.create(output_idx=current_monitor_idx-1, output_color="BGR")

    if camera is None:
        print("错误: DXCam 初始化失败，回退到 mss")
        use_dxcam = False
    else:
        print(f"DXCam 初始化成功！")
        use_dxcam = True

    print()

    # 捕获分辨率设置 - 默认480p获得最快速度
    capture_mode = '480p'  # 默认 480p (最快)
    capture_resolution_map = {
        '480p': (854, 480),   # 超快模式
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '2k': (2560, 1440),
        '4k': None  # 原始分辨率
    }

    # 创建窗口
    window_name = 'GPU Ultra Fast Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("="*60)
    print("开始超高速推理...")
    print("控制按键:")
    print("  ESC/q - 退出")
    print("  s - 保存截图")
    print("  0 - 捕获 480p (极速)")
    print("  1 - 捕获 720p")
    print("  2 - 捕获 1080p")
    print("  3 - 捕获 2K")
    print("  4 - 捕获 4K (原始)")
    print("  m - 切换显示器")
    print("="*60)
    print()

    # 统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0

    inference_times = []
    capture_times = []

    try:
        while True:
            # 硬件加速屏幕捕获
            cap_start = time.time()

            if use_dxcam:
                # DXCam 捕获（超快！）
                frame = camera.grab()
                if frame is not None:
                    target_res = capture_resolution_map[capture_mode]
                    if target_res:
                        frame = cv2.resize(frame, target_res, interpolation=cv2.INTER_LINEAR)
            else:
                # 回退到 mss
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                target_res = capture_resolution_map[capture_mode]
                if target_res:
                    frame = cv2.resize(frame, target_res, interpolation=cv2.INTER_LINEAR)

            if frame is None:
                continue

            capture_time = (time.time() - cap_start) * 1000

            capture_times.append(capture_time)
            if len(capture_times) > 100:
                capture_times.pop(0)

            # 将帧调整为 640x640 用于推理和显示
            frame = cv2.resize(frame, (640, 640))

            # 预处理
            input_frame = frame.transpose(2, 0, 1)
            input_frame = np.expand_dims(input_frame, axis=0)
            input_frame = input_frame.astype(np.float32) / 255.0

            # GPU 推理
            infer_start = time.time()
            results = compiled_model([input_frame])[output_layer]
            infer_time = (time.time() - infer_start) * 1000

            inference_times.append(infer_time)
            if len(inference_times) > 100:
                inference_times.pop(0)

            avg_infer_time = sum(inference_times) / len(inference_times)

            # 解析检测结果并绘制框
            detections = results[0].T
            class_scores = detections[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            mask = confidences > 0.3  # 降低阈值以便看到更多检测
            boxes = detections[mask, :4]
            confidences_filtered = confidences[mask]
            class_ids_filtered = class_ids[mask]

            # 绘制检测框 - 不转换坐标，直接使用原始坐标
            h, w = frame.shape[:2]
            person_count = 0
            total_detections = len(boxes)

            for box, conf, class_id in zip(boxes, confidences_filtered, class_ids_filtered):
                x_center, y_center, width, height = box

                # 直接使用 YOLO 输出的坐标（像素坐标）
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                if class_id == 0:  # person
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 计算 FPS
            frame_count += 1
            current_time = time.time()

            if current_time - fps_update_time >= 1.0:
                current_fps = frame_count / (current_time - start_time)
                fps_update_time = current_time

            # 显示
            avg_cap_time = sum(capture_times) / len(capture_times) if capture_times else 0

            cv2.putText(frame, f"Monitor {current_monitor_idx} | {capture_mode.upper()} | FPS: {current_fps:.1f} | People: {person_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Capture: {avg_cap_time:.1f}ms | GPU: {avg_infer_time:.1f}ms",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "DXCam HW Accel" if use_dxcam else "MSS Software",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            cv2.imshow(window_name, frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"screen_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存: {filename}")
            elif key == ord('0'):
                capture_mode = '480p'
                print(f"捕获分辨率切换到 480p - 极速模式")
            elif key == ord('1'):
                capture_mode = '720p'
                print(f"捕获分辨率切换到 720p")
            elif key == ord('2'):
                capture_mode = '1080p'
                print(f"捕获分辨率切换到 1080p")
            elif key == ord('3'):
                capture_mode = '2k'
                print(f"捕获分辨率切换到 2K")
            elif key == ord('4'):
                capture_mode = '4k'
                print(f"捕获分辨率切换到 4K (原始)")
            elif key == ord('m'):
                # 切换显示器
                current_monitor_idx = (current_monitor_idx % len(monitors)) + 1
                monitor = monitors[current_monitor_idx - 1]

                # 重新初始化 DXCam
                if use_dxcam and camera:
                    camera.stop()
                    camera = dxcam.create(output_idx=current_monitor_idx-1, output_color="BGR")
                    if camera is None:
                        print(f"警告: 显示器 {current_monitor_idx} DXCam 初始化失败，使用 MSS")
                        use_dxcam = False
                    else:
                        print(f"已切换到显示器 {current_monitor_idx} ({monitor['width']}x{monitor['height']})")
                else:
                    print(f"已切换到显示器 {current_monitor_idx} ({monitor['width']}x{monitor['height']})")

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        print("正在释放资源...")
        if use_dxcam and camera:
            camera.stop()
        cv2.destroyAllWindows()

        for i in range(10):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0

        print()
        print("="*60)
        print("性能统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  平均屏幕捕获: {avg_capture:.1f} ms")
        print(f"  平均 GPU 推理: {avg_infer:.1f} ms")
        print(f"  捕获方式: {'DXCam 硬件加速' if use_dxcam else 'MSS 软件'}")
        print("="*60)
        print("程序已退出")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

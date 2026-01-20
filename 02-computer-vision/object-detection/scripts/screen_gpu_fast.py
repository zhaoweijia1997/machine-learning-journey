# -*- coding: utf-8 -*-
"""
GPU 高速屏幕检测 - 强制 GPU 加速
使用 yolov8n (最小模型) + 直接 OpenVINO GPU 调用
"""

import cv2
import numpy as np
import time
import sys

try:
    import mss
except ImportError:
    print("错误: mss 库未安装")
    print("安装命令: pip install mss")
    sys.exit(1)

try:
    from openvino import Core
except ImportError:
    print("错误: openvino 库未安装")
    sys.exit(1)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='GPU 高速屏幕检测 - 强制 GPU 模式')
    parser.add_argument('--monitor', type=int, default=2,
                       help='显示器编号 (1, 2, 3..., 默认=2)')
    args = parser.parse_args()

    print("="*60)
    print("GPU 高速屏幕检测 - 强制 GPU 模式")
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

    # 加载模型 - 强制使用 GPU
    model_path = "yolov8n_openvino_model/yolov8n.xml"
    print(f"加载模型: {model_path}")
    print("强制设备: GPU")

    model = ie.read_model(model=model_path)

    # GPU 优化配置
    config = {
        "PERFORMANCE_HINT": "LATENCY",  # 延迟优先（实时应用）
        "NUM_STREAMS": "1",  # 单流实时
    }

    compiled_model = ie.compile_model(model=model, device_name="GPU", config=config)

    print("模型已编译到 GPU (延迟优化模式)")
    print()

    # 获取输入输出信息
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(f"输入shape: {input_layer.shape}")
    print(f"输出shape: {output_layer.shape}")
    print()

    # 获取显示器
    sct = mss.mss()
    monitors = sct.monitors[1:]

    print("可用显示器:")
    for i, mon in enumerate(monitors, 1):
        print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")

    current_monitor_idx = min(args.monitor, len(monitors))
    monitor = monitors[current_monitor_idx - 1]
    print(f"\n使用显示器 {current_monitor_idx}: {monitor['width']}x{monitor['height']}")
    print()

    # 推理分辨率
    inference_size = 640  # YOLOv8 标准输入

    # 创建窗口
    window_name = f'GPU Fast Detection - Monitor {current_monitor_idx}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 移动窗口并设置为 4K 原始分辨率
    if len(monitors) >= 2:
        other_monitor = monitors[0] if current_monitor_idx == 2 else monitors[1]
        cv2.resizeWindow(window_name, monitor['width'], monitor['height'])
        cv2.moveWindow(window_name, other_monitor['left'] + 100, other_monitor['top'] + 100)
        print(f"窗口已移动到显示器 {1 if current_monitor_idx == 2 else 2} (4K原始分辨率)")
    else:
        cv2.resizeWindow(window_name, monitor['width'], monitor['height'])
        cv2.moveWindow(window_name, monitor['left'] + 50, monitor['top'] + 50)

    print()
    print("="*60)
    print("开始 GPU 高速推理...")
    print("控制按键:")
    print("  ESC/q - 退出")
    print("  s - 保存截图")
    print("  1 - 捕获 720p (最快)")
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

    # 屏幕捕获分辨率（影响性能！）
    capture_mode = '1080p'  # 默认 1080p
    capture_resolution_map = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '2k': (2560, 1440),
        '4k': (monitor['width'], monitor['height'])  # 原始分辨率
    }

    inference_times = []
    capture_times = []
    display_times = []

    try:
        while True:
            loop_start = time.time()

            # 捕获屏幕并立即调整到目标分辨率（减少数据量）
            cap_start = time.time()
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 根据捕获模式立即缩放（关键优化！）
            target_cap_width, target_cap_height = capture_resolution_map[capture_mode]
            if capture_mode != '4k':
                frame = cv2.resize(frame, (target_cap_width, target_cap_height), interpolation=cv2.INTER_LINEAR)

            capture_time = (time.time() - cap_start) * 1000

            capture_times.append(capture_time)
            if len(capture_times) > 100:
                capture_times.pop(0)

            # 预处理：缩放到 640x640
            input_frame = cv2.resize(frame, (inference_size, inference_size))
            input_frame = input_frame.transpose(2, 0, 1)  # HWC -> CHW
            input_frame = np.expand_dims(input_frame, axis=0)  # 添加 batch 维度
            input_frame = input_frame.astype(np.float32) / 255.0  # 归一化

            # GPU 推理
            infer_start = time.time()
            results = compiled_model([input_frame])[output_layer]
            infer_time = (time.time() - infer_start) * 1000  # ms

            inference_times.append(infer_time)
            if len(inference_times) > 100:
                inference_times.pop(0)

            avg_infer_time = sum(inference_times) / len(inference_times)

            # 解析检测结果并绘制框
            # YOLOv8 输出格式: [batch, 84, 8400] -> [batch, num_detections, 84]
            # 84 = 4 (bbox) + 80 (classes)
            detections = results[0].T  # 转置为 [8400, 84]

            # 获取置信度最高的类别
            class_scores = detections[:, 4:]  # [8400, 80]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            # 过滤低置信度检测（阈值 0.5）
            mask = confidences > 0.5
            boxes = detections[mask, :4]
            confidences_filtered = confidences[mask]
            class_ids_filtered = class_ids[mask]

            # 绘制检测框
            h, w = frame.shape[:2]
            person_count = 0
            for box, conf, class_id in zip(boxes, confidences_filtered, class_ids_filtered):
                # YOLO 输出是归一化的 (x_center, y_center, width, height)
                x_center, y_center, width, height = box

                # 转换为像素坐标
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h

                # 转换为左上角坐标
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)

                # 只画人（class_id=0）
                if class_id == 0:
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

            # 显示（直接使用捕获的分辨率）
            disp_start = time.time()
            display_frame = frame  # 直接显示，不再缩放

            # 根据分辨率调整字体大小
            if capture_mode in ['2k', '4k']:
                font_scale = 1.5
                thickness = 3
            elif capture_mode == '1080p':
                font_scale = 0.8
                thickness = 2
            else:  # 720p
                font_scale = 0.6
                thickness = 2

            # 性能统计
            avg_cap_time = sum(capture_times) / len(capture_times) if capture_times else 0

            cv2.putText(display_frame, f"Monitor {current_monitor_idx} | {capture_mode.upper()} | FPS: {current_fps:.1f}",
                       (10, int(40*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            cv2.putText(display_frame, f"Cap: {avg_cap_time:.1f}ms | GPU: {avg_infer_time:.1f}ms",
                       (10, int(80*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

            cv2.imshow(window_name, display_frame)

            display_time = (time.time() - disp_start) * 1000
            display_times.append(display_time)
            if len(display_times) > 100:
                display_times.pop(0)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"screen_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存: {filename}")
            elif key == ord('1'):
                capture_mode = '720p'
                print(f"捕获分辨率切换到 720p (1280x720) - 最快")
            elif key == ord('2'):
                capture_mode = '1080p'
                print(f"捕获分辨率切换到 1080p (1920x1080)")
            elif key == ord('3'):
                capture_mode = '2k'
                print(f"捕获分辨率切换到 2K (2560x1440)")
            elif key == ord('4'):
                capture_mode = '4k'
                print(f"捕获分辨率切换到 4K ({monitor['width']}x{monitor['height']}) - 原始")
            elif key == ord('m'):
                # 切换显示器
                current_monitor_idx = (current_monitor_idx % len(monitors)) + 1
                monitor = monitors[current_monitor_idx - 1]
                print(f"已切换到显示器 {current_monitor_idx} ({monitor['width']}x{monitor['height']})")

                # 更新 4K 分辨率映射
                capture_resolution_map['4k'] = (monitor['width'], monitor['height'])

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        print("正在释放资源...")
        sct.close()
        cv2.destroyAllWindows()

        for i in range(10):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0
        avg_display = sum(display_times) / len(display_times) if display_times else 0

        print()
        print("="*60)
        print("性能分析:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  平均屏幕捕获: {avg_capture:.1f} ms")
        print(f"  平均 GPU 推理: {avg_infer:.1f} ms")
        print(f"  平均显示处理: {avg_display:.1f} ms")
        print(f"  总延迟: {avg_capture + avg_infer + avg_display:.1f} ms")
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

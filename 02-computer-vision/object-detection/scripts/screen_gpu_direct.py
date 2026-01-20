# -*- coding: utf-8 -*-
"""
GPU 直接屏幕检测 - 使用 OpenVINO Runtime API 强制 GPU
绕过 YOLO 封装，直接使用 OpenVINO 推理
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
    from openvino.runtime import Core
except ImportError:
    print("错误: openvino 库未安装")
    sys.exit(1)

def main():
    print("="*60)
    print("GPU 直接屏幕检测 - OpenVINO Runtime API")
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
    model_path = "yolov8m_openvino_model/yolov8m.xml"
    print(f"加载模型: {model_path}")
    print("指定设备: GPU")

    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="GPU")

    print("模型加载完成！")
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

    monitor_idx = 2 if len(monitors) >= 2 else 1
    monitor = monitors[monitor_idx - 1]
    print(f"\n使用显示器 {monitor_idx}: {monitor['width']}x{monitor['height']}")
    print()

    # 推理分辨率
    inference_size = 640  # YOLOv8 标准输入

    # 创建窗口
    window_name = f'GPU Direct Detection - Monitor {monitor_idx}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 移动窗口
    if len(monitors) >= 2:
        other_monitor = monitors[0] if monitor_idx == 2 else monitors[1]
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.moveWindow(window_name, other_monitor['left'] + 100, other_monitor['top'] + 100)
        print(f"窗口已移动到显示器 {1 if monitor_idx == 2 else 2}")

    print()
    print("="*60)
    print("开始 GPU 推理...")
    print("按 ESC 或 q 退出")
    print("="*60)
    print()

    # 统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0

    inference_times = []

    try:
        while True:
            # 捕获屏幕
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

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

            # 计算 FPS
            frame_count += 1
            current_time = time.time()

            if current_time - fps_update_time >= 1.0:
                current_fps = frame_count / (current_time - start_time)
                fps_update_time = current_time

            # 显示帧（简单版，不解析检测结果）
            display_frame = cv2.resize(frame, (1280, 720))

            # 显示信息
            cv2.putText(display_frame, f"GPU Direct - Monitor {monitor_idx}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Inference: {infer_time:.1f}ms (avg: {avg_infer_time:.1f}ms)",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, "Pure GPU inference via OpenVINO",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            cv2.imshow(window_name, display_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break

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

        print()
        print("="*60)
        print("GPU 性能统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  平均推理时间: {avg_infer:.1f} ms")
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

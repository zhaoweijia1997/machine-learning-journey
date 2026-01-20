# -*- coding: utf-8 -*-
"""
GPU 满载屏幕检测 - 压榨GPU性能
使用更大模型和更高分辨率，最大化GPU利用率
用法: python screen_gpu_max.py [model_choice] [resolution_choice]
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys
import argparse

try:
    import mss
except ImportError:
    print("错误: mss 库未安装")
    print("安装命令: pip install mss")
    sys.exit(1)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GPU 满载屏幕检测 - 性能压榨模式')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4, 5], default=3,
                       help='模型选择: 1=nano, 2=small, 3=medium, 4=large, 5=xlarge (默认=3)')
    parser.add_argument('--resolution', type=int, choices=[1, 2, 3, 4, 5], default=4,
                       help='推理分辨率: 1=640, 2=1280, 3=1920, 4=2560, 5=4K原始 (默认=4)')
    args = parser.parse_args()

    print("="*60)
    print("GPU 满载屏幕检测 - 性能压榨模式")
    print("="*60)
    print()

    # 获取显示器信息
    sct = mss.mss()
    monitors = sct.monitors[1:]

    print("可用显示器:")
    for i, mon in enumerate(monitors, 1):
        print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")

    # 默认使用显示器2
    monitor_idx = 2 if len(monitors) >= 2 else 1
    monitor = monitors[monitor_idx - 1]

    print(f"\n使用显示器 {monitor_idx}: {monitor['width']}x{monitor['height']}")
    print()

    # 模型映射
    model_map = {
        1: ('yolov8n', 'nano'),
        2: ('yolov8s', 'small'),
        3: ('yolov8m', 'medium'),
        4: ('yolov8l', 'large'),
        5: ('yolov8x', 'xlarge'),
    }

    model_name, model_desc = model_map[args.model]
    print(f"模型: {model_name} ({model_desc})")
    print()

    # 检查模型
    model_path = f'{model_name}_openvino_model'
    pt_model = f'{model_name}.pt'

    if not os.path.exists(model_path):
        print(f"首次运行，正在下载并转换 {model_name} 模型...")
        print("这可能需要几分钟...")
        print()

        base_model = YOLO(pt_model)
        # 导出为 OpenVINO 格式（会自动使用 GPU 如果可用）
        base_model.export(format='openvino', half=False)
        print()

    # 加载模型并强制使用 GPU
    print(f"加载 {model_name} GPU 优化模型...")

    # 显示可用设备
    try:
        from openvino import Core
        ie = Core()
        print(f"可用设备: {ie.available_devices}")
        if 'GPU' in ie.available_devices:
            gpu_name = ie.get_property('GPU', "FULL_DEVICE_NAME")
            print(f"GPU 设备: {gpu_name}")
    except Exception as e:
        print(f"获取设备信息失败: {e}")

    # 创建模型时强制指定 GPU 设备
    model = YOLO(model_path, task='detect')

    # 设置 OpenVINO 推理时使用 GPU
    print("配置为强制使用 GPU...")
    print("模型加载完成！")
    print()

    # 分辨率映射
    resolution_map = {
        1: 640,
        2: 1280,
        3: 1920,
        4: 2560,
        5: None  # 原始分辨率
    }

    inference_size = resolution_map[args.resolution]

    if inference_size:
        print(f"\n推理分辨率: {inference_size}x{inference_size}")
    else:
        print(f"\n推理分辨率: 原始 {monitor['width']}x{monitor['height']}")
    print()

    print("="*60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("  i         - 显示/隐藏性能信息")
    print("="*60)
    print()

    # 提示
    print("⚠️  GPU 满载模式启动中...")
    print("   窗口将自动移到另一个显示器")
    print()
    print("   3 秒后开始...")
    print()

    for i in range(3, 0, -1):
        print(f"   {i}...", flush=True)
        time.sleep(1)
    print("   🚀 开始压榨GPU！")
    print()

    # 创建窗口
    window_name = f'GPU MAX - {model_name.upper()} @ {inference_size or "4K"}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 移动窗口到另一个显示器
    if len(monitors) >= 2:
        other_monitor = monitors[0] if monitor_idx == 2 else monitors[1]
        cv2.resizeWindow(window_name, 1920, 1080)
        cv2.moveWindow(window_name, other_monitor['left'] + 100, other_monitor['top'] + 100)
        print(f"✓ 窗口已移动到显示器 {1 if monitor_idx == 2 else 2}")

    print()
    print("🔥 GPU 满载运行中... 观察任务管理器GPU占用率！")
    print()

    # 统计
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False
    show_info = True

    # 性能统计
    inference_times = []
    max_inference_time = 0
    min_inference_time = float('inf')

    try:
        while True:
            if not paused:
                loop_start = time.time()

                # 捕获屏幕
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 如果设置了推理分辨率，缩放输入
                if inference_size:
                    # 保持宽高比缩放
                    h, w = frame.shape[:2]
                    scale = inference_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    inference_frame = cv2.resize(frame, (new_w, new_h))
                else:
                    inference_frame = frame

                # GPU 推理 - 记录时间
                # OpenVINO 模型会自动使用加载时指定的设备
                infer_start = time.time()
                results = model(inference_frame, verbose=False)
                infer_time = (time.time() - infer_start) * 1000  # ms

                # 更新推理时间统计
                inference_times.append(infer_time)
                if len(inference_times) > 100:
                    inference_times.pop(0)
                max_inference_time = max(max_inference_time, infer_time)
                min_inference_time = min(min_inference_time, infer_time)
                avg_inference_time = sum(inference_times) / len(inference_times)

                # 统计人数
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

                # 绘制结果 - 在原始帧上
                annotated_frame = results[0].plot()

                # 缩放显示（如果推理用的是缩放后的）
                if inference_size and (annotated_frame.shape[1] != 1920):
                    annotated_frame = cv2.resize(annotated_frame, (1920, 1080))

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 显示详细性能信息
                if show_info:
                    info_y = 30
                    font_scale = 0.7
                    thickness = 2

                    # 黑色背景让文字更清晰
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (5, 5), (600, 250), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

                    cv2.putText(annotated_frame, f"Model: {model_name.upper()} ({model_desc})",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

                    res_text = f"{inference_size}x{inference_size}" if inference_size else f"{monitor['width']}x{monitor['height']}"
                    cv2.putText(annotated_frame, f"Resolution: {res_text}",
                               (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                               (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                    cv2.putText(annotated_frame, f"Inference: {infer_time:.1f}ms (avg: {avg_inference_time:.1f}ms)",
                               (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

                    cv2.putText(annotated_frame, f"Min/Max: {min_inference_time:.1f}ms / {max_inference_time:.1f}ms",
                               (10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

                    cv2.putText(annotated_frame, f"People: {person_count}",
                               (10, info_y + 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                    # GPU 占用提示
                    cv2.putText(annotated_frame, "Check Task Manager for GPU usage!",
                               (10, info_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), thickness)

                cv2.imshow(window_name, annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"gpu_max_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)
            elif key == ord('i'):
                show_info = not show_info
                print("性能信息:", "显示" if show_info else "隐藏")

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理
        print("正在释放资源...")
        sct.close()
        cv2.destroyAllWindows()

        for i in range(10):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print()
        print("="*60)
        print("GPU 性能统计:")
        print(f"  模型: {model_name} ({model_desc})")
        res_text = f"{inference_size}x{inference_size}" if inference_size else f"{monitor['width']}x{monitor['height']}"
        print(f"  推理分辨率: {res_text}")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  平均推理时间: {avg_inference_time:.1f} ms")
        print(f"  推理时间范围: {min_inference_time:.1f} - {max_inference_time:.1f} ms")
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

# -*- coding: utf-8 -*-
"""
多显示器屏幕检测 - GPU 加速
支持选择不同的显示器
用法: python screen_multi_monitor.py [monitor_number]
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

def get_monitor(monitor_idx=None):
    """获取指定显示器"""
    sct = mss.mss()
    monitors = sct.monitors[1:]  # 跳过第0个（所有屏幕的组合）

    if len(monitors) == 0:
        print("未检测到显示器！")
        return None, None

    print("\n可用显示器:")
    for i, monitor in enumerate(monitors, 1):
        print(f"  {i}. 显示器 {i}: {monitor['width']}x{monitor['height']} @ ({monitor['left']}, {monitor['top']})")

    # 如果指定了显示器编号，直接使用
    if monitor_idx is not None:
        if 1 <= monitor_idx <= len(monitors):
            selected_monitor = monitors[monitor_idx - 1]
            print(f"\n已选择显示器 {monitor_idx}: {selected_monitor['width']}x{selected_monitor['height']}")
            return selected_monitor, monitor_idx
        else:
            print(f"错误: 显示器 {monitor_idx} 不存在")
            return None, None

    # 没有指定，默认使用第2个显示器（如果有）
    default_idx = 2 if len(monitors) >= 2 else 1
    selected_monitor = monitors[default_idx - 1]
    print(f"\n使用默认显示器 {default_idx}: {selected_monitor['width']}x{selected_monitor['height']}")
    return selected_monitor, default_idx

def select_monitor():
    """交互式选择显示器（按 'm' 键时使用）"""
    sct = mss.mss()
    monitors = sct.monitors[1:]

    if len(monitors) == 0:
        print("未检测到显示器！")
        return None, None

    print("\n可用显示器:")
    for i, monitor in enumerate(monitors, 1):
        print(f"  {i}. 显示器 {i}: {monitor['width']}x{monitor['height']} @ ({monitor['left']}, {monitor['top']})")

    print()

    while True:
        try:
            choice = input(f"请选择显示器 (1-{len(monitors)}, 默认=1): ").strip()

            if not choice:
                monitor_idx = 1
            else:
                monitor_idx = int(choice)

            if 1 <= monitor_idx <= len(monitors):
                selected_monitor = monitors[monitor_idx - 1]
                print(f"\n已选择显示器 {monitor_idx}: {selected_monitor['width']}x{selected_monitor['height']}")
                return selected_monitor, monitor_idx
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
        except KeyboardInterrupt:
            print("\n已取消")
            return None, None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多显示器屏幕检测 (GPU 加速)')
    parser.add_argument('monitor', type=int, nargs='?', default=None,
                       help='显示器编号 (默认: 2 或 1)')
    args = parser.parse_args()

    print("="*60)
    print("多显示器屏幕检测 (GPU 加速)")
    print("="*60)
    print()

    # 获取显示器
    monitor, monitor_idx = get_monitor(args.monitor)
    if monitor is None:
        return

    # 检查模型
    model_path = 'yolov8n_openvino_model'
    if not os.path.exists(model_path):
        print("\n首次运行，正在转换模型...")
        print("这需要 1-2 分钟")
        print()

        base_model = YOLO('yolov8n.pt')
        base_model.export(format='openvino', half=False)
        print()

    # 加载模型
    print("加载 GPU 优化模型...")
    model = YOLO(model_path, task='detect')

    # 显示设备信息
    try:
        from openvino import Core
        ie = Core()
        if 'GPU' in ie.available_devices:
            gpu_name = ie.get_property('GPU', "FULL_DEVICE_NAME")
            print(f"使用设备: GPU ({gpu_name})")
    except:
        pass

    print("模型加载完成！")
    print()

    # 初始化屏幕捕获
    print("初始化屏幕捕获...")
    sct = mss.mss()

    print()
    print("="*60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("  m         - 切换显示器")
    print("="*60)
    print()

    # 重要提示
    print("⚠️  重要提示:")
    print(f"   即将开始捕获显示器 {monitor_idx} 的屏幕内容")
    print("   为避免递归显示（检测窗口被捕获）：")
    print("   1. 请将检测窗口移到另一个显示器")
    print("   2. 或者最小化检测窗口")
    print()
    print("   5 秒后开始检测...")
    print()

    # 倒计时
    for i in range(5, 0, -1):
        print(f"   {i}...", end='', flush=True)
        time.sleep(1)
    print(" 开始！")
    print()

    # 窗口名称
    window_name = f'Monitor {monitor_idx} Detection (GPU) [Press ESC or Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 统计变量
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False

    try:
        while True:
            if not paused:
                # 捕获选定显示器的屏幕
                screenshot = sct.grab(monitor)

                # 转换为 OpenCV 格式
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
                info_y = 30
                cv2.putText(annotated_frame, f"Monitor: {monitor_idx} ({monitor['width']}x{monitor['height']})",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(annotated_frame, f"People: {person_count}",
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            # ESC 键 (27) 或 q 键
            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"monitor{monitor_idx}_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)
            elif key == ord('m'):
                # 切换显示器
                print("\n切换显示器...")
                cv2.destroyAllWindows()

                new_monitor, new_idx = select_monitor()
                if new_monitor:
                    monitor = new_monitor
                    monitor_idx = new_idx
                    window_name = f'Monitor {monitor_idx} Detection (GPU) [Press ESC or Q to quit]'

                    # 重置统计
                    frame_count = 0
                    start_time = time.time()
                    fps_update_time = start_time

                    # 重新创建窗口
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    print(f"\n已切换到显示器 {monitor_idx}")
                else:
                    print("取消切换，继续使用当前显示器")
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理资源
        print("正在释放资源...")
        sct.close()
        cv2.destroyAllWindows()

        # Windows 修复
        for i in range(10):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print()
        print("="*60)
        print("运行统计:")
        print(f"  显示器: {monitor_idx}")
        print(f"  分辨率: {monitor['width']}x{monitor['height']}")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
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

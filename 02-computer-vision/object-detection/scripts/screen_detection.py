# -*- coding: utf-8 -*-
"""
屏幕实时检测 - GPU 加速
捕获屏幕内容进行人形检测
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys

try:
    import mss
except ImportError:
    print("错误: mss 库未安装")
    print("安装命令: pip install mss")
    sys.exit(1)

def main():
    print("="*60)
    print("屏幕实时人形检测 (GPU 加速)")
    print("="*60)
    print()

    # 检查模型
    model_path = 'yolov8n_openvino_model'
    if not os.path.exists(model_path):
        print("首次运行，正在转换模型...")
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

    # 使用主显示器
    monitor = sct.monitors[1]
    print(f"屏幕尺寸: {monitor['width']}x{monitor['height']}")
    print()

    print("="*60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("  r         - 选择区域（全屏/半屏）")
    print("="*60)
    print()
    print("检测开始！正在捕获屏幕...")
    print()

    # 窗口名称
    window_name = 'Screen Detection (GPU) [Press ESC or Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 统计变量
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False
    region_mode = 'full'  # full, half

    try:
        while True:
            if not paused:
                # 根据区域模式调整捕获范围
                if region_mode == 'full':
                    capture_region = monitor
                else:  # half - 捕获右半屏
                    capture_region = {
                        'left': monitor['left'] + monitor['width'] // 2,
                        'top': monitor['top'],
                        'width': monitor['width'] // 2,
                        'height': monitor['height']
                    }

                # 捕获屏幕
                screenshot = sct.grab(capture_region)

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
                region_text = "全屏" if region_mode == 'full' else "右半屏"
                cv2.putText(annotated_frame, f"Source: Screen ({region_text})",
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
                filename = f"screen_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)
            elif key == ord('r'):
                region_mode = 'half' if region_mode == 'full' else 'full'
                region_text = "全屏" if region_mode == 'full' else "右半屏"
                print(f"切换到: {region_text}")

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
        print(f"  输入源: 屏幕捕获")
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

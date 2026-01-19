# -*- coding: utf-8 -*-
"""
多源输入检测 - 支持摄像头和屏幕捕获
GPU 加速实时人形检测
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("警告: mss 未安装，屏幕捕获功能不可用")
    print("安装命令: pip install mss")

def select_input_source():
    """选择输入源"""
    print("选择输入源:")
    print("  1. 摄像头 (Webcam)")
    print("  2. 屏幕捕获 (Screen Capture)")

    if not MSS_AVAILABLE:
        print("\n注意: 屏幕捕获需要安装 mss 库")
        print("      pip install mss")

    print()

    while True:
        try:
            choice = input("请选择 (1-2, 默认=摄像头): ").strip()

            if not choice or choice == '1':
                return 'webcam', None
            elif choice == '2':
                if not MSS_AVAILABLE:
                    print("错误: mss 未安装，无法使用屏幕捕获")
                    continue

                # 显示可用的屏幕
                with mss.mss() as sct:
                    print(f"\n可用屏幕数量: {len(sct.monitors) - 1}")
                    for i, monitor in enumerate(sct.monitors[1:], 1):
                        print(f"  屏幕 {i}: {monitor['width']}x{monitor['height']}")

                screen_choice = input(f"\n选择屏幕 (1-{len(sct.monitors)-1}, 默认=1): ").strip()
                screen_idx = int(screen_choice) if screen_choice else 1

                return 'screen', screen_idx
            else:
                print("无效选择，请输入 1 或 2")
        except ValueError:
            print("请输入数字")
        except KeyboardInterrupt:
            print("\n已取消")
            sys.exit(0)

class ScreenCapture:
    """屏幕捕获类"""
    def __init__(self, monitor_number=1):
        if not MSS_AVAILABLE:
            raise ImportError("mss 库未安装")

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_number]
        self.width = self.monitor['width']
        self.height = self.monitor['height']
        print(f"屏幕捕获: {self.width}x{self.height}")

    def read(self):
        """捕获屏幕，返回 OpenCV 格式的图像"""
        try:
            # 捕获屏幕
            screenshot = self.sct.grab(self.monitor)

            # 转换为 numpy 数组
            img = np.array(screenshot)

            # BGRA -> BGR (OpenCV 格式)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return True, img
        except Exception as e:
            print(f"屏幕捕获错误: {e}")
            return False, None

    def release(self):
        """释放资源"""
        self.sct.close()

    def isOpened(self):
        """检查是否可用"""
        return True

def main():
    print("="*60)
    print("多源输入人形检测 (GPU 加速)")
    print("="*60)
    print()

    # 选择输入源
    source_type, source_param = select_input_source()
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

    # 打开输入源
    if source_type == 'webcam':
        print("正在打开摄像头...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("无法打开摄像头！")
            return

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头已打开: {width}x{height}")
        source_name = "Webcam"

    else:  # screen
        print(f"正在初始化屏幕捕获 (屏幕 {source_param})...")
        try:
            cap = ScreenCapture(monitor_number=source_param)
            width = cap.width
            height = cap.height
            source_name = f"Screen {source_param}"
        except Exception as e:
            print(f"屏幕捕获初始化失败: {e}")
            return

    print()
    print("="*60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("  c         - 显示/隐藏置信度")
    print("="*60)
    print()
    print(f"检测开始！输入源: {source_name}")
    print()

    # 窗口名称
    window_name = f'GPU Detection - {source_name} [Press ESC or Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 统计变量
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    paused = False
    show_confidence = True

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧")
                    break

                # GPU 推理
                results = model(frame, verbose=False)

                # 统计人数
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

                # 绘制结果
                if show_confidence:
                    annotated_frame = results[0].plot()
                else:
                    # 不显示置信度，只画框
                    annotated_frame = frame.copy()
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0:  # person
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 显示信息
                info_y = 30
                cv2.putText(annotated_frame, f"Source: {source_name}",
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
                filename = f"{source_type}_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)
            elif key == ord('c'):
                show_confidence = not show_confidence
                status = "显示" if show_confidence else "隐藏"
                print(f"置信度: {status}")

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理资源
        print("正在释放资源...")
        cap.release()
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
        print(f"  输入源: {source_name}")
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

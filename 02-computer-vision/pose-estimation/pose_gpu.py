# -*- coding: utf-8 -*-
"""
GPU 加速实时姿态估计 - 综合版
支持摄像头和屏幕捕获切换
按 C 键切换摄像头，按 1-9 切换显示器
"""

from ultralytics import YOLO
import cv2
import time
import os
import signal
import sys
import argparse
import numpy as np

# 全局变量
running = True

def signal_handler(sig, frame):
    global running
    print("\n检测到 Ctrl+C，正在退出...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# 尝试导入 DXCam（屏幕捕获加速）
try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False

# MSS 作为备用
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class ScreenCapture:
    """屏幕捕获类"""
    def __init__(self, monitor_idx=0):
        self.monitor_idx = monitor_idx
        self.camera = None
        self.sct = None
        self.use_dxcam = False

        if HAS_DXCAM:
            try:
                self.camera = dxcam.create(output_idx=monitor_idx, output_color="BGR")
                self.camera.start(target_fps=60, video_mode=True)
                self.use_dxcam = True
                print(f"  屏幕捕获: DXCam (显示器 {monitor_idx})")
            except Exception as e:
                print(f"  DXCam 初始化失败: {e}")
                self.camera = None

        if not self.use_dxcam and HAS_MSS:
            self.sct = mss.mss()
            print(f"  屏幕捕获: MSS (显示器 {monitor_idx})")

    def grab(self):
        if self.use_dxcam and self.camera:
            frame = self.camera.get_latest_frame()
            if frame is not None:
                return frame

        if self.sct:
            monitors = self.sct.monitors
            if self.monitor_idx + 1 < len(monitors):
                mon = monitors[self.monitor_idx + 1]
            else:
                mon = monitors[1]
            img = self.sct.grab(mon)
            return np.array(img)[:, :, :3]

        return None

    def release(self):
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
        if self.sct:
            try:
                self.sct.close()
            except:
                pass


def get_monitor_count():
    """获取显示器数量"""
    if HAS_MSS:
        with mss.mss() as sct:
            return len(sct.monitors) - 1
    return 1


def main():
    parser = argparse.ArgumentParser(description='GPU 加速姿态估计')
    parser.add_argument('--mode', type=str, default='camera', choices=['camera', 'screen'],
                       help='初始模式: camera=摄像头, screen=屏幕捕获')
    parser.add_argument('--monitor', type=int, default=0,
                       help='屏幕模式时的显示器索引 (默认=0)')
    parser.add_argument('--resize', type=int, default=0,
                       help='推理分辨率 (0=原始, 640/1080等)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值 (默认=0.5)')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU 加速实时姿态估计 - 综合版")
    print("=" * 60)
    print()

    # 检查 OpenVINO 模型
    openvino_model_path = 'yolov8n-pose_openvino_model'

    if not os.path.exists(openvino_model_path):
        print("首次运行，正在转换模型...")
        model = YOLO('yolov8n-pose.pt')
        model.export(format='openvino', half=False)
        print("模型转换完成！")
        print()

    print("加载模型...")
    model = YOLO(openvino_model_path, task='pose')

    # 检测设备
    try:
        from openvino.runtime import Core
        ie = Core()
        devices = ie.available_devices
        print(f"可用设备: {', '.join(devices)}")
        use_gpu = 'GPU' in devices
    except:
        use_gpu = False

    # 获取显示器信息
    monitor_count = get_monitor_count()
    print(f"检测到 {monitor_count} 个显示器")
    print()

    # 初始化捕获源
    mode = args.mode  # 'camera' 或 'screen'
    monitor_idx = args.monitor
    cap = None
    screen_cap = None

    def init_camera():
        nonlocal cap
        if cap:
            cap.release()
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"摄像头已打开: {w}x{h}")
            return True
        print("无法打开摄像头")
        return False

    def init_screen(idx):
        nonlocal screen_cap
        if screen_cap:
            screen_cap.release()
        screen_cap = ScreenCapture(idx)
        return True

    # 初始化
    if mode == 'camera':
        if not init_camera():
            print("切换到屏幕模式")
            mode = 'screen'
            init_screen(monitor_idx)
    else:
        init_screen(monitor_idx)

    print()
    print("=" * 60)
    print("按键说明:")
    print("  C - 切换到摄像头 (Camera)")
    print("  1-9 - 切换到屏幕捕获 (显示器 1-9)")
    print("  ESC/Q - 退出")
    print("  X 按钮 - 关闭窗口")
    print("  S - 保存截图")
    print("  空格 - 暂停/继续")
    print("=" * 60)
    print()

    # 创建窗口
    window_name = 'Pose Estimation [C=Camera, 1-9=Screen, ESC=Quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # 状态变量
    frame_count = 0
    start_time = time.time()
    current_fps = 0
    fps_update_time = start_time
    paused = False
    inference_times = []
    capture_times = []

    global running
    annotated_frame = None

    try:
        while running:
            # 检查窗口
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n窗口已关闭...")
                    break
            except:
                break

            if not paused:
                # 捕获
                capture_start = time.time()

                if mode == 'camera' and cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        frame = None
                elif mode == 'screen' and screen_cap:
                    frame = screen_cap.grab()
                else:
                    frame = None

                capture_time = (time.time() - capture_start) * 1000

                if frame is None:
                    cv2.waitKey(10)
                    continue

                capture_times.append(capture_time)
                if len(capture_times) > 50:
                    capture_times.pop(0)

                # 调整推理分辨率
                if args.resize > 0:
                    h, w = frame.shape[:2]
                    scale = args.resize / max(h, w)
                    infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    infer_frame = frame

                # GPU 推理
                infer_start = time.time()
                results = model(infer_frame, conf=args.conf, verbose=False)
                infer_time = (time.time() - infer_start) * 1000

                inference_times.append(infer_time)
                if len(inference_times) > 50:
                    inference_times.pop(0)

                # 如果 resize 了，需要将结果缩放回原始尺寸
                if args.resize > 0 and len(results[0].boxes) > 0:
                    h_orig, w_orig = frame.shape[:2]
                    h_infer, w_infer = infer_frame.shape[:2]
                    scale_x = w_orig / w_infer
                    scale_y = h_orig / h_infer

                    # 绘制在原始帧上
                    annotated_frame = frame.copy()

                    # 绘制关键点和骨架
                    if results[0].keypoints is not None:
                        for person_kp in results[0].keypoints:
                            kp_xy = person_kp.xy[0].cpu().numpy()  # (17, 2)
                            kp_conf = person_kp.conf[0].cpu().numpy() if person_kp.conf is not None else np.ones(17)

                            # 缩放关键点
                            kp_xy[:, 0] *= scale_x
                            kp_xy[:, 1] *= scale_y

                            # 骨架连接
                            skeleton = [
                                (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
                                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
                                (5, 11), (6, 12), (11, 12),  # 躯干
                                (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
                            ]

                            # 绘制骨架线
                            for i, j in skeleton:
                                if kp_conf[i] > 0.5 and kp_conf[j] > 0.5:
                                    pt1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
                                    pt2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
                                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)

                            # 绘制关键点
                            for idx, (x, y) in enumerate(kp_xy):
                                if kp_conf[idx] > 0.5:
                                    cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 0, 255), -1)

                    # 绘制边界框
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                else:
                    annotated_frame = results[0].plot()

                # 统计
                person_count = len(results[0].keypoints) if results[0].keypoints is not None else 0
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 0.5:
                    elapsed = current_time - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    fps_update_time = current_time

                # 计算平均时间
                avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0
                avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0

                # 显示信息（右下角）
                h_frame, w_frame = annotated_frame.shape[:2]
                line_height = 22
                font_scale = 0.45
                thickness = 1

                mode_text = "Camera" if mode == 'camera' else f"Screen {monitor_idx}"
                resize_text = f"{args.resize}" if args.resize > 0 else "Ori"

                texts = [
                    (f"FPS: {current_fps:.1f} | Mode: {mode_text} | GPU: {'ON' if use_gpu else 'OFF'}", (0, 255, 0)),
                    (f"People: {person_count} | Resize: {resize_text}", (0, 255, 0)),
                    (f"Capture: {avg_capture:.1f}ms | Inference: {avg_inference:.1f}ms", (0, 255, 255)),
                    (f"[C]=Camera [1-9]=Screen [ESC]=Quit", (200, 200, 200))
                ]

                for i, (text, color) in enumerate(reversed(texts)):
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = w_frame - text_size[0] - 10
                    text_y = h_frame - 10 - (i * line_height)
                    cv2.putText(annotated_frame, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                cv2.imshow(window_name, annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q') or key == ord('Q'):
                print("\n正在退出...")
                break
            elif key == ord('c') or key == ord('C'):
                print("\n切换到摄像头模式...")
                if init_camera():
                    mode = 'camera'
                else:
                    print("摄像头不可用")
            elif ord('1') <= key <= ord('9'):
                new_monitor = key - ord('1')
                if new_monitor < monitor_count:
                    print(f"\n切换到屏幕 {new_monitor}...")
                    monitor_idx = new_monitor
                    init_screen(monitor_idx)
                    mode = 'screen'
                else:
                    print(f"\n显示器 {new_monitor} 不存在")
            elif key == ord('s') or key == ord('S'):
                if annotated_frame is not None:
                    filename = f"pose_{mode}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}")

    except Exception as e:
        print(f"\n错误: {e}")

    finally:
        print("正在释放资源...")
        if cap:
            cap.release()
        if screen_cap:
            screen_cap.release()

        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

        # 统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print()
        print("=" * 60)
        print(f"运行统计: {frame_count} 帧, {total_time:.1f} 秒, 平均 {avg_fps:.1f} FPS")
        print("=" * 60)


if __name__ == "__main__":
    main()

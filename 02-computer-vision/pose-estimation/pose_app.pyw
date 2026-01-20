# -*- coding: utf-8 -*-
"""
GPU 加速姿态估计 - GUI 版本 (带实时控制面板)
运行时可通过 UI 切换摄像头/屏幕
"""

import tkinter as tk
from tkinter import ttk
import threading
import cv2
import numpy as np
import time
import os
import ctypes

# Windows 高分辨率 DPI 适配
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor DPI Aware
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

# 全局变量
running = False
model = None

try:
    import mss
    HAS_MSS = True
except:
    HAS_MSS = False


class FastScreenCapture:
    """屏幕捕获 - MSS 稳定版"""
    def __init__(self, monitor_idx=0):
        self.monitor_idx = monitor_idx
        self.sct = None
        self.monitor = None

        if HAS_MSS:
            self.sct = mss.mss()
            monitors = self.sct.monitors
            idx = monitor_idx + 1
            if idx < len(monitors):
                self.monitor = monitors[idx]
            else:
                self.monitor = monitors[1]

    def grab(self):
        if self.sct and self.monitor:
            try:
                img = self.sct.grab(self.monitor)
                frame = np.array(img)
                return frame[:, :, :3].copy()
            except:
                pass
        return None

    def release(self):
        if self.sct:
            try:
                self.sct.close()
            except:
                pass
            self.sct = None


def get_monitor_count():
    if HAS_MSS:
        with mss.mss() as sct:
            return len(sct.monitors) - 1
    return 1


class PoseApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("姿态估计 - Pose Estimation")
        self.root.geometry("400x550")
        self.root.resizable(False, False)
        self.center_window()
        self.is_running = False
        self.detection_running = False

        # 运行时控制变量
        self.mode_switch_request = None  # ('camera', None) or ('screen', monitor_idx)
        self.stop_request = False

        self.create_ui()

    def center_window(self):
        self.root.update_idletasks()
        w, h = 400, 550
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def create_ui(self):
        # 标题
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="姿态估计 Pose Estimation",
                font=('Microsoft YaHei UI', 16, 'bold'),
                fg='white', bg='#2c3e50').pack(pady=20)

        main_frame = tk.Frame(self.root, padx=30, pady=15)
        main_frame.pack(fill='both', expand=True)

        # 输入源选择
        mode_frame = tk.LabelFrame(main_frame, text="输入源 (运行时可切换)", font=('Microsoft YaHei UI', 10))
        mode_frame.pack(fill='x', pady=8)

        self.mode_var = tk.StringVar(value='screen')

        # 摄像头按钮
        cam_frame = tk.Frame(mode_frame)
        cam_frame.pack(fill='x', padx=10, pady=5)

        ttk.Radiobutton(cam_frame, text="摄像头",
                       variable=self.mode_var, value='camera').pack(side='left')

        self.cam_btn = tk.Button(cam_frame, text="切换", width=6,
                                command=lambda: self.switch_to_camera(),
                                state='disabled', bg='#3498db', fg='white')
        self.cam_btn.pack(side='right', padx=5)

        # 屏幕捕获
        screen_frame = tk.Frame(mode_frame)
        screen_frame.pack(fill='x', padx=10, pady=5)

        ttk.Radiobutton(screen_frame, text="屏幕捕获",
                       variable=self.mode_var, value='screen').pack(side='left')

        # 显示器选择和切换按钮
        mon_frame = tk.Frame(mode_frame)
        mon_frame.pack(fill='x', padx=30, pady=5)

        tk.Label(mon_frame, text="显示器:").pack(side='left')
        self.monitor_var = tk.StringVar(value='0')
        self.monitor_combo = ttk.Combobox(mon_frame, textvariable=self.monitor_var,
                    values=[str(i) for i in range(get_monitor_count())],
                    width=5, state='readonly')
        self.monitor_combo.pack(side='left', padx=5)

        self.screen_btn = tk.Button(mon_frame, text="切换", width=6,
                                   command=lambda: self.switch_to_screen(),
                                   state='disabled', bg='#9b59b6', fg='white')
        self.screen_btn.pack(side='right', padx=5)

        # 性能设置
        perf_frame = tk.LabelFrame(main_frame, text="性能设置", font=('Microsoft YaHei UI', 10))
        perf_frame.pack(fill='x', pady=8)

        # 分辨率
        res_frame = tk.Frame(perf_frame)
        res_frame.pack(fill='x', padx=20, pady=3)
        tk.Label(res_frame, text="推理分辨率:").pack(side='left')
        self.resize_var = tk.StringVar(value='480')
        ttk.Combobox(res_frame, textvariable=self.resize_var,
                    values=['320', '480', '640', '720'],
                    width=8, state='readonly').pack(side='left', padx=5)
        tk.Label(res_frame, text="(320最快)", fg='gray').pack(side='left')

        # 置信度
        conf_frame = tk.Frame(perf_frame)
        conf_frame.pack(fill='x', padx=20, pady=3)
        tk.Label(conf_frame, text="置信度:").pack(side='left')
        self.conf_var = tk.StringVar(value='0.5')
        ttk.Combobox(conf_frame, textvariable=self.conf_var,
                    values=['0.3', '0.4', '0.5', '0.6'],
                    width=8, state='readonly').pack(side='left', padx=5)

        # 状态显示
        self.status_label = tk.Label(main_frame, text="准备就绪",
                                    font=('Microsoft YaHei UI', 10), fg='#27ae60')
        self.status_label.pack(pady=8)

        # 实时信息
        self.info_label = tk.Label(main_frame, text="",
                                  font=('Microsoft YaHei UI', 9), fg='#7f8c8d')
        self.info_label.pack(pady=2)

        # 按钮区域
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="启动检测",
                                  font=('Microsoft YaHei UI', 11, 'bold'),
                                  bg='#27ae60', fg='white', width=12, height=2,
                                  command=self.start_detection, cursor='hand2')
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = tk.Button(btn_frame, text="停止",
                                 font=('Microsoft YaHei UI', 11, 'bold'),
                                 bg='#e74c3c', fg='white', width=8, height=2,
                                 command=self.stop_detection, cursor='hand2',
                                 state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        tk.Label(main_frame, text="快捷键: C=摄像头 | 1-9=屏幕 | ESC=退出",
                font=('Microsoft YaHei UI', 9), fg='gray').pack(pady=5)

    def switch_to_camera(self):
        """运行时切换到摄像头"""
        if self.detection_running:
            self.mode_switch_request = ('camera', None)
            self.update_status("切换到摄像头...", '#f39c12')

    def switch_to_screen(self):
        """运行时切换到屏幕"""
        if self.detection_running:
            monitor_idx = int(self.monitor_var.get())
            self.mode_switch_request = ('screen', monitor_idx)
            self.update_status(f"切换到屏幕 {monitor_idx}...", '#f39c12')

    def update_status(self, text, color='#27ae60'):
        try:
            self.status_label.config(text=text, fg=color)
            self.root.update()
        except:
            pass

    def update_info(self, text):
        try:
            self.info_label.config(text=text)
            self.root.update()
        except:
            pass

    def start_detection(self):
        if self.is_running:
            return
        self.is_running = True
        self.stop_request = False
        self.start_btn.config(state='disabled', bg='gray')
        self.stop_btn.config(state='normal')
        self.cam_btn.config(state='normal')
        self.screen_btn.config(state='normal')
        self.update_status("加载模型...", '#f39c12')
        threading.Thread(target=self.run_detection, daemon=True).start()

    def stop_detection(self):
        """停止检测"""
        self.stop_request = True
        self.update_status("正在停止...", '#f39c12')

    def run_detection(self):
        global model, running

        try:
            from ultralytics import YOLO

            model_path = 'yolov8n-pose_openvino_model'
            if not os.path.exists(model_path):
                self.update_status("转换模型中...", '#f39c12')
                YOLO('yolov8n-pose.pt').export(format='openvino', half=False)

            self.update_status("加载中...", '#f39c12')
            model = YOLO(model_path, task='pose')

            # 预热
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(3):
                model(dummy, verbose=False)

            # 参数
            mode = self.mode_var.get()
            monitor_idx = int(self.monitor_var.get())
            resize_val = int(self.resize_var.get())
            conf_val = float(self.conf_var.get())

            self.update_status("运行中", '#27ae60')
            self.detection_running = True
            self.detection_loop(mode, monitor_idx, resize_val, conf_val)

        except Exception as e:
            self.update_status(f"错误: {e}", '#e74c3c')
        finally:
            self.detection_running = False
            self.reset_ui()

    def detection_loop(self, mode, monitor_idx, resize_val, conf_val):
        global model, running

        # 初始化捕获
        cap = None
        screen_cap = None

        if mode == 'camera':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            screen_cap = FastScreenCapture(monitor_idx)

        window_name = 'Pose Estimation [C=Cam, 1-9=Screen, ESC=Quit]'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        running = True
        frame_count = 0
        start_time = time.time()
        fps = 0
        inference_times = []
        last_keypoints_scaled = []
        monitor_count = get_monitor_count()
        last_update_time = time.time()

        try:
            while running and not self.stop_request:
                # 检查窗口
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    break

                # 检查模式切换请求
                if self.mode_switch_request:
                    new_mode, new_monitor = self.mode_switch_request
                    self.mode_switch_request = None

                    # 释放旧资源
                    if cap:
                        cap.release()
                        cap = None
                    if screen_cap:
                        screen_cap.release()
                        screen_cap = None

                    # 切换
                    if new_mode == 'camera':
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        mode = 'camera'
                        self.update_status("摄像头模式", '#27ae60')
                    else:
                        monitor_idx = new_monitor
                        screen_cap = FastScreenCapture(monitor_idx)
                        mode = 'screen'
                        self.update_status(f"屏幕 {monitor_idx} 模式", '#27ae60')

                # 捕获
                if mode == 'camera' and cap:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                else:
                    frame = screen_cap.grab() if screen_cap else None
                    if frame is None:
                        time.sleep(0.001)
                        continue

                frame_count += 1
                h_orig, w_orig = frame.shape[:2]

                # 推理
                scale = resize_val / max(h_orig, w_orig)
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                infer_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                t1 = time.perf_counter()
                results = model(infer_frame, conf=conf_val, verbose=False)
                infer_time = (time.perf_counter() - t1) * 1000

                inference_times.append(infer_time)
                if len(inference_times) > 30:
                    inference_times.pop(0)

                # 缩放结果
                scale_x = w_orig / new_w
                scale_y = h_orig / new_h

                last_keypoints_scaled = []
                if results[0].keypoints is not None:
                    for kp in results[0].keypoints:
                        kp_xy = kp.xy[0].cpu().numpy().copy()
                        kp_conf = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(17)
                        kp_xy[:, 0] *= scale_x
                        kp_xy[:, 1] *= scale_y
                        last_keypoints_scaled.append((kp_xy, kp_conf))

                # 绘制
                display = frame.copy()
                skeleton = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                           (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

                for kp_xy, kp_conf in last_keypoints_scaled:
                    for i, j in skeleton:
                        if kp_conf[i] > 0.5 and kp_conf[j] > 0.5:
                            pt1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
                            pt2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
                            cv2.line(display, pt1, pt2, (0, 255, 0), 2)

                    for idx, (x, y) in enumerate(kp_xy):
                        if kp_conf[idx] > 0.5:
                            cv2.circle(display, (int(x), int(y)), 5, (0, 0, 255), -1)

                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                avg_inf = sum(inference_times) / len(inference_times) if inference_times else 0

                # 信息显示
                h_disp, w_disp = display.shape[:2]
                texts = [
                    (f"FPS: {fps:.1f} | People: {len(last_keypoints_scaled)}", (0, 255, 0)),
                    (f"Inference: {avg_inf:.1f}ms", (0, 255, 255)),
                    (f"Mode: {'Camera' if mode=='camera' else f'Screen {monitor_idx}'}", (200, 200, 200))
                ]

                for i, (text, color) in enumerate(reversed(texts)):
                    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(display, text, (w_disp - sz[0] - 10, h_disp - 10 - i * 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imshow(window_name, display)

                # 更新 UI 信息 (每秒)
                current_time = time.time()
                if current_time - last_update_time >= 1.0:
                    mode_text = "摄像头" if mode == 'camera' else f"屏幕 {monitor_idx}"
                    self.update_info(f"FPS: {fps:.1f} | 推理: {avg_inf:.1f}ms | 人数: {len(last_keypoints_scaled)} | {mode_text}")
                    last_update_time = current_time

                # 按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    self.mode_switch_request = ('camera', None)
                elif ord('1') <= key <= ord('9'):
                    new_mon = key - ord('1')
                    if new_mon < monitor_count:
                        self.mode_switch_request = ('screen', new_mon)
                elif key == ord('s'):
                    cv2.imwrite(f"pose_{int(time.time())}.jpg", display)

        finally:
            if cap:
                cap.release()
            if screen_cap:
                screen_cap.release()
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)

    def reset_ui(self):
        self.is_running = False
        self.detection_running = False
        try:
            self.start_btn.config(state='normal', bg='#27ae60')
            self.stop_btn.config(state='disabled')
            self.cam_btn.config(state='disabled')
            self.screen_btn.config(state='disabled')
            self.update_status("准备就绪", '#27ae60')
            self.update_info("")
        except:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PoseApp()
    app.run()

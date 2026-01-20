# -*- coding: utf-8 -*-
"""
人脸识别应用 - GUI 版本
支持摄像头/屏幕识别 + 人脸录入
"""

import ctypes
# Windows 高分辨率 DPI 适配
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor DPI Aware
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import os

# 导入人脸数据库
from face_register import FaceDatabase, HAS_OPENVINO

# 屏幕捕获
try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class FaceRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("人脸识别系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # 状态变量
        self.running = False
        self.mode = None  # "camera" or "screen"
        self.mode_switch_request = None
        self.current_monitor = 0

        # 人脸数据库
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "face_db")
        self.db = FaceDatabase(db_path)

        # 屏幕捕获
        self.dxcam_camera = None
        self.mss_sct = None

        # 性能统计
        self.fps = 0
        self.inference_time = 0
        self.last_frame_time = time.time()
        self.frame_count = 0

        # 录入模式
        self.register_mode = False
        self.register_name = ""
        self.register_title = ""

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        """设置UI"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧 - 视频显示
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 右侧 - 控制面板
        right_frame = ttk.Frame(main_frame, width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # 模式选择
        mode_frame = ttk.LabelFrame(right_frame, text="输入模式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.cam_btn = ttk.Button(mode_frame, text="摄像头", command=self._start_camera)
        self.cam_btn.pack(fill=tk.X, pady=2)

        self.screen_btn = ttk.Button(mode_frame, text="屏幕捕获", command=self._start_screen)
        self.screen_btn.pack(fill=tk.X, pady=2)

        # 显示器选择
        monitor_frame = ttk.Frame(mode_frame)
        monitor_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(monitor_frame, text="显示器:").pack(side=tk.LEFT)

        self.monitor_var = tk.StringVar(value="0")
        self.monitor_combo = ttk.Combobox(monitor_frame, textvariable=self.monitor_var,
                                          values=["0", "1", "2"], width=5, state="readonly")
        self.monitor_combo.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(mode_frame, text="停止", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=(10, 0))

        # 人脸录入
        register_frame = ttk.LabelFrame(right_frame, text="人脸录入", padding=10)
        register_frame.pack(fill=tk.X, pady=(0, 10))

        self.register_btn = ttk.Button(register_frame, text="开始录入", command=self._toggle_register)
        self.register_btn.pack(fill=tk.X, pady=2)

        self.capture_btn = ttk.Button(register_frame, text="拍照录入 (S)", command=self._capture_face, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, pady=2)

        # 数据库管理
        db_frame = ttk.LabelFrame(right_frame, text="数据库管理", padding=10)
        db_frame.pack(fill=tk.X, pady=(0, 10))

        self.list_btn = ttk.Button(db_frame, text="查看已录入", command=self._show_faces)
        self.list_btn.pack(fill=tk.X, pady=2)

        self.delete_btn = ttk.Button(db_frame, text="删除人脸", command=self._delete_face)
        self.delete_btn.pack(fill=tk.X, pady=2)

        # 已录入人脸列表
        list_frame = ttk.LabelFrame(right_frame, text="已录入人脸", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.face_listbox = tk.Listbox(list_frame, height=8)
        self.face_listbox.pack(fill=tk.BOTH, expand=True)
        self._refresh_face_list()

        # 状态显示
        status_frame = ttk.LabelFrame(right_frame, text="状态", padding=10)
        status_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(anchor=tk.W)

        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)

        self.inference_label = ttk.Label(status_frame, text="推理: 0 ms")
        self.inference_label.pack(anchor=tk.W)

        # 识别信息
        self.recognized_label = ttk.Label(status_frame, text="识别: -", wraplength=250)
        self.recognized_label.pack(anchor=tk.W, pady=(5, 0))

    def _bind_events(self):
        """绑定事件"""
        self.root.bind('<s>', lambda e: self._capture_face())
        self.root.bind('<S>', lambda e: self._capture_face())
        self.root.bind('<Escape>', lambda e: self._stop())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_face_list(self):
        """刷新已录入人脸列表"""
        self.face_listbox.delete(0, tk.END)
        for name, info in self.db.face_info.items():
            title = info.get("title", "")
            count = info.get("photo_count", 0)
            display = f"{name} ({title}) - {count}张" if title else f"{name} - {count}张"
            self.face_listbox.insert(tk.END, display)

    def _start_camera(self):
        """启动摄像头"""
        if self.running and self.mode == "camera":
            return
        self.mode_switch_request = "camera"
        if not self.running:
            self._start_capture()

    def _start_screen(self):
        """启动屏幕捕获"""
        if self.running and self.mode == "screen":
            return
        self.current_monitor = int(self.monitor_var.get())
        self.mode_switch_request = "screen"
        if not self.running:
            self._start_capture()

    def _start_capture(self):
        """启动捕获"""
        self.running = True
        self.stop_btn.config(state=tk.NORMAL)
        self.cam_btn.config(state=tk.DISABLED)
        self.screen_btn.config(state=tk.DISABLED)

        # 启动捕获线程
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _stop(self):
        """停止捕获"""
        self.running = False
        self.mode = None
        self.register_mode = False

        self.stop_btn.config(state=tk.DISABLED)
        self.cam_btn.config(state=tk.NORMAL)
        self.screen_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.register_btn.config(text="开始录入")

        self.status_label.config(text="已停止")

    def _capture_loop(self):
        """捕获主循环"""
        cap = None
        dxcam_camera = None

        while self.running:
            # 检查模式切换请求
            if self.mode_switch_request:
                # 清理旧资源
                if cap:
                    cap.release()
                    cap = None
                if dxcam_camera:
                    try:
                        dxcam_camera.stop()
                    except:
                        pass
                    dxcam_camera = None

                self.mode = self.mode_switch_request
                self.mode_switch_request = None

                if self.mode == "camera":
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.root.after(0, lambda: self.status_label.config(text="摄像头模式"))
                elif self.mode == "screen":
                    if HAS_DXCAM:
                        try:
                            dxcam_camera = dxcam.create(output_idx=self.current_monitor)
                            dxcam_camera.start(target_fps=30)
                        except Exception as e:
                            print(f"DXCam 初始化失败: {e}")
                            dxcam_camera = None
                    self.root.after(0, lambda: self.status_label.config(text=f"屏幕模式 (显示器 {self.current_monitor})"))

            # 获取帧
            frame = None
            if self.mode == "camera" and cap:
                ret, frame = cap.read()
                if not ret:
                    continue
            elif self.mode == "screen":
                if dxcam_camera:
                    frame = dxcam_camera.get_latest_frame()
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif HAS_MSS:
                    with mss.mss() as sct:
                        monitor = sct.monitors[self.current_monitor + 1]
                        screenshot = sct.grab(monitor)
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if frame is None:
                time.sleep(0.01)
                continue

            # 缩放大帧以提高处理速度
            h, w = frame.shape[:2]
            max_size = 1280
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # 人脸识别
            start_time = time.time()
            results = self.db.recognize_face(frame, threshold=0.5)
            self.inference_time = (time.time() - start_time) * 1000

            # 绘制结果
            display_frame = self._draw_results(frame, results)

            # 更新 FPS
            self.frame_count += 1
            elapsed = time.time() - self.last_frame_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_frame_time = time.time()

            # 更新显示
            self._update_display(display_frame, results)

            time.sleep(0.01)

        # 清理
        if cap:
            cap.release()
        if dxcam_camera:
            try:
                dxcam_camera.stop()
            except:
                pass

    def _draw_results(self, frame, results):
        """绘制识别结果"""
        display = frame.copy()

        for result in results:
            bbox = result["bbox"]
            name = result["name"]
            title = result.get("title", "")
            confidence = result["confidence"]

            # 颜色: 已知-绿色, 未知-红色
            color = (0, 255, 0) if name != "未知" else (0, 0, 255)

            # 绘制边框
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # 绘制标签背景
            label = f"{name}"
            if title:
                label += f" ({title})"
            if confidence > 0:
                label += f" {confidence:.0%}"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (bbox[0], bbox[1] - 25),
                         (bbox[0] + label_size[0] + 10, bbox[1]), color, -1)

            # 绘制文字
            cv2.putText(display, label, (bbox[0] + 5, bbox[1] - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 录入模式提示
        if self.register_mode:
            cv2.putText(display, f"REGISTER MODE: {self.register_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Press S to capture", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display

    def _update_display(self, frame, results):
        """更新显示"""
        # 转换为 PIL 图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # 缩放以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            img_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo)

        # 更新状态
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.inference_label.config(text=f"推理: {self.inference_time:.1f} ms")

        # 更新识别信息
        if results:
            recognized = [f"{r['name']}" + (f"({r['title']})" if r.get('title') else "") for r in results if r['name'] != "未知"]
            unknown_count = sum(1 for r in results if r['name'] == "未知")

            info = ""
            if recognized:
                info += "已识别: " + ", ".join(recognized)
            if unknown_count > 0:
                if info:
                    info += f" | 未知: {unknown_count}"
                else:
                    info = f"未知: {unknown_count}"
            self.recognized_label.config(text=info if info else "识别: -")
        else:
            self.recognized_label.config(text="识别: 未检测到人脸")

    def _toggle_register(self):
        """切换录入模式"""
        if self.register_mode:
            self.register_mode = False
            self.register_btn.config(text="开始录入")
            self.capture_btn.config(state=tk.DISABLED)
        else:
            # 获取姓名
            name = simpledialog.askstring("人脸录入", "请输入姓名:", parent=self.root)
            if not name or not name.strip():
                return

            title = simpledialog.askstring("人脸录入", "请输入职称/备注 (可选):", parent=self.root)

            self.register_name = name.strip()
            self.register_title = title.strip() if title else ""
            self.register_mode = True
            self.register_btn.config(text="取消录入")
            self.capture_btn.config(state=tk.NORMAL)

            messagebox.showinfo("提示", f"录入模式已开启\n姓名: {self.register_name}\n\n请面向摄像头，按 S 键拍照录入")

    def _capture_face(self):
        """拍照录入"""
        if not self.register_mode or not self.running:
            return

        # 获取当前帧进行录入
        # 这里简化处理，使用 db 直接从摄像头获取
        if self.mode == "camera":
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                success, message = self.db.register_face(frame, self.register_name, self.register_title)
                if success:
                    messagebox.showinfo("成功", message)
                    self._refresh_face_list()
                else:
                    messagebox.showwarning("失败", message)
        else:
            messagebox.showwarning("提示", "请使用摄像头模式进行人脸录入")

    def _show_faces(self):
        """显示已录入人脸"""
        if not self.db.face_info:
            messagebox.showinfo("信息", "数据库为空，请先录入人脸")
            return

        info = "已录入人脸:\n" + "=" * 30 + "\n"
        for name, face_info in self.db.face_info.items():
            title = face_info.get("title", "")
            count = face_info.get("photo_count", 0)
            added = face_info.get("added_time", "")
            info += f"\n{name}"
            if title:
                info += f" ({title})"
            info += f"\n  照片数: {count}\n  添加时间: {added}\n"

        messagebox.showinfo("已录入人脸", info)

    def _delete_face(self):
        """删除人脸"""
        selection = self.face_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择要删除的人脸")
            return

        # 解析选中项
        selected_text = self.face_listbox.get(selection[0])
        name = selected_text.split(" ")[0]

        if messagebox.askyesno("确认", f"确定要删除 {name} 的人脸数据吗？"):
            success, message = self.db.delete_face(name)
            if success:
                messagebox.showinfo("成功", message)
                self._refresh_face_list()
            else:
                messagebox.showerror("失败", message)

    def _on_close(self):
        """关闭窗口"""
        self.running = False
        time.sleep(0.1)
        self.root.destroy()

    def run(self):
        """运行应用"""
        if not HAS_OPENVINO:
            messagebox.showerror("错误", "OpenVINO 未安装，请运行:\npip install openvino")
            return

        self.root.mainloop()


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()

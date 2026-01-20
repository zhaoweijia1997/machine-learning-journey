# -*- coding: utf-8 -*-
"""
系统托盘屏幕检测 - GPU 加速
后台运行，通过系统托盘控制
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys
import threading

try:
    import mss
except ImportError:
    print("错误: mss 库未安装")
    print("安装命令: pip install mss")
    sys.exit(1)

try:
    import pystray
    from pystray import MenuItem as item
    from PIL import Image, ImageDraw
except ImportError:
    print("错误: pystray 库未安装")
    print("安装命令: pip install pystray pillow")
    sys.exit(1)

try:
    import tkinter as tk
except ImportError:
    print("错误: tkinter 未安装")
    sys.exit(1)

class ScreenDetector:
    """屏幕检测器类"""
    def __init__(self):
        self.model = None
        self.sct = mss.mss()
        self.region = None
        self.running = False
        self.paused = False
        self.detection_thread = None
        self.window_name = 'Screen Detection (GPU)'

        # 统计
        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0

    def load_model(self):
        """加载模型"""
        model_path = 'yolov8n_openvino_model'
        if not os.path.exists(model_path):
            print("首次运行，正在转换模型...")
            base_model = YOLO('yolov8n.pt')
            base_model.export(format='openvino', half=False)

        self.model = YOLO(model_path, task='detect')
        print("模型加载完成！")

    def select_region(self):
        """选择区域"""
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)
        root.attributes('-topmost', True)

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        canvas = tk.Canvas(root, cursor="cross", bg='black')
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.create_text(
            screen_width // 2, 50,
            text="拖动鼠标选择区域 | Enter=全屏 | ESC=取消",
            fill="white", font=("Arial", 16, "bold")
        )

        start_x = None
        start_y = None
        rect_id = None
        selected_region = None

        def on_button_press(event):
            nonlocal start_x, start_y, rect_id
            start_x = event.x
            start_y = event.y
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(
                start_x, start_y, start_x, start_y,
                outline='red', width=3
            )

        def on_mouse_drag(event):
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, event.x, event.y)

        def on_button_release(event):
            nonlocal selected_region
            x1 = min(start_x, event.x)
            y1 = min(start_y, event.y)
            x2 = max(start_x, event.x)
            y2 = max(start_y, event.y)

            width = x2 - x1
            height = y2 - y1

            if width > 50 and height > 50:
                selected_region = {
                    'left': x1, 'top': y1,
                    'width': width, 'height': height
                }
                root.quit()
                root.destroy()

        def on_escape(event):
            root.quit()
            root.destroy()

        def on_enter(event):
            nonlocal selected_region
            selected_region = {
                'left': 0, 'top': 0,
                'width': screen_width, 'height': screen_height
            }
            root.quit()
            root.destroy()

        canvas.bind('<ButtonPress-1>', on_button_press)
        canvas.bind('<B1-Motion>', on_mouse_drag)
        canvas.bind('<ButtonRelease-1>', on_button_release)
        root.bind('<Escape>', on_escape)
        root.bind('<Return>', on_enter)

        root.mainloop()

        return selected_region

    def start_detection(self):
        """开始检测"""
        if self.running:
            print("检测已在运行")
            return

        if self.region is None:
            print("请先选择区域")
            return

        self.running = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 启动检测线程
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

        print(f"开始检测: {self.region['width']}x{self.region['height']}")

    def _detection_loop(self):
        """检测循环"""
        fps_update_time = time.time()

        try:
            while self.running:
                if not self.paused:
                    # 捕获屏幕
                    screenshot = self.sct.grab(self.region)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # 推理
                    results = self.model(frame, verbose=False)
                    person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

                    # 绘制
                    annotated_frame = results[0].plot()

                    # 计算 FPS
                    self.frame_count += 1
                    current_time = time.time()

                    if current_time - fps_update_time >= 1.0:
                        self.current_fps = self.frame_count / (current_time - self.start_time)
                        fps_update_time = current_time

                    # 显示信息
                    cv2.putText(annotated_frame, f"Region: {self.region['width']}x{self.region['height']}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"People: {person_count}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "右键托盘图标控制",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow(self.window_name, annotated_frame)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or q
                    self.stop_detection()
                    break
                elif key == ord(' '):
                    self.toggle_pause()

        except Exception as e:
            print(f"检测错误: {e}")
        finally:
            cv2.destroyAllWindows()
            for i in range(10):
                cv2.waitKey(1)

    def stop_detection(self):
        """停止检测"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print("检测已停止")

    def toggle_pause(self):
        """暂停/继续"""
        self.paused = not self.paused
        status = "已暂停" if self.paused else "继续检测"
        print(status)

class TrayApp:
    """系统托盘应用"""
    def __init__(self):
        self.detector = ScreenDetector()
        self.icon = None

    def create_icon_image(self):
        """创建托盘图标"""
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), color='black')
        dc = ImageDraw.Draw(image)

        # 画一个相机图标
        dc.rectangle([10, 20, 54, 50], outline='white', width=3)
        dc.ellipse([25, 30, 39, 44], fill='white')
        dc.rectangle([20, 15, 30, 20], fill='white')

        return image

    def on_select_region(self, icon, item):
        """选择区域"""
        print("\n请选择屏幕区域...")
        region = self.detector.select_region()
        if region:
            self.detector.region = region
            print(f"区域已选择: {region['width']}x{region['height']}")

    def on_start(self, icon, item):
        """开始检测"""
        if self.detector.model is None:
            print("正在加载模型...")
            self.detector.load_model()
        self.detector.start_detection()

    def on_stop(self, icon, item):
        """停止检测"""
        self.detector.stop_detection()

    def on_pause(self, icon, item):
        """暂停/继续"""
        self.detector.toggle_pause()

    def on_quit(self, icon, item):
        """退出"""
        self.detector.stop_detection()
        self.detector.sct.close()
        icon.stop()
        print("程序已退出")

    def run(self):
        """运行托盘应用"""
        # 预加载模型
        print("正在加载模型...")
        self.detector.load_model()

        # 创建托盘图标
        menu = pystray.Menu(
            item('选择区域', self.on_select_region),
            item('开始检测', self.on_start),
            item('停止检测', self.on_stop),
            item('暂停/继续', self.on_pause),
            pystray.Menu.SEPARATOR,
            item('退出', self.on_quit)
        )

        self.icon = pystray.Icon(
            'screen_detector',
            self.create_icon_image(),
            '屏幕人形检测',
            menu
        )

        print("\n" + "="*60)
        print("系统托盘屏幕检测已启动")
        print("="*60)
        print("\n使用说明:")
        print("1. 右键点击系统托盘图标")
        print("2. 选择 '选择区域' 框选检测区域")
        print("3. 选择 '开始检测' 开始实时检测")
        print("4. 选择 '暂停/继续' 暂停或继续检测")
        print("5. 选择 '停止检测' 停止检测")
        print("6. 选择 '退出' 关闭程序")
        print("\n程序已最小化到系统托盘...")
        print("="*60)
        print()

        # 运行托盘图标
        self.icon.run()

def main():
    try:
        app = TrayApp()
        app.run()
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

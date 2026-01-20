# -*- coding: utf-8 -*-
"""
系统托盘屏幕检测 - GPU 加速（简化版）
后台运行，通过系统托盘控制
支持多显示器选择
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

class ScreenDetector:
    """屏幕检测器类"""
    def __init__(self):
        self.model = None
        self.sct = mss.mss()
        self.monitor = None
        self.monitor_idx = 2  # 默认显示器2
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
        print("正在加载模型...")
        model_path = 'yolov8n_openvino_model'
        if not os.path.exists(model_path):
            print("首次运行，正在转换模型（约1-2分钟）...")
            base_model = YOLO('yolov8n.pt')
            base_model.export(format='openvino', half=False)

        self.model = YOLO(model_path, task='detect')
        print("模型加载完成！")

    def list_monitors(self):
        """列出可用显示器"""
        monitors = self.sct.monitors[1:]
        print("\n可用显示器:")
        for i, mon in enumerate(monitors, 1):
            print(f"  {i}. {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")
        return monitors

    def set_monitor(self, idx):
        """设置要捕获的显示器"""
        monitors = self.sct.monitors[1:]
        if 1 <= idx <= len(monitors):
            self.monitor = monitors[idx - 1]
            self.monitor_idx = idx
            print(f"\n设置为显示器 {idx}: {self.monitor['width']}x{self.monitor['height']}")
            return True
        return False

    def start_detection(self):
        """开始检测"""
        if self.running:
            print("检测已在运行")
            return

        if self.monitor is None:
            print("请先设置显示器")
            return

        if self.model is None:
            print("正在加载模型...")
            self.load_model()

        self.running = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()

        # 启动检测线程
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

        print(f"\n✓ 开始检测显示器 {self.monitor_idx}")

    def _detection_loop(self):
        """检测循环"""
        fps_update_time = time.time()

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        print("提示: 请将检测窗口移到其他显示器，避免递归显示")
        print("按 ESC 或 q 键可关闭检测窗口\n")

        try:
            while self.running:
                if not self.paused:
                    # 捕获屏幕
                    screenshot = self.sct.grab(self.monitor)
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
                    cv2.putText(annotated_frame, f"Monitor {self.monitor_idx}: {self.monitor['width']}x{self.monitor['height']}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"People: {person_count}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Right-click tray icon for controls",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow(self.window_name, annotated_frame)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or q
                    self.stop_detection()
                    break

        except Exception as e:
            print(f"检测错误: {e}")
        finally:
            cv2.destroyAllWindows()
            # Windows 清理
            for i in range(10):
                cv2.waitKey(1)

    def stop_detection(self):
        """停止检测"""
        if not self.running:
            return

        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)

        # 统计
        if self.start_time and self.frame_count > 0:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time
            print(f"\n✓ 检测已停止")
            print(f"  总帧数: {self.frame_count}")
            print(f"  运行时间: {total_time:.1f}s")
            print(f"  平均 FPS: {avg_fps:.1f}")

    def toggle_pause(self):
        """暂停/继续"""
        self.paused = not self.paused
        status = "已暂停" if self.paused else "继续检测"
        print(f"✓ {status}")

class TrayApp:
    """系统托盘应用"""
    def __init__(self):
        self.detector = ScreenDetector()
        self.icon = None

    def create_icon_image(self):
        """创建托盘图标"""
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), color='#1a1a1a')
        dc = ImageDraw.Draw(image)

        # 画一个简单的显示器图标
        dc.rectangle([10, 15, 54, 45], outline='#00ff00', width=3)
        dc.rectangle([28, 45, 36, 50], fill='#00ff00')
        dc.line([20, 50, 44, 50], fill='#00ff00', width=3)

        return image

    def on_monitor1(self, icon, item):
        """选择显示器1"""
        self.detector.stop_detection()
        if self.detector.set_monitor(1):
            self.detector.start_detection()

    def on_monitor2(self, icon, item):
        """选择显示器2"""
        self.detector.stop_detection()
        if self.detector.set_monitor(2):
            self.detector.start_detection()

    def on_start(self, icon, item):
        """开始检测"""
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
        print("\n程序已退出")

    def run(self):
        """运行托盘应用"""
        print("="*60)
        print("系统托盘屏幕检测 - GPU 加速")
        print("="*60)
        print()

        # 列出显示器
        monitors = self.detector.list_monitors()

        # 设置默认显示器
        default_idx = 2 if len(monitors) >= 2 else 1
        self.detector.set_monitor(default_idx)

        # 预加载模型
        self.detector.load_model()

        # 创建托盘菜单
        monitor_menu = []
        for i in range(1, len(monitors) + 1):
            mon = monitors[i-1]
            label = f"显示器 {i} ({mon['width']}x{mon['height']})"
            if i == 1:
                monitor_menu.append(item(label, self.on_monitor1))
            elif i == 2:
                monitor_menu.append(item(label, self.on_monitor2))

        menu = pystray.Menu(
            pystray.Menu.SEPARATOR if len(monitor_menu) > 0 else None,
            *monitor_menu,
            pystray.Menu.SEPARATOR,
            item('▶️ 开始检测', self.on_start),
            item('⏸️ 暂停/继续', self.on_pause),
            item('⏹️ 停止检测', self.on_stop),
            pystray.Menu.SEPARATOR,
            item('❌ 退出', self.on_quit)
        )

        self.icon = pystray.Icon(
            'screen_detector',
            self.create_icon_image(),
            '屏幕人形检测',
            menu
        )

        print()
        print("="*60)
        print("使用说明:")
        print("  1. 程序已最小化到系统托盘（任务栏右下角）")
        print(f"  2. 默认监控: 显示器 {default_idx}")
        print("  3. 右键点击托盘图标可以:")
        print("     - 切换显示器")
        print("     - 开始/暂停/停止检测")
        print("     - 退出程序")
        print("  4. 检测窗口出现后，请移到其他显示器")
        print("="*60)
        print()

        # 运行托盘图标
        self.icon.run()

def main():
    try:
        app = TrayApp()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

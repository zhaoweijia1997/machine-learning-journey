# -*- coding: utf-8 -*-
"""
屏幕区域选择检测 - GPU 加速
支持鼠标框选屏幕区域进行检测
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
    import mss.tools
except ImportError:
    print("错误: mss 库未安装")
    print("安装命令: pip install mss")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("错误: tkinter 未安装")
    sys.exit(1)

class RegionSelector:
    """区域选择器"""
    def __init__(self):
        self.region = None
        self.start_x = None
        self.start_y = None
        self.selecting = False

    def select_region(self):
        """使用 Tkinter 创建全屏透明窗口进行区域选择"""
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)  # 半透明
        root.attributes('-topmost', True)

        # 获取屏幕尺寸
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # 创建画布
        canvas = tk.Canvas(root, cursor="cross", bg='black')
        canvas.pack(fill=tk.BOTH, expand=True)

        # 提示文字
        canvas.create_text(
            screen_width // 2,
            50,
            text="按住鼠标左键拖动选择区域 | ESC 取消 | Enter 全屏",
            fill="white",
            font=("Arial", 16, "bold")
        )

        rect_id = None

        def on_button_press(event):
            nonlocal rect_id
            self.start_x = event.x
            self.start_y = event.y
            self.selecting = True

            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(
                self.start_x, self.start_y,
                self.start_x, self.start_y,
                outline='red', width=3
            )

        def on_mouse_drag(event):
            if self.selecting and rect_id:
                canvas.coords(
                    rect_id,
                    self.start_x, self.start_y,
                    event.x, event.y
                )

        def on_button_release(event):
            if self.selecting:
                self.selecting = False
                end_x = event.x
                end_y = event.y

                # 确保坐标正确（左上到右下）
                x1 = min(self.start_x, end_x)
                y1 = min(self.start_y, end_y)
                x2 = max(self.start_x, end_x)
                y2 = max(self.start_y, end_y)

                width = x2 - x1
                height = y2 - y1

                # 最小区域 50x50
                if width > 50 and height > 50:
                    self.region = {
                        'left': x1,
                        'top': y1,
                        'width': width,
                        'height': height
                    }
                    root.quit()
                    root.destroy()
                else:
                    messagebox.showwarning("区域太小", "请选择至少 50x50 的区域")

        def on_escape(event):
            self.region = None
            root.quit()
            root.destroy()

        def on_enter(event):
            # 全屏
            self.region = {
                'left': 0,
                'top': 0,
                'width': screen_width,
                'height': screen_height
            }
            root.quit()
            root.destroy()

        canvas.bind('<ButtonPress-1>', on_button_press)
        canvas.bind('<B1-Motion>', on_mouse_drag)
        canvas.bind('<ButtonRelease-1>', on_button_release)
        root.bind('<Escape>', on_escape)
        root.bind('<Return>', on_enter)

        root.mainloop()

        return self.region

def main():
    print("="*60)
    print("屏幕区域检测 (GPU 加速)")
    print("="*60)
    print()

    # 选择区域
    print("请在屏幕上框选检测区域...")
    print("提示:")
    print("  - 按住鼠标左键拖动选择区域")
    print("  - 按 Enter 键使用全屏")
    print("  - 按 ESC 键取消")
    print()

    selector = RegionSelector()
    region = selector.select_region()

    if region is None:
        print("已取消区域选择")
        return

    print(f"\n选择的区域: {region['width']}x{region['height']}")
    print(f"位置: ({region['left']}, {region['top']})")
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

    print()
    print("="*60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("  r         - 重新选择区域")
    print("="*60)
    print()
    print("检测开始！")
    print()

    # 窗口名称
    window_name = 'Region Detection (GPU) [Press ESC or Q to quit]'
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
                # 捕获屏幕区域
                screenshot = sct.grab(region)

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
                cv2.putText(annotated_frame, f"Region: {region['width']}x{region['height']}",
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
                filename = f"region_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)
            elif key == ord('r'):
                # 重新选择区域
                print("\n请重新选择区域...")
                cv2.destroyAllWindows()

                new_region = selector.select_region()
                if new_region:
                    region = new_region
                    print(f"\n新区域: {region['width']}x{region['height']}")
                    print(f"位置: ({region['left']}, {region['top']})")

                    # 重置统计
                    frame_count = 0
                    start_time = time.time()
                    fps_update_time = start_time

                    # 重新创建窗口
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                else:
                    print("取消重新选择，继续使用当前区域")
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
        print(f"  区域大小: {region['width']}x{region['height']}")
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

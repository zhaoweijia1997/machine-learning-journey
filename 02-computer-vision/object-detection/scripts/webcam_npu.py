# -*- coding: utf-8 -*-
"""
NPU 加速实时摄像头检测
使用 Intel AI Boost (NPU) 进行推理
"""

from ultralytics import YOLO
import cv2
import time
import os
import sys

def main():
    print("="*60)
    print("NPU 加速实时人形检测")
    print("="*60)
    print()

    # 检查 OpenVINO 模型
    model_path = 'yolov8n_openvino_model'

    if not os.path.exists(model_path):
        print("首次运行，正在转换模型...")
        print("这需要 1-2 分钟，只需要做一次")
        print()

        base_model = YOLO('yolov8n.pt')
        base_model.export(format='openvino', half=False)
        print()

    # 加载模型
    print("加载 OpenVINO 模型...")
    model = YOLO(model_path, task='detect')

    # 显示设备信息
    try:
        from openvino import Core
        ie = Core()
        if 'NPU' in ie.available_devices:
            npu_name = ie.get_property('NPU', "FULL_DEVICE_NAME")
            print(f"使用设备: NPU ({npu_name})")
        else:
            print("警告: NPU 不可用，将使用其他设备")
    except:
        pass

    print("模型加载完成！")
    print()

    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头！")
        print("请检查:")
        print("  1. 摄像头是否连接")
        print("  2. 摄像头权限是否开启")
        print("  3. 是否被其他程序占用")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开: {width}x{height}")
    print()
    print("="*60)
    print("控制说明:")
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("="*60)
    print()
    print("检测开始！使用 Intel AI Boost (NPU)")
    print()

    # 设置窗口
    window_name = 'NPU Detection [Press ESC or Q to quit]'
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
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧")
                    break

                # NPU 推理
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
                cv2.putText(annotated_frame, f"Device: NPU (Intel AI Boost)",
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
                filename = f"npu_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)

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
        print(f"  设备: NPU (Intel AI Boost)")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {total_time:.2f} 秒")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print("="*60)
        print("程序已退出")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

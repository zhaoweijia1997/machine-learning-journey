# -*- coding: utf-8 -*-
"""
可选择设备的实时摄像头检测
支持 CPU、GPU、NPU 切换
"""

from ultralytics import YOLO
import cv2
import time
import os
import sys

def select_device():
    """让用户选择推理设备"""
    try:
        from openvino import Core
        ie = Core()
        devices = ie.available_devices

        print("可用的设备:")
        for i, device in enumerate(devices, 1):
            device_name = ie.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {i}. {device}: {device_name}")

        print(f"  {len(devices)+1}. AUTO (自动选择)")
        print()

        while True:
            try:
                choice = input(f"请选择设备 (1-{len(devices)+1}, 默认=NPU): ").strip()

                if not choice:
                    # 默认使用 NPU
                    if 'NPU' in devices:
                        return 'NPU', 'Intel AI Boost'
                    elif 'GPU' in devices:
                        return 'GPU', 'Intel Arc GPU'
                    else:
                        return 'AUTO', 'AUTO'

                choice_num = int(choice)

                if 1 <= choice_num <= len(devices):
                    selected = devices[choice_num - 1]
                    name = ie.get_property(selected, "FULL_DEVICE_NAME")
                    return selected, name
                elif choice_num == len(devices) + 1:
                    return 'AUTO', 'AUTO'
                else:
                    print("无效选择，请重新输入")
            except ValueError:
                print("请输入数字")
            except KeyboardInterrupt:
                print("\n已取消")
                sys.exit(0)

    except Exception as e:
        print(f"检测设备失败: {e}")
        return 'AUTO', 'AUTO'

def export_model_for_device(device):
    """为特定设备导出优化模型"""
    model_dir = f'yolov8n_{device.lower()}_model'

    if os.path.exists(model_dir):
        print(f"使用已有的 {device} 优化模型")
        return model_dir

    print(f"\n首次使用 {device}，正在导出优化模型...")
    print("这需要 1-2 分钟，只需要做一次")

    model = YOLO('yolov8n.pt')

    # 为不同设备设置导出参数
    if device == 'NPU':
        # NPU 需要 INT8 量化以获得最佳性能
        print("正在为 NPU 导出 INT8 量化模型...")
        model.export(format='openvino', half=False, int8=True)
        # 重命名输出目录
        if os.path.exists('yolov8n_openvino_model'):
            import shutil
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            shutil.move('yolov8n_openvino_model', model_dir)
    else:
        model.export(format='openvino', half=False)
        if device != 'CPU' and device != 'GPU':
            if os.path.exists('yolov8n_openvino_model'):
                import shutil
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                shutil.move('yolov8n_openvino_model', model_dir)

    print(f"{device} 模型准备完成！")
    return model_dir if os.path.exists(model_dir) else 'yolov8n_openvino_model'

def main():
    print("="*60)
    print("实时人形检测 (多设备支持)")
    print("="*60)
    print()

    # 选择设备
    device, device_name = select_device()
    print(f"\n选择的设备: {device} ({device_name})")
    print()

    # 准备模型
    model_path = export_model_for_device(device)

    # 加载模型
    print("加载模型...")
    model = YOLO(model_path, task='detect')
    print("模型加载完成！")
    print()

    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头！")
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
    print("  ESC 或 q  - 退出")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("="*60)
    print()
    print(f"检测开始！使用 {device} 推理")
    print()

    # 窗口名称
    window_name = f'{device} Detection [Press ESC or Q to quit]'
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

                # 推理
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
                cv2.putText(annotated_frame, f"Device: {device}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(annotated_frame, f"People: {person_count}",
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"{device.lower()}_snapshot_{int(time.time())}.jpg"
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
        print(f"  设备: {device}")
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
        sys.exit(1)

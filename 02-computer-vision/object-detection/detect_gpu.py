# -*- coding: utf-8 -*-
"""
GPU 加速人形检测（OpenVINO）
使用 Intel Arc GPU 进行加速推理
"""

from ultralytics import YOLO
import cv2
import os
import time

def main():
    print("=" * 60)
    print("GPU 加速人形检测")
    print("=" * 60)
    print()

    # 查找图片
    image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("未找到图片文件！")
        print("请将图片放到当前目录")
        return

    image_path = image_files[0]
    print(f"图片: {image_path}")
    print()

    # 检查 OpenVINO 模型是否存在
    openvino_model_path = 'yolov8n_openvino_model'

    if not os.path.exists(openvino_model_path):
        print("首次运行，正在转换模型为 OpenVINO 格式...")
        print("这需要 1-2 分钟，只需要做一次")
        print()

        # 加载原始模型
        model = YOLO('yolov8n.pt')

        # 导出为 OpenVINO 格式
        model.export(format='openvino', half=False)
        print()
        print("模型转换完成！")
        print()

    # 加载 OpenVINO 模型
    print("加载 OpenVINO 优化模型...")
    model = YOLO(openvino_model_path, task='detect')

    # 检测可用设备
    try:
        from openvino.runtime import Core
        ie = Core()
        devices = ie.available_devices
        print(f"可用设备: {', '.join(devices)}")

        # 优先使用 GPU，否则使用 AUTO
        if 'GPU' in devices:
            device = 'GPU'
            print("使用: Intel Arc GPU")
        else:
            device = 'AUTO'
            print("使用: AUTO (自动选择最优设备)")
    except:
        device = 'AUTO'
        print("使用: AUTO")

    print()
    print("开始检测...")

    # OpenVINO 模型不需要指定 device 参数
    # 它会自动使用转换时指定的设备

    # 预热
    _ = model(image_path, verbose=False)

    # 正式检测（计时）
    start_time = time.time()
    results = model(image_path, verbose=False)
    inference_time = (time.time() - start_time) * 1000

    # 统计人数
    person_count = 0
    all_detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = results[0].names[cls_id]

        all_detections.append((cls_name, conf))

        if cls_id == 0:  # person
            person_count += 1

    print()
    print("=" * 60)
    print("检测结果:")
    print("=" * 60)
    print(f"检测到: {person_count} 个人")
    print(f"推理时间: {inference_time:.1f} ms")
    print(f"FPS: {1000/inference_time:.1f}")
    print()

    if all_detections:
        print("所有检测对象:")
        for obj_name, conf in all_detections:
            print(f"  - {obj_name}: {conf:.2%}")

    # 保存结果
    annotated = results[0].plot()
    output_path = 'result_gpu.jpg'
    cv2.imwrite(output_path, annotated)

    print()
    print(f"结果已保存: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

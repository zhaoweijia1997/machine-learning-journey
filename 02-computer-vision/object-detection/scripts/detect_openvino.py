# -*- coding: utf-8 -*-
"""
OpenVINO 直接推理（完全 GPU 加速）
"""

from openvino import Core
import cv2
import numpy as np
import time
import os

def main():
    print("=" * 60)
    print("OpenVINO GPU 加速推理")
    print("=" * 60)
    print()

    # 查找图片
    image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("未找到图片！")
        return

    image_path = image_files[0]
    print(f"图片: {image_path}")

    # 初始化 OpenVINO
    print("\n初始化 OpenVINO...")
    ie = Core()

    # 显示可用设备
    devices = ie.available_devices
    print(f"可用设备: {', '.join(devices)}")

    # 选择设备
    if 'GPU' in devices:
        device = 'GPU'
        print(f"使用设备: GPU (Intel Arc)")
    else:
        device = 'CPU'
        print(f"使用设备: CPU")

    # 加载模型
    print("\n加载模型...")
    model_path = 'yolov8n_openvino_model/yolov8n.xml'

    if not os.path.exists(model_path):
        print("模型未找到！请先运行 detect_gpu.py 转换模型")
        return

    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)

    # 获取输入输出信息
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(f"输入形状: {input_layer.shape}")
    print(f"输出形状: {output_layer.shape}")

    # 读取图片
    print(f"\n读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片！")
        return

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 预处理
    print("\n预处理...")
    input_h, input_w = 640, 640
    resized = cv2.resize(image, (input_w, input_h))
    input_image = resized.transpose(2, 0, 1)  # HWC -> CHW
    input_image = input_image.reshape(1, 3, input_h, input_w).astype(np.float32) / 255.0

    # 预热
    print("预热...")
    for _ in range(3):
        _ = compiled_model([input_image])

    # 推理（计时）
    print(f"\n使用 {device} 推理...")
    times = []

    for i in range(10):
        start = time.time()
        result = compiled_model([input_image])
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    fps = 1000 / avg_time

    # 输出结果
    output = result[output_layer]
    print(f"\n输出形状: {output.shape}")

    print("\n" + "=" * 60)
    print("性能结果:")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"平均推理时间: {avg_time:.1f} ms")
    print(f"FPS: {fps:.1f}")
    print(f"最快: {min(times):.1f} ms")
    print(f"最慢: {max(times):.1f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()

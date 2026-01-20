# -*- coding: utf-8 -*-
"""
NPU 加速测试
对比 CPU、GPU、NPU 三种设备的性能
"""

from openvino import Core
import cv2
import numpy as np
import time
import os

def preprocess_image(image, input_h=640, input_w=640):
    """预处理图片"""
    resized = cv2.resize(image, (input_w, input_h))
    input_image = resized.transpose(2, 0, 1)  # HWC -> CHW
    input_image = input_image.reshape(1, 3, input_h, input_w).astype(np.float32) / 255.0
    return input_image

def benchmark_device(compiled_model, input_image, device_name, warmup=3, runs=20):
    """性能测试"""
    print(f"\n{'='*60}")
    print(f"测试设备: {device_name}")
    print(f"{'='*60}")

    # 预热
    print("预热中...")
    for _ in range(warmup):
        _ = compiled_model([input_image])

    # 正式测试
    print(f"运行 {runs} 次推理...")
    times = []

    for i in range(runs):
        start = time.time()
        result = compiled_model([input_image])
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        if (i + 1) % 5 == 0:
            print(f"  进度: {i+1}/{runs}")

    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000 / avg_time

    print(f"\n结果:")
    print(f"  平均: {avg_time:.1f} ms ({fps:.1f} FPS)")
    print(f"  最快: {min_time:.1f} ms")
    print(f"  最慢: {max_time:.1f} ms")

    return {
        'device': device_name,
        'avg_ms': avg_time,
        'fps': fps,
        'min_ms': min_time,
        'max_ms': max_time
    }

def main():
    print("="*60)
    print("NPU vs GPU vs CPU 性能对比")
    print("="*60)

    # 查找图片
    image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("\n未找到测试图片！")
        return

    image_path = image_files[0]
    print(f"\n测试图片: {image_path}")

    # 检查模型
    model_path = 'yolov8n_openvino_model/yolov8n.xml'
    if not os.path.exists(model_path):
        print("\n模型未找到！请先运行: python detect_gpu.py")
        return

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("\n无法读取图片！")
        return

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 预处理
    input_image = preprocess_image(image)

    # 初始化 OpenVINO
    print("\n初始化 OpenVINO...")
    ie = Core()

    devices = ie.available_devices
    print(f"可用设备: {', '.join(devices)}")

    # 加载模型
    print("\n加载模型...")
    model = ie.read_model(model=model_path)

    results = []

    # 测试所有设备
    for device in ['CPU', 'GPU', 'NPU']:
        if device in devices:
            print(f"\n编译模型 ({device})...")
            compiled_model = ie.compile_model(model=model, device_name=device)

            result = benchmark_device(compiled_model, input_image, device)
            results.append(result)
        else:
            print(f"\n⚠️ {device} 不可用，跳过")

    # 对比结果
    print("\n")
    print("="*60)
    print("性能对比汇总")
    print("="*60)
    print(f"{'设备':<10} {'平均时间':<15} {'FPS':<10} {'相对速度'}")
    print("-"*60)

    if results:
        baseline_fps = results[0]['fps']

        for r in results:
            speedup = r['fps'] / baseline_fps
            print(f"{r['device']:<10} {r['avg_ms']:>6.1f} ms      {r['fps']:>6.1f}    {speedup:.2f}x")

    print("="*60)

    # 推荐
    if results:
        best = max(results, key=lambda x: x['fps'])
        print(f"\n推荐设备: {best['device']} ({best['fps']:.1f} FPS)")

if __name__ == "__main__":
    main()

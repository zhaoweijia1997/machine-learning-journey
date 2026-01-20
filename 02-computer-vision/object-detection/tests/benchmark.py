# -*- coding: utf-8 -*-
"""
性能对比测试
对比 CPU vs GPU 性能差异
"""

import time
from ultralytics import YOLO
import os

def benchmark_model(model_path, device, runs=20, name="Model"):
    """
    性能测试

    参数:
        model_path: 模型路径
        device: 设备 ('cpu', 'gpu', 'AUTO')
        runs: 测试次数
        name: 显示名称
    """
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"设备: {device}")
    print(f"{'='*60}")

    # 查找测试图片
    image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("未找到测试图片！")
        return None

    test_image = image_files[0]
    print(f"测试图片: {test_image}")

    try:
        # 加载模型
        if 'openvino' in model_path.lower():
            model = YOLO(model_path, task='detect')
        else:
            model = YOLO(model_path)

        print("预热中...")
        # 预热 (5次)
        for _ in range(5):
            _ = model(test_image, device=device, verbose=False)

        print(f"正式测试 ({runs} 次)...")
        times = []

        for i in range(runs):
            start = time.time()
            results = model(test_image, device=device, verbose=False)
            elapsed = (time.time() - start) * 1000  # 转换为毫秒
            times.append(elapsed)

            if (i + 1) % 5 == 0:
                print(f"  完成 {i+1}/{runs} 次")

        # 计算统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1000 / avg_time

        print()
        print(f"结果:")
        print(f"  平均推理时间: {avg_time:.1f} ms")
        print(f"  最快: {min_time:.1f} ms")
        print(f"  最慢: {max_time:.1f} ms")
        print(f"  平均 FPS: {fps:.1f}")

        return {
            'name': name,
            'device': device,
            'avg_time': avg_time,
            'fps': fps,
            'min_time': min_time,
            'max_time': max_time
        }

    except Exception as e:
        print(f"错误: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("YOLOv8 性能对比测试")
    print("Intel Ultra 9 185H")
    print("="*60)

    results = []

    # 测试 1: CPU (原始 PyTorch)
    print("\n【测试 1/3】CPU 版本 (PyTorch)")
    result = benchmark_model('yolov8n.pt', 'cpu', runs=20, name='CPU (PyTorch)')
    if result:
        results.append(result)

    # 测试 2: OpenVINO AUTO
    openvino_path = 'yolov8n_openvino_model'
    if os.path.exists(openvino_path):
        print("\n【测试 2/3】OpenVINO AUTO")
        result = benchmark_model(openvino_path, 'AUTO', runs=20, name='OpenVINO AUTO')
        if result:
            results.append(result)

        # 测试 3: OpenVINO GPU
        try:
            from openvino.runtime import Core
            ie = Core()
            if 'GPU' in ie.available_devices:
                print("\n【测试 3/3】OpenVINO GPU")
                result = benchmark_model(openvino_path, 'GPU', runs=20, name='OpenVINO GPU')
                if result:
                    results.append(result)
            else:
                print("\n【测试 3/3】跳过 - 未检测到 GPU")
        except:
            print("\n【测试 3/3】跳过 - OpenVINO 未正确配置")
    else:
        print("\n跳过 OpenVINO 测试 - 模型未转换")
        print("请先运行: python detect_gpu.py")

    # 显示对比结果
    if len(results) > 1:
        print("\n" + "="*60)
        print("性能对比总结")
        print("="*60)
        print()
        print(f"{'配置':<20} {'推理时间':<15} {'FPS':<10} {'提升':<10}")
        print("-" * 60)

        baseline = results[0]['avg_time']

        for r in results:
            speedup = baseline / r['avg_time']
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else "-"

            print(f"{r['name']:<20} {r['avg_time']:>6.1f} ms      {r['fps']:>6.1f}    {speedup_str}")

        print("-" * 60)

        # 找出最快的
        fastest = min(results, key=lambda x: x['avg_time'])
        print()
        print(f"最快配置: {fastest['name']}")
        print(f"性能提升: {baseline/fastest['avg_time']:.2f}x")
        print()

    print("="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()

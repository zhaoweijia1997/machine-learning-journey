# -*- coding: utf-8 -*-
"""
检查 GPU 是否被正确使用
"""

from openvino import Core
import numpy as np
import time

def main():
    print("="*60)
    print("OpenVINO GPU 诊断工具")
    print("="*60)
    print()

    ie = Core()

    print("可用设备:")
    for device in ie.available_devices:
        print(f"  - {device}")
        if device == 'GPU':
            try:
                full_name = ie.get_property(device, "FULL_DEVICE_NAME")
                print(f"    名称: {full_name}")
            except:
                pass
    print()

    # 加载模型
    model_path = "yolov8m_openvino_model/yolov8m.xml"
    print(f"加载模型: {model_path}")

    model = ie.read_model(model=model_path)

    # 测试不同设备的性能
    devices = ['CPU', 'GPU']

    for device_name in devices:
        if device_name not in ie.available_devices:
            continue

        print()
        print(f"{'='*60}")
        print(f"测试设备: {device_name}")
        print(f"{'='*60}")

        # 编译模型
        compiled = ie.compile_model(model=model, device_name=device_name)

        # 创建测试输入
        input_layer = compiled.input(0)
        test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        # 预热
        for _ in range(5):
            _ = compiled([test_input])

        # 性能测试
        times = []
        test_count = 20

        print(f"运行 {test_count} 次推理测试...")

        for i in range(test_count):
            start = time.time()
            _ = compiled([test_input])
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  第 {i+1}/{test_count} 次: {elapsed:.1f}ms")

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1000 / avg_time

        print()
        print(f"结果统计:")
        print(f"  平均推理时间: {avg_time:.1f}ms")
        print(f"  最小推理时间: {min_time:.1f}ms")
        print(f"  最大推理时间: {max_time:.1f}ms")
        print(f"  平均 FPS: {fps:.1f}")
        print()

    print("="*60)
    print("诊断完成！")
    print("如果 GPU 比 CPU 慢，说明可能存在驱动或配置问题")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

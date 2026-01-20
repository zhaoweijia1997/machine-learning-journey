# -*- coding: utf-8 -*-
"""
GPU 压力测试 - 最大化 GPU 利用率
使用 yolov8l (large) 模型 + 高分辨率 + 并行推理
"""

from openvino import Core
import numpy as np
import time
import sys

def main():
    import argparse

    parser = argparse.ArgumentParser(description='GPU 压力测试')
    parser.add_argument('--model', type=int, choices=[1,2,3,4,5], default=3,
                       help='模型: 1=nano, 2=small, 3=medium, 4=large, 5=xlarge')
    parser.add_argument('--resolution', type=int, choices=[1,2,3], default=2,
                       help='分辨率: 1=640, 2=1280, 3=1920')
    parser.add_argument('--batch', type=int, choices=[1,2,4], default=4,
                       help='批量大小: 1/2/4')
    parser.add_argument('--duration', type=int, default=60,
                       help='测试时长（秒）')
    args = parser.parse_args()

    print("="*60)
    print("GPU 压力测试 - 极限性能模式")
    print("="*60)
    print()

    # 初始化 OpenVINO
    ie = Core()

    if 'GPU' not in ie.available_devices:
        print("错误: GPU 不可用")
        return

    gpu_name = ie.get_property('GPU', "FULL_DEVICE_NAME")
    print(f"GPU: {gpu_name}")
    print()

    # 模型映射
    models = {
        1: ('yolov8n', 'nano'),
        2: ('yolov8s', 'small'),
        3: ('yolov8m', 'medium'),
        4: ('yolov8l', 'large'),
        5: ('yolov8x', 'xlarge'),
    }

    model_name, model_desc = models[args.model]
    model_path = f"{model_name}_openvino_model/{model_name}.xml"

    print(f"模型: {model_name} ({model_desc})")
    print(f"路径: {model_path}")
    print()

    # 加载并编译到 GPU
    print("加载模型到 GPU...")
    model = ie.read_model(model=model_path)

    # 配置 GPU 性能优化
    config = {
        "PERFORMANCE_HINT": "THROUGHPUT",  # 吞吐量优先
        "NUM_STREAMS": "4",  # 4 个推理流并行
    }

    compiled_model = ie.compile_model(
        model=model,
        device_name="GPU",
        config=config
    )

    print("模型已编译到 GPU（吞吐量优化模式）")
    print()

    # 模型输入形状是固定的，从模型获取
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape

    print(f"模型输入形状: {input_shape}")

    # 提取实际的分辨率
    model_resolution = int(input_shape[2])  # height

    print(f"推理分辨率: {model_resolution}x{model_resolution} (固定)")
    print()

    # 并行推理：多次调用
    num_parallel = args.batch

    print(f"并行推理策略: {num_parallel} 个请求")
    print()

    # 创建测试输入
    print("="*60)
    print("开始 GPU 压力测试...")
    print("="*60)
    print()

    test_input = np.random.rand(1, 3, model_resolution, model_resolution).astype(np.float32)

    # 预热
    print("预热 GPU (10次)...")
    for _ in range(10):
        _ = compiled_model([test_input])
    print("预热完成")
    print()

    # 持续压力测试
    print(f"开始持续推理（{args.duration}秒）...")
    print("请打开任务管理器观察 GPU 占用率")
    print()

    inference_count = 0
    start_time = time.time()
    last_report = start_time

    try:
        while time.time() - start_time < args.duration:
            # 并行推理：快速连续多次调用以充分利用 GPU
            for _ in range(num_parallel):
                _ = compiled_model([test_input])
                inference_count += 1

            # 每秒报告一次
            current_time = time.time()
            if current_time - last_report >= 1.0:
                elapsed = current_time - start_time
                fps = inference_count / elapsed

                print(f"时间: {elapsed:.1f}s | 推理次数: {inference_count} | FPS: {fps:.1f}")

                last_report = current_time

    except KeyboardInterrupt:
        print("\n\n用户中断")

    # 最终统计
    total_time = time.time() - start_time
    avg_fps = inference_count / total_time
    avg_latency = (total_time / inference_count) * 1000  # ms

    print()
    print("="*60)
    print("GPU 压力测试结果:")
    print("="*60)
    print(f"模型: {model_name} ({model_desc})")
    print(f"分辨率: {model_resolution}x{model_resolution}")
    print(f"并行请求: {num_parallel}")
    print(f"总推理次数: {inference_count}")
    print(f"测试时长: {total_time:.1f} 秒")
    print(f"平均 FPS: {avg_fps:.1f}")
    print(f"平均延迟: {avg_latency:.1f} ms/帧")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

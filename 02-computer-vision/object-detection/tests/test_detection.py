# -*- coding: utf-8 -*-
"""
测试检测功能 - 验证 YOLO 输出
"""

import cv2
import numpy as np
from openvino import Core

# 初始化 OpenVINO
ie = Core()
print(f"可用设备: {ie.available_devices}")

# 加载模型
model_path = "yolov8x_openvino_model/yolov8x.xml"
print(f"加载模型: {model_path}")

model = ie.read_model(model=model_path)
compiled_model = ie.compile_model(model=model, device_name="GPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print(f"输入 shape: {input_layer.shape}")
print(f"输出 shape: {output_layer.shape}")

# 创建测试图像 (640x640, 随机噪声)
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 预处理
input_frame = test_image.transpose(2, 0, 1)
input_frame = np.expand_dims(input_frame, axis=0)
input_frame = input_frame.astype(np.float32) / 255.0

print(f"输入数据 shape: {input_frame.shape}")

# 推理
results = compiled_model([input_frame])[output_layer]

print(f"输出结果 shape: {results.shape}")
print(f"输出结果类型: {results.dtype}")

# 解析输出
detections = results[0].T
print(f"Detections shape (转置后): {detections.shape}")
print(f"前5个检测的前10个值:")
print(detections[:5, :10])

# 获取类别置信度
class_scores = detections[:, 4:]
class_ids = np.argmax(class_scores, axis=1)
confidences = np.max(class_scores, axis=1)

print(f"\nClass scores shape: {class_scores.shape}")
print(f"最大置信度: {np.max(confidences):.4f}")
print(f"置信度 > 0.5 的数量: {np.sum(confidences > 0.5)}")
print(f"置信度 > 0.3 的数量: {np.sum(confidences > 0.3)}")
print(f"置信度 > 0.1 的数量: {np.sum(confidences > 0.1)}")

# 过滤
mask = confidences > 0.3
boxes = detections[mask, :4]
confidences_filtered = confidences[mask]
class_ids_filtered = class_ids[mask]

print(f"\n过滤后检测数量: {len(boxes)}")
if len(boxes) > 0:
    print(f"检测到的类别: {np.unique(class_ids_filtered)}")
    for i, (box, conf, cls) in enumerate(zip(boxes[:5], confidences_filtered[:5], class_ids_filtered[:5])):
        print(f"  {i+1}. Class: {cls}, Conf: {conf:.4f}, Box: {box}")

print("\n测试完成!")

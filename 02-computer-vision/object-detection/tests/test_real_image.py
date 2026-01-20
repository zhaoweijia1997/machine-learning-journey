# -*- coding: utf-8 -*-
"""
使用真实图片测试检测
"""

import cv2
import numpy as np
from openvino import Core
import urllib.request

# 下载一张包含人的测试图片
print("下载测试图片...")
url = "https://ultralytics.com/images/bus.jpg"
urllib.request.urlretrieve(url, "test_image.jpg")
print("下载完成!")

# 读取图片
image = cv2.imread("test_image.jpg")
print(f"原始图片大小: {image.shape}")

# 初始化 OpenVINO
ie = Core()
model_path = "yolov8x_openvino_model/yolov8x.xml"
model = ie.read_model(model=model_path)
compiled_model = ie.compile_model(model=model, device_name="GPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 预处理
input_frame = cv2.resize(image, (640, 640))
input_frame = input_frame.transpose(2, 0, 1)
input_frame = np.expand_dims(input_frame, axis=0)
input_frame = input_frame.astype(np.float32) / 255.0

# 推理
print("开始推理...")
results = compiled_model([input_frame])[output_layer]

# 解析输出
detections = results[0].T
class_scores = detections[:, 4:]
class_ids = np.argmax(class_scores, axis=1)
confidences = np.max(class_scores, axis=1)

print(f"最大置信度: {np.max(confidences):.4f}")
print(f"置信度 > 0.5 的数量: {np.sum(confidences > 0.5)}")

# 过滤并绘制
mask = confidences > 0.5
boxes = detections[mask, :4]
confidences_filtered = confidences[mask]
class_ids_filtered = class_ids[mask]

print(f"\n检测到 {len(boxes)} 个物体")

# 绘制检测框
output_image = image.copy()
h, w = image.shape[:2]

for box, conf, class_id in zip(boxes, confidences_filtered, class_ids_filtered):
    x_center, y_center, width, height = box

    # 转换为像素坐标
    x_center_px = x_center * w / 640  # 因为推理输入是640x640
    y_center_px = y_center * h / 640
    width_px = width * w / 640
    height_px = height * h / 640

    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)

    print(f"  Class: {class_id}, Conf: {conf:.2f}, Box: ({x1},{y1}) -> ({x2},{y2})")

    # 绘制框
    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
    label = f"Class{class_id} {conf:.2f}"
    cv2.putText(output_image, label, (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存结果
cv2.imwrite("test_result.jpg", output_image)
print("\n结果已保存到 test_result.jpg")
print("测试完成!")

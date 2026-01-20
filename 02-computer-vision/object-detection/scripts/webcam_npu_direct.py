# -*- coding: utf-8 -*-
"""
NPU 直接推理 - 实时摄像头检测
使用 OpenVINO API 直接控制 NPU
"""

from openvino import Core
import cv2
import numpy as np
import time
import os

# COCO 类别名称（YOLOv8）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess(frame, input_h=640, input_w=640):
    """预处理图像"""
    # 保存原始尺寸用于后处理
    orig_h, orig_w = frame.shape[:2]

    # Resize
    resized = cv2.resize(frame, (input_w, input_h))

    # HWC -> CHW
    input_image = resized.transpose(2, 0, 1)

    # 归一化并添加 batch 维度
    input_image = input_image.reshape(1, 3, input_h, input_w).astype(np.float32) / 255.0

    return input_image, orig_w, orig_h

def postprocess(output, orig_w, orig_h, conf_threshold=0.5):
    """后处理 YOLOv8 输出"""
    # YOLOv8 输出格式: (1, 84, 8400)
    # 84 = 4 (box) + 80 (classes)

    predictions = output[0]  # (84, 8400)
    predictions = predictions.T  # (8400, 84)

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        # 前4个是box坐标
        x, y, w, h = pred[:4]

        # 后80个是类别分数
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > conf_threshold:
            # 转换坐标 (center -> corner)
            x1 = int((x - w/2) * orig_w / 640)
            y1 = int((y - h/2) * orig_h / 640)
            x2 = int((x + w/2) * orig_w / 640)
            y2 = int((y + h/2) * orig_h / 640)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(int(class_id))

    # NMS
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.4)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            scores = [scores[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]

    return boxes, scores, class_ids

def draw_boxes(frame, boxes, scores, class_ids):
    """绘制检测框"""
    person_count = 0

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box

        # 统计人数
        if class_id == 0:  # person
            person_count += 1
            color = (0, 255, 0)  # 绿色
        else:
            color = (255, 0, 0)  # 蓝色

        # 绘制框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f'{COCO_CLASSES[class_id]}: {score:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return person_count

def main():
    print("="*60)
    print("NPU 直接推理 - 实时人形检测")
    print("="*60)
    print()

    # 检查模型
    model_path = 'yolov8n_openvino_model/yolov8n.xml'
    if not os.path.exists(model_path):
        print("模型未找到！请先运行: python detect_gpu.py")
        return

    # 初始化 OpenVINO
    print("初始化 OpenVINO...")
    ie = Core()

    devices = ie.available_devices
    print(f"可用设备: {', '.join(devices)}")

    if 'NPU' not in devices:
        print("\n警告: NPU 不可用！")
        print("将使用 GPU 代替")
        device = 'GPU' if 'GPU' in devices else 'CPU'
    else:
        device = 'NPU'
        device_name = ie.get_property('NPU', "FULL_DEVICE_NAME")
        print(f"使用设备: NPU ({device_name})")

    # 编译模型
    print(f"\n编译模型到 {device}...")
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(f"模型输入: {input_layer.shape}")
    print(f"模型输出: {output_layer.shape}")

    # 打开摄像头
    print("\n正在打开摄像头...")
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
    print("  ESC 或 q  - 退出程序")
    print("  s         - 保存截图")
    print("  空格      - 暂停/继续")
    print("="*60)
    print()
    print(f"检测开始！使用 {device}")
    print()

    # 窗口
    window_name = f'{device} Direct Inference [Press ESC or Q to quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 预热
    print("预热 NPU...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_input, _, _ = preprocess(dummy_frame)
    for _ in range(3):
        _ = compiled_model([dummy_input])
    print("预热完成！")
    print()

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

                # 预处理
                input_image, orig_w, orig_h = preprocess(frame)

                # NPU 推理
                result = compiled_model([input_image])
                output = result[output_layer]

                # 后处理
                boxes, scores, class_ids = postprocess(output, orig_w, orig_h, conf_threshold=0.5)

                # 绘制
                person_count = draw_boxes(frame, boxes, scores, class_ids)

                # 计算 FPS
                frame_count += 1
                current_time = time.time()

                if current_time - fps_update_time >= 1.0:
                    current_fps = frame_count / (current_time - start_time)
                    fps_update_time = current_time

                # 显示信息
                info_y = 30
                cv2.putText(frame, f"Device: {device}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"FPS: {current_fps:.1f}",
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"People: {person_count}",
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                print("\n正在退出...")
                break
            elif key == ord('s'):
                filename = f"{device.lower()}_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存: {filename}")
            elif key == ord(' '):
                paused = not paused
                status = "暂停" if paused else "继续"
                print(status)

    except KeyboardInterrupt:
        print("\n检测到中断...")

    finally:
        # 清理
        print("正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()

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
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()

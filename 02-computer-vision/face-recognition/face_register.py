# -*- coding: utf-8 -*-
"""
人脸录入工具 - OpenVINO 版本
用于录入老师/学生的人脸到数据库
适配 Python 3.14+ (不依赖 onnxruntime)
"""

import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime
import urllib.request
import zipfile

# OpenVINO 人脸识别
try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("警告: openvino 未安装，请运行: pip install openvino")


class FaceAnalyzerOV:
    """OpenVINO 人脸分析器"""

    MODEL_URLS = {
        "face-detection": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP16/",
        "face-reidentification": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP16/"
    }

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.core = ov.Core()

        # 下载并加载模型
        print("加载人脸检测模型...")
        det_model_path = self._ensure_model("face-detection", "face-detection-retail-0005")
        self.det_model = self.core.compile_model(det_model_path, "AUTO")
        self.det_input = self.det_model.input(0)
        self.det_output = self.det_model.output(0)

        print("加载人脸识别模型...")
        reid_model_path = self._ensure_model("face-reidentification", "face-reidentification-retail-0095")
        self.reid_model = self.core.compile_model(reid_model_path, "AUTO")
        self.reid_input = self.reid_model.input(0)
        self.reid_output = self.reid_model.output(0)

        print("人脸分析器就绪")

    def _ensure_model(self, model_type, model_name):
        """确保模型存在，否则下载"""
        model_path = os.path.join(self.model_dir, f"{model_name}.xml")
        bin_path = os.path.join(self.model_dir, f"{model_name}.bin")

        if not os.path.exists(model_path) or not os.path.exists(bin_path):
            print(f"  下载 {model_name}...")
            base_url = self.MODEL_URLS[model_type]

            try:
                urllib.request.urlretrieve(f"{base_url}{model_name}.xml", model_path)
                urllib.request.urlretrieve(f"{base_url}{model_name}.bin", bin_path)
                print(f"  ✓ {model_name} 下载完成")
            except Exception as e:
                print(f"  下载失败: {e}")
                raise

        return model_path

    def detect_faces(self, image, threshold=0.5):
        """
        检测人脸

        Args:
            image: BGR 图像
            threshold: 检测阈值

        Returns:
            faces: list of dict {bbox, confidence}
        """
        # 预处理 - 人脸检测模型输入 [1, 3, 300, 300]
        h, w = image.shape[:2]
        input_shape = self.det_input.shape
        input_h, input_w = input_shape[2], input_shape[3]

        blob = cv2.resize(image, (input_w, input_h))
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = blob.reshape(1, 3, input_h, input_w).astype(np.float32)

        # 推理
        result = self.det_model([blob])[self.det_output]

        # 解析结果 [1, 1, N, 7] -> [image_id, label, conf, x_min, y_min, x_max, y_max]
        faces = []
        for detection in result[0][0]:
            confidence = detection[2]
            if confidence > threshold:
                x_min = int(detection[3] * w)
                y_min = int(detection[4] * h)
                x_max = int(detection[5] * w)
                y_max = int(detection[6] * h)

                # 边界检查
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                faces.append({
                    "bbox": np.array([x_min, y_min, x_max, y_max]),
                    "confidence": float(confidence)
                })

        return faces

    def get_embedding(self, image, bbox):
        """
        获取人脸特征向量

        Args:
            image: BGR 图像
            bbox: 人脸边界框 [x1, y1, x2, y2]

        Returns:
            embedding: 256维特征向量
        """
        x1, y1, x2, y2 = bbox.astype(int)
        face_img = image[y1:y2, x1:x2]

        if face_img.size == 0:
            return None

        # 预处理 - 人脸识别模型输入 [1, 3, 128, 128]
        input_shape = self.reid_input.shape
        input_h, input_w = input_shape[2], input_shape[3]

        blob = cv2.resize(face_img, (input_w, input_h))
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = blob.reshape(1, 3, input_h, input_w).astype(np.float32)

        # 推理
        embedding = self.reid_model([blob])[self.reid_output]
        embedding = embedding.flatten()

        # L2 归一化
        embedding = embedding / np.linalg.norm(embedding)

        return embedding


class FaceDatabase:
    """人脸数据库管理"""

    def __init__(self, db_path="face_db"):
        self.db_path = db_path
        self.embeddings_file = os.path.join(db_path, "embeddings.pkl")
        self.info_file = os.path.join(db_path, "face_info.json")

        # 创建数据库目录
        os.makedirs(db_path, exist_ok=True)
        os.makedirs(os.path.join(db_path, "photos"), exist_ok=True)

        # 加载已有数据
        self.embeddings = {}  # {name: [embedding1, embedding2, ...]}
        self.face_info = {}   # {name: {title, added_time, photo_count}}
        self.load_database()

        # 初始化人脸分析器
        if HAS_OPENVINO:
            model_dir = os.path.join(db_path, "models")
            self.analyzer = FaceAnalyzerOV(model_dir)
        else:
            self.analyzer = None

    def load_database(self):
        """加载数据库"""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"已加载 {len(self.embeddings)} 个人脸特征")

        if os.path.exists(self.info_file):
            with open(self.info_file, 'r', encoding='utf-8') as f:
                self.face_info = json.load(f)

    def save_database(self):
        """保存数据库"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

        with open(self.info_file, 'w', encoding='utf-8') as f:
            json.dump(self.face_info, f, ensure_ascii=False, indent=2)
        print("数据库已保存")

    def detect_faces(self, image):
        """检测人脸"""
        if self.analyzer is None:
            return []
        return self.analyzer.detect_faces(image)

    def register_face(self, image, name, title=""):
        """
        注册人脸

        Args:
            image: BGR 图像
            name: 人名
            title: 职称/备注

        Returns:
            success: bool
            message: str
        """
        if self.analyzer is None:
            return False, "人脸分析器未初始化"

        faces = self.detect_faces(image)

        if len(faces) == 0:
            return False, "未检测到人脸"

        if len(faces) > 1:
            return False, f"检测到 {len(faces)} 张人脸，请确保画面中只有一个人"

        face = faces[0]
        embedding = self.analyzer.get_embedding(image, face["bbox"])

        if embedding is None:
            return False, "无法提取人脸特征"

        # 添加到数据库
        if name not in self.embeddings:
            self.embeddings[name] = []
            self.face_info[name] = {
                "title": title,
                "added_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "photo_count": 0
            }

        self.embeddings[name].append(embedding)
        self.face_info[name]["photo_count"] += 1

        # 保存照片
        photo_path = os.path.join(
            self.db_path, "photos",
            f"{name}_{self.face_info[name]['photo_count']}.jpg"
        )
        cv2.imwrite(photo_path, image)

        # 保存数据库
        self.save_database()

        return True, f"已录入 {name} ({title}) 的第 {self.face_info[name]['photo_count']} 张照片"

    def recognize_face(self, image, threshold=0.5):
        """
        识别人脸

        Args:
            image: BGR 图像
            threshold: 相似度阈值（余弦相似度，越大越相似）

        Returns:
            results: list of dict
        """
        if self.analyzer is None:
            return []

        faces = self.detect_faces(image)
        results = []

        for face in faces:
            bbox = face["bbox"]
            embedding = self.analyzer.get_embedding(image, bbox)

            if embedding is None:
                continue

            best_match = None
            best_similarity = -1

            # 与数据库中所有人比对
            for name, embeddings in self.embeddings.items():
                for db_embedding in embeddings:
                    # 计算余弦相似度
                    similarity = np.dot(embedding, db_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name

            if best_similarity > threshold:
                title = self.face_info.get(best_match, {}).get("title", "")
                results.append({
                    "name": best_match,
                    "title": title,
                    "confidence": float(best_similarity),
                    "bbox": bbox.astype(int)
                })
            else:
                results.append({
                    "name": "未知",
                    "title": "",
                    "confidence": float(best_similarity) if best_similarity > 0 else 0,
                    "bbox": bbox.astype(int)
                })

        return results

    def list_faces(self):
        """列出所有已录入的人脸"""
        print("\n" + "=" * 50)
        print("已录入人脸列表:")
        print("=" * 50)

        if not self.face_info:
            print("  (空)")
            return

        for name, info in self.face_info.items():
            print(f"  {name} ({info.get('title', '')}) - {info.get('photo_count', 0)} 张照片")

        print("=" * 50)

    def delete_face(self, name):
        """删除人脸"""
        if name in self.embeddings:
            del self.embeddings[name]
            del self.face_info[name]
            self.save_database()
            return True, f"已删除 {name}"
        return False, f"未找到 {name}"


def register_from_camera():
    """从摄像头录入人脸"""
    db = FaceDatabase()
    db.list_faces()

    print("\n人脸录入模式")
    print("=" * 50)
    name = input("请输入姓名: ").strip()
    if not name:
        print("姓名不能为空")
        return

    title = input("请输入职称/备注 (可选): ").strip()

    print(f"\n准备录入: {name} ({title})")
    print("按 S 键拍照录入，Q 键退出")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = f'Face Register - {name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    photo_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # 检测人脸
        faces = db.detect_faces(frame)

        for face in faces:
            bbox = face["bbox"].astype(int)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # 显示状态
        status = f"Detected: {len(faces)} face(s) | Photos: {photo_count}"
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Press S to capture, Q to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            success, message = db.register_face(frame, name, title)
            print(message)
            if success:
                photo_count += 1
                # 显示成功提示
                cv2.putText(display, "Captured!", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow(window_name, display)
                cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n录入完成: {name} 共 {photo_count} 张照片")


def register_from_image(image_path, name, title=""):
    """从图片文件录入人脸"""
    db = FaceDatabase()

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    success, message = db.register_face(image, name, title)
    print(message)


def main():
    print("=" * 60)
    print("人脸录入工具 (OpenVINO)")
    print("=" * 60)
    print()
    print("1. 从摄像头录入")
    print("2. 查看已录入人脸")
    print("3. 删除人脸")
    print("4. 退出")
    print()

    choice = input("请选择: ").strip()

    if choice == "1":
        register_from_camera()
    elif choice == "2":
        db = FaceDatabase()
        db.list_faces()
    elif choice == "3":
        db = FaceDatabase()
        db.list_faces()
        name = input("请输入要删除的姓名: ").strip()
        if name:
            success, message = db.delete_face(name)
            print(message)
    elif choice == "4":
        print("退出")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
